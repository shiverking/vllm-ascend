// SPDX-License-Identifier: Apache-2.0
// Protected AIV W16/R4 implementation for Ascend910B3.
// Explicit standalone build; the default plugin build remains unchanged.

#include "kernel_operator.h"

using namespace AscendC;

namespace {
constexpr uint32_t D = 128;
constexpr uint32_t N = D * D;
constexpr uint32_t W = 16;
constexpr uint32_t RANK = 4;
constexpr uint32_t POWER_ITERS = 8;
constexpr uint32_t PACKET = 27744;
constexpr uint32_t SCALE = 16384;
constexpr uint32_t SMOOTH = 16896;
constexpr uint32_t LEFT = 17408;  // [rank, value], contiguous factors.
constexpr uint32_t RIGHT = 18432;
constexpr uint32_t KEYS = 19456;
constexpr uint32_t VALUES = 23552;  // [window, value].
constexpr uint32_t DECAY = 27648;
constexpr uint32_t CURSOR = 27652;
constexpr uint32_t TAG = 27656;
constexpr uint32_t LOG_TIMES = 27680;
constexpr uint32_t MAGIC = 0x4c513034;  // LQ04 packet format marker.
constexpr uint32_t VEC_BYTES = 36864;
// FP32 Brcb with 8 repeats writes 8 * 8 * 32 = 2048 bytes.
// The broadcast workspace at vec_[68*D] therefore occupies groups 68..71.
constexpr uint32_t BRDC_START_BYTES = 68 * D * sizeof(float);
constexpr uint32_t BRDC_LENGTH_BYTES = 8 * 8 * 32;
constexpr uint32_t FUSED_DECAY_SCRATCH_BYTES = 3 * D * sizeof(float);
static_assert(BRDC_START_BYTES + BRDC_LENGTH_BYTES <= VEC_BYTES, "Brcb workspace outside vecBuf");
static_assert(FUSED_DECAY_SCRATCH_BYTES + D * sizeof(float) <= BRDC_START_BYTES,
              "Reconstruct decayScale overlaps ScaleRows Brcb workspace");

constexpr float Q_SCALE = 0.08838834764831845f;

__aicore__ inline void VSync() { PipeBarrier<PIPE_V>(); }
__aicore__ inline void VS() {
  SetFlag<HardEvent::V_S>(0);
  WaitFlag<HardEvent::V_S>(0);
}
__aicore__ inline void SV() {
  SetFlag<HardEvent::S_V>(0);
  WaitFlag<HardEvent::S_V>(0);
}
__aicore__ inline void LoadSync() {
  SetFlag<HardEvent::MTE2_V>(0);
  WaitFlag<HardEvent::MTE2_V>(0);
  SetFlag<HardEvent::MTE2_S>(0);
  WaitFlag<HardEvent::MTE2_S>(0);
}
__aicore__ inline void StoreSync() {
  SetFlag<HardEvent::V_MTE3>(0);
  WaitFlag<HardEvent::V_MTE3>(0);
  SetFlag<HardEvent::S_MTE3>(0);
  WaitFlag<HardEvent::S_MTE3>(0);
}
__aicore__ inline void StoreDone() {
  SetFlag<HardEvent::MTE3_V>(0);
  WaitFlag<HardEvent::MTE3_V>(0);
  SetFlag<HardEvent::MTE3_S>(0);
  WaitFlag<HardEvent::MTE3_S>(0);
}

static_assert(PACKET + 2 * N * sizeof(float) + VEC_BYTES <= 192 * 1024,
              "Full decode UB exceeds 192KiB");
class LeapQuant {
 public:
  __aicore__ inline void Init(TPipe* pipe, GM_ADDR packet, GM_ADDR indices, uint32_t batch, uint32_t heads,
                              uint32_t blocks, uint32_t packetStride) {
    batch_ = batch;
    heads_ = heads;
    blocks_ = blocks;
    packetStride_ = packetStride;
    packetGm_.SetGlobalBuffer((__gm__ uint8_t*)packet);
    indexGm_.SetGlobalBuffer((__gm__ int32_t*)indices);
    pipe->InitBuffer(packetBuf_, PACKET);
    pipe->InitBuffer(stateBuf_, N * sizeof(float));
    pipe->InitBuffer(workBuf_, N * sizeof(float));
    pipe->InitBuffer(vecBuf_, VEC_BYTES);
    s_ = stateBuf_.Get<float>();
    work_ = workBuf_.Get<float>();
    vec_ = vecBuf_.Get<float>();
    packet_ = packetBuf_.Get<uint8_t>();
    scale_ = packetBuf_.GetWithOffset<float>(D, SCALE);
    smooth_ = packetBuf_.GetWithOffset<float>(D, SMOOTH);
    left_ = packetBuf_.GetWithOffset<half>(RANK * D, LEFT);
    right_ = packetBuf_.GetWithOffset<half>(RANK * D, RIGHT);
    keys_ = packetBuf_.GetWithOffset<half>(W * D, KEYS);
    values_ = packetBuf_.GetWithOffset<half>(W * D, VALUES);
    decay_ = packetBuf_.GetWithOffset<float>(1, DECAY);
    counter_ = packetBuf_.GetWithOffset<int32_t>(1, CURSOR);
    tag_ = packetBuf_.GetWithOffset<uint32_t>(6, TAG);
    logTimes_ = packetBuf_.GetWithOffset<float>(W, LOG_TIMES);
    coefficients_ = vec_[65 * D];
    // Only the four padding rows in each of the two factor matrices are
    // read without being overwritten. Everything else has a producer before
    // its first use.
    Duplicate(vec_[16 * D + RANK * D], 0.0f, (8 - RANK) * D);
    Duplicate(vec_[40 * D + RANK * D], 0.0f, (8 - RANK) * D);
    VSync();
  }

  __aicore__ inline void TransposeState() {
    auto offset = vecBuf_.Get<uint32_t>();
    CreateVecIndex(offset.ReinterpretCast<int32_t>(), static_cast<int32_t>(0), D);
    VSync();
    ShiftLeft(offset, offset, static_cast<uint32_t>(9), D);
    VSync();
    for (uint32_t row = 1; row < 16; ++row) {
      Adds(offset[row * D].ReinterpretCast<int32_t>(), offset.ReinterpretCast<int32_t>(), static_cast<int32_t>(row * sizeof(float)), D);
    }
    VSync();
    for (uint32_t row = 0; row < D; row += 16) {
      Gather(work_[row * D], s_, offset, row * sizeof(float), 16 * D);
    }
    VSync();
  }

  __aicore__ inline void Encode(GM_ADDR state, bool transposed = false) {
    GlobalTensor<float> input;
    input.SetGlobalBuffer((__gm__ float*)state);
    for (uint32_t job = GetBlockIdx(); job < batch_ * heads_; job += GetBlockNum()) {
      uint32_t b = job / heads_, h = job % heads_;
      int32_t slot = indexGm_.GetValue(b);
      if (slot < 0 || static_cast<uint32_t>(slot) >= blocks_) {
        continue;
      }
      DataCopy(s_, input[static_cast<uint64_t>(job) * N], N);
      LoadSync();
      if (transposed) {
        TransposeState();
        DataCopy(s_, work_, N);
        VSync();
      }
      for (uint32_t pad = 0; pad < 6; ++pad) {
        tag_.SetValue(pad, 0);
      }
      Fold();
      Duplicate(keys_, (half)0, 2 * W * D);
      VSync();
      tag_.SetValue(0, MAGIC);
      StoreSync();
      DataCopy(packetGm_[(static_cast<uint64_t>(slot) * heads_ + h) * packetStride_], packet_, PACKET);
      StoreDone();
    }
  }

  // Materialize selected prefix/chunk states only. Decode never allocates
  // a persistent FP32 state pool. A false mask must not read an old slot.
  __aicore__ inline void Export(GM_ADDR output, GM_ADDR hasState, bool transposed = false) {
    GlobalTensor<float> outGm;
    GlobalTensor<uint8_t> maskGm;
    outGm.SetGlobalBuffer((__gm__ float*)output);
    maskGm.SetGlobalBuffer((__gm__ uint8_t*)hasState);
    for (uint32_t job = GetBlockIdx(); job < batch_ * heads_; job += GetBlockNum()) {
      uint32_t b = job / heads_, h = job % heads_;
      int32_t slot = indexGm_.GetValue(b);
      if (maskGm.GetValue(b) && slot >= 0 && static_cast<uint32_t>(slot) < blocks_) {
        DataCopy(packet_, packetGm_[(static_cast<uint64_t>(slot) * heads_ + h) * packetStride_], PACKET);
        LoadSync();
        int32_t count = counter_.GetValue(0);
        if (tag_.GetValue(0) != MAGIC || count < 0 || count >= W) {
          Duplicate(s_, __builtin_nanf(""), N);
          VSync();
        } else {
          MaterializeResidual();
          auto kr = vec_[24 * D], vr = vec_[48 * D];
          // Only [0,count) entries are live in the window.
          if (count) {
            Cast(kr, keys_, RoundMode::CAST_NONE, count * D);
            Cast(vr, values_, RoundMode::CAST_NONE, count * D);
          }
          float logDecay = decay_.GetValue(0);
          if (count) {
            Muls(coefficients_[8], logTimes_, -1.f, count);
            VSync();
            Adds(coefficients_[8], coefficients_[8], logDecay, count);
          }
          Duplicate(coefficients_, logDecay, 8);
          VSync();
          Exp(coefficients_, coefficients_, 8 + count);
          VSync();
          VS();
          float decay = coefficients_.GetValue(0);
          Reconstruct(decay, kr, vr, count);
        }
      } else {
        // No recurrent state for this (batch, head): the caller expects
        // zeros, not whatever a previous loop iteration left in s_. Only
        // this branch needs the zero-fill now, instead of every job.
        Duplicate(s_, 0.f, N);
        VSync();
      }
      if (transposed) { TransposeState(); }
      StoreSync();
      DataCopy(outGm[static_cast<uint64_t>(job) * N], transposed ? work_ : s_, N);
      StoreDone();
    }
  }

  __aicore__ inline void Decode(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR beta, GM_ADDR g, GM_ADDR output,
                                uint32_t keyHeads) {
    GlobalTensor<bfloat16_t> qGm, kGm, vGm, bGm, outGm;
    GlobalTensor<float> gGm;
    qGm.SetGlobalBuffer((__gm__ bfloat16_t*)q);
    kGm.SetGlobalBuffer((__gm__ bfloat16_t*)k);
    vGm.SetGlobalBuffer((__gm__ bfloat16_t*)v);
    bGm.SetGlobalBuffer((__gm__ bfloat16_t*)beta);
    gGm.SetGlobalBuffer((__gm__ float*)g);
    outGm.SetGlobalBuffer((__gm__ bfloat16_t*)output);
    auto io = vecBuf_.GetWithOffset<bfloat16_t>(D, 14 * D * sizeof(float));
    auto qv = vec_[0 * D], kv = vec_[1 * D], vv = vec_[2 * D];
    auto pred = vec_[6 * D], out = vec_[7 * D];
    auto baseQ = vec_[5 * D];
    for (uint32_t job = GetBlockIdx(); job < batch_ * heads_; job += GetBlockNum()) {
      uint32_t b = job / heads_, h = job % heads_;
      int32_t slot = indexGm_.GetValue(b);
      if (slot < 0 || static_cast<uint32_t>(slot) >= blocks_) {
        Duplicate(out, 0.0f, D);
        VSync();
        Cast(io, out, RoundMode::CAST_RINT, D);
        VSync();
        StoreSync();
        DataCopy(outGm[static_cast<uint64_t>(job) * D], io, D);
        StoreDone();
        continue;
      }
      uint64_t offset = (static_cast<uint64_t>(slot) * heads_ + h) * packetStride_;
      DataCopy(packet_, packetGm_[offset], PACKET);
      uint64_t qOff = (static_cast<uint64_t>(b) * keyHeads + h / (heads_ / keyHeads)) * D;
      auto ioK = vecBuf_.GetWithOffset<bfloat16_t>(D, 14 * D * sizeof(float) + D * sizeof(bfloat16_t));
      auto ioV = vecBuf_.GetWithOffset<bfloat16_t>(D, 15 * D * sizeof(float));
      auto ioBeta = vecBuf_.GetWithOffset<bfloat16_t>(16, 15 * D * sizeof(float) + D * sizeof(bfloat16_t));
      auto gateBlock = vec_[9 * D], gateVec = vec_[10 * D];
      DataCopy(io, qGm[qOff], D);
      DataCopy(ioK, kGm[qOff], D);
      DataCopy(ioV, vGm[static_cast<uint64_t>(job) * D], D);
      // Aligned gate blocks are safe because each request has 32 heads.
      DataCopy(gateBlock, gGm[job / 8 * 8], 8);
      DataCopy(ioBeta, bGm[job / 16 * 16], 16);
      LoadSync();
      if (tag_.GetValue(0) != MAGIC || counter_.GetValue(0) < 0 || counter_.GetValue(0) >= W) {
        // An invalid/foreign packet is visible as a nonfinite result.
        Duplicate(out, __builtin_nanf(""), D);
        VSync();
        Cast(io, out, RoundMode::CAST_RINT, D);
        VSync();
        StoreSync();
        DataCopy(outGm[static_cast<uint64_t>(job) * D], io, D);
        StoreDone();
        continue;
      }
      uint32_t count = counter_.GetValue(0);

      Cast(qv, io, RoundMode::CAST_NONE, D);
      Cast(kv, ioK, RoundMode::CAST_NONE, D);
      Cast(vv, ioV, RoundMode::CAST_NONE, D);
      Cast(gateVec, ioBeta, RoundMode::CAST_NONE, 16);
      VSync();
      Muls(qv, qv, Q_SCALE, D);
      VSync();
      VS();
      float gate = gateBlock.GetValue(job % 8);
      float betaValue = gateVec.GetValue(job % 16);
      float currentLog = decay_.GetValue(0) + gate;
      Duplicate(gateVec, currentLog, 8);
      VSync();
      Exp(gateVec, gateVec, 8);
      VSync();
      VS();
      float decay = gateVec.GetValue(0);

      ReadBaseHalf(pred, baseQ, kv, qv);
      // Retain rank-major factors in vector storage for both Q and K readouts.
      Cast(vec_[40 * D], left_, RoundMode::CAST_NONE, RANK * D);
      Cast(vec_[16 * D], right_, RoundMode::CAST_NONE, RANK * D);

      auto keyRecords = vec_[24 * D], valRecords = vec_[48 * D], weights = vec_[64 * D];
      // Cast only the live records; JointRead contracts 8 + count rows.
      if (count) {
        Cast(keyRecords, keys_, RoundMode::CAST_NONE, count * D);
        Cast(valRecords, values_, RoundMode::CAST_NONE, count * D);
      }
      VSync();
      // Preserve FP16 updates exactly as inserted. All intervening decay is
      // represented in FP32 log coefficients, avoiding per-token FP16 rounding.
      // [0,count) is the only live history. The current record's
      // coefficient is explicitly set to 1 below, after its insertion.
      if (count) {
        Muls(coefficients_[8], logTimes_, -1.f, count);
        VSync();
        Adds(coefficients_[8], coefficients_[8], currentLog, count);
        VSync();
        Exp(coefficients_[8], coefficients_[8], count);
        VSync();
      }
      Duplicate(coefficients_, decay, 8);
      VSync();
      Muls(pred, pred, decay, D);
      JointRead(out, weights, kv, count);
      Add(pred, pred, out, D);
      VSync();
      Sub(vv, vv, pred, D);
      VSync();
      Muls(vv, vv, betaValue, D);
      VSync();
      // Match the public kernel: clamp only the buffered FP16 record, not
      // the FP32 update used to produce the current token's output.
      auto buffered = vec_[8 * D];
      Mins(buffered, vv, 65504.f, D);
      VSync();
      Maxs(buffered, buffered, -65504.f, D);
      VSync();
      Cast(values_[count * D], buffered, RoundMode::CAST_RINT, D);
      DataCopy(valRecords[count * D], vv, D);
      Cast(keys_[count * D], kv, RoundMode::CAST_RINT, D);
      DataCopy(keyRecords[count * D], kv, D);
      VSync();
      logTimes_.SetValue(count, currentLog);
      coefficients_.SetValue(8 + count, 1.f);
      SV();
      DataCopy(pred, baseQ, D);
      VSync();
      Muls(pred, pred, decay, D);
      JointRead(out, weights, qv, count + 1);
      Add(out, out, pred, D);
      VSync();
      Cast(io, out, RoundMode::CAST_RINT, D);
      VSync();
      StoreSync();
      DataCopy(outGm[static_cast<uint64_t>(job) * D], io, D);
      StoreDone();

      if (count + 1 == W) {
        // Fold the FP16 buffer, while this token's output used FP32 delta.
        Cast(valRecords[count * D], values_[count * D], RoundMode::CAST_NONE, D);
        VSync();
        MaterializeResidual();
        Reconstruct(decay, keyRecords, valRecords);
        Fold(true);
      } else {
        decay_.SetValue(0, currentLog);
        counter_.SetValue(0, count + 1);
      }
      StoreSync();
      if (count + 1 == W) {
        DataCopy(packetGm_[offset], packet_, PACKET);
      } else {
        // The low-bit checkpoint is immutable inside the window.
        // Write the new raw key/update and FP32 log-time metadata only.
        DataCopy(packetGm_[offset + KEYS + count * D * sizeof(half)], packet_[KEYS + count * D * sizeof(half)],
                 D * sizeof(half));
        DataCopy(packetGm_[offset + VALUES + count * D * sizeof(half)], packet_[VALUES + count * D * sizeof(half)],
                 D * sizeof(half));
        DataCopy(packetGm_[offset + DECAY], packet_[DECAY], PACKET - DECAY);
      }
      StoreDone();
    }
  }

 private:
  // Matrix scratch stays in UB; decode has no persistent FP32 HBM pool.
  __aicore__ inline bool NewtonSafe(LocalTensor<float> vector, uint32_t count) {
    auto magnitude = vec_[66 * D], low = vec_[12 * D], high = vec_[13 * D];
    Abs(magnitude, vector, count);
    VSync();
    for (uint32_t offset = 0; offset < count; offset += 64) {
      const uint32_t lanes = count - offset < 64 ? count - offset : 64;
      WholeReduceMin(low, magnitude[offset], lanes, 1, 1, 1, 8, ReduceOrder::ORDER_ONLY_VALUE);
      WholeReduceMax(high, magnitude[offset], lanes, 1, 1, 1, 8, ReduceOrder::ORDER_ONLY_VALUE);
      VSync();
      VS();
      if (!(low.GetValue(0) >= 1.e-30f && high.GetValue(0) <= 1.e30f)) return false;
    }
    return true;
  }
  __aicore__ inline void ElemColumns(LocalTensor<float> dst, LocalTensor<float> matrix, LocalTensor<float> vector,
                                     uint32_t rows, bool divide = false) {
    if (divide && NewtonSafe(vector, D)) {
      auto inverse = vec_[67 * D];
      Reciprocal(inverse, vector, D);
      VSync();
      // Two Newton steps repair the approximate reciprocal before quantization.
      auto error = vec_[66 * D];
      for (uint32_t refine = 0; refine < 2; ++refine) {
        Mul(error, vector, inverse, D); VSync();
        Muls(error, error, -1.f, D); VSync();
        Adds(error, error, 2.f, D); VSync();
        Mul(inverse, inverse, error, D); VSync();
      }
      vector = inverse; divide = false;
    }
    for (uint32_t col = 0; col < D; col += 64) {
      if (divide) {
        Div(dst[col], matrix[col], vector[col], 64, rows, {1, 1, 1, 16, 16, 0});
      } else {
        Mul(dst[col], matrix[col], vector[col], 64, rows, {1, 1, 1, 16, 16, 0});
      }
    }
    VSync();
  }
  __aicore__ inline void RowsReduce(LocalTensor<float> dst, uint32_t rows, bool maximum = false) {
    if (maximum) {
      Max(work_[64], work_, work_[64], 64, rows, {1, 1, 1, 16, 16, 16});
      VSync();
      WholeReduceMax(dst, work_[64], 64, rows, 1, 1, 16, ReduceOrder::ORDER_ONLY_VALUE);
    } else {
      Add(work_[64], work_, work_[64], 64, rows, {1, 1, 1, 16, 16, 16});
      VSync();
      WholeReduceSum(dst, work_[64], 64, rows, 1, 1, 16);
    }
    VSync();
  }
  __aicore__ inline void MatRows(LocalTensor<float> dst, LocalTensor<float> matrix, LocalTensor<float> vector,
                                 uint32_t rows) {
    ElemColumns(work_, matrix, vector, rows);
    RowsReduce(dst, rows);
  }
  __aicore__ inline void ColsReduce(LocalTensor<float> dst, uint32_t rows, bool maximum = false) {
    uint32_t power = 1;
    while (power * 2 <= rows) {
      power *= 2;
    }
    if (rows > power) {
      if (maximum) {
        Max(work_, work_, work_[power * D], (rows - power) * D);
      } else {
        Add(work_, work_, work_[power * D], (rows - power) * D);
      }
      VSync();
    }
    for (uint32_t stride = power / 2; stride; stride /= 2) {
      if (maximum) {
        Max(work_, work_, work_[stride * D], stride * D);
      } else {
        Add(work_, work_, work_[stride * D], stride * D);
      }
      VSync();
    }
    DataCopy(dst, work_, D);
    VSync();
  }
  __aicore__ inline void ScaleRows(LocalTensor<float> dst, LocalTensor<float> matrix, LocalTensor<float> vector,
                                   uint32_t rows, bool divide = false) {
    if (divide && NewtonSafe(vector, rows)) {
      auto inverse = vec_[67 * D];
      Reciprocal(inverse, vector, rows);
      VSync();
      // Two Newton steps repair the approximate reciprocal before quantization.
      auto error = vec_[66 * D];
      for (uint32_t refine = 0; refine < 2; ++refine) {
        Mul(error, vector, inverse, rows); VSync();
        Muls(error, error, -1.f, rows); VSync();
        Adds(error, error, 2.f, rows); VSync();
        Mul(inverse, inverse, error, rows); VSync();
      }
      vector = inverse; divide = false;
    }
    auto repeated = vec_[68 * D];
    for (uint32_t row = 0; row < rows; row += 64) {
      uint32_t count = rows - row < 64 ? rows - row : 64;
      Brcb(repeated, vector[row], (count + 7) / 8, {1, 8});
      VSync();
      for (uint32_t col = 0; col < D; col += 64) {
        if (divide) {
          Div(dst[row * D + col], matrix[row * D + col], repeated, 64, count, {1, 1, 0, 16, 16, 1});
        } else {
          Mul(dst[row * D + col], matrix[row * D + col], repeated, 64, count, {1, 1, 0, 16, 16, 1});
        }
      }
      VSync();
    }
  }
  __aicore__ inline void MatCols(LocalTensor<float> dst, LocalTensor<float> matrix, LocalTensor<float> vector,
                                 uint32_t rows) {
    ScaleRows(work_, matrix, vector, rows);
    ColsReduce(dst, rows);
  }
  __aicore__ inline void Outer(LocalTensor<float> u, LocalTensor<float> v, bool subtract) {
    auto factor = u;
    if (subtract) {
      factor = vec_[9 * D];
      Muls(factor, u, -1.f, D);
      VSync();
    }
    auto repeated = vec_[68 * D];
    // A 32-byte block per row is sufficient for hardware-strided broadcast.
    // Avoid materializing a 64-KiB broadcast matrix for each rank-one update.
    for (uint32_t row = 0; row < D; row += 64) {
      Brcb(repeated, factor[row], 8, {1, 8});
      VSync();
      for (uint32_t col = 0; col < D; col += 64) {
        MulAddDst(s_[row * D + col], v[col], repeated, 64, 64, {1, 1, 0, 16, 0, 1});
      }
      VSync();
    }
  }
  __aicore__ inline float MaximumMagnitude(LocalTensor<float> value) {
    auto tmp = vec_[11 * D];
    Abs(tmp, value, D);
    VSync();
    Max(tmp[64], tmp, tmp[64], 64);
    VSync();
    WholeReduceMax(tmp, tmp[64], 64, 1, 1, 1, 8, ReduceOrder::ORDER_ONLY_VALUE);
    VSync();
    VS();
    return tmp.GetValue(0);
  }
  __aicore__ inline void Normalize(LocalTensor<float> value, float maximum = -1.f) {
    if (maximum < 0.f) {
      maximum = MaximumMagnitude(value);
    }
    float inverse = maximum > 1.e-30f ? 1.f / maximum : 0.f;
    Muls(value, value, inverse, D);
    VSync();
    auto tmp = vec_[11 * D];
    Mul(tmp, value, value, D);
    VSync();
    Add(tmp[64], tmp, tmp[64], 64);
    VSync();
    WholeReduceSum(tmp, tmp[64], 64, 1, 1, 1, 8);
    VSync();
    Maxs(tmp, tmp, 1.f, 8);
    VSync();
    Rsqrt(tmp, tmp, 8);
    VSync();
    VS();
    Muls(value, value, tmp.GetValue(0), D);
    VSync();
  }
  __aicore__ inline void MaterializeResidual() {
    auto halfTmp = workBuf_.Get<half>();
    auto quant = packetBuf_.Get<int8_t>();
    Cast(halfTmp, quant, RoundMode::CAST_NONE, N);
    VSync();
    Cast(s_, halfTmp, RoundMode::CAST_NONE, N);
    VSync();
  }
  __aicore__ inline void ReadBaseHalf(LocalTensor<float> dstK, LocalTensor<float> dstQ,
                                     LocalTensor<float> key, LocalTensor<float> query) {
    auto scaledK = vec_[66 * D], scaledQ = vec_[67 * D];
    Mul(scaledK, key, smooth_, D);
    Mul(scaledQ, query, smooth_, D);
    VSync();
    // INT8 residual <=127 and 128 terms. Bound absolute half sum below 32768.
    auto magnitudes = vec_[11 * D];
    auto maximum = vec_[13 * D];
    Abs(magnitudes, scaledK, 2 * D);
    VSync();
    WholeReduceMax(maximum, magnitudes, 64, 4, 1, 1, 8, ReduceOrder::ORDER_ONLY_VALUE);
    VSync();
    WholeReduceMax(magnitudes, maximum, 4, 1, 1, 1, 8, ReduceOrder::ORDER_ONLY_VALUE);
    VSync();
    VS();
    if (magnitudes.GetValue(0) > 2.f) {
      MaterializeResidual();
      ReadBasePair(dstK, dstQ, key, query);
      return;
    }
    auto halfMatrix = stateBuf_.Get<half>();
    auto halfWork = workBuf_.Get<half>();
    auto halfK = vecBuf_.GetWithOffset<half>(D, 8 * D * sizeof(float));
    auto halfQ = vecBuf_.GetWithOffset<half>(D, 8 * D * sizeof(float) + D * sizeof(half));
    auto halfOutput = vecBuf_.GetWithOffset<half>(D, 10 * D * sizeof(float));
    Cast(halfMatrix, packetBuf_.Get<int8_t>(), RoundMode::CAST_NONE, N);
    Cast(halfK, scaledK, RoundMode::CAST_RINT, D);
    Cast(halfQ, scaledQ, RoundMode::CAST_RINT, D);
    VSync();
    Mul(halfWork, halfMatrix, halfK, 128, D, {1, 1, 1, 8, 8, 0});
    VSync();
    WholeReduceSum(halfOutput, halfWork, 128, D, 1, 1, 8);
    VSync();
    Cast(dstK, halfOutput, RoundMode::CAST_NONE, D);
    Mul(halfWork, halfMatrix, halfQ, 128, D, {1, 1, 1, 8, 8, 0});
    VSync();
    WholeReduceSum(halfOutput, halfWork, 128, D, 1, 1, 8);
    VSync();
    Cast(dstQ, halfOutput, RoundMode::CAST_NONE, D);
    VSync();
    Mul(dstK, dstK, scale_, D);
    Mul(dstQ, dstQ, scale_, D);
    VSync();
  }
  __aicore__ inline void ReadBasePair(LocalTensor<float> dstK, LocalTensor<float> dstQ, LocalTensor<float> key,
                                      LocalTensor<float> query) {
    auto scaledK = vec_[66 * D], scaledQ = vec_[67 * D];
    Mul(scaledK, key, smooth_, D);
    Mul(scaledQ, query, smooth_, D);
    VSync();
    MatRows(dstK, s_, scaledK, D);
    MatRows(dstQ, s_, scaledQ, D);
    Mul(dstK, dstK, scale_, D);
    Mul(dstQ, dstQ, scale_, D);
    VSync();
  }
  __aicore__ inline void JointRead(LocalTensor<float> dst, LocalTensor<float> weights, LocalTensor<float> vector,
                                   uint32_t count) {
    // Four compensators and four zero padding rows precede the 16 real
    // records. Shared contraction removes two pairs of small matrix ops.
    // Rows [8 + count, 24) correspond to window slots that have not been
    // written yet this window; MatCols' ColsReduce handles any row count
    // (not just powers of two), so there is no need to pad the
    // contraction out to the full 24 rows just to multiply zeros.
    // MatRows writes every coefficient that MatCols will consume.
    MatRows(weights, vec_[16 * D], vector, 8 + count);
    Mul(weights, weights, coefficients_, 8 + count);
    VSync();
    MatCols(dst, vec_[40 * D], weights, 8 + count);
  }
  __aicore__ inline void Reconstruct(float decay, LocalTensor<float> kr, LocalTensor<float> vr, uint32_t count = W) {
    // Avoid one full N-element multiply by fusing decay into row scales.
    // IMPORTANT: ScaleRows Brcb writes 2048 bytes into vec_[68*D ... 72*D).
    // The old vec_[69*D] scratch overlapped this region and was corrupted.
    // vec_[3*D] is dead during Reconstruct (used again only by Fold after it),
    // and does not overlap broadcast scratch or live kr/vr/coefficients.
    auto decayScale = vec_[3 * D];
    Muls(decayScale, scale_, decay, D);
    VSync();
    ScaleRows(s_, s_, decayScale, D);
    ElemColumns(s_, s_, smooth_, D);
    auto lv = vec_[12 * D], rv = vec_[13 * D];
    for (uint32_t r = 0; r < RANK; ++r) {
      Cast(lv, left_[r * D], RoundMode::CAST_NONE, D);
      VSync();
      Cast(rv, right_[r * D], RoundMode::CAST_NONE, D);
      VSync();
      Muls(lv, lv, decay, D);
      VSync();
      Outer(lv, rv, false);
    }
    if (count) {
      ScaleRows(vr, vr, coefficients_[8], count);
    }
    for (uint32_t w = 0; w < count; ++w) {
      Outer(vr[w * D], kr[w * D], false);
    }
  }
  __aicore__ inline void Fold(bool warm = false) {
    auto u = vec_[3 * D], direction = vec_[4 * D];
    for (uint32_t r = 0; r < RANK; ++r) {
      bool restart = !warm;
      float maximum = -1.f;
      if (warm) {
        Cast(u, left_[r * D], RoundMode::CAST_NONE, D);
        VSync();
        MatCols(direction, s_, u, D);
        maximum = MaximumMagnitude(direction);
        restart = maximum <= 1.e-30f;
      }
      if (restart) {
        maximum = -1.f;
        Mul(work_, s_, s_, N);
        VSync();
        ColsReduce(direction, D);
        Sqrt(direction, direction, D);
        VSync();
      }
      Normalize(direction, maximum);
      if (restart) {
        for (uint32_t it = 0; it < POWER_ITERS; ++it) {
          MatRows(u, s_, direction, D);
          MatCols(direction, s_, u, D);
          Normalize(direction);
        }
      }
      MatRows(u, s_, direction, D);
      Mins(u, u, 65504.f, D);
      VSync();
      Maxs(u, u, -65504.f, D);
      VSync();
      Cast(left_[r * D], u, RoundMode::CAST_RINT, D);
      VSync();
      Cast(right_[r * D], direction, RoundMode::CAST_RINT, D);
      VSync();
      Cast(u, left_[r * D], RoundMode::CAST_NONE, D);
      VSync();
      Cast(direction, right_[r * D], RoundMode::CAST_NONE, D);
      VSync();
      Outer(u, direction, true);
    }
    Abs(work_, s_, N);
    VSync();
    ColsReduce(smooth_, D);
    Muls(smooth_, smooth_, 1.f / D, D);
    VSync();
    Sqrt(smooth_, smooth_, D);
    VSync();
    Maxs(smooth_, smooth_, 1.e-6f, D);
    VSync();
    // No producer needs the unnormalized residual after this point.
    ElemColumns(s_, s_, smooth_, D, true);
    Abs(work_, s_, N);
    VSync();
    RowsReduce(scale_, D, true);
    auto mask = vecBuf_.GetWithOffset<uint8_t>(32, 15 * D * sizeof(float));
    CompareScalar(mask, scale_, 1.e-30f, CMPMODE::GT, D);
    VSync();
    Muls(scale_, scale_, 1.f / 127.f, D);
    VSync();
    Select(scale_, mask, scale_, 1.f, SELMODE::VSEL_TENSOR_SCALAR_MODE, D);
    VSync();
    ScaleRows(s_, s_, scale_, D, true);
    // Round in FP32 before the FP16 bridge: direct FP32->FP16->INT8
    // double-rounds values close to half integers and changes the quantizer.
    auto integers = workBuf_.Get<int32_t>();
    Cast(integers, s_, RoundMode::CAST_RINT, N);
    VSync();
    SetDeqScale((half)1.f);
    auto quantHalf = stateBuf_.Get<half>();
    Cast(quantHalf, integers, RoundMode::CAST_ROUND, N);
    VSync();
    Cast(packetBuf_.Get<int8_t>(), quantHalf, RoundMode::CAST_TRUNC, N);
    VSync();
    Duplicate(logTimes_, 0.f, W);
    VSync();
    decay_.SetValue(0, 0.f);
    counter_.SetValue(0, 0);
  }

  TBuf<QuePosition::VECCALC> packetBuf_, stateBuf_, workBuf_, vecBuf_;
  LocalTensor<uint8_t> packet_;
  LocalTensor<float> s_, work_, vec_, scale_, smooth_, decay_, logTimes_, coefficients_;
  LocalTensor<half> left_, right_, keys_, values_;
  LocalTensor<int32_t> counter_;
  LocalTensor<uint32_t> tag_;
  GlobalTensor<uint8_t> packetGm_;
  GlobalTensor<int32_t> indexGm_;
  uint32_t batch_, heads_, blocks_, packetStride_;
};
}  // namespace

extern "C" uint32_t leapquant_packet_bytes() { return PACKET; }
extern "C" uint32_t leapquant_format_tag() { return MAGIC; }

// Serving pools require one MiB of SSM storage to match the K plane's
// block stride. Content stays LQ04; each head has unused trailing padding.
extern "C" __global__ __aicore__ void leapquant_padded_encode(GM_ADDR state, GM_ADDR packet, GM_ADDR indices,
                                                              uint32_t batch, uint32_t heads, uint32_t blocks,
                                                              uint32_t stride) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  TPipe pipe;
  LeapQuant op;
  op.Init(&pipe, packet, indices, batch, heads, blocks, stride);
  op.Encode(state);
}
extern "C" void leapquant_padded_encode_launch(uint32_t cores, void* stream, void* state, void* packet, void* indices,
                                               uint32_t batch, uint32_t heads, uint32_t blocks, uint32_t stride) {
  leapquant_padded_encode<<<cores, nullptr, stream>>>((uint8_t*)state, (uint8_t*)packet, (uint8_t*)indices, batch,
                                                      heads, blocks, stride);
}
extern "C" __global__ __aicore__ void leapquant_padded_decode(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR beta, GM_ADDR g,
                                                              GM_ADDR packet, GM_ADDR indices, GM_ADDR output,
                                                              uint32_t batch, uint32_t heads, uint32_t keyHeads,
                                                              uint32_t blocks, uint32_t stride) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  TPipe pipe;
  LeapQuant op;
  op.Init(&pipe, packet, indices, batch, heads, blocks, stride);
  op.Decode(q, k, v, beta, g, output, keyHeads);
}
extern "C" void leapquant_padded_decode_launch(uint32_t cores, void* stream, void* q, void* k, void* v, void* beta,
                                               void* g, void* packet, void* indices, void* output, uint32_t batch,
                                               uint32_t heads, uint32_t keyHeads, uint32_t blocks, uint32_t stride) {
  leapquant_padded_decode<<<cores, nullptr, stream>>>((uint8_t*)q, (uint8_t*)k, (uint8_t*)v, (uint8_t*)beta,
                                                      (uint8_t*)g, (uint8_t*)packet, (uint8_t*)indices,
                                                      (uint8_t*)output, batch, heads, keyHeads, blocks, stride);
}
extern "C" __global__ __aicore__ void leapquant_padded_export(GM_ADDR packet, GM_ADDR indices, GM_ADDR mask,
                                                              GM_ADDR output, uint32_t batch, uint32_t heads,
                                                              uint32_t blocks, uint32_t stride) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  TPipe pipe;
  LeapQuant op;
  op.Init(&pipe, packet, indices, batch, heads, blocks, stride);
  op.Export(output, mask);
}
extern "C" void leapquant_padded_export_launch(uint32_t cores, void* stream, void* packet, void* indices, void* mask,
                                               void* output, uint32_t batch, uint32_t heads, uint32_t blocks,
                                               uint32_t stride) {
  leapquant_padded_export<<<cores, nullptr, stream>>>((uint8_t*)packet, (uint8_t*)indices, (uint8_t*)mask,
                                                      (uint8_t*)output, batch, heads, blocks, stride);
}

// Fused prefill [K,V] layout conversion; packets retain native [V,K] order.
extern "C" __global__ __aicore__ void leapquant_layout_encode(GM_ADDR state, GM_ADDR packet, GM_ADDR indices, uint32_t batch, uint32_t heads, uint32_t blocks, uint32_t stride) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  TPipe pipe; LeapQuant op; op.Init(&pipe, packet, indices, batch, heads, blocks, stride); op.Encode(state, true);
}
extern "C" void leapquant_layout_encode_launch(uint32_t cores, void* stream, void* state, void* packet, void* indices, uint32_t batch, uint32_t heads, uint32_t blocks, uint32_t stride) {
  leapquant_layout_encode<<<cores, nullptr, stream>>>((uint8_t*)state, (uint8_t*)packet, (uint8_t*)indices, batch, heads, blocks, stride);
}
extern "C" __global__ __aicore__ void leapquant_layout_export(GM_ADDR packet, GM_ADDR indices, GM_ADDR mask, GM_ADDR output, uint32_t batch, uint32_t heads, uint32_t blocks, uint32_t stride) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  TPipe pipe; LeapQuant op; op.Init(&pipe, packet, indices, batch, heads, blocks, stride); op.Export(output, mask, true);
}
extern "C" void leapquant_layout_export_launch(uint32_t cores, void* stream, void* packet, void* indices, void* mask, void* output, uint32_t batch, uint32_t heads, uint32_t blocks, uint32_t stride) {
  leapquant_layout_export<<<cores, nullptr, stream>>>((uint8_t*)packet, (uint8_t*)indices, (uint8_t*)mask, (uint8_t*)output, batch, heads, blocks, stride);
}
