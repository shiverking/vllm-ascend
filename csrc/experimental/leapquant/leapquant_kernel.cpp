// SPDX-License-Identifier: Apache-2.0
// Isolated Ascend910B3 experiment. Not included in the production build.
#include "kernel_operator.h"

using namespace AscendC;

namespace {
constexpr uint32_t D = 128;
constexpr uint32_t N = D * D;
constexpr uint32_t W = 16;
constexpr uint32_t RANK = 4;
constexpr uint32_t PACKET = 27680;
constexpr uint32_t SCALE = 16384;
constexpr uint32_t SMOOTH = 16896;
constexpr uint32_t LEFT = 17408;  // Native v2: [rank, value], contiguous factors.
constexpr uint32_t RIGHT = 18432;
constexpr uint32_t KEYS = 19456;
constexpr uint32_t VALUES = 23552;  // Native v2: [window, value].
constexpr uint32_t DECAY = 27648;
constexpr uint32_t CURSOR = 27652;
constexpr uint32_t TAG = 27656;
constexpr uint32_t MAGIC = 0x4c513032;  // LQ02: prevents mixing with Triton v1 packets.
constexpr uint32_t VEC_BYTES = 32768;
#ifndef LEAPQUANT_POWER_ITERS
  #define LEAPQUANT_POWER_ITERS 6
#endif
static_assert(LEAPQUANT_POWER_ITERS >= 1 && LEAPQUANT_POWER_ITERS <= 6);
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

class LeapQuant {
 public:
  __aicore__ inline void Init(TPipe* pipe, GM_ADDR packet, GM_ADDR indices, uint32_t batch, uint32_t heads,
                              uint32_t blocks) {
    batch_ = batch;
    heads_ = heads;
    blocks_ = blocks;
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
    Duplicate(vec_, 0.0f, VEC_BYTES / sizeof(float));
    VSync();
  }

  __aicore__ inline void Encode(GM_ADDR state) {
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
      for (uint32_t pad = 0; pad < 6; ++pad) {
        tag_.SetValue(pad, 0);
      }
      Fold();
      Duplicate(keys_, (half)0, 2 * W * D);
      VSync();
      tag_.SetValue(0, MAGIC);
      StoreSync();
      DataCopy(packetGm_[(static_cast<uint64_t>(slot) * heads_ + h) * PACKET], packet_, PACKET);
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
      uint64_t offset = (static_cast<uint64_t>(slot) * heads_ + h) * PACKET;
      DataCopy(packet_, packetGm_[offset], PACKET);
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
      uint64_t qOff = (static_cast<uint64_t>(b) * keyHeads + h / (heads_ / keyHeads)) * D;
      DataCopy(io, qGm[qOff], D);
      LoadSync();
      Cast(qv, io, RoundMode::CAST_NONE, D);
      VSync();
      Muls(qv, qv, Q_SCALE, D);
      VSync();
      DataCopy(io, kGm[qOff], D);
      LoadSync();
      Cast(kv, io, RoundMode::CAST_NONE, D);
      VSync();
      DataCopy(io, vGm[static_cast<uint64_t>(job) * D], D);
      LoadSync();
      Cast(vv, io, RoundMode::CAST_NONE, D);
      VSync();
      // Scalar gates stay on device; no host synchronization is required.
      float gate = gGm.GetValue(job);
      auto gateVec = vec_[10 * D];
      gateVec.SetValue(0, gate);
      SV();
      Exp(gateVec, gateVec, 8);
      VSync();
      VS();
      float alpha = gateVec.GetValue(0);
      float decay = decay_.GetValue(0) * alpha;
      bfloat16_t bfBeta = bGm.GetValue(job);
      io.SetValue(0, bfBeta);
      SV();
      Cast(gateVec, io, RoundMode::CAST_NONE, 8);
      VSync();
      VS();
      float betaValue = gateVec.GetValue(0);

      MaterializeResidual();
      // Retain rank-major factors in vector storage for both Q and K readouts.
      Cast(vec_[52 * D], left_, RoundMode::CAST_NONE, RANK * D);
      VSync();
      Cast(vec_[56 * D], right_, RoundMode::CAST_NONE, RANK * D);
      VSync();
      auto keyRecords = vec_[16 * D], valRecords = vec_[32 * D], weights = vec_[48 * D];
      Cast(keyRecords, keys_, RoundMode::CAST_NONE, W * D);
      VSync();
      Cast(valRecords, values_, RoundMode::CAST_NONE, W * D);
      VSync();
      Muls(valRecords, valRecords, alpha, W * D);
      VSync();
      Cast(values_, valRecords, RoundMode::CAST_RINT, W * D);
      VSync();
      Cast(valRecords, values_, RoundMode::CAST_NONE, W * D);
      VSync();
      ReadBase(pred, kv);
      Muls(pred, pred, decay, D);
      VSync();
      RecordRead(out, keyRecords, valRecords, weights, kv, count);
      Add(pred, pred, out, D);
      VSync();
      Sub(vv, vv, pred, D);
      VSync();
      Muls(vv, vv, betaValue, D);
      VSync();
      Cast(values_[count * D], vv, RoundMode::CAST_RINT, D);
      VSync();
      Cast(valRecords[count * D], values_[count * D], RoundMode::CAST_NONE, D);
      VSync();
      Cast(keys_[count * D], kv, RoundMode::CAST_RINT, D);
      VSync();
      DataCopy(keyRecords[count * D], kv, D);
      VSync();
      ReadBase(pred, qv);
      Muls(pred, pred, decay, D);
      VSync();
      RecordRead(out, keyRecords, valRecords, weights, qv, count + 1);
      Add(out, out, pred, D);
      VSync();
      Cast(io, out, RoundMode::CAST_RINT, D);
      VSync();
      StoreSync();
      DataCopy(outGm[static_cast<uint64_t>(job) * D], io, D);
      StoreDone();

      if (count + 1 == W) {
        Reconstruct(decay, keyRecords, valRecords);
        Fold();
      } else {
        decay_.SetValue(0, decay);
        counter_.SetValue(0, count + 1);
      }
      StoreSync();
      if (count + 1 == W) {
        DataCopy(packetGm_[offset], packet_, PACKET);
      } else {
        // The low-bit checkpoint is immutable inside the window.
        // Write only the new key, decayed values and cursor/decay.
        DataCopy(packetGm_[offset + KEYS + count * D * sizeof(half)], packet_[KEYS + count * D * sizeof(half)],
                 D * sizeof(half));
        DataCopy(packetGm_[offset + VALUES], packet_[VALUES], (count + 1) * D * sizeof(half));
        DataCopy(packetGm_[offset + DECAY], packet_[DECAY], PACKET - DECAY);
      }
      StoreDone();
    }
  }

 private:
  // One matrix and one scratch matrix occupy 128 KiB of the 192 KiB UB.
  // All fitting passes reuse them; no FP32 recurrent matrix goes to HBM.
  __aicore__ inline void ElemColumns(LocalTensor<float> dst, LocalTensor<float> matrix, LocalTensor<float> vector,
                                     uint32_t rows, bool divide = false) {
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
    for (uint32_t stride = rows / 2; stride; stride /= 2) {
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
  __aicore__ inline void MatCols(LocalTensor<float> dst, LocalTensor<float> matrix, LocalTensor<float> vector,
                                 uint32_t rows) {
    uint32_t shape[2] = {rows, D}, sourceShape[2] = {rows, 1};
    Broadcast<float, 2, 1>(work_, vector, shape, sourceShape);
    VSync();
    Mul(work_, work_, matrix, rows * D);
    VSync();
    ColsReduce(dst, rows);
  }
  __aicore__ inline void Outer(LocalTensor<float> u, LocalTensor<float> v, bool subtract) {
    uint32_t shape[2] = {D, D}, src[2] = {D, 1};
    Broadcast<float, 2, 1>(work_, u, shape, src);
    VSync();
    ElemColumns(work_, work_, v, D);
    if (subtract) {
      Sub(s_, s_, work_, N);
    } else {
      Add(s_, s_, work_, N);
    }
    VSync();
  }
  __aicore__ inline void Normalize(LocalTensor<float> value) {
    auto tmp = vec_[11 * D];
    Mul(tmp, value, value, D);
    VSync();
    Add(tmp[64], tmp, tmp[64], 64);
    VSync();
    WholeReduceSum(tmp, tmp[64], 64, 1, 1, 1, 8);
    VSync();
    Maxs(tmp, tmp, 1.e-24f, 8);
    VSync();
    Rsqrt(tmp, tmp, 8);
    VSync();
    VS();
    float norm = tmp.GetValue(0);
    Muls(value, value, norm, D);
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
  __aicore__ inline void ReadBase(LocalTensor<float> dst, LocalTensor<float> vector) {
    auto scaled = vec_[49 * D];
    Mul(scaled, vector, smooth_, D);
    VSync();
    MatRows(dst, s_, scaled, D);
    Mul(dst, dst, scale_, D);
    VSync();
    auto dots = vec_[50 * D], tmp = vec_[5 * D];
    MatRows(dots, vec_[56 * D], vector, RANK);
    MatCols(tmp, vec_[52 * D], dots, RANK);
    Add(dst, dst, tmp, D);
    VSync();
  }
  __aicore__ inline void RecordRead(LocalTensor<float> dst, LocalTensor<float> kr, LocalTensor<float> vr,
                                    LocalTensor<float> weights, LocalTensor<float> vector, uint32_t count) {
    Duplicate(weights, 0.f, D);
    VSync();
    if (count) {
      MatRows(weights, kr, vector, count);
    }
    MatCols(dst, vr, weights, W);
  }
  __aicore__ inline void Reconstruct(float decay, LocalTensor<float> kr, LocalTensor<float> vr) {
    uint32_t shape[2] = {D, D}, src[2] = {D, 1};
    Broadcast<float, 2, 1>(work_, scale_, shape, src);
    VSync();
    Mul(s_, s_, work_, N);
    VSync();
    ElemColumns(s_, s_, smooth_, D);
    auto lv = vec_[12 * D], rv = vec_[13 * D];
    for (uint32_t r = 0; r < RANK; ++r) {
      Cast(lv, left_[r * D], RoundMode::CAST_NONE, D);
      VSync();
      Cast(rv, right_[r * D], RoundMode::CAST_NONE, D);
      VSync();
      Outer(lv, rv, false);
    }
    Muls(s_, s_, decay, N);
    VSync();
    for (uint32_t w = 0; w < W; ++w) {
      Outer(vr[w * D], kr[w * D], false);
    }
  }
  __aicore__ inline void Fold() {
    auto u = vec_[3 * D], direction = vec_[4 * D];
    for (uint32_t r = 0; r < RANK; ++r) {
      Duplicate(direction, 1.f, D);
      VSync();
      for (uint32_t it = 0; it < LEAPQUANT_POWER_ITERS; ++it) {
        MatRows(u, s_, direction, D);
        Normalize(u);
        MatCols(direction, s_, u, D);
        Normalize(direction);
      }
      MatRows(u, s_, direction, D);
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
    ColsReduce(smooth_, D, true);
    Maxs(smooth_, smooth_, 1.e-12f, D);
    VSync();
    ElemColumns(work_, s_, smooth_, D, true);
    // RowsReduce uses work_, so preserve normalized residual in s_.
    DataCopy(s_, work_, N);
    VSync();
    Abs(work_, work_, N);
    VSync();
    RowsReduce(scale_, D, true);
    Maxs(scale_, scale_, 1.e-12f, D);
    VSync();
    Muls(scale_, scale_, 1.f / 127.f, D);
    VSync();
    uint32_t shape[2] = {D, D}, src[2] = {D, 1};
    Broadcast<float, 2, 1>(work_, scale_, shape, src);
    VSync();
    Div(s_, s_, work_, N);
    VSync();
    auto quantHalf = workBuf_.Get<half>();
    Cast(quantHalf, s_, RoundMode::CAST_RINT, N);
    VSync();
    Cast(packetBuf_.Get<int8_t>(), quantHalf, RoundMode::CAST_RINT, N);
    VSync();
    decay_.SetValue(0, 1.f);
    counter_.SetValue(0, 0);
  }

  TBuf<QuePosition::VECCALC> packetBuf_, stateBuf_, workBuf_, vecBuf_;
  LocalTensor<uint8_t> packet_;
  LocalTensor<float> s_, work_, vec_, scale_, smooth_, decay_;
  LocalTensor<half> left_, right_, keys_, values_;
  LocalTensor<int32_t> counter_;
  LocalTensor<uint32_t> tag_;
  GlobalTensor<uint8_t> packetGm_;
  GlobalTensor<int32_t> indexGm_;
  uint32_t batch_, heads_, blocks_;
};
}  // namespace

extern "C" __global__ __aicore__ void leapquant_encode(GM_ADDR state, GM_ADDR packet, GM_ADDR indices, uint32_t batch,
                                                       uint32_t heads, uint32_t blocks) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  TPipe pipe;
  LeapQuant op;
  op.Init(&pipe, packet, indices, batch, heads, blocks);
  op.Encode(state);
}
extern "C" __global__ __aicore__ void leapquant_decode(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR beta, GM_ADDR g,
                                                       GM_ADDR packet, GM_ADDR indices, GM_ADDR output, uint32_t batch,
                                                       uint32_t heads, uint32_t keyHeads, uint32_t blocks) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  TPipe pipe;
  LeapQuant op;
  op.Init(&pipe, packet, indices, batch, heads, blocks);
  op.Decode(q, k, v, beta, g, output, keyHeads);
}
extern "C" void leapquant_encode_launch(uint32_t cores, void* stream, void* state, void* packet, void* indices,
                                        uint32_t batch, uint32_t heads, uint32_t blocks) {
  leapquant_encode<<<cores, nullptr, stream>>>((uint8_t*)state, (uint8_t*)packet, (uint8_t*)indices, batch, heads,
                                               blocks);
}
extern "C" void leapquant_decode_launch(uint32_t cores, void* stream, void* q, void* k, void* v, void* beta, void* g,
                                        void* packet, void* indices, void* output, uint32_t batch, uint32_t heads,
                                        uint32_t keyHeads, uint32_t blocks) {
  leapquant_decode<<<cores, nullptr, stream>>>((uint8_t*)q, (uint8_t*)k, (uint8_t*)v, (uint8_t*)beta, (uint8_t*)g,
                                               (uint8_t*)packet, (uint8_t*)indices, (uint8_t*)output, batch, heads,
                                               keyHeads, blocks);
}
