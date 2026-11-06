// SPDX-License-Identifier: Apache-2.0
// Isolated Ascend910B3 experiment. Not included in the production build.
#include "kernel_operator.h"
#include "leapquant_build_config.h"

using namespace AscendC;

namespace {
constexpr uint32_t D = 128;
constexpr uint32_t N = D * D;
constexpr uint32_t W = 16;
constexpr uint32_t RANK = 4;
constexpr uint32_t PACKET = 27744;
constexpr uint32_t SCALE = 16384;
constexpr uint32_t SMOOTH = 16896;
constexpr uint32_t LEFT = 17408;  // Native v2: [rank, value], contiguous factors.
constexpr uint32_t RIGHT = 18432;
constexpr uint32_t KEYS = 19456;
constexpr uint32_t VALUES = 23552;  // Native v2: [window, value].
constexpr uint32_t DECAY = 27648;
constexpr uint32_t CURSOR = 27652;
constexpr uint32_t TAG = 27656;
constexpr uint32_t LOG_TIMES = 27680;
constexpr uint32_t MAGIC = 0x4c513034;  // LQ04: prevents mixing with Triton v1 packets.
constexpr uint32_t VEC_BYTES = 36864;
#ifndef LEAPQUANT_POWER_ITERS
  #define LEAPQUANT_POWER_ITERS 8
#endif
static_assert(LEAPQUANT_POWER_ITERS >= 1 && LEAPQUANT_POWER_ITERS <= 8);
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
                              uint32_t blocks, uint32_t tile = D) {
    batch_ = batch;
    heads_ = heads;
    blocks_ = blocks;
    packetGm_.SetGlobalBuffer((__gm__ uint8_t*)packet);
    indexGm_.SetGlobalBuffer((__gm__ int32_t*)indices);
    pipe->InitBuffer(packetBuf_, PACKET);
    pipe->InitBuffer(stateBuf_, tile * D * sizeof(float));
    pipe->InitBuffer(workBuf_, tile * D * sizeof(float));
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
      Duplicate(logTimes_, 0.f, W);
      VSync();
      Duplicate(keys_, (half)0, 2 * W * D);
      VSync();
      tag_.SetValue(0, MAGIC);
      StoreSync();
      DataCopy(packetGm_[(static_cast<uint64_t>(slot) * heads_ + h) * PACKET], packet_, PACKET);
      StoreDone();
    }
  }

  __aicore__ inline void Decode(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR beta, GM_ADDR g, GM_ADDR output,
                                uint32_t keyHeads, bool streaming = false) {
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
    auto nextPacket = stateBuf_.GetWithOffset<uint8_t>(PACKET, N * sizeof(float) / 2);
    bool prefetched = false;
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
      if (prefetched) {
        LoadSync();
        DataCopy(packet_, nextPacket, PACKET);
        VSync();
        VS();
        prefetched = false;
      } else {
        DataCopy(packet_, packetGm_[offset], PACKET);
      }
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

      if (streaming) {
        if (LEAPQUANT_STREAM_PREFETCH && count + 1 != W && job + GetBlockNum() < batch_ * heads_) {
          uint32_t nextJob = job + GetBlockNum(), nextHead = nextJob % heads_;
          int32_t nextSlot = indexGm_.GetValue(nextJob / heads_);
          if (nextSlot >= 0 && static_cast<uint32_t>(nextSlot) < blocks_ && (nextSlot != slot || nextHead != h)) {
            DataCopy(nextPacket, packetGm_[(static_cast<uint64_t>(nextSlot) * heads_ + nextHead) * PACKET], PACKET);
            prefetched = true;
          }
        }
        ReadResidualPair(pred, baseQ, kv, qv);
      } else {
        MaterializeResidual();
      }
      // Retain rank-major factors in vector storage for both Q and K readouts.
      Cast(vec_[40 * D], left_, RoundMode::CAST_NONE, RANK * D);
      VSync();
      Cast(vec_[16 * D], right_, RoundMode::CAST_NONE, RANK * D);
      VSync();
      auto keyRecords = vec_[24 * D], valRecords = vec_[48 * D], weights = vec_[64 * D];
      if (!streaming) {
        Cast(keyRecords, keys_, RoundMode::CAST_NONE, W * D);
        VSync();
        Cast(valRecords, values_, RoundMode::CAST_NONE, W * D);
        VSync();
      } else if (count) {
        Cast(keyRecords, keys_, RoundMode::CAST_NONE, count * D);
        Cast(valRecords, values_, RoundMode::CAST_NONE, count * D);
        VSync();
      }
      // Preserve FP16 updates exactly as inserted. All intervening decay is
      // represented in FP32 log coefficients, avoiding per-token FP16 rounding.
      Muls(coefficients_[8], logTimes_, -1.f, W);
      VSync();
      Adds(coefficients_[8], coefficients_[8], currentLog, W);
      VSync();
      Exp(coefficients_[8], coefficients_[8], W);
      VSync();
      Duplicate(coefficients_, decay, 8);
      VSync();
      if (!streaming) {
        ReadBase(pred, kv);
      }
      Muls(pred, pred, decay, D);
      VSync();
      JointRead(out, weights, kv, count, streaming);
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
      VSync();
      DataCopy(valRecords[count * D], vv, D);
      VSync();
      Cast(keys_[count * D], kv, RoundMode::CAST_RINT, D);
      VSync();
      DataCopy(keyRecords[count * D], kv, D);
      VSync();
      logTimes_.SetValue(count, currentLog);
      coefficients_.SetValue(8 + count, 1.f);
      SV();
      if (streaming) {
        DataCopy(pred, baseQ, D);
        VSync();
      } else {
        ReadBase(pred, qv);
      }
      Muls(pred, pred, decay, D);
      VSync();
      JointRead(out, weights, qv, count + 1, streaming);
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
        if (streaming) {
          MaterializeResidual();
        }
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

  __aicore__ inline void DecodeTiles(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR beta, GM_ADDR g, GM_ADDR output,
                                     uint32_t keyHeads, uint32_t tile) {
    GlobalTensor<bfloat16_t> qGm, kGm, vGm, bGm, outGm;
    GlobalTensor<float> gGm;
    qGm.SetGlobalBuffer((__gm__ bfloat16_t*)q);
    kGm.SetGlobalBuffer((__gm__ bfloat16_t*)k);
    vGm.SetGlobalBuffer((__gm__ bfloat16_t*)v);
    bGm.SetGlobalBuffer((__gm__ bfloat16_t*)beta);
    gGm.SetGlobalBuffer((__gm__ float*)g);
    outGm.SetGlobalBuffer((__gm__ bfloat16_t*)output);
    auto io = vecBuf_.GetWithOffset<bfloat16_t>(D, 14 * D * sizeof(float));
    auto ioK = vecBuf_.GetWithOffset<bfloat16_t>(D, 14 * D * sizeof(float) + D * sizeof(bfloat16_t));
    auto ioV = vecBuf_.GetWithOffset<bfloat16_t>(D, 15 * D * sizeof(float));
    auto ioBeta = vecBuf_.GetWithOffset<bfloat16_t>(16, 15 * D * sizeof(float) + D * sizeof(bfloat16_t));
    auto qv = vec_, kv = vec_[D], vv = vec_[2 * D];
    auto pred = vec_[6 * D], out = vec_[7 * D], weights = vec_[64 * D];
    auto gateBlock = vec_[9 * D], gateVec = vec_[10 * D];
    for (uint32_t item = GetBlockIdx(); item < batch_ * heads_ * (D / tile); item += GetBlockNum()) {
      uint32_t job = item / (D / tile), start = (item % (D / tile)) * tile;
      uint32_t b = job / heads_, h = job % heads_;
      int32_t slot = indexGm_.GetValue(b);
      if (slot < 0 || static_cast<uint32_t>(slot) >= blocks_) {
        Duplicate(out, 0.f, tile);
        VSync();
        Cast(io, out, RoundMode::CAST_RINT, tile);
        VSync();
        StoreSync();
        DataCopy(outGm[static_cast<uint64_t>(job) * D + start], io, tile);
        StoreDone();
        continue;
      }
      uint64_t offset = (static_cast<uint64_t>(slot) * heads_ + h) * PACKET;
      // Only this value slice is read or written. Shared keys, scales and
      // metadata remain immutable until the following commit kernel.
      DataCopy(packet_, packetGm_[offset + start * D], tile * D);
      DataCopy(packet_[SCALE], packetGm_[offset + SCALE + start * sizeof(float)], tile * sizeof(float));
      DataCopy(packet_[SMOOTH], packetGm_[offset + SMOOTH], D * sizeof(float));
      DataCopy(packet_[RIGHT], packetGm_[offset + RIGHT], (RANK + W) * D * sizeof(half));
      DataCopyParams leftCopy{RANK, static_cast<uint16_t>(tile * sizeof(half) / 32),
                              static_cast<uint16_t>((D - tile) * sizeof(half) / 32), 0};
      DataCopyParams valueCopy{W, static_cast<uint16_t>(tile * sizeof(half) / 32),
                               static_cast<uint16_t>((D - tile) * sizeof(half) / 32), 0};
      DataCopy(packet_[LEFT], packetGm_[offset + LEFT + start * sizeof(half)], leftCopy);
      DataCopy(packet_[VALUES], packetGm_[offset + VALUES + start * sizeof(half)], valueCopy);
      DataCopy(packet_[DECAY], packetGm_[offset + DECAY], PACKET - DECAY);
      uint64_t qOff = (static_cast<uint64_t>(b) * keyHeads + h / (heads_ / keyHeads)) * D;
      DataCopy(io, qGm[qOff], D);
      DataCopy(ioK, kGm[qOff], D);
      DataCopy(ioV, vGm[static_cast<uint64_t>(job) * D + start], tile);
      DataCopy(gateBlock, gGm[job / 8 * 8], 8);
      DataCopy(ioBeta, bGm[job / 16 * 16], 16);
      LoadSync();
      if (tag_.GetValue(0) != MAGIC || counter_.GetValue(0) < 0 || counter_.GetValue(0) >= W) {
        Duplicate(out, __builtin_nanf(""), tile);
        VSync();
        Cast(io, out, RoundMode::CAST_RINT, tile);
        VSync();
        StoreSync();
        DataCopy(outGm[static_cast<uint64_t>(job) * D + start], io, tile);
        StoreDone();
        continue;
      }
      uint32_t count = counter_.GetValue(0);
      Cast(qv, io, RoundMode::CAST_NONE, D);
      Cast(kv, ioK, RoundMode::CAST_NONE, D);
      Cast(vv, ioV, RoundMode::CAST_NONE, tile);
      Cast(gateVec, ioBeta, RoundMode::CAST_NONE, 16);
      VSync();
      Muls(qv, qv, Q_SCALE, D);
      VSync();
      VS();
      float betaValue = gateVec.GetValue(job % 16);
      float currentLog = decay_.GetValue(0) + gateBlock.GetValue(job % 8);
      Duplicate(gateVec, currentLog, 8);
      VSync();
      Exp(gateVec, gateVec, 8);
      VSync();
      VS();
      float decay = gateVec.GetValue(0);
      auto halfTmp = workBuf_.Get<half>();
      Cast(halfTmp, packetBuf_.Get<int8_t>(), RoundMode::CAST_NONE, tile * D);
      VSync();
      Cast(s_, halfTmp, RoundMode::CAST_NONE, tile * D);
      Cast(vec_[16 * D], right_, RoundMode::CAST_NONE, RANK * D);
      Cast(vec_[24 * D], keys_, RoundMode::CAST_NONE, W * D);
      Cast(vec_[40 * D], left_, RoundMode::CAST_NONE, RANK * tile);
      Cast(vec_[40 * D + 8 * tile], values_, RoundMode::CAST_NONE, W * tile);
      VSync();
      Muls(coefficients_[8], logTimes_, -1.f, W);
      VSync();
      Adds(coefficients_[8], coefficients_[8], currentLog, W);
      VSync();
      Exp(coefficients_[8], coefficients_[8], W);
      Duplicate(coefficients_, decay, 8);
      VSync();
      ReadTile(pred, kv, weights, count, tile, decay);
      Sub(vv, vv, pred, tile);
      VSync();
      Muls(vv, vv, betaValue, tile);
      VSync();
      ReadTile(out, qv, weights, count, tile, decay);
      auto dot = vec_[3 * D];
      Mul(dot, qv, kv, D);
      VSync();
      Add(dot[64], dot, dot[64], 64);
      VSync();
      WholeReduceSum(dot, dot[64], 64, 1, 1, 1, 8);
      VSync();
      VS();
      Muls(pred, vv, dot.GetValue(0), tile);
      VSync();
      Add(out, out, pred, tile);
      VSync();
      Cast(io, out, RoundMode::CAST_RINT, tile);
      Mins(pred, vv, 65504.f, tile);
      VSync();
      Maxs(pred, pred, -65504.f, tile);
      VSync();
      Cast(values_, pred, RoundMode::CAST_RINT, tile);
      VSync();
      StoreSync();
      DataCopy(outGm[static_cast<uint64_t>(job) * D + start], io, tile);
      DataCopy(packetGm_[offset + VALUES + (count * D + start) * sizeof(half)], packet_[VALUES], tile * sizeof(half));
      StoreDone();
    }
  }

  __aicore__ inline void CommitTiles(GM_ADDR k, GM_ADDR g, uint32_t keyHeads) {
    GlobalTensor<bfloat16_t> kGm;
    GlobalTensor<float> gGm;
    kGm.SetGlobalBuffer((__gm__ bfloat16_t*)k);
    gGm.SetGlobalBuffer((__gm__ float*)g);
    auto io = vecBuf_.GetWithOffset<bfloat16_t>(D, 14 * D * sizeof(float));
    auto gates = vec_[10 * D];
    for (uint32_t job = GetBlockIdx(); job < batch_ * heads_; job += GetBlockNum()) {
      uint32_t b = job / heads_, h = job % heads_;
      int32_t slot = indexGm_.GetValue(b);
      if (slot < 0 || static_cast<uint32_t>(slot) >= blocks_) {
        continue;
      }
      uint64_t offset = (static_cast<uint64_t>(slot) * heads_ + h) * PACKET;
      DataCopy(packet_[DECAY], packetGm_[offset + DECAY], PACKET - DECAY);
      DataCopy(gates, gGm[job / 8 * 8], 8);
      LoadSync();
      if (tag_.GetValue(0) != MAGIC || counter_.GetValue(0) < 0 || counter_.GetValue(0) >= W) {
        continue;
      }
      uint32_t count = counter_.GetValue(0);
      float currentLog = decay_.GetValue(0) + gates.GetValue(job % 8);
      if (count + 1 == W) {
        DataCopy(packet_, packetGm_[offset], PACKET);
      }
      uint64_t kOff = (static_cast<uint64_t>(b) * keyHeads + h / (heads_ / keyHeads)) * D;
      DataCopy(io, kGm[kOff], D);
      LoadSync();
      auto key = vec_[D];
      Cast(key, io, RoundMode::CAST_NONE, D);
      VSync();
      Cast(keys_[count * D], key, RoundMode::CAST_RINT, D);
      VSync();
      logTimes_.SetValue(count, currentLog);
      SV();
      if (count + 1 == W) {
        MaterializeResidual();
        auto kr = vec_[24 * D], vr = vec_[48 * D];
        Cast(kr, keys_, RoundMode::CAST_NONE, W * D);
        Cast(vr, values_, RoundMode::CAST_NONE, W * D);
        Muls(coefficients_[8], logTimes_, -1.f, W);
        VSync();
        Adds(coefficients_[8], coefficients_[8], currentLog, W);
        VSync();
        Exp(coefficients_[8], coefficients_[8], W);
        Duplicate(gates, currentLog, 8);
        VSync();
        Exp(gates, gates, 8);
        VSync();
        VS();
        float decay = gates.GetValue(0);
        Reconstruct(decay, kr, vr);
        Fold(true);
        StoreSync();
        DataCopy(packetGm_[offset], packet_, PACKET);
      } else {
        decay_.SetValue(0, currentLog);
        counter_.SetValue(0, count + 1);
        StoreSync();
        DataCopy(packetGm_[offset + KEYS + count * D * sizeof(half)], packet_[KEYS + count * D * sizeof(half)],
                 D * sizeof(half));
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
  __aicore__ inline void ReadResidualPair(LocalTensor<float> pred, LocalTensor<float> output, LocalTensor<float> key,
                                          LocalTensor<float> query) {
    auto scaledK = vec_[66 * D], scaledQ = vec_[67 * D];
    auto partK = vec_[3 * D], partQ = vec_[4 * D];
    Mul(scaledK, key, smooth_, D);
    Mul(scaledQ, query, smooth_, D);
    VSync();
    for (uint32_t col = 0; col < D; col += 64) {
      auto packed = stateBuf_.Get<int8_t>();
      auto halfTmp = workBuf_.Get<half>();
      DataCopyParams copy{D, 2, 2, 0};
      DataCopy(packed, packetBuf_.Get<int8_t>()[col], copy);
      VSync();
      Cast(halfTmp, packed, RoundMode::CAST_NONE, N / 2);
      VSync();
      Cast(s_, halfTmp, RoundMode::CAST_NONE, N / 2);
      VSync();
      Mul(work_, s_, scaledK[col], 64, D, {1, 1, 1, 8, 8, 0});
      VSync();
      WholeReduceSum(partK, work_, 64, D, 1, 1, 8);
      VSync();
      Mul(work_, s_, scaledQ[col], 64, D, {1, 1, 1, 8, 8, 0});
      VSync();
      WholeReduceSum(partQ, work_, 64, D, 1, 1, 8);
      VSync();
      if (col == 0) {
        DataCopy(pred, partK, D);
        DataCopy(output, partQ, D);
      } else {
        Add(pred, pred, partK, D);
        Add(output, output, partQ, D);
      }
      VSync();
    }
    Mul(pred, pred, scale_, D);
    Mul(output, output, scale_, D);
    VSync();
  }
  __aicore__ inline void ReadBase(LocalTensor<float> dst, LocalTensor<float> vector) {
    auto scaled = vec_[66 * D];
    Mul(scaled, vector, smooth_, D);
    VSync();
    MatRows(dst, s_, scaled, D);
    Mul(dst, dst, scale_, D);
    VSync();
  }
  __aicore__ inline void JointRead(LocalTensor<float> dst, LocalTensor<float> weights, LocalTensor<float> vector,
                                   uint32_t count, bool activeOnly = false) {
    // Four compensators and four zero padding rows precede the 16 real
    // records. Shared contraction removes two pairs of small matrix ops.
    Duplicate(weights, 0.f, D);
    VSync();
    MatRows(weights, vec_[16 * D], vector, 8 + count);
    uint32_t rows = activeOnly ? 8 + count : 24;
    Mul(weights, weights, coefficients_, rows);
    VSync();
    MatCols(dst, vec_[40 * D], weights, rows);
  }
  __aicore__ inline void ReadTile(LocalTensor<float> dst, LocalTensor<float> vector, LocalTensor<float> weights,
                                  uint32_t count, uint32_t tile, float decay) {
    auto scaled = vec_[66 * D], repeated = vec_[68 * D];
    Mul(scaled, vector, smooth_, D);
    VSync();
    MatRows(dst, s_, scaled, tile);
    Mul(dst, dst, scale_, tile);
    VSync();
    Muls(dst, dst, decay, tile);
    Duplicate(weights, 0.f, D);
    VSync();
    MatRows(weights, vec_[16 * D], vector, 8 + count);
    Mul(weights, weights, coefficients_, 24);
    VSync();
    Brcb(repeated, weights, 3, {1, 8});
    VSync();
    Mul(work_, vec_[40 * D], repeated, tile, 24,
        {1, 1, 0, static_cast<uint8_t>(tile / 8), static_cast<uint8_t>(tile / 8), 1});
    VSync();
    Add(work_, work_, work_[16 * tile], 8 * tile);
    VSync();
    for (uint32_t stride = 8; stride; stride /= 2) {
      Add(work_, work_, work_[stride * tile], stride * tile);
      VSync();
    }
    Add(dst, dst, work_, tile);
    VSync();
  }
  __aicore__ inline void Reconstruct(float decay, LocalTensor<float> kr, LocalTensor<float> vr) {
    ScaleRows(s_, s_, scale_, D);
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
    ScaleRows(vr, vr, coefficients_[8], W);
    for (uint32_t w = 0; w < W; ++w) {
      Outer(vr[w * D], kr[w * D], false);
    }
  }
  __aicore__ inline void Fold(bool warm = false) {
    auto u = vec_[3 * D], direction = vec_[4 * D];
    uint32_t coldRestarts = 0;
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
        ++coldRestarts;
        maximum = -1.f;
        Mul(work_, s_, s_, N);
        VSync();
        ColsReduce(direction, D);
        Sqrt(direction, direction, D);
        VSync();
      }
      Normalize(direction, maximum);
      if (restart) {
        for (uint32_t it = 0; it < LEAPQUANT_POWER_ITERS; ++it) {
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
    ElemColumns(work_, s_, smooth_, D, true);
    // RowsReduce uses work_, so preserve normalized residual in s_.
    DataCopy(s_, work_, N);
    VSync();
    Abs(work_, work_, N);
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
    tag_.SetValue(1, coldRestarts);
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

extern "C" uint32_t leapquant_packet_bytes() { return PACKET; }
extern "C" uint32_t leapquant_format_tag() { return MAGIC; }
extern "C" uint32_t leapquant_power_iterations() { return LEAPQUANT_POWER_ITERS; }

extern "C" __global__ __aicore__ void leapquant_tiles(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR beta, GM_ADDR g,
                                                      GM_ADDR packet, GM_ADDR indices, GM_ADDR output, uint32_t batch,
                                                      uint32_t heads, uint32_t keyHeads, uint32_t blocks,
                                                      uint32_t tile) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  TPipe pipe;
  LeapQuant op;
  op.Init(&pipe, packet, indices, batch, heads, blocks, tile);
  op.DecodeTiles(q, k, v, beta, g, output, keyHeads, tile);
}
extern "C" __global__ __aicore__ void leapquant_commit(GM_ADDR k, GM_ADDR g, GM_ADDR packet, GM_ADDR indices,
                                                       uint32_t batch, uint32_t heads, uint32_t keyHeads,
                                                       uint32_t blocks) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  TPipe pipe;
  LeapQuant op;
  op.Init(&pipe, packet, indices, batch, heads, blocks);
  op.CommitTiles(k, g, keyHeads);
}
extern "C" void leapquant_tiled_launch(uint32_t cores, void* stream, void* q, void* k, void* v, void* beta, void* g,
                                       void* packet, void* indices, void* output, uint32_t batch, uint32_t heads,
                                       uint32_t keyHeads, uint32_t blocks, uint32_t tile) {
  leapquant_tiles<<<cores, nullptr, stream>>>((uint8_t*)q, (uint8_t*)k, (uint8_t*)v, (uint8_t*)beta, (uint8_t*)g,
                                              (uint8_t*)packet, (uint8_t*)indices, (uint8_t*)output, batch, heads,
                                              keyHeads, blocks, tile);
  leapquant_commit<<<cores, nullptr, stream>>>((uint8_t*)k, (uint8_t*)g, (uint8_t*)packet, (uint8_t*)indices, batch,
                                               heads, keyHeads, blocks);
}

extern "C" __global__ __aicore__ void leapquant_stream(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR beta, GM_ADDR g,
                                                       GM_ADDR packet, GM_ADDR indices, GM_ADDR output, uint32_t batch,
                                                       uint32_t heads, uint32_t keyHeads, uint32_t blocks) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  TPipe pipe;
  LeapQuant op;
  op.Init(&pipe, packet, indices, batch, heads, blocks);
  op.Decode(q, k, v, beta, g, output, keyHeads, true);
}
extern "C" void leapquant_stream_launch(uint32_t cores, void* stream, void* q, void* k, void* v, void* beta, void* g,
                                        void* packet, void* indices, void* output, uint32_t batch, uint32_t heads,
                                        uint32_t keyHeads, uint32_t blocks) {
  leapquant_stream<<<cores, nullptr, stream>>>((uint8_t*)q, (uint8_t*)k, (uint8_t*)v, (uint8_t*)beta, (uint8_t*)g,
                                               (uint8_t*)packet, (uint8_t*)indices, (uint8_t*)output, batch, heads,
                                               keyHeads, blocks);
}

// Standalone residual-contraction helpers. They do not update recurrent state.
extern "C" __global__ __aicore__ void leapquant_digit_pack(GM_ADDR query, GM_ADDR packed, GM_ADDR scales,
                                                           uint32_t heads) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  TPipe pipe;
  TBuf<QuePosition::VECCALC> dataBuf, packedBuf;
  pipe.InitBuffer(dataBuf, 8192);
  pipe.InitBuffer(packedBuf, 2048);
  auto data = dataBuf.Get<float>();
  auto remaining = data[2 * D], temp = data[3 * D], reduced = data[4 * D];
  auto intTmp = dataBuf.GetWithOffset<int32_t>(D, 5 * D * sizeof(float));
  auto halfTmp = dataBuf.GetWithOffset<half>(D, 6 * D * sizeof(float));
  auto scale = data[7 * D];
  GlobalTensor<float> queryGm, scaleGm;
  GlobalTensor<int8_t> packedGm;
  queryGm.SetGlobalBuffer((__gm__ float*)query);
  scaleGm.SetGlobalBuffer((__gm__ float*)scales);
  packedGm.SetGlobalBuffer((__gm__ int8_t*)packed);
  for (uint32_t head = GetBlockIdx(); head < heads; head += GetBlockNum()) {
    DataCopy(data, queryGm[static_cast<uint64_t>(head) * 2 * D], 2 * D);
    Duplicate(packedBuf.Get<int32_t>(), 0, 16 * D / 4);
    Duplicate(scale, 0.f, 8);
    LoadSync();
    VSync();
    for (uint32_t q = 0; q < 2; ++q) {
      DataCopy(remaining, data[q * D], D);
      Abs(reduced, remaining, D);
      VSync();
      Max(reduced[64], reduced, reduced[64], 64);
      VSync();
      WholeReduceMax(reduced, reduced[64], 64, 1, 1, 1, 8, ReduceOrder::ORDER_ONLY_VALUE);
      VSync();
      VS();
      float maximum = reduced.GetValue(0);
      float unit = (maximum > 1.e-30f ? maximum : 1.e-30f) / 127.f;
      scale.SetValue(q, unit);
      for (uint32_t digit = 0; digit < 4; ++digit) {
        Muls(temp, remaining, 1.f / unit, D);
        VSync();
        Cast(intTmp, temp, RoundMode::CAST_RINT, D);
        VSync();
        SetDeqScale((half)1.f);
        Cast(halfTmp, intTmp, RoundMode::CAST_ROUND, D);
        VSync();
        Cast(packedBuf.Get<int8_t>()[(q * 4 + digit) * D], halfTmp, RoundMode::CAST_TRUNC, D);
        Cast(temp, halfTmp, RoundMode::CAST_NONE, D);
        VSync();
        Muls(temp, temp, unit, D);
        VSync();
        Sub(remaining, remaining, temp, D);
        VSync();
        unit /= 128.f;
      }
    }
    StoreSync();
    DataCopy(packedGm[static_cast<uint64_t>(head) * 16 * D], packedBuf.Get<int8_t>(), 16 * D);
    DataCopy(scaleGm[static_cast<uint64_t>(head) * 8], scale, 8);
    StoreDone();
  }
}

extern "C" __global__ __aicore__ void leapquant_digit_unpack(GM_ADDR products, GM_ADDR scales, GM_ADDR output,
                                                             uint32_t heads) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  TPipe pipe;
  TBuf<QuePosition::VECCALC> integerBuf, dataBuf;
  pipe.InitBuffer(integerBuf, 8 * D * sizeof(int32_t));
  pipe.InitBuffer(dataBuf, 12 * D * sizeof(float));
  auto integer = integerBuf.Get<int32_t>();
  auto data = dataBuf.Get<float>(), temp = data[8 * D], out = data[9 * D], scale = data[10 * D];
  GlobalTensor<int32_t> productGm;
  GlobalTensor<float> scaleGm, outGm;
  productGm.SetGlobalBuffer((__gm__ int32_t*)products);
  scaleGm.SetGlobalBuffer((__gm__ float*)scales);
  outGm.SetGlobalBuffer((__gm__ float*)output);
  for (uint32_t head = GetBlockIdx(); head < heads; head += GetBlockNum()) {
    DataCopy(integer, productGm[static_cast<uint64_t>(head) * 16 * D], 8 * D);
    DataCopy(scale, scaleGm[static_cast<uint64_t>(head) * 8], 8);
    LoadSync();
    Cast(data, integer, RoundMode::CAST_NONE, 8 * D);
    VSync();
    for (uint32_t q = 0; q < 2; ++q) {
      DataCopy(out, data[q * 4 * D], D);
      VSync();
      float multiplier = 1.f / 128.f;
      for (uint32_t digit = 1; digit < 4; ++digit) {
        Muls(temp, data[(q * 4 + digit) * D], multiplier, D);
        VSync();
        Add(out, out, temp, D);
        VSync();
        multiplier /= 128.f;
      }
      Muls(out, out, scale.GetValue(q), D);
      VSync();
      StoreSync();
      DataCopy(outGm[(static_cast<uint64_t>(head) * 2 + q) * D], out, D);
      StoreDone();
    }
  }
}
extern "C" void leapquant_digit_pack_launch(uint32_t cores, void* stream, void* query, void* packed, void* scales,
                                            uint32_t heads) {
  leapquant_digit_pack<<<cores, nullptr, stream>>>((uint8_t*)query, (uint8_t*)packed, (uint8_t*)scales, heads);
}
extern "C" void leapquant_digit_unpack_launch(uint32_t cores, void* stream, void* products, void* scales, void* output,
                                              uint32_t heads) {
  leapquant_digit_unpack<<<cores, nullptr, stream>>>((uint8_t*)products, (uint8_t*)scales, (uint8_t*)output, heads);
}

extern "C" __global__ __aicore__ void leapquant_residual_transpose(GM_ADDR source, GM_ADDR output, uint32_t heads,
                                                                   uint32_t sourceStride) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  TPipe pipe;
  TBuf<QuePosition::VECCALC> bytesBuf, inputBuf, outputBuf;
  pipe.InitBuffer(bytesBuf, N);
  pipe.InitBuffer(inputBuf, N * sizeof(half));
  pipe.InitBuffer(outputBuf, N * sizeof(half));
  auto bytes = bytesBuf.Get<int8_t>();
  auto input = inputBuf.Get<half>(), result = outputBuf.Get<half>();
  GlobalTensor<int8_t> sourceGm, outputGm;
  sourceGm.SetGlobalBuffer((__gm__ int8_t*)source);
  outputGm.SetGlobalBuffer((__gm__ int8_t*)output);
  for (uint32_t head = GetBlockIdx(); head < heads; head += GetBlockNum()) {
    DataCopy(bytes, sourceGm[static_cast<uint64_t>(head) * sourceStride], N);
    LoadSync();
    Cast(input, bytes, RoundMode::CAST_NONE, N);
    VSync();
    for (uint32_t row = 0; row < D; row += 16) {
      LocalTensor<half> srcList[16], dstList[16];
      for (uint32_t i = 0; i < 16; ++i) {
        srcList[i] = input[(row + i) * D];
        dstList[i] = result[i * D + row];
      }
      TransDataTo5HD<half>(dstList, srcList, {false, false, 8, 128, 1});
      VSync();
    }
    Cast(bytes, result, RoundMode::CAST_TRUNC, N);
    VSync();
    StoreSync();
    DataCopy(outputGm[static_cast<uint64_t>(head) * N], bytes, N);
    StoreDone();
  }
}
extern "C" void leapquant_residual_transpose_launch(uint32_t cores, void* stream, void* source, void* output,
                                                    uint32_t heads, uint32_t sourceStride) {
  leapquant_residual_transpose<<<cores, nullptr, stream>>>((uint8_t*)source, (uint8_t*)output, heads, sourceStride);
}
