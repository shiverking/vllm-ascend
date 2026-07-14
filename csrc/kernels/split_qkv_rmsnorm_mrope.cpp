/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "kernel_operator.h"

using namespace AscendC;

namespace {

constexpr uint32_t BYTES_PER_BLOCK = 32;

class SplitQkvRmsNormMropeKernel {
public:
    __aicore__ inline SplitQkvRmsNormMropeKernel() = default;

    __aicore__ inline void Init(
        __gm__ void* qkv,
        __gm__ void* qWeight,
        __gm__ void* kWeight,
        __gm__ void* cosSinCache,
        __gm__ void* positions,
        __gm__ void* qOut,
        __gm__ void* kOut,
        __gm__ void* vOut,
        uint32_t numTokens,
        uint32_t maxPositions,
        uint32_t numQHeads,
        uint32_t numKvHeads,
        uint32_t headSize,
        uint32_t ropeDim,
        float epsilon,
        uint32_t mropeSectionT,
        uint32_t mropeSectionH,
        uint32_t mropeSectionW,
        uint32_t isInterleaved,
        uint32_t blockDim)
    {
        numTokens_ = numTokens;
        maxPositions_ = maxPositions;
        numQHeads_ = numQHeads;
        numKvHeads_ = numKvHeads;
        headSize_ = headSize;
        ropeDim_ = ropeDim;
        halfRopeDim_ = ropeDim / 2;
        epsilon_ = epsilon;
        mropeSectionT_ = mropeSectionT;
        mropeSectionH_ = mropeSectionH;
        mropeSectionW_ = mropeSectionW;
        isInterleaved_ = isInterleaved != 0;
        blockDim_ = blockDim;

        qSize_ = numQHeads_ * headSize_;
        kvSize_ = numKvHeads_ * headSize_;
        qkvStride_ = qSize_ + 2 * kvSize_;
        workPerToken_ = numQHeads_ + 2 * numKvHeads_;
        totalWork_ = numTokens_ * workPerToken_;

        qkvGlobal_.SetGlobalBuffer(static_cast<__gm__ bfloat16_t*>(qkv), numTokens_ * qkvStride_);
        qWeightGlobal_.SetGlobalBuffer(static_cast<__gm__ bfloat16_t*>(qWeight), headSize_);
        kWeightGlobal_.SetGlobalBuffer(static_cast<__gm__ bfloat16_t*>(kWeight), headSize_);
        cosSinCacheGlobal_.SetGlobalBuffer(
            static_cast<__gm__ bfloat16_t*>(cosSinCache), maxPositions_ * ropeDim_);
        positionsGlobal_.SetGlobalBuffer(static_cast<__gm__ int64_t*>(positions), 3 * numTokens_);
        qOutGlobal_.SetGlobalBuffer(static_cast<__gm__ bfloat16_t*>(qOut), numTokens_ * qSize_);
        kOutGlobal_.SetGlobalBuffer(static_cast<__gm__ bfloat16_t*>(kOut), numTokens_ * kvSize_);
        vOutGlobal_.SetGlobalBuffer(static_cast<__gm__ bfloat16_t*>(vOut), numTokens_ * kvSize_);

        alignedHeadSize_ = AlignElements(headSize_, sizeof(bfloat16_t));
        alignedRopeDim_ = AlignElements(ropeDim_, sizeof(bfloat16_t));

        pipe_.InitBuffer(inputBuf_, alignedHeadSize_ * sizeof(bfloat16_t));
        pipe_.InitBuffer(weightBuf_, alignedHeadSize_ * sizeof(bfloat16_t));
        pipe_.InitBuffer(weightFloatBuf_, alignedHeadSize_ * sizeof(float));
        pipe_.InitBuffer(normalizedBuf_, alignedHeadSize_ * sizeof(float));
        pipe_.InitBuffer(rotatedBuf_, alignedHeadSize_ * sizeof(float));
        pipe_.InitBuffer(outputBuf_, alignedHeadSize_ * sizeof(bfloat16_t));
        pipe_.InitBuffer(tCacheBuf_, alignedRopeDim_ * sizeof(bfloat16_t));
        pipe_.InitBuffer(hCacheBuf_, alignedRopeDim_ * sizeof(bfloat16_t));
        pipe_.InitBuffer(wCacheBuf_, alignedRopeDim_ * sizeof(bfloat16_t));
        pipe_.InitBuffer(tCacheFloatBuf_, alignedRopeDim_ * sizeof(float));
        pipe_.InitBuffer(hCacheFloatBuf_, alignedRopeDim_ * sizeof(float));
        pipe_.InitBuffer(wCacheFloatBuf_, alignedRopeDim_ * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        for (uint32_t work = GetBlockIdx(); work < totalWork_; work += blockDim_) {
            const uint32_t tokenIdx = work / workPerToken_;
            const uint32_t taskIdx = work % workPerToken_;

            if (taskIdx < numQHeads_) {
                ProcessQHead(tokenIdx, taskIdx);
            } else if (taskIdx < numQHeads_ + numKvHeads_) {
                ProcessKHead(tokenIdx, taskIdx - numQHeads_);
            } else {
                CopyVHead(tokenIdx, taskIdx - numQHeads_ - numKvHeads_);
            }
        }
    }

private:
    static __aicore__ inline uint32_t AlignElements(uint32_t count, uint32_t elementSize)
    {
        const uint32_t elementsPerBlock = BYTES_PER_BLOCK / elementSize;
        return (count + elementsPerBlock - 1) / elementsPerBlock * elementsPerBlock;
    }

    __aicore__ inline void LoadMropeCache(uint32_t tokenIdx)
    {
        const int64_t tPosition = positionsGlobal_.GetValue(tokenIdx);
        const int64_t hPosition = positionsGlobal_.GetValue(numTokens_ + tokenIdx);
        const int64_t wPosition = positionsGlobal_.GetValue(2 * numTokens_ + tokenIdx);

        LocalTensor<bfloat16_t> tCache = tCacheBuf_.Get<bfloat16_t>();
        LocalTensor<bfloat16_t> hCache = hCacheBuf_.Get<bfloat16_t>();
        LocalTensor<bfloat16_t> wCache = wCacheBuf_.Get<bfloat16_t>();

        DataCopy(tCache, cosSinCacheGlobal_[tPosition * ropeDim_], alignedRopeDim_);
        DataCopy(hCache, cosSinCacheGlobal_[hPosition * ropeDim_], alignedRopeDim_);
        DataCopy(wCache, cosSinCacheGlobal_[wPosition * ropeDim_], alignedRopeDim_);
        pipe_barrier(PIPE_ALL);

        Cast(tCacheFloatBuf_.Get<float>(), tCache, RoundMode::CAST_NONE, alignedRopeDim_);
        Cast(hCacheFloatBuf_.Get<float>(), hCache, RoundMode::CAST_NONE, alignedRopeDim_);
        Cast(wCacheFloatBuf_.Get<float>(), wCache, RoundMode::CAST_NONE, alignedRopeDim_);
        pipe_barrier(PIPE_ALL);
    }

    __aicore__ inline uint32_t MropeAxis(uint32_t dim) const
    {
        if (isInterleaved_) {
            if ((dim % 3 == 1) && (dim <= 3 * mropeSectionH_)) {
                return 1;
            }
            if ((dim % 3 == 2) && (dim <= 3 * mropeSectionW_)) {
                return 2;
            }
            return 0;
        }

        if (dim < mropeSectionT_) {
            return 0;
        }
        if (dim < mropeSectionT_ + mropeSectionH_) {
            return 1;
        }
        return 2;
    }

    __aicore__ inline float CacheValue(uint32_t axis, uint32_t offset)
    {
        if (axis == 0) {
            return tCacheFloatBuf_.Get<float>().GetValue(offset);
        }
        if (axis == 1) {
            return hCacheFloatBuf_.Get<float>().GetValue(offset);
        }
        return wCacheFloatBuf_.Get<float>().GetValue(offset);
    }

    __aicore__ inline void NormalizeAndRotate(
        GlobalTensor<bfloat16_t>& output,
        uint32_t inputOffset,
        uint32_t outputOffset,
        bool useQWeight)
    {
        LocalTensor<bfloat16_t> input = inputBuf_.Get<bfloat16_t>();
        LocalTensor<bfloat16_t> weight = weightBuf_.Get<bfloat16_t>();
        LocalTensor<float> weightFloat = weightFloatBuf_.Get<float>();
        LocalTensor<float> normalized = normalizedBuf_.Get<float>();
        LocalTensor<float> rotatedOutput = rotatedBuf_.Get<float>();
        LocalTensor<bfloat16_t> outputLocal = outputBuf_.Get<bfloat16_t>();

        DataCopy(input, qkvGlobal_[inputOffset], alignedHeadSize_);
        if (useQWeight) {
            DataCopy(weight, qWeightGlobal_, alignedHeadSize_);
        } else {
            DataCopy(weight, kWeightGlobal_, alignedHeadSize_);
        }
        pipe_barrier(PIPE_ALL);

        Cast(normalized, input, RoundMode::CAST_NONE, alignedHeadSize_);
        Cast(weightFloat, weight, RoundMode::CAST_NONE, alignedHeadSize_);
        pipe_barrier(PIPE_ALL);

        float squareSum = 0.0F;
        for (uint32_t dim = 0; dim < headSize_; ++dim) {
            const float value = normalized.GetValue(dim);
            squareSum += value * value;
        }

        normalized.SetValue(0, squareSum / static_cast<float>(headSize_) + epsilon_);
        pipe_barrier(PIPE_ALL);
        Sqrt(normalized, normalized, 1);
        pipe_barrier(PIPE_ALL);
        const float reciprocalStd = 1.0F / normalized.GetValue(0);

        for (uint32_t dim = 0; dim < headSize_; ++dim) {
            normalized.SetValue(
                dim, normalized.GetValue(dim) * reciprocalStd * weightFloat.GetValue(dim));
        }

        for (uint32_t dim = 0; dim < headSize_; ++dim) {
            float result = normalized.GetValue(dim);
            if (dim < ropeDim_) {
                const uint32_t ropeOffset = dim % halfRopeDim_;
                const uint32_t axis = MropeAxis(ropeOffset);
                const float cosValue = CacheValue(axis, ropeOffset);
                const float sinValue = CacheValue(axis, halfRopeDim_ + ropeOffset);
                const uint32_t pairedDim = dim < halfRopeDim_ ? dim + halfRopeDim_ : dim - halfRopeDim_;
                float rotated = normalized.GetValue(pairedDim);
                if (dim < halfRopeDim_) {
                    rotated = -rotated;
                }
                result = result * cosValue + rotated * sinValue;
            }
            rotatedOutput.SetValue(dim, result);
        }

        pipe_barrier(PIPE_ALL);
        Cast(outputLocal, rotatedOutput, RoundMode::CAST_RINT, alignedHeadSize_);
        pipe_barrier(PIPE_ALL);
        DataCopy(output[outputOffset], outputLocal, alignedHeadSize_);
    }

    __aicore__ inline void ProcessQHead(uint32_t tokenIdx, uint32_t headIdx)
    {
        LoadMropeCache(tokenIdx);
        const uint32_t inputOffset = tokenIdx * qkvStride_ + headIdx * headSize_;
        const uint32_t outputOffset = tokenIdx * qSize_ + headIdx * headSize_;
        NormalizeAndRotate(qOutGlobal_, inputOffset, outputOffset, true);
    }

    __aicore__ inline void ProcessKHead(uint32_t tokenIdx, uint32_t headIdx)
    {
        LoadMropeCache(tokenIdx);
        const uint32_t inputOffset = tokenIdx * qkvStride_ + qSize_ + headIdx * headSize_;
        const uint32_t outputOffset = tokenIdx * kvSize_ + headIdx * headSize_;
        NormalizeAndRotate(kOutGlobal_, inputOffset, outputOffset, false);
    }

    __aicore__ inline void CopyVHead(uint32_t tokenIdx, uint32_t headIdx)
    {
        LocalTensor<bfloat16_t> input = inputBuf_.Get<bfloat16_t>();
        const uint32_t inputOffset = tokenIdx * qkvStride_ + qSize_ + kvSize_ + headIdx * headSize_;
        const uint32_t outputOffset = tokenIdx * kvSize_ + headIdx * headSize_;
        DataCopy(input, qkvGlobal_[inputOffset], alignedHeadSize_);
        pipe_barrier(PIPE_ALL);
        DataCopy(vOutGlobal_[outputOffset], input, alignedHeadSize_);
    }

    TPipe pipe_;
    TBuf<TPosition::VECCALC> inputBuf_;
    TBuf<TPosition::VECCALC> weightBuf_;
    TBuf<TPosition::VECCALC> weightFloatBuf_;
    TBuf<TPosition::VECCALC> normalizedBuf_;
    TBuf<TPosition::VECCALC> rotatedBuf_;
    TBuf<TPosition::VECCALC> outputBuf_;
    TBuf<TPosition::VECCALC> tCacheBuf_;
    TBuf<TPosition::VECCALC> hCacheBuf_;
    TBuf<TPosition::VECCALC> wCacheBuf_;
    TBuf<TPosition::VECCALC> tCacheFloatBuf_;
    TBuf<TPosition::VECCALC> hCacheFloatBuf_;
    TBuf<TPosition::VECCALC> wCacheFloatBuf_;

    GlobalTensor<bfloat16_t> qkvGlobal_;
    GlobalTensor<bfloat16_t> qWeightGlobal_;
    GlobalTensor<bfloat16_t> kWeightGlobal_;
    GlobalTensor<bfloat16_t> cosSinCacheGlobal_;
    GlobalTensor<int64_t> positionsGlobal_;
    GlobalTensor<bfloat16_t> qOutGlobal_;
    GlobalTensor<bfloat16_t> kOutGlobal_;
    GlobalTensor<bfloat16_t> vOutGlobal_;

    uint32_t numTokens_;
    uint32_t maxPositions_;
    uint32_t numQHeads_;
    uint32_t numKvHeads_;
    uint32_t headSize_;
    uint32_t ropeDim_;
    uint32_t halfRopeDim_;
    uint32_t qSize_;
    uint32_t kvSize_;
    uint32_t qkvStride_;
    uint32_t workPerToken_;
    uint32_t totalWork_;
    uint32_t blockDim_;
    uint32_t alignedHeadSize_;
    uint32_t alignedRopeDim_;
    uint32_t mropeSectionT_;
    uint32_t mropeSectionH_;
    uint32_t mropeSectionW_;
    float epsilon_;
    bool isInterleaved_;
};

}  // namespace

extern "C" __global__ __aicore__ void split_qkv_rmsnorm_mrope_kernel(
    __gm__ void* qkv,
    __gm__ void* qWeight,
    __gm__ void* kWeight,
    __gm__ void* cosSinCache,
    __gm__ void* positions,
    __gm__ void* qOut,
    __gm__ void* kOut,
    __gm__ void* vOut,
    uint32_t numTokens,
    uint32_t maxPositions,
    uint32_t numQHeads,
    uint32_t numKvHeads,
    uint32_t headSize,
    uint32_t ropeDim,
    float epsilon,
    uint32_t mropeSectionT,
    uint32_t mropeSectionH,
    uint32_t mropeSectionW,
    uint32_t isInterleaved,
    uint32_t blockDim)
{
    SplitQkvRmsNormMropeKernel op;
    op.Init(qkv, qWeight, kWeight, cosSinCache, positions, qOut, kOut, vOut,
            numTokens, maxPositions, numQHeads, numKvHeads, headSize, ropeDim,
            epsilon, mropeSectionT, mropeSectionH, mropeSectionW, isInterleaved,
            blockDim);
    op.Process();
}

namespace vllm_ascend {

void split_qkv_rmsnorm_mrope_impl(
    void* stream,
    void* qkv,
    void* qWeight,
    void* kWeight,
    void* cosSinCache,
    void* positions,
    void* qOut,
    void* kOut,
    void* vOut,
    uint32_t numTokens,
    uint32_t maxPositions,
    uint32_t numQHeads,
    uint32_t numKvHeads,
    uint32_t headSize,
    uint32_t ropeDim,
    float epsilon,
    uint32_t mropeSectionT,
    uint32_t mropeSectionH,
    uint32_t mropeSectionW,
    bool isInterleaved,
    uint32_t blockDim)
{
    split_qkv_rmsnorm_mrope_kernel<<<blockDim, nullptr, stream>>>(
        qkv, qWeight, kWeight, cosSinCache, positions, qOut, kOut, vOut,
        numTokens, maxPositions, numQHeads, numKvHeads, headSize, ropeDim,
        epsilon, mropeSectionT, mropeSectionH, mropeSectionW,
        static_cast<uint32_t>(isInterleaved), blockDim);
}

}  // namespace vllm_ascend
