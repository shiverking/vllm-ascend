/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
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

template <typename T>
class SplitQkvRmsNormMropeKernel {
public:
    __aicore__ inline void Init(
        __gm__ void* qkv, __gm__ void* qWeight, __gm__ void* kWeight,
        __gm__ void* cos, __gm__ void* sin,
        __gm__ void* qOut, __gm__ void* kOut, uint32_t numTokens,
        uint32_t numQHeads,
        uint32_t numKvHeads, uint32_t headSize, uint32_t ropeDim,
        float epsilon, float invHeadSize, uint32_t blockDim)
    {
        numTokens_ = numTokens;
        numQHeads_ = numQHeads;
        numKvHeads_ = numKvHeads;
        headSize_ = headSize;
        ropeDim_ = ropeDim;
        halfRopeDim_ = ropeDim / 2;
        epsilon_ = epsilon;
        invHeadSize_ = invHeadSize;
        blockDim_ = blockDim;
        qSize_ = numQHeads * headSize;
        kvSize_ = numKvHeads * headSize;
        qkvStride_ = qSize_ + 2 * kvSize_;
        workPerToken_ = numQHeads + numKvHeads;
        totalWork_ = numTokens * workPerToken_;

        qkvGm_.SetGlobalBuffer(static_cast<__gm__ T*>(qkv), numTokens * qkvStride_);
        qWeightGm_.SetGlobalBuffer(static_cast<__gm__ T*>(qWeight), headSize);
        kWeightGm_.SetGlobalBuffer(static_cast<__gm__ T*>(kWeight), headSize);
        cosGm_.SetGlobalBuffer(static_cast<__gm__ T*>(cos), numTokens * ropeDim);
        sinGm_.SetGlobalBuffer(static_cast<__gm__ T*>(sin), numTokens * ropeDim);
        qOutGm_.SetGlobalBuffer(static_cast<__gm__ T*>(qOut), numTokens * qSize_);
        kOutGm_.SetGlobalBuffer(static_cast<__gm__ T*>(kOut), numTokens * kvSize_);

        alignedHeadSize_ = AlignElements(headSize, sizeof(T));
        alignedRopeDim_ = AlignElements(ropeDim, sizeof(T));
        pipe_.InitBuffer(inputBuf_, alignedHeadSize_ * sizeof(T));
        pipe_.InitBuffer(weightBuf_, alignedHeadSize_ * sizeof(T));
        pipe_.InitBuffer(weightFloatBuf_, alignedHeadSize_ * sizeof(float));
        pipe_.InitBuffer(normalizedBuf_, alignedHeadSize_ * sizeof(float));
        pipe_.InitBuffer(normBuf_, BYTES_PER_BLOCK);
        pipe_.InitBuffer(rotatedBuf_, alignedHeadSize_ * sizeof(float));
        pipe_.InitBuffer(outputBuf_, alignedHeadSize_ * sizeof(T));
        pipe_.InitBuffer(cosBuf_, alignedRopeDim_ * sizeof(T));
        pipe_.InitBuffer(sinBuf_, alignedRopeDim_ * sizeof(T));
        pipe_.InitBuffer(cosFloatBuf_, alignedRopeDim_ * sizeof(float));
        pipe_.InitBuffer(sinFloatBuf_, alignedRopeDim_ * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        for (uint32_t work = GetBlockIdx(); work < totalWork_; work += blockDim_) {
            uint32_t token = work / workPerToken_;
            uint32_t task = work % workPerToken_;
            if (task < numQHeads_) {
                ProcessQ(token, task);
            } else {
                ProcessK(token, task - numQHeads_);
            }
        }
    }

private:
    static __aicore__ inline uint32_t AlignElements(uint32_t count, uint32_t elementSize)
    {
        uint32_t elementsPerBlock = BYTES_PER_BLOCK / elementSize;
        return (count + elementsPerBlock - 1) / elementsPerBlock * elementsPerBlock;
    }

    __aicore__ inline void LoadCosSin(uint32_t token)
    {
        LocalTensor<T> cos = cosBuf_.Get<T>();
        LocalTensor<T> sin = sinBuf_.Get<T>();
        uint32_t offset = token * ropeDim_;
        DataCopy(cos, cosGm_[offset], alignedRopeDim_);
        DataCopy(sin, sinGm_[offset], alignedRopeDim_);
        pipe_barrier(PIPE_ALL);
        Cast(cosFloatBuf_.Get<float>(), cos, RoundMode::CAST_NONE, alignedRopeDim_);
        Cast(sinFloatBuf_.Get<float>(), sin, RoundMode::CAST_NONE, alignedRopeDim_);
        pipe_barrier(PIPE_ALL);
    }

    __aicore__ inline void NormalizeRotate(
        GlobalTensor<T>& output, uint32_t inputOffset,
        uint32_t outputOffset, bool useQWeight)
    {
        LocalTensor<T> input = inputBuf_.Get<T>();
        LocalTensor<T> weight = weightBuf_.Get<T>();
        LocalTensor<float> weightFloat = weightFloatBuf_.Get<float>();
        LocalTensor<float> normalized = normalizedBuf_.Get<float>();
        LocalTensor<float> norm = normBuf_.Get<float>();
        LocalTensor<float> rotated = rotatedBuf_.Get<float>();
        LocalTensor<T> outputLocal = outputBuf_.Get<T>();
        DataCopy(input, qkvGm_[inputOffset], alignedHeadSize_);
        if (useQWeight) {
            DataCopy(weight, qWeightGm_, alignedHeadSize_);
        } else {
            DataCopy(weight, kWeightGm_, alignedHeadSize_);
        }
        pipe_barrier(PIPE_ALL);
        Cast(normalized, input, RoundMode::CAST_NONE, alignedHeadSize_);
        Cast(weightFloat, weight, RoundMode::CAST_NONE, alignedHeadSize_);
        pipe_barrier(PIPE_ALL);

        float squareSum = 0.0F;
        for (uint32_t dim = 0; dim < headSize_; ++dim) {
            float value = normalized.GetValue(dim);
            squareSum += value * value;
        }
        norm.SetValue(0, squareSum * invHeadSize_ + epsilon_);
        pipe_barrier(PIPE_ALL);
        Sqrt(norm, norm, 1);
        pipe_barrier(PIPE_ALL);
        float reciprocalStd = 1.0F / norm.GetValue(0);
        for (uint32_t dim = 0; dim < headSize_; ++dim) {
            normalized.SetValue(dim, normalized.GetValue(dim) * reciprocalStd * weightFloat.GetValue(dim));
        }
        for (uint32_t dim = 0; dim < headSize_; ++dim) {
            float result = normalized.GetValue(dim);
            if (dim < ropeDim_) {
                uint32_t pair = dim < halfRopeDim_ ? dim + halfRopeDim_ : dim - halfRopeDim_;
                float pairValue = normalized.GetValue(pair);
                if (dim < halfRopeDim_) pairValue = -pairValue;
                result = result * cosFloatBuf_.Get<float>().GetValue(dim) +
                         pairValue * sinFloatBuf_.Get<float>().GetValue(dim);
            }
            rotated.SetValue(dim, result);
        }
        pipe_barrier(PIPE_ALL);
        // Ascend 310P supports FP32 -> FP16 with CAST_NONE only.
        Cast(outputLocal, rotated, RoundMode::CAST_NONE, alignedHeadSize_);
        pipe_barrier(PIPE_ALL);
        DataCopy(output[outputOffset], outputLocal, alignedHeadSize_);
    }

    __aicore__ inline void ProcessQ(uint32_t token, uint32_t head)
    {
        LoadCosSin(token);
        NormalizeRotate(qOutGm_, token * qkvStride_ + head * headSize_,
                        token * qSize_ + head * headSize_, true);
    }

    __aicore__ inline void ProcessK(uint32_t token, uint32_t head)
    {
        LoadCosSin(token);
        NormalizeRotate(kOutGm_, token * qkvStride_ + qSize_ + head * headSize_,
                        token * kvSize_ + head * headSize_, false);
    }

    TPipe pipe_;
    TBuf<TPosition::VECCALC> inputBuf_, weightBuf_, weightFloatBuf_;
    TBuf<TPosition::VECCALC> normalizedBuf_, normBuf_, rotatedBuf_, outputBuf_;
    TBuf<TPosition::VECCALC> cosBuf_, sinBuf_, cosFloatBuf_, sinFloatBuf_;
    GlobalTensor<T> qkvGm_, qWeightGm_, kWeightGm_, cosGm_, sinGm_;
    GlobalTensor<T> qOutGm_, kOutGm_;
    uint32_t numTokens_, numQHeads_, numKvHeads_, headSize_, ropeDim_, halfRopeDim_;
    uint32_t qSize_, kvSize_, qkvStride_, workPerToken_, totalWork_, blockDim_;
    uint32_t alignedHeadSize_, alignedRopeDim_;
    float epsilon_, invHeadSize_;
};

template <typename T>
__aicore__ inline void RunKernel(
    __gm__ void* qkv, __gm__ void* qWeight, __gm__ void* kWeight,
    __gm__ void* cos, __gm__ void* sin, __gm__ void* qOut,
    __gm__ void* kOut, uint32_t numTokens, uint32_t numQHeads,
    uint32_t numKvHeads,
    uint32_t headSize, uint32_t ropeDim, float epsilon, float invHeadSize,
    uint32_t blockDim)
{
    SplitQkvRmsNormMropeKernel<T> op;
    op.Init(qkv, qWeight, kWeight, cos, sin, qOut, kOut,
            numTokens, numQHeads, numKvHeads, headSize, ropeDim,
            epsilon, invHeadSize, blockDim);
    op.Process();
}

}  // namespace

#define MROPE_KERNEL_ARGS \
    __gm__ void* qkv, __gm__ void* qWeight, __gm__ void* kWeight, \
    __gm__ void* cos, __gm__ void* sin, __gm__ void* qOut, \
    __gm__ void* kOut, uint32_t numTokens, uint32_t numQHeads, \
    uint32_t numKvHeads, \
    uint32_t headSize, uint32_t ropeDim, float epsilon, float invHeadSize, \
    uint32_t blockDim

#define MROPE_KERNEL_CALL \
    qkv, qWeight, kWeight, cos, sin, qOut, kOut, numTokens, \
    numQHeads, numKvHeads, headSize, ropeDim, epsilon, invHeadSize, blockDim

extern "C" __global__ __aicore__ void split_qkv_rmsnorm_mrope_fp16_kernel(MROPE_KERNEL_ARGS)
{
    RunKernel<half>(MROPE_KERNEL_CALL);
}

namespace vllm_ascend {

void split_qkv_rmsnorm_mrope_impl(
    void* stream, void* qkv, void* qWeight, void* kWeight,
    void* cos, void* sin, void* qOut, void* kOut, uint32_t numTokens,
    uint32_t numQHeads,
    uint32_t numKvHeads, uint32_t headSize, uint32_t ropeDim, float epsilon,
    uint32_t blockDim)
{
    float invHeadSize = 1.0F / static_cast<float>(headSize);
    split_qkv_rmsnorm_mrope_fp16_kernel<<<blockDim, nullptr, stream>>>(
        qkv, qWeight, kWeight, cos, sin, qOut, kOut, numTokens,
        numQHeads, numKvHeads, headSize, ropeDim, epsilon, invHeadSize,
        blockDim);
}

}  // namespace vllm_ascend
