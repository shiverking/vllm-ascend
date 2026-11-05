// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#include <torch/extension.h>
#include <xlite/xlite.h>
#include <limits>

// Version-pinned declarations from xLite 0.2.0rc4 op.h. These exported
// asynchronous entrypoints avoid the synchronization in its Python SiLU API.
void XliteOpMatmul(XRuntime&, XTensor&, XTensor&, XTensor&, bool, const XTensor&, const XTensor&, bool, uint64_t,
                   uint64_t, uint64_t);
void XliteOpSiluAndMul(XRuntime&, XTensor&, XTensor&, const XTensor&, float);

class StreamGuard {
 public:
  StreamGuard(XRuntime& runtime, uintptr_t stream) : runtime_(runtime), previous_(runtime.stream) {
    runtime_.stream = reinterpret_cast<aclrtStream>(stream);
  }
  ~StreamGuard() { runtime_.stream = previous_; }

 private:
  XRuntime& runtime_;
  aclrtStream previous_;
};

void DecodeMLP(XRuntime& runtime, const at::Tensor& x, const at::Tensor& up_weight, const at::Tensor& down_weight,
               const at::Tensor& up, const at::Tensor& act, const at::Tensor& out, bool up_nz, bool down_nz,
               uintptr_t stream) {
  XTensor input, wu, wd, u, a, o, empty;
  InitXTensor(input, x);
  InitXTensor(wu, up_weight);
  InitXTensor(wd, down_weight);
  InitXTensor(u, up);
  InitXTensor(a, act);
  InitXTensor(o, out);
  StreamGuard guard(runtime, stream);
  constexpr auto automatic = std::numeric_limits<uint64_t>::max();
  XliteOpMatmul(runtime, input, wu, u, up_nz, empty, empty, false, automatic, automatic, automatic);
  XliteOpSiluAndMul(runtime, u, a, empty, 0.0f);
  XliteOpMatmul(runtime, a, wd, o, down_nz, empty, empty, false, automatic, automatic, automatic);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) { module.def("mlp", &DecodeMLP); }
