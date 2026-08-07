# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from unittest.mock import MagicMock, patch

import torch

from tests.ut.base import TestBase
from vllm_ascend._310p.quantization.methods.w8a8_dynamic import (
    AscendW8A8DynamicLinearMethod310,
)


class TestAscendW8A8DynamicLinearLayout310(TestBase):
    def setUp(self):
        self.method = AscendW8A8DynamicLinearMethod310()

    @patch("torch_npu.npu_dynamic_quant", create=True)
    @patch("torch_npu.npu_quant_matmul")
    def test_2d_input_is_not_squeezed_or_unsqueezed(self, mock_npu_quant_matmul, mock_npu_dynamic_quant):
        layer = MagicMock()
        layer.weight = torch.randint(-128, 127, (128, 256), dtype=torch.int8)
        layer.weight_scale = torch.randn(256, dtype=torch.float32)

        x = torch.randn(32, 128, dtype=torch.float16)
        quantized_x = torch.randint(-128, 127, x.shape, dtype=torch.int8)
        pertoken_scale = torch.randn(32, 1, dtype=torch.float32)
        mock_npu_dynamic_quant.return_value = quantized_x, pertoken_scale
        expected_output = torch.randn(32, 256, dtype=torch.float16)
        mock_npu_quant_matmul.return_value = expected_output

        output = self.method.apply(layer, x)

        args, kwargs = mock_npu_quant_matmul.call_args
        self.assertEqual(args[0].shape, (32, 128))
        self.assertEqual(kwargs["pertoken_scale"].shape, (32, 1))
        self.assertEqual(output.shape, (32, 256))
