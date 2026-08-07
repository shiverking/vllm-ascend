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
from vllm_ascend.utils import ACL_FORMAT_FRACTAL_ND, ACL_FORMAT_FRACTAL_NZ


class TestAscendW8A8DynamicLinearLayout310(TestBase):
    def setUp(self):
        self.method = AscendW8A8DynamicLinearMethod310()

    @patch("torch_npu.npu_dynamic_quant", create=True)
    @patch("torch_npu.get_npu_format", return_value=ACL_FORMAT_FRACTAL_ND, create=True)
    @patch("torch_npu.npu_format_cast")
    @patch("torch_npu.npu_quant_matmul")
    def test_2d_input_is_not_squeezed_or_unsqueezed(
        self,
        mock_npu_quant_matmul,
        mock_npu_format_cast,
        _mock_get_npu_format,
        mock_npu_dynamic_quant,
    ):
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
        mock_npu_format_cast.assert_not_called()

    @patch("torch_npu.npu_dynamic_quant", create=True)
    @patch("torch_npu.get_npu_format", return_value=ACL_FORMAT_FRACTAL_NZ, create=True)
    @patch("torch_npu.npu_format_cast")
    @patch("torch_npu.npu_quant_matmul")
    def test_nz_activation_is_converted_to_nd_before_quant_matmul(
        self,
        mock_npu_quant_matmul,
        mock_npu_format_cast,
        _mock_get_npu_format,
        mock_npu_dynamic_quant,
    ):
        layer = MagicMock()
        layer.weight = torch.randint(-128, 127, (128, 256), dtype=torch.int8)
        layer.weight_scale = torch.randn(256, dtype=torch.float32)

        x = torch.randn(32, 128, dtype=torch.float16)
        quantized_x = torch.randint(-128, 127, x.shape, dtype=torch.int8)
        quantized_x_nd = quantized_x.clone()
        pertoken_scale = torch.randn(32, 1, dtype=torch.float32)
        mock_npu_dynamic_quant.return_value = quantized_x, pertoken_scale
        mock_npu_format_cast.return_value = quantized_x_nd
        mock_npu_quant_matmul.return_value = torch.randn(32, 256, dtype=torch.float16)

        self.method.apply(layer, x)

        mock_npu_format_cast.assert_called_once_with(quantized_x, ACL_FORMAT_FRACTAL_ND)
        self.assertIs(mock_npu_quant_matmul.call_args.args[0], quantized_x_nd)

    @patch("torch_npu.npu_format_cast")
    def test_weight_is_transposed_and_kept_in_nd(self, mock_npu_format_cast):
        layer = MagicMock()
        loaded_weight = torch.randint(-127, 128, (128, 256), dtype=torch.int8)
        layer.weight.data = loaded_weight
        layer.weight_scale.data = torch.randn(128, 1, dtype=torch.float32)
        layer.weight_offset.data = torch.randn(128, 1, dtype=torch.float32)

        self.method.process_weights_after_loading(layer)

        self.assertEqual(layer.weight.data.shape, (256, 128))
        self.assertTrue(layer.weight.data.is_contiguous())
        self.assertTrue(torch.equal(layer.weight.data, loaded_weight.transpose(0, 1)))
        mock_npu_format_cast.assert_not_called()
