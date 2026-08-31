import os
from pathlib import Path

import pytest

from tests.e2e.conftest import RemoteOpenAIServer


MODEL_PATH = os.getenv("QWEN3_ASR_MTP5_MODEL_PATH") or os.getenv(
    "QWEN3_ASR_MTP_MODEL_PATH"
)
TEST_AUDIO = os.getenv("QWEN3_ASR_MTP_TEST_AUDIO")

pytestmark = pytest.mark.skipif(
    not MODEL_PATH or not TEST_AUDIO,
    reason="QWEN3_ASR_MTP5_MODEL_PATH and QWEN3_ASR_MTP_TEST_AUDIO are required",
)


def _transcribe(num_speculative_tokens: int | None) -> str:
    args = [
        "--dtype",
        "bfloat16",
        "--tensor-parallel-size",
        "1",
        "--max-model-len",
        "8192",
        "--max-num-seqs",
        "1",
        "--served-model-name",
        "qwen3-asr-mtp5",
    ]
    if num_speculative_tokens is not None:
        args.extend(
            [
                "--speculative-config",
                (
                    '{"method":"mtp","num_speculative_tokens":'
                    f"{num_speculative_tokens}}}"
                ),
            ]
        )
    with RemoteOpenAIServer(MODEL_PATH, args) as server:
        client = server.get_client()
        with Path(TEST_AUDIO).open("rb") as audio:
            response = client.audio.transcriptions.create(
                model="qwen3-asr-mtp5",
                file=(Path(TEST_AUDIO).name, audio.read(), "audio/wav"),
                temperature=0,
            )
        assert response.text
        return response.text


@pytest.mark.parametrize("num_speculative_tokens", [3, 4, 5])
def test_qwen3_asr_mtp5_matches_greedy_target_on_310p(
    num_speculative_tokens: int,
):
    baseline = _transcribe(num_speculative_tokens=None)
    mtp = _transcribe(num_speculative_tokens=num_speculative_tokens)
    assert mtp == baseline
