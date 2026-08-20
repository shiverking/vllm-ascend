import os
from pathlib import Path

import pytest

from tests.e2e.conftest import RemoteOpenAIServer


MODEL_PATH = os.getenv("QWEN3_ASR_MTP_MODEL_PATH")
TEST_AUDIO = os.getenv("QWEN3_ASR_MTP_TEST_AUDIO")

pytestmark = pytest.mark.skipif(
    not MODEL_PATH or not TEST_AUDIO,
    reason="QWEN3_ASR_MTP_MODEL_PATH and QWEN3_ASR_MTP_TEST_AUDIO are required",
)


def _transcribe(speculative: bool) -> str:
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
        "qwen3-asr-mtp3",
    ]
    if speculative:
        args.extend(
            [
                "--speculative-config",
                '{"method":"mtp","num_speculative_tokens":3}',
            ]
        )
    with RemoteOpenAIServer(MODEL_PATH, args) as server:
        client = server.get_client()
        with Path(TEST_AUDIO).open("rb") as audio:
            response = client.audio.transcriptions.create(
                model="qwen3-asr-mtp3",
                file=(Path(TEST_AUDIO).name, audio.read(), "audio/wav"),
                temperature=0,
            )
        assert response.text
        return response.text


def test_qwen3_asr_mtp_matches_greedy_target_on_310p():
    baseline = _transcribe(speculative=False)
    mtp = _transcribe(speculative=True)
    assert mtp == baseline
