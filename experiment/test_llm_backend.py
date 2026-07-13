"""
Unit tests for llm_backend.py.

All tests run without real AWS credentials or GPU — models and Bedrock calls are mocked.
Prompts mirror the two experiment styles:
  - llm-fs: system + user messages with similar-query context, predict a float weight
  - dat:    system + user messages with sparse/dense top-1 passages, predict two int scores
"""

import asyncio
import json
import sys
import os
import unittest
from unittest.mock import MagicMock, patch, call

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from llm_backend import BedrockBackend, LocalQwen3Backend, LocalMistralBackend


# ---------------------------------------------------------------------------
# Representative prompts (adapted from experiment scripts)
# ---------------------------------------------------------------------------

# llm-fs style: few-shot context queries → predict fusion weight
LLM_FS_MESSAGES = [
    {
        "role": "system",
        "content": (
            "You are an expert at predicting the optimal BM25 fusion weight (alpha in [0,1]) "
            "for hybrid sparse+dense retrieval. Higher alpha favours sparse retrieval."
        ),
    },
    {
        "role": "user",
        "content": (
            "Context queries with known best weights (highest_ndcg@10):\n"
            "1. Query: 'patent infringement damages calculation' | mean_best_weight: 0.70 | highest_ndcg@10: 0.843\n"
            "2. Query: 'employment discrimination settlement amounts' | mean_best_weight: 0.60 | highest_ndcg@10: 0.812\n"
            "3. Query: 'contract breach remedies available' | mean_best_weight: 0.55 | highest_ndcg@10: 0.798\n\n"
            "Predict only the numeric mean_best_weight for:\n"
            "Query: 'wrongful termination claims procedures'"
        ),
    },
]

# dat style: evaluate sparse vs dense top-1 hit quality → predict two int scores
DAT_SYSTEM = (
    "You are an evaluator assessing the retrieval effectiveness of dense retrieval and sparse "
    "retrieval for finding the correct answer. Return two integers separated by a space: "
    "sparse score then dense score (0–5 each)."
)
DAT_USER = (
    '- **Question :** "What are the legal requirements for filing a patent application?"\n'
    '- **sparse retrieval Top1 Result:** "A patent application must include a written description, '
    'claims defining the invention, and drawings where necessary to understand it."\n'
    '- **dense retrieval Top1 Result:** "Intellectual property protection for inventions is '
    'governed by patent law, requiring novelty, utility, and non-obviousness criteria."'
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bedrock_response_bytes(content: str) -> bytes:
    return json.dumps({"choices": [{"message": {"content": content}}]}).encode()


def _make_bedrock_mock_client(response_content: str):
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.__getitem__.side_effect = lambda k: (
        _body_mock(response_content) if k == "body" else MagicMock()
    )
    mock_client.invoke_model.return_value = mock_response
    return mock_client


def _body_mock(content: str):
    m = MagicMock()
    m.read.return_value = _bedrock_response_bytes(content)
    return m


class _MockTensorDict(dict):
    """dict that supports .to(device) and .input_ids shorthand — mimics a tokenizer output."""
    def to(self, device):
        return self

    @property
    def input_ids(self):
        return self["input_ids"]


def _make_mock_tensor(shape_tuple):
    t = MagicMock()
    t.shape = shape_tuple
    t.to.return_value = t
    return t


def _make_qwen3_backend(response_text="0.62", fail_times=0):
    """Build a LocalQwen3Backend with all heavy dependencies mocked out."""
    mock_id = _make_mock_tensor((1, 10))
    mock_inputs = _MockTensorDict({"input_ids": mock_id, "attention_mask": MagicMock()})

    mock_tokenizer = MagicMock()
    mock_tokenizer.pad_token = None
    mock_tokenizer.eos_token = "<eos>"
    mock_tokenizer.eos_token_id = 0
    mock_tokenizer.apply_chat_template.return_value = "<mock_prompt>"
    mock_tokenizer.return_value = mock_inputs
    mock_tokenizer.batch_decode.return_value = [response_text]

    mock_outputs = MagicMock()
    mock_llm = MagicMock()
    mock_llm.device = "cpu"
    if fail_times:
        mock_llm.generate.side_effect = (
            [RuntimeError("CUDA OOM")] * fail_times + [mock_outputs]
        )
    else:
        mock_llm.generate.return_value = mock_outputs

    backend = object.__new__(LocalQwen3Backend)
    backend._tokenizer = mock_tokenizer
    backend._llm = mock_llm
    backend._max_new_tokens = 10
    backend._temperature = 0.1
    backend._max_retries = 5
    return backend, mock_tokenizer, mock_llm


def _make_mistral_backend(response_text="0.70", fail_times=0):
    """Build a LocalMistralBackend with all heavy dependencies mocked out."""
    mock_id = _make_mock_tensor((1, 10))
    mock_attn = MagicMock()
    mock_attn.to.return_value = mock_attn
    mock_pixel = MagicMock()  # should be filtered out in generate()

    mock_tokenizer = MagicMock()
    mock_tokenizer.pad_token = None
    mock_tokenizer.eos_token = "<eos>"
    mock_tokenizer.apply_chat_template.return_value = "<mock_prompt>"
    mock_tokenizer.return_value = {
        "input_ids": mock_id,
        "attention_mask": mock_attn,
        "pixel_values": mock_pixel,
    }
    mock_tokenizer.batch_decode.return_value = [response_text]

    mock_outputs = MagicMock()
    mock_llm = MagicMock()
    mock_llm.device = "cpu"
    if fail_times:
        mock_llm.generate.side_effect = (
            [RuntimeError("CUDA OOM")] * fail_times + [mock_outputs]
        )
    else:
        mock_llm.generate.return_value = mock_outputs

    backend = object.__new__(LocalMistralBackend)
    backend._tokenizer = mock_tokenizer
    backend._llm = mock_llm
    backend._max_new_tokens = 100
    backend._temperature = 0.1
    backend._max_retries = 5
    return backend, mock_tokenizer, mock_llm


# ---------------------------------------------------------------------------
# BedrockBackend tests
# ---------------------------------------------------------------------------

class TestBedrockBackendGenerate(unittest.TestCase):

    def _make_backend(self, response="0.62"):
        with patch("boto3.client") as mock_boto3:
            mock_boto3.return_value = _make_bedrock_mock_client(response)
            backend = BedrockBackend(model_id="qwen.qwen3-32b-v1:0", max_retries=5)
            backend._client = mock_boto3.return_value
        return backend

    def test_generate_returns_model_response(self):
        backend = self._make_backend("0.62")
        result = backend.generate(LLM_FS_MESSAGES)
        self.assertEqual(result, "0.62")

    def test_generate_merges_system_and_user_into_single_user_message(self):
        backend = self._make_backend("0.62")
        backend.generate(LLM_FS_MESSAGES)

        _, kwargs = backend._client.invoke_model.call_args
        body = json.loads(kwargs["body"])
        self.assertEqual(len(body["messages"]), 1)
        self.assertEqual(body["messages"][0]["role"], "user")
        combined = body["messages"][0]["content"]
        self.assertIn(LLM_FS_MESSAGES[0]["content"], combined)
        self.assertIn(LLM_FS_MESSAGES[1]["content"], combined)

    def test_generate_uses_correct_model_id(self):
        backend = self._make_backend("0.62")
        backend.generate(LLM_FS_MESSAGES)
        _, kwargs = backend._client.invoke_model.call_args
        self.assertEqual(kwargs["modelId"], "qwen.qwen3-32b-v1:0")

    def test_generate_retries_on_exception_and_returns_empty_after_exhaustion(self):
        with patch("boto3.client") as mock_boto3:
            mock_client = MagicMock()
            mock_client.invoke_model.side_effect = Exception("throttled")
            mock_boto3.return_value = mock_client
            backend = BedrockBackend(model_id="qwen.qwen3-32b-v1:0", max_retries=3)
            backend._client = mock_client

        result = backend.generate(LLM_FS_MESSAGES)
        self.assertEqual(result, "")
        self.assertEqual(mock_client.invoke_model.call_count, 3)

    def test_generate_succeeds_after_transient_failure(self):
        with patch("boto3.client") as mock_boto3:
            mock_client = MagicMock()
            good_response = MagicMock()
            good_response.__getitem__.side_effect = lambda k: (
                _body_mock("0.55") if k == "body" else MagicMock()
            )
            mock_client.invoke_model.side_effect = [Exception("throttled"), good_response]
            mock_boto3.return_value = mock_client
            backend = BedrockBackend(model_id="qwen.qwen3-32b-v1:0", max_retries=5)
            backend._client = mock_client

        result = backend.generate(LLM_FS_MESSAGES)
        self.assertEqual(result, "0.55")
        self.assertEqual(mock_client.invoke_model.call_count, 2)

    def test_generate_dat_messages_no_system_role(self):
        """When messages list has no system role, combined content equals user content."""
        messages = [{"role": "user", "content": DAT_USER}]
        with patch("boto3.client") as mock_boto3:
            mock_boto3.return_value = _make_bedrock_mock_client("4 3")
            backend = BedrockBackend(model_id="mistral.ministral-3-14b-instruct")
            backend._client = mock_boto3.return_value

        result = backend.generate(messages)
        self.assertEqual(result, "4 3")
        _, kwargs = backend._client.invoke_model.call_args
        body = json.loads(kwargs["body"])
        self.assertEqual(body["messages"][0]["content"], DAT_USER)


class TestBedrockBackendInvoke(unittest.TestCase):

    def _make_backend(self, response="4 3"):
        with patch("boto3.client") as mock_boto3:
            mock_boto3.return_value = _make_bedrock_mock_client(response)
            backend = BedrockBackend(model_id="qwen.qwen3-32b-v1:0", max_retries=3, max_workers=1)
            backend._client = mock_boto3.return_value
        return backend

    def test_invoke_returns_model_response(self):
        backend = self._make_backend("4 3")
        result = asyncio.run(backend.invoke(DAT_SYSTEM, DAT_USER))
        self.assertEqual(result, "4 3")

    def test_invoke_combines_system_and_user(self):
        backend = self._make_backend("4 3")
        asyncio.run(backend.invoke(DAT_SYSTEM, DAT_USER))
        _, kwargs = backend._client.invoke_model.call_args
        body = json.loads(kwargs["body"])
        combined = body["messages"][0]["content"]
        self.assertIn(DAT_SYSTEM, combined)
        self.assertIn(DAT_USER, combined)

    def test_invoke_retries_on_exception(self):
        with patch("boto3.client") as mock_boto3:
            mock_client = MagicMock()
            mock_client.invoke_model.side_effect = Exception("timeout")
            mock_boto3.return_value = mock_client
            backend = BedrockBackend(model_id="qwen.qwen3-32b-v1:0", max_retries=3, max_workers=1)
            backend._client = mock_client

        result = asyncio.run(backend.invoke(DAT_SYSTEM, DAT_USER))
        self.assertEqual(result, "")
        self.assertEqual(mock_client.invoke_model.call_count, 3)


# ---------------------------------------------------------------------------
# LocalQwen3Backend tests
# ---------------------------------------------------------------------------

class TestLocalQwen3BackendGenerate(unittest.TestCase):

    def test_generate_returns_decoded_response(self):
        backend, _, _ = _make_qwen3_backend("0.62")
        result = backend.generate(LLM_FS_MESSAGES)
        self.assertEqual(result, "0.62")

    def test_generate_uses_enable_thinking_false(self):
        backend, mock_tokenizer, _ = _make_qwen3_backend()
        backend.generate(LLM_FS_MESSAGES)
        mock_tokenizer.apply_chat_template.assert_called_once_with(
            LLM_FS_MESSAGES,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    def test_generate_passes_do_sample_and_pad_token_id(self):
        backend, _, mock_llm = _make_qwen3_backend()
        backend.generate(LLM_FS_MESSAGES)
        _, kwargs = mock_llm.generate.call_args
        self.assertTrue(kwargs.get("do_sample"))
        self.assertEqual(kwargs.get("pad_token_id"), backend._tokenizer.eos_token_id)

    def test_generate_passes_max_new_tokens(self):
        backend, _, mock_llm = _make_qwen3_backend()
        backend.generate(LLM_FS_MESSAGES)
        _, kwargs = mock_llm.generate.call_args
        self.assertEqual(kwargs.get("max_new_tokens"), 10)

    def test_generate_retries_and_returns_empty_after_exhaustion(self):
        backend, _, mock_llm = _make_qwen3_backend(fail_times=5)
        mock_llm.generate.side_effect = RuntimeError("CUDA OOM")
        result = backend.generate(LLM_FS_MESSAGES)
        self.assertEqual(result, "")
        self.assertEqual(mock_llm.generate.call_count, 5)

    def test_generate_succeeds_after_transient_failure(self):
        backend, _, mock_llm = _make_qwen3_backend(response_text="0.45", fail_times=2)
        result = backend.generate(LLM_FS_MESSAGES)
        self.assertEqual(result, "0.45")
        self.assertEqual(mock_llm.generate.call_count, 3)

    def test_invoke_wraps_generate_as_async(self):
        backend, mock_tokenizer, _ = _make_qwen3_backend("0.62")
        result = asyncio.run(backend.invoke(DAT_SYSTEM, DAT_USER))
        self.assertEqual(result, "0.62")
        # invoke() builds the messages list and forwards to generate()
        called_messages = mock_tokenizer.apply_chat_template.call_args[0][0]
        self.assertEqual(called_messages[0]["role"], "system")
        self.assertEqual(called_messages[0]["content"], DAT_SYSTEM)
        self.assertEqual(called_messages[1]["role"], "user")
        self.assertEqual(called_messages[1]["content"], DAT_USER)


# ---------------------------------------------------------------------------
# LocalMistralBackend tests
# ---------------------------------------------------------------------------

class TestLocalMistralBackendGenerate(unittest.TestCase):

    def test_generate_returns_decoded_response(self):
        backend, _, _ = _make_mistral_backend("0.70")
        result = backend.generate(LLM_FS_MESSAGES)
        self.assertEqual(result, "0.70")

    def test_generate_does_not_use_enable_thinking(self):
        backend, mock_tokenizer, _ = _make_mistral_backend()
        backend.generate(LLM_FS_MESSAGES)
        _, kwargs = mock_tokenizer.apply_chat_template.call_args
        self.assertNotIn("enable_thinking", kwargs)

    def test_generate_filters_out_pixel_values(self):
        """Only input_ids and attention_mask should reach llm.generate()."""
        backend, _, mock_llm = _make_mistral_backend()
        backend.generate(LLM_FS_MESSAGES)
        _, kwargs = mock_llm.generate.call_args
        self.assertIn("input_ids", kwargs)
        self.assertIn("attention_mask", kwargs)
        self.assertNotIn("pixel_values", kwargs)

    def test_generate_does_not_pass_do_sample(self):
        """Mistral original does not use do_sample."""
        backend, _, mock_llm = _make_mistral_backend()
        backend.generate(LLM_FS_MESSAGES)
        _, kwargs = mock_llm.generate.call_args
        self.assertNotIn("do_sample", kwargs)

    def test_generate_passes_max_new_tokens(self):
        backend, _, mock_llm = _make_mistral_backend()
        backend.generate(LLM_FS_MESSAGES)
        _, kwargs = mock_llm.generate.call_args
        self.assertEqual(kwargs.get("max_new_tokens"), 100)

    def test_generate_retries_and_returns_empty_after_exhaustion(self):
        backend, _, mock_llm = _make_mistral_backend()
        mock_llm.generate.side_effect = RuntimeError("CUDA OOM")
        result = backend.generate(LLM_FS_MESSAGES)
        self.assertEqual(result, "")
        self.assertEqual(mock_llm.generate.call_count, 5)

    def test_generate_succeeds_after_transient_failure(self):
        backend, _, mock_llm = _make_mistral_backend(response_text="0.55", fail_times=1)
        result = backend.generate(LLM_FS_MESSAGES)
        self.assertEqual(result, "0.55")
        self.assertEqual(mock_llm.generate.call_count, 2)

    def test_invoke_wraps_generate_as_async(self):
        backend, mock_tokenizer, _ = _make_mistral_backend("3 4")
        result = asyncio.run(backend.invoke(DAT_SYSTEM, DAT_USER))
        self.assertEqual(result, "3 4")
        called_messages = mock_tokenizer.apply_chat_template.call_args[0][0]
        self.assertEqual(called_messages[0]["role"], "system")
        self.assertEqual(called_messages[1]["role"], "user")


if __name__ == "__main__":
    unittest.main(verbosity=2)
