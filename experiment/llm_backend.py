import asyncio
import json
import warnings
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor

import boto3
import torch


class LLMBackend(ABC):
    """Abstract base for all LLM inference backends.

    Two call interfaces:
    - generate(messages)        sync,  OpenAI-format list — used by llm-fs scripts
    - await invoke(sys, user)   async, two strings       — used by dynamic-alpha scripts
    """

    @abstractmethod
    def generate(self, messages: list) -> str:
        ...

    async def invoke(self, system_message: str, user_message: str) -> str:
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.generate, messages)


class BedrockBackend(LLMBackend):
    """AWS Bedrock inference via invoke_model (OpenAI-compatible response format).

    Overrides invoke() with a dedicated ThreadPoolExecutor so that high-concurrency
    async callers (dynamic_alpha_tuning.py) can dispatch many requests in parallel.
    """

    def __init__(self, model_id, region="us-east-1", max_tokens=32, temperature=0.1,
                 max_retries=5, max_workers=50):
        self._model_id = model_id
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._max_retries = max_retries
        self._client = boto3.client("bedrock-runtime", region_name=region)
        self._executor = ThreadPoolExecutor(max_workers=max_workers)

    def generate(self, messages: list) -> str:
        system = next((m["content"] for m in messages if m["role"] == "system"), "")
        user = next((m["content"] for m in messages if m["role"] == "user"), "")
        combined = f"{system}\n\n{user}" if system else user
        request_body = {
            "messages": [{"role": "user", "content": combined}],
            "max_tokens": self._max_tokens,
            "temperature": self._temperature,
        }
        for attempt in range(self._max_retries):
            try:
                response = self._client.invoke_model(
                    modelId=self._model_id,
                    body=json.dumps(request_body),
                    contentType="application/json",
                    accept="application/json",
                )
                return json.loads(response["body"].read())["choices"][0]["message"]["content"]
            except Exception as e:
                print(f"  Bedrock attempt {attempt + 1}/{self._max_retries} failed: {e}. Retrying...")
        print(f"  FAILED after {self._max_retries} retries, returning empty string.")
        return ""

    async def invoke(self, system_message: str, user_message: str) -> str:
        combined = f"{system_message}\n\n{user_message}"
        request_body = {
            "messages": [{"role": "user", "content": combined}],
            "max_tokens": self._max_tokens,
            "temperature": self._temperature,
        }
        loop = asyncio.get_running_loop()
        last_error = None
        for attempt in range(self._max_retries):
            try:
                response = await loop.run_in_executor(
                    self._executor,
                    lambda: self._client.invoke_model(
                        modelId=self._model_id,
                        body=json.dumps(request_body),
                        contentType="application/json",
                        accept="application/json",
                    ),
                )
                return json.loads(response["body"].read())["choices"][0]["message"]["content"]
            except Exception as e:
                last_error = e
                print(f"  Bedrock attempt {attempt + 1}/{self._max_retries} failed: {e}. Retrying...")
        print(f"  FAILED after {self._max_retries} retries: {last_error}")
        return ""


class LocalQwen3Backend(LLMBackend):
    """Local Qwen3 inference via AutoModelForCausalLM with 4-bit quantization."""

    def __init__(self, model="Qwen/Qwen3-32B", max_new_tokens=10, temperature=0.1, max_retries=5):
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
        self._tokenizer = AutoTokenizer.from_pretrained(model)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._tokenizer.padding_side = "left"
        self._llm = AutoModelForCausalLM.from_pretrained(
            model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            quantization_config=quant_config,
        )
        self._max_new_tokens = max_new_tokens
        self._temperature = temperature
        self._max_retries = max_retries

    def generate(self, messages: list) -> str:
        prompt = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._llm.device)
        input_length = inputs.input_ids.shape[1]
        last_error = None
        for attempt in range(self._max_retries):
            try:
                with torch.inference_mode():
                    outputs = self._llm.generate(
                        **inputs,
                        max_new_tokens=self._max_new_tokens,
                        temperature=self._temperature,
                        do_sample=True,
                        pad_token_id=self._tokenizer.eos_token_id,
                    )
                return self._tokenizer.batch_decode(
                    outputs[:, input_length:], skip_special_tokens=True
                )[0]
            except Exception as e:
                last_error = e
                print(f"  LocalQwen3 attempt {attempt + 1}/{self._max_retries} failed: {e}. Retrying...")
        print(f"  FAILED after {self._max_retries} retries: {last_error}")
        return ""


class LocalMistralBackend(LLMBackend):
    """Local Ministral-3 inference via Mistral3ForConditionalGeneration with FP8 quantization."""

    def __init__(self, model="mistralai/Ministral-3-14B-Instruct-2512", max_new_tokens=100,
                 temperature=0.1, max_retries=5):
        from transformers import FineGrainedFP8Config, Mistral3ForConditionalGeneration, MistralCommonBackend

        self._tokenizer = MistralCommonBackend.from_pretrained(model)
        if getattr(self._tokenizer, "pad_token", None) is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._tokenizer.padding_side = "left"
        self._llm = Mistral3ForConditionalGeneration.from_pretrained(
            model,
            device_map="auto",
            quantization_config=FineGrainedFP8Config(dequantize=True),
        )
        self._max_new_tokens = max_new_tokens
        self._temperature = temperature
        self._max_retries = max_retries

    def generate(self, messages: list) -> str:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*tokenize=False.*")
            prompt = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        inputs = self._tokenizer(prompt, return_tensors="pt", padding=True)
        gen_inputs = {
            k: v.to(self._llm.device)
            for k, v in inputs.items()
            if k in ("input_ids", "attention_mask")
        }
        input_length = gen_inputs["input_ids"].shape[1]
        last_error = None
        for attempt in range(self._max_retries):
            try:
                outputs = self._llm.generate(
                    **gen_inputs,
                    max_new_tokens=self._max_new_tokens,
                    temperature=self._temperature,
                )
                return self._tokenizer.batch_decode(
                    outputs[:, input_length:], skip_special_tokens=True
                )[0]
            except Exception as e:
                last_error = e
                print(f"  LocalMistral attempt {attempt + 1}/{self._max_retries} failed: {e}. Retrying...")
        print(f"  FAILED after {self._max_retries} retries: {last_error}")
        return ""
