"""
Frozen Code LLM Inference Engine.
Owner: Member 3 (LLM & Experimentation Engineer)

Provides reproducible inference over frozen code language models (e.g. Qwen2.5-Coder-3B-Instruct)
with token counting, latency measurement, VRAM profiling, stochastic sampling, and
yes/no logit probability extraction.
"""

from __future__ import annotations

import logging
import math
import os
from pathlib import Path
import time
from typing import Any

import torch
from repoguard.exceptions import ModelInferenceError
from repoguard.schemas import GenerationResult

logger = logging.getLogger("repoguard.llm")


class MockLLM:
    """
    Deterministic mock LLM for testing, CPU-only environments, and CI.
    Generates structured, reproducible answers without GPU or network access.
    """

    def __init__(self, model_name: str = "mock-qwen2.5-coder") -> None:
        self.model_name = model_name

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> GenerationResult:
        start_time = time.perf_counter()
        prompt_words = prompt.split()
        prompt_tokens = max(1, len(prompt_words) * 4 // 3)

        # Context-aware mock responses based on prompt keywords
        q_lower = prompt.lower()
        if "sync_all_users" in q_lower or "cancel_all_async_jobs" in q_lower or "does not exist" in q_lower:
            text = (
                "Based on the repository analysis, this symbol or functionality does not exist in the codebase. "
                "I must abstain from generating an unsupported answer."
            )
        elif "load_config" in q_lower:
            text = (
                "The `load_config` function is defined in `config.py` at line 12. "
                "It takes parameter `path` and returns a configuration dictionary."
            )
        elif "scheduler" in q_lower or "every" in q_lower:
            text = (
                "In `schedule/__init__.py`, the `Scheduler.every` method schedules a new periodic job. "
                "It takes parameters `self` and `interval` (defaulting to 1) and returns a `Job` instance."
            )
        elif "job" in q_lower or "do" in q_lower:
            text = (
                "The `Job.do` method defines the job's target execution function and associated arguments. "
                "It accepts `job_func` and `*args`, `**kwargs`."
            )
        else:
            text = (
                f"Generated response based on repository prompt context. Mentioned symbol `sample_function` "
                f"defined in repository file `schedule/__init__.py`."
            )

        completion_tokens = max(1, len(text.split()) * 4 // 3)
        latency_ms = (time.perf_counter() - start_time) * 1000.0

        return GenerationResult(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            latency_ms=round(latency_ms, 2),
            peak_vram_mb=0.0,
            finish_reason="stop",
            is_mock=True,
            model_name=self.model_name,
        )

    def sample(
        self,
        prompt: str,
        k: int = 5,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
    ) -> list[str]:
        """Generate k mock samples to test semantic agreement signals."""
        res = self.generate(prompt, max_new_tokens=max_new_tokens, temperature=temperature)
        return [res.text for _ in range(k)]

    def yes_no_prob(self, prompt: str) -> float:
        """Return deterministic pseudo-probability of 'yes' for self-check queries."""
        q_lower = prompt.lower()
        if "exist" in q_lower and ("sync_all" in q_lower or "cancel_async" in q_lower or "fake" in q_lower):
            return 0.05
        if "exist" in q_lower and ("every" in q_lower or "scheduler" in q_lower or "job" in q_lower):
            return 0.94
        if "in scope" in q_lower or "about this repository" in q_lower:
            return 0.88
        return 0.50


class LLMService:
    """
    Frozen LLM Service supporting Qwen2.5-Coder model family.
    Operates in 4-bit (NF4 via bitsandbytes), 8-bit, FP16/BF16, CPU, or Mock mode.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-Coder-3B-Instruct",
        device: str | None = None,
        quantization_bits: int = 4,
        torch_dtype: str = "float16",
        mock_mode: bool = False,
        cache_dir: str | None = None,
    ) -> None:
        self.model_name = model_name

        # Prefer GPU (CUDA) by default; fall back to CPU if GPU or CUDA build is unavailable
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        elif device == "cuda" and not torch.cuda.is_available():
            logger.warning(
                "CUDA device was requested, but torch.cuda.is_available() is False. "
                "Falling back to CPU."
            )
            self.device = "cpu"
        else:
            self.device = device

        self.quantization_bits = quantization_bits
        self.torch_dtype = torch_dtype
        self.mock_mode = mock_mode
        self.cache_dir = cache_dir or os.environ.get("HF_HOME")

        self._tokenizer = None
        self._model = None
        self._mock_engine = MockLLM(model_name=model_name) if mock_mode else None

    def _init_model(self) -> None:
        """Lazily load tokenizer and model weights."""
        if self.mock_mode or self._model is not None:
            return

        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info("Loading tokenizer for %s (cache_dir=%s)...", self.model_name, self.cache_dir)
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir,
            trust_remote_code=True,
        )

        use_cuda = self.device == "cuda" and torch.cuda.is_available()
        logger.info("Initializing %s on %s (CUDA available: %s)", self.model_name, self.device, use_cuda)

        if use_cuda and self.quantization_bits == 4:
            try:
                from transformers import BitsAndBytesConfig
                from transformers.modeling_utils import PreTrainedModel

                # Monkey-patch PreTrainedModel.to so accelerate's dispatch_model doesn't crash
                # on transformers 4.41.x with bitsandbytes 4-bit models
                orig_to = PreTrainedModel.to

                def safe_to(m_self, *args, **kwargs):
                    try:
                        return orig_to(m_self, *args, **kwargs)
                    except ValueError as err:
                        if "not supported" in str(err) and ("4-bit" in str(err) or "8-bit" in str(err)):
                            return torch.nn.Module.to(m_self, *args, **kwargs)
                        raise

                PreTrainedModel.to = safe_to

                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                )
                logger.info("Loading %s with 4-bit NF4 quantization for 6GB VRAM budget...", self.model_name)
                self._model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    quantization_config=bnb_config,
                    device_map="auto",
                    cache_dir=self.cache_dir,
                    trust_remote_code=True,
                )
                # Ensure all buffers and parameters (such as RoPE rotary embeddings) are on GPU
                torch.nn.Module.to(self._model, "cuda:0")
            except Exception as e:
                logger.warning(
                    "4-bit quantization unavailable (%s). Falling back to float16 on %s.",
                    e,
                    "GPU (CUDA)" if use_cuda else "CPU",
                )

        if not self._model:
            dtype = torch.float16 if use_cuda else torch.float32
            target_device = "cuda" if use_cuda else "cpu"

            try:
                import accelerate  # noqa: F401
                has_accelerate = True
            except ImportError:
                has_accelerate = False

            logger.info("Loading %s in %s on %s...", self.model_name, dtype, target_device)
            if use_cuda and has_accelerate:
                self._model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=dtype,
                    device_map="auto",
                    cache_dir=self.cache_dir,
                    trust_remote_code=True,
                )
            else:
                self._model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=dtype,
                    cache_dir=self.cache_dir,
                    low_cpu_mem_usage=False,
                    trust_remote_code=True,
                )
                self._model = self._model.to(target_device)

        self._model.eval()
        logger.info("Model %s loaded successfully.", self.model_name)

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.2,
        top_p: float = 0.95,
        repetition_penalty: float = 1.05,
        **kwargs: Any,
    ) -> GenerationResult:
        """
        Generate answer from frozen LLM with token counting and latency tracking.
        """
        if self.mock_mode:
            return self._mock_engine.generate(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )

        self._init_model()
        start_time = time.perf_counter()

        # Tokenize prompt
        inputs = self._tokenizer(prompt, return_tensors="pt")
        prompt_tokens = inputs.input_ids.shape[1]
        input_ids = inputs.input_ids.to(self._model.device)
        attention_mask = inputs.attention_mask.to(self._model.device)

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        # Inference with no gradients
        do_sample = temperature > 0.0
        with torch.no_grad():
            outputs = self._model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                top_p=top_p if do_sample else None,
                repetition_penalty=repetition_penalty,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        latency_ms = (time.perf_counter() - start_time) * 1000.0
        generated_ids = outputs[0][prompt_tokens:]
        completion_tokens = len(generated_ids)
        output_text = self._tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        peak_vram_mb = 0.0
        if torch.cuda.is_available():
            peak_vram_mb = round(torch.cuda.max_memory_allocated() / (1024 * 1024), 2)

        # Free cached intermediate activation tensors so VRAM doesn't accumulate across 100 questions
        del outputs, input_ids, attention_mask
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return GenerationResult(
            text=output_text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            latency_ms=round(latency_ms, 2),
            peak_vram_mb=peak_vram_mb,
            finish_reason="stop",
            is_mock=False,
            model_name=self.model_name,
        )

    def sample(
        self,
        prompt: str,
        k: int = 5,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.90,
    ) -> list[str]:
        """
        Generate k diverse samples to measure semantic consensus across completions.
        """
        if self.mock_mode:
            return self._mock_engine.sample(
                prompt=prompt,
                k=k,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )

        self._init_model()
        samples: list[str] = []
        for _ in range(k):
            res = self.generate(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            samples.append(res.text)
        return samples

    def yes_no_prob(self, prompt: str) -> float:
        """
        Calculate probability P('yes') vs P('no') from next-token logits without generating text.
        Used for the 4 self-check probability signals in RepoGuard.
        """
        if self.mock_mode:
            return self._mock_engine.yes_no_prob(prompt)

        self._init_model()
        inputs = self._tokenizer(prompt, return_tensors="pt")
        input_ids = inputs.input_ids.to(self._model.device)

        with torch.no_grad():
            outputs = self._model(input_ids)
            next_token_logits = outputs.logits[0, -1, :]

        # Look up token IDs for Yes / No variations
        yes_candidates = ["Yes", "yes", " YES"]
        no_candidates = ["No", "no", " NO"]

        yes_ids = [self._tokenizer.encode(t, add_special_tokens=False)[0] for t in yes_candidates]
        no_ids = [self._tokenizer.encode(t, add_special_tokens=False)[0] for t in no_candidates]

        yes_logits = [next_token_logits[tid].item() for tid in yes_ids]
        no_logits = [next_token_logits[tid].item() for tid in no_ids]

        max_yes = max(yes_logits)
        max_no = max(no_logits)

        # Softmax over binary decision
        exp_yes = math.exp(max_yes)
        exp_no = math.exp(max_no)
        p_yes = exp_yes / (exp_yes + exp_no + 1e-12)
        return round(p_yes, 4)


# Module-level default instance
_default_service: LLMService | None = None


def get_llm_service(mock_mode: bool = True, device: str | None = None) -> LLMService:
    """Return or initialize default LLM service (prefers CUDA GPU with fallback to CPU)."""
    global _default_service
    if _default_service is None or _default_service.mock_mode != mock_mode:
        _default_service = LLMService(
            model_name="Qwen/Qwen2.5-Coder-3B-Instruct",
            device=device,
            mock_mode=mock_mode,
        )
    return _default_service


def generate(prompt: str, **kwargs: Any) -> GenerationResult:
    """Convenience functional wrapper."""
    return get_llm_service().generate(prompt, **kwargs)


def sample(prompt: str, k: int = 5, **kwargs: Any) -> list[str]:
    """Convenience sampling wrapper."""
    return get_llm_service().sample(prompt, k=k, **kwargs)


def yes_no_prob(prompt: str) -> float:
    """Convenience yes/no probability wrapper."""
    return get_llm_service().yes_no_prob(prompt)
