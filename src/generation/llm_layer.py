"""
AGHR System — LLM Layer (Phase 8)
====================================
Unified LLM interface supporting multiple models:
  - FLAN-T5 (seq2seq baseline)
  - TinyLlama (causal, lightweight)
  - Fine-tuned QLoRA models
"""

import torch
from typing import Dict, Optional

from loguru import logger


class LLMLayer:
    """
    Unified LLM generation interface.

    Supports seq2seq (FLAN-T5) and causal (TinyLlama) models
    with optional 4-bit quantization for low-memory inference.
    """

    def __init__(self, model_name: str = "google/flan-t5-base",
                 model_type: str = "seq2seq", max_new_tokens: int = 512,
                 temperature: float = 0.3, top_p: float = 0.9,
                 quantize: bool = False, device: str = None):
        self.model_name = model_name
        self.model_type = model_type
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self._load_model(quantize)

    def _load_model(self, quantize: bool) -> None:
        """Load model and tokenizer."""
        from transformers import AutoTokenizer

        logger.info(f"Loading {self.model_name} (type={self.model_type})...")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        if quantize and torch.cuda.is_available():
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            quant_kwargs = {"quantization_config": bnb_config, "device_map": "auto"}
        else:
            quant_kwargs = {}

        if self.model_type == "seq2seq":
            from transformers import AutoModelForSeq2SeqLM
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_name, **quant_kwargs
            )
        else:
            from transformers import AutoModelForCausalLM
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name, **quant_kwargs
            )

        if not quant_kwargs:
            self.model = self.model.to(self.device)

        self.model.eval()
        logger.info(f"Model loaded on {self.device}")

    def generate(self, prompt: str, max_new_tokens: int = None,
                 temperature: float = None, top_p: float = None) -> str:
        """
        Generate text from a prompt.

        Args:
            prompt: Input prompt string.
            max_new_tokens: Override default max tokens.
            temperature: Override default temperature.
            top_p: Override default top_p.

        Returns:
            Generated text string.
        """
        max_tokens = max_new_tokens or self.max_new_tokens
        temp = temperature or self.temperature
        top_p_val = top_p or self.top_p

        # For small models like flan-t5-small, limit input to 512 tokens
        is_small_model = "small" in self.model_name.lower()
        max_input = 512 if is_small_model else 1024

        inputs = self.tokenizer(
            prompt, return_tensors="pt", max_length=max_input, truncation=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                min_new_tokens=10,  # Prevent single-word answers like 'NO'
                temperature=temp,
                top_p=top_p_val,
                do_sample=False,    # Use greedy/beam for more factual output
                num_beams=3,        # Beam search for better quality
                no_repeat_ngram_size=3,
                early_stopping=True,
            )

        # For causal models, strip the input tokens from output
        if self.model_type == "causal":
            outputs = outputs[:, inputs["input_ids"].shape[-1]:]

        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return text.strip()

    def generate_with_context(self, question: str, context: str,
                               prompt_engine=None, strategy: str = "chain_of_thought") -> Dict:
        """
        Generate answer using a prompt engine and context.

        Args:
            question: User's question.
            context: Retrieved context.
            prompt_engine: PromptEngine instance.
            strategy: Prompting strategy.

        Returns:
            Dict with answer, parsed output, and metadata.
        """
        # Limit context length based on model size
        is_small_model = "small" in self.model_name.lower()
        ctx_limit = 400 if is_small_model else 1500
        trimmed_context = context[:ctx_limit] if context else ""

        if prompt_engine:
            prompt = prompt_engine.format_prompt(strategy, question, context=trimmed_context)
        else:
            # Simple, directive prompt optimized for small seq2seq models
            prompt = (f"Answer the medical question based on the context.\n\n"
                      f"Context: {trimmed_context}\n\n"
                      f"Question: {question}\n\n"
                      f"Detailed answer:")

        raw_output = self.generate(prompt)

        # Parse structured output if applicable
        if prompt_engine and strategy in ("chain_of_thought", "structured"):
            parsed = prompt_engine.parse_structured_output(raw_output)
        else:
            parsed = {"answer": raw_output, "raw": raw_output, "reasoning": "",
                      "sources": "", "confidence": 0.0}

        return {
            "question": question,
            "answer": parsed.get("answer", raw_output),
            "reasoning": parsed.get("reasoning", ""),
            "sources": parsed.get("sources", ""),
            "confidence": parsed.get("confidence", 0.0),
            "raw_output": raw_output,
            "strategy": strategy,
            "model": self.model_name,
        }

    def generate_stream(self, prompt: str, max_new_tokens: int = None,
                        temperature: float = None, top_p: float = None):
        """Yields tokens iteratively as they are generated using TextIteratorStreamer."""
        from transformers import TextIteratorStreamer
        from threading import Thread

        max_tokens = max_new_tokens or self.max_new_tokens
        temp = temperature or self.temperature
        top_p_val = top_p or self.top_p

        inputs = self.tokenizer(
            prompt, return_tensors="pt", max_length=1024, truncation=True
        ).to(self.device)

        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        
        generation_kwargs = dict(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temp,
            top_p=top_p_val,
            do_sample=temp > 0,
            num_beams=1,
            streamer=streamer,
        )

        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        for new_text in streamer:
            yield new_text

    def generate_with_context_stream(self, question: str, context: str,
                                     prompt_engine=None, strategy: str = "chain_of_thought"):
        """Returns a generator yielding raw text tokens for the answer."""
        # Limit context length based on model size
        is_small_model = "small" in self.model_name.lower()
        ctx_limit = 400 if is_small_model else 1500
        trimmed_context = context[:ctx_limit] if context else ""

        if prompt_engine:
            prompt = prompt_engine.format_prompt(strategy, question, context=trimmed_context)
        else:
            prompt = (f"Answer the medical question based on the context.\n\n"
                      f"Context: {trimmed_context}\n\n"
                      f"Question: {question}\n\n"
                      f"Detailed answer:")

        return self.generate_stream(prompt)
