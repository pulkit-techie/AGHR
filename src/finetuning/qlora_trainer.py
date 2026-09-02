"""
AGHR System — QLoRA Trainer (Phase 9.2)
=========================================
Fine-tunes LLMs using QLoRA (Quantized Low-Rank Adaptation).
Supports FLAN-T5 and TinyLlama with 4-bit quantization.

NOTE: Requires GPU (CUDA). For CPU-only, use the base models directly.
"""

import os
import json
from pathlib import Path
from typing import Dict, Optional

import torch
from loguru import logger


class QLoRATrainer:
    """
    QLoRA fine-tuning pipeline using HuggingFace PEFT + TRL.

    Configuration:
      - LoRA: r=8, alpha=32, dropout=0.05
      - Quantization: 4-bit NF4
      - Training: SFTTrainer from trl
    """

    def __init__(self, config: Dict):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.trainer = None

    def setup(self, model_name: str = "google/flan-t5-base",
              model_type: str = "seq2seq") -> None:
        """
        Set up model with QLoRA configuration.

        Args:
            model_name: HuggingFace model name.
            model_type: "seq2seq" for T5, "causal" for TinyLlama.
        """
        from transformers import AutoTokenizer

        self.model_type = model_type
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model with quantization if GPU available
        if torch.cuda.is_available():
            self._load_quantized(model_name, model_type)
        else:
            self._load_standard(model_name, model_type)
            logger.warning("No GPU — training will be slow. Consider Google Colab.")

        logger.info(f"Model ready: {model_name} ({model_type})")

    def _load_quantized(self, model_name: str, model_type: str) -> None:
        """Load model with 4-bit quantization."""
        from transformers import BitsAndBytesConfig

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        if model_type == "seq2seq":
            from transformers import AutoModelForSeq2SeqLM
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name, quantization_config=bnb_config, device_map="auto"
            )
        else:
            from transformers import AutoModelForCausalLM
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, quantization_config=bnb_config, device_map="auto"
            )

        self.model.gradient_checkpointing_enable()

    def _load_standard(self, model_name: str, model_type: str) -> None:
        """Load model without quantization (CPU fallback)."""
        if model_type == "seq2seq":
            from transformers import AutoModelForSeq2SeqLM
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        else:
            from transformers import AutoModelForCausalLM
            self.model = AutoModelForCausalLM.from_pretrained(model_name)

    def apply_lora(self) -> None:
        """Apply LoRA adapter to the model."""
        from peft import LoraConfig, get_peft_model, TaskType

        lora_cfg = self.config.get("lora", {})

        task_type = TaskType.SEQ_2_SEQ_LM if self.model_type == "seq2seq" else TaskType.CAUSAL_LM

        # Determine target modules based on model type
        if self.model_type == "seq2seq":
            target_modules = ["q", "v"]
        else:
            target_modules = lora_cfg.get("target_modules", ["q_proj", "v_proj"])

        peft_config = LoraConfig(
            r=lora_cfg.get("r", 8),
            lora_alpha=lora_cfg.get("lora_alpha", 32),
            lora_dropout=lora_cfg.get("lora_dropout", 0.05),
            target_modules=target_modules,
            bias=lora_cfg.get("bias", "none"),
            task_type=task_type,
        )

        self.model = get_peft_model(self.model, peft_config)
        trainable, total = self.model.get_nb_trainable_parameters()
        logger.info(f"LoRA applied: {trainable:,} / {total:,} trainable params "
                     f"({100*trainable/total:.2f}%)")

    def train(self, train_dataset, eval_dataset=None) -> None:
        """
        Run QLoRA training using standard Trainer (version-safe).

        Args:
            train_dataset: HuggingFace Dataset with 'text' field.
            eval_dataset: Optional evaluation dataset.
        """
        from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling

        train_cfg = self.config.get("training", {})
        output_dir = train_cfg.get("output_dir", "models/finetuned")
        max_seq_length = train_cfg.get("max_seq_length", 1024)

        # Step 1: Tokenize the dataset ourselves (bypasses all SFTTrainer issues)
        tokenizer = self.tokenizer

        def tokenize_fn(examples):
            out = tokenizer(
                examples["text"],
                truncation=True,
                max_length=max_seq_length,
                padding="max_length",
            )
            # Create a safe copy of input_ids for labels
            out["labels"] = [list(seq) for seq in out["input_ids"]]
            return out

        logger.info("Tokenizing dataset...")
        tokenized_train = train_dataset.map(
            tokenize_fn, batched=True, remove_columns=train_dataset.column_names
        )
        tokenized_eval = None
        if eval_dataset:
            tokenized_eval = eval_dataset.map(
                tokenize_fn, batched=True, remove_columns=eval_dataset.column_names
            )

        # Step 2: Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm=False
        )

        # Step 3: Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=train_cfg.get("epochs", 3),
            per_device_train_batch_size=train_cfg.get("batch_size", 4),
            gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 4),
            learning_rate=train_cfg.get("learning_rate", 2e-4),
            warmup_steps=train_cfg.get("warmup_steps", 10),
            logging_steps=10,
            save_steps=train_cfg.get("save_steps", 100),
            save_total_limit=2,
            fp16=torch.cuda.is_available(),
            report_to="none",
        )

        # Step 4: Use standard Trainer (rock-solid, works on ALL versions)
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_eval,
            data_collator=data_collator,
        )

        logger.info("Starting QLoRA training...")
        self.trainer.train()
        logger.info("Training complete!")

    def save_model(self, output_dir: str = None) -> None:
        """Save fine-tuned LoRA adapter."""
        output_dir = output_dir or self.config.get("training", {}).get(
            "output_dir", "models/finetuned"
        )
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        logger.info(f"Model saved → {output_dir}")

    def load_finetuned(self, adapter_path: str, base_model: str) -> None:
        """Load a fine-tuned LoRA adapter on top of base model."""
        from peft import PeftModel
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(adapter_path)
        self._load_standard(base_model, self.model_type)
        self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.model.eval()
        logger.info(f"Loaded fine-tuned model from {adapter_path}")


def create_hf_dataset(samples, tokenizer=None):
    """Convert list of dicts to HuggingFace Dataset."""
    from datasets import Dataset
    return Dataset.from_list(samples)
