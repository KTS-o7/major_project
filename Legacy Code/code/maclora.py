import argparse
import json
import logging
import os
import random

import matplotlib.pyplot as plt
import torch

# Ensure you have installed the `peft` library: pip install peft
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# --- LoRA Configuration ---
DEFAULT_LORA_CONFIG = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# --- Dataset and Plotting Classes (Unchanged) ---

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        logger.info("Initialized CustomDataset with %d entries", len(data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        # Format the text for instruction finetuning
        text = f"Instruction: {entry['instruction']}\nInput: {entry['input']}\nOutput: {entry['output']}"
        tokenized = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
        )
        return {key: torch.tensor(val) for key, val in tokenized.items()}

def load_jsonl_dataset(file_path):
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
                data.append(entry)
            except json.JSONDecodeError:
                logger.exception("Failed to decode JSON line in %s", file_path)
                continue
    logger.info("Loaded %d entries from %s", len(data), file_path)
    return data

class PlotMetricsCallback(TrainerCallback):
    def __init__(self, plot_dir, file_prefix="metrics"):
        self.plot_dir = plot_dir
        self.file_prefix = file_prefix
        self.metrics = {"loss": [], "step": [], "learning_rate": []}

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            if "loss" in logs:
                self.metrics["loss"].append(logs["loss"])
                self.metrics["step"].append(state.global_step)
            if "learning_rate" in logs:
                self.metrics["learning_rate"].append(logs["learning_rate"])

    def on_train_end(self, args, state, control, **kwargs):
        os.makedirs(self.plot_dir, exist_ok=True)
        if self.metrics["loss"]:
            plt.figure()
            plt.plot(self.metrics["step"], self.metrics["loss"], label="Training Loss")
            plt.xlabel("Step")
            plt.ylabel("Loss")
            plt.title("Training Loss Curve")
            plt.legend()
            loss_plot_file = os.path.join(self.plot_dir, f"{self.file_prefix}_loss.png")
            plt.savefig(loss_plot_file)
            plt.close()
            logger.info(f"Saved loss curve plot to {loss_plot_file}")

# --- Main Training Function (Modified for Apple Silicon) ---

def train_model(
    model_name,
    dataset_path,
    max_length=1024,
    output_dir="./lora_finetuned_mac",
    num_epochs=3,
    per_device_train_batch_size=1,
):
    logger.info("Starting training for model: %s on Apple Silicon", model_name)

    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is not available. This script is designed for Apple Silicon.")
    device = torch.device("mps")
    logger.info("Using device: %s", device)

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
        use_cache=False,
    )

    model = get_peft_model(model, DEFAULT_LORA_CONFIG)
    model.print_trainable_parameters()

    dataset_entries = load_jsonl_dataset(dataset_path)
    random.shuffle(dataset_entries)
    train_dataset = CustomDataset(dataset_entries, tokenizer, max_length)

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        num_train_epochs=num_epochs,
        learning_rate=2e-4,
        logging_steps=10,
        save_steps=50,
        bf16=True,
        optim="adamw_torch",
        gradient_accumulation_steps=4,
        gradient_checkpointing=False, # Disabled to fix gradients issue on MPS
        dataloader_num_workers=0,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        remove_unused_columns=False,
        use_mps_device=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    plot_callback = PlotMetricsCallback(plot_dir=os.path.join(output_dir, "plots"))
    trainer.add_callback(plot_callback)

    logger.info("Starting Trainer...")
    trainer.train()

    logger.info("Saving final model to %s", output_dir)
    model.save_pretrained(output_dir)

def main():
    parser = argparse.ArgumentParser(
        description="Finetune a Causal LM on a Mac with LoRA"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Pre-trained model name. A smaller model like 'PY007/TinyLlama-1.1B-Chat-v0.3' is recommended.",
    )
    parser.add_argument(
        "--dataset_path", type=str, required=True, help="Path to the JSONL dataset."
    )
    parser.add_argument(
        "--max_length", type=int, default=1024, help="Max token length for inputs."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./lora_finetuned_mac",
        help="Directory for the saved model.",
    )
    parser.add_argument(
        "--num_epochs", type=int, default=3, help="Number of training epochs."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size per device during training.",
    )
    args = parser.parse_args()

    logger.info("Arguments received: %s", args)
    train_model(
        model_name=args.model_name,
        dataset_path=args.dataset_path,
        max_length=args.max_length,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
    )

if __name__ == "__main__":
    main()