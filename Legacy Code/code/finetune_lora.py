import argparse
import json
import logging
import os
import random

import matplotlib.pyplot as plt
import torch

# Ensure you have installed the `peft` library: pip install peft
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# --- New: Define custom LoRA configurations for specific models ---
# The LoRA hyperparameters here balance the trade-off between training efficiency and adaptation capacity.
# - r: Rank of the low-rank update matrices. A higher 'r' increases the expressiveness but also the number of trainable parameters.
# - lora_alpha: A scaling factor applied to the low-rank updates. Higher values result in stronger adaptation.
# - target_modules: The modules in the transformer where LoRA updates will be applied (commonly the query and value projections).
# - lora_dropout: A dropout probability to regularize the low-rank updates.
# - bias: Set to "none" for keeping the original bias parameters frozen.
DEFAULT_LORA_CONFIG = LoraConfig(
    r=8,  # Standard rank; chosen for a good balance between capacity & efficiency.
    lora_alpha=32,  # Scaling factor to ensure sufficient adaptation without overfitting.
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,  # 10% dropout to help regularize and prevent overfitting.
    bias="none",
    task_type="CAUSAL_LM",
)

LORA_CONFIGS = {
    "tinyllama": LoraConfig(
        r=8,  # For TinyLlama, a smaller model, r=8 is often sufficient.
        lora_alpha=32,  # Scaling factor aligns with the capacity of a 1.1B model.
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
    ),
    "h2o-danube": LoraConfig(
        r=12,  # Slightly higher rank for the 1.8B model to capture its complexity.
        lora_alpha=64,  # Increased scaling to better suit a larger parameter space.
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
    ),
    "fox-1": LoraConfig(
        r=8,  # Consistent with the other 1.6B models; maintaining balance between model size and efficiency.
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
    ),
    "bitnet": LoraConfig(
        r=8,  # For BitNet b1.58, similar hyperparameters ensure parameter-efficiency.
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
    ),
    "smol": LoraConfig(
        r=8,  # SmolLM v2 is chosen with these settings to support its 8,000-token limit.
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
    ),
    "deepseek": LoraConfig(
        r=8,  # DeepSeek Coder benefits from this setup while supporting long-context (16,384 tokens).
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
    ),
}


def get_lora_config(model_name: str) -> LoraConfig:
    model_name_lower = model_name.lower()
    if "tinyllama" in model_name_lower:
        logger.debug("Using LoRA config for TinyLlama")
        return LORA_CONFIGS["tinyllama"]
    elif "h2o" in model_name_lower or "danube" in model_name_lower:
        logger.debug("Using LoRA config for H2O-Danube")
        return LORA_CONFIGS["h2o-danube"]
    elif "fox" in model_name_lower:
        logger.debug("Using LoRA config for Fox-1")
        return LORA_CONFIGS["fox-1"]
    elif "bitnet" in model_name_lower:
        logger.debug("Using LoRA config for BitNet")
        return LORA_CONFIGS["bitnet"]
    elif "smol" in model_name_lower:
        logger.debug("Using LoRA config for SmolLM")
        return LORA_CONFIGS["smol"]
    elif "deepseek" in model_name_lower or "coder" in model_name_lower:
        logger.debug("Using LoRA config for DeepSeek Coder")
        return LORA_CONFIGS["deepseek"]
    else:
        logger.warning(
            "No specific LoRA config found for %s, using default", model_name
        )
        return DEFAULT_LORA_CONFIG


# --- End new configuration block ---


# Custom dataset class to read our JSONL entries.
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        logger.debug("Initialized CustomDataset with %d entries", len(data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        # Combine the fields: instruction, input, and output – adjust as needed.
        text = ""
        if "instruction" in entry:
            text += entry["instruction"] + "\n"
        if "input" in entry:
            text += entry["input"] + "\n"
        if "output" in entry:
            text += entry["output"]
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
    logger.debug("Loaded %d entries from %s", len(data), file_path)
    return data


class PlotMetricsCallback(TrainerCallback):
    """
    Callback to record and plot training metrics (loss, learning rate) during training.
    At the end of training, the curves are saved to the specified folder.
    """

    def __init__(self, plot_dir, file_prefix="metrics"):
        self.plot_dir = plot_dir
        self.file_prefix = file_prefix
        self.metrics = {"loss": [], "step": [], "learning_rate": []}

    def on_log(self, args, state, control, logs=None, **kwargs):
        # Record loss and learning rate (if available) from each logging step.
        if logs is not None:
            if "loss" in logs:
                self.metrics["loss"].append(logs["loss"])
                self.metrics["step"].append(state.global_step)
            if "learning_rate" in logs:
                self.metrics["learning_rate"].append(logs["learning_rate"])

    def on_train_end(self, args, state, control, **kwargs):
        # Ensure the plot directory exists
        os.makedirs(self.plot_dir, exist_ok=True)

        # Plot the training loss curve.
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

        # Optionally, plot the learning rate curve if available.
        if self.metrics["learning_rate"]:
            plt.figure()
            # Align the number of steps with available learning rate data.
            plt.plot(
                self.metrics["step"][: len(self.metrics["learning_rate"])],
                self.metrics["learning_rate"],
                label="Learning Rate",
            )
            plt.xlabel("Step")
            plt.ylabel("Learning Rate")
            plt.title("Learning Rate Curve")
            plt.legend()
            lr_plot_file = os.path.join(self.plot_dir, f"{self.file_prefix}_lr.png")
            plt.savefig(lr_plot_file)
            plt.close()
            logger.info(f"Saved learning rate plot to {lr_plot_file}")


def train_model(
    model_name,
    dataset_path,
    max_length=1024,
    output_dir="./lora_finetuned",
    num_epochs=3,
    per_device_train_batch_size=1,
):
    logger.info("Starting training for model: %s", model_name)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Configure 4-bit quantization
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    # Load model with 4-bit quantization
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto",
        use_cache=False,
        low_cpu_mem_usage=True,
    )

    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)

    # Get LoRA configuration
    lora_config = get_lora_config(model_name)

    # Enable gradient checkpointing
    model.gradient_checkpointing_enable()

    # Apply LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load and prepare dataset
    dataset_entries = load_jsonl_dataset(dataset_path)
    random.shuffle(dataset_entries)
    split_idx = int(0.8 * len(dataset_entries))
    train_entries = dataset_entries[:split_idx]
    eval_entries = dataset_entries[split_idx:]

    train_dataset = CustomDataset(train_entries, tokenizer, max_length)
    eval_dataset = CustomDataset(eval_entries, tokenizer, max_length)

    # Optimized training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        num_train_epochs=num_epochs,
        learning_rate=2e-4,
        logging_steps=10,
        save_steps=50,
        fp16=True,
        optim="paged_adamw_8bit",
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        ddp_find_unused_parameters=False,
        dataloader_num_workers=2,
        dataloader_pin_memory=True,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        remove_unused_columns=False,  # Important for custom datasets
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    # Add metrics plotting callback
    plot_callback = PlotMetricsCallback(plot_dir=os.path.join(output_dir, "plots"))
    trainer.add_callback(plot_callback)

    # Start training
    trainer.train()

    # Save the final model
    model.save_pretrained(output_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Finetune a causal language model with LoRA using a custom JSONL dataset"
    )
    parser.add_argument(
        "--model_name", type=str, required=True, help="Pre-trained model name or path"
    )
    parser.add_argument(
        "--dataset_path", type=str, required=True, help="Path to the JSONL dataset"
    )
    parser.add_argument(
        "--max_length", type=int, default=1024, help="Max token length for inputs"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./lora_finetuned",
        help="Directory for the saved fine-tuned model",
    )
    parser.add_argument(
        "--num_epochs", type=int, default=3, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size per device during training",
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
