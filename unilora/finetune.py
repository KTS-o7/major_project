import argparse
import json
import logging
import math
import os
import random

import matplotlib.pyplot as plt
import torch
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# --- LoRA Configurations ---
LORA_CONFIGS = {
    "tinyllama": LoraConfig(r=8, lora_alpha=32, target_modules=["q_proj", "v_proj"], lora_dropout=0.1, bias="none", task_type="CAUSAL_LM"),
    "h2o-danube": LoraConfig(r=12, lora_alpha=64, target_modules=["q_proj", "v_proj"], lora_dropout=0.1, bias="none", task_type="CAUSAL_LM"),
    "fox": LoraConfig(r=8, lora_alpha=32, target_modules=["q_proj", "v_proj"], lora_dropout=0.1, bias="none", task_type="CAUSAL_LM"),
    "bitnet": LoraConfig(r=8, lora_alpha=32, target_modules=["q_proj", "v_proj"], lora_dropout=0.1, bias="none", task_type="CAUSAL_LM"),
    "smol": LoraConfig(r=8, lora_alpha=32, target_modules=["q_proj", "v_proj"], lora_dropout=0.1, bias="none", task_type="CAUSAL_LM"),
    "deepseek": LoraConfig(r=8, lora_alpha=32, target_modules=["q_proj", "v_proj"], lora_dropout=0.1, bias="none", task_type="CAUSAL_LM"),
}
DEFAULT_LORA_CONFIG = LoraConfig(r=8, lora_alpha=32, target_modules=["q_proj", "v_proj"], lora_dropout=0.1, bias="none", task_type="CAUSAL_LM")

def get_lora_config(model_name: str) -> LoraConfig:
    """Selects the appropriate LoRA config based on the model name."""
    model_name_lower = model_name.lower()
    for key, config in LORA_CONFIGS.items():
        if key in model_name_lower:
            logger.info(f"Using LoRA config for '{key}'")
            return config
    logger.warning(f"No specific LoRA config found for {model_name}. Using default.")
    return DEFAULT_LORA_CONFIG

# --- Dataset Class ---
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        entry = self.data[idx]
        text = f"Instruction: {entry['instruction']}\nInput: {entry['input']}\nOutput: {entry['output']}"
        tokenized = self.tokenizer(text, truncation=True, max_length=self.max_length, padding="max_length")
        return {key: torch.tensor(val) for key, val in tokenized.items()}

def load_jsonl_dataset(file_path):
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                continue
    return data

# --- NEW: Enhanced Plotting Callback ---
class PlotMetricsCallback(TrainerCallback):
    def __init__(self, plot_dir, file_prefix="metrics"):
        self.plot_dir = plot_dir
        self.file_prefix = file_prefix
        # Add learning_rate to the metrics to track
        self.metrics = {"loss": [], "step": [], "eval_loss": [], "eval_step": [], "learning_rate": []}

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            if "loss" in logs:
                self.metrics["loss"].append(logs["loss"])
                # The step is always the global_step
                self.metrics["step"].append(state.global_step)
            if "eval_loss" in logs:
                self.metrics["eval_loss"].append(logs["eval_loss"])
                self.metrics["eval_step"].append(state.global_step)
            # Capture learning rate from the logs
            if "learning_rate" in logs:
                self.metrics["learning_rate"].append(logs["learning_rate"])

    def on_train_end(self, args, state, control, **kwargs):
        os.makedirs(self.plot_dir, exist_ok=True)
        
        # --- Plot 1: Training and Evaluation Loss ---
        # Always plot training loss if it exists.
        if self.metrics["loss"]:
            plt.style.use('seaborn-v0_8-whitegrid')
            fig, ax = plt.subplots(figsize=(12, 7))
            
            # Plot training loss
            ax.plot(self.metrics["step"], self.metrics["loss"], label="Training Loss", color="royalblue", lw=2)
            
            # If evaluation loss also exists, plot it on the same graph
            if self.metrics["eval_loss"]:
                ax.plot(self.metrics["eval_step"], self.metrics["eval_loss"], label="Evaluation Loss", color="darkorange", marker='o', linestyle='--', ms=6)

            ax.set_xlabel("Steps", fontsize=12)
            ax.set_ylabel("Loss", fontsize=12)
            ax.set_title("Model Loss Curves", fontsize=16, fontweight='bold')
            ax.legend(fontsize=12)
            ax.grid(True, which='both', linestyle='--', linewidth=0.5)
            fig.tight_layout()

            plot_file = os.path.join(self.plot_dir, f"{self.file_prefix}_loss_curves.png")
            plt.savefig(plot_file, dpi=300)
            plt.close(fig)
            logger.info(f"Saved loss curve plot to {plot_file}")

        # --- Plot 2: Learning Rate ---
        # Separately plot learning rate if it was tracked.
        if self.metrics["learning_rate"]:
            plt.style.use('seaborn-v0_8-whitegrid')
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # The number of LR logs might differ from loss logs, so we plot against its own progression
            lr_steps = self.metrics["step"][:len(self.metrics["learning_rate"])]
            ax.plot(lr_steps, self.metrics["learning_rate"], label="Learning Rate", color="mediumseagreen", lw=2)
            
            ax.set_xlabel("Steps", fontsize=12)
            ax.set_ylabel("Learning Rate", fontsize=12)
            ax.set_title("Learning Rate Schedule", fontsize=16, fontweight='bold')
            ax.legend(fontsize=12)
            ax.grid(True)
            fig.tight_layout()

            lr_plot_file = os.path.join(self.plot_dir, f"{self.file_prefix}_lr_curve.png")
            plt.savefig(lr_plot_file, dpi=300)
            plt.close(fig)
            logger.info(f"Saved learning rate plot to {lr_plot_file}")


# --- Main Training Function ---
def train_model(model_name, dataset_path, max_length, output_dir, num_epochs, batch_size):
    logger.info(f"Starting training for model: {model_name} on Apple Silicon")
    if not torch.backends.mps.is_available(): raise RuntimeError("MPS not available.")
    device = torch.device("mps")

    trust_code = "bitnet" in model_name.lower()

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_code)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
        use_cache=False,
        trust_remote_code=trust_code
    )

    lora_config = get_lora_config(model_name)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dataset_entries = load_jsonl_dataset(dataset_path)
    random.shuffle(dataset_entries)
    split_idx = int(0.9 * len(dataset_entries))
    train_dataset = CustomDataset(dataset_entries[:split_idx], tokenizer, max_length)
    eval_dataset = CustomDataset(dataset_entries[split_idx:], tokenizer, max_length)

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        learning_rate=2e-4,
        logging_steps=10,
        eval_steps=10,
        save_steps=50,
        bf16=False,
        optim="adamw_torch",
        gradient_accumulation_steps=4,
        gradient_checkpointing=False,
        dataloader_num_workers=0,
        use_mps_device=True,
    )

    trainer = Trainer(
        model=model, args=training_args,
        train_dataset=train_dataset, eval_dataset=eval_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )
    trainer.add_callback(PlotMetricsCallback(plot_dir=os.path.join(output_dir, "plots")))
    trainer.train()
    model.save_pretrained(output_dir)
    logger.info(f"Training complete. LoRA adapter saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Finetune a Causal LM on a Mac with LoRA")
    parser.add_argument("--model_name", type=str, required=True, help="Hugging Face model name")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the JSONL dataset")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the LoRA adapter")
    parser.add_argument("--max_length", type=int, default=1024, help="Max token length")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="Training batch size")
    args = parser.parse_args()
    train_model(**vars(args))

if __name__ == "__main__":
    main()