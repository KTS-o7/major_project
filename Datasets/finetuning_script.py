import os
import json
import torch
import argparse
import random
import re
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoTokenizer,
    TrainingArguments,
    Seq2SeqTrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    prepare_model_for_kbit_training
)
try:
    import wandb
except ImportError:
    wandb = None
from tqdm import tqdm
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def augment_log_data(log: str) -> str:
    """Apply various augmentation techniques to log data."""
    augmented = log
    
    # 1. Log Level Variation (Most Important)
    if random.random() < 0.3:
        log_levels = ['INFO', 'WARNING', 'ERROR', 'DEBUG', 'CRITICAL']
        for level in log_levels:
            if level in augmented:
                augmented = augmented.replace(level, random.choice(log_levels))
                break
    
    # 2. IP Address Variation
    if random.random() < 0.2:
        ip_pattern = r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}'
        if re.search(ip_pattern, augmented):
            new_ip = f"{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}"
            augmented = re.sub(ip_pattern, new_ip, augmented)
    
    # 3. Timestamp Format Variation
    if random.random() < 0.2:
        timestamp_patterns = [
            r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}',  # 2024-03-14 15:30:45
            r'\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2}',  # 14/03/2024 15:30:45
            r'\d{2}-\d{2}-\d{4} \d{2}:\d{2}:\d{2}'  # 14-03-2024 15:30:45
        ]
        for pattern in timestamp_patterns:
            if re.search(pattern, augmented):
                # Convert to different format
                if pattern == timestamp_patterns[0]:
                    new_format = f"{random.randint(1, 31):02d}/{random.randint(1, 12):02d}/{random.randint(2020, 2024)} {random.randint(0, 23):02d}:{random.randint(0, 59):02d}:{random.randint(0, 59):02d}"
                else:
                    new_format = f"{random.randint(2020, 2024)}-{random.randint(1, 12):02d}-{random.randint(1, 31):02d} {random.randint(0, 23):02d}:{random.randint(0, 59):02d}:{random.randint(0, 59):02d}"
                augmented = re.sub(pattern, new_format, augmented)
                break
    
    # 4. Process ID Variation
    if random.random() < 0.2:
        pid_pattern = r'\[pid:\d+\]'
        if re.search(pid_pattern, augmented):
            new_pid = f"[pid:{random.randint(1000, 9999)}]"
            augmented = re.sub(pid_pattern, new_pid, augmented)
    
    # 5. Error Code Variation
    if random.random() < 0.2:
        error_pattern = r'error_code=\d+'
        if re.search(error_pattern, augmented):
            new_error = f"error_code={random.randint(100, 999)}"
            augmented = re.sub(error_pattern, new_error, augmented)
    
    # 6. Case Variation (for non-keywords)
    if random.random() < 0.3:
        # Preserve important keywords
        preserved_words = {'ERROR', 'WARNING', 'INFO', 'DEBUG', 'CRITICAL', 'pid', 'error_code'}
        parts = augmented.split()
        for i in range(len(parts)):
            if parts[i] not in preserved_words and random.random() < 0.3:
                parts[i] = parts[i].lower() if random.random() < 0.5 else parts[i].upper()
        augmented = ' '.join(parts)
    
    # 7. Whitespace Normalization
    if random.random() < 0.2:
        augmented = re.sub(r'\s+', ' ', augmented)
        if random.random() < 0.5:
            augmented = augmented.replace(' ', '  ') # Double space for some variation
    
    return augmented

def get_model_config(model_name: str) -> Dict:
    """Get model-specific configuration."""
    configs = {
        "Qwen/Qwen2-1.5B-Instruct":{
            "task_type": TaskType.CAUSAL_LM,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
            "max_input_length": 8192,
            "max_target_length": 512,
            "requires_4bit": True,
            "is_causal": True,
            "model_type": "qwen"
        },
        "t5-small": {
            "task_type": TaskType.SEQ_2_SEQ_LM,
            "target_modules": ["q", "k", "v", "o"],  # Standard attention layers for T5
            "max_input_length": 512,
            "max_target_length": 128,
            "requires_4bit": False,
            "is_causal": False,
            "model_type": "t5"
        },
        "distilbert/distilbert-base-uncased": {
            "task_type": TaskType.CAUSAL_LM, # Changed to CAUSAL_LM for text generation, though BERT is typically MLM.
                                            # If you intend to use DistilBERT for summarization, you might need a different approach
                                            # or a Seq2Seq model. For this script, assuming generative task.
            "target_modules": ["q_lin", "k_lin", "v_lin"], # Common for DistilBERT attention
            "max_input_length": 512,
            "max_target_length": 128,
            "requires_4bit": False,
            "is_causal": True, # Treat as causal for text generation
            "model_type": "distilbert"
        },
        "mistralai/Mistral-7B-Instruct-v0.3": {
            "task_type": TaskType.CAUSAL_LM,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"], # Standard for many transformer-based Causal LMs
            "max_input_length": 4096,
            "max_target_length": 512,
            "requires_4bit": True,
            "is_causal": True
        },
        "microsoft/Phi-3-mini-128k-instruct": {
            "task_type": TaskType.CAUSAL_LM,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"], # Standard for many transformer-based Causal LMs
            "max_input_length": 2048,
            "max_target_length": 256,
            "requires_4bit": True,
            "is_causal": True
        },
        "TinyLlama/TinyLlama_v1.1":{
            "task_type": TaskType.CAUSAL_LM,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"], # Standard for many transformer-based Causal LMs
            "max_input_length": 2048,
            "max_target_length": 256,
            "requires_4bit": True,
            "is_causal": True
        }
    }
    return configs.get(model_name, {
        "task_type": TaskType.SEQ_2_SEQ_LM, # Default to Seq2Seq
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "max_input_length": 1024,
        "max_target_length": 256,
        "requires_4bit": False,
        "is_causal": False
    })

def prepare_model_and_tokenizer(
    model_name: str,
    use_lora: bool = True,
    lora_r: int = 32,
    lora_alpha: int = 64,
    lora_dropout: float = 0.1
):
    """Prepare the model and tokenizer for training."""
    logger.info(f"Loading model and tokenizer from {model_name}")
    start_time = time.time()
    
    # Check for GPU availability
    if not torch.cuda.is_available():
        raise RuntimeError("This script requires a GPU to run. Please ensure CUDA is available.")
    
    device = "cuda"
    logger.info(f"CUDA available: {torch.cuda.is_available()} - Using device: {torch.cuda.get_device_name(0)}")
    
    try:
        # Get model-specific configuration
        model_config = get_model_config(model_name)
        
        # Load tokenizer with model-specific settings
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            local_files_only=False
        )
        
        # Set pad token for all models
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                # For models like T5, we should be careful about using eos_token as pad_token
                if "t5" in model_name.lower():
                    logger.warning(
                        "Using eos_token as pad_token for T5 model. This might affect generation performance. "
                        "Consider using a dedicated pad token if available."
                    )
                tokenizer.pad_token = tokenizer.eos_token
                tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
            else:
                # For models without eos_token, use a generic pad token
                logger.warning(
                    f"No eos_token found for {model_name}. Using a generic pad token. "
                    "This might affect model performance."
                )
                tokenizer.pad_token = "[PAD]"
                tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
        
        logger.info(f"Tokenizer vocabulary size: {len(tokenizer)}")
        logger.info(f"Tokenizer pad token: {tokenizer.pad_token}")
        logger.info(f"Tokenizer pad token id: {tokenizer.pad_token_id}")
        
        # Configure quantization if needed
        if model_config["requires_4bit"]:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float32,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True
            )
        else:
            quantization_config = None
        
        # Load appropriate model based on task type
        if model_config["task_type"] == TaskType.CAUSAL_LM:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map=None,
                torch_dtype=torch.float32,
                quantization_config=quantization_config,
                local_files_only=False
            )
        else:
            # For sequence-to-sequence models like T5 and BART
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                device_map=None,
                torch_dtype=torch.float32,
                quantization_config=quantization_config,
                local_files_only=False
            )
        
        if model_config["requires_4bit"]:
            model = prepare_model_for_kbit_training(model)
        
        # Configure LoRA with model-specific settings
        logger.info("Configuring LoRA parameters...")
        peft_config = LoraConfig(
            task_type=model_config["task_type"],
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=model_config["target_modules"],
            bias="none",
            inference_mode=False
        )
        model = get_peft_model(model, peft_config)
        
        # Move model to GPU and ensure parameters have gradients
        model = model.to(device)
        model.train()
        
        # Enable gradients for all parameters
        for param in model.parameters():
            param.requires_grad = True
        
        # Verify that some parameters have gradients enabled
        has_gradients = any(p.requires_grad for p in model.parameters())
        if not has_gradients:
            raise RuntimeError("No parameters have gradients enabled. Check model configuration.")
        
        model.print_trainable_parameters()
        logger.info(f"Model loaded successfully. Device: {next(model.parameters()).device}")
        logger.info(f"Model preparation completed in {time.time() - start_time:.2f} seconds")
        return model, tokenizer, model_config
        
    except Exception as e:
        logger.error(f"Error loading model or tokenizer: {str(e)}")
        raise

def load_training_data(data_path: str, augment: bool = True) -> Dataset:
    """Load and prepare the training data with optional augmentation."""
    logger.info(f"Loading training data from {data_path}")
    start_time = time.time()
    
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    logger.info(f"Loaded {len(data)} original examples")
    
    formatted_data = []
    for item in data:
        # Original data
        formatted_data.append({
            "input_text": item['input'],
            "target_text": item['output']
        })
        
        # Augmented data (create multiple variations)
        if augment:
            for _ in range(2):  # Create 2 variations per log
                augmented_input = augment_log_data(item['input'])
                formatted_data.append({
                    "input_text": augmented_input,
                    "target_text": item['output']
                })
    
    # Create dataset and split into train/validation
    dataset = Dataset.from_list(formatted_data)
    dataset = dataset.train_test_split(test_size=0.1, seed=42)
    
    logger.info(f"Created {len(dataset['train'])} training examples and {len(dataset['test'])} validation examples")
    logger.info(f"Data loading completed in {time.time() - start_time:.2f} seconds")
    
    return dataset

def train(
    model_name: str,
    data_path: str,
    output_dir: str,
    num_train_epochs: int = 3,
    per_device_train_batch_size: int = 8,  # Reduced for 8GB VRAM
    gradient_accumulation_steps: int = 4,  # Increased for 8GB VRAM
    learning_rate: float = 2e-5,
    use_wandb: bool = True,
    wandb_project: str = "log-summarizer",
    wandb_run_name: Optional[str] = None,
    augment_data: bool = True,
    early_stopping_patience: int = 3,
    fp16: bool = True,  # Added fp16 parameter
    gradient_checkpointing: bool = True  # Added gradient_checkpointing parameter
):
    """Train the model on the log summarization dataset."""
    logger.info("Starting training process...")
    start_time = time.time()
    
    # Initialize wandb if enabled and available
    if use_wandb and wandb is not None:
        logger.info(f"Initializing Weights & Biases with project: {wandb_project}")
        wandb.init(
            project=wandb_project,
            name=wandb_run_name or f"log-summarizer-{model_name.split('/')[-1]}"
        )
    elif use_wandb and wandb is None:
        logger.warning("Weights & Biases requested but not installed. Running without wandb logging.")
        use_wandb = False
    
    # Load and prepare data
    dataset = load_training_data(data_path, augment=augment_data)
    
    # Prepare model and tokenizer
    model, tokenizer, model_config = prepare_model_and_tokenizer(
        model_name,
        lora_r=8,  # Reduced for 8GB VRAM
        lora_alpha=16,  # Reduced for 8GB VRAM
        lora_dropout=0.1
    )
    
    # Tokenize dataset
    logger.info("Tokenizing dataset...")
    def tokenize_function(examples):
        model_inputs = tokenizer(
            examples["input_text"],
            max_length=model_config["max_input_length"],
            padding="max_length",
            truncation=True,
            return_tensors=None  # Let HF Datasets handle tensor conversion
        )
        
        # Tokenize targets
        labels = tokenizer(
            examples["target_text"],
            max_length=model_config["max_target_length"],
            padding="max_length",
            truncation=True,
            return_tensors=None  # Let HF Datasets handle tensor conversion
        )
        
        # For Causal LMs, labels should be input_ids, and attention_mask is also used.
        # For Seq2Seq, labels are usually just input_ids for the decoder.
        model_inputs["labels"] = labels["input_ids"]

        # Ensure attention_mask is present for labels in Causal LMs if needed for loss masking
        if model_config["is_causal"]:
            # In causal LMs, we often mask out the input part of the labels for loss calculation.
            # However, DataCollatorForLanguageModeling handles this by default.
            # If not using DataCollatorForLanguageModeling, you'd set labels to -100 for input tokens.
            pass 

        return model_inputs
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset["train"].column_names
    )
    logger.info(f"Tokenized dataset size: {len(tokenized_dataset['train'])} training, {len(tokenized_dataset['test'])} validation")
    
    # Configure training arguments
    logger.info("Configuring training arguments...")
    if model_config["task_type"] == TaskType.SEQ_2_SEQ_LM:
        training_args = Seq2SeqTrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            fp16=fp16,
            bf16=False,  # Disable bf16 when using fp16
            logging_steps=10,
            save_strategy="epoch",
            eval_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            report_to=["wandb"] if use_wandb else [],
            remove_unused_columns=False,
            push_to_hub=False,
            warmup_ratio=0.1,
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            gradient_checkpointing=gradient_checkpointing,
            max_grad_norm=1.0,
            optim="adamw_torch",
            save_total_limit=2,
            dataloader_num_workers=2,
            eval_accumulation_steps=2,
            logging_first_step=True,
            logging_dir=os.path.join(output_dir, "logs"),
            seed=42,
            disable_tqdm=False
        )
    else: # For Causal LMs (TaskType.CAUSAL_LM)
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            fp16=fp16,
            bf16=False,  # Disable bf16 when using fp16
            logging_steps=10,
            save_strategy="epoch",
            eval_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            report_to=["wandb"] if use_wandb else [],
            remove_unused_columns=False,
            push_to_hub=False,
            warmup_ratio=0.1,
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            gradient_checkpointing=gradient_checkpointing,
            max_grad_norm=1.0,
            optim="adamw_torch",
            save_total_limit=2,
            dataloader_num_workers=2,
            eval_accumulation_steps=2,
            logging_first_step=True,
            logging_dir=os.path.join(output_dir, "logs"),
            seed=42,
            disable_tqdm=False
        )
    
    # Choose appropriate data collator
    logger.info("Initializing data collator...")
    if model_config["is_causal"]:
        # For Causal LMs, DataCollatorForLanguageModeling is appropriate for next token prediction
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    else:
        # For Seq2Seq LMs, DataCollatorForSeq2Seq handles encoder-decoder inputs/labels
        data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    
    # Initialize trainer
    logger.info("Initializing trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)]
    )
    
    # Train the model
    logger.info("Starting model training...")
    trainer.train()
    
    # Save the model
    logger.info(f"Saving model to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    total_time = time.time() - start_time
    logger.info(f"Training completed in {total_time:.2f} seconds ({total_time/3600:.2f} hours)")
    
    if use_wandb and wandb is not None:
        wandb.finish()
        logger.info("Weights & Biases logging completed")

def main():
    parser = argparse.ArgumentParser(description="Fine-tune a language model for log summarization")
    
    # Model and data arguments
    parser.add_argument("--model_name", type=str, default="distilbert/distilbert-base-uncased",
                        help="Name or path of the base model to fine-tune")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to the training data JSON file")
    parser.add_argument("--output_dir", type=str, default="./output",
                        help="Directory to save the fine-tuned model")
    
    # Training arguments
    parser.add_argument("--num_train_epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8,  # Reduced for 8GB VRAM
                        help="Batch size per device")
    parser.add_argument("--grad_accum_steps", type=int, default=4,  # Increased for 8GB VRAM
                        help="Number of steps to accumulate gradients")
    parser.add_argument("--lr", type=float, default=2e-5,
                        help="Learning rate")
    parser.add_argument("--early_stop", type=int, default=3,
                        help="Number of epochs to wait before early stopping")
    
    # LoRA arguments
    parser.add_argument("--lora_r", type=int, default=8,  # Reduced for 8GB VRAM
                        help="LoRA attention dimension")
    parser.add_argument("--lora_alpha", type=int, default=16,  # Reduced for 8GB VRAM
                        help="LoRA alpha parameter")
    parser.add_argument("--lora_dropout", type=float, default=0.1,
                        help="LoRA dropout probability")
    
    # Memory optimization arguments
    parser.add_argument("--gradient_checkpointing", action="store_true",
                        help="Enable gradient checkpointing to save memory")
    parser.add_argument("--fp16", action="store_true",
                        help="Enable mixed precision training")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                        help="Maximum gradient norm for gradient clipping")
    
    # Logging arguments
    parser.add_argument("--wandb", action="store_true",
                        help="Use Weights & Biases for experiment tracking")
    parser.add_argument("--wandb_project", type=str, default="log-summarizer",
                        help="Weights & Biases project name")
    parser.add_argument("--wandb_run_name", type=str,
                        help="Weights & Biases run name")
    
    # Data augmentation
    parser.add_argument("--no_augment", action="store_true",
                        help="Disable data augmentation")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Train the model
    train(
        model_name=args.model_name,
        data_path=args.data_path,
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        learning_rate=args.lr,
        use_wandb=args.wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        augment_data=not args.no_augment,
        early_stopping_patience=args.early_stop,
        fp16=args.fp16,
        gradient_checkpointing=args.gradient_checkpointing
    )

if __name__ == "__main__":
    main()
