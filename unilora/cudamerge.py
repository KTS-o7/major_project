import argparse
import logging
import os
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def merge_and_save(base_model_path: str, lora_adapter_path: str, output_dir: str):
    logger.info(f"Loading base model from: {base_model_path}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path, torch_dtype=torch.bfloat16, device_map="auto"
    )
    logger.info(f"Loading LoRA adapter from: {lora_adapter_path}")
    peft_model = PeftModel.from_pretrained(base_model, lora_adapter_path)
    logger.info("Merging the LoRA adapter into the base model...")
    merged_model = peft_model.merge_and_unload()
    logger.info(f"Saving the merged model to: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    merged_model.save_pretrained(output_dir)
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    tokenizer.save_pretrained(output_dir)
    logger.info(f"Merge complete! Final model is in {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Merge a LoRA adapter with its base model.")
    parser.add_argument("--base_model_path", type=str, required=True, help="Original base model name/path")
    parser.add_argument("--lora_adapter_path", type=str, required=True, help="Path to the saved LoRA adapter")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the final merged model")
    args = parser.parse_args()
    merge_and_save(**vars(args))

if __name__ == "__main__":
    main()