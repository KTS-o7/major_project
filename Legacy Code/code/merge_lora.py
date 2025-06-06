import argparse
import logging
import os

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def merge_and_save(base_model_path: str, lora_adapter_path: str, output_dir: str):
    """
    Merges a LoRA adapter with a base model and saves the resulting model.

    Args:
        base_model_path (str): The path or Hugging Face name of the base model.
        lora_adapter_path (str): The path to the directory containing the LoRA adapter files.
        output_dir (str): The directory where the merged model will be saved.
    """
    logger.info(f"Loading base model from: {base_model_path}")
    
    # Load the base model. It's recommended to load in the same precision
    # as the training was done (bfloat16).
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",  # Let transformers handle device placement
    )

    logger.info(f"Loading LoRA adapter from: {lora_adapter_path}")
    
    # Load the PeftModel by combining the base model with the LoRA adapter.
    # This will load the adapter weights on top of the base model.
    peft_model = PeftModel.from_pretrained(base_model, lora_adapter_path)
    
    logger.info("Merging the LoRA adapter into the base model...")
    
    # The magic happens here: merge_and_unload() combines the weights.
    # This returns a new, standard AutoModelForCausalLM.
    merged_model = peft_model.merge_and_unload()
    
    logger.info(f"Saving the merged model to: {output_dir}")
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the merged model using the standard save_pretrained method.
    merged_model.save_pretrained(output_dir)
    
    # We also need to save the tokenizer for easy loading and use later.
    logger.info("Saving the tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    tokenizer.save_pretrained(output_dir)
    
    logger.info("Merge complete! Your final model is ready in %s", output_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Merge a LoRA finetuned adapter with its base model."
    )
    parser.add_argument(
        "--base_model_path",
        type=str,
        required=True,
        help="The name or path of the original base model (e.g., 'PY007/TinyLlama-1.1B-Chat-v0.3')."
    )
    parser.add_argument(
        "--lora_adapter_path",
        type=str,
        required=True,
        help="Path to the directory containing the finetuned LoRA adapter (your training output directory)."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="The directory where the final, merged model will be saved."
    )
    args = parser.parse_args()
    
    merge_and_save(
        base_model_path=args.base_model_path,
        lora_adapter_path=args.lora_adapter_path,
        output_dir=args.output_dir,
    )

if __name__ == "__main__":
    main()