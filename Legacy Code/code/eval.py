import argparse
import json
import logging

import nltk
import torch
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import single_meteor_score
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# --- Download NLTK data (only needs to be done once) ---
try:
    # Check for all necessary NLTK data
    nltk.data.find("tokenizers/punkt")
    nltk.data.find("tokenizers/punkt_tab")
    nltk.data.find("corpora/wordnet")
    nltk.data.find("corpora/omw-1.4")
except LookupError:
    logger.info("One or more NLTK resources not found. Downloading...")
    # Add all required resources for download
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)
    logger.info("NLTK resources downloaded.")


def load_test_data(file_path: str):
    """Loads test data from a JSONL file."""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                continue
    return data


def generate_response(model, tokenizer, prompt: str) -> str:
    """Generates a response from a model given a prompt."""
    with torch.no_grad():
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False,
        )
        
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return full_response[len(prompt):].strip()


def evaluate_and_compare(
    base_model_path: str,
    finetuned_model_path: str,
    test_data_path: str,
    num_samples: int,
):
    """
    Loads models, runs inference on test data, and evaluates BLEU/METEOR scores.
    """
    logger.info("Loading models... This may take a moment.")
    
    dtype = torch.bfloat16
    
    base_tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path, torch_dtype=dtype, device_map="auto"
    )
    
    finetuned_tokenizer = AutoTokenizer.from_pretrained(finetuned_model_path)
    finetuned_model = AutoModelForCausalLM.from_pretrained(
        finetuned_model_path, torch_dtype=dtype, device_map="auto"
    )

    if base_tokenizer.pad_token is None:
        base_tokenizer.pad_token = base_tokenizer.eos_token
    if finetuned_tokenizer.pad_token is None:
        finetuned_tokenizer.pad_token = finetuned_tokenizer.eos_token

    test_data = load_test_data(test_data_path)
    if num_samples > 0 and len(test_data) > num_samples:
        test_data = test_data[:num_samples]
    logger.info(f"Evaluating on {len(test_data)} samples.")
    
    base_scores = {"bleu": [], "meteor": []}
    finetuned_scores = {"bleu": [], "meteor": []}

    for sample in tqdm(test_data, desc="Evaluating Samples"):
        instruction = sample["instruction"]
        input_text = sample["input"]
        reference_output = sample["output"]
        
        prompt = f"Instruction: {instruction}\nInput: {input_text}\nOutput:"
        
        base_generated = generate_response(base_model, base_tokenizer, prompt)
        finetuned_generated = generate_response(finetuned_model, finetuned_tokenizer, prompt)
        
        reference_tokens = nltk.word_tokenize(reference_output)
        base_tokens = nltk.word_tokenize(base_generated)
        finetuned_tokens = nltk.word_tokenize(finetuned_generated)

        base_scores["bleu"].append(sentence_bleu([reference_tokens], base_tokens, weights=(0.5, 0.5)))
        finetuned_scores["bleu"].append(sentence_bleu([reference_tokens], finetuned_tokens, weights=(0.5, 0.5)))
        
        base_scores["meteor"].append(single_meteor_score(reference_tokens, base_tokens))
        finetuned_scores["meteor"].append(single_meteor_score(reference_tokens, finetuned_tokens))

    avg_base_bleu = sum(base_scores["bleu"]) / len(base_scores["bleu"])
    avg_finetuned_bleu = sum(finetuned_scores["bleu"]) / len(finetuned_scores["bleu"])
    avg_base_meteor = sum(base_scores["meteor"]) / len(base_scores["meteor"])
    avg_finetuned_meteor = sum(finetuned_scores["meteor"]) / len(finetuned_scores["meteor"])

    print("\n--- Model Evaluation Results ---")
    print(f"{'Metric':<15} | {'Base Model':<15} | {'Finetuned Model':<15}")
    print("-" * 50)
    print(f"{'Avg. BLEU':<15} | {avg_base_bleu:<15.4f} | {avg_finetuned_bleu:<15.4f}")
    print(f"{'Avg. METEOR':<15} | {avg_base_meteor:<15.4f} | {avg_finetuned_meteor:<15.4f}")
    print("-" * 50)
    
    if avg_finetuned_bleu > avg_base_bleu and avg_finetuned_meteor > avg_base_meteor:
        print("\nConclusion: The finetuned model shows a clear improvement in generating accurate and relevant text.")
    else:
        print("\nConclusion: The finetuned model's performance is comparable to or lower than the base model. Review training data and parameters.")


def main():
    parser = argparse.ArgumentParser(description="Evaluate and compare a base model vs. a finetuned model.")
    parser.add_argument("--base_model_path", type=str, required=True, help="Path or name of the original base model.")
    parser.add_argument("--finetuned_model_path", type=str, required=True, help="Path to the merged finetuned model directory.")
    parser.add_argument("--test_data_path", type=str, required=True, help="Path to the JSONL file with test data.")
    parser.add_argument("--num_samples", type=int, default=20, help="Number of samples to evaluate on. Set to 0 to run on all.")
    args = parser.parse_args()
    
    evaluate_and_compare(
        base_model_path=args.base_model_path,
        finetuned_model_path=args.finetuned_model_path,
        test_data_path=args.test_data_path,
        num_samples=args.num_samples,
    )

if __name__ == "__main__":
    main()