import argparse
import json
import logging
import nltk
import torch
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import single_meteor_score
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# --- NLTK Downloader ---
try:
    nltk.data.find("tokenizers/punkt")
    nltk.data.find("corpora/wordnet")
    nltk.data.find("corpora/omw-1.4")
except LookupError:
    logger.info("Downloading NLTK resources...")
    nltk.download("punkt", quiet=True)
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)

def load_test_data(file_path: str):
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            try: data.append(json.loads(line.strip()))
            except json.JSONDecodeError: continue
    return data

def generate_response(model, tokenizer, prompt: str) -> str:
    with torch.no_grad():
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs, max_new_tokens=256, pad_token_id=tokenizer.eos_token_id, do_sample=False
        )
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return full_response[len(prompt):].strip()

def evaluate_and_compare(base_model_path, finetuned_model_path, test_data_path, num_samples):
    logger.info("Loading models...")
    dtype = torch.bfloat16
    base_tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    base_model = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype=dtype, device_map="auto")
    finetuned_tokenizer = AutoTokenizer.from_pretrained(finetuned_model_path)
    finetuned_model = AutoModelForCausalLM.from_pretrained(finetuned_model_path, torch_dtype=dtype, device_map="auto")
    
    for tokenizer in [base_tokenizer, finetuned_tokenizer]:
        if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    test_data = load_test_data(test_data_path)
    if num_samples > 0: test_data = test_data[:num_samples]
    logger.info(f"Evaluating on {len(test_data)} samples.")

    base_scores, finetuned_scores = {"bleu": [], "meteor": []}, {"bleu": [], "meteor": []}
    for sample in tqdm(test_data, desc="Evaluating Samples"):
        prompt = f"Instruction: {sample['instruction']}\nInput: {sample['input']}\nOutput:"
        base_gen = generate_response(base_model, base_tokenizer, prompt)
        finetuned_gen = generate_response(finetuned_model, finetuned_tokenizer, prompt)
        ref_tokens = nltk.word_tokenize(sample["output"])
        base_tokens = nltk.word_tokenize(base_gen)
        finetuned_tokens = nltk.word_tokenize(finetuned_gen)

        base_scores["bleu"].append(sentence_bleu([ref_tokens], base_tokens, weights=(0.5, 0.5)))
        finetuned_scores["bleu"].append(sentence_bleu([ref_tokens], finetuned_tokens, weights=(0.5, 0.5)))
        base_scores["meteor"].append(single_meteor_score(ref_tokens, base_tokens))
        finetuned_scores["meteor"].append(single_meteor_score(ref_tokens, finetuned_tokens))

    avg_base_bleu = sum(base_scores["bleu"]) / len(base_scores["bleu"])
    avg_finetuned_bleu = sum(finetuned_scores["bleu"]) / len(finetuned_scores["bleu"])
    avg_base_meteor = sum(base_scores["meteor"]) / len(base_scores["meteor"])
    avg_finetuned_meteor = sum(finetuned_scores["meteor"]) / len(finetuned_scores["meteor"])

    print("\n--- Model Evaluation Results ---")
    print(f"{'Metric':<15} | {'Base Model':<15} | {'Finetuned Model':<15}")
    print("-" * 50)
    print(f"{'Avg. BLEU':<15} | {avg_base_bleu:<15.4f} | {avg_finetuned_bleu:<15.4f}")
    print(f"{'Avg. METEOR':<15} | {avg_base_meteor:<15.4f} | {avg_finetuned_meteor:<15.4f}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate and compare base vs. finetuned models.")
    parser.add_argument("--base_model_path", type=str, required=True, help="Original base model name/path")
    parser.add_argument("--finetuned_model_path", type=str, required=True, help="Path to the merged finetuned model")
    parser.add_argument("--test_data_path", type=str, required=True, help="Path to the JSONL test data")
    parser.add_argument("--num_samples", type=int, default=20, help="Number of samples to evaluate")
    args = parser.parse_args()
    evaluate_and_compare(**vars(args))

if __name__ == "__main__":
    main()