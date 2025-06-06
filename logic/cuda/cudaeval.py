import argparse
import json
import logging
import nltk
import torch
import textstat
import matplotlib.pyplot as plt # NEW: Import Matplotlib
import numpy as np # NEW: Import NumPy for bar chart positioning
import os # NEW: Import OS for creating plot directory

from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import single_meteor_score
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from rouge_score import rouge_scorer
from bert_score import score as bert_score_func

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
        inputs = tokenizer(prompt, return_tensors="pt")
        # Move each tensor in the inputs to the device
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        outputs = model.generate(
            **inputs, max_new_tokens=256, pad_token_id=tokenizer.eos_token_id, do_sample=False
        )
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return full_response[len(prompt):].strip()

# NEW: Function to plot evaluation scores
def plot_evaluation_scores(base_scores_avg, finetuned_scores_avg, output_dir):
    # Validate inputs
    if not base_scores_avg or not finetuned_scores_avg:
        raise ValueError("Score dictionaries cannot be empty")
    
    # Check if both dictionaries have the same keys
    if set(base_scores_avg.keys()) != set(finetuned_scores_avg.keys()):
        raise KeyError("Base scores and finetuned scores must have the same metrics")
    
    plot_output_dir = os.path.join(output_dir, "evaluation_plots")
    os.makedirs(plot_output_dir, exist_ok=True)
    logger.info(f"Saving evaluation plots to: {plot_output_dir}")

    metrics_to_plot = list(base_scores_avg.keys())
    
    for metric_name in metrics_to_plot:
        base_val = base_scores_avg[metric_name]
        finetuned_val = finetuned_scores_avg[metric_name]
        
        labels = ['Base Model', 'Finetuned Model']
        values = [base_val, finetuned_val]
        
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(8, 6))
        
        bar_colors = ['skyblue', 'lightcoral']
        bars = ax.bar(labels, values, color=bar_colors, width=0.5)

        # Add text labels on top of bars
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01 * max(values), f'{yval:.4f}', ha='center', va='bottom')

        ax.set_ylabel('Score', fontsize=12)
        ax.set_title(f'Comparison for {metric_name.upper()}', fontsize=14, fontweight='bold')
        ax.set_ylim(0, max(values) * 1.15 if max(values) > 0 else 0.1) # Adjust y-limit for better visualization
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plot_file_path = os.path.join(plot_output_dir, f"{metric_name}_comparison.png")
        plt.savefig(plot_file_path, dpi=300)
        plt.close(fig)
        logger.info(f"Saved plot: {plot_file_path}")


def evaluate_and_compare(base_model_path, finetuned_model_path, test_data_path, num_samples):
    logger.info("Loading models...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16
    
    base_tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    base_model = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype=dtype, device_map=device)
    finetuned_tokenizer = AutoTokenizer.from_pretrained(finetuned_model_path)
    finetuned_model = AutoModelForCausalLM.from_pretrained(finetuned_model_path, torch_dtype=dtype, device_map=device)
    
    for tokenizer in [base_tokenizer, finetuned_tokenizer]:
        if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    test_data = load_test_data(test_data_path)
    if num_samples > 0 and len(test_data) > num_samples:
        test_data = test_data[:num_samples]
    logger.info(f"Evaluating on {len(test_data)} samples.")

    score_types = ["bleu", "meteor", "rouge1", "rouge2", "rougeL", "ttr", "readability"]
    base_scores_all_samples = {stype: [] for stype in score_types} # Stores all individual scores
    finetuned_scores_all_samples = {stype: [] for stype in score_types}
    
    rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    all_refs, all_base_gens, all_finetuned_gens = [], [], []

    for sample in tqdm(test_data, desc="Generating Responses"):
        prompt = f"Instruction: {sample['instruction']}\nInput: {sample['input']}\nOutput:"
        ref_output = sample["output"]
        
        base_gen = generate_response(base_model, base_tokenizer, prompt)
        finetuned_gen = generate_response(finetuned_model, finetuned_tokenizer, prompt)

        all_refs.append(ref_output)
        all_base_gens.append(base_gen)
        all_finetuned_gens.append(finetuned_gen)

        ref_tokens = nltk.word_tokenize(ref_output)
        base_tokens = nltk.word_tokenize(base_gen)
        finetuned_tokens = nltk.word_tokenize(finetuned_gen)

        base_scores_all_samples["bleu"].append(sentence_bleu([ref_tokens], base_tokens, weights=(0.5, 0.5)))
        finetuned_scores_all_samples["bleu"].append(sentence_bleu([ref_tokens], finetuned_tokens, weights=(0.5, 0.5)))
        base_scores_all_samples["meteor"].append(single_meteor_score(ref_tokens, base_tokens))
        finetuned_scores_all_samples["meteor"].append(single_meteor_score(ref_tokens, finetuned_tokens))
        
        base_rouge = rouge.score(ref_output, base_gen)
        finetuned_rouge = rouge.score(ref_output, finetuned_gen)
        for rouge_type in ['rouge1', 'rouge2', 'rougeL']:
            base_scores_all_samples[rouge_type].append(base_rouge[rouge_type].fmeasure)
            finetuned_scores_all_samples[rouge_type].append(finetuned_rouge[rouge_type].fmeasure)

        if len(base_tokens) > 0:
            base_scores_all_samples["ttr"].append(len(set(base_tokens)) / len(base_tokens))
            base_scores_all_samples["readability"].append(textstat.flesch_reading_ease(base_gen))
        else:
            base_scores_all_samples["ttr"].append(0)
            base_scores_all_samples["readability"].append(0)

        if len(finetuned_tokens) > 0:
            finetuned_scores_all_samples["ttr"].append(len(set(finetuned_tokens)) / len(finetuned_tokens))
            finetuned_scores_all_samples["readability"].append(textstat.flesch_reading_ease(finetuned_gen))
        else:
            finetuned_scores_all_samples["ttr"].append(0)
            finetuned_scores_all_samples["readability"].append(0)

    logger.info("Calculating BERTScore for all samples...")
    base_p, base_r, base_f1_bert = bert_score_func(all_base_gens, all_refs, lang="en", device=device, verbose=False)
    finetuned_p, finetuned_r, finetuned_f1_bert = bert_score_func(all_finetuned_gens, all_refs, lang="en", device=device, verbose=False)
    
    # Store average BERTScore F1
    base_scores_avg = {stype: sum(scores) / len(scores) if len(scores) > 0 else 0 for stype, scores in base_scores_all_samples.items()}
    finetuned_scores_avg = {stype: sum(scores) / len(scores) if len(scores) > 0 else 0 for stype, scores in finetuned_scores_all_samples.items()}
    
    base_scores_avg["bert_f1"] = base_f1_bert.mean().item()
    finetuned_scores_avg["bert_f1"] = finetuned_f1_bert.mean().item()


    print("\n" + "="*60)
    print("--- Comprehensive Model Evaluation Results ---")
    print("="*60)
    print(f"{'Metric':<18} | {'Base Model':<15} | {'Finetuned Model':<15}")
    print("-" * 60)
    print("--- Accuracy & Overlap ---")
    for stype in ["bleu", "meteor", "rouge1", "rouge2", "rougeL"]:
        print(f"{stype.upper():<18} | {base_scores_avg[stype]:<15.4f} | {finetuned_scores_avg[stype]:<15.4f}")
    print("--- Semantic Similarity ---")
    print(f"{'BERTScore F1':<18} | {base_scores_avg['bert_f1']:<15.4f} | {finetuned_scores_avg['bert_f1']:<15.4f}")
    print("--- Quality & Style ---")
    for stype in ["ttr", "readability"]:
        print(f"{stype.upper() + ' (Higher is Better)':<18} | {base_scores_avg[stype]:<15.4f} | {finetuned_scores_avg[stype]:<15.4f}")
    print("="*60)

    # NEW: Call the plotting function
    # Plots will be saved in a subfolder of the finetuned model's directory
    plot_evaluation_scores(base_scores_avg, finetuned_scores_avg, finetuned_model_path)


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