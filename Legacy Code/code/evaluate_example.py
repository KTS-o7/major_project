#!/usr/bin/env python3
"""
Example script demonstrating how to use the model evaluation script.
This script shows different ways to evaluate your fine-tuned models.
"""

import os


def run_evaluation_examples():
    """Run various evaluation examples."""

    print("=" * 60)
    print("MODEL EVALUATION EXAMPLES")
    print("=" * 60)

    # Example paths (adjust these to your actual paths)
    model_path = "./lora_finetuned"  # Path to your fine-tuned model
    dataset_path = "log_summarization_dataset.jsonl"  # Your evaluation dataset

    print("\n1. Basic single model evaluation:")
    print("-" * 40)
    cmd1 = [
        "python",
        "eval.py",
        "--model_path",
        model_path,
        "--dataset_path",
        dataset_path,
        "--output_file",
        "./evaluation_results/detailed_results.json",
    ]
    print(" ".join(cmd1))

    print("\n2. Evaluate with a sample of 100 examples:")
    print("-" * 40)
    cmd2 = [
        "python",
        "evaluate_model.py",
        "--model_path",
        model_path,
        "--dataset_path",
        dataset_path,
        "--sample_size",
        "100",
        "--output_file",
        "./evaluation_results/sample_results.json",
    ]
    print(" ".join(cmd2))

    print("\n3. Compare multiple models:")
    print("-" * 40)
    cmd3 = [
        "python",
        "evaluate_model.py",
        "--compare_models",
        "./tinyllama_finetuned",
        "./h2o_danube_finetuned",
        "./deepseek_finetuned",
        "--dataset_path",
        dataset_path,
        "--comparison_output_dir",
        "./model_comparison_results",
    ]
    print(" ".join(cmd3))

    print("\n4. Evaluate with specific base model:")
    print("-" * 40)
    cmd4 = [
        "python",
        "evaluate_model.py",
        "--model_path",
        model_path,
        "--base_model",
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "--dataset_path",
        dataset_path,
        "--output_file",
        "./evaluation_results/tinyllama_results.json",
    ]
    print(" ".join(cmd4))

    print("\n" + "=" * 60)
    print("EVALUATION METRICS INCLUDED:")
    print("=" * 60)
    print("• BLEU Score - Measures n-gram overlap with reference")
    print("• METEOR Score - Considers synonyms and word order")
    print("• ROUGE-1/2/L - Measures recall of unigrams/bigrams/longest sequences")
    print("• BERTScore - Semantic similarity using BERT embeddings")
    print("• Custom Metrics:")
    print("  - Term Coverage: How many reference terms appear in prediction")
    print("  - Structure Score: Presence of log summary structure elements")
    print("  - Length Ratio: Comparison of prediction vs reference length")

    print("\n" + "=" * 60)
    print("OUTPUT FILES:")
    print("=" * 60)
    print("• detailed_results.json - Full evaluation results with predictions")
    print("• metrics.csv - Summary metrics in CSV format")
    print("• model_comparison.json - Comparison results for multiple models")
    print("• model_comparison.csv - Comparison metrics in tabular format")

    print("\n" + "=" * 60)
    print("TO RUN THESE EXAMPLES:")
    print("=" * 60)
    print("1. Make sure your fine-tuned models are saved in the expected paths")
    print("2. Ensure you have a test dataset in JSONL format")
    print("3. Install requirements: pip install -r requirements_evaluation.txt")
    print("4. Copy and run any of the commands above")


def create_sample_evaluation_script():
    """Create a sample script for batch evaluation."""

    script_content = """#!/bin/bash
# Batch evaluation script for all fine-tuned models

echo "Starting batch evaluation of fine-tuned models..."

# Define common parameters
DATASET_PATH="log_summarization_dataset.jsonl"
OUTPUT_DIR="./evaluation_results"
SAMPLE_SIZE=200

# Create output directory
mkdir -p $OUTPUT_DIR

# List of models to evaluate (adjust paths as needed)
MODELS=(
    "./tinyllama_finetuned"
    "./h2o_danube_finetuned"
    "./fox1_finetuned"
    "./bitnet_finetuned"
    "./smol_finetuned"
    "./deepseek_finetuned"
)

echo "Evaluating individual models..."

# Evaluate each model individually
for model in "${MODELS[@]}"; do
    if [ -d "$model" ]; then
        echo "Evaluating: $model"
        model_name=$(basename "$model")
        python evaluate_model.py \\
            --model_path "$model" \\
            --dataset_path "$DATASET_PATH" \\
            --sample_size $SAMPLE_SIZE \\
            --output_file "$OUTPUT_DIR/${model_name}_results.json"
    else
        echo "Model not found: $model"
    fi
done

echo "Comparing all models..."

# Compare all available models
existing_models=()
for model in "${MODELS[@]}"; do
    if [ -d "$model" ]; then
        existing_models+=("$model")
    fi
done

if [ ${#existing_models[@]} -gt 1 ]; then
    python evaluate_model.py \\
        --compare_models "${existing_models[@]}" \\
        --dataset_path "$DATASET_PATH" \\
        --comparison_output_dir "$OUTPUT_DIR/comparison"
else
    echo "Need at least 2 models for comparison"
fi

echo "Evaluation complete! Results saved in: $OUTPUT_DIR"
echo "Check the following files:"
echo "- Individual model results: ${OUTPUT_DIR}/*_results.json"
echo "- Model comparison: ${OUTPUT_DIR}/comparison/model_comparison.csv"
"""

    with open("code/batch_evaluate.sh", "w") as f:
        f.write(script_content)

    # Make the script executable
    os.chmod("code/batch_evaluate.sh", 0o755)

    print("\nCreated batch evaluation script: code/batch_evaluate.sh")
    print("Make it executable and run with: ./batch_evaluate.sh")


if __name__ == "__main__":
    run_evaluation_examples()
    create_sample_evaluation_script()
