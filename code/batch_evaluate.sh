#!/bin/bash
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
        python evaluate_model.py \
            --model_path "$model" \
            --dataset_path "$DATASET_PATH" \
            --sample_size $SAMPLE_SIZE \
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
    python evaluate_model.py \
        --compare_models "${existing_models[@]}" \
        --dataset_path "$DATASET_PATH" \
        --comparison_output_dir "$OUTPUT_DIR/comparison"
else
    echo "Need at least 2 models for comparison"
fi

echo "Evaluation complete! Results saved in: $OUTPUT_DIR"
echo "Check the following files:"
echo "- Individual model results: ${OUTPUT_DIR}/*_results.json"
echo "- Model comparison: ${OUTPUT_DIR}/comparison/model_comparison.csv"
