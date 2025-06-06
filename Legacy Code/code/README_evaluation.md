# Log Summarization Model Evaluation

This directory contains comprehensive evaluation tools for fine-tuned log summarization models using LoRA (Low-Rank Adaptation). The evaluation system supports multiple metrics including BLEU, METEOR, ROUGE, BERTScore, and custom log-specific metrics.

## Files Overview

- `evaluate_model.py` - Main evaluation script
- `finetune_lora.py` - LoRA fine-tuning script
- `generate_dataset.py` - Dataset generation script
- `requirements_evaluation.txt` - Python dependencies for evaluation
- `evaluate_example.py` - Usage examples and batch evaluation script
- `batch_evaluate.sh` - Shell script for batch evaluation (auto-generated)

## Installation

1. Install the required dependencies:

```bash
pip install -r requirements_evaluation.txt
```

2. Ensure you have your fine-tuned models and evaluation dataset ready.

## Evaluation Metrics

### Standard NLG Metrics

- **BLEU Score**: Measures n-gram overlap between prediction and reference
- **METEOR Score**: Considers synonyms, stemming, and word order
- **ROUGE-1/2/L**: Measures recall of unigrams, bigrams, and longest common subsequences
- **BERTScore**: Semantic similarity using BERT embeddings

### Custom Log Summarization Metrics

- **Term Coverage**: Percentage of reference terms that appear in predictions
- **Structure Score**: Presence of log summary structure elements (system, status, duration, etc.)
- **Length Ratio**: Comparison of prediction vs reference text length
- **Average Lengths**: Statistics on prediction and reference text lengths

## Usage Examples

### 1. Basic Single Model Evaluation

Evaluate a single fine-tuned model:

```bash
python evaluate_model.py \
    --model_path ./lora_finetuned \
    --dataset_path log_summarization_dataset.jsonl \
    --output_file ./evaluation_results/detailed_results.json
```

### 2. Sample Evaluation

Evaluate on a subset of your dataset for faster testing:

```bash
python evaluate_model.py \
    --model_path ./lora_finetuned \
    --dataset_path log_summarization_dataset.jsonl \
    --sample_size 100 \
    --output_file ./evaluation_results/sample_results.json
```

### 3. Multiple Model Comparison

Compare multiple fine-tuned models on the same dataset:

```bash
python evaluate_model.py \
    --compare_models ./tinyllama_finetuned ./h2o_danube_finetuned ./deepseek_finetuned \
    --dataset_path log_summarization_dataset.jsonl \
    --comparison_output_dir ./model_comparison_results
```

### 4. Specify Base Model

When the base model differs from the one in the config:

```bash
python evaluate_model.py \
    --model_path ./lora_finetuned \
    --base_model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --dataset_path log_summarization_dataset.jsonl \
    --output_file ./evaluation_results/results.json
```

### 5. Batch Evaluation

Run the example script to generate a batch evaluation shell script:

```bash
python evaluate_example.py
chmod +x batch_evaluate.sh
./batch_evaluate.sh
```

## Output Files

### Individual Model Evaluation

- `detailed_results.json` - Complete evaluation results including:
  - All computed metrics
  - Individual predictions and references
  - Model configuration details
- `detailed_results_metrics.csv` - Summary metrics in CSV format for easy analysis

### Model Comparison

- `model_comparison.json` - Comparison results for all models
- `model_comparison.csv` - Tabular comparison of metrics across models
- Individual detailed results for each model

## Dataset Format

Your evaluation dataset should be in JSONL format with the following structure:

```json
{"instruction": "Analyze the following log lines...", "input": "log content here", "output": "expected summary"}
{"instruction": "Analyze the following log lines...", "input": "log content here", "output": "expected summary"}
```

## Interpreting Results

### BLEU Score (0-1, higher is better)

- **> 0.3**: Excellent quality
- **0.2-0.3**: Good quality
- **0.1-0.2**: Moderate quality
- **< 0.1**: Poor quality

### METEOR Score (0-1, higher is better)

- **> 0.5**: Excellent
- **0.3-0.5**: Good
- **0.2-0.3**: Moderate
- **< 0.2**: Poor

### ROUGE Scores (0-1, higher is better)

- **ROUGE-1**: Unigram overlap (vocabulary coverage)
- **ROUGE-2**: Bigram overlap (fluency indicator)
- **ROUGE-L**: Longest common subsequence (structural similarity)

### Custom Metrics

- **Term Coverage (0-1)**: Higher values indicate better coverage of key terms
- **Structure Score (0-1)**: Higher values indicate better structured summaries
- **Length Ratio**: Values close to 1.0 indicate appropriate summary length

## Advanced Usage

### Customizing Generation Parameters

You can modify the generation parameters in the `generate_summary()` method:

```python
generation_config = GenerationConfig(
    max_new_tokens=150,    # Maximum tokens to generate
    temperature=0.7,       # Randomness (0.0-1.0)
    do_sample=True,        # Enable sampling
    top_p=0.9,            # Nucleus sampling threshold
)
```

### Adding Custom Metrics

To add your own evaluation metrics, extend the `_compute_custom_metrics()` method in the `ModelEvaluator` class.

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size or use smaller models
2. **BERTScore Not Available**: Install with `pip install bert-score`
3. **Model Loading Errors**: Ensure the model path and base model are correct
4. **Empty Predictions**: Check if the model is properly fine-tuned and prompts are formatted correctly

### Performance Tips

- Use `--sample_size` for quick testing
- Evaluate on GPU for faster inference
- Use mixed precision (`torch_dtype=torch.float16`) to save memory

## Model-Specific Notes

### TinyLlama

- Fast inference, good for testing
- May need specific prompting format

### H2O-Danube

- Larger model, better quality summaries
- Requires more GPU memory

### DeepSeek Coder

- Excellent for technical log analysis
- Supports longer context windows

### SmolLM v2

- Efficient for deployment
- Good balance of speed and quality

## Example Output

```
EVALUATION RESULTS
==================================================
bleu: 0.2847
meteor: 0.4521
rouge1: 0.5234
rouge2: 0.3876
rougeL: 0.4892
bertscore_f1: 0.7123
term_coverage: 0.6543
structure_score: 0.7890
length_ratio: 1.0234
num_samples: 500
```

## Integration with Training

This evaluation system integrates seamlessly with the LoRA fine-tuning pipeline:

1. **Generate Dataset**: Use `generate_dataset.py`
2. **Fine-tune Models**: Use `finetune_lora.py`
3. **Evaluate Performance**: Use `evaluate_model.py`
4. **Compare Models**: Use comparison features to select the best model

## Contributing

To extend the evaluation system:

1. Add new metrics in the `compute_metrics()` method
2. Implement custom scoring in `_compute_custom_metrics()`
3. Add support for new model architectures
4. Enhance the comparison visualization features

For questions or issues, please refer to the Hugging Face documentation for PEFT and Transformers libraries.
