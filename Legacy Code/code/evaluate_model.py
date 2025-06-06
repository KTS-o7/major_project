import argparse
import json
import logging
import os

import evaluate
import numpy as np
import pandas as pd
import torch
from peft import PeftConfig, PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Comprehensive evaluation class for fine-tuned log summarization models.
    Supports multiple metrics: BLEU, METEOR, ROUGE, BERTScore, and custom metrics.
    """

    def __init__(self, model_path: str, base_model_name: str = None):
        """
        Initialize the evaluator with a fine-tuned model.

        Args:
            model_path: Path to the fine-tuned LoRA model
            base_model_name: Name of the base model (if different from config)
        """
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load model and tokenizer
        self._load_model_and_tokenizer(base_model_name)

        # Initialize evaluation metrics
        self._initialize_metrics()

        logger.info(f"Evaluator initialized with model: {model_path}")
        logger.info(f"Device: {self.device}")

    def _load_model_and_tokenizer(self, base_model_name: str = None):
        """Load the fine-tuned model and tokenizer."""
        try:
            # Load PEFT config to get base model name
            peft_config = PeftConfig.from_pretrained(self.model_path)
            base_model = base_model_name or peft_config.base_model_name_or_path

            logger.info(f"Loading base model: {base_model}")
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Load base model
            self.base_model = AutoModelForCausalLM.from_pretrained(
                base_model,
                device_map="auto",
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
            )

            # Load fine-tuned model with LoRA weights
            self.model = PeftModel.from_pretrained(self.base_model, self.model_path)
            self.model.eval()

            logger.info("Model and tokenizer loaded successfully")

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def _initialize_metrics(self):
        """Initialize evaluation metrics."""
        try:
            # Standard NLG metrics
            self.bleu_metric = evaluate.load("bleu")
            self.meteor_metric = evaluate.load("meteor")
            self.rouge_metric = evaluate.load("rouge")

            # Semantic similarity metric
            try:
                self.bertscore_metric = evaluate.load("bertscore")
                self.has_bertscore = True
            except:
                logger.warning("BERTScore not available, skipping this metric")
                self.has_bertscore = False

            logger.info("Evaluation metrics initialized")

        except Exception as e:
            logger.error(f"Error initializing metrics: {e}")
            raise

    def generate_summary(
        self, log_text: str, max_length: int = 512, max_new_tokens: int = 150
    ) -> str:
        """
        Generate a summary for the given log text.

        Args:
            log_text: Input log text to summarize
            max_length: Maximum input length
            max_new_tokens: Maximum number of new tokens to generate

        Returns:
            Generated summary text
        """
        # Prepare input text with instruction
        prompt = f"Analyze the following log lines and generate a structured summary:\n{log_text}"

        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
            padding=True,
        ).to(self.device)

        # Generation configuration
        generation_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        # Generate summary
        with torch.no_grad():
            outputs = self.model.generate(**inputs, generation_config=generation_config)

        # Decode and clean output
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove the input prompt from the generated text
        summary = generated_text[len(prompt) :].strip()

        return summary

    def compute_metrics(self, predictions: list, references: list) -> dict:
        """
        Compute comprehensive evaluation metrics.

        Args:
            predictions: List of predicted summaries
            references: List of reference summaries

        Returns:
            Dictionary containing all computed metrics
        """
        metrics = {}

        # Ensure we have valid text for evaluation
        valid_pairs = [
            (pred, ref)
            for pred, ref in zip(predictions, references)
            if pred.strip() and ref.strip()
        ]

        if not valid_pairs:
            logger.warning("No valid prediction-reference pairs found")
            return metrics

        valid_predictions, valid_references = zip(*valid_pairs)

        try:
            # BLEU Score
            bleu_result = self.bleu_metric.compute(
                predictions=valid_predictions,
                references=[[ref] for ref in valid_references],
            )
            metrics["bleu"] = bleu_result["bleu"]

            # METEOR Score
            meteor_result = self.meteor_metric.compute(
                predictions=valid_predictions, references=valid_references
            )
            metrics["meteor"] = meteor_result["meteor"]

            # ROUGE Scores
            rouge_result = self.rouge_metric.compute(
                predictions=valid_predictions, references=valid_references
            )
            metrics["rouge1"] = rouge_result["rouge1"]
            metrics["rouge2"] = rouge_result["rouge2"]
            metrics["rougeL"] = rouge_result["rougeL"]
            metrics["rougeLsum"] = rouge_result["rougeLsum"]

            # BERTScore (if available)
            if self.has_bertscore:
                try:
                    bert_result = self.bertscore_metric.compute(
                        predictions=valid_predictions,
                        references=valid_references,
                        lang="en",
                    )
                    metrics["bertscore_precision"] = np.mean(bert_result["precision"])
                    metrics["bertscore_recall"] = np.mean(bert_result["recall"])
                    metrics["bertscore_f1"] = np.mean(bert_result["f1"])
                except Exception as e:
                    logger.warning(f"BERTScore computation failed: {e}")

            # Custom log summarization metrics
            metrics.update(
                self._compute_custom_metrics(valid_predictions, valid_references)
            )

        except Exception as e:
            logger.error(f"Error computing metrics: {e}")

        return metrics

    def _compute_custom_metrics(self, predictions: list, references: list) -> dict:
        """
        Compute custom metrics specific to log summarization.

        Args:
            predictions: List of predicted summaries
            references: List of reference summaries

        Returns:
            Dictionary with custom metrics
        """
        custom_metrics = {}

        # Average length metrics
        pred_lengths = [len(pred.split()) for pred in predictions]
        ref_lengths = [len(ref.split()) for ref in references]

        custom_metrics["avg_prediction_length"] = np.mean(pred_lengths)
        custom_metrics["avg_reference_length"] = np.mean(ref_lengths)
        custom_metrics["length_ratio"] = np.mean(pred_lengths) / np.mean(ref_lengths)

        # Coverage metrics (how many key terms from reference appear in prediction)
        coverage_scores = []
        for pred, ref in zip(predictions, references):
            pred_words = set(pred.lower().split())
            ref_words = set(ref.lower().split())
            if ref_words:
                coverage = len(pred_words.intersection(ref_words)) / len(ref_words)
                coverage_scores.append(coverage)

        if coverage_scores:
            custom_metrics["term_coverage"] = np.mean(coverage_scores)

        # Check for key log summarization elements
        structure_scores = []
        for pred in predictions:
            score = 0
            pred_lower = pred.lower()

            # Check for structured elements
            if "system:" in pred_lower or "operation:" in pred_lower:
                score += 0.2
            if "status:" in pred_lower:
                score += 0.2
            if "duration:" in pred_lower or "time:" in pred_lower:
                score += 0.2
            if "summary:" in pred_lower:
                score += 0.2
            if "issues:" in pred_lower or "errors:" in pred_lower:
                score += 0.2

            structure_scores.append(score)

        custom_metrics["structure_score"] = np.mean(structure_scores)

        return custom_metrics

    def evaluate_dataset(
        self, dataset_path: str, output_file: str = None, sample_size: int = None
    ) -> dict:
        """
        Evaluate the model on a dataset.

        Args:
            dataset_path: Path to JSONL dataset file
            output_file: Optional path to save detailed results
            sample_size: Number of samples to evaluate (None for all)

        Returns:
            Dictionary containing evaluation metrics
        """
        logger.info(f"Evaluating on dataset: {dataset_path}")

        # Load dataset
        dataset = self._load_dataset(dataset_path)

        if sample_size and sample_size < len(dataset):
            dataset = dataset[:sample_size]
            logger.info(f"Using sample of {sample_size} examples")

        predictions = []
        references = []
        detailed_results = []

        # Generate predictions
        for i, entry in enumerate(tqdm(dataset, desc="Generating summaries")):
            try:
                log_input = entry.get("input", "")
                reference = entry.get("output", "")

                if not log_input or not reference:
                    logger.warning(f"Skipping entry {i}: missing input or output")
                    continue

                # Generate prediction
                prediction = self.generate_summary(log_input)

                predictions.append(prediction)
                references.append(reference)

                # Store detailed result
                detailed_results.append(
                    {
                        "id": i,
                        "input": (
                            log_input[:200] + "..."
                            if len(log_input) > 200
                            else log_input
                        ),
                        "reference": reference,
                        "prediction": prediction,
                    }
                )

            except Exception as e:
                logger.error(f"Error processing entry {i}: {e}")
                continue

        # Compute metrics
        metrics = self.compute_metrics(predictions, references)
        metrics["num_samples"] = len(predictions)

        # Save detailed results if requested
        if output_file:
            self._save_detailed_results(detailed_results, metrics, output_file)

        logger.info("Evaluation completed")
        return metrics

    def _load_dataset(self, dataset_path: str) -> list:
        """Load dataset from JSONL file."""
        dataset = []
        try:
            with open(dataset_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        entry = json.loads(line.strip())
                        dataset.append(entry)
                    except json.JSONDecodeError:
                        continue
            logger.info(f"Loaded {len(dataset)} entries from dataset")
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise

        return dataset

    def _save_detailed_results(
        self, detailed_results: list, metrics: dict, output_file: str
    ):
        """Save detailed evaluation results to file."""
        try:
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(output_file), exist_ok=True)

            # Save as JSON
            results = {
                "metrics": metrics,
                "detailed_results": detailed_results,
                "model_path": self.model_path,
            }

            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

            logger.info(f"Detailed results saved to: {output_file}")

            # Also save metrics as CSV for easy viewing
            csv_file = output_file.replace(".json", "_metrics.csv")
            metrics_df = pd.DataFrame([metrics])
            metrics_df.to_csv(csv_file, index=False)
            logger.info(f"Metrics saved to: {csv_file}")

        except Exception as e:
            logger.error(f"Error saving results: {e}")


def compare_models(model_paths: list, dataset_path: str, output_dir: str):
    """
    Compare multiple models on the same dataset.

    Args:
        model_paths: List of paths to fine-tuned models
        dataset_path: Path to evaluation dataset
        output_dir: Directory to save comparison results
    """
    logger.info(f"Comparing {len(model_paths)} models")

    os.makedirs(output_dir, exist_ok=True)

    all_results = {}

    for model_path in model_paths:
        model_name = os.path.basename(model_path)
        logger.info(f"Evaluating model: {model_name}")

        try:
            evaluator = ModelEvaluator(model_path)
            metrics = evaluator.evaluate_dataset(
                dataset_path,
                output_file=os.path.join(
                    output_dir, f"{model_name}_detailed_results.json"
                ),
            )
            all_results[model_name] = metrics

        except Exception as e:
            logger.error(f"Error evaluating model {model_name}: {e}")
            continue

    # Save comparison results
    comparison_file = os.path.join(output_dir, "model_comparison.json")
    with open(comparison_file, "w") as f:
        json.dump(all_results, f, indent=2)

    # Create comparison DataFrame
    comparison_df = pd.DataFrame(all_results).T
    comparison_csv = os.path.join(output_dir, "model_comparison.csv")
    comparison_df.to_csv(comparison_csv)

    logger.info(f"Model comparison saved to: {comparison_file}")
    logger.info(f"Comparison CSV saved to: {comparison_csv}")

    # Print summary
    print("\n" + "=" * 50)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 50)

    key_metrics = ["bleu", "meteor", "rouge1", "rouge2", "rougeL", "structure_score"]
    for metric in key_metrics:
        if metric in comparison_df.columns:
            print(f"\n{metric.upper()}:")
            sorted_models = comparison_df[metric].sort_values(ascending=False)
            for model, score in sorted_models.items():
                print(f"  {model}: {score:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate fine-tuned log summarization models"
    )
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to fine-tuned LoRA model"
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to evaluation dataset (JSONL format)",
    )
    parser.add_argument(
        "--output_file", type=str, help="Output file for detailed results"
    )
    parser.add_argument(
        "--base_model", type=str, help="Base model name (if different from config)"
    )
    parser.add_argument("--sample_size", type=int, help="Number of samples to evaluate")
    parser.add_argument(
        "--compare_models", nargs="+", help="List of model paths to compare"
    )
    parser.add_argument(
        "--comparison_output_dir",
        type=str,
        default="./evaluation_results",
        help="Directory for comparison results",
    )

    args = parser.parse_args()

    if args.compare_models:
        # Compare multiple models
        compare_models(
            args.compare_models, args.dataset_path, args.comparison_output_dir
        )
    else:
        # Evaluate single model
        evaluator = ModelEvaluator(args.model_path, args.base_model)

        metrics = evaluator.evaluate_dataset(
            args.dataset_path, args.output_file, args.sample_size
        )

        # Print results
        print("\n" + "=" * 50)
        print("EVALUATION RESULTS")
        print("=" * 50)

        for metric, value in metrics.items():
            if isinstance(value, float):
                print(f"{metric}: {value:.4f}")
            else:
                print(f"{metric}: {value}")


if __name__ == "__main__":
    main()
