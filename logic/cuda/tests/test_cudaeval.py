import os
import pytest
import torch
import numpy as np
from unittest.mock import patch, Mock, MagicMock
from logic.cuda.cudaeval import (
    evaluate_and_compare,
    generate_response,
    load_test_data,
    plot_evaluation_scores
)

def test_generate_response(mock_model, mock_tokenizer):
    """Test response generation functionality."""
    prompt = "Test prompt"
    expected_response = "Test response"
    
    # Create mock tensors that have .to() method
    mock_input_ids = torch.tensor([1, 2, 3])
    mock_attention_mask = torch.tensor([1, 1, 1])
    
    # Setup mock tokenizer to return dict with tensors that have .to() method
    mock_tokenized_output = {
        "input_ids": mock_input_ids,
        "attention_mask": mock_attention_mask
    }
    mock_tokenizer.return_value = mock_tokenized_output
    
    # Setup mock model
    mock_model.device = torch.device("cpu")
    mock_model.generate.return_value = torch.tensor([1, 2, 3, 4])
    mock_tokenizer.decode.return_value = prompt + expected_response
    mock_tokenizer.eos_token_id = 2
    
    # Test response generation
    response = generate_response(mock_model, mock_tokenizer, prompt)
    assert response == expected_response
    
    # Verify tokenizer and model calls
    mock_tokenizer.assert_called_once_with(prompt, return_tensors="pt")
    mock_model.generate.assert_called_once()
    mock_tokenizer.decode.assert_called_once()

def test_load_test_data(sample_jsonl_file):
    """Test loading test data from JSONL file."""
    data = load_test_data(sample_jsonl_file)
    assert len(data) == 2
    assert all(isinstance(item, dict) for item in data)
    assert all("instruction" in item for item in data)
    assert all("input" in item for item in data)
    assert all("output" in item for item in data)

def test_plot_evaluation_scores(temp_dir):
    """Test evaluation score plotting functionality."""
    base_scores = {
        "bleu": 0.8,
        "meteor": 0.7,
        "rouge1": 0.75,
        "rouge2": 0.65,
        "rougeL": 0.7,
        "ttr": 0.6,
        "readability": 0.85,
        "bert_f1": 0.75
    }
    
    finetuned_scores = {
        "bleu": 0.85,
        "meteor": 0.75,
        "rouge1": 0.8,
        "rouge2": 0.7,
        "rougeL": 0.75,
        "ttr": 0.65,
        "readability": 0.9,
        "bert_f1": 0.8
    }
    
    output_dir = temp_dir
    plot_evaluation_scores(base_scores, finetuned_scores, output_dir)
    
    # Verify plot directory and files
    plots_dir = os.path.join(output_dir, "evaluation_plots")
    assert os.path.exists(plots_dir)
    for metric in base_scores.keys():
        plot_file = os.path.join(plots_dir, f"{metric}_comparison.png")
        assert os.path.exists(plot_file)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_evaluate_and_compare(temp_dir, sample_jsonl_file):
    """Test model evaluation and comparison functionality."""
    base_model_path = "test_base_model"
    finetuned_model_path = temp_dir  # Use temp_dir as finetuned model path
    
    with patch("transformers.AutoModelForCausalLM.from_pretrained") as mock_model_class, \
         patch("transformers.AutoTokenizer.from_pretrained") as mock_tokenizer_class, \
         patch("bert_score.score") as mock_bert_score, \
         patch("nltk.translate.bleu_score.sentence_bleu") as mock_bleu, \
         patch("nltk.translate.meteor_score.single_meteor_score") as mock_meteor, \
         patch("rouge_score.rouge_scorer.RougeScorer") as mock_rouge_class, \
         patch("nltk.word_tokenize") as mock_word_tokenize, \
         patch("textstat.flesch_reading_ease") as mock_flesch:
        
        # Setup mock models and tokenizers
        mock_model = Mock()
        mock_model.device = torch.device("cpu")
        mock_model.generate.return_value = torch.tensor([1, 2, 3, 4])
        mock_model_class.return_value = mock_model
        
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "</s>"
        mock_tokenizer.eos_token_id = 2
        mock_tokenizer.return_value = {
            "input_ids": torch.tensor([1, 2, 3]),
            "attention_mask": torch.tensor([1, 1, 1])
        }
        mock_tokenizer.decode.return_value = "Test prompt Test response"
        mock_tokenizer_class.return_value = mock_tokenizer
        
        # Setup NLTK mock
        mock_word_tokenize.return_value = ["test", "response"]
        
        # Setup metric mocks
        mock_bert_score.return_value = (
            torch.tensor([0.8]), torch.tensor([0.8]), torch.tensor([0.8])
        )
        mock_bleu.return_value = 0.8
        mock_meteor.return_value = 0.7
        mock_flesch.return_value = 80.0
        
        # Setup rouge scorer mock
        mock_rouge_instance = Mock()
        mock_rouge_class.return_value = mock_rouge_instance
        mock_rouge_instance.score.return_value = {
            'rouge1': Mock(fmeasure=0.75),
            'rouge2': Mock(fmeasure=0.65),
            'rougeL': Mock(fmeasure=0.7)
        }
        
        # Run evaluation
        evaluate_and_compare(
            base_model_path=base_model_path,
            finetuned_model_path=finetuned_model_path,
            test_data_path=sample_jsonl_file,
            num_samples=2
        )
        
        # Verify outputs
        plots_dir = os.path.join(finetuned_model_path, "evaluation_plots")
        assert os.path.exists(plots_dir)
        
        # Verify mock calls
        assert mock_model_class.call_count == 2  # Called for both base and finetuned models
        assert mock_tokenizer_class.call_count == 2  # Called for both tokenizers
        mock_bert_score.assert_called()
        mock_bleu.assert_called()
        mock_meteor.assert_called()
        mock_rouge_class.assert_called()

def test_evaluate_and_compare_error_cases(temp_dir):
    """Test error handling in model evaluation."""
    # Test with nonexistent model paths
    with pytest.raises((Exception, OSError, ValueError)):
        evaluate_and_compare(
            base_model_path="nonexistent_model",
            finetuned_model_path="nonexistent_model", 
            test_data_path="nonexistent.jsonl",
            num_samples=2
        )

def test_plot_evaluation_scores_error_cases(temp_dir):
    """Test error handling in score plotting."""
    # Test with empty scores - should raise ValueError or KeyError
    with pytest.raises((ValueError, KeyError)):
        plot_evaluation_scores({}, {}, temp_dir)
    
    # Test with mismatched metrics - should raise KeyError  
    base_scores = {"bleu": 0.8}
    finetuned_scores = {"meteor": 0.7}
    with pytest.raises(KeyError):
        plot_evaluation_scores(base_scores, finetuned_scores, temp_dir) 