import os
import pytest
import torch
from unittest.mock import patch, Mock
from logic.cuda.cudafft import (
    train_model,
    CustomDataset,
    get_lora_config,
    PlotMetricsCallback,
    load_jsonl_dataset
)

def test_get_lora_config():
    """Test LoRA configuration selection for different models."""
    # Test default config
    config = get_lora_config("unknown_model")
    assert config.r == 8
    assert config.lora_alpha == 32
    assert config.lora_dropout == 0.1
    assert config.bias == "none"
    assert config.task_type == "CAUSAL_LM"
    
    # Test specific model configs
    model_configs = {
        "tinyllama": {"r": 8, "lora_alpha": 32},
        "h2o-danube": {"r": 12, "lora_alpha": 64},
        "fox": {"r": 8, "lora_alpha": 32},
        "bitnet": {"r": 8, "lora_alpha": 32},
        "smol": {"r": 8, "lora_alpha": 32},
        "deepseek": {"r": 8, "lora_alpha": 32}
    }
    
    for model_name, expected_config in model_configs.items():
        config = get_lora_config(model_name)
        assert config.r == expected_config["r"]
        assert config.lora_alpha == expected_config["lora_alpha"]

def test_custom_dataset(sample_dataset, mock_tokenizer):
    """Test CustomDataset class functionality."""
    max_length = 128
    
    # Setup mock tokenizer to return all required keys
    mock_tokenizer.return_value = {
        "input_ids": [1, 2, 3],
        "attention_mask": [1, 1, 1]
    }
    
    dataset = CustomDataset(sample_dataset, mock_tokenizer, max_length)
    
    # Test dataset length
    assert len(dataset) == len(sample_dataset)
    
    # Test __getitem__
    item = dataset[0]
    assert isinstance(item, dict)
    assert "input_ids" in item
    assert "attention_mask" in item
    
    # Test tokenizer calls
    mock_tokenizer.assert_called()
    call_args = mock_tokenizer.call_args[1]
    assert call_args["truncation"] is True
    assert call_args["max_length"] == max_length
    assert call_args["padding"] == "max_length"

def test_load_jsonl_dataset(sample_jsonl_file):
    """Test loading JSONL dataset."""
    data = load_jsonl_dataset(sample_jsonl_file)
    assert len(data) == 2
    assert all(isinstance(item, dict) for item in data)
    assert all("instruction" in item for item in data)
    assert all("input" in item for item in data)
    assert all("output" in item for item in data)

def test_plot_metrics_callback(temp_dir):
    """Test PlotMetricsCallback functionality."""
    plot_dir = os.path.join(temp_dir, "plots")
    callback = PlotMetricsCallback(plot_dir)
    
    # Test initialization
    assert callback.plot_dir == plot_dir
    assert isinstance(callback.metrics, dict)
    assert all(key in callback.metrics for key in ["loss", "step", "eval_loss", "eval_step", "learning_rate"])
    
    # Test on_log
    mock_args = Mock()
    mock_state = Mock()
    mock_state.global_step = 1
    mock_logs = {
        "loss": 0.5,
        "eval_loss": 0.4,
        "learning_rate": 0.001
    }
    
    callback.on_log(mock_args, mock_state, None, mock_logs)
    assert callback.metrics["loss"] == [0.5]
    assert callback.metrics["step"] == [1]
    assert callback.metrics["eval_loss"] == [0.4]
    assert callback.metrics["eval_step"] == [1]
    assert callback.metrics["learning_rate"] == [0.001]

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_train_model(temp_dir, sample_jsonl_file):
    """Test model training functionality."""
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    output_dir = os.path.join(temp_dir, "output")
    
    with patch("transformers.AutoModelForCausalLM.from_pretrained") as mock_model_class, \
         patch("transformers.AutoTokenizer.from_pretrained") as mock_tokenizer_class, \
         patch("transformers.Trainer") as mock_trainer_class, \
         patch("peft.get_peft_model") as mock_peft, \
         patch("torch.cuda.is_available", return_value=True):
        
        # Setup mock model
        mock_model = Mock()
        mock_model.print_trainable_parameters.return_value = None
        mock_model.save_pretrained.return_value = None
        mock_model_class.return_value = mock_model
        
        # Setup mock tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "</s>"
        mock_tokenizer_class.return_value = mock_tokenizer
        
        # Setup mock PEFT model
        mock_peft.return_value = mock_model
        
        # Setup mock trainer
        mock_trainer = Mock()
        mock_trainer.train.return_value = None
        mock_trainer.add_callback.return_value = None
        mock_trainer_class.return_value = mock_trainer
        
        # Run training
        train_model(
            model_name=model_name,
            dataset_path=sample_jsonl_file,
            max_length=128,
            output_dir=output_dir,
            num_epochs=1,
            batch_size=1
        )
        
        # Verify outputs
        assert os.path.exists(output_dir)
        mock_model_class.assert_called_once()
        mock_tokenizer_class.assert_called_once()
        mock_peft.assert_called_once()
        mock_trainer.train.assert_called_once()

def test_train_model_error_cases(temp_dir):
    """Test error handling in model training."""
    # Test CUDA not available
    with pytest.raises(RuntimeError):
        with patch("torch.cuda.is_available", return_value=False):
            train_model(
                model_name="test_model",
                dataset_path="nonexistent.jsonl",
                max_length=128,
                output_dir=temp_dir,
                num_epochs=1,
                batch_size=1
            )
    
    # Test invalid dataset path
    with pytest.raises((Exception, FileNotFoundError)):
        with patch("torch.cuda.is_available", return_value=True):
            train_model(
                model_name="test_model",
                dataset_path="nonexistent.jsonl",
                max_length=128,
                output_dir=temp_dir,
                num_epochs=1,
                batch_size=1
            ) 