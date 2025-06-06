import os
import json
import pytest
import torch
import tempfile
from pathlib import Path
from unittest.mock import Mock

@pytest.fixture(scope="session")
def cuda_available():
    """Check if CUDA is available and skip tests if not."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

@pytest.fixture(scope="session")
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir

@pytest.fixture(scope="session")
def sample_dataset():
    """Create a sample dataset for testing."""
    return [
        {
            "instruction": "Translate to French",
            "input": "Hello world",
            "output": "Bonjour le monde"
        },
        {
            "instruction": "Summarize",
            "input": "This is a long text",
            "output": "Short summary"
        }
    ]

@pytest.fixture(scope="session")
def sample_jsonl_file(temp_dir, sample_dataset):
    """Create a sample JSONL file for testing."""
    file_path = os.path.join(temp_dir, "test.jsonl")
    with open(file_path, "w", encoding="utf-8") as f:
        for item in sample_dataset:
            f.write(json.dumps(item) + "\n")
    return file_path

@pytest.fixture(scope="session")
def mock_model():
    """Create a mock model for testing."""
    model = Mock()
    model.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.generate.return_value = torch.tensor([1, 2, 3, 4])
    model.save_pretrained = Mock()
    model.merge_and_unload = Mock(return_value=model)
    model.print_trainable_parameters = Mock()
    return model

@pytest.fixture(scope="session")
def mock_tokenizer():
    """Create a mock tokenizer for testing."""
    tokenizer = Mock()
    tokenizer.pad_token = "<pad>"
    tokenizer.eos_token = "</s>"
    tokenizer.eos_token_id = 2
    tokenizer.save_pretrained = Mock()
    tokenizer.decode = Mock(return_value="Test response")
    
    # Make the tokenizer callable and return proper format
    def mock_tokenize(text, return_tensors=None, truncation=False, max_length=None, padding=None):
        return {
            "input_ids": torch.tensor([1, 2, 3]) if return_tensors == "pt" else [1, 2, 3],
            "attention_mask": torch.tensor([1, 1, 1]) if return_tensors == "pt" else [1, 1, 1]
        }
    
    tokenizer.side_effect = mock_tokenize
    tokenizer.__call__ = mock_tokenize
    return tokenizer

@pytest.fixture(scope="session")
def mock_trainer():
    """Create a mock trainer for testing."""
    trainer = Mock()
    trainer.state.global_step = 0
    trainer.train = Mock()
    trainer.add_callback = Mock()
    return trainer

@pytest.fixture(scope="session")
def mock_plot_metrics_callback():
    """Create a mock PlotMetricsCallback for testing."""
    callback = Mock()
    callback.metrics = {
        "loss": [],
        "step": [],
        "eval_loss": [],
        "eval_step": [],
        "learning_rate": []
    }
    return callback

def create_mock_model_output():
    """Helper function to create mock model outputs."""
    return {
        "input_ids": torch.tensor([1, 2, 3]),
        "attention_mask": torch.tensor([1, 1, 1]),
        "labels": torch.tensor([1, 2, 3])
    }

def create_mock_training_args():
    """Helper function to create mock training arguments."""
    return Mock(
        output_dir="test_output",
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        num_train_epochs=1,
        learning_rate=2e-4,
        logging_steps=10,
        eval_steps=10,
        save_steps=50,
        bf16=True,
        optim="adamw_torch",
        gradient_accumulation_steps=4,
        gradient_checkpointing=False,
        dataloader_num_workers=0
    ) 