import os
import pytest
import torch
from unittest.mock import patch, Mock
from logic.cuda.cudamerge import merge_and_save

def test_merge_and_save(temp_dir, mock_model, mock_tokenizer):
    """Test model merging and saving functionality."""
    base_model_path = "test_base_model"
    lora_adapter_path = "test_lora"
    output_dir = os.path.join(temp_dir, "merged_model")
    
    with patch("transformers.AutoModelForCausalLM.from_pretrained", return_value=mock_model) as mock_load_model, \
         patch("peft.PeftModel.from_pretrained", return_value=mock_model) as mock_load_peft, \
         patch("transformers.AutoTokenizer.from_pretrained", return_value=mock_tokenizer) as mock_load_tokenizer:
        
        # Setup mock behavior
        mock_model.merge_and_unload.return_value = mock_model
        
        # Run merge and save
        merge_and_save(base_model_path, lora_adapter_path, output_dir)
        
        # Verify outputs
        assert os.path.exists(output_dir)
        mock_model.save_pretrained.assert_called_once_with(output_dir)
        mock_tokenizer.save_pretrained.assert_called_once_with(output_dir)
        
        # Verify model loading calls
        mock_load_model.assert_called_once_with(
            base_model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        mock_load_peft.assert_called_once_with(mock_model, lora_adapter_path)
        mock_load_tokenizer.assert_called_once_with(base_model_path)
        
        # Verify merge operation
        mock_model.merge_and_unload.assert_called_once()

def test_merge_and_save_error_cases(temp_dir):
    """Test error handling in model merging."""
    # Test with nonexistent model paths - should raise OSError or ValueError from transformers
    with pytest.raises((OSError, ValueError, Exception)):
        merge_and_save(
            base_model_path="nonexistent_model",
            lora_adapter_path="nonexistent_lora",
            output_dir=temp_dir
        )

def test_merge_and_save_cuda_availability(temp_dir):
    """Test CUDA availability handling."""
    with patch("transformers.AutoModelForCausalLM.from_pretrained") as mock_load_model, \
         patch("transformers.AutoTokenizer.from_pretrained") as mock_load_tokenizer, \
         patch("peft.PeftModel.from_pretrained") as mock_load_peft:
        
        # Setup mocks
        mock_model = Mock()
        mock_model.merge_and_unload.return_value = mock_model
        mock_load_model.return_value = mock_model
        mock_load_peft.return_value = mock_model
        
        mock_tokenizer = Mock()
        mock_load_tokenizer.return_value = mock_tokenizer
        
        # Test should work regardless of CUDA availability since cudamerge doesn't explicitly check for CUDA
        merge_and_save(
            base_model_path="test_model",
            lora_adapter_path="test_lora",
            output_dir=temp_dir
        )
        
        # Verify the function completed successfully
        assert mock_load_model.called
        assert mock_load_peft.called
        assert mock_load_tokenizer.called

def test_merge_and_save_model_loading(temp_dir):
    """Test model loading behavior."""
    with patch("transformers.AutoModelForCausalLM.from_pretrained") as mock_load_model:
        # Test model loading failure
        mock_load_model.side_effect = Exception("Model loading failed")
        with pytest.raises(Exception):
            merge_and_save(
                base_model_path="test_model",
                lora_adapter_path="test_lora",
                output_dir=temp_dir
            )

def test_merge_and_save_tokenizer_handling(temp_dir, mock_model):
    """Test tokenizer handling during merge and save."""
    with patch("transformers.AutoModelForCausalLM.from_pretrained", return_value=mock_model), \
         patch("peft.PeftModel.from_pretrained", return_value=mock_model), \
         patch("transformers.AutoTokenizer.from_pretrained") as mock_load_tokenizer:
        
        # Test tokenizer loading failure
        mock_load_tokenizer.side_effect = Exception("Tokenizer loading failed")
        with pytest.raises(Exception):
            merge_and_save(
                base_model_path="test_model",
                lora_adapter_path="test_lora",
                output_dir=temp_dir
            )

def test_merge_and_save_output_handling(temp_dir, mock_model, mock_tokenizer):
    """Test output directory handling during merge and save."""
    with patch("transformers.AutoModelForCausalLM.from_pretrained", return_value=mock_model), \
         patch("peft.PeftModel.from_pretrained", return_value=mock_model), \
         patch("transformers.AutoTokenizer.from_pretrained", return_value=mock_tokenizer):
        
        # Test with existing output directory
        os.makedirs(temp_dir, exist_ok=True)
        merge_and_save(
            base_model_path="test_model",
            lora_adapter_path="test_lora",
            output_dir=temp_dir
        )
        assert os.path.exists(temp_dir)
        
        # Test with non-existent output directory
        new_output_dir = os.path.join(temp_dir, "new_dir")
        merge_and_save(
            base_model_path="test_model",
            lora_adapter_path="test_lora",
            output_dir=new_output_dir
        )
        assert os.path.exists(new_output_dir) 