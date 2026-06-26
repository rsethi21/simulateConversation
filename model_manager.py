import os
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class ModelManager:
    def __init__(self, cache_dir: str = "./models"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.loaded_models = {}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def load_model(self, model_name: str, token: str = None):
        """Load model and tokenizer from Hugging Face with local caching."""
        if token:
            tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=str(self.cache_dir),
            clean_up_tokenization_spaces=False,
            token=token
            )
            tokenizer.pad_token = tokenizer.eos_token  # Ensure pad token is set
            tokenizer.padding_side = "left"  # Ensure padding is on the right
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                cache_dir=str(self.cache_dir),
                device_map="auto",
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                token=token
            )
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=str(self.cache_dir),
                clean_up_tokenization_spaces=False
            )
            tokenizer.pad_token = tokenizer.eos_token  # Ensure pad token is set
            tokenizer.padding_side = "left"  # Ensure padding is on the right
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                cache_dir=str(self.cache_dir),
                device_map="auto",
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
        return {"model": model, "tokenizer": tokenizer}
    
    def unload_model(self, model_name: str):
        """Free memory by unloading a model."""
        if model_name in self.loaded_models:
            del self.loaded_models[model_name]