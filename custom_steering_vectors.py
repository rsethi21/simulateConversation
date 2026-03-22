import json
import numpy as np
from pathlib import Path
from typing import Optional, Dict
import torch
import pdb
from steering_vectors import SteeringVector  # Import the base class for type hinting

class CustomSteeringVector:
    def __init__(self, vector: SteeringVector, intensity: float = 1.0):
        self.vector = vector
        self.intensity = intensity
        self.device = 0
    
    @staticmethod
    def load_from_pt(file_path: str) -> "CustomSteeringVector":
        """Load steering vector from PyTorch .pt file with multi-layer activations."""
        data = torch.load(file_path, map_location='cpu', weights_only=False)  # Load on CPU to avoid device issues
        return CustomSteeringVector(data)

        layer_activations = {}
        for layer, tensor in data.layer_activations.items():
            layer_activations[layer] = tensor.float().numpy()
        
        return CustomSteeringVector(layer_activations)  # No intensity loaded
    
    def save_to_pt(self, file_path: str):
        """Save steering vector to PyTorch .pt file."""
        data = {
            'layer_activations': {layer: torch.from_numpy(vector) for layer, vector in self.layer_activations.items()}
        }  # No intensity saved
        torch.save(data, file_path)
    
    def set_intensity(self, intensity: float):
        """Update the intensity of the steering vector."""
        self.intensity = intensity
    
    def apply(self, activations: torch.Tensor, layer: int) -> torch.Tensor:
        """Apply steering vector to model activations for a specific layer."""
        if activations is None or layer not in self.layer_activations:
            return activations
        
        vector_tensor = torch.from_numpy(self.layer_activations[layer]).to(self.device)
        # Scale the vector by intensity and any additional scaling
        scaled_vector = vector_tensor * self.intensity
        
        # Add the scaled vector to activations
        if type(activations) == tuple:
            temporary_activations = activations[0] + scaled_vector
            activations = (temporary_activations,) + activations[1:]
        elif activations.shape == vector_tensor.shape:
            activations = activations + scaled_vector
        else:
            # Handle dimension mismatch gracefully
            min_dim = min(activations.shape[-1], vector_tensor.shape[-1])
            activations[..., :min_dim] = activations[..., :min_dim] + scaled_vector[:min_dim]
        
        return activations

class SteeringVectorManager:
    def __init__(self, vector_dir: str = "./vectors", default_intensity: float = 1.0):
        self.vector_dir = Path(vector_dir)
        self.vector_dir.mkdir(parents=True, exist_ok=True)
        self.vectors = {}
        self.default_intensity = default_intensity
    
    def load_vector(self, name: str, file_path: str) -> CustomSteeringVector:
        """Load a steering vector from .pt file."""
        vector = CustomSteeringVector.load_from_pt(file_path)
        vector.set_intensity(self.default_intensity)  # Set from config
        self.vectors[name] = vector
        return vector
    
    def preload_vectors(self, vector_a_path: Optional[str], vector_b_path: Optional[str]):
        """Preload vectors for Model A and Model B on launch, skipping if path is None."""
        if vector_a_path:
            self.load_vector('vector_a', vector_a_path)
        if vector_b_path:
            self.load_vector('vector_b', vector_b_path)
    
    def get_vector(self, name: str) -> Optional[CustomSteeringVector]:
        """Retrieve a loaded vector."""
        return self.vectors.get(name)