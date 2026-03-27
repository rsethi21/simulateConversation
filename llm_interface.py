import pdb
from typing import Optional, Generator
from model_manager import ModelManager
from custom_steering_vectors import CustomSteeringVector
import torch
from steering_vectors import SteeringVector

class LLMInterface:
    def __init__(self, model_name: str, model_display_name: str, manager: ModelManager):
        self.model_name = model_name
        self.model_display_name = model_display_name
        self.manager = manager
        self.model_data = manager.load_model(model_name)
        self.model = self.model_data["model"]
        self.tokenizer = self.model_data["tokenizer"]
        self.system_prompt = ""
        self.steering_vector: Optional[CustomSteeringVector] = None
        self.hook_handles = []
        self.token_index = 0  # Track position in generation
        self.decay_rate = 0.95  # Exponential decay factor (0.0-1.0)
        self.device = 0
    
    def set_decay_rate(self, decay_rate: float):
        """Set the exponential decay rate for steering intensity (0.0-1.0)."""
        if 0.0 <= decay_rate <= 1.0:
            self.decay_rate = decay_rate
        else:
            raise ValueError("decay_rate must be between 0.0 and 1.0")
    
    def _calculate_decay(self, token_position: int) -> float:
        """Calculate decay factor at given token position."""
        return self.decay_rate ** token_position
    
    def set_personality(self, personality: str):
        """Set the system prompt (personality) for this model."""
        self.system_prompt = personality
    
    def update_personality(self, personality: str):
        """Update personality on-the-fly without reloading models."""
        self.set_personality(personality)
    
    def update_steering_intensity(self, intensity: float):
        """Update the intensity of the steering vector on-the-fly."""
        if self.steering_vector is not None:
            self.steering_vector.set_intensity(intensity)

    def set_steering_vector(self, vector: Optional[CustomSteeringVector]):
        """Set steering vector for this model and register hooks."""
        # self._remove_steering_hooks()  # Clean up old hooks
        self.steering_vector = vector
        # if vector is not None:
        #     self._register_steering_hooks()
    
    # def _register_steering_hooks(self):
    #     """Register forward hooks to apply steering vectors for each layer."""
    #     if self.steering_vector is None:
    #         return
        
    #     # Register hooks for each layer in the steering vector
    #     for target_layer in self.steering_vector.layer_activations.keys():
    #         layer_count = 0
            
    #         for name, module in self.model.named_modules():
    #             if layer_count == target_layer:
    #                 def steering_hook(module, input, output, layer=target_layer, vector=self.steering_vector):
    #                     if vector is not None:
    #                         decay_factor = self._calculate_decay(self.token_index)
    #                         adjusted_scale = vector.intensity * decay_factor
    #                         vector.set_intensity(adjusted_scale)
    #                         print(name)
    #                         return vector.apply(output, layer)
    #                     return output
                    
    #                 handle = module.register_forward_hook(steering_hook)
    #                 self.hook_handles.append(handle)
    #                 break
                
    #             layer_count += 1
    
    # def _remove_steering_hooks(self):
    #     """Remove all registered hooks."""
    #     for handle in self.hook_handles:
    #         handle.remove()
    #     self.hook_handles = []
    
    def _format_prompt(self, user_message: str) -> str:
        """Format conversation using chat template if available."""
        messages = []
        
        # Add system prompt if present
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        
        # Add user message
        messages.append({"role": "user", "content": user_message})
        
        # Use apply_chat_template if available, otherwise fallback to manual format
        if hasattr(self.tokenizer, 'apply_chat_template'):
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            # Fallback manual format
            prompt = ""
            if self.system_prompt:
                prompt += f"System: {self.system_prompt}\n"
            prompt += f"User: {user_message}\nAssistant:"
            return prompt
    
    def generate(self, prompt: str, temperature: float = 0.7, 
                 top_k: int = 50, max_tokens: int = 256) -> str:
        """Generate a response with optional steering applied via registered hooks."""
        full_prompt = self._format_prompt(prompt)
        inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.model.device)
        self.token_index = 0  # Reset decay tracking
        
        # Hooks are automatically invoked during forward pass within generate()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            self.token_index += max_tokens  # Update after generation
        
        response = self.tokenizer.decode(outputs[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
        return response
    
    def generate_stream(self, prompt: str, temperature: float = 0.7,
                       top_k: int = 50, max_tokens: int = 256) -> Generator[str, None, None]:
        """Stream tokens one at a time with decay applied."""
        full_prompt = self._format_prompt(prompt)
        inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.model.device)
        self.token_index = 0  # Reset decay tracking
        # attention_mask = inputs["attention_mask"]
        with torch.no_grad():
            # for i in range(max_tokens):
            with self.steering_vector.vector.apply(self.model, multiplier=self.steering_vector.intensity):
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
        self.token_index += max_tokens  # Update after generation
        response = self.tokenizer.decode(outputs[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
        return response
            # self.token_index += 1  # Increment for each token
            # decay_factor = self._calculate_decay(self.token_index)
            # adjusted_scale = self.steering_vector.intensity * decay_factor
            # self.steering_vector.set_intensity(adjusted_scale)
            # new_token = outputs[0, -1].item()
            # token_str = self.tokenizer.decode([new_token])
            # inputs = outputs
            #     yield token_str
    
    # def cleanup(self):
    #     """Clean up resources."""
    #     self._remove_steering_hooks()
    #     if hasattr(self.model, 'cpu'):
    #         self.model.cpu()
    #     del self.model
    #     del self.tokenizer