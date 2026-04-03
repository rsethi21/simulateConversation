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
        self.device = self.model.device # Use the model's device
    
    def set_decay_rate(self, decay_rate: float):
        """Set the exponential decay rate for steering intensity (0.0-1.0)."""
        if 0.0 <= decay_rate <= 1.0:
            self.decay_rate = decay_rate
        else:
            raise ValueError("decay_rate must be between 0.0 and 1.0")
    
    def _calculate_decay(self, token_position: int) -> float:
        """Calculate decay factor at given token position."""
        # Start decay after the prompt tokens, so token_position is relative to generated tokens
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
        self._remove_steering_hooks()  # Clean up old hooks before setting a new vector
        self.steering_vector = vector
        if vector is not None:
            self._register_steering_hooks()
    
    def _register_steering_hooks(self):
        """Register forward hooks to apply steering vectors for each target layer."""
        if self.steering_vector is None:
            return
        
        if self.steering_vector.layer_activations and \
           all(isinstance(k, str) for k in self.steering_vector.layer_activations.keys()):
            # If keys are already module names, we can use them directly
            named_modules = dict(self.model.named_modules())
            for index, vector in self.steering_vector.layer_activations.items():
                name = f"model.layers.{index}"
                if name in named_modules.keys():
                    module = named_modules.get(name)
                    if module is not None:
                        def steering_hook_factory(layer_name, steering_tensor):
                            def steering_hook(module, input, output):
                                # The steering vector apply method often expects an output tuple
                                # if it's a multi-output module, but generally just modifies the tensor.
                                # Ensure output is a tensor or the first element of a tuple.
                                original_output_tensor = output[0] if isinstance(output, tuple) else output
                                
                                decay_factor = self._calculate_decay(self.token_index)
                                adjusted_intensity = self.steering_vector.intensity * decay_factor
                                
                                # Ensure steering_tensor has the correct shape for broadcasting or addition
                                # You might need to expand dimensions of steering_tensor if it's 1D
                                # or reshape it to match original_output_tensor.shape[1:] if it's per-batch.
                                
                                # Simple addition, assuming shapes are compatible
                                steered_output = original_output_tensor + adjusted_intensity * steering_tensor.to(self.device)
                                
                                if isinstance(output, tuple):
                                    return (steered_output,) + output[1:]
                                return steered_output
                            return steering_hook

                    handle = module.register_forward_hook(
                        steering_hook_factory(name, vector)
                    )
                    self.hook_handles.append(handle)
        else:
            # If your CustomSteeringVector.layer_activations uses integer keys (e.g., {0: tensor1, 1: tensor2})
            # You need a mapping from your integer index to actual module names.
            # This is a placeholder and MUST BE CUSTOMIZED.
            print("Warning: CustomSteeringVector.layer_activations keys are not strings (module names).")
            print("Please customize _register_steering_hooks to map your numerical layer indices to model module names.")
            # Example for a Llama-like model where layer_activations is {0: vec_for_layer0, 1: vec_for_layer1, ...}
            pass


    def _remove_steering_hooks(self):
        """Remove all registered hooks."""
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles = []
    
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
        """Stream tokens one at a time with decay applied via registered hooks."""
        full_prompt = self._format_prompt(prompt)
        inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.model.device)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        input_len = input_ids.shape[1]
        
        self.token_index = 0  # Reset decay tracking for generated tokens

        with torch.no_grad():
            past_key_values = None
            generated_tokens_count = 0
            
            while generated_tokens_count < max_tokens:
                if past_key_values:
                    # For subsequent tokens, only pass the last generated token
                    current_input_ids = input_ids[:, -1].unsqueeze(-1)
                    current_attention_mask = torch.ones(current_input_ids.shape, dtype=torch.long, device=self.device)
                else:
                    # For the first token, pass the full prompt
                    current_input_ids = input_ids
                    current_attention_mask = attention_mask

                outputs = self.model(
                    input_ids=current_input_ids,
                    attention_mask=current_attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                
                logits = outputs.logits[:, -1, :] # Get logits for the last token
                past_key_values = outputs.past_key_values

                # Apply sampling
                if temperature == 0:
                    next_token_logits = logits
                else:
                    next_token_logits = logits / temperature
                    
                # Apply top_k sampling
                if top_k > 0:
                    top_k_values, top_k_indices = torch.topk(next_token_logits, top_k)
                    next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                    next_token_logits.scatter_(1, top_k_indices, top_k_values)
                
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # If it's the first token, combine with prompt for next iteration's attention mask
                if generated_tokens_count == 0:
                    input_ids = torch.cat([input_ids, next_token], dim=-1)
                    attention_mask = torch.cat([attention_mask, current_attention_mask], dim=-1)
                else:
                    input_ids = torch.cat([input_ids, next_token], dim=-1)
                
                # Check for EOS token
                if next_token.item() == self.tokenizer.eos_token_id:
                    break

                token_str = self.tokenizer.decode(next_token[0], skip_special_tokens=True)
                yield token_str
                
                self.token_index += 1  # Increment for each generated token, allowing decay in hooks
                generated_tokens_count += 1
            
    # def cleanup(self):
    #     """Clean up resources."""
    #     self._remove_steering_hooks()
    #     if hasattr(self.model, 'cpu'):
    #         self.model.cpu()
    #     del self.model
    #     del self.tokenizer