import pdb
from typing import Optional, Generator, List
from model_manager import ModelManager
from custom_steering_vectors import CustomSteeringVector
import torch
import numpy as np
from transformers import LogitsProcessor, LogitsProcessorList # Import these

class DecayTokenIndexLogitsProcessor(LogitsProcessor):
    """
    A custom LogitsProcessor to update the LLMInterface's token_index 
    for progressive decay calculations during model.generate().
    It needs to track for each item in the batch.
    However, since ConversationManager is not doing true batching across conversations,
    this will operate on a batch of size 1 where input_ids.shape[-1] is the current sequence length.
    """
    def __init__(self, llm_interface_instance, initial_prompt_lengths: List[int]):
        self.llm_interface = llm_interface_instance
        # Store initial prompt lengths for each item in the batch.
        # If ConversationManager sends a batch of 1, this will be List[int] of size 1.
        self.initial_prompt_lengths = initial_prompt_lengths 

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # input_ids shape: (batch_size, current_sequence_length)
        # Assuming we only care about the first item in the batch for self.token_index (for single conversation flow)
        if input_ids.shape[0] > 0:
            current_sequence_length = input_ids[0].shape[-1]
            initial_prompt_length = self.initial_prompt_lengths[0] # Take the length for the first (only) item in batch
            
            generated_tokens_count = max(0, current_sequence_length - initial_prompt_length)
            
            # Update the token_index in the LLMInterface instance
            # This self.token_index will then be used by the steering_hook when it's called
            self.llm_interface.token_index = generated_tokens_count
        
        # This processor doesn't modify logits, so return them as is
        return scores

class LLMInterface:
    def __init__(self, model_name: str, model_display_name: str, manager: ModelManager, token: str = None):
        self.model_name = model_name
        self.model_display_name = model_display_name
        self.manager = manager
        self.model_data = manager.load_model(model_name, token=token)
        self.model = self.model_data["model"]
        self.tokenizer = self.model_data["tokenizer"]
        self.system_prompt = ""
        self.steering_vector: Optional[CustomSteeringVector] = None
        self.hook_handles = []
        self.token_index = 0  # Track position in generation
        self.decay_rate = 0.95  # Exponential decay factor after warmup (0.0-1.0)
        self.warmup_tokens = 0  # Number of generated tokens used for linear warmup
        self.min_intensity = 0.0  # Minimum scheduled intensity floor
        self.device = self.model.device # Use the model's device
        self.system_prompt_tokens_length = 0
    
    def set_decay_rate(self, decay_rate: float):
        """Set the exponential decay rate for steering intensity after warmup."""
        if 0.0 <= decay_rate <= 1.0:
            self.decay_rate = decay_rate
        else:
            raise ValueError("decay_rate must be between 0.0 and 1.0")

    def set_warmup_tokens(self, warmup_tokens: int):
        """Set the number of generated tokens to warm up intensity linearly."""
        if warmup_tokens < 0:
            raise ValueError("warmup_tokens must be non-negative")
        self.warmup_tokens = warmup_tokens

    def set_min_intensity(self, min_intensity: float):
        """Set the minimum intensity floor for the warmup + decay schedule."""
        self.min_intensity = min_intensity
    
    def _calculate_decay(self, token_position: int) -> float:
        """Calculate the warmup + decay intensity factor at a given generated token position."""
        if token_position < 0:
            return 1.0

        # Linear warmup over the first warmup_tokens generated tokens.
        if self.warmup_tokens > 0 and token_position < self.warmup_tokens:
            warmup_factor = float(token_position + 1) / float(self.warmup_tokens)
            return max(self.min_intensity, warmup_factor)

        # After warmup, apply exponential decay on the relative token index.
        decay_position = max(0, token_position - self.warmup_tokens)
        decay_factor = self.decay_rate ** decay_position
        return max(self.min_intensity, decay_factor)
    
    def _calculate_intensity(self, steering_tensor: torch.Tensor, original_output: torch.Tensor) -> float:
        norm_steering = torch.norm(steering_tensor)
        norm_output = torch.norm(original_output, dim=-1, keepdim=True)  # Compute norm across the last dimension for each token in the batch
        sim = 1 - torch.nn.functional.cosine_similarity(original_output, steering_tensor, dim=-1).unsqueeze(-1)  # Compute cosine similarity for each token in the batch
        try:
            base_intensity = (norm_output / norm_steering.item()) * sim
            return base_intensity
        except ZeroDivisionError:
            return torch.ones((norm_output.shape[0], norm_output.shape[1], 1))  # Default to 1.0 if steering vector is zero

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
                name = f"model.layers.{index}.{self.steering_vector.layer_to_apply}"  # Construct module name based on index and layer type
                name_2 = f"model.language_model.layers.{index}.{self.steering_vector.layer_to_apply}"  # Alternative name pattern for some models
                if name in named_modules.keys() or name_2 in named_modules.keys():
                    module = named_modules.get(name) or named_modules.get(name_2)
                    if module is not None:
                        def steering_hook_factory(layer_name, steering_vector_data):
                            # Ensure steering_vector_data is a torch.Tensor and move to device
                            # This tensor will be kept in memory with its original dtype (likely float32 if loaded from numpy)
                            if isinstance(steering_vector_data, np.ndarray):
                                base_steering_tensor = torch.from_numpy(steering_vector_data).to(self.device)
                            else:
                                base_steering_tensor = steering_vector_data.to(self.device)

                            def steering_hook(module, input, output):
                                # The steering vector apply method often expects an output tuple
                                # if it's a multi-output module, but generally just modifies the tensor.
                                # Ensure output is a tensor or the first element of a tuple.
                                original_output_tensor = output[0] if isinstance(output, tuple) else output
                                
                                # Get the dtype of the model's activation
                                target_dtype = original_output_tensor.dtype
                                
                                # decay_factor = self._calculate_decay(self.token_index) # self.token_index will be updated by LogitsProcessor
                                adjusted_intensity = self._calculate_intensity(base_steering_tensor, original_output_tensor)
                                # self.steering_vector.intensity * decay_factor
                                
                                # Cast the base_steering_tensor to the target_dtype before addition
                                # This ensures dtype consistency with the model's activations
                                steering_tensor_on_device_and_dtype = base_steering_tensor.to(target_dtype)
                                
                                # Perform the addition directly on tensors, maintaining dtype
                                steered_output = original_output_tensor.clone()
                                if steered_output.shape[1] > 1: # First pass (Prompt processing)
                                    # steered_output[:,self.system_prompt_tokens_length:,:] = original_output_tensor[:,self.system_prompt_tokens_length:,:] + adjusted_intensity * steering_tensor_on_device_and_dtype
                                    steered_output = original_output_tensor + self.steering_vector.intensity * adjusted_intensity * steering_tensor_on_device_and_dtype
                                else:
                                    steered_output = original_output_tensor + self.steering_vector.intensity * adjusted_intensity * steering_tensor_on_device_and_dtype
                                
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
    
    def _format_prompt(self, user_messages: List[str]) -> List[str]: # Updated signature
        """Format conversation using chat template if available, handling multiple messages."""
        formatted_prompts = []
        for user_message in user_messages:
            messages = []
            if self.system_prompt:
                messages.append({"role": "system", "content": self.system_prompt})
            messages.append({"role": "user", "content": user_message})
            
            if hasattr(self.tokenizer, 'apply_chat_template'):
                formatted_prompts.append(self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
            else:
                prompt = ""
                if self.system_prompt:
                    prompt += f"System: {self.system_prompt}\n"
                prompt += f"User: {user_message}\nAssistant:"
                formatted_prompts.append(prompt)
        return formatted_prompts # Always return a list of strings
    
    def generate(self, prompts: List[str], temperature: float = 0.7, # Updated signature
                 top_k: int = 50, max_tokens: int = 256, 
                 num_beams: int = 1, length_penalty: float = 1.0) -> List[str]: # Updated return type
        """Generate responses for a batch of prompts with optional steering applied via registered hooks."""
        full_formatted_prompts = self._format_prompt(prompts) # _format_prompt now returns List[str]

        # Calculate system_prompt_tokens_length based on the first prompt in the batch.
        # This assumes the system prompt portion for steering start is consistent across the batch.
        try:
            # Look for "Knowledge Base:" in the first prompt of the batch
            prompt_to_cache_index = full_formatted_prompts[0].index("Knowledge Base:")
        except ValueError:
            prompt_to_cache_index = None

        # Tokenize the batch of prompts
        # Add padding and truncation for proper batching
        inputs = self.tokenizer(full_formatted_prompts, return_tensors="pt", padding=True, truncation=True).to(self.model.device)

        if prompt_to_cache_index is not None:
            # Calculate length based on the first prompt in the batch
            self.system_prompt_tokens_length = len(self.tokenizer(full_formatted_prompts[0][:prompt_to_cache_index], add_special_tokens=True)["input_ids"])
        else:
            self.system_prompt_tokens_length = 0

        self.token_index = 0  # Reset decay tracking before generation starts

        # Instantiate the custom logits processor
        # Pass initial prompt lengths for each item in the batch.
        # inputs["input_ids"] shape is (batch_size, sequence_length), so we need lengths for each.
        initial_prompt_token_lengths = [len(ids) for ids in inputs["input_ids"]]
        decay_processor = DecayTokenIndexLogitsProcessor(self, initial_prompt_token_lengths)
        
        # Create a LogitsProcessorList and add your custom processor
        logits_processor_list = LogitsProcessorList([decay_processor])

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                num_beams=num_beams,
                length_penalty=length_penalty,
                logits_processor=logits_processor_list # Pass the custom processor here
            )
        
        # Decode the batch of outputs
        # outputs is typically (batch_size, generated_sequence_length)
        # inputs["input_ids"] is (batch_size, prompt_length)
        # We need to slice each generated sequence by its corresponding prompt length.
        responses = []
        for i in range(len(outputs)):
            # Determine the length of the actual prompt for this specific item in the batch
            # Note: For padded inputs, inputs["input_ids"][i] will be padded. We want the original non-padded length.
            # However, the tokenizer's padding makes it such that the output will include all prompt tokens up to the pad_token_id.
            # So, `inputs["input_ids"][i]` length is fine.
            prompt_len = len(inputs["input_ids"][i])
            generated_text = self.tokenizer.decode(outputs[i][prompt_len:], skip_special_tokens=True)
            responses.append(generated_text)
        
        return responses
    
    def generate_stream(self, prompt: str, temperature: float = 0.7,
                       top_k: int = 50, max_tokens: int = 256) -> Generator[str, None, None]:
        """Stream tokens one at a time with decay applied via registered hooks."""
        full_prompt = self._format_prompt(prompt)
        try:
            prompt_to_cache_index = full_prompt.index("Knowledge Base:") # Calculate the length of the system prompt to determine where generated tokens start
        except ValueError:
            prompt_to_cache_index = None  # Default to the beginning if the substring is not found
        inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.model.device)
        if prompt_to_cache_index is not None:
            self.system_prompt_tokens_length = len(self.tokenizer(full_prompt[:prompt_to_cache_index], add_special_tokens=True)["input_ids"])
        else:
            self.system_prompt_tokens_length = 0
        
        input_ids = inputs["input_ids"]
        # `current_attention_mask` will track the full attention mask for the entire sequence
        current_attention_mask = inputs["attention_mask"] 
        
        self.token_index = 0  # Reset decay tracking for generated tokens (relative to generated tokens)

        generated_ids = input_ids # This will accumulate prompt tokens + generated tokens
        # if self.kv_cache is not None:
        #     past_key_values = self.kv_cache
        #     cache_len = past_key_values[0][0].shape[2]  # Get sequence length from the first layer's key tensor
        #     cache_mask = torch.ones((1, 1, cache_len), dtype=torch.long, device=self.device)  # Mask for the cached tokens
        #     current_attention_mask = torch.cat([cache_mask, current_attention_mask], dim=-1)
        # else:
        past_key_values = None
        
        with torch.no_grad():
            for i in range(max_tokens):
                # Prepare inputs for the model call
                if past_key_values is None:
                    # First call: pass the full prompt input_ids and its attention_mask
                    model_inputs_for_call = {
                        "input_ids": generated_ids,
                        "attention_mask": current_attention_mask,
                    }
                # elif i == 0 and past_key_values is not None:
                #     model_inputs_for_call = {
                #         "input_ids": generated_ids,
                #         "attention_mask": current_attention_mask,
                #         "past_key_values": past_key_values,
                #     }
                else:
                    # Subsequent calls: pass only the most recently generated token
                    # The attention_mask needs to be extended by one for the new token
                    # and the past_key_values from the previous step are used.
                    new_attention_mask = torch.cat([current_attention_mask, 
                                                    torch.ones((1, 1), dtype=torch.long, device=self.device)], dim=-1)
                    model_inputs_for_call = {
                        "input_ids": generated_ids[:, -1].unsqueeze(-1), # Only the last token
                        "attention_mask": new_attention_mask,
                        "past_key_values": past_key_values,
                    }
                    # Update `current_attention_mask` for the next iteration to include the new token
                    current_attention_mask = new_attention_mask

                # Make the model call using the prepared `model_inputs_for_call`
                outputs = self.model(**model_inputs_for_call, use_cache=True)
                
                logits = outputs.logits[:, -1, :] # Get logits for the last token
                past_key_values = outputs.past_key_values

                # Apply sampling
                if temperature == 0:
                    next_token_logits = logits
                else:
                    next_token_logits = logits / temperature
                    
                # Apply top_k sampling
                if top_k > 0:
                    # Set all logits below the top_k threshold to -inf
                    top_k_values, top_k_indices = torch.topk(next_token_logits, top_k)
                    next_token_logits = torch.full_like(next_token_logits, float('-inf'), device=self.device)
                    next_token_logits.scatter_(1, top_k_indices, top_k_values)
                
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Check for EOS token
                generated_ids = torch.cat([generated_ids, next_token], dim=-1)


                if next_token.item() in self.model.generation_config.eos_token_id:
                    break

                # Append the new token to the sequence of generated IDs for further processing/decoding
                token_str = self.tokenizer.decode(next_token[0], skip_special_tokens=False, clean_up_tokenization_spaces=False)
                yield token_str
                
                self.token_index += 1  # Increment for each generated token, allowing decay in hooks
            
        # It's good practice to remove hooks after generation to prevent side effects
        # self._remove_steering_hooks()

    # def cleanup(self):
    #     """Clean up resources."""
    #     self._remove_steering_hooks()
    #     if hasattr(self.model, 'cpu'):
    #         self.model.cpu()
    #     del self.model
    #     del self.tokenizer