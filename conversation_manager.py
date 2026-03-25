from typing import List, Dict, Optional
from llm_interface import LLMInterface
import torch

class ConversationManager:
    def __init__(self, model_a: LLMInterface, model_b: LLMInterface, starting_model: str = "model_a"):
        self.model_a = model_a
        self.model_b = model_b
        self.starting_model = starting_model  # "model_a" or "model_b"
        self.history: List[Dict[str, str]] = []
        self.current_turn = starting_model  # Tracks whose turn it is next
    
    def add_user_message(self, message: str):
        """Add a user message to the conversation."""
        self.history.append({"role": "user", "content": message})
    
    def get_context(self, max_messages: int = 5) -> str:
        """Get recent conversation context (only responses, no system prompts)."""
        recent = self.history[-max_messages:]
        context_lines = []
        for msg in recent:
            if msg["role"] == "user":
                not_starting_model = self.model_b if self.starting_model == "model_a" else self.model_a
                context_lines.append(f"{not_starting_model.model_display_name}: {msg['content']}")
            elif msg["role"] == "model_a":
                context_lines.append(f"{self.model_a.model_display_name}: {msg['content']}")
            elif msg["role"] == "model_b":
                context_lines.append(f"{self.model_b.model_display_name}: {msg['content']}")
        
        # Make it explicit who should speak next
        if self.current_turn == "model_a":
            context_lines.append("Next response (Model A):")
        else:
            context_lines.append("Next response (Model B):")
        
        return "\n".join(context_lines)
    
    def _last_message_from(self, role: str) -> Optional[str]:
        """Return the most recent message from the given role, or None."""
        for msg in reversed(self.history):
            name = self.model_a.model_display_name if role == "model_a" else self.model_b.model_display_name
            final_message = ""
            if msg["role"] == role:
                final_message += f"{name}: {msg['content']}"
                if msg["role"] == "model_a":
                    final_message += f"\nNext response by {self.model_a.model_display_name}:"
                elif msg["role"] == "model_b":
                    final_message += f"\nNext response by {self.model_b.model_display_name}:"
                return final_message
        return None
    
    def generate_response(self, model: LLMInterface, temperature: float = 0.7,
                         max_tokens: int = 256) -> str:
        """Generate a response from the specified model using only the other side’s last message."""
        if model is self.model_a:
            user_prompt = self._last_message_from("model_b") or self._last_message_from("user") or ""
        else:
            user_prompt = self._last_message_from("model_a") or self._last_message_from("user") or ""

        if model.steering_vector:
            response = model.generate_stream(user_prompt, temperature=temperature, max_tokens=max_tokens)
        else:
            response = model.generate(user_prompt, temperature=temperature, max_tokens=max_tokens)
        return response
    
    def start_conversation(self, user_message: str, temperature: float = 0.7,
                          max_tokens: int = 256) -> Dict[str, str]:
        """Start the conversation: Add user message and generate first response from starting model."""
        self.add_user_message(user_message)
        
        if self.starting_model == "model_a":
            response = "".join(self.generate_response(self.model_a, temperature, max_tokens))
            self.history.append({"role": "model_a", "content": response})
            self.current_turn = "model_b"  # Next turn is Model B
        else:
            response = self.generate_response(self.model_b, temperature, max_tokens)
            self.history.append({"role": "model_b", "content": response})
            self.current_turn = "model_a"  # Next turn is Model A
        
        return {self.starting_model: response}
    
    def continue_conversation(self, temperature: float = 0.7, max_tokens: int = 256) -> Dict[str, str]:
        """Continue the conversation: Generate the next response from the current turn's model."""
        if self.current_turn == "model_a":
            response = self.generate_response(self.model_a, temperature, max_tokens)
            self.history.append({"role": "model_a", "content": response})
            self.current_turn = "model_b"
            return {"model_a": response}
        else:
            response = self.generate_response(self.model_b, temperature, max_tokens)
            self.history.append({"role": "model_b", "content": response})
            self.current_turn = "model_a"
            return {"model_b": response}
    
    def clear_history(self):
        """Clear conversation history."""
        self.history = []
        self.current_turn = self.starting_model  # Reset to starting model
    
    def export_history(self) -> List[Dict[str, str]]:
        """Export conversation history."""
        return self.history.copy()
    
    def is_conversation_started(self) -> bool:
        """Check if the conversation has started (i.e., has at least one response)."""
        return any(msg["role"] in ["model_a", "model_b"] for msg in self.history)