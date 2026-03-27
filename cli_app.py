import yaml
import sys
import os
import time
import glob

from model_manager import ModelManager
from llm_interface import LLMInterface
from custom_steering_vectors import SteeringVectorManager
from conversation_manager import ConversationManager

# Helper function to compose the system prompt from role and knowledge base
def compose_system_prompt(role_content, knowledge_base_content):
    prompt_parts = []
    if role_content:
        prompt_parts.append(role_content.strip())
    if knowledge_base_content:
        prompt_parts.append("Knowledge Base:\n" + knowledge_base_content.strip())
    return "\n\n".join(prompt_parts)

def load_config(config_path="cli_config.yaml"):
    """Loads configuration from cli_config.yaml."""
    try:
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: cli_config.yaml not found at {config_path}")
        sys.exit(1)

def load_prompt_from_file(file_path):
    """Loads a single prompt from a specified text file."""
    if not os.path.exists(file_path):
        print(f"Error: Prompt file not found at {file_path}")
        return None
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read().strip()

def run_cli_conversation():
    config = load_config()

    print("--- Initializing CLI Dual LLM Conversation ---")

    # Get CLI specific settings
    cli_input_folder = config.get("cli_input_folder", "./cli_inputs")
    cli_max_turns = config.get("cli_max_turns", 5)
    cli_output_folder = config.get("cli_output_folder", "./cli_conversations")
    cli_starting_model = config.get("cli_starting_model", "model_a")
    knowledge_base_dir = config.get("knowledge_base_dir", "./knowledge_bases")
    default_kb_a_path = config.get("default_knowledge_base_a_path")
    default_kb_b_path = config.get("default_knowledge_base_b_path")

    if default_kb_a_path != None:
        full_path_a = os.path.join(knowledge_base_dir, default_kb_a_path)
        if os.path.exists(full_path_a):
            with open(full_path_a, "r", encoding="utf-8") as kb_file:
                knowledge_base_a_content = kb_file.read()
        else:
            knowledge_base_a_content = None
    else:
        knowledge_base_a_content = None
    
    if default_kb_b_path != None:
        full_path_b = os.path.join(knowledge_base_dir, default_kb_b_path)
        if os.path.exists(full_path_b):
            with open(full_path_b, "r", encoding="utf-8") as kb_file:
                knowledge_base_b_content = kb_file.read()
        else:
            knowledge_base_b_content = None
    else:
        knowledge_base_b_content = None

    # Ensure output folder exists
    os.makedirs(cli_output_folder, exist_ok=True)
    # Ensure knowledge base directory exists
    knowledge_base_dir = config.get("knowledge_base_dir", "./knowledge_bases")
    if not os.path.exists(knowledge_base_dir):
        os.makedirs(knowledge_base_dir)


    # Initialize shared managers once
    model_manager = ModelManager(cache_dir=config["model_cache_dir"])
    vector_manager = SteeringVectorManager(vector_dir=config["vector_dir"], default_intensity=config["default_intensity"])
    vector_manager.preload_vectors(config.get("vector_a"), config.get("vector_b"))

    # Set common generation settings from config defaults
    temp_a = config.get("default_temperature", 0.7)
    temp_b = config.get("default_temperature", 0.7)
    top_k = config.get("default_top_k", 50)
    max_tokens = config.get("max_tokens", 256)
    decay_rate = config.get("decay_rate", 0.95)
    max_context = config.get("max_context_messages", 2)

    # Load display names for models
    display_name_a = config.get("default_name_a", "Model A")
    display_name_b = config.get("default_name_b", "Model B")

    # Initialize LLM interfaces (can be reused across conversations)
    model_a_id = config["model_a"]
    model_b_id = config["model_b"]

    llm_a = LLMInterface(model_a_id, display_name_a, model_manager) # Pass display_name_a
    llm_b = LLMInterface(model_b_id, display_name_b, model_manager) # Pass display_name_b

    llm_a.set_decay_rate(decay_rate)
    llm_b.set_decay_rate(decay_rate)

    # Set personalities
    role_a = config.get("default_role_a", "")
    role_b = config.get("default_role_b", "")
    if role_a:
        full_system_prompt_a = compose_system_prompt(role_a, knowledge_base_a_content)
        llm_a.set_personality(full_system_prompt_a)
    if role_b:
        full_system_prompt_b = compose_system_prompt(role_b, knowledge_base_b_content)
        llm_b.set_personality(full_system_prompt_b)

    # Set preloaded steering vectors and initial intensity
    vector_a_obj = vector_manager.get_vector('vector_a')
    vector_b_obj = vector_manager.get_vector('vector_b')
    if vector_a_obj:
        llm_a.set_steering_vector(vector_a_obj)
        llm_a.update_steering_intensity(config.get("default_intensity", 0.05))
    if vector_b_obj:
        llm_b.set_steering_vector(vector_b_obj)
        llm_b.update_steering_intensity(config.get("default_intensity", 0.05))

    # Find all question files in the input folder
    question_files = glob.glob(os.path.join(cli_input_folder, "*.txt"))
    if not question_files:
        print(f"No .txt files found in input folder: {cli_input_folder}. Exiting.")
        sys.exit(1)

    print(f"Found {len(question_files)} conversation prompts in '{cli_input_folder}'")
    print("-" * 40)

    for i, question_file_path in enumerate(question_files):
        file_base_name = os.path.basename(question_file_path)
        print(f"\n--- Starting Conversation {i+1}/{len(question_files)}: {file_base_name} ---")

        starting_prompt = load_prompt_from_file(question_file_path)
        if not starting_prompt:
            print(f"Skipping {file_base_name} due to empty or unreadable prompt.")
            continue

        # Start timer for this conversation
        start_time = time.time()

        # Re-initialize ConversationManager for each new conversation
        conversation_manager = ConversationManager(llm_a, llm_b, cli_starting_model)
        conversation_history = []

        # The initial user prompt is technically from the "user" role, but for CLI output, we can attribute it
        # to the non-starting model's personality, if that's how it's framed.
        # However, for consistency with app.py's "Initial Query (on behalf of model not starting)",
        # we'll just log it as "Initial Prompt".
        print(f"[INITIAL PROMPT]: {starting_prompt}")
        conversation_history.append(f"[INITIAL PROMPT]: {starting_prompt}")
        
        current_temp = temp_a if cli_starting_model == "model_a" else temp_b
        try:
            # start_conversation returns a dict like {"model_a": response_content}
            responses_dict = conversation_manager.start_conversation(
                starting_prompt,
                temperature=current_temp,
                max_tokens=max_tokens
            )
            for model_key, content in responses_dict.items():
                display_name = llm_a.model_display_name if model_key == "model_a" else llm_b.model_display_name
                print(f"[{display_name}]: {content}")
                conversation_history.append(f"[{display_name}]: {content}")
        except Exception as e:
            print(f"Error starting conversation for {file_base_name}: {e}")
            continue # Move to the next file

        for turn in range(1, cli_max_turns):
            print(f"\n--- Turn {turn+1} ---")
            current_model_turn = conversation_manager.current_turn
            current_temp = temp_a if current_model_turn == "model_a" else temp_b
            try:
                # continue_conversation returns a dict like {"model_b": response_content}
                responses_dict = conversation_manager.continue_conversation(
                    temperature=current_temp,
                    max_tokens=max_tokens,
                    max_context=max_context
                )
                for model_key, content in responses_dict.items():
                    display_name = llm_a.model_display_name if model_key == "model_a" else llm_b.model_display_name
                    print(f"[{display_name}]: {content}")
                    conversation_history.append(f"[{display_name}]: {content}")
            except Exception as e:
                print(f"Error during turn {turn+1} for {file_base_name}: {e}")
                break # Exit on error for this conversation

        # End timer for this conversation
        end_time = time.time()
        runtime = end_time - start_time
        print(f"\n--- Conversation for {file_base_name} Finished (Runtime: {runtime:.2f} seconds) ---")

        # Save conversation history to a file in the output folder
        output_filename = os.path.splitext(file_base_name)[0] + "_conversation.txt"
        output_file_path = os.path.join(cli_output_folder, output_filename)
        try:
            with open(output_file_path, "w", encoding="utf-8") as f:
                f.write(f"--- Conversation for: {file_base_name} ---\n")
                f.write(f"--- Runtime: {runtime:.2f} seconds ---\n\n")
                for line in conversation_history:
                    f.write(line + "\n")
            print(f"Conversation history saved to {output_file_path}")
        except IOError as e:
            print(f"Error saving conversation history to {output_file_path}: {e}")
        print("-" * 40)

    print("\n--- All CLI Conversations Finished ---")

if __name__ == "__main__":
    run_cli_conversation()