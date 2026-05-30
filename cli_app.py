import pdb

import yaml
import sys
import os
import time
import glob
import pandas as pd # Added pandas import

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

def load_config(config_path="cli_config.yaml"): # Kept original config file name
    """Loads configuration from cli_config.yaml."""
    try:
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: cli_config.yaml not found at {config_path}")
        sys.exit(1)

# Removed load_prompt_from_file as input now comes from DataFrame

def run_cli_conversation():
    config = load_config()

    try:
        token = load_config(config_path="hf_token.yaml").get("token") # Load token from separate config file
    except FileNotFoundError:
        print("Error: hf_token.yaml not found")
        token=None

    print("--- Initializing CLI Dual LLM Conversation ---")

    # Get CLI specific settings
    # cli_input_folder = config.get("cli_input_folder", "./cli_inputs") # Removed
    cli_max_turns = config.get("cli_max_turns", 5)
    cli_input_data = config.get("cli_input_data") # New: Path to input CSV
    cli_output_folder = config.get("cli_output_folder", "./cli_conversations")
    cli_starting_model = config.get("cli_starting_model", "model_a")
    knowledge_base_dir = config.get("knowledge_base_dir", "./knowledge_bases")
    # Changed from default_kb_a_path to default_kb_a_folder for dynamic loading
    default_kb_a_folder = config.get("default_knowledge_base_a_folder")
    # Changed from default_kb_b_path to default_kb_b_folder for dynamic loading
    default_kb_b_folder = config.get("default_knowledge_base_b_folder")

    # Ensure output folder exists
    os.makedirs(cli_output_folder, exist_ok=True)
    if not os.path.exists(knowledge_base_dir):
        os.makedirs(knowledge_base_dir)


    # Initialize shared managers once
    model_manager = ModelManager(cache_dir=config["model_cache_dir"])
    vector_manager = SteeringVectorManager(vector_dir=config["vector_dir"])
    vector_manager.preload_vectors(config.get("vector_a"), config.get("vector_b"), config.get("steering_vector_layer_a", "post_attention_layernorm"), config.get("steering_vector_layer_b", "post_attention_layernorm"))

    # Set common generation settings from config defaults
    temp_a = config.get("default_temperature", 0.7)
    temp_b = config.get("default_temperature", 0.7)
    top_k = config.get("default_top_k", 50)
    max_tokens = config.get("max_tokens", 256)
    decay_rate = config.get("decay_rate", 0.95) # Changed key from default_decay_rate
    max_context = config.get("max_context_messages", 2)
    num_beams = config.get("default_num_beams", 1) # New
    length_penalty = config.get("default_length_penalty", 1.0) # New

    # Load display names for models
    display_name_a = config.get("default_name_a", "Model A")
    display_name_b = config.get("default_name_b", "Model B")

    # Initialize LLM interfaces (can be reused across conversations)
    model_a_id = config["model_a"]
    model_b_id = config["model_b"]

    llm_a = LLMInterface(model_a_id, display_name_a, model_manager, token=token) # Pass display_name_a and token
    llm_b = LLMInterface(model_b_id, display_name_b, model_manager, token=token) # Pass display_name_b and token

    llm_a.set_decay_rate(decay_rate)
    llm_b.set_decay_rate(decay_rate)
    llm_a.set_warmup_tokens(config.get("default_warmup_tokens", 0))
    llm_b.set_warmup_tokens(config.get("default_warmup_tokens", 0))
    llm_a.set_min_intensity(config.get("default_min_intensity_a", 0.0))
    llm_b.set_min_intensity(config.get("default_min_intensity_b", 0.0))

    # Set personalities
    role_a = config.get("default_role_a", "")
    role_b = config.get("default_role_b", "")
    if role_a:
        full_system_prompt_a = compose_system_prompt(role_a, None)
        llm_a.set_personality(full_system_prompt_a)
    if role_b:
        full_system_prompt_b = compose_system_prompt(role_b, None)
        llm_b.set_personality(full_system_prompt_b)

    # Set preloaded steering vectors and initial intensity
    vector_a_obj = vector_manager.get_vector('vector_a')
    vector_b_obj = vector_manager.get_vector('vector_b')
    if vector_a_obj:
        llm_a.set_steering_vector(vector_a_obj)
        llm_a.update_steering_intensity(config.get("default_intensity_a", 0.05))
    if vector_b_obj:
        llm_b.set_steering_vector(vector_b_obj)
        llm_b.update_steering_intensity(config.get("default_intensity_b", 0.05))

    # Find all question files in the input folder
    # ...existing code...
    # question_files = glob.glob(os.path.join(cli_input_folder, "*.txt")) # Removed
    cli_input_df = pd.read_csv(cli_input_data, encoding_errors='ignore') # New: Read input from CSV
    # if not question_files: # Removed
        # print(f"No .txt files found in input folder: {cli_input_folder}. Exiting.") # Removed
        # sys.exit(1) # Removed

    print(f"Found {len(cli_input_df.index)} conversation prompts in '{cli_input_data}'") # Updated message
    print("-" * 40)
    
    # New: DataFrame for structured output
    output_filename_df = "conversations_df.csv"
    output_file_path_df = os.path.join(cli_output_folder, output_filename_df)
    df = pd.DataFrame(columns=["question_id", "patient_question", "provider_response", "turn_id"])

    # ...existing code...
    # for i, question_file_path in enumerate(question_files): # Replaced with DataFrame iteration
    for i, row in cli_input_df.iterrows():
        # pdb.set_trace()
        # file_base_name = os.path.basename(question_file_path) # Removed
        # print(f"\n--- Starting Conversation {i+1}/{len(question_files)}: {file_base_name} ---") # Removed
        print(f"\n--- Starting Conversation {i+1}/{len(cli_input_df.index)}: QA ID {row.qa_id} ---") # Updated message

        # starting_prompt = load_prompt_from_file(question_file_path) # Removed
        starting_prompt = row.question # New: Get prompt from DataFrame row
        # if not starting_prompt: # Removed
            # print(f"Skipping {file_base_name} due to empty or unreadable prompt.") # Removed
            # continue # Removed
        
        # New: Dynamic knowledge base loading per conversation
        knowledge_base_a_content = None
        knowledge_base_b_content = None
        if default_kb_a_folder and row.trial_num:
            path = os.path.join(default_kb_a_folder, f"{row.trial_num}.txt")
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as file:
                    knowledge_base_a_content = file.read()

        if default_kb_b_folder and row.trial_num:
            path = os.path.join(default_kb_b_folder, f"{row.trial_num}.txt")
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as file:
                    knowledge_base_b_content = file.read()

        # Start timer for this conversation
        start_time = time.time()

        # Re-initialize ConversationManager for each new conversation
        conversation_manager = ConversationManager(llm_a, llm_b, cli_starting_model, knowledge_base_a=knowledge_base_a_content, knowledge_base_b=knowledge_base_b_content)
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
                max_tokens=max_tokens,
                num_beams=num_beams,
                length_penalty=length_penalty
            )
            for model_key, content in responses_dict.items():
                display_name = llm_a.model_display_name if model_key == "model_a" else llm_b.model_display_name
                print(f"[{display_name}]: {content}")
                conversation_history.append(f"[{display_name}]: {content}")

        except Exception as e:
            print(f"Error starting conversation for {row.qa_id}: {e}") # Updated error message
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
                    max_context=max_context,
                    num_beams=num_beams,
                    length_penalty=length_penalty
                )
                for model_key, content in responses_dict.items():
                    display_name = llm_a.model_display_name if model_key == "model_a" else llm_b.model_display_name
                    print(f"[{display_name}]: {content}")
                    conversation_history.append(f"[{display_name}]: {content}")
            except Exception as e:
                print(f"Error during turn {turn+1} for {row.qa_id}: {e}") # Updated error message
                break # Exit on error for this conversation

        # End timer for this conversation
        end_time = time.time()
        runtime = end_time - start_time
        print(f"\n--- Conversation for {row.qa_id} Finished (Runtime: {runtime:.2f} seconds) ---") # Updated message

        # Save conversation history to a file in the output folder
        output_filename = str(int(row.qa_id)) + "_conversation.txt" # Updated filename generation
        output_file_path = os.path.join(cli_output_folder, output_filename)
        try:
            with open(output_file_path, "w", encoding="utf-8") as f:
                f.write(f"--- Conversation for: Question {row.qa_id} ---\n") # Updated header
                f.write(f"--- Runtime: {runtime:.2f} seconds ---\n\n")
                for line in conversation_history:
                    f.write(line + "\n--------------------\n") # Added separator for readability
            print(f"Conversation history saved to {output_file_path}")
        except IOError as e:
            print(f"Error saving conversation history to {output_file_path}: {e}")
        print("-" * 40)

        # New: Save conversation history to DataFrame
        try:
            skip = 2 # Assuming patient and provider alternate
            question_id = row.qa_id
            for j in range(0, len(conversation_history), skip):
                turn = j/skip + 1
                patient = conversation_history[j]
                provider = conversation_history[j+1]
                df.loc[len(df)] = {"question_id": question_id, "patient_question": patient, "provider_response": provider, "turn_id": turn}
        except Exception as e: # Catch specific exception
            print(f"Error saving conversation history to dataframe for QA ID {row.qa_id}: {e}") # More informative error

    df.to_csv(output_file_path_df, index=False) # New: Save DataFrame to CSV
    print(f"All conversation data saved to {output_file_path_df}") # New: Confirmation message
    print("\n--- All CLI Conversations Finished ---")
    

if __name__ == "__main__":
    run_cli_conversation()
