import streamlit as st
import yaml
import tempfile
import os # New: Import os for path operations
from model_manager import ModelManager
from llm_interface import LLMInterface
from custom_steering_vectors import SteeringVectorManager, CustomSteeringVector
from conversation_manager import ConversationManager

# Helper function to compose the system prompt from role and knowledge base
def compose_system_prompt(role_content, knowledge_base_content):
    prompt_parts = []
    if role_content:
        prompt_parts.append(role_content.strip())
    if knowledge_base_content:
        prompt_parts.append("Knowledge Base:\n" + knowledge_base_content.strip())
    return "\n\n".join(prompt_parts)

# Load config
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Ensure knowledge base directory exists
knowledge_base_dir = config.get("knowledge_base_dir", "./knowledge_bases")
if not os.path.exists(knowledge_base_dir):
    os.makedirs(knowledge_base_dir)

st.set_page_config(page_title="Dual LLM Conversation", layout="wide")
st.title("🤖 Dual LLM Conversation with Steering Vectors")

# Initialize managers and session state for KB
if "model_manager" not in st.session_state:
    st.session_state.model_manager = ModelManager(cache_dir=config["model_cache_dir"])
    st.session_state.vector_manager = SteeringVectorManager(vector_dir=config["vector_dir"], default_intensity=config["default_intensity"])
    st.session_state.vector_manager.preload_vectors(config.get("vector_a"), config.get("vector_b"))
    st.session_state.conversation = None
    st.session_state.vector_a = st.session_state.vector_manager.get_vector('vector_a')
    st.session_state.vector_b = st.session_state.vector_manager.get_vector('vector_b')
    st.session_state.knowledge_base_a_content = "" # New: Store KB content in session state
    st.session_state.knowledge_base_b_content = "" # New: Store KB content in session state
    st.session_state.knowledge_base_a_name = "None" # New: Store KB filename
    st.session_state.knowledge_base_b_name = "None" # New: Store KB filename

    # Load default knowledge base A if path is specified and no content yet
    default_kb_a_path = config.get("default_knowledge_base_a_path")
    if default_kb_a_path and not st.session_state.knowledge_base_a_content:
        full_path_a = os.path.join(knowledge_base_dir, default_kb_a_path)
        if os.path.exists(full_path_a):
            try:
                with open(full_path_a, "r", encoding="utf-8") as kb_file:
                    st.session_state.knowledge_base_a_content = kb_file.read()
                    st.session_state.knowledge_base_a_name = os.path.basename(full_path_a)
            except Exception as e:
                st.error(f"Error loading default knowledge base A from {full_path_a}: {e}")
        else:
            st.session_state.knowledge_base_a_content = None
            st.session_state.knowledge_base_a_name = "None"
    else:
        st.session_state.knowledge_base_a_content = None
        st.session_state.knowledge_base_a_name = "None"

    # Load default knowledge base B if path is specified and no content yet
    default_kb_b_path = config.get("default_knowledge_base_b_path")
    if default_kb_b_path and not st.session_state.knowledge_base_b_content:
        full_path_b = os.path.join(knowledge_base_dir, default_kb_b_path)
        if os.path.exists(full_path_b):
            try:
                with open(full_path_b, "r", encoding="utf-8") as kb_file:
                    st.session_state.knowledge_base_b_content = kb_file.read()
                    st.session_state.knowledge_base_b_name = os.path.basename(full_path_b)
            except Exception as e:
                st.error(f"Error loading default knowledge base B from {full_path_b}: {e}")
        else:
            st.session_state.knowledge_base_b_content = None
            st.session_state.knowledge_base_b_name = "None"
    else:
        st.session_state.knowledge_base_b_content = None
        st.session_state.knowledge_base_b_name = "None"
# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    model_a = st.selectbox("Model A", config["available_models"], key="model_a_select")
    model_b = st.selectbox("Model B", config["available_models"], key="model_b_select")
    
    st.subheader("Roles") # Changed: "Personalities" to "Roles"
    # Changed: personality_a to role_a, default_personality_a to default_role_a, key="pers_a" to key="role_a"
    role_a = st.text_area("Model A Role", value=config.get("default_role_a", ""), placeholder="Enter role prompt...", key="role_a")
    # Changed: personality_b to role_b, default_personality_b to default_role_b, key="pers_b" to key="role_b"
    role_b = st.text_area("Model B Role", value=config.get("default_role_b", ""), placeholder="Enter role prompt...", key="role_b")

    col_role_a, col_role_b = st.columns(2) # Changed: col_pers_a to col_role_a
    with col_role_a:
        # Changed: "Update Personality A" to "Update Role A" and logic to update system prompt
        if st.button("Update Role A") and st.session_state.conversation:
            current_kb_content_a = st.session_state.knowledge_base_a_content
            full_system_prompt_a = compose_system_prompt(role_a, None)
            # Assumed change in LLMInterface: update_personality -> set_personality
            st.session_state.conversation.model_a.set_personality(full_system_prompt_a) 
            st.success("Model A role updated!")
    with col_role_b:
        # Changed: "Update Personality B" to "Update Role B" and logic to update system prompt
        if st.button("Update Role B") and st.session_state.conversation:
            current_kb_content_b = st.session_state.knowledge_base_b_content
            full_system_prompt_b = compose_system_prompt(role_b, None)
            # Assumed change in LLMInterface: update_personality -> set_personality
            st.session_state.conversation.model_b.set_personality(None)
            st.success("Model B role updated!")
    
    st.subheader("Knowledge Bases") # New: Subheader for Knowledge Bases
    
    # New: File uploader and update button for Knowledge Base A
    uploaded_file_a = st.file_uploader("Upload Knowledge Base A", type=["txt"], key="upload_kb_a")
    if uploaded_file_a is not None:
        string_data = uploaded_file_a.getvalue().decode("utf-8")
        st.session_state.knowledge_base_a_content = string_data
        st.session_state.knowledge_base_a_name = uploaded_file_a.name
        st.success(f"Loaded Knowledge Base A: {uploaded_file_a.name}")
    st.info(f"Current Knowledge Base A: {st.session_state.knowledge_base_a_name}")

    if st.button("Update Knowledge Base A") and st.session_state.conversation:
        # Get current role content from the UI widget
        current_role_a_from_ui = st.session_state.get("role_a", "")
        full_system_prompt_a = compose_system_prompt(current_role_a_from_ui, None)
        # Assumed change in LLMInterface: update_personality -> set_personality
        st.session_state.conversation.model_a.set_personality(full_system_prompt_a)
        st.success("Model A knowledge base updated!")

    st.markdown("---") # Separator

    # New: File uploader and update button for Knowledge Base B
    uploaded_file_b = st.file_uploader("Upload Knowledge Base B", type=["txt"], key="upload_kb_b")
    if uploaded_file_b is not None:
        string_data = uploaded_file_b.getvalue().decode("utf-8")
        st.session_state.knowledge_base_b_content = string_data
        st.session_state.knowledge_base_b_name = uploaded_file_b.name
        st.success(f"Loaded Knowledge Base B: {uploaded_file_b.name}")
    st.info(f"Current Knowledge Base B: {st.session_state.knowledge_base_b_name}")

    if st.button("Update Knowledge Base B") and st.session_state.conversation:
        # Get current role content from the UI widget
        current_role_b_from_ui = st.session_state.get("role_b", "")
        full_system_prompt_b = compose_system_prompt(current_role_b_from_ui, None)
        # Assumed change in LLMInterface: update_personality -> set_personality
        st.session_state.conversation.model_b.set_personality(full_system_prompt_b)
        st.success("Model B knowledge base updated!")


    st.subheader("Names")
    name_a = st.text_input("Model A Name", value=config.get("default_name_a", "Model A"), key="name_a")
    name_b = st.text_input("Model B Name", value=config.get("default_name_b", "Model B"), key="name_b")

    st.subheader("Steering Vector Intensity")
    intensity_a = st.slider("Intensity A", -2.0, 2.0, config["default_intensity"], step=0.01, key="intensity_a")
    intensity_b = st.slider("Intensity B", -2.0, 2.0, config["default_intensity"], step=0.01, key="intensity_b")
    col_intensity_a, col_intensity_b = st.columns(2)
    with col_intensity_a:
        if st.button("Update Steering Intensity A") and st.session_state.conversation:
            st.session_state.conversation.model_a.update_steering_intensity(intensity_a)
            st.success("Model A steering intensity updated!")
    with col_intensity_b:
        if st.button("Update Steering Intensity B") and st.session_state.conversation:
            st.session_state.conversation.model_b.update_steering_intensity(intensity_b)
            st.success("Model B steering intensity updated!")

    decay_rate = st.slider("Steering Decay Rate", 0.0, 1.0, 0.95, step=0.01, 
                          help="Lower = faster decay. 1.0 = no decay", key="decay_rate")
    
    st.subheader("Generation Settings")
    temp_a = st.slider("Temperature A", 0.0, 2.0, 0.7, key="temp_a")
    temp_b = st.slider("Temperature B", 0.0, 2.0, 0.7, key="temp_b")
    top_k = st.slider("Top-K", 1, 100, 50, key="top_k")
    max_tokens = st.slider("Max Tokens", 50, 500, 256)
    max_context = st.slider("Max Context Messages", 1, 20, config.get("max_context_messages", 2))

    starting_model = st.selectbox("Starting Model", ["model_a", "model_b"], key="starting_model")
    
    st.subheader("Conversation Setup")
    starting_prompt = st.text_area("Starting Prompt", placeholder="Enter the initial conversation topic...", key="starting_prompt", height=100)
    
    if st.button("Initialize Models"):
        llm_a = LLMInterface(model_a, name_a, st.session_state.model_manager)
        llm_b = LLMInterface(model_b, name_b, st.session_state.model_manager)
        
        llm_a.set_decay_rate(decay_rate)
        llm_b.set_decay_rate(decay_rate)
        
        # New: Compose full system prompts for initialization
        full_system_prompt_a = compose_system_prompt(role_a, None)
        full_system_prompt_b = compose_system_prompt(role_b, None)

        # Assumed change in LLMInterface: set_personality -> set_system_prompt
        if full_system_prompt_a:
            llm_a.set_personality(full_system_prompt_a)
        if full_system_prompt_b:
            llm_b.set_personality(full_system_prompt_b)
        
        # Set preloaded steering vectors only if available
        if st.session_state.vector_a:
            llm_a.set_steering_vector(st.session_state.vector_a)
            llm_a.update_steering_intensity(intensity_a)  # Set initial intensity from slider
        if st.session_state.vector_b:
            llm_b.set_steering_vector(st.session_state.vector_b)
            llm_b.update_steering_intensity(intensity_b)  # Set initial intensity from slider
        st.session_state.conversation = ConversationManager(llm_a, llm_b, starting_model, knowledge_base_a=st.session_state.knowledge_base_a_content, knowledge_base_b=st.session_state.knowledge_base_b_content)
        st.success("Models initialized! (Steering vectors applied only where configured)")

# Main conversation area
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("Start Conversation") and starting_prompt and st.session_state.conversation and not st.session_state.conversation.is_conversation_started():
        with st.spinner("Generating first response..."):
            temp = temp_a if starting_model == "model_a" else temp_b
            responses = st.session_state.conversation.start_conversation(
                starting_prompt,
                temperature=temp
            )
        st.success("Conversation started!")

with col2:
    if st.button("Continue Conversation") and st.session_state.conversation and st.session_state.conversation.is_conversation_started():
        with st.spinner("Generating next response..."):
            temp = temp_a if st.session_state.conversation.current_turn == "model_a" else temp_b
            responses = st.session_state.conversation.continue_conversation(
                temperature=temp,
                max_tokens=max_tokens,
                max_context=max_context
            )
        st.success("Response generated!")

with col3:
    if st.button("Clear History") and st.session_state.conversation:
        st.session_state.conversation.clear_history()
        st.success("History cleared!")

# Display conversation history (always visible)
st.subheader("Conversation History")
if st.session_state.conversation and st.session_state.conversation.history:
    for msg in st.session_state.conversation.history:
        if msg["role"] == "user":
            st.write(f"**Initial Query (on behalf of model not starting):** {msg['content']}")
        elif msg["role"] == "model_a":
            st.write(f"**{name_a}:** {msg['content']}")
        elif msg["role"] == "model_b":
            st.write(f"**{name_b}:** {msg['content']}")
else:
    st.write("No conversation history yet. Initialize models and start a conversation!")