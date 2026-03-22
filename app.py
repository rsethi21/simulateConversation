import streamlit as st
import yaml
import tempfile
from model_manager import ModelManager
from llm_interface import LLMInterface
from custom_steering_vectors import SteeringVectorManager, CustomSteeringVector
from conversation_manager import ConversationManager

# Load config
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

st.set_page_config(page_title="Dual LLM Conversation", layout="wide")
st.title("🤖 Dual LLM Conversation with Steering Vectors")

# Initialize managers
if "model_manager" not in st.session_state:
    st.session_state.model_manager = ModelManager(cache_dir=config["model_cache_dir"])
    st.session_state.vector_manager = SteeringVectorManager(vector_dir=config["vector_dir"], default_intensity=config["default_intensity"])
    st.session_state.vector_manager.preload_vectors(config.get("vector_a"), config.get("vector_b"))  # Use .get() for safety
    st.session_state.conversation = None
    st.session_state.vector_a = st.session_state.vector_manager.get_vector('vector_a')
    st.session_state.vector_b = st.session_state.vector_manager.get_vector('vector_b')

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    model_a = st.selectbox("Model A", config["available_models"], key="model_a_select")
    model_b = st.selectbox("Model B", config["available_models"], key="model_b_select")
    
    st.subheader("Personalities")
    personality_a = st.text_area("Model A Personality", value=config.get("default_personality_a", ""), placeholder="Enter personality prompt...", key="pers_a")
    personality_b = st.text_area("Model B Personality", value=config.get("default_personality_b", ""), placeholder="Enter personality prompt...", key="pers_b")
    
    col_pers_a, col_pers_b = st.columns(2)
    with col_pers_a:
        if st.button("Update Personality A") and st.session_state.conversation:
            st.session_state.conversation.model_a.update_personality(personality_a)
            st.success("Model A personality updated!")
    with col_pers_b:
        if st.button("Update Personality B") and st.session_state.conversation:
            st.session_state.conversation.model_b.update_personality(personality_b)
            st.success("Model B personality updated!")
    
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
    
    starting_model = st.selectbox("Starting Model", ["model_a", "model_b"], key="starting_model")
    
    st.subheader("Conversation Setup")
    starting_prompt = st.text_area("Starting Prompt", placeholder="Enter the initial conversation topic...", key="starting_prompt", height=100)
    
    if st.button("Initialize Models"):
        llm_a = LLMInterface(model_a, st.session_state.model_manager)
        llm_b = LLMInterface(model_b, st.session_state.model_manager)
        
        llm_a.set_decay_rate(decay_rate)
        llm_b.set_decay_rate(decay_rate)
        
        # Set personalities
        if personality_a:
            llm_a.set_personality(personality_a)
        if personality_b:
            llm_b.set_personality(personality_b)
        
        # Set preloaded steering vectors only if available
        if st.session_state.vector_a:
            llm_a.set_steering_vector(st.session_state.vector_a)
            llm_a.update_steering_intensity(intensity_a)  # Set initial intensity from slider
        if st.session_state.vector_b:
            llm_b.set_steering_vector(st.session_state.vector_b)
            llm_b.update_steering_intensity(intensity_b)  # Set initial intensity from slider
        st.session_state.conversation = ConversationManager(llm_a, llm_b, starting_model)
        st.success("Models initialized! (Steering vectors applied only where configured)")

# Main conversation area
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("Start Conversation") and starting_prompt and st.session_state.conversation and not st.session_state.conversation.is_conversation_started():
        with st.spinner("Generating first response..."):
            temp = temp_a if starting_model == "model_a" else temp_b
            responses = st.session_state.conversation.start_conversation(
                starting_prompt,
                temperature=temp,
                max_tokens=max_tokens
            )
        st.success("Conversation started!")

with col2:
    if st.button("Continue Conversation") and st.session_state.conversation and st.session_state.conversation.is_conversation_started():
        with st.spinner("Generating next response..."):
            temp = temp_a if st.session_state.conversation.current_turn == "model_a" else temp_b
            responses = st.session_state.conversation.continue_conversation(
                temperature=temp,
                max_tokens=max_tokens
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
            st.write(f"**User:** {msg['content']}")
        elif msg["role"] == "model_a":
            st.write(f"**Model A:** {msg['content']}")
        elif msg["role"] == "model_b":
            st.write(f"**Model B:** {msg['content']}")
else:
    st.write("No conversation history yet. Initialize models and start a conversation!")