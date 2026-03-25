# Motivation
Utilizing multi-agent interactions fosters emergent behavior and authentic information asymmetry that a single-instance narrator cannot replicate. By isolating the internal state of each entity, you prevent the "homogenization" of personas, allowing for more rigorous adversarial testing and distinct semantic profiles. This setup is particularly effective for uncovering edge cases in logic or tone, as each turn acts as an independent probe that forces the opposing instance to respond to unpredictable stimuli rather than a self-generated, globally-consistent script.

# On command line:
Some versions specific to testing machine (i.e. pytorch), but can change.
```
pip install -r requirements.txt 
```

```
streamlit run app.py
```

# Configs:
You can change which open source models to have as options by listing model names from huggingface under "available_models" key.

Change "vector_a" and/or "vector_b" to null if you do not have steering vectors. Steering vectors created using https://github.com/steering-vectors/steering-vectors.

# On UI:
1. Select model configurations
- Model Name
- System Prompts or Personality
- Steering Vectors
- Steering Intensities
- Which model starts?
2. Click "Initialize Models"
3. Type in starting prompt/question (this will serve as a question that the model not selected to start will "ask")
4. Click "Start Conversation"
5. Click "Continue Conversation" after the first model responds and to continue the conversation after model turns
6. "Clear History" to restart conversation

# On CLI:
This will allow you to generate conversations on the CLI in a streamlined fashion. 

Create a folder called "cli_inputs" in the main directory. Add in text file for each question separately (i.e. question1.txt).

Adjust he cli_config.yaml file similar to what is explained on the UI config and run the following command:
```
python cli_app.py
```
Outputs will be stored in a newly created folder with the name of the question file, runtime statistics, and conversation.
