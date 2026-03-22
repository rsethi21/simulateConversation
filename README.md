# On command line:
Some versions specific to testing machine (i.e. pytorch), but can change.
```
pip install -r requirements.txt 
```

```
streamlit run app.py
```

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
