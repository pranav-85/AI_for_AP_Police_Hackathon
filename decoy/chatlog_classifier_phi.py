import pandas as pd
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM

# Load CSV
df = pd.read_csv("chat_log.csv")

# Normalize column names
df.columns = [col.strip().lower() for col in df.columns]

# Initialize phi3 model via Ollama
model_name = "phi3:instruct"
llm = OllamaLLM(model=model_name, temperature=0.1, max_tokens=1000)

# Define prompt template
prompt_template = PromptTemplate.from_template(
    """You are analyzing a short cluster of messages from one user in a group chat.

Definition:
A decoy agent is someone who BOTH:
1) Clearly claims they personally received money (or payment), AND
2) Encourages other people to trust the work, send money, or join in,

with these statements happening in the same message or within a short time window.

Below is the cluster of messages:

------------------------
{messages}
------------------------

If the user clearly claims they received money AND encourages others to trust or pay, classify as 'decoy'.
If not, classify as 'not_decoy'.

Respond with just one word: decoy OR not_decoy.
"""
)


# Function to classify a message
def classify_message(message):
    try:
        prompt = prompt_template.format(message=message)
        response = llm.invoke(prompt).strip().lower()
        if "suspicious" in response:
            return "suspicious"
        elif "benign" in response:
            return "benign"
        else:
            print(f"Unclear response: '{response}'")
            return "unknown"
    except Exception as e:
        print(f"Error classifying message: {e}")
        return "error"

# Run classification
print("Classifying messages using phi3 + LangChain...")
df["predicted"] = df["message"].apply(classify_message)

# Save results
df.to_excel("chatlog_classified_phi3.xlsx", index=False)
print("Results saved to 'chatlog_classified.xlsx'.")