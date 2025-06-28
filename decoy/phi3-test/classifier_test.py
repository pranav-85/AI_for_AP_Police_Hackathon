import pandas as pd
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load Excel
df = pd.read_excel("test_cleaned.xlsx")


# Initialize phi3 model via Ollama
model_name = "phi3:instruct"
llm = OllamaLLM(model=model_name, temperature=0.1, max_tokens=1000)

# Define a prompt template
prompt_template = PromptTemplate.from_template(
    """You are an AI assistant that classifies individual chat messages as either:

- suspicious: includes anything harmful, spam-like, or intended to distract or mislead (this includes decoy messages).
- benign: normal, safe, and friendly conversation.

Classify the following message into one of these two categories.

Message: "{message}"

Respond with just one word: suspicious or benign.
"""
)

# Function to classify a message
def classify_message(message):
    try:
        prompt = prompt_template.format(message=message)
        response = llm.invoke(prompt).lower()

        print(f"Response: {response}")

        if "suspicious" in response:
            return "suspicious"
        elif "benign" in response:
            return "benign"
        else:
            return "unknown"
    except Exception as e:
        print(f"Error classifying message: {e}")
        return "error"

# Run classification
print("Classifying messages using phi3 + LangChain...")
df["Predicted"] = df["Message"].apply(classify_message)

# Save results
df.to_excel("phi3-test/classified_output.xlsx", index=False)
print("Results saved to 'classified_output.xlsx'.")

# Evaluation
valid_df = df[~df["Predicted"].isin(["error", "unknown"])]
acc = accuracy_score(valid_df["Label"], valid_df["Predicted"])
print(f"Accuracy: {acc:.2%}")

# Confusion matrix
cm = confusion_matrix(valid_df["Label"], valid_df["Predicted"], labels=["suspicious", "benign"])
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["suspicious", "benign"], yticklabels=["suspicious", "benign"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
