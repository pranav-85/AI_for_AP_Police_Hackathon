import pandas as pd
import json
from collections import defaultdict
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from datetime import timedelta

# Load chat log
df = pd.read_excel("chatlog_classified_phi3.xlsx")
df["mobile_number"] = df["mobile_number"].astype(str).str.strip()
df["message"] = df["message"].astype(str).str.strip()
df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
df = df.sort_values(by="timestamp")

# Group messages by mobile_number
user_messages = defaultdict(list)
for _, row in df.iterrows():
    ts = row["timestamp"]
    msg = row["message"]
    mobile = row["mobile_number"]
    user_messages[mobile].append((ts, msg))

# Initialize phi3
llm = OllamaLLM(model="phi3:instruct", temperature=0, max_tokens=512)

# Prompt: clear decoy rule
prompt_template = PromptTemplate.from_template(
    """You are analyzing a group chat for scam support.

Definition:
A decoy agent is someone who tries to convince others that the work or scam is real by **claiming they got paid
and encouraging others.**

Below is a short cluster of messages from one user:

------------------------
{messages}
------------------------

If these messages show that the user BOTH claims they got paid AND encourages others,
and these parts happen in the same message or close together in time,
classify as 'decoy'. Otherwise, 'not_decoy'.

One word only: decoy OR not_decoy.
"""
)

results = {}

# For each user
for mobile, msg_list in user_messages.items():
    msg_list = sorted(msg_list, key=lambda x: x[0])  # sort by time
    decoy_votes = 0
    total_clusters = 0

    i = 0
    while i < len(msg_list):
        cluster = [msg_list[i]]
        j = i + 1

        # Add messages within short time window (e.g., 1 hour)
        while j < len(msg_list):
            delta = msg_list[j][0] - msg_list[j-1][0]
            if delta <= timedelta(hours=1):
                cluster.append(msg_list[j])
                j += 1
            else:
                break

        # Build prompt
        cluster_text = "\n".join([f"[{ts.strftime('%Y-%m-%d %H:%M')}] {msg}" for ts, msg in cluster])
        prompt = prompt_template.format(messages=cluster_text)

        try:
            response = llm.invoke(prompt).strip().lower()
            first_word = response.split()[0]
            if "decoy" in first_word and "not" not in first_word:
                decoy_votes += 1
        except Exception as e:
            print(f"Error for {mobile}: {e}")

        total_clusters += 1

        # Move to next cluster window
        i = j

    print(f"{mobile}: {decoy_votes} decoy votes out of {total_clusters}")

    if total_clusters > 0 and decoy_votes > 0:
        user_label = "decoy"
    else:
        user_label = "not_decoy"

    results[mobile] = user_label

# Save to JSON
with open("decoy_classification.json", "w") as f:
    json.dump(results, f, indent=4)

print("Done! Saved results to 'decoy_classification.json'.")
