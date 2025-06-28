import pandas as pd
from transformers import pipeline
import re

emoji_pattern = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map symbols
    "\U0001F1E0-\U0001F1FF"  # flags
    "\U00002500-\U00002BEF"  # CJK characters
    "\U00002702-\U000027B0"
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "\U0001f926-\U0001f937"
    "\U00010000-\U0010ffff"
    "\u2640-\u2642"
    "\u2600-\u2B55"
    "\u200d"
    "\u23cf"
    "\u23e9"
    "\u231a"
    "\ufe0f"
    "\u3030"
    "]+",
    flags=re.UNICODE
)

def remove_emojis(text):
    return emoji_pattern.sub(r'', text)

classifier = pipeline("text-classification", model="mshenoda/roberta-spam")

df = pd.read_csv("chat_log.csv")
df.columns = [col.strip().lower() for col in df.columns]

df['message'] = df['message'].astype(str).apply(remove_emojis)

def predict_label(text):
    result = classifier(str(text)[:512])[0]
    return 'suspicious' if result['label'] == 'LABEL_1' else 'benign'

df['predicted'] = df['message'].apply(predict_label)

df.to_csv("chatlog_classified.csv", index=False)
print("\nResults saved to 'chatlog_classified.csv'")
