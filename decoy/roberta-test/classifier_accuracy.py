import pandas as pd
from transformers import pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

classifier = pipeline("text-classification", model="mshenoda/roberta-spam")

df = pd.read_excel("test_cleaned.xlsx")

df.columns = [col.strip().lower() for col in df.columns]

def predict_label(text):
    result = classifier(str(text)[:512])[0]
    return 'suspicious' if result['label'] == 'LABEL_1' else 'benign'

df['predicted'] = df['message'].apply(predict_label)

y_true = df['label'].str.lower().str.strip()
y_pred = df['predicted']

accuracy = accuracy_score(y_true, y_pred)
print(f"\nAccuracy: {accuracy:.4f}")

conf_mat = confusion_matrix(y_true, y_pred)
print("\nConfusion Matrix:")
print(conf_mat)

print("\nClassification Report:")
print(classification_report(y_true, y_pred))

df.to_excel("roberta-test/test_with_predictions.xlsx", index=False)
print("\nResults saved to 'test_with_predictions.xlsx'")

plt.figure(figsize=(6, 4))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', xticklabels=['benign', 'suspicious'], yticklabels=['benign', 'suspicious'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()