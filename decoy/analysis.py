import pandas as pd

df = pd.read_excel("chatlog_classified_phi3.xlsx")

print(df.columns)

df["mobile_number"] = df["mobile_number"].astype(str).str.strip()
df["predicted"] = df["predicted"].str.lower().str.strip()

counts = df.groupby(["mobile_number", "predicted"]).size().reset_index(name="count")

pivot_df = counts.pivot(index="mobile_number", columns="predicted", values="count").fillna(0)

pivot_df["total_messages"] = pivot_df.sum(axis=1)

pivot_df = pivot_df.reset_index()

print(pivot_df)

pivot_df.to_excel("mobile_suspicious_benign_counts.xlsx", index=False)
print("Saved results to 'mobile_suspicious_benign_counts.xlsx'")
