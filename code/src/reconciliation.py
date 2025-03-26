import pandas as pd
from sentence_transformers import SentenceTransformer, util
import openai
import os

# Set your OpenAI API key
openai.api_key = "your_openai_api_key"

# Load AI Model for Text Matching
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load data from two source systems
df1 = pd.read_csv("data/source1.csv")
df2 = pd.read_csv("data/source2.csv")

# Merge datasets on a common key (Transaction ID)
key_column = "transaction_id"
merged = df1.merge(df2, on=key_column, how="outer", suffixes=('_src1', '_src2'), indicator=True)

# Identify matching and missing records
matched = merged[merged['_merge'] == 'both']
missing_in_src2 = merged[merged['_merge'] == 'left_only']
missing_in_src1 = merged[merged['_merge'] == 'right_only']

# Identify mismatches (AI-powered fuzzy matching)
def ai_fuzzy_match(row, column):
    text1 = str(row[f"{column}_src1"])
    text2 = str(row[f"{column}_src2"])
    embedding1 = model.encode(text1, convert_to_tensor=True)
    embedding2 = model.encode(text2, convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(embedding1, embedding2).item()
    return similarity

mismatches = matched.copy()
mismatches['similarity_score'] = mismatches.apply(lambda row: ai_fuzzy_match(row, "amount"), axis=1)
mismatches = mismatches[mismatches['similarity_score'] < 0.9]  # Threshold for mismatch

# AI-powered discrepancy explanation
def explain_discrepancy(row):
    prompt = f"""
    Given the following records from two systems:
    System A: {row['amount_src1']}
    System B: {row['amount_src2']}
    
    Identify discrepancies and suggest a resolution.
    """
    response = openai.ChatCompletion.create(model="gpt-4", messages=[{"role": "system", "content": "You are an AI expert in data reconciliation."}, {"role": "user", "content": prompt}])
    return response["choices"][0]["message"]["content"]

mismatches['ai_explanation'] = mismatches.apply(lambda row: explain_discrepancy(row), axis=1)

# Save reports
matched.to_csv("output/matched_records.csv", index=False)
missing_in_src2.to_csv("output/missing_in_src2.csv", index=False)
missing_in_src1.to_csv("output/missing_in_src1.csv", index=False)
mismatches.to_csv("output/mismatches.csv", index=False)

print("✅ Reconciliation completed! Reports saved in 'output' folder.")
