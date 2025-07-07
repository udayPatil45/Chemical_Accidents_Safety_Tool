import os
import re
import pdfplumber
import pandas as pd
import numpy as np
import faiss
import streamlit as st
import requests
from sentence_transformers import SentenceTransformer

# --- Setup paths ---
pdf_folder = r"C:\Users\Hp\Desktop\ACCIDENTS"
csv_path = os.path.join(pdf_folder, "your_research_papers_extracted.csv")

# --- Extract case details from PDF text ---
def extract_case_details(text):
    text = text.replace('\n', ' ').replace('  ', ' ')
    title = re.search(r'Title:\s*(.*?)\.', text)
    location = re.search(r'Location:\s*(.*?)\.', text)
    date = re.search(r'Dt\.:\s*(\d{2}/\d{2}/\d{4})', text)
    return {
        'title': title.group(1) if title else '',
        'location': location.group(1) if location else '',
        'date': date.group(1) if date else '',
        'text': text
    }

# --- Read PDFs and extract data ---
data_rows = []

for filename in os.listdir(pdf_folder):
    if filename.lower().endswith(".pdf"):
        with pdfplumber.open(os.path.join(pdf_folder, filename)) as pdf:
            full_text = ""
            for page in pdf.pages:
                full_text += page.extract_text() or ""
            if full_text.strip():
                data_rows.append(extract_case_details(full_text))

# --- Create DataFrame and CSV ---
df = pd.DataFrame(data_rows, columns=['title', 'location', 'date', 'text'])
if df.empty:
    raise ValueError("No valid PDF text extracted. Check your files.")
df.to_csv(csv_path, index=False)

# --- Embedding text ---
df = pd.read_csv(csv_path).fillna("N/A")
texts = df['text'].tolist()

embedder = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = embedder.encode(texts, show_progress_bar=True, convert_to_numpy=True).astype("float32")

# --- FAISS index ---
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

import requests
import json

def generate_answer(question, context):
    prompt = f"""Use the following context to answer the question:

Context:
{context}

Question:
{question}

Answer:"""

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "tinyllama",
            "prompt": prompt,
            "stream": True
        },
        stream=True
    )

    output = ""
    for line in response.iter_lines():
        if line:
            data = json.loads(line.decode('utf-8'))
            output += data.get("response", "")

    return output


# --- Streamlit app ---
st.title("📄 Accident Case Chatbot")
query = st.text_input("🔍 Ask a question:")

if query:
    q_emb = embedder.encode([query], convert_to_numpy=True).astype("float32")
    _, I = index.search(q_emb, k=3)

    context = "\n\n".join(texts[i] for i in I[0])
    answer = generate_answer(query, context)

    st.markdown("### ✅ Answer:")
    st.write(answer)

    st.markdown("### 📚 Context Used:")
    for i in I[0]:
        st.write(f"- {texts[i][:300]}...")