import pandas as pd
import numpy as np
import json, re, os, sys
import requests
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

# ────────── CONFIG ──────────
CSV_PATH        = "sample_products.csv"        
SCHEMA_PATH     = "schema.json"               
INDEX_NAME      = "product_vectors"
OPENSEARCH_URL  = "http://localhost:9200"     
MODEL_NAME      = "intfloat/multilingual-e5-base"
BATCH_SIZE_DOCS = 100
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"

def find_col(df, patterns):
    for col in df.columns:
        for p in patterns:
            if re.search(p, col, re.IGNORECASE):
                return col
    return None

print(f"Loading model: {MODEL_NAME} on {DEVICE}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE).eval()

@torch.no_grad()
def embed(text_batch):
    inputs = tokenizer(
        [f"passage: {t}" for t in text_batch],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256
    ).to(DEVICE)

    output = model(**inputs).last_hidden_state[:, 0, :]
    normalized = torch.nn.functional.normalize(output, p=2, dim=1)
    return normalized.cpu().numpy()

if not os.path.isfile(CSV_PATH):
    sys.exit(f"CSV not found: {CSV_PATH}")

df = pd.read_csv(CSV_PATH).fillna("")

id_col   = find_col(df, ["^id$", "product[_-]?id", "id_product"])
name_col = find_col(df, ["name", "title", "name_product"])
desc_col = find_col(df, ["description", "desc"])
cat_col  = find_col(df, ["category", "type", "section"])

if not all([id_col, name_col, desc_col, cat_col]):
    sys.exit("Could not detect all required columns: id, name, description, category.")

print("Detected columns:")
print(f"   id           → {id_col}")
print(f"   name         → {name_col}")
print(f"   description  → {desc_col}")
print(f"   category     → {cat_col}")

def column_to_vectors(series, label):
    print(f"🔄 Embedding '{label}' …")
    texts = series.astype(str).tolist()
    all_vectors = []
    for i in tqdm(range(0, len(texts), 64)):
        batch = texts[i:i+64]
        vectors = embed(batch)
        all_vectors.append(vectors)
    return np.vstack(all_vectors)

name_vecs = column_to_vectors(df[name_col], "name")
desc_vecs = column_to_vectors(df[desc_col], "description")
cat_vecs  = column_to_vectors(df[cat_col], "category")
combo_vecs = (name_vecs + desc_vecs + cat_vecs) / 3.0

if not os.path.exists(SCHEMA_PATH):
    sys.exit(f"Schema file not found: {SCHEMA_PATH}")

with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
    mapping = json.load(f)

print(f"Creating or resetting index '{INDEX_NAME}'")
requests.delete(f"{OPENSEARCH_URL}/{INDEX_NAME}", auth=OPENSEARCH_AUTH)

response = requests.put(
    f"{OPENSEARCH_URL}/{INDEX_NAME}",
    headers={"Content-Type": "application/json"},
    data=json.dumps(mapping),
    auth=OPENSEARCH_AUTH
)

if response.status_code not in (200, 201):
    sys.exit(f"Failed to create index: {response.text}")
    
def bulk_upload(start, end):
    lines = []
    for i in range(start, end):
        doc = {
            "id_product":         str(df.iloc[i][id_col]),
            "name_vector":        name_vecs[i].tolist(),
            "description_vector": desc_vecs[i].tolist(),
            "category_vector":    cat_vecs[i].tolist(),
            "combination_vector": combo_vecs[i].tolist()
        }
        lines.append(json.dumps({ "index": { "_index": INDEX_NAME } }))
        lines.append(json.dumps(doc, ensure_ascii=False))
    
    body = "\n".join(lines) + "\n"
    res = requests.post(
        f"{OPENSEARCH_URL}/_bulk",
        headers={"Content-Type": "application/json"},
        data=body.encode("utf-8"),
        auth=OPENSEARCH_AUTH
    )
    
    if res.status_code != 200 or res.json().get("errors"):
        print("Bulk upload failed")
        print(res.text)
    else:
        print(f"Uploaded documents {start} to {end}")

print("Uploading vectors to OpenSearch")
for start in range(0, len(df), BATCH_SIZE_DOCS):
    end = min(start + BATCH_SIZE_DOCS, len(df))
    bulk_upload(start, end)

print("Done uploading all data.")