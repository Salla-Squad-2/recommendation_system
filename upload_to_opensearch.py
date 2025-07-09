import pandas as pd
import json
import requests
import os
import re

# Configuration
FILES = [
    {
        'csv': './dataset/gucci_cleaned_with_en.csv',
        'schema': 'schema_product_en.json',
        'index': 'products-en'
    },
    {
        'csv': './dataset/gucci_cleaned_with_ar.csv',
        'schema': 'schema_product_ar.json',
        'index': 'products'
    }
]

OPENSEARCH_URL = 'http://localhost:9200'
HEADERS = {'Content-Type': 'application/json'}
BATCH_SIZE = 10     # You can change it

def upload_batch(index_name, batch_df):
    for _, row in batch_df.iterrows():
        bulk_data = json.dumps({"index": {"_index": index_name}}) + '\n'
        doc = row.to_dict()

        for key, value in doc.items():
            if isinstance(value, pd.Timestamp):
                doc[key] = value.isoformat()
            elif hasattr(value, 'astype'): 
                try:
                    doc[key] = pd.to_datetime(value).isoformat()
                except Exception:
                    pass
        
        bulk_data += json.dumps(doc, ensure_ascii=False) + '\n'
        response = requests.post(f"{OPENSEARCH_URL}/_bulk", headers=HEADERS, data=bulk_data.encode('utf-8'))

    if response.status_code != 200:
        print(f"Batch upload failed: {response.status_code} {response.text}")
        return False

    result = response.json()
    if result.get('errors'):
        print(f"Errors in bulk upload for {index_name}:")
        for item in result['items']:
            if 'error' in item['index']:
                print(json.dumps(item['index']['error'], indent=2))
        return False

    print(f"Batch uploaded successfully: {len(batch_df)} records")
    return True

def process_file(csv_file, schema_file, index_name):

    try:
        df = pd.read_csv(csv_file, encoding='utf-8')
        print(f"Loaded {len(df)} records from '{csv_file}'")
    except Exception as e:
        print(f"Error loading CSV '{csv_file}': {e}")
        return

    if df is None:
        print(f"Data conversion failed for '{csv_file}'")
        return

    df = df.where(pd.notnull(df), None) 

    print(f"Uploading data to index '{index_name}' in batches...")
    for start in range(0, len(df), BATCH_SIZE):
        end = min(start + BATCH_SIZE, len(df))
        print(f"Uploading records {start + 1} to {end}...")
        batch_df = df.iloc[start:end]
        if not upload_batch(index_name, batch_df):
            print("Stopping upload due to errors.")
            break
    else:
        print(f"All batches uploaded successfully for '{csv_file}'.")

if __name__ == '__main__':
    for f in FILES:
        print(f"\n--- Processing {f['csv']} ---")
        process_file(f['csv'], f['schema'], f['index'])
