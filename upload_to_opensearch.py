import pandas as pd
import json
import requests
import os

# Configuration
FILES = [
    {
        'csv': 'ecommerce_customer_data_en.csv',
        'schema': 'schema_en.json',
        'index': 'data-recommendation-en'
    },
    {
        'csv': 'ecommerce_customer_data_ar.csv',
        'schema': 'schema_ar.json',
        'index': 'data-recommendation-ar'
    }
]

OPENSEARCH_URL = 'http://localhost:9200'
HEADERS = {'Content-Type': 'application/json'}
BATCH_SIZE = 1000


def delete_index(index_name):
    print(f"Deleting index '{index_name}' if exists...")
    res = requests.delete(f"{OPENSEARCH_URL}/{index_name}", headers=HEADERS)
    if res.status_code == 200:
        print(f"Index '{index_name}' deleted.")
    elif res.status_code == 404:
        print(f"Index '{index_name}' does not exist, no need to delete.")
    else:
        print(f"Failed to delete index '{index_name}': {res.status_code} {res.text}")

def create_index(index_name, schema_file):
    if not os.path.exists(schema_file):
        print(f"Schema file '{schema_file}' not found!")
        return False

    with open(schema_file, 'r', encoding='utf-8') as f:
        schema = json.load(f)

    print(f"Creating index '{index_name}'...")
    res = requests.put(f"{OPENSEARCH_URL}/{index_name}", headers=HEADERS, json=schema)
    if res.status_code in (200, 201):
        print(f"Index '{index_name}' created or already exists.")
        return True
    elif res.status_code == 400 and 'resource_already_exists_exception' in res.text:
        print(f"Index '{index_name}' already exists.")
        return True
    else:
        print(f"Failed to create index '{index_name}': {res.status_code} {res.text}")
        return False

def convert_dates_and_ids(df):
    if 'Customer ID' not in df.columns or 'Purchase Date' not in df.columns:
        print("CSV missing required columns.")
        return None

    try:
        df['Customer ID'] = df['Customer ID'].astype(int)
    except Exception as e:
        print(f"Error converting 'Customer ID' to int: {e}")
        return None

    try:
        df['Purchase Date'] = pd.to_datetime(df['Purchase Date'], errors='raise')
    except Exception as e:
        print(f"Date conversion error: {e}")
        return None

    return df

def upload_batch(index_name, batch_df):
    bulk_data = ''
    for _, row in batch_df.iterrows():
        bulk_data += json.dumps({"index": {"_index": index_name}}) + '\n'
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
    delete_index(index_name)

    if not create_index(index_name, schema_file):
        print(f"Skipping uploading data for index '{index_name}' due to index creation failure.")
        return

    try:
        df = pd.read_csv(csv_file, encoding='utf-8')
        print(f"Loaded {len(df)} records from '{csv_file}'")
    except Exception as e:
        print(f"Error loading CSV '{csv_file}': {e}")
        return

    df = convert_dates_and_ids(df)
    if df is None:
        print(f"Data conversion failed for '{csv_file}'")
        return

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