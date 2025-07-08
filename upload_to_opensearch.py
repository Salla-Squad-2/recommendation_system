import pandas as pd
import json
import requests
import os

# Configuration
FILES = [
    {
        'csv': '/data/ecommerce_customer_data_en.csv',
        'schema': 'schema_en.json',
        'index': 'data-recommendation-en'
    },
    {
        'csv': '/data/ecommerce_customer_data_ar.csv',
        'schema': 'schema_ar.json',
        'index': 'data-recommendation-ar'
    }
]
OPENSEARCH_URL = 'http://localhost:9200'
HEADERS = {'Content-Type': 'application/json'}
BATCH_SIZE = 1000

def create_index(index_name, schema_file):
    if not os.path.exists(schema_file):
        print(f"Schema file '{schema_file}' not found!")
        return False

    with open(schema_file, 'r', encoding='utf-8') as f:
        schema = json.load(f)

    print(f"Creating index '{index_name}'...")
    res = requests.put(f"{OPENSEARCH_URL}/{index_name}", headers=HEADERS, json=schema)
    if res.status_code in (200, 201):
        print(f"Index '{index_name}' created.")
        return True
    elif 'resource_already_exists_exception' in res.text:
        print(f"Index '{index_name}' already exists.")
        return True
    else:
        print(f"Failed to create index '{index_name}': {res.text}")
        return False

def convert_dates_and_ids(df):
    try:
        df['Customer ID'] = df['Customer ID'].astype(int)
    except Exception as e:
        print(f"Error converting 'Customer ID' to int: {e}")
        return None

    try:
        df['Purchase Date'] = pd.to_datetime(df['Purchase Date']).dt.strftime('%-m/%-d/%Y %H:%M')
    except Exception as e:
        print(f"Date conversion error: {e}, trying Windows format...")
        try:
            df['Purchase Date'] = pd.to_datetime(df['Purchase Date']).dt.strftime('%#m/%#d/%Y %H:%M')
        except Exception as e2:
            print(f"Windows date conversion also failed: {e2}")
            return None
    return df

def upload_batch(index_name, batch_df):
    bulk_data = ''
    for _, row in batch_df.iterrows():
        bulk_data += json.dumps({"index": {"_index": index_name}}) + '\n'
        doc = row.to_dict()
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
                print(item['index']['error'])
        return False
    return True

def process_file(csv_file, schema_file, index_name):
    if not create_index(index_name, schema_file):
        return
    try:
        df = pd.read_csv(csv_file, encoding='utf-8')
        print(f"Loaded {len(df)} records from {csv_file}")
    except Exception as e:
        print(f"Error loading CSV {csv_file}: {e}")
        return

    df = convert_dates_and_ids(df)
    if df is None:
        print(f"Data conversion failed for {csv_file}")
        return

    print(f"Uploading data for {index_name} in batches...")
    for start in range(0, len(df), BATCH_SIZE):
        end = min(start + BATCH_SIZE, len(df))
        print(f"Uploading records {start + 1} to {end} for {index_name}...")
        batch_df = df.iloc[start:end]
        if not upload_batch(index_name, batch_df):
            print("Stopping due to errors.")
            break
    print(f"Finished uploading {csv_file}.\n")

if __name__ == '__main__':
    for f in FILES:
        process_file(f['csv'], f['schema'], f['index'])