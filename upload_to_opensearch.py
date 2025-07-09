import pandas as pd
import json
import requests
import os
import re

# Configuration
FILES = [
    {
        'csv': 'product_data_en.csv',
        'schema': 'schema_product_en.json',
        'index': 'product-data-en'
    },
    {
        'csv': 'product_data_ar.csv',
        'schema': 'schema_product_ar.json',
        'index': 'product-data-ar'
    }
]

OPENSEARCH_URL = 'http://localhost:9200'
HEADERS = {'Content-Type': 'application/json'}
BATCH_SIZE = 1000

def clean_price(price):
    if pd.isna(price):
        return None
    if isinstance(price, (int, float)):
        return float(price)
    if isinstance(price, str):
        # Remove $ sign, commas, spaces, keep digits and dot
        cleaned = re.sub(r'[^\d\.]', '', price)
        try:
            return float(cleaned)
        except:
            return None
    return None

def parse_prices(prices_str):
    if not prices_str or prices_str == '[]' or pd.isna(prices_str):
        return []

    cleaned = prices_str.strip()
    cleaned = cleaned.replace("'", '"')
    cleaned = re.sub(r'[\n\r\t]', ' ', cleaned)
    cleaned = re.sub(r'}\s*{', '},{', cleaned)
    cleaned = re.sub(r'\bTrue\b', 'true', cleaned)
    cleaned = re.sub(r'\bFalse\b', 'false', cleaned)
    cleaned = re.sub(r'\bNaN\b', 'null', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r',\s*([\]}])', r'\1', cleaned)

    try:
        parsed = json.loads(cleaned)
        for item in parsed:
            if 'price' in item:
                try:
                    item['price'] = float(item['price'])
                except Exception:
                    item['price'] = None
            if 'lastFullPrice' in item:
                try:
                    if item['lastFullPrice'] == '':
                        item['lastFullPrice'] = None
                    else:
                        item['lastFullPrice'] = float(item['lastFullPrice'])
                except Exception:
                    item['lastFullPrice'] = None
        return parsed
    except json.JSONDecodeError as e:
        print(f"JSON decode error in prices: {e}")
        print(f"Problematic string (snippet): {cleaned[:200]}...")
        return []

def convert_dates_and_ids(df):
    if 'lastUpdated' in df.columns:
        try:
            df['lastUpdated'] = pd.to_datetime(df['lastUpdated'], errors='coerce')
        except Exception as e:
            print(f"Error converting 'lastUpdated': {e}")
            return None
    return df

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

def upload_batch(index_name, batch_df):
    bulk_data = ''
    for _, row in batch_df.iterrows():
        bulk_data += json.dumps({"index": {"_index": index_name}}) + '\n'
        doc = row.to_dict()

        # Convert datetime to ISO format string
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
    if not create_index(index_name, schema_file):
        print(f"Skipping uploading data for index '{index_name}' due to index creation failure.")
        return

    try:
        df = pd.read_csv(csv_file, encoding='utf-8')
        print(f"Loaded {len(df)} records from '{csv_file}'")
    except Exception as e:
        print(f"Error loading CSV '{csv_file}': {e}")
        return

    if 'prices' in df.columns:
        df['prices'] = df['prices'].apply(parse_prices)

    if 'price' in df.columns:
        df['price'] = df['price'].apply(clean_price)
    if 'rawPrice' in df.columns:
        df['rawPrice'] = df['rawPrice'].apply(clean_price)

    df = convert_dates_and_ids(df)
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
