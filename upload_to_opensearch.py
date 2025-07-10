import os
import json
import pandas as pd
import numpy as np
import torch
import requests
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

# Configuration 
OPENSEARCH_URL = 'http://localhost:9200'
HEADERS = {'Content-Type': 'application/json'}
MODEL_NAME = 'intfloat/multilingual-e5-base'
BATCH_SIZE_UPLOAD = 10
BATCH_SIZE_EMBED = 64
BATCH_SIZE = 10

print('Loading model and tokenizer...')
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.eval()
    print('Model loaded successfully')
except Exception as e:
    print(f'Error loading model: {str(e)}')
    exit(1)

FILES = [
    {
        'csv': './dataset/customer_history_data_ar.xlsx',
        'schema': './schema/schema_history_customer.json',
        'index': 'products-history-ar',
        'with_vectors': False
    },
    {
        'csv': './dataset/customer_history_data_ar_with_vectors.csv',
        'schema': './schema/schema_product_ar_with_vectors.json',
        'index': 'products-history-vectors',
        'with_vectors': True
    }
]

# Vector embedding generation
@torch.no_grad()
def generate_embeddings(texts):
    """Generate embeddings for a list of texts using the language model"""
    inputs = tokenizer([f'passage: {t}' for t in texts], 
                       return_tensors='pt',
                       padding=True, 
                       truncation=True, 
                       max_length=256)
    outputs = model(**inputs).last_hidden_state[:, 0, :]
    return torch.nn.functional.normalize(outputs, p=2, dim=1).cpu().numpy()

def process_vectors(df, text_columns):
    """Generate vectors for specified text columns"""
    vectors = {}
    for col in text_columns:
        print(f'Generating vectors for {col}...')
        batch_vectors = []
        for i in tqdm(range(0, len(df), BATCH_SIZE)):
            texts = df[col].iloc[i:i+BATCH_SIZE].fillna('').astype(str).tolist()
            batch_vectors.append(generate_embeddings(texts))
        vectors[f'{col}_vector'] = np.vstack(batch_vectors)
    
    for col, vec in vectors.items():
        df[col] = vec.tolist()
    if len(vectors) > 0:
        all_vecs = np.stack(list(vectors.values()))
        df['combination_vector'] = (all_vecs.mean(axis=0)).tolist()
    return df

def check_index_exists(name):
    """Check if an index exists and how many documents it has"""
    auth = None
    if os.getenv('OPENSEARCH_USER') and os.getenv('OPENSEARCH_PASS'):
        auth = (os.getenv('OPENSEARCH_USER'), os.getenv('OPENSEARCH_PASS'))
    
    try:
        response = requests.get(f'{OPENSEARCH_URL}/{name}/_count', auth=auth)
        if response.status_code == 200:
            count = response.json().get('count', 0)
            return True, count
        return False, 0
    except Exception as e:
        print(f'Error checking index: {str(e)}')
        return False, 0

def recreate_index(name, schema_path, force=True):
    """Recreate the OpenSearch index with the given schema"""
    if not os.path.exists(schema_path):
        print(f'Schema file not found: {schema_path}')
        return False
        
    auth = None
    if os.getenv('OPENSEARCH_USER') and os.getenv('OPENSEARCH_PASS'):
        auth = (os.getenv('OPENSEARCH_USER'), os.getenv('OPENSEARCH_PASS'))
    
    exists, count = check_index_exists(name)
    if exists:
        if not force:
            print(f'Index {name} already exists with {count} documents. Set force=True to recreate.')
            return False
        print(f'Deleting existing index {name} with {count} documents...')
        response = requests.delete(f'{OPENSEARCH_URL}/{name}', auth=auth)
        if response.status_code not in (200, 404):
            print(f'Failed to delete index: {response.text}')
            return False
    
    print(f'Creating index {name} with schema from {schema_path}')
    with open(schema_path, 'r', encoding='utf-8') as f:
        schema = json.load(f)
    
    response = requests.put(
        f'{OPENSEARCH_URL}/{name}',
        headers=HEADERS,
        json=schema,
        auth=auth
    )
    
    if response.status_code not in (200, 201):
        print(f'Failed to create index: {response.text}')
        return False
    
    print(f'Successfully created index: {name}')
    return True

def upload_batch(index_name, batch_df):
    """Upload a batch of documents to OpenSearch"""
    auth = None
    if os.getenv('OPENSEARCH_USER') and os.getenv('OPENSEARCH_PASS'):
        auth = (os.getenv('OPENSEARCH_USER'), os.getenv('OPENSEARCH_PASS'))
    bulk_data = ''
    for _, row in batch_df.iterrows():
        bulk_data += json.dumps({'index': {'_index': index_name}}) + '\n'
        
        doc = {}
        for key, value in row.items():
            if key.endswith('_vector'):
                try:
                    if isinstance(value, str):
                        value = eval(value)
                    if isinstance(value, np.ndarray):
                        value = value.tolist()
                    if isinstance(value, list):
                        doc[key] = [float(x) for x in value]
                    else:
                        print(f'Warning: Invalid vector format for {key}: {type(value)}')
                        continue
                except Exception as e:
                    print(f'Error processing vector {key}: {str(e)}')
                    continue
            else:
                if isinstance(value, (np.int64, np.float64)):
                    value = value.item()
                if pd.isna(value) or value is None:
                    if key in ['quantity_of_product', 'price']:
                        doc[key] = 0
                    else:
                        doc[key] = ''
                elif isinstance(value, pd.Timestamp):
                    doc[key] = value.isoformat()
                else:
                    doc[key] = str(value)
        vector_fields = {k: v for k, v in doc.items() if k.endswith('_vector')}
        if vector_fields:
            print(f'Vector fields being uploaded: {list(vector_fields.keys())}')
            for k, v in vector_fields.items():
                if isinstance(v, list):
                    print(f'{k} length: {len(v)}')
        
        bulk_data += json.dumps(doc, ensure_ascii=False) + '\n'
    response = requests.post(
        f'{OPENSEARCH_URL}/_bulk',
        headers=HEADERS,
        data=bulk_data.encode('utf-8'),
        auth=auth
    )
    
    if response.status_code != 200:
        print(f'Upload failed: {response.text}')
        return False
    
    result = response.json()
    if result.get('errors'):
        print('Errors in bulk upload:')
        for item in result['items']:
            if 'error' in item['index']:
                print(item['index']['error'])
        return False
    return True

def process_file(config):
    """Process a single file according to its configuration"""
    file_path = config['csv']
    print(f'\nProcessing: {file_path}')
    try:
        if config['csv'].endswith('.xlsx'):
            df = pd.read_excel(config['csv'])
        else:
            df = pd.read_csv(config['csv'])
        print(f'Loaded {len(df)} records')
    except Exception as e:
        print(f'Error reading file: {str(e)}')
        return
    for col in df.columns:
        if col not in ['name_vector', 'description_vector', 'category_vector', 'combination_vector']:
            df[col] = df[col].fillna('')
    print('\nColumns before renaming:\n' + '\n'.join(df.columns.tolist()))
    
    if config.get('with_vectors'):
        vector_columns = ['name_vector', 'description_vector', 'category_vector', 'combination_vector']
        has_vectors = all(col in df.columns for col in vector_columns)
        
        if not has_vectors:
            if 'name_product' in df.columns:
                text_columns = ['name_product', 'description', 'category']
            else:
                text_columns = ['name', 'description', 'category']
            df = process_vectors(df, text_columns)
            print('Vector embeddings generated')

            out_path = './dataset/customer_history_data_ar_with_vectors.csv'
            df.to_csv(out_path, index=False)
            print(f'Saved processed data to {out_path}')
        else:
            print('Vector embeddings already exist in the data')

    df = df.rename(columns={
        'name_product': 'name'  
    })
    
    required_columns = ['id_customer', 'productCode', 'name', 'category', 'purchase_date', 
                       'description', 'quantity_of_product', 'price', 'order_id']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f'Warning: Missing required columns: {missing_columns}')
    
    print('\nColumns after renaming:\n' + '\n'.join(df.columns.tolist()))
    
    if not recreate_index(config['index'], config['schema'], force=True):
        return
    
    print(f'\nUploading {len(df)} documents to {config["index"]}...')
    total_batches = len(df) // BATCH_SIZE + (1 if len(df) % BATCH_SIZE > 0 else 0)
    for i in range(0, len(df), BATCH_SIZE):
        batch_df = df.iloc[i:i+BATCH_SIZE]
        print(f'Uploading batch {i//BATCH_SIZE + 1}/{total_batches} ({i} to {i+len(batch_df)-1})...')
        if not upload_batch(config['index'], batch_df):
            print('Upload failed, stopping')
            return
    
    _, final_count = check_index_exists(config['index'])
    print(f'Successfully processed {config["csv"]}')
    print(f'Final document count in {config["index"]}: {final_count}')

if __name__ == '__main__':
    print('Starting data processing and upload to OpenSearch...')
    for config in FILES:
        process_file(config)
    print('\nProcessing complete!')