from elasticsearch import Elasticsearch
import pandas as pd


es = Elasticsearch([{'host': 'localhost', 'port': 9200}])


df = pd.read_csv('customer_history_data_ar.xlsx')


for index, row in df.iterrows():
    document = {
        'Customer ID': row['Customer ID'],
        'Product Code': row['Product Code'],
        'Product Name': row['Product Name'],
        'Category': row['Category'],
        'Purchase Date': row['Purchase Date'],
        'Description': row['Description'],
        'Quantity': row['Quantity'],
        'Price': row['Price']
    }
   
    es.index(index="customer_data", body=document)
