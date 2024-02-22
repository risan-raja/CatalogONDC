from pymongo import MongoClient
import os
import json
from collections import defaultdict
from time import perf_counter, sleep
from pymongo import UpdateOne
from pymongo.errors import BulkWriteError
import hashlib

import tqdm


def get_mongodb():
    client = MongoClient(os.environ['MONGO_URI'])
    db = client['catalogStore']
    return db

def get_key_prefs():
    with open('key_preferences.json', 'r') as f:
        data = json.load(f)
    return data


def clean_key(k):
    if '_' in k:
        return k.replace('_', ' ')
    else:
        return k
    

def get_text_and_metadata(doc,key_prefs):
    # key_prefs = get_key_prefs()
    text_payload = []
    skip_keys = ['l1', 'l2', 'l3', 'l4']
    for key in doc:
        if key in key_prefs['indexed_cols']:
            if key in skip_keys:
                text_payload.append(("", doc[key]))
                continue
            text_payload.append((clean_key(key), doc[key]))
    # total_text = [f'{x}->{y}'.strip() for x,y in text_payload]
    total_text = []
    for x, y in text_payload:
        if x == '':
            total_text.append(y)
        else:
            total_text.append(f'{x}->{y}'.replace('\n',' '))
            
    total_text = '|'.join(total_text).strip()
    del text_payload
    metadata_available = [k for k in doc.keys() if k in key_prefs['shared_id']+key_prefs['indexed_cols'] and k not in key_prefs['ignored_cols']]
    metadata_payload = {k: doc[k] for k in metadata_available}
    return total_text, metadata_payload


def insert_all_text() -> list[dict[str,str]]:
    key_prefs = get_key_prefs()
    db = get_mongodb()
    catalogs = db['catalogs']
    total_documents = catalogs.count_documents({})
    documents = catalogs.find({})
    doc_text_collection = []
    with tqdm.tqdm(total=total_documents) as pbar:
        for document in documents:
            doc_text, doc_meta = get_text_and_metadata(document,key_prefs=key_prefs)
            doc_text_collection.append({
                "_id": document['_id'],
                "text": doc_text
                })
            pbar.update(1)
        text_4_embedding = db['doc_text']
        text_4_embedding.insert_many(doc_text_collection)
    return doc_text_collection


def update_md5_hashes():
    db = get_mongodb()
    text_4_embedding = db['doc_text']
    documents = text_4_embedding.find({})
    total_documents = text_4_embedding.count_documents({})
    update_requests = []
    with tqdm.tqdm(total=total_documents) as pbar:
        for document in documents:
            text = document['text']
            md5_hash = hashlib.md5(text.encode()).hexdigest()
            update_requests.append(UpdateOne({"_id": document['_id']}, {"$set": {"md5_hash": md5_hash}}))
            pbar.update(1)
    print("Updating MD5 Hashes")
    try:
        text_4_embedding.bulk_write(update_requests)
    except BulkWriteError as bwe:
        print(bwe.details)
    print('Done')


def find_duplicates():
    db = get_mongodb()
    text_4_embedding = db['doc_text']
    documents = text_4_embedding.find({}).batch_size(3000)
    total_documents = text_4_embedding.count_documents({})
    md5_hashes = defaultdict(list)
    with tqdm.tqdm(total=total_documents) as pbar:
        for document in documents:
            md5_hash = document['md5_hash']
            md5_hashes[md5_hash].append(document['_id'])
            pbar.update(1)
    duplicates = {k: v for k, v in md5_hashes.items() if len(v) > 1}
    return duplicates

def create_indexed_products():
    db = get_mongodb()
    text_4_embedding = db['doc_text']
    documents = text_4_embedding.find({})
    total_documents = text_4_embedding.count_documents({})
    indexed_products = []
    with tqdm.tqdm(total=total_documents) as pbar:
        for document in documents:
            indexed_products.append(document['_id'])
            pbar.update(1)
    with open('indexed_products.json', 'w') as f:
        json.dump(indexed_products, f)
    print('Done')
