"""
# Raw Product Sample
{
  "_id": "a3e0cd2c_0acd074e",
  "catalog_id": "a3e0cd2c",
  "product_sku": "0acd074e",
  "url_uid": 271333,
  "l0": "RETAIL",
  "l1": "Fashion",
  "l2": "Men's Apparel",
  "l3": "Topwear",
  "l4": "Shirts",
  "brand": "Allen Solly",
  "product_name": "Men Blue Tartan Checked Slim Fit Pure Cotton Casual Shirt",
  "short_product_description": "Blue tartan checks checked opaque Casual shirt ,has a spread collar, button placket, 1 patch pocket, long regular sleeves, curved hem",
  "size": "40",
  "colour": "Blue",
  "country_of_origin": "India",
  "pattern": "Checked",
  "occasion": "Casual",
  "material_or_content": "Cotton",
  "fabric": "Pure Cotton,Cotton",
  "cuff": "Button",
  "pockets": "1",
  "manufacturer_name_and_address": "Westbury Holdings Pvt Ltd_Madura, Aditya Birla Fashion and Retail,No.6, Belakhahalli,Retail Ltd),Arehalligudhahalli, Doddaballapura,561203,Doddaballapura,Karnataka,India",
  "fit": "Slim Fit",
  "print_or_pattern_type": "Tartan Checks",
  "length": "Regular",
  "collar": "Spread Collar",
  "placket": "Button Placket",
  "pocket_type": "Patch",
  "pack_qty": 1,
  "placket_length": "Full",
  "weave_pattern": "Regular",
  "trend": "New Basics",
  "gender": "Men",
  "selling_price": 11260,
  "sleeve_length": "Long Sleeves",
  "hemline": "Curved",
  "transparency": "Opaque",
  "mrp": 2299,
  "all_urls": [
    "https://datalabs.siva3.io/images/MenShirts_AllenSolly_20459562MenBlueTartanCheckedSlimFitPureCottonCasualShirt_2.jpg",
    "https://datalabs.siva3.io/images/MenShirts_AllenSolly_20459562MenBlueTartanCheckedSlimFitPureCottonCasualShirt_4.jpg",
    "https://datalabs.siva3.io/images/MenShirts_AllenSolly_20459562MenBlueTartanCheckedSlimFitPureCottonCasualShirt_3.jpg",
    "https://datalabs.siva3.io/images/MenShirts_AllenSolly_20459562MenBlueTartanCheckedSlimFitPureCottonCasualShirt_5.jpg",
    "https://datalabs.siva3.io/images/MenShirts_AllenSolly_20459562MenBlueTartanCheckedSlimFitPureCottonCasualShirt_1.jpg"
  ]
}
"""
from dataclasses import dataclass, field
from hashlib import md5
import datetime
from typing import Any
from .category import Category
from ..clients.data_client import data_client
from tqdm import tqdm
@dataclass
class BaseProduct:
    vendor_sku: str # product_sku
    vendor_id: str # catalog_id
    urls: list[str] # all_urls
    metadata: dict[str, Any] # all other fields


class Product(BaseProduct):
    def __init__(self, document):
        vendor_sku = document['product_sku']
        vendor_id = document['catalog_id']
        self.urls = document['all_urls']
        self.metadata = {k: v for k, v in document.items() if k not in ['product_sku', 'catalog_id', 'l0', 'l1', 'l2', 'l3', 'l4', 'all_urls','_id','url_uid']}
        self.L0 = Category(document['l0'], 'l0',{'name': document['l0'], 'domain': 'l0'})
        self.L1 = Category(document['l1'], 'l1',{'name': document['l0'], 'domain': 'l0'})
        self.L2 = Category(document['l2'], 'l2',{'name': document['l1'], 'domain': 'l1'})
        self.L3 = Category(document['l3'], 'l3',{'name': document['l2'], 'domain': 'l2'})
        self.L4 = Category(document['l4'], 'l4',{'name': document['l3'], 'domain': 'l3'})
        hash = data_client.catalogStore.doc_text.find_one({"_id":document['_id']})['md5_hash'] # type: ignore
        if hash:
            self.hash = hash
        indexed = data_client.catalogStore.indexed_products.find_one({"hash":self.hash}) # type: ignore
        self.id = indexed['_id'] # type: ignore
        self.sellers = indexed['sellers'] # type: ignore
        tmp = data_client.catalogStore['indexed_payload'].find_one({"hash":self.hash})
        if tmp:
            pass
        else:
            vec_payload = {k:v for k,v in self.metadata.items()}
            vec_payload['L0'] = document['l0']
            vec_payload['L1'] = document['l1']
            vec_payload['L2'] = document['l2']
            vec_payload['L3'] = document['l3']
            vec_payload['L4'] = document['l4']
            vec_payload['hash'] = self.hash
            vec_payload['_id'] = self.id
            vec_payload['urls'] = self.urls
            vec_payload['sellers'] = ['_'.join(list(p.values())) for p in self.sellers]
            data_client.catalogStore['indexed_payload'].insert_one(
                vec_payload
            )
# Unique url_uids  only
            
url_uids = []

with tqdm(total=360000) as pbar:
  for product in data_client.catalogStore.catalogs.find():
    if product['url_uid'] not in url_uids:
      Product(product)
      pbar.update(1)




