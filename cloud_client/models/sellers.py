from dataclasses import dataclass, field
from hashlib import md5
import datetime

@dataclass
class Vendor:
    id: str # _id
    name: str # catalog_name
    url: str # catalog_url
    hash: str # hash
    products: list[dict[str,str]] # product_ids
    last_updated: datetime.datetime
    
    def __post_init__(self):
        self.product_count = len(self.products)


class iVendor(Vendor):
    def __init__(self, Vendor:Vendor):
        super().__init__(id=Vendor.id, name=Vendor.name, url=Vendor.url, hash=Vendor.hash, products=Vendor.products, last_updated=Vendor.last_updated)
        self.product_count = Vendor.product_count
    
    def process_products(self):
        pass
