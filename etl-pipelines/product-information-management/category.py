from dataclasses import dataclass, field
from uuid import uuid4
from typing import Any, Optional
from enum import Enum
from .clients.data_client import data_client
from functools import lru_cache


@dataclass
class Category:
    name: str
    domain: str
    parent: dict[str, str]

    def __post_init__(self):
        domain_map = {
            "l0":0,
            "l1":1,
            "l2":2,
            "l3":3,
            "l4":4,
        }
        category_index = data_client.catalogStore.category_index
        category_document = {
            "name": self.name,
            "domain": self.domain,
            "level": domain_map[self.domain],
            "parent":{
                "name": self.parent["name"],
                "domain": self.parent["domain"],
                "level": domain_map[self.parent["domain"]]
            }
        }
        self.level = domain_map[self.domain]
        self.parent_name = self.parent
        self.parent_domain = self.parent["domain"]
        self.parent_level = domain_map[self.parent["domain"]]
        tmp =  category_index.find_one({"name": self.name, "domain": self.domain, "parent.name": self.parent["name"]})
        if tmp:
            self.id = tmp["_id"]
        else:
            id = category_index.insert_one(category_document)
            self.id = id
