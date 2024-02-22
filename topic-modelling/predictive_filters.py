from qdrant_client import QdrantClient,models
from pymongo import MongoClient
from pprint import pprint
import polars as pl
from collections import defaultdict
import random
import json



qclient = QdrantClient(host="localhost",port=6333)
ondc_index = qclient.get_collection(collection_name="ondc-index")
schema = ondc_index.payload_schema
field_names = list(schema.keys())
field_types = [schema[field_name].data_type for field_name in field_names]
field_points = [schema[field_name].points for field_name in field_names]
df = pl.DataFrame({
    "field_name": field_names,
    "field_type": field_types,
    "field_points": field_points
})
non_keywords = df.filter(
    pl.col("field_type") != "keyword"
)['field_name'].to_list()

keyword_fields = list(set(field_names).difference(set(non_keywords)))
key_enum = {f:defaultdict(int) for f in keyword_fields} 

def update_key_enum(key_enum,points):
    keyword_fields = ['weave_type', 'cushioning', 'pockets', 'placket_length', 'sports_type', 'wiring', 'product_type', 'add_ons', 'L4', 'heel_type', 'waist_band', 'features', 'cuff', 'face_shape', 'tshirt_type', 'gender', 'blouse', 'cleats', 'technology', 'type_of_distress', 'shape', 'surface_styling', 'stone_type', 'frame_material', 'pocket_type', 'fabric_type', 'knit_or_woven', 'lehenga_fabric', 'front_styling', 'back', 'heel_height', 'top_length', 'hood', 'base_metal', 'design_type', 'toe_shape', 'brand_fit_name', 'type', 'pattern', 'fusion_wear', 'running_type', 'slit_detail', 'bundles_or_multipack_sets', 'top_design_styling', 'sleeve_length', 'padding', 'frame_colour', 'lens_type', 'coverage', 'brand', 'design_or_style_type', 'sports_bra_support', 'design', 'trend', 'hemline', 'bottom_length', 'season', 'sport_type', 'lehenga_stitch', 'pattern_size', 'set_content', 'ornamentation', 'seam', 'weave_pattern', 'style', 'players', 'dupatta', 'sleeve_styling', 'fly_type', 'border', 'center_front_open', 'case', 'lining_fabric', 'top_type', 'top_pattern', 'outsole_type', 'colour', 'pronation_for_running_shoes', 'closure', 'character', 'distress', 'straps', 'bottom_type', 'bottom_pattern', 'sport_team', 'length', 'surface_type', 'lehenga_lining_fabric', 'assorted', 'fastening', 'plating', 'stitch', 'bottom_fabric', 'neck', 'placket', 'number_of_pockets', 'shade', 'L3', 'size', 'print_or_pattern_type', 'closure_type', 'saree_fabric', 'transparency', 'kurta_fabric', 'effects', 'blouse_fabric', 'L1', 'top_shape', 'country_of_origin', 'sole_material', 'L2', 'fabric', 'top_hemline', 'waist_rise', 'top_fabric', 'dupatta_pattern', 'choli_stitch', 'distance', 'material_or_content', 'colour_family', 'fabric_finish', 'lens_colour', 'type_of_pleat', 'pattern_coverage', 'fade', 'ankle_height', 'occasion', 'dupatta_border', 'fit', 'dupatta_fabric', 'design_styling', 'collar', 'technique', 'stretch', 'lining', 'fastening_and_back_detail', 'arch_type', 'kurta_pattern', 'reversible', 'bottom_closure']    
    for point in points:
        payload = point.payload
        for k in payload:
            if k in keyword_fields:
                key_enum[k][payload[k]] += 1
    return key_enum


def get_all_points(collection_name:str,limit:int=10000):
    offset = 0
    points,offset = qclient.scroll(
        collection_name=collection_name,
        offset=offset,
        limit=limit
    )
    all_points = points
    while offset:
        points,offset = qclient.scroll(
            collection_name=collection_name,
            offset=offset,
            limit=limit,
            with_vectors=False
        )
        if offset is None:
            break
        else:
            all_points += points
    return all_points


all_points = get_all_points("ondc-index")
l3_categories = defaultdict(int)
l2_categories = defaultdict(int)
l4_categories = defaultdict(int)
# Load all L2 categories

for point in all_points:
    payload = point.payload
    l2_categories[payload['L2']] += 1  # type: ignore

l3_categories = {k:defaultdict(int) for k in l2_categories}
for point in all_points:
    payload = point.payload
    l3_categories[payload['L2']][payload['L3']] += 1 # type: ignore

l4_categories = {}
for l2 in l3_categories:
    l4_categories[l2] = {k:defaultdict(int) for k in l3_categories[l2]}

for point in all_points:
    payload = point.payload
    l4_categories[payload['L2']][payload['L3']][payload['L4']] += 1 # type: ignore

def get_attributes_in_category(all_points):
    l2_category_keys = {k: [] for k in l2_categories}
    l3_category_keys = {k: {k2:[] for k2 in l3_categories[k]} for k in l3_categories}
    l4_category_keys = {k: {k2: {k3:[] for k3 in l4_categories[k][k2]} for k2 in l4_categories[k]} for k in l4_categories}
    for point in all_points:
        payload = point.payload
        l2_category_keys[payload['L2']] += list(payload.keys())
        l2_category_keys[payload['L2']] = list(set(l2_category_keys[payload['L2']]))
        l3_category_keys[payload['L2']][payload['L3']] += list(payload.keys())
        l3_category_keys[payload['L2']][payload['L3']] = list(set(l3_category_keys[payload['L2']][payload['L3']]))
        l4_category_keys[payload['L2']][payload['L3']][payload['L4']] += list(payload.keys())
        l4_category_keys[payload['L2']][payload['L3']][payload['L4']] = list(set(l4_category_keys[payload['L2']][payload['L3']][payload['L4']]))
    return l2_category_keys,l3_category_keys,l4_category_keys

l2_category_keys,l3_category_keys,l4_category_keys = get_attributes_in_category(all_points)


other_common_attributes = [
    "wash_care",
    "seller_name_and_address",
    "supplier_name",
    "care_instructions",
    "query_contact",
    "bundles_or_multipack_sets",
    "product_code"
]
common_l2_attributes = []
tmp_category = random.choice(list(l2_category_keys.keys()))
keys = set(l2_category_keys[tmp_category])
for k in l2_category_keys:
    keys = keys.intersection(set(l2_category_keys[k]))
common_l2_attributes = list(keys)
common_l2_attributes = list(set(common_l2_attributes).union(set(other_common_attributes)))
common_l3_attributes = []
tmp_category = random.choice(list(l3_category_keys.keys()))
tmp_sub_category = random.choice(list(l3_category_keys[tmp_category].keys()))
keys = set(l3_category_keys[tmp_category][tmp_sub_category])
for k in l3_category_keys:
    for k2 in l3_category_keys[k]:
        keys = keys.intersection(set(l3_category_keys[k][k2]))
common_l3_attributes = list(keys)
common_l3_attributes = list(set(common_l3_attributes).union(set(common_l2_attributes)).union(set(other_common_attributes)))
common_l4_attributes = list(set(common_l3_attributes).union(set(common_l2_attributes)).union(set(other_common_attributes)))

# Remove common attributes from the category keys
for k in l2_category_keys:
    l2_category_keys[k] = list(set(l2_category_keys[k]).difference(set(common_l2_attributes)))

for k in l3_category_keys:
    for k2 in l3_category_keys[k]:
        l3_category_keys[k][k2] = list(set(l3_category_keys[k][k2]).difference(set(common_l3_attributes)))

for k in l4_category_keys:
    for k2 in l4_category_keys[k]:
        for k3 in l4_category_keys[k][k2]:
            l4_category_keys[k][k2][k3] = list(set(l4_category_keys[k][k2][k3]).difference(set(common_l3_attributes)))
            l4_category_keys[k][k2][k3] = list(set(l4_category_keys[k][k2][k3]).difference(set(common_l2_attributes)))


with open("l2_category_keys.json","w") as f:
    json.dump(l2_category_keys,f)

with open("l3_category_keys.json","w") as f:
    json.dump(l3_category_keys,f)

with open("l4_category_keys.json","w") as f:
    json.dump(l4_category_keys,f)


# category_keys = l4_category_keys["Men\'s Apparel"]["Topwear"]["T Shirts"]
l4_category_key_enums_master = {k: {k2: {k3:defaultdict(int) for k3 in l4_categories[k][k2]} for k2 in l4_categories[k]} for k in l4_categories}
for l2 in l4_category_keys:
    for l3 in l4_category_keys[l2]:
        for l4 in l4_category_keys[l2][l3]:
            category_keys = l4_category_keys[l2][l3][l4]
            category_key_enums = {k:defaultdict(int) for k in category_keys}
            filter = models.Filter(
                must=[
                    models.FieldCondition(key="L2",match=models.MatchValue(value=l2)),
                    models.FieldCondition(key="L3",match=models.MatchValue(value=l3)),
                    models.FieldCondition(key="L4",match=models.MatchValue(value=l4))
                ]
            )
            offset=0
            while offset is not None:
                points,offset = qclient.scroll(
                    collection_name="ondc-index",
                    scroll_filter= filter,
                    offset=offset,
                    limit=10000
                )
                for point in points:
                    payload = point.payload
                    for k in payload: # type: ignore
                        if k in category_keys:
                            category_key_enums[k][payload[k]] += 1 # type: ignore
            l4_category_key_enums_master[l2][l3][l4] = category_key_enums # type: ignore



l3_category_key_enums_master = {k: {k2:defaultdict(int) for k2 in l3_categories[k]} for k in l3_categories}
for l2 in l3_category_keys:
    for l3 in l3_category_keys[l2]:
        category_keys = l3_category_keys[l2][l3]
        category_key_enums = {k:defaultdict(int) for k in category_keys}
        filter = models.Filter(
            must=[
                models.FieldCondition(key="L2",match=models.MatchValue(value=l2)),
                models.FieldCondition(key="L3",match=models.MatchValue(value=l3))
            ]
        )
        offset=0
        while offset is not None:
            points,offset = qclient.scroll(
                collection_name="ondc-index",
                scroll_filter= filter,
                offset=offset,
                limit=10000
            )
            for point in points:
                payload = point.payload
                for k in payload: # type: ignore
                    if k in category_keys:
                        category_key_enums[k][payload[k]] += 1 # type: ignore
        l3_category_key_enums_master[l2][l3] = category_key_enums # type: ignore


l2_category_key_enums_master = {k:defaultdict(int) for k in l2_categories}
for l2 in l2_category_keys:
    category_keys = l2_category_keys[l2]
    category_key_enums = {k:defaultdict(int) for k in category_keys}
    filter = models.Filter(
        must=[
            models.FieldCondition(key="L2",match=models.MatchValue(value=l2))
        ]
    )
    offset=0
    while offset is not None:
        points,offset = qclient.scroll(
            collection_name="ondc-index",
            scroll_filter= filter,
            offset=offset,
            limit=10000
        )
        for point in points:
            payload = point.payload
            for k in payload: # type: ignore
                if k in category_keys:
                    category_key_enums[k][payload[k]] += 1 # type: ignore
    l2_category_key_enums_master[l2] = category_key_enums # type: ignore


with open("l2_category_key_enums_master.json","w") as f:
    json.dump(l2_category_key_enums_master,f)

with open("l3_category_key_enums_master.json","w") as f:
    json.dump(l3_category_key_enums_master,f)

with open("l4_category_key_enums_master.json","w") as f:
    json.dump(l4_category_key_enums_master,f)

