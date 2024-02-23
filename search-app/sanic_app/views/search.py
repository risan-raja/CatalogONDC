import asyncio
from sanic.views import HTTPMethodView
from sanic import text
from sanic.response import json
from qdrant_client import models
import uuid
import orjson

# app = Sanic.get_app("ONDC_Index")


def process_filters(request) -> tuple[models.Filter | None, models.Filter | None]:
    filters = request.json.get("filters")
    search_text = request.json.get("search_text")
    must_only_filters = None
    filters_with_text = None
    must_filters = []
    if filters:
        for key, value in filters.items():
            # If the value is a list, then it is a match any condition
            # Multi select filters
            if isinstance(value, list):
                filter = models.FieldCondition(
                    key=key, match=models.MatchAny(any=value)
                )
                must_filters.append(filter)
            elif key == "mrp":
                # If the key is mrp, then it is a range condition
                filter = models.FieldCondition(
                    key=key, range=models.Range(gte=value[0], lte=value[1])
                )
                must_filters.append(filter)
            elif isinstance(value, str):
                # If the value is a string, then it is a match condition
                filter = models.FieldCondition(
                    key=key, match=models.MatchValue(value=value)
                )
                must_filters.append(filter)
        must_only_filters = models.Filter(must=must_filters)  # type: ignore
    if search_text:
        text_match_filters = [
            models.FieldCondition(
                key="product_name",
                match=models.MatchText(text=search_text),
            ),
            models.FieldCondition(
                key="short_product_description",
                match=models.MatchText(text=search_text),
            ),
        ]
        filters_with_text = models.Filter(should=text_match_filters, must=must_filters)  # type: ignore
    return must_only_filters, filters_with_text


def create_vector_models(query_embedding):
    dense_vector = models.NamedVector(
        name="dense",
        vector=query_embedding["dense"],
    )
    sparse_vector = models.NamedSparseVector(
        name="sparse",
        vector=models.SparseVector(
            indices=query_embedding["sparse"]["indices"],
            values=query_embedding["sparse"]["values"],
        ),
    )
    return dense_vector, sparse_vector


class SearchView(HTTPMethodView):
    async def post(self, request):
        """
        Search object
        {
            "search_text": "string",
            "filters":{
                "key1": "string"| "list",
                "key2": "string"| "list",
                ....
            },
            limit: "int",
            offset: "int"|"string"
        }
        """
        query_embedding = request.ctx.query_embedding
        pure_structured_query = False
        if query_embedding is None:
            pure_structured_query = True
        else:
            query_embedding = query_embedding[0]
            # Construct Vectors
            dense_vector, sparse_vector = create_vector_models(query_embedding)
        qclient = request.app.ctx.vector_db
        # Process All the filters
        must_only_filters, filters_with_text = process_filters(request)
        # Process Limit
        if request.json.get("limit"):
            limit = request.json["limit"]
        else:
            limit = 20
        # Process Offset
        if request.ctx.offset:
            offset_values = request.ctx.offset
        else:
            offset_values = {
                "structured_search": 0,
                "dense_vector_search": 0,
                "sparse_vector_search": 0,
            }
        # Structured search query
        if filters_with_text:
            structured_filter = filters_with_text
        else:
            structured_filter = must_only_filters
        structured_search = qclient.scroll(
            collection_name="ondc-index",
            scroll_filter=structured_filter,
            limit=limit,
            offset=offset_values["structured_search"],
            with_payload=['product_name'],
            with_vectors=False,
        )
        if pure_structured_query:
            # Fetch Structured Search Results
            structured_search_results, structured_offset = await structured_search
            # Check if the offset is already present
            if request.ctx.offset:
                unique_offset = request.json.get("offset")
            else:
                unique_offset = str(uuid.uuid4())
            redis = request.app.ctx.redis
            # Store the offset in the redis
            await redis.set(
                f"{unique_offset}:offset",
                orjson.dumps({"structured_search": structured_offset}),
            )
            results = {
                "results": [point.payload for point in structured_search_results],
                "offset": unique_offset,
            }
            return json(results)
        # Dense vector search query
        dense_vector_search = qclient.search(
            collection_name="ondc-index",
            query_vector=dense_vector,
            limit=limit,
            offset=offset_values["dense_vector_search"],
            with_payload=['product_name'],
            with_vectors=False,
        )
        # Sparse vector search query
        sparse_vector_search = qclient.search(
            collection_name="ondc-index",
            query_vector=sparse_vector,
            limit=limit,
            offset=offset_values["sparse_vector_search"],
            with_payload=['product_name'],
            with_vectors=False,
        )
        all_searches = [structured_search, dense_vector_search, sparse_vector_search]
        all_results = await asyncio.gather(*all_searches)
        structured_search_results, structured_offset = all_results[0]
        offset_values["structured_search"] = structured_offset
        offset_values["dense_vector_search"] += limit
        offset_values["sparse_vector_search"] += limit
        # Check if the offset is already present
        if request.ctx.offset:
            unique_offset = request.json.get("offset")
        else:
            unique_offset = str(uuid.uuid4())
        redis = request.app.ctx.redis
        # Store the offset in the redis
        await redis.set(
            f"{unique_offset}:offset",
            orjson.dumps(offset_values),
        )
        # TODO: Perform Rank Fusion here
        results ={
            "results": [point.payload for point in structured_search_results+all_results[1]+all_results[2]],
            "offset": unique_offset,
        }
        print("******Structured Query Results*********\n")
        for point in structured_search_results:
            print("*********\n")
            print(point)
            print("*********\n\n")
        print("******Dense Vector Query Results*********\n")
        for point in all_results[1]:
            print("*********\n")
            print(point)
            print("*********\n\n")
        print("******Sparse Vector Query Results*********\n")
        for point in all_results[2]:
            print("*********\n")
            print(point)
            print("*********\n\n")
        return text("Done")
