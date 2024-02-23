import orjson

# app = Sanic.get_app("ONDC_Index")


async def embed_query(request):
    if request.json:
        if "search_text" in request.json:
            redis = request.app.ctx.redis
            search_text = request.json["search_text"]
            # async with redis.conn as r:
            red = await redis.get(f'{search_text}:query_embedding')
            if red:
                request.ctx.query_embedding = orjson.loads(red)
            else:
                query_embedding = await request.app.ctx.embedding_service.async_infer([search_text])
                request.ctx.query_embedding = query_embedding
                await redis.set(f'{search_text}:query_embedding', orjson.dumps(request.ctx.query_embedding))
        else:
            request.ctx.query_embedding = None

