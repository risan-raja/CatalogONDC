import orjson

async def fetch_offsets(request):
    """
    Store the offsets of the multiple search requests
    """
    request.ctx.offset = None
    if request.json:
        if "offset" in request.json:
            redis = request.app.ctx.redis
            offset = request.json.get("offset")
            if isinstance(offset, str) and len(offset) > 0:
                red = await redis.get(f'{offset}:offset')
                if red:
                    request.ctx.offset = orjson.loads(red)