from .drivers.vector_index import qclient
from .drivers.embedding import QueryEmbedding
from .drivers.redis_cache import redis
from sanic import Sanic


def create_app():
    app = Sanic("ONDC_Index")
    app.ctx.embedding_service = QueryEmbedding()
    app.config.update(
        {
            "REDIS": "redis://localhost:6379/0",
        }
    )
    redis.init_app(app)
    app.ctx.vector_db = qclient
    return app