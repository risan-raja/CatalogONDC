from .drivers.vector_index import qclient
from .drivers.embedding import QueryEmbedding
from .filter_enums.enums import L2_Enums, L3_Enums, L4_Enums
from .drivers.redis_cache import redis
from .middleware.embed_query import embed_query
from .middleware.offset import fetch_offsets
from .views.search import SearchView
from sanic import Sanic


def create_app():
    app = Sanic("ONDC_Index")
    app.ctx.embedding_service = QueryEmbedding()
    app.ctx.l2_enums = L2_Enums()
    app.ctx.l3_enums = L3_Enums()
    app.ctx.l4_enums = L4_Enums()
    app.config.update(
        {
            "REDIS": "redis://localhost:6379/0",
        }
    )
    redis.init_app(app)
    app.ctx.vector_db = qclient
    app.register_middleware(embed_query, "request",priority=3)
    app.register_middleware(fetch_offsets, "request",priority=2)
    # app.register_middleware(fuse_rank, "response")
    app.add_route(SearchView.as_view(), "/search")
    return app