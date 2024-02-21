from .drivers.vector_index import qclient
from .drivers.embedding import QueryEmbedding
from sanic import Sanic


def create_app():
    app = Sanic("ONDC_Index")
    app.ctx.embedding_service = QueryEmbedding()
    app.ctx.vector_db = qclient
    return app