import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.routes import analytics, classify, compare, conversations, documents, health, qa, search, summarize
from app.config import get_settings
from app.core.errors import register_exception_handlers
from app.db.base import Base
from app.db.fts import ensure_fts
from app.db.session import engine
from app.logging_config import configure_logging, new_request_id, request_id_ctx

settings = get_settings()
configure_logging(settings.log_level)
logger = logging.getLogger("app")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Import models so their tables are registered on Base.metadata before create_all.
    import app.models  # noqa: F401

    Base.metadata.create_all(bind=engine)
    ensure_fts(engine)
    logger.info("Database tables ensured at %s", settings.sqlite_path)
    yield
    logger.info("Application shutting down")


def create_app() -> FastAPI:
    app = FastAPI(
        title="AI Research & Knowledge Assistant",
        description="RAG backend for uploading, searching, and reasoning over technical documents.",
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def request_context_middleware(request: Request, call_next):
        request_id = new_request_id()
        token = request_id_ctx.set(request_id)
        start = time.perf_counter()
        try:
            response = await call_next(request)
        finally:
            request_id_ctx.reset(token)
        duration_ms = int((time.perf_counter() - start) * 1000)
        response.headers["X-Request-ID"] = request_id
        logger.info("%s %s -> %s (%sms)", request.method, request.url.path, response.status_code, duration_ms)
        return response

    register_exception_handlers(app)
    app.include_router(health.router)
    app.include_router(documents.router)
    app.include_router(search.router)
    app.include_router(analytics.router)
    app.include_router(qa.router)
    app.include_router(conversations.router)
    app.include_router(compare.router)
    app.include_router(summarize.router)
    app.include_router(classify.router)

    return app


app = create_app()
