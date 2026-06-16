"""
Production-Ready FastAPI + LangGraph RAG Application

Wires together:
- Security pipeline (input sanitization, PII masking)
- Response caching
- Rate Limiting (slowapi)
- LangGraph RAG agent (retrieve → LLM → fallback)
- Structured JSON logging + metrics
- LangSmith tracing
- Health checks
"""

from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from langsmith import traceable
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app.agent import ProductionAgent
from app.cache import ResponseCache
from app.config import get_settings
from app.models import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    MetricsResponse,
)
from app.monitoring import MetricsCollector, RequestTimer, get_logger
from app.security import SecurityPipeline

load_dotenv()

logger = get_logger("production-api")

# Globals (initialised in lifespan)

security: SecurityPipeline
cache: ResponseCache
metrics: MetricsCollector
agent: ProductionAgent


# Lifespan


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Modern FastAPI lifespan pattern (replaces @app.on_event).

    All heavy initialisation happens here so:
    1. The app is ready before accepting traffic.
    2. Components are shared across all requests (not re-created per request).
    3. Shutdown is clean (logs final metrics summary).
    """
    global security, cache, metrics, agent

    settings = get_settings()

    logger.info(
        "Starting production API...",
        extra={
            "extra_data": {
                "environment": settings.app_env,
                "primary_model": settings.primary_model,
                "tracing_enabled": settings.langchain_tracing_v2,
            }
        },
    )

    security = SecurityPipeline()
    cache = ResponseCache(ttl_seconds=settings.cache_ttl_seconds)
    metrics = MetricsCollector()
    agent = ProductionAgent()

    logger.info("All components initialised. Ready to serve requests.")

    yield  # <- app is running here

    logger.info(
        "Shutting down...",
        extra={"extra_data": metrics.summary},
    )


# App

limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title="Production LangGraph RAG API",
    description="Production-ready Chat API with RAG, Security, Caching, and Observability.",
    version="1.0.0",
    lifespan=lifespan,
)
app.state.limiter = limiter

# WHY this line matters:
# Without registering the exception handler, hitting the rate limit raises
# an unhandled RateLimitExceeded exception -> 500 Internal Server Error.
# With it, slowapi returns a proper 429 Too Many Requests with a
# Retry-After header.
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# Endpoints


@app.post("/chat", response_model=ChatResponse)
@limiter.limit(get_settings().rate_limit)
@traceable(name="chat_endpoint")
async def chat(request: Request, body: ChatRequest):
    """
    Main chat endpoint.

    Flow:
        1. Security check    -> injection detection + PII masking on input
        2. Cache lookup      -> return instantly if seen before
        3. RAG agent         -> retrieve → LLM (with fallback + error handler)
        4. Output validation -> PII masking + harmful-content check on output
        5. Cache store       -> persist for future identical queries
        6. Log + metrics     -> record latency, tokens, cache miss
    """
    security_notes: list[str] = []

    with RequestTimer() as timer:
        # 1. Security
        is_allowed, cleaned_message, notes = security.check_input(body.message)
        security_notes.extend(notes)

        if not is_allowed:
            logger.warning(
                "Request blocked by security",
                extra={
                    "extra_data": {
                        "reason": notes,
                        "thread_id": body.thread_id,
                    }
                },
            )
            metrics.record_request(latency_ms=timer.elapsed_ms, error=True)
            raise HTTPException(
                status_code=400,
                detail="Your message was blocked by our security filters.",
            )

        # 2. Cache lookup
        cached_response = cache.get(cleaned_message)
        if cached_response is not None:
            metrics.record_request(latency_ms=timer.elapsed_ms, cache_hit=True)
            logger.info(
                "Cache hit",
                extra={"extra_data": {"thread_id": body.thread_id}},
            )
            return ChatResponse(
                response=cached_response,
                thread_id=body.thread_id,
                model_used="cache",
                cached=True,
                proccesing_time_ms=round(timer.elapsed_ms, 2),
            )

        # 3. RAG Agent
        try:
            result = agent.invoke(cleaned_message)
        except Exception as e:
            logger.error(
                f"Agent invocation failed: {e}",
                extra={
                    "extra_data": {
                        "thread_id": body.thread_id,
                        "error": str(e),
                    }
                },
            )
            metrics.record_request(latency_ms=timer.elapsed_ms, error=True)
            raise HTTPException(
                status_code=500,
                detail="An error occurred while processing your request.",
            )

        response_text = result["response"]
        model_used = result["model_used"]

        # 4. Output validation
        validated_response, output_warnings = security.check_output(response_text)
        security_notes.extend(output_warnings)

        # 5. Cache store
        cache.set(cleaned_message, validated_response)

    # 6. Metrics + log
    # Rough token estimate (actual tokens come from the LLM response object
    # in a production setup - wire up via LangSmith callbacks for accuracy).
    input_tokens = int(len(cleaned_message.split()) * 1.3)
    output_tokens = int(len(validated_response.split()) * 1.3)

    metrics.record_request(
        latency_ms=timer.elapsed_ms,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_hit=False,
    )

    logger.info(
        "Request completed",
        extra={
            "extra_data": {
                "thread_id": body.thread_id,
                "model_used": model_used,
                "latency_ms": round(timer.elapsed_ms, 2),
                "security_notes": security_notes,
            }
        },
    )

    return ChatResponse(
        response=validated_response,
        thread_id=body.thread_id,
        model_used=model_used,
        cached=False,
        proccesing_time_ms=round(timer.elapsed_ms, 2),
    )


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check for Docker / Kubernetes liveness probes."""
    settings = get_settings()

    checks = {
        "agent": agent is not None,
        "security": security is not None,
        "cache": cache is not None,
    }

    return HealthResponse(
        status="healty" if all(checks.values()) else "degraded",
        environment=settings.app_env,
        checks=checks,
    )


@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    """Prometheus-style metrics for monitoring dashboards."""
    return MetricsResponse(**metrics.summary)


@app.get("/cache/stats")
async def cache_stats():
    """Cache hit/miss statistics."""
    return cache.stats
