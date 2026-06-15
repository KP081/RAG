"""
Production-Ready FastAPI + LangGraph Application
Wires together:
- Security pipeline (input sanitization, PII masking)
- Response caching
- Rate Limiting (slowapi)
- LangGraph agent (with retries + fallback)
- Structured logging + metrics
- LangSmith tracing
- Health checks
"""

import time
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from langsmith import traceable
from dotenv import load_dotenv

from app.config import get_settings
from app.models import (
    ChatRequest, ChatResponse,
    HealthResponse, MetricsResponse, ErrorResponse
)
from app.security import SecurityPipeline
from app.cache import ResponseCache
from app.agent import ProductionAgent
from app.monitoring import get_logger, MetricsCollector # RequestTimer()

load_dotenv()

# lifespan (startup/shutdown)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Initialize all componets on startup, clean up on shutdown.
    This is modern FastApi pattern (replace @app.on_event).
    """
    global security, cache, metrics, agent
    
    settings = get_settings()
    
    # logger. info ("Starting production API...", extra={"extra_data": {
    #     "environment": settings.app_env,
    #     "primary model": settings.primary_model,
    #     "tracing_enabled": settings.langchain_tracing_v2,
    # }})
    
    # initialize componets
    security = SecurityPipeline()
    cache = ResponseCache(ttl_seconds=settings.cache_ttl_seconds)
    metrics = MetricsCollector()
    agent = ProductionAgent()
    
    # logger.info("All conponets are initialized, Ready to serve requests.")
    
    yield # app is running
    
    # logger.info("Shutting down...", extra={"extra_data": metrics.summary})
    
    
# Rate limiter setup
limiter = Limiter(key_func=get_remote_address)

# FastApi app
app = FastAPI(
    title="Production langGraph API",
    description="A production ready ChatApi with Security, Cacheing, and Observability.",
    version="1.0.0",
    lifespan=lifespan
)
app.state.limiter = limiter


# Endpoints

@app.post("/chat", response_model=ChatResponse)
@limiter.limit(get_settings().rate_limit)
@traceable(name="chat_endpoint")
async def chat(request: Request, body: ChatRequest):
    """
    Main chat endpoint.
    
    Flow:
    1. Security Check (Injection + PII masking)
    2. cache lookup
    3. Langgrpah agent invoke (if cache miss)
    4. Output validation
    5. Cache store
    6. Return response
    """
    # with RequestTimer() as timer:
        security_notes = []
        
        # step 1 security check
        is_allowed, cleand_message, notes = security.check_input(body.message)
        security_notes.extend(notes)
        
        if not is_allowed:
            # logger.warning("Request blocked by security", extra={f"extra_data": {
            #     "reason": notes,
            #     "thread_id": body. thread_id,
            # }})
            metrics.record_request(latency_ms=0, error=True)
            raise HTTPException(
                status_code=400,
                detail="Your message blocked by our security filters."
            )
            
        # step 2 cache lookup
        cached_response = cache.get(cleand_message)
        if cached_response is not None:
            metrics.record_request(latency_ms=0, cache_hit=True)
            # logger.info("Cache hit", extra={"extra_data": {
            #     "thred_id": body.thread_id
            # }})
            return ChatResponse(
                response=cached_response,
                thread_id=body.thread_id,
                model_used="cache",
                cached=True,
                proccesing_time_ms=0
            )
            
        # step 3 invoke LangGraph Agent
        try:
            result = agent.invoke(cleand_message)
        except Exception as e:
            # logger. error (f"Agent invocation failed: {e}", extra={"extra_data": {
            #     "thread_id": body.thread_id,
            #     "error": str(e),
            # }})
            metrics. record_request (latency_ms=0, error=True)
            raise HTTPException(
                status_code=500,
                detail="An error occurred while processing your request."
            )
            
        response_text = result["response"]
        model_used = result["model_used"]
        
        # step 4 output validation
        validated_response, output_warnings = security.check_output(response_text)
        security_notes.extend(output_warnings)
        
        # step 5 cache store
        cache.set(cleand_message, validated_response)
        
    # step 6 Log and Record Metrics
    input_tokens = int(len(cleand_message.split()) * 1.3)
    output_tokens = int(len(validated_response.split()) * 1.3)

    # metrics.record_request(
    #     latency_ms=timer.elapsed_ms,
    #     input_tokens=input_tokens,
    #     output_tokens=output_tokens,
    #     cache_hit=False
    # )
    
    # logger. info("Request completed", extra={"extra_data": {
    #     "thread_id": body.thread_id,
    #     "model_used": model_used,
    #     "latency_ms": round (timer. elapsed_ms, 2),
    # }})
    
    return ChatResponse(
        response=validated_response,
        thread_id=body.thread_id,
        model_used=model_used,
        cached=False,
        proccesing_time_ms=round(timer.elapsed_ms, 2)
    )
    
    
@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check for docker/kubernates."""
    settings = get_settings()
    
    checks = {
        "agent": agent is not None,
        "security": security is not None,
        "cache": cache is not None
    }
    
    all_healthy = all(checks.values())
    
    return HealthResponse(
        status="healty" if all_healthy else "degraded",
        environment=settings.app_env,
        checks=checks
    )
    
    
@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    """Metrics for monitoring dashboards."""
    summary = metrics.summary
    return MetricsResponse(**summary)


@app.get("/cache/stats")
async def cache_stats():
    """cache performance statistics."""
    return cache.stats