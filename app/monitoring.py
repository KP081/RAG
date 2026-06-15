"""
Monitoring & Structured Logging

Production-grade metrics collection and JSON logging.
"""

import json
import logging
import time

from datetime import datetime, timezone
from functools import wraps
from typing import Any, Callable


# JSON LOGGER


class JSONFormatter(logging.Formatter):
    """Format log records as JSON."""

    def format(self, record: logging.LogRecord) -> str:
        log_obj = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
        }

        if hasattr(record, "extra_data"):
            log_obj.update(record.extra_data)

        return json.dumps(log_obj)


def get_logger(name: str = "production-api") -> logging.Logger:
    """Create structured JSON logger."""

    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(JSONFormatter())

        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    return logger


# METRICS COLLECTOR


class MetricsCollector:
    """
    Collects application metrics.

    In production:
        Prometheus
        OpenTelemetry
        Datadog
        New Relic
    """

    def __init__(self):
        self._request_total = 0
        self._errors_total = 0

        self._latency_sum = 0.0
        self._latency_count = 0

        self._tokens_input = 0
        self._tokens_output = 0

        self._cache_hits = 0
        self._cache_misses = 0

    def record_request(
        self,
        latency_ms: float,
        input_tokens: int = 0,
        output_tokens: int = 0,
        error: bool = False,
        cache_hit: bool = False,
    ) -> None:
        """
        Record request metrics.
        """

        self._request_total += 1

        if error:
            self._errors_total += 1

        self._latency_sum += latency_ms
        self._latency_count += 1

        self._tokens_input += input_tokens
        self._tokens_output += output_tokens

        if cache_hit:
            self._cache_hits += 1
        else:
            self._cache_misses += 1

    @property
    def metrics(self) -> dict[str, Any]:
        """
        Return current metrics snapshot.
        """

        avg_latency = (
            self._latency_sum / self._latency_count if self._latency_count else 0.0
        )

        cache_total = self._cache_hits + self._cache_misses

        cache_hit_rate = self._cache_hits / cache_total if cache_total else 0.0

        error_rate = (
            self._errors_total / self._request_total if self._request_total else 0.0
        )

        return {
            "requests_total": self._request_total,
            "errors_total": self._errors_total,
            "error_rate": round(error_rate, 4),
            "avg_latency_ms": round(avg_latency, 2),
            "input_tokens_total": self._tokens_input,
            "output_tokens_total": self._tokens_output,
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "cache_hit_rate": round(cache_hit_rate, 4),
        }


# GLOBALS

logger = get_logger()

metrics = MetricsCollector()


# DECORATOR

def monitor(
    fn: Callable,
) -> Callable:
    """
    Automatically monitor requests.

    Measures:
        - latency
        - success/failure
        - logs
    """

    @wraps(fn)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()

        try:
            result = fn(*args, **kwargs)

            latency_ms = (time.perf_counter() - start) * 1000

            metrics.record_request(
                latency_ms=latency_ms,
                error=False,
            )

            logger.info(
                "request_success",
                extra={
                    "extra_data": {
                        "latency_ms": round(latency_ms, 2),
                        "function": fn.__name__,
                    }
                },
            )

            return result

        except Exception as exc:
            latency_ms = (time.perf_counter() - start) * 1000

            metrics.record_request(
                latency_ms=latency_ms,
                error=True,
            )

            logger.error(
                "request_failed",
                extra={
                    "extra_data": {
                        "latency_ms": round(latency_ms, 2),
                        "function": fn.__name__,
                        "error": str(exc),
                    }
                },
            )

            raise

    return wrapper
