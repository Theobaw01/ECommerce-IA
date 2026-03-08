"""
============================================
ECommerce-IA — Monitoring & Structured Logging
============================================
Module de monitoring production pour l'API FastAPI.

Fonctionnalités :
- Logging structuré JSON (analyse par ELK / CloudWatch / Datadog)
- Middleware de suivi des requêtes (latence, status, route)
- Métriques de performance IA (classification, NLP, recommandation)
- Health check avancé avec état des composants
- Alertes seuils de performance

Auteur : BAWANA Théodore — Projet SAHELYS
============================================
"""

import time
import json
import logging
import traceback
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, Optional
from collections import defaultdict
from functools import wraps

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware


# ============================================
# 1. Structured JSON Logger
# ============================================
class StructuredFormatter(logging.Formatter):
    """Formatter JSON structuré pour la production."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Ajouter les champs extra s'ils existent
        for key in ("request_id", "method", "path", "status_code",
                     "duration_ms", "client_ip", "user_agent",
                     "model", "accuracy", "latency_ms", "error_type"):
            if hasattr(record, key):
                log_entry[key] = getattr(record, key)

        # Ajouter l'exception si présente
        if record.exc_info and record.exc_info[0]:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": traceback.format_exception(*record.exc_info),
            }

        return json.dumps(log_entry, ensure_ascii=False, default=str)


def setup_structured_logging(
    level: str = "INFO",
    log_file: Optional[str] = "logs/api.jsonl"
) -> logging.Logger:
    """Configure le logging structuré JSON.

    Args:
        level: Niveau de log (DEBUG, INFO, WARNING, ERROR)
        log_file: Chemin du fichier de log (None = console uniquement)

    Returns:
        Logger configuré
    """
    logger = logging.getLogger("ecommerce_ia")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.handlers.clear()

    formatter = StructuredFormatter()

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (optionnel)
    if log_file:
        import os
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


# ============================================
# 2. Métriques de Performance
# ============================================
class PerformanceMetrics:
    """Collecteur de métriques de performance en mémoire."""

    def __init__(self):
        self.reset()

    def reset(self):
        """Réinitialise toutes les métriques."""
        self._request_count = 0
        self._error_count = 0
        self._latencies: Dict[str, list] = defaultdict(list)
        self._status_codes: Dict[int, int] = defaultdict(int)
        self._ai_metrics: Dict[str, list] = defaultdict(list)
        self._start_time = time.time()

    def record_request(self, path: str, status_code: int, duration_ms: float):
        """Enregistre une requête HTTP."""
        self._request_count += 1
        self._status_codes[status_code] += 1
        self._latencies[path].append(duration_ms)
        if status_code >= 400:
            self._error_count += 1

    def record_ai_inference(self, model: str, latency_ms: float,
                            success: bool = True, **kwargs):
        """Enregistre une inférence IA.

        Args:
            model: Nom du modèle (classification, nlp, recommendation, chatbot)
            latency_ms: Latence en ms
            success: Si l'inférence a réussi
            **kwargs: Métriques additionnelles (accuracy, confidence, etc.)
        """
        self._ai_metrics[model].append({
            "latency_ms": latency_ms,
            "success": success,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **kwargs
        })

    def get_summary(self) -> Dict[str, Any]:
        """Retourne un résumé des métriques."""
        uptime = time.time() - self._start_time

        # Calcul des latences par route
        route_stats = {}
        for path, latencies in self._latencies.items():
            if latencies:
                route_stats[path] = {
                    "count": len(latencies),
                    "avg_ms": round(sum(latencies) / len(latencies), 2),
                    "min_ms": round(min(latencies), 2),
                    "max_ms": round(max(latencies), 2),
                    "p95_ms": round(sorted(latencies)[int(len(latencies) * 0.95)], 2)
                    if len(latencies) >= 20 else None,
                }

        # Calcul des métriques IA
        ai_stats = {}
        for model, records in self._ai_metrics.items():
            latencies = [r["latency_ms"] for r in records]
            successes = sum(1 for r in records if r.get("success", True))
            ai_stats[model] = {
                "total_inferences": len(records),
                "success_rate": round(successes / len(records) * 100, 1) if records else 0,
                "avg_latency_ms": round(sum(latencies) / len(latencies), 2) if latencies else 0,
                "max_latency_ms": round(max(latencies), 2) if latencies else 0,
            }

        return {
            "uptime_seconds": round(uptime, 1),
            "total_requests": self._request_count,
            "total_errors": self._error_count,
            "error_rate_percent": round(
                self._error_count / self._request_count * 100, 2
            ) if self._request_count > 0 else 0,
            "status_codes": dict(self._status_codes),
            "routes": route_stats,
            "ai_models": ai_stats,
        }


# Instance globale
metrics = PerformanceMetrics()


# ============================================
# 3. Middleware de Requêtes
# ============================================
class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware pour logger chaque requête avec sa latence."""

    SKIP_PATHS = {"/health", "/metrics", "/favicon.ico", "/docs", "/redoc", "/openapi.json"}

    def __init__(self, app, logger: Optional[logging.Logger] = None):
        super().__init__(app)
        self.logger = logger or logging.getLogger("ecommerce_ia")

    async def dispatch(self, request: Request, call_next) -> Response:
        # Générer un ID unique pour la requête
        request_id = str(uuid.uuid4())[:8]
        request.state.request_id = request_id
        request.state.start_time = time.time()

        # Traiter la requête
        try:
            response = await call_next(request)
        except Exception as exc:
            duration_ms = (time.time() - request.state.start_time) * 1000
            self.logger.error(
                f"Erreur non gérée : {exc}",
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "duration_ms": round(duration_ms, 2),
                    "error_type": type(exc).__name__,
                },
                exc_info=True,
            )
            metrics.record_request(request.url.path, 500, duration_ms)
            raise

        # Calculer la durée
        duration_ms = (time.time() - request.state.start_time) * 1000

        # Enregistrer les métriques
        metrics.record_request(request.url.path, response.status_code, duration_ms)

        # Logger (sauf endpoints de monitoring)
        if request.url.path not in self.SKIP_PATHS:
            log_level = logging.WARNING if response.status_code >= 400 else logging.INFO
            self.logger.log(
                log_level,
                f"{request.method} {request.url.path} → {response.status_code} ({duration_ms:.1f}ms)",
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": response.status_code,
                    "duration_ms": round(duration_ms, 2),
                    "client_ip": request.client.host if request.client else "unknown",
                    "user_agent": request.headers.get("user-agent", ""),
                },
            )

        # Ajouter les headers de traçabilité
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time"] = f"{duration_ms:.2f}ms"

        return response


# ============================================
# 4. Décorateur pour tracer les inférences IA
# ============================================
def track_inference(model_name: str):
    """Décorateur pour tracer automatiquement les inférences IA.

    Usage:
        @track_inference("classification")
        async def classify_image(image):
            ...
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start = time.time()
            success = True
            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as exc:
                success = False
                raise
            finally:
                latency_ms = (time.time() - start) * 1000
                metrics.record_ai_inference(
                    model=model_name,
                    latency_ms=latency_ms,
                    success=success,
                )
        return wrapper
    return decorator


# ============================================
# 5. Health Check Avancé
# ============================================
def get_health_status() -> Dict[str, Any]:
    """Retourne l'état de santé détaillé du système.

    Returns:
        Dictionnaire avec le statut de chaque composant
    """
    import os

    components = {}

    # Vérifier les modèles IA
    model_paths = {
        "classification_cnn": "models/classification/efficientnet_b4_best.pth",
        "classification_vit": "models/classification/vit_base_best.pth",
        "recommendation_svd": "models/recommendation/svd_model.pkl",
        "faiss_index": "models/search/faiss_index.bin",
        "chatbot_chromadb": "models/chatbot/chroma_db",
    }
    for name, path in model_paths.items():
        full_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), path)
        components[name] = "loaded" if os.path.exists(full_path) else "not_found"

    # Vérifier le NLP engine
    try:
        from src.nlp_engine import NLPEngine
        engine = NLPEngine()
        test = engine.analyser("bonjour")
        components["nlp_engine"] = "healthy" if test.get("intent") else "degraded"
    except Exception as e:
        components["nlp_engine"] = f"error: {str(e)[:50]}"

    # Statut global
    healthy_count = sum(1 for v in components.values() if v in ("loaded", "healthy"))
    total = len(components)
    overall = "healthy" if healthy_count == total else (
        "degraded" if healthy_count > total // 2 else "unhealthy"
    )

    return {
        "status": overall,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "components": components,
        "metrics_summary": {
            "uptime_seconds": round(time.time() - metrics._start_time, 1),
            "total_requests": metrics._request_count,
            "error_rate": f"{metrics.get_summary()['error_rate_percent']}%",
        },
    }


# ============================================
# 6. Alertes de Performance
# ============================================
class PerformanceAlerts:
    """Système d'alertes basé sur des seuils."""

    DEFAULT_THRESHOLDS = {
        "max_latency_ms": 5000,       # Alerte si requête > 5s
        "max_error_rate": 10.0,       # Alerte si > 10% d'erreurs
        "max_ai_latency_ms": 10000,   # Alerte si inférence > 10s
        "min_ai_success_rate": 90.0,  # Alerte si succès < 90%
    }

    def __init__(self, thresholds: Optional[Dict] = None):
        self.thresholds = thresholds or self.DEFAULT_THRESHOLDS
        self.alerts: list = []

    def check(self) -> list:
        """Vérifie les seuils et retourne les alertes actives."""
        self.alerts.clear()
        summary = metrics.get_summary()

        # Taux d'erreur global
        if summary["error_rate_percent"] > self.thresholds["max_error_rate"]:
            self.alerts.append({
                "level": "critical",
                "type": "high_error_rate",
                "message": f"Taux d'erreur élevé : {summary['error_rate_percent']}%",
                "threshold": self.thresholds["max_error_rate"],
                "current": summary["error_rate_percent"],
            })

        # Latence IA
        for model, stats in summary.get("ai_models", {}).items():
            if stats["max_latency_ms"] > self.thresholds["max_ai_latency_ms"]:
                self.alerts.append({
                    "level": "warning",
                    "type": "high_ai_latency",
                    "model": model,
                    "message": f"Latence IA élevée pour {model}: {stats['max_latency_ms']}ms",
                    "threshold": self.thresholds["max_ai_latency_ms"],
                    "current": stats["max_latency_ms"],
                })

            if stats["success_rate"] < self.thresholds["min_ai_success_rate"]:
                self.alerts.append({
                    "level": "critical",
                    "type": "low_ai_success_rate",
                    "model": model,
                    "message": f"Taux de succès bas pour {model}: {stats['success_rate']}%",
                    "threshold": self.thresholds["min_ai_success_rate"],
                    "current": stats["success_rate"],
                })

        return self.alerts


# Instance globale
alerts = PerformanceAlerts()
