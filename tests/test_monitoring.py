"""
Tests pour le module de monitoring et structured logging.
"""

import json
import logging
import time
import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from src.monitoring import (
    StructuredFormatter,
    setup_structured_logging,
    PerformanceMetrics,
    PerformanceAlerts,
    get_health_status,
    metrics,
)


# ============================================
# Tests StructuredFormatter
# ============================================
class TestStructuredFormatter:
    """Tests du formatter JSON structuré."""

    def test_format_basic_log(self):
        """Un log basique produit du JSON valide."""
        formatter = StructuredFormatter()
        record = logging.LogRecord(
            name="test", level=logging.INFO,
            pathname="test.py", lineno=42,
            msg="Test message", args=(), exc_info=None,
        )
        output = formatter.format(record)
        parsed = json.loads(output)

        assert parsed["level"] == "INFO"
        assert parsed["message"] == "Test message"
        assert parsed["line"] == 42
        assert "timestamp" in parsed

    def test_format_with_extra_fields(self):
        """Les champs extra (request_id, method...) sont inclus."""
        formatter = StructuredFormatter()
        record = logging.LogRecord(
            name="test", level=logging.INFO,
            pathname="test.py", lineno=1,
            msg="Request", args=(), exc_info=None,
        )
        record.request_id = "abc123"
        record.method = "GET"
        record.path = "/api/test"
        record.status_code = 200
        record.duration_ms = 42.5

        output = formatter.format(record)
        parsed = json.loads(output)

        assert parsed["request_id"] == "abc123"
        assert parsed["method"] == "GET"
        assert parsed["path"] == "/api/test"
        assert parsed["status_code"] == 200
        assert parsed["duration_ms"] == 42.5

    def test_format_with_exception(self):
        """Les exceptions sont sérialisées dans le log."""
        formatter = StructuredFormatter()
        try:
            raise ValueError("test error")
        except ValueError:
            import sys
            record = logging.LogRecord(
                name="test", level=logging.ERROR,
                pathname="test.py", lineno=1,
                msg="Error", args=(), exc_info=sys.exc_info(),
            )

        output = formatter.format(record)
        parsed = json.loads(output)

        assert "exception" in parsed
        assert parsed["exception"]["type"] == "ValueError"
        assert "test error" in parsed["exception"]["message"]


# ============================================
# Tests PerformanceMetrics
# ============================================
class TestPerformanceMetrics:
    """Tests du collecteur de métriques."""

    def setup_method(self):
        self.metrics = PerformanceMetrics()

    def test_record_request(self):
        """Enregistrement de requêtes HTTP."""
        self.metrics.record_request("/api/test", 200, 15.5)
        self.metrics.record_request("/api/test", 200, 20.3)
        self.metrics.record_request("/api/error", 500, 100.0)

        summary = self.metrics.get_summary()
        assert summary["total_requests"] == 3
        assert summary["total_errors"] == 1
        assert summary["status_codes"][200] == 2
        assert summary["status_codes"][500] == 1

    def test_error_rate_calculation(self):
        """Calcul du taux d'erreur."""
        for _ in range(8):
            self.metrics.record_request("/ok", 200, 10.0)
        for _ in range(2):
            self.metrics.record_request("/err", 500, 10.0)

        summary = self.metrics.get_summary()
        assert summary["error_rate_percent"] == 20.0

    def test_route_latency_stats(self):
        """Statistiques de latence par route."""
        self.metrics.record_request("/api/classify", 200, 100.0)
        self.metrics.record_request("/api/classify", 200, 200.0)
        self.metrics.record_request("/api/classify", 200, 150.0)

        summary = self.metrics.get_summary()
        route = summary["routes"]["/api/classify"]
        assert route["count"] == 3
        assert route["avg_ms"] == 150.0
        assert route["min_ms"] == 100.0
        assert route["max_ms"] == 200.0

    def test_record_ai_inference(self):
        """Enregistrement d'inférences IA."""
        self.metrics.record_ai_inference("classification", 250.0, success=True)
        self.metrics.record_ai_inference("classification", 300.0, success=True)
        self.metrics.record_ai_inference("classification", 500.0, success=False)

        summary = self.metrics.get_summary()
        ai = summary["ai_models"]["classification"]
        assert ai["total_inferences"] == 3
        assert ai["success_rate"] == 66.7
        assert ai["avg_latency_ms"] == 350.0

    def test_reset(self):
        """Reset remet les compteurs à zéro."""
        self.metrics.record_request("/test", 200, 10.0)
        self.metrics.reset()

        summary = self.metrics.get_summary()
        assert summary["total_requests"] == 0
        assert summary["total_errors"] == 0

    def test_empty_metrics(self):
        """Métriques vides retournent des valeurs par défaut."""
        summary = self.metrics.get_summary()
        assert summary["total_requests"] == 0
        assert summary["error_rate_percent"] == 0
        assert summary["routes"] == {}
        assert summary["ai_models"] == {}


# ============================================
# Tests PerformanceAlerts
# ============================================
class TestPerformanceAlerts:
    """Tests du système d'alertes."""

    def setup_method(self):
        self.metrics = PerformanceMetrics()
        self.alerts = PerformanceAlerts()

    def test_no_alerts_when_healthy(self):
        """Aucune alerte quand tout va bien."""
        # Patch le module metrics global
        with patch("src.monitoring.metrics", self.metrics):
            self.metrics.record_request("/ok", 200, 10.0)
            active = self.alerts.check()
            assert len(active) == 0

    def test_high_error_rate_alert(self):
        """Alerte déclenchée si taux d'erreur > seuil."""
        with patch("src.monitoring.metrics", self.metrics):
            for _ in range(5):
                self.metrics.record_request("/ok", 200, 10.0)
            for _ in range(5):
                self.metrics.record_request("/err", 500, 10.0)

            active = self.alerts.check()
            error_alerts = [a for a in active if a["type"] == "high_error_rate"]
            assert len(error_alerts) == 1
            assert error_alerts[0]["level"] == "critical"

    def test_high_ai_latency_alert(self):
        """Alerte si latence IA dépasse le seuil."""
        with patch("src.monitoring.metrics", self.metrics):
            self.metrics.record_ai_inference("classification", 15000.0)  # 15s > 10s

            active = self.alerts.check()
            latency_alerts = [a for a in active if a["type"] == "high_ai_latency"]
            assert len(latency_alerts) == 1

    def test_low_success_rate_alert(self):
        """Alerte si taux de succès IA < seuil."""
        with patch("src.monitoring.metrics", self.metrics):
            for _ in range(2):
                self.metrics.record_ai_inference("nlp", 10.0, success=True)
            for _ in range(8):
                self.metrics.record_ai_inference("nlp", 10.0, success=False)

            active = self.alerts.check()
            success_alerts = [a for a in active if a["type"] == "low_ai_success_rate"]
            assert len(success_alerts) == 1

    def test_custom_thresholds(self):
        """Les seuils personnalisés fonctionnent."""
        custom_alerts = PerformanceAlerts(thresholds={
            "max_latency_ms": 100,
            "max_error_rate": 5.0,
            "max_ai_latency_ms": 1000,
            "min_ai_success_rate": 99.0,
        })
        with patch("src.monitoring.metrics", self.metrics):
            self.metrics.record_ai_inference("test", 2000.0)
            active = custom_alerts.check()
            assert len(active) >= 1


# ============================================
# Tests Setup Logging
# ============================================
class TestSetupLogging:
    """Tests de la configuration du logging."""

    def test_setup_returns_logger(self):
        """setup_structured_logging retourne un logger configuré."""
        logger = setup_structured_logging(level="DEBUG", log_file=None)
        assert isinstance(logger, logging.Logger)
        assert logger.name == "ecommerce_ia"

    def test_logger_has_handlers(self):
        """Le logger a au moins un handler."""
        logger = setup_structured_logging(level="INFO", log_file=None)
        assert len(logger.handlers) >= 1


# ============================================
# Tests Health Check
# ============================================
class TestHealthCheck:
    """Tests du health check."""

    def test_health_status_structure(self):
        """Le health check retourne la bonne structure."""
        health = get_health_status()
        assert "status" in health
        assert "timestamp" in health
        assert "components" in health
        assert "metrics_summary" in health
        assert health["status"] in ("healthy", "degraded", "unhealthy")

    def test_health_metrics_summary(self):
        """Les métriques résumées sont présentes."""
        health = get_health_status()
        summary = health["metrics_summary"]
        assert "uptime_seconds" in summary
        assert "total_requests" in summary
        assert "error_rate" in summary
