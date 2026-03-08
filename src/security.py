"""
============================================
ECommerce-IA — Sécurité API
============================================
Module de sécurité pour l'API FastAPI :
- Rate Limiting (limitation de débit par IP)
- Validation et assainissement des entrées
- Headers de sécurité (CSP, HSTS, XSS-Protection)
- Protection contre les injections

Auteur : BAWANA Théodore — Projet SAHELYS
============================================
"""

import re
import time
import logging
from typing import Dict, Optional, Tuple
from collections import defaultdict
from datetime import datetime, timezone

from fastapi import Request, Response, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger("ecommerce_ia.security")


# ============================================
# 1. Rate Limiter (Token Bucket)
# ============================================
class RateLimiter:
    """
    Limitation de débit par IP utilisant l'algorithme Token Bucket.

    Chaque IP dispose d'un bucket de tokens qui se remplit au fil du temps.
    Chaque requête consomme un token. Quand le bucket est vide, la requête
    est rejetée avec HTTP 429 (Too Many Requests).

    Args:
        requests_per_minute: Nombre max de requêtes par minute par IP.
        burst_size: Taille du bucket (permet un burst initial).

    Exemple:
        limiter = RateLimiter(requests_per_minute=60, burst_size=10)
        allowed, retry_after = limiter.check("192.168.1.1")
    """

    def __init__(self, requests_per_minute: int = 60, burst_size: int = 10):
        self.rate = requests_per_minute / 60.0  # tokens par seconde
        self.burst_size = burst_size
        self._buckets: Dict[str, Tuple[float, float]] = {}  # ip -> (tokens, last_time)

    def check(self, client_ip: str) -> Tuple[bool, float]:
        """
        Vérifie si une requête est autorisée pour cette IP.

        Args:
            client_ip: Adresse IP du client.

        Returns:
            Tuple (allowed, retry_after_seconds).
            Si allowed=False, retry_after indique combien de secondes attendre.
        """
        now = time.monotonic()

        if client_ip not in self._buckets:
            self._buckets[client_ip] = (self.burst_size - 1, now)
            return True, 0.0

        tokens, last_time = self._buckets[client_ip]
        elapsed = now - last_time

        # Remplir le bucket (tokens accumulés depuis la dernière requête)
        tokens = min(self.burst_size, tokens + elapsed * self.rate)

        if tokens >= 1:
            self._buckets[client_ip] = (tokens - 1, now)
            return True, 0.0
        else:
            # Calculer le temps d'attente
            retry_after = (1 - tokens) / self.rate
            self._buckets[client_ip] = (tokens, now)
            return False, retry_after

    def cleanup(self, max_age_seconds: float = 300.0):
        """
        Supprime les entrées inactives depuis plus de max_age_seconds.

        Appeler périodiquement pour éviter les fuites mémoire.
        """
        now = time.monotonic()
        stale = [
            ip for ip, (_, last) in self._buckets.items()
            if now - last > max_age_seconds
        ]
        for ip in stale:
            del self._buckets[ip]

    @property
    def active_clients(self) -> int:
        """Nombre de clients actuellement suivis."""
        return len(self._buckets)


# ============================================
# 2. Rate Limiting Middleware
# ============================================
class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Middleware FastAPI pour appliquer le rate limiting à toutes les requêtes.

    Usage:
        app.add_middleware(RateLimitMiddleware, requests_per_minute=60)

    Headers ajoutés:
        X-RateLimit-Limit: nombre max de requêtes/minute
        X-RateLimit-Remaining: requêtes restantes (estimé)
        Retry-After: secondes à attendre (si 429)
    """

    def __init__(self, app, requests_per_minute: int = 60, burst_size: int = 15):
        super().__init__(app)
        self.limiter = RateLimiter(
            requests_per_minute=requests_per_minute,
            burst_size=burst_size,
        )
        self.requests_per_minute = requests_per_minute
        self._cleanup_counter = 0

    async def dispatch(self, request: Request, call_next):
        # Identifier le client
        client_ip = request.client.host if request.client else "unknown"

        # Exclure les routes de santé
        if request.url.path in ("/health", "/docs", "/redoc", "/openapi.json", "/"):
            return await call_next(request)

        # Vérifier
        allowed, retry_after = self.limiter.check(client_ip)

        if not allowed:
            logger.warning(
                f"Rate limit exceeded for {client_ip} on {request.url.path}",
                extra={"client_ip": client_ip, "path": str(request.url.path)},
            )
            return Response(
                content='{"detail":"Trop de requêtes. Réessayez plus tard."}',
                status_code=429,
                media_type="application/json",
                headers={
                    "Retry-After": str(int(retry_after) + 1),
                    "X-RateLimit-Limit": str(self.requests_per_minute),
                },
            )

        response = await call_next(request)

        # Ajouter les headers de rate limiting
        response.headers["X-RateLimit-Limit"] = str(self.requests_per_minute)

        # Nettoyage périodique (toutes les 100 requêtes)
        self._cleanup_counter += 1
        if self._cleanup_counter >= 100:
            self.limiter.cleanup()
            self._cleanup_counter = 0

        return response


# ============================================
# 3. Security Headers Middleware
# ============================================
class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Middleware ajoutant les headers de sécurité HTTP standard.

    Headers appliqués:
        X-Content-Type-Options: nosniff
        X-Frame-Options: DENY
        X-XSS-Protection: 1; mode=block
        Strict-Transport-Security: max-age=31536000
        Referrer-Policy: strict-origin-when-cross-origin
        Permissions-Policy: camera=(), microphone=(), geolocation=()
    """

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)

        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = (
            "max-age=31536000; includeSubDomains"
        )
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = (
            "camera=(), microphone=(), geolocation=()"
        )

        return response


# ============================================
# 4. Input Sanitization
# ============================================
class InputSanitizer:
    """
    Assainit les entrées utilisateur pour prévenir les injections.

    Méthodes:
        sanitize_text: nettoyage de texte (HTML, scripts, SQL)
        validate_file_type: validation du type de fichier uploadé
        sanitize_filename: nettoyage du nom de fichier

    Exemple:
        sanitizer = InputSanitizer()
        clean = sanitizer.sanitize_text("<script>alert('xss')</script>")
        # → "alert('xss')"
    """

    # Tags HTML dangereux
    _HTML_TAG_RE = re.compile(r"<[^>]+>")
    # Patterns d'injection SQL courants
    _SQL_INJECTION_RE = re.compile(
        r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|UNION|ALTER|CREATE|EXEC)\b"
        r"|\b(OR|AND)\s+\d+\s*=\s*\d+|--|\;|\/\*)",
        re.IGNORECASE,
    )
    # Extensions de fichiers image autorisées
    ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}
    # Taille max d'image (10 Mo)
    MAX_IMAGE_SIZE = 10 * 1024 * 1024

    @classmethod
    def sanitize_text(cls, text: str, max_length: int = 5000) -> str:
        """
        Nettoie un texte utilisateur.

        - Supprime les tags HTML
        - Tronque à max_length caractères
        - Supprime les caractères nuls

        Args:
            text: Texte brut à nettoyer.
            max_length: Longueur maximale autorisée.

        Returns:
            Texte assaini.
        """
        if not text:
            return ""
        # Supprimer les caractères nuls
        text = text.replace("\x00", "")
        # Supprimer les tags HTML
        text = cls._HTML_TAG_RE.sub("", text)
        # Tronquer
        return text[:max_length].strip()

    @classmethod
    def check_sql_injection(cls, text: str) -> bool:
        """
        Détecte les patterns d'injection SQL potentiels.

        Args:
            text: Texte à vérifier.

        Returns:
            True si un pattern suspect est détecté.
        """
        return bool(cls._SQL_INJECTION_RE.search(text))

    @classmethod
    def validate_file_upload(
        cls,
        filename: str,
        content_length: Optional[int] = None,
    ) -> Tuple[bool, str]:
        """
        Valide un fichier uploadé (nom, extension, taille).

        Args:
            filename: Nom du fichier.
            content_length: Taille en octets (si disponible).

        Returns:
            Tuple (valid, message).
        """
        if not filename:
            return False, "Nom de fichier manquant"

        # Extension
        ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        if ext not in cls.ALLOWED_IMAGE_EXTENSIONS:
            return False, (
                f"Extension '{ext}' non autorisée. "
                f"Formats acceptés : {', '.join(sorted(cls.ALLOWED_IMAGE_EXTENSIONS))}"
            )

        # Taille
        if content_length and content_length > cls.MAX_IMAGE_SIZE:
            max_mb = cls.MAX_IMAGE_SIZE / (1024 * 1024)
            return False, f"Fichier trop volumineux. Maximum : {max_mb:.0f} Mo"

        return True, "OK"

    @classmethod
    def sanitize_filename(cls, filename: str) -> str:
        """
        Nettoie un nom de fichier (supprime les caractères dangereux).

        Args:
            filename: Nom de fichier original.

        Returns:
            Nom de fichier assaini.
        """
        # Garder uniquement alphanum, tirets, underscores, points
        clean = re.sub(r"[^\w\-.]", "_", filename)
        # Empêcher la traversée de répertoire
        clean = clean.replace("..", "_")
        return clean


# Instance globale pour usage dans l'API
sanitizer = InputSanitizer()
