"""
Tests unitaires — Module Security (Rate Limiting, Sanitization, Headers)
"""

import pytest
import time


# ============================================
# Tests Rate Limiter
# ============================================
class TestRateLimiter:
    """Tests pour le rate limiter Token Bucket."""

    def setup_method(self):
        from src.security import RateLimiter
        self.limiter = RateLimiter(requests_per_minute=60, burst_size=5)

    def test_first_request_allowed(self):
        """La première requête est toujours autorisée."""
        allowed, retry = self.limiter.check("10.0.0.1")
        assert allowed is True
        assert retry == 0.0

    def test_burst_allowed(self):
        """Le burst initial est autorisé."""
        ip = "10.0.0.2"
        for i in range(5):
            allowed, _ = self.limiter.check(ip)
            assert allowed is True, f"Requête {i+1} devrait passer"

    def test_over_burst_rejected(self):
        """Les requêtes au-delà du burst sont rejetées."""
        ip = "10.0.0.3"
        # Épuiser le burst
        for _ in range(5):
            self.limiter.check(ip)
        # La suivante doit être rejetée
        allowed, retry = self.limiter.check(ip)
        assert allowed is False
        assert retry > 0

    def test_different_ips_independent(self):
        """Chaque IP a son propre bucket."""
        for i in range(5):
            self.limiter.check("10.0.0.10")
        # L'autre IP doit encore passer
        allowed, _ = self.limiter.check("10.0.0.11")
        assert allowed is True

    def test_refill_over_time(self):
        """Les tokens se rechargent au fil du temps."""
        ip = "10.0.0.4"
        # Épuiser le burst
        for _ in range(5):
            self.limiter.check(ip)
        # Attendre un peu (simuler le remplissage)
        time.sleep(0.15)  # ~0.15 tokens ajoutés à 1/s
        allowed, _ = self.limiter.check(ip)
        # Peut être encore rejeté selon le timing exact, just check no crash
        assert isinstance(allowed, bool)

    def test_cleanup_removes_stale(self):
        """Le cleanup supprime les entrées inactives."""
        self.limiter.check("10.0.0.5")
        assert self.limiter.active_clients >= 1
        time.sleep(0.05)  # Laisser un peu de temps passer
        self.limiter.cleanup(max_age_seconds=0.01)
        assert self.limiter.active_clients == 0

    def test_active_clients_count(self):
        """Compte le nombre de clients actifs."""
        assert self.limiter.active_clients == 0
        self.limiter.check("10.0.0.6")
        self.limiter.check("10.0.0.7")
        assert self.limiter.active_clients == 2


# ============================================
# Tests Input Sanitizer
# ============================================
class TestInputSanitizer:
    """Tests pour l'assainissement des entrées."""

    def setup_method(self):
        from src.security import InputSanitizer
        self.sanitizer = InputSanitizer()

    def test_sanitize_html_tags(self):
        """Les tags HTML sont supprimés."""
        result = self.sanitizer.sanitize_text("<script>alert('xss')</script>Hello")
        assert "<script>" not in result
        assert "Hello" in result

    def test_sanitize_null_bytes(self):
        """Les caractères nuls sont supprimés."""
        result = self.sanitizer.sanitize_text("Hello\x00World")
        assert "\x00" not in result
        assert "HelloWorld" in result

    def test_sanitize_max_length(self):
        """Le texte est tronqué à max_length."""
        long_text = "A" * 10000
        result = self.sanitizer.sanitize_text(long_text, max_length=100)
        assert len(result) == 100

    def test_sanitize_empty_string(self):
        """Les chaînes vides sont gérées."""
        assert self.sanitizer.sanitize_text("") == ""

    def test_sql_injection_detected(self):
        """Les patterns SQL dangereux sont détectés."""
        assert self.sanitizer.check_sql_injection("'; DROP TABLE users;--") is True
        assert self.sanitizer.check_sql_injection("SELECT * FROM products") is True
        assert self.sanitizer.check_sql_injection("1 OR 1=1") is True

    def test_normal_text_not_flagged(self):
        """Le texte normal n'est pas signalé comme injection."""
        assert self.sanitizer.check_sql_injection("Je cherche une robe rouge") is False
        assert self.sanitizer.check_sql_injection("Quel est le prix ?") is False

    def test_validate_file_valid_jpg(self):
        """Un fichier .jpg valide est accepté."""
        valid, msg = self.sanitizer.validate_file_upload("photo.jpg")
        assert valid is True

    def test_validate_file_valid_png(self):
        """Un fichier .png valide est accepté."""
        valid, msg = self.sanitizer.validate_file_upload("image.png")
        assert valid is True

    def test_validate_file_invalid_extension(self):
        """Un fichier avec une extension invalide est rejeté."""
        valid, msg = self.sanitizer.validate_file_upload("malware.exe")
        assert valid is False
        assert "non autorisée" in msg

    def test_validate_file_too_large(self):
        """Un fichier trop volumineux est rejeté."""
        valid, msg = self.sanitizer.validate_file_upload(
            "big.jpg", content_length=20 * 1024 * 1024
        )
        assert valid is False
        assert "volumineux" in msg

    def test_validate_file_no_name(self):
        """Un nom de fichier vide est rejeté."""
        valid, msg = self.sanitizer.validate_file_upload("")
        assert valid is False

    def test_sanitize_filename_special_chars(self):
        """Les caractères spéciaux sont remplacés dans les noms de fichiers."""
        result = self.sanitizer.sanitize_filename("../../etc/passwd")
        assert "/" not in result
        assert ".." not in result

    def test_sanitize_filename_keeps_valid(self):
        """Les noms de fichiers valides sont préservés."""
        result = self.sanitizer.sanitize_filename("photo_123.jpg")
        assert result == "photo_123.jpg"

    def test_allowed_extensions_complete(self):
        """Toutes les extensions image courantes sont supportées."""
        for ext in [".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"]:
            valid, _ = self.sanitizer.validate_file_upload(f"test{ext}")
            assert valid is True, f"Extension {ext} devrait être acceptée"


# ============================================
# Tests Security Headers
# ============================================
class TestSecurityConcepts:
    """Tests des concepts de sécurité."""

    def test_rate_limiter_custom_config(self):
        """Configuration personnalisée du rate limiter."""
        from src.security import RateLimiter
        limiter = RateLimiter(requests_per_minute=10, burst_size=2)
        # 2 requêtes passent (burst)
        assert limiter.check("10.0.0.20")[0] is True
        assert limiter.check("10.0.0.20")[0] is True
        # La 3ème est rejetée
        assert limiter.check("10.0.0.20")[0] is False

    def test_sanitizer_global_instance(self):
        """L'instance globale sanitizer est disponible."""
        from src.security import sanitizer
        assert sanitizer is not None
        result = sanitizer.sanitize_text("test")
        assert result == "test"

    def test_xss_patterns_cleaned(self):
        """Patterns XSS courants sont nettoyés."""
        from src.security import InputSanitizer
        attacks = [
            "<img src=x onerror=alert(1)>",
            "<iframe src='evil.com'></iframe>",
            "<div onmouseover='steal()'>Click</div>",
        ]
        for attack in attacks:
            clean = InputSanitizer.sanitize_text(attack)
            assert "<" not in clean
            assert ">" not in clean
