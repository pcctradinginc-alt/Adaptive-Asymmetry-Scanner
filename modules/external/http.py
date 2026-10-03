"""
modules/external/http.py – höfliche HTTP-Schicht für offizielle Datenquellen.

- Nur offizielle APIs/Downloads; kein Umgehen von Anti-Bot/Auth.
- Retries mit Backoff nur für 5xx/Netzfehler; 401/403 → AuthError (nicht retryen).
- Liefert Bytes + Metadaten (Status, Content-Type, Größe, Hash, Fingerprint).
- Credentials werden nie geloggt oder gespeichert (request_fingerprint filtert sie).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import requests

from modules.external.pit import payload_hash, request_fingerprint, utc_now

log = logging.getLogger(__name__)

USER_AGENT = "AdaptiveAsymmetryScanner/1.0 (research; official-data-only)"
DEFAULT_TIMEOUT = 30


_SECRET_PARAM_RE = None


def redact_secrets(text: str) -> str:
    """Maskiert Credentials in URLs/Fehlertexten (api_key=, securityToken=,
    appId=, token=, ...). requests-Exceptions enthalten die volle URL MIT
    Query-String; ohne Maskierung landete ein Key über repr(e) in der
    committeten source_health.json (Audit 2026-09-29, kein Leak gefunden)."""
    import re
    global _SECRET_PARAM_RE
    if _SECRET_PARAM_RE is None:
        _SECRET_PARAM_RE = re.compile(
            r"(?i)([?&;\s](?:[a-z_]*api[_-]?key|[a-z_]*token|app[_-]?id|secret|password|passwd|key)=)[^&\s'\"<>)]+")
    return _SECRET_PARAM_RE.sub(r"\1***", str(text))


class FetchError(Exception):
    """Netz-/Serverfehler nach allen Retries."""


class AuthError(FetchError):
    """401/403 oder fehlende Credentials — nie automatisch retryen."""


class SchemaError(Exception):
    """Antwort passt nicht zum erwarteten Schema → laut scheitern, nie umdeuten."""


@dataclass
class FetchResult:
    url: str
    status: int
    content: bytes
    content_type: str
    retrieved_at: object          # datetime UTC
    content_hash: str
    fingerprint: str
    bytes: int

    def json(self):
        import json
        return json.loads(self.content.decode("utf-8"))


def fetch(url: str, params: dict | None = None, headers: dict | None = None,
          timeout: int = DEFAULT_TIMEOUT, retries: int = 3, backoff: float = 2.0,
          session: requests.Session | None = None, method: str = "GET",
          data: dict | None = None, json_body: dict | None = None) -> FetchResult:
    """method='POST' schickt `data` als application/x-www-form-urlencoded
    Body (z.B. für GENESIS-Online, das POST für data/tablefile bevorzugt);
    `params` bleibt in diesem Fall die Query-String-Ergänzung (falls nötig).
    Credentials in `data`/`headers` werden nie geloggt (request_fingerprint
    filtert nur den GET-params-Pfad; POST-Bodies landen nicht im Fingerprint)."""
    hdrs = {"User-Agent": USER_AGENT, "Accept": "*/*"}
    hdrs.update(headers or {})
    sess = session or requests
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            if method.upper() == "POST" and json_body is not None:
                r = sess.post(url, params=params, json=json_body, headers=hdrs, timeout=timeout)
            elif method.upper() == "POST":
                r = sess.post(url, params=params, data=data, headers=hdrs, timeout=timeout)
            else:
                r = sess.get(url, params=params, headers=hdrs, timeout=timeout)
            if r.status_code in (401, 403):
                raise AuthError(f"{r.status_code} für {url}")
            if r.status_code == 429 or r.status_code >= 500:
                raise FetchError(f"{r.status_code} für {url}")
            if r.status_code >= 400:
                # Fehlertext des Servers (gekürzt) für die Diagnose; enthält
                # keine Request-Daten/Credentials, nur die Server-Antwort.
                detail = ""
                try:
                    detail = (r.text or "")[:200].replace("\n", " ").strip()
                except Exception:  # noqa: BLE001
                    pass
                raise FetchError(f"{r.status_code} für {url} (nicht retrybar)"
                                 + (f": {detail}" if detail else ""))
            content = r.content or b""
            return FetchResult(
                url=url, status=r.status_code, content=content,
                content_type=r.headers.get("Content-Type", ""),
                retrieved_at=utc_now(), content_hash=payload_hash(content),
                fingerprint=request_fingerprint("GET", url, params), bytes=len(content),
            )
        except AuthError:
            raise
        except (requests.RequestException, FetchError) as e:
            last_err = e
            if isinstance(e, FetchError) and "nicht retrybar" in str(e):
                break
            if attempt < retries - 1:
                time.sleep(backoff * (2 ** attempt))
    raise FetchError(redact_secrets(str(last_err)))
