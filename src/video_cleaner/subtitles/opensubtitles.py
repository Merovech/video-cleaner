"""A client for the OpenSubtitles REST API (v1): log in, search by hash or title, download.

The server is not trusted. Every URL the client is about to contact (the host the login names,
the link a download names) must be HTTPS on the service's domain (``opensubtitles.com`` unless
the ``opensubtitles_domain`` setting says otherwise) or one of its subdomains;
redirects are held to the same rule; every response is read with a size cap and its JSON is
checked before use, so an odd or hostile reply raises `ProviderError` (`AuthError` when a login
reply names an API host outside opensubtitles.com, `QuotaExhausted` when the limit is reached)
and never a parser or type error; every request times out. No secret (key, user name,
password, session token) appears in a URL, a log record or an exception message, and each is
registered with `logging_setup.register_secrets` as soon as the client has it.
File names the server sends are never read.

Only the standard library does the HTTP (`UrllibTransport`); tests substitute a `Transport`.
"""

from __future__ import annotations

import contextlib
import email.utils
import functools
import http.client
import io
import json
import logging
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from http.client import HTTPMessage
from typing import IO, Any, Literal, Protocol

from video_cleaner import __version__
from video_cleaner.config import (
    DEFAULT_OPENSUBTITLES_API_HOST,
    DEFAULT_OPENSUBTITLES_API_PATH,
    DEFAULT_OPENSUBTITLES_DOMAIN,
    DEFAULT_SUBTITLE_LANGUAGE,
    Settings,
)
from video_cleaner.credentials import TOKEN, CredentialError, CredentialStore
from video_cleaner.credentials import OpenSubtitlesCredentials as OpenSubtitlesCredentials
from video_cleaner.logging_setup import register_secrets
from video_cleaner.progress import Reporter
from video_cleaner.subtitles.models import QuotaInfo
from video_cleaner.tools import heartbeat

_log = logging.getLogger("video_cleaner.subtitles.opensubtitles")


@dataclass(frozen=True)
class ApiEndpoint:
    """Where the service is and which language is wanted; the defaults are today's values.

    *domain* is the trust boundary: every URL the client contacts, a login's ``base_url``, a
    download link and each redirect must be HTTPS on *domain* or a subdomain of it. *host* is
    the API host used before a login names one, and must lie on *domain* (`load_settings`
    checks it). *language* is compared with the results' language in lower case.
    """

    domain: str = DEFAULT_OPENSUBTITLES_DOMAIN
    host: str = DEFAULT_OPENSUBTITLES_API_HOST
    path: str = DEFAULT_OPENSUBTITLES_API_PATH
    language: str = DEFAULT_SUBTITLE_LANGUAGE

    @property
    def base(self) -> str:
        return f"https://{self.host}{self.path}"

    @classmethod
    def from_settings(cls, settings: Settings) -> ApiEndpoint:
        return cls(
            domain=settings.opensubtitles_domain,
            host=settings.opensubtitles_api_host,
            path=settings.opensubtitles_api_path,
            language=settings.subtitle_language,
        )


REQUEST_TIMEOUT_S = 30.0
MAX_TOTAL_S = 120.0  # one request, however slowly the server drips its reply
MAX_RETRIES = 3  # after the first attempt: 429, 5xx, timeouts and connection errors
BACKOFF_BASE_S = 1.0
MAX_RETRY_DELAY_S = 120.0
HEARTBEAT_S = 15.0

MAX_JSON_BYTES = 4 * 1024 * 1024  # a search reply; login and download replies are far smaller
MAX_URL_CHARS = 2048
MAX_TOKEN_CHARS = 4096
MAX_TEXT_CHARS = 300
MAX_ITEMS = 500
MAX_COUNT = 10**12
MAX_FILE_ID = 2**53
_READ_CHUNK = 64 * 1024

_TOKEN = re.compile(rf"[A-Za-z0-9._~+/=-]{{1,{MAX_TOKEN_CHARS}}}")
_MOVIEHASH = re.compile(r"[0-9a-f]{16}")
_RETRY_AFTER_SECONDS = re.compile(r"\d{1,6}(?:\.\d{1,3})?")


class AuthError(Exception):
    """The login was refused or answered with something the client will not trust."""


class ProviderError(Exception):
    """The service failed, was unreachable or answered with something unusable."""


class DownloadNotCounted(ProviderError):
    """A download failed in a way that cannot have been counted against the allowance: the
    request never reached the server (no connection, a failed login, every attempt refused
    with 429). Any other `ProviderError` from `OpenSubtitlesClient.download` may have been
    counted, so the id is not asked for again."""


class QuotaExhausted(Exception):
    """The download allowance is used up; *reset_at* is when it renews, if the server said."""

    def __init__(self, message: str, reset_at: datetime | None = None) -> None:
        super().__init__(message)
        self.reset_at = reset_at


class TransportError(Exception):
    """A request could not be completed (connection, TLS, protocol, refused URL or redirect)."""


class TransportTimeout(TransportError):
    """A request, or the reading of its reply, took too long."""


class RequestNotSent(TransportError):
    """The request failed before it was sent (DNS, connect, TLS), so the server never saw it."""


class RedirectRefused(TransportError):
    """A redirect that must not or cannot be followed. Deterministic: never worth a retry."""


class ConnectTimeout(TransportTimeout, RequestNotSent):
    """A timeout while connecting: a timeout, and one that happened before anything was sent."""


class ResponseTooLarge(TransportError):
    """A reply was longer than the cap the caller set."""


@dataclass(frozen=True)
class HttpResponse:
    status: int
    headers: Mapping[str, str]
    body: bytes


class Transport(Protocol):
    def request(
        self,
        method: str,
        url: str,
        *,
        headers: Mapping[str, str],
        body: bytes | None,
        timeout_s: float,
        max_bytes: int,
    ) -> HttpResponse:
        """Send one request. Raises `TransportError` (`TransportTimeout`, `ResponseTooLarge`)."""
        ...


# --- the host rule ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=8)
def _host_pattern(domain: str) -> re.Pattern[str]:
    return re.compile(rf"(?:[a-z0-9](?:[a-z0-9-]{{0,61}}[a-z0-9])?\.)*{re.escape(domain.lower())}")


def is_opensubtitles_host(host: str | None, domain: str = DEFAULT_OPENSUBTITLES_DOMAIN) -> bool:
    """True for *domain* (``opensubtitles.com`` unless told otherwise) and its subdomains,
    with a dot boundary.

    ``evilopensubtitles.com``, ``opensubtitles.com.evil.net``, a trailing-dot name and anything
    that is not plain ASCII are refused.
    """
    if not host or len(host) > 253 or not host.isascii():
        return False
    return _host_pattern(domain).fullmatch(host.lower()) is not None


def https_host(url: object, domain: str = DEFAULT_OPENSUBTITLES_DOMAIN) -> str | None:
    """The lower-case host of *url* when it is an acceptable HTTPS URL on *domain*.

    Acceptable: ``https``, no user information, no port other than 443, no whitespace, control
    character or backslash anywhere, a host `is_opensubtitles_host` accepts for *domain*. None
    otherwise.
    """
    if not isinstance(url, str) or not url or len(url) > MAX_URL_CHARS:
        return None
    if any(ch <= " " or ch == "\x7f" or ch == "\\" for ch in url):
        return None
    try:
        parts = urllib.parse.urlsplit(url)
        port = parts.port
    except ValueError:
        return None
    if parts.scheme != "https" or "@" in parts.netloc or port not in (None, 443):
        return None
    host = parts.hostname
    if host is None or not is_opensubtitles_host(host, domain):
        return None
    return host.lower()


# --- the real transport ----------------------------------------------------------------------


class _GuardedRedirects(urllib.request.HTTPRedirectHandler):
    """Follows a redirect only to HTTPS on the service's domain; anything else is refused.

    A redirect's body is never read: urllib would read it whole before following (no size cap,
    no time limit), so it is closed unread first and urllib is handed an empty one.
    """

    def __init__(self, domain: str = DEFAULT_OPENSUBTITLES_DOMAIN) -> None:
        super().__init__()
        self._domain = domain

    def http_error_301(
        self, req: urllib.request.Request, fp: IO[bytes], code: int, msg: str, headers: HTTPMessage
    ) -> Any:
        fp.close()
        return super().http_error_301(req, io.BytesIO(), code, msg, headers)

    def http_error_302(
        self, req: urllib.request.Request, fp: IO[bytes], code: int, msg: str, headers: HTTPMessage
    ) -> Any:
        fp.close()
        return super().http_error_302(req, io.BytesIO(), code, msg, headers)

    def http_error_303(
        self, req: urllib.request.Request, fp: IO[bytes], code: int, msg: str, headers: HTTPMessage
    ) -> Any:
        fp.close()
        return super().http_error_303(req, io.BytesIO(), code, msg, headers)

    def http_error_307(
        self, req: urllib.request.Request, fp: IO[bytes], code: int, msg: str, headers: HTTPMessage
    ) -> Any:
        fp.close()
        return super().http_error_307(req, io.BytesIO(), code, msg, headers)

    def http_error_308(
        self, req: urllib.request.Request, fp: IO[bytes], code: int, msg: str, headers: HTTPMessage
    ) -> Any:
        fp.close()
        return super().http_error_308(req, io.BytesIO(), code, msg, headers)

    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: object,
        code: int,
        msg: str,
        headers: object,
        newurl: str,
    ) -> urllib.request.Request | None:
        if https_host(newurl, self._domain) is None:
            raise RedirectRefused(f"redirect to a URL that is not https on {self._domain} refused")
        return super().redirect_request(req, fp, code, msg, headers, newurl)  # type: ignore[arg-type]


class UrllibTransport:
    """`Transport` over ``urllib``: HTTPS to the service's *domain* only, capped reads.

    *https_handler* and *monotonic* exist for tests. A reply is read in pieces and stops with
    `ResponseTooLarge` the moment it passes the cap, or with `TransportTimeout` when the whole
    exchange has taken more than *max_duration_s* (a server that sends one byte at a time never
    trips the per-read timeout).
    """

    def __init__(
        self,
        *,
        domain: str = DEFAULT_OPENSUBTITLES_DOMAIN,
        https_handler: urllib.request.HTTPSHandler | None = None,
        max_duration_s: float = MAX_TOTAL_S,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self._domain = domain
        self._opener = urllib.request.build_opener(
            _GuardedRedirects(domain), https_handler or urllib.request.HTTPSHandler()
        )
        self._max_duration_s = max_duration_s
        self._monotonic = monotonic

    def request(
        self,
        method: str,
        url: str,
        *,
        headers: Mapping[str, str],
        body: bytes | None,
        timeout_s: float,
        max_bytes: int,
    ) -> HttpResponse:
        if https_host(url, self._domain) is None:
            raise TransportError(f"refusing a URL that is not https on {self._domain}")
        deadline = self._monotonic() + self._max_duration_s
        try:
            return self._exchange(method, url, headers, body, timeout_s, max_bytes, deadline)
        except TransportError:
            raise
        except TimeoutError:
            raise TransportTimeout("the request timed out") from None
        except urllib.error.URLError as exc:
            # urllib raises URLError only for what fails while connecting or sending; a failure
            # while waiting for the reply surfaces as the bare OSError / HTTPException below.
            if isinstance(exc.reason, TimeoutError):
                raise ConnectTimeout("the request timed out before it was sent") from None
            raise RequestNotSent("network error before the request was sent") from None
        except (OSError, http.client.HTTPException, ValueError) as exc:
            # The text of these can carry a request detail; the type name is enough.
            raise TransportError(f"network error ({type(exc).__name__})") from None

    def _exchange(
        self,
        method: str,
        url: str,
        headers: Mapping[str, str],
        body: bytes | None,
        timeout_s: float,
        max_bytes: int,
        deadline: float,
    ) -> HttpResponse:
        # The URL was checked above (https, the service's domain), and so is every redirect.
        req = urllib.request.Request(url, data=body, headers=dict(headers), method=method)  # noqa: S310
        try:
            with contextlib.closing(self._opener.open(req, timeout=timeout_s)) as response:  # nosec B310
                return self._read(response, response.getcode(), max_bytes, deadline)
        except urllib.error.HTTPError as err:
            with contextlib.closing(err):
                if 300 <= err.code < 400:
                    raise RedirectRefused("redirect not followed (or too many)") from None
                return self._read(err, err.code, max_bytes, deadline)

    def _read(self, response: object, status: int, max_bytes: int, deadline: float) -> HttpResponse:
        headers = {k.lower(): v for k, v in response.headers.items()}  # type: ignore[attr-defined]
        declared = headers.get("content-length", "")
        if declared.isascii() and declared.isdigit() and int(declared) > max_bytes:
            raise ResponseTooLarge("the reply is larger than allowed")
        read = getattr(response, "read1", None) or response.read  # type: ignore[attr-defined]
        chunks: list[bytes] = []
        total = 0
        while True:
            if self._monotonic() > deadline:
                raise TransportTimeout("the reply took too long")
            chunk: bytes = read(min(_READ_CHUNK, max_bytes + 1 - total))
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                raise ResponseTooLarge("the reply is larger than allowed")
            chunks.append(chunk)
        return HttpResponse(status, headers, b"".join(chunks))


# --- results ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class SubtitleResult:
    file_id: int
    release: str
    title: str
    year: int | None
    download_count: int
    hash_match: bool
    hearing_impaired: bool
    foreign_parts_only: bool


def rank_results(results: Sequence[SubtitleResult]) -> list[SubtitleResult]:
    """Best first: hash match, then not foreign-parts-only, then most downloaded, then not SDH.

    Stable: results that tie keep the order they came in.
    """
    return sorted(
        results,
        key=lambda r: (
            not r.hash_match,
            r.foreign_parts_only,
            -r.download_count,
            r.hearing_impaired,
        ),
    )


# --- defensive parsing -----------------------------------------------------------------------


def _reject_constant(name: str) -> object:
    raise ValueError(f"JSON constant {name} is not allowed")


def _json_object(body: bytes, what: str) -> dict[str, object]:
    try:
        data = json.loads(body.decode("utf-8"), parse_constant=_reject_constant)
    except (ValueError, RecursionError):  # UnicodeDecodeError is a ValueError
        raise ProviderError(f"{what}: the reply is not valid JSON") from None
    if not isinstance(data, dict):
        raise ProviderError(f"{what}: the reply is not a JSON object")
    return data


def _int(value: object, low: int, high: int) -> int | None:
    """*value* when it is a real integer within [low, high]; booleans, floats and text are not."""
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        return None
    return value


def _clean(value: object, limit: int = MAX_TEXT_CHARS) -> str:
    """Printable characters only (no escape sequences or bidi controls), at most *limit* long."""
    if not isinstance(value, str):
        return ""
    return "".join(ch for ch in value[: limit * 2] if ch.isprintable()).strip()[:limit]


def _flag(attributes: Mapping[str, object], name: str) -> bool:
    return attributes.get(name) is True


def _parse_item(item: object, language: str) -> SubtitleResult | Literal[False] | None:
    """A result, None when it is well formed but unusable (not in *language*, no single file),
    or False when it is malformed."""
    if not isinstance(item, dict):
        return False
    attributes = item.get("attributes")
    if not isinstance(attributes, dict):
        return False
    found_language = attributes.get("language")
    files = attributes.get("files")
    if not isinstance(found_language, str) or not isinstance(files, list):
        return False
    if found_language.lower() != language.lower():
        return None
    if not files:
        return None
    if len(files) > 1:  # a multi-part release; one file of it is not a subtitle
        return None
    entry = files[0]
    if not isinstance(entry, dict):
        return False
    file_id = _int(entry.get("file_id"), 1, MAX_FILE_ID)
    if file_id is None:
        return False
    details = attributes.get("feature_details")
    details = details if isinstance(details, dict) else {}
    return SubtitleResult(
        file_id=file_id,
        release=_clean(attributes.get("release")),
        title=_clean(details.get("title")) or _clean(details.get("movie_name")),
        year=_int(details.get("year"), 1800, 2200),
        download_count=_int(attributes.get("download_count"), 0, MAX_COUNT) or 0,
        hash_match=_flag(attributes, "moviehash_match"),
        hearing_impaired=_flag(attributes, "hearing_impaired"),
        foreign_parts_only=_flag(attributes, "foreign_parts_only"),
    )


def _parse_search(body: bytes, language: str) -> list[SubtitleResult]:
    payload = _json_object(body, "search")
    data = payload.get("data")
    if not isinstance(data, list):
        raise ProviderError("search: the reply has no result list")
    items = data[:MAX_ITEMS]
    found: list[SubtitleResult] = []
    malformed = 0
    for entry in items:
        parsed = _parse_item(entry, language)
        if isinstance(parsed, SubtitleResult):
            found.append(parsed)
        elif parsed is False:
            malformed += 1
    if items and malformed == len(items):
        raise ProviderError("search: no result in the reply could be read")
    skipped = len(items) - len(found)
    if skipped:
        _log.debug(
            "OpenSubtitles search: kept %d, skipped %d unusable results", len(found), skipped
        )
    return found


def _parse_reset(value: object) -> datetime | None:
    if not isinstance(value, str) or len(value) > 64:
        return None
    text = value.strip()
    if text.upper().endswith(" UTC"):
        text = text[:-4]
    try:
        moment = datetime.fromisoformat(text)
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=UTC)  # the field is "reset_time_utc"
        return moment.astimezone(UTC)
    except (ValueError, OverflowError):
        return None


def _header(headers: Mapping[str, str], name: str) -> str | None:
    for key, value in headers.items():
        if key.lower() == name:
            return value
    return None


# --- the client ------------------------------------------------------------------------------


def _utc_now() -> datetime:
    return datetime.now(UTC)


@contextlib.contextmanager
def _before_the_request(counted: bool) -> Iterator[None]:
    """For the steps that come before a request the server counts (a login): a `ProviderError`
    from them is `DownloadNotCounted` when the request is *counted*, since it was not sent."""
    try:
        yield
    except DownloadNotCounted:
        raise
    except ProviderError as exc:
        if not counted:
            raise
        raise DownloadNotCounted(str(exc)) from None


class OpenSubtitlesClient:
    """Search and download through the OpenSubtitles REST API.

    The first call logs in (user name and password, over HTTPS) and caches the session token in
    *store* under `credentials.TOKEN`; a later run reuses it, and a 401 triggers one fresh login
    and one retry. Only subtitles in the endpoint's language (English unless configured) are
    asked for and returned.
    """

    def __init__(
        self,
        creds: OpenSubtitlesCredentials,
        store: CredentialStore,
        transport: Transport,
        reporter: Reporter,
        *,
        endpoint: ApiEndpoint | None = None,
        sleep: Callable[[float], None] = time.sleep,
        now: Callable[[], datetime] = _utc_now,
    ) -> None:
        # Registered here, not left to `load_credentials`, so no code path can log them.
        register_secrets([creds.api_key, creds.username, creds.password])
        if not creds.api_key.isascii() or any(ch <= " " or ch == "\x7f" for ch in creds.api_key):
            raise AuthError("the API key contains characters that cannot be sent in a header")
        self._creds = creds
        self._store = store
        self._transport = transport
        self._reporter = reporter
        self._sleep = sleep
        self._now = now
        self._endpoint = endpoint or ApiEndpoint()
        self._token: str | None = None
        self._base = self._endpoint.base
        self._quota: QuotaInfo | None = None
        self._auth_failure: AuthError | None = None

    def last_quota(self) -> QuotaInfo | None:
        """The allowance the last download reply reported, or None when none did yet."""
        return self._quota

    # -- public operations --

    def search(
        self, *, moviehash: str | None, title: str | None, year: int | None
    ) -> list[SubtitleResult]:
        """Subtitles in the configured language for a file hash and/or a title, best first
        (`rank_results`)."""
        title = (title or "").strip()[:200].lower() or None
        if moviehash is not None and not _MOVIEHASH.fullmatch(moviehash):
            raise ValueError("moviehash must be 16 lower-case hex digits")
        if moviehash is None and title is None:
            raise ValueError("a search needs a moviehash or a title")
        params = {"languages": self._endpoint.language.lower()}
        if moviehash is not None:
            params["moviehash"] = moviehash
        if title is not None:
            params["query"] = title
            if year is not None:
                params["year"] = str(year)
        query = urllib.parse.urlencode(sorted(params.items()), quote_via=urllib.parse.quote_plus)
        resp = self._api("GET", "/subtitles", "OpenSubtitles search", query=query)
        if resp.status != 200:
            raise ProviderError(f"OpenSubtitles search failed: HTTP {resp.status}")
        return rank_results(_parse_search(resp.body, self._endpoint.language))

    def download(self, result: SubtitleResult, *, max_bytes: int) -> bytes:
        """The subtitle file for *result*, at most *max_bytes* long.

        Raises `QuotaExhausted` when the allowance is used up, `DownloadNotCounted` when the
        request provably never reached the server, `ProviderError` otherwise (the download may
        have been counted).
        """
        if max_bytes <= 0:
            raise ValueError("max_bytes must be positive")
        label = "OpenSubtitles download"
        # The server counts the download when it gets this POST, so it is never repeated after
        # it may have arrived (counted=True).
        resp = self._api(
            "POST", "/download", label, payload={"file_id": result.file_id}, counted=True
        )
        if resp.status == 406:
            raise self._exhausted(self._lenient_json(resp.body))
        if resp.status != 200:
            raise ProviderError(f"{label} failed: HTTP {resp.status}")
        info = _json_object(resp.body, "download")
        remaining = _int(info.get("remaining"), -(10**9), 10**9)
        reset_at = _parse_reset(info.get("reset_time_utc"))
        if remaining is not None and remaining < 0:
            raise self._exhausted(info)
        if remaining is not None and reset_at is not None:
            self._quota = QuotaInfo(remaining, reset_at)
        link = info.get("link")
        if https_host(link, self._endpoint.domain) is None or not isinstance(link, str):
            raise ProviderError(f"download: the reply names a link outside {self._endpoint.domain}")
        # The link is fetched with nothing but the User-Agent: no key, no token.
        reply = self._send("GET", link, label, self._plain_headers(), None, max_bytes)
        if reply.status != 200:
            raise ProviderError(f"{label} failed: HTTP {reply.status} from the download link")
        if not reply.body:
            raise ProviderError(f"{label}: the downloaded file is empty")
        return reply.body

    # -- session --

    def _api(
        self,
        method: str,
        endpoint: str,
        label: str,
        *,
        query: str = "",
        payload: Mapping[str, object] | None = None,
        counted: bool = False,
    ) -> HttpResponse:
        """One authenticated API call; a 401 earns one fresh login and one retry.

        A refused login is remembered: the service rate-limits logins and the same credentials
        will be refused again, so this client sends nothing more after one `AuthError`.
        """
        if self._auth_failure is not None:
            raise AuthError(str(self._auth_failure))
        try:
            return self._api_checked(method, endpoint, label, query, payload, counted)
        except AuthError as exc:
            self._auth_failure = exc
            raise

    def _api_checked(
        self,
        method: str,
        endpoint: str,
        label: str,
        query: str,
        payload: Mapping[str, object] | None,
        counted: bool,
    ) -> HttpResponse:
        with _before_the_request(counted):
            fresh = self._ensure_session()
        resp = self._api_once(method, endpoint, label, query, payload, counted)
        if resp.status != 401:
            return resp
        self._forget_token()
        if fresh:
            raise AuthError("OpenSubtitles rejected a session token it had just issued")
        with _before_the_request(counted):  # the refused request was not counted
            self._login()
        resp = self._api_once(method, endpoint, label, query, payload, counted)
        if resp.status == 401:
            raise AuthError("OpenSubtitles rejected the new session token")
        return resp

    def _api_once(
        self,
        method: str,
        endpoint: str,
        label: str,
        query: str,
        payload: Mapping[str, object] | None,
        counted: bool,
    ) -> HttpResponse:
        headers = self._plain_headers()
        headers["Api-Key"] = self._creds.api_key
        headers["Accept"] = "application/json"
        headers["Authorization"] = f"Bearer {self._token}"
        body = None
        if payload is not None:
            headers["Content-Type"] = "application/json"
            body = json.dumps(payload).encode("utf-8")
        url = f"{self._base}{endpoint}" + (f"?{query}" if query else "")
        return self._send(method, url, label, headers, body, MAX_JSON_BYTES, counted=counted)

    def _ensure_session(self) -> bool:
        """Have a token; True when this call had to log in to get one."""
        if self._token is not None:
            return False
        cached = self._read_cached_token()
        if cached is not None:
            self._token = cached
            return False
        self._login()
        return True

    def _read_cached_token(self) -> str | None:
        try:
            value = self._store.get(TOKEN)
        except CredentialError:
            _log.warning("could not read the cached OpenSubtitles session token")
            return None
        if not value:
            return None
        register_secrets([value])
        return value if _TOKEN.fullmatch(value) else None

    def _forget_token(self) -> None:
        self._token = None
        try:
            self._store.delete(TOKEN)
        except CredentialError:
            _log.warning("could not remove the cached OpenSubtitles session token")

    def _login(self) -> None:
        headers = self._plain_headers()
        headers["Api-Key"] = self._creds.api_key
        headers["Accept"] = "application/json"
        headers["Content-Type"] = "application/json"
        body = json.dumps({"username": self._creds.username, "password": self._creds.password})
        resp = self._send(
            "POST",
            f"{self._endpoint.base}/login",
            "OpenSubtitles login",
            headers,
            body.encode("utf-8"),
            MAX_JSON_BYTES,
        )
        if 400 <= resp.status < 500:
            raise AuthError(f"OpenSubtitles refused the login: HTTP {resp.status}")
        if resp.status != 200:
            raise ProviderError(f"OpenSubtitles login failed: HTTP {resp.status}")
        payload = _json_object(resp.body, "login")
        token = payload.get("token")
        if not isinstance(token, str) or not token:
            raise ProviderError("login: the reply has no session token")
        register_secrets([token])
        if not _TOKEN.fullmatch(token):
            raise ProviderError("login: the session token is not in the expected form")
        base = self._trusted_base(payload.get("base_url"))
        self._token = token
        self._base = base
        try:
            self._store.set(TOKEN, token)
        except CredentialError:
            _log.warning("could not cache the OpenSubtitles session token; it lasts for this run")

    def _trusted_base(self, value: object) -> str:
        """The API root the login names, only if it is on the service's domain; else
        `AuthError`."""
        if value is None:
            return self._endpoint.base
        if not isinstance(value, str) or not value.strip():
            raise AuthError("login: the reply names an unusable API host")
        text = value.strip()
        host = https_host(text if "://" in text else f"https://{text}", self._endpoint.domain)
        if host is None:
            raise AuthError(f"login: the reply names an API host outside {self._endpoint.domain}")
        return f"https://{host}{self._endpoint.path}"

    # -- one request, with retries --

    @staticmethod
    def _plain_headers() -> dict[str, str]:
        return {"User-Agent": f"video-cleaner v{__version__}"}

    def _send(
        self,
        method: str,
        url: str,
        label: str,
        headers: Mapping[str, str],
        body: bytes | None,
        max_bytes: int,
        *,
        counted: bool = False,
    ) -> HttpResponse:
        """One request, retried on 429, 5xx, timeouts and connection errors (`MAX_RETRIES`).

        With *counted* (the request the server charges to the user's quota when it receives it)
        only a 429 and a failure before the request was sent are retried: after a timeout, a lost
        connection or a 5xx the server may already have counted it, and asking again could spend
        another download for nothing.
        """
        self._reporter.stage(label)
        with heartbeat(self._reporter, label, HEARTBEAT_S):
            for attempt in range(MAX_RETRIES + 1):
                last = attempt == MAX_RETRIES
                resp = self._attempt(method, url, label, headers, body, max_bytes, last, counted)
                if resp is None:  # a timeout or connection error that is worth another try
                    self._sleep(self._retry_delay(None, attempt))
                    continue
                _log.debug("%s: HTTP %d", label, resp.status)
                if len(resp.body) > max_bytes:
                    raise ProviderError(f"{label}: the reply is larger than allowed")
                if resp.status >= 500 and counted:
                    raise ProviderError(
                        f"{label} failed: HTTP {resp.status}; not retried, "
                        "the server may have counted the download"
                    )
                if resp.status == 429 or resp.status >= 500:
                    if last:
                        message = f"{label} failed: HTTP {resp.status} after {MAX_RETRIES} retries"
                        if counted:  # only 429s and unsent requests got here: none was counted
                            raise DownloadNotCounted(message)
                        raise ProviderError(message)
                    self._sleep(self._retry_delay(resp, attempt))
                    continue
                return resp
        raise AssertionError("unreachable")  # pragma: no cover - the loop returns or raises

    def _attempt(
        self,
        method: str,
        url: str,
        label: str,
        headers: Mapping[str, str],
        body: bytes | None,
        max_bytes: int,
        last: bool,
        counted: bool,
    ) -> HttpResponse | None:
        try:
            return self._transport.request(
                method,
                url,
                headers=headers,
                body=body,
                timeout_s=REQUEST_TIMEOUT_S,
                max_bytes=max_bytes,
            )
        except ResponseTooLarge:
            raise ProviderError(f"{label}: the reply is larger than allowed") from None
        except RedirectRefused:
            raise ProviderError(f"{label}: a redirect was refused") from None
        except TransportError as exc:
            what = "timed out" if isinstance(exc, TransportTimeout) else "network error"
            if counted and not isinstance(exc, RequestNotSent):
                raise ProviderError(
                    f"{label} failed: {what}; not retried, the server may have counted the download"
                ) from None
        if last:
            message = f"{label} failed: {what} after {MAX_RETRIES} retries"
            if counted:  # a counted request only gets here when it was never sent
                raise DownloadNotCounted(message)
            raise ProviderError(message)
        _log.debug("%s: %s, retrying", label, what)
        return None

    def _retry_delay(self, resp: HttpResponse | None, attempt: int) -> float:
        backoff = float(BACKOFF_BASE_S * 2**attempt)
        if resp is None or resp.status != 429:
            return backoff
        value = _header(resp.headers, "retry-after")
        seconds = self._retry_after_seconds(value.strip()) if value is not None else None
        if seconds is None:
            return backoff
        return min(max(seconds, 0.0), MAX_RETRY_DELAY_S)

    def _retry_after_seconds(self, value: str) -> float | None:
        if _RETRY_AFTER_SECONDS.fullmatch(value):
            return float(value)
        try:
            when = email.utils.parsedate_to_datetime(value)
        except (TypeError, ValueError, IndexError):
            return None
        if when.tzinfo is None:
            when = when.replace(tzinfo=UTC)
        return (when - self._now()).total_seconds()

    # -- quota --

    @staticmethod
    def _lenient_json(body: bytes) -> dict[str, object]:
        try:
            return _json_object(body, "quota")
        except ProviderError:
            return {}

    def _exhausted(self, info: Mapping[str, object]) -> QuotaExhausted:
        reset_at = _parse_reset(info.get("reset_time_utc"))
        if reset_at is not None:
            self._quota = QuotaInfo(0, reset_at)
        when = f"; it renews at {reset_at:%Y-%m-%d %H:%M} UTC" if reset_at else ""
        return QuotaExhausted(f"the OpenSubtitles download allowance is used up{when}", reset_at)
