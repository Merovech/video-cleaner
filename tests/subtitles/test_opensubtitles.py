"""The OpenSubtitles client against a fake transport that replays synthesised fixtures."""

# Every key, password and token below is an obviously fake placeholder.
# ruff: noqa: S105, S106

from __future__ import annotations

import json
import logging
import urllib.parse
from collections.abc import Callable, Iterator, Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from media import RecordingReporter
from video_cleaner import __version__
from video_cleaner.credentials import TOKEN, CredentialError, OpenSubtitlesCredentials
from video_cleaner.logging_setup import redact
from video_cleaner.subtitles import opensubtitles as module
from video_cleaner.subtitles.models import QuotaInfo
from video_cleaner.subtitles.opensubtitles import (
    MAX_JSON_BYTES,
    MAX_RETRIES,
    ApiEndpoint,
    AuthError,
    ConnectTimeout,
    HttpResponse,
    OpenSubtitlesClient,
    ProviderError,
    QuotaExhausted,
    RedirectRefused,
    RequestNotSent,
    ResponseTooLarge,
    SubtitleResult,
    TransportError,
    TransportTimeout,
    https_host,
    is_opensubtitles_host,
    rank_results,
)

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures" / "opensubtitles"

API_KEY = "FAKE-API-KEY-0000"
USERNAME = "fake-user-0000"
PASSWORD = "fake-pass-0000"
SESSION_TOKEN = "FAKE-SESSION-TOKEN-0000.FAKE.FAKE"
SECRETS = (API_KEY, USERNAME, PASSWORD, SESSION_TOKEN)
HASH = "0123456789abcdef"
SERVER_FILE_NAME = "FAKE-FILE-NAME-DO-NOT-USE"
NOW = datetime(2030, 1, 1, 12, 0, tzinfo=UTC)


def fixture_bytes(name: str) -> bytes:
    return (FIXTURES / name).read_bytes()


def fixture_json(name: str) -> dict[str, object]:
    data = json.loads(fixture_bytes(name))
    assert isinstance(data, dict)
    return data


def reply(
    status: int = 200,
    body: bytes | str | Mapping[str, object] | list[object] | None = b"",
    headers: Mapping[str, str] | None = None,
) -> HttpResponse:
    if body is None:
        raw = b""
    elif isinstance(body, bytes):
        raw = body
    elif isinstance(body, str):
        raw = body.encode("utf-8")
    else:
        raw = json.dumps(body).encode("utf-8")
    return HttpResponse(status, dict(headers or {}), raw)


def fx(name: str, status: int = 200, headers: Mapping[str, str] | None = None) -> HttpResponse:
    return reply(status, fixture_bytes(name), headers)


class Recorded:
    def __init__(
        self,
        method: str,
        url: str,
        headers: Mapping[str, str],
        body: bytes | None,
        timeout_s: float,
        max_bytes: int,
    ) -> None:
        self.method = method
        self.url = url
        self.headers = dict(headers)
        self.body = body
        self.timeout_s = timeout_s
        self.max_bytes = max_bytes

    @property
    def path(self) -> str:
        return urllib.parse.urlsplit(self.url).path

    @property
    def query(self) -> dict[str, list[str]]:
        return urllib.parse.parse_qs(urllib.parse.urlsplit(self.url).query)

    @property
    def host(self) -> str | None:
        return urllib.parse.urlsplit(self.url).hostname

    def json_body(self) -> object:
        assert self.body is not None
        return json.loads(self.body)


class FakeTransport:
    """Plays queued replies (or raises queued exceptions) in order and records every request."""

    def __init__(self, *script: HttpResponse | Exception) -> None:
        self.script = list(script)
        self.requests: list[Recorded] = []

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
        self.requests.append(Recorded(method, url, headers, body, timeout_s, max_bytes))
        assert self.script, f"unexpected request {method} {url}"
        step = self.script.pop(0)
        if isinstance(step, Exception):
            raise step
        return step

    def paths(self) -> list[str]:
        return [r.path for r in self.requests]


class MemoryStore:
    def __init__(self, **values: str) -> None:
        self.values = dict(values)
        self.fail_get = False
        self.fail_set = False

    def get(self, name: str) -> str | None:
        if self.fail_get:
            raise CredentialError("keyring read failed")
        return self.values.get(name)

    def set(self, name: str, value: str) -> None:
        if self.fail_set:
            raise CredentialError("keyring write failed")
        self.values[name] = value

    def delete(self, name: str) -> None:
        self.values.pop(name, None)


class Harness:
    def __init__(
        self,
        *script: HttpResponse | Exception,
        token: str | None = None,
        now: Callable[[], datetime] | None = None,
        endpoint: ApiEndpoint | None = None,
    ) -> None:
        self.transport = FakeTransport(*script)
        self.store = MemoryStore(**({TOKEN: token} if token else {}))
        self.reporter = RecordingReporter()
        self.sleeps: list[float] = []
        self.client = OpenSubtitlesClient(
            OpenSubtitlesCredentials(API_KEY, USERNAME, PASSWORD),
            self.store,
            self.transport,
            self.reporter,
            endpoint=endpoint,
            sleep=self.sleeps.append,
            now=now or (lambda: NOW),
        )

    def done(self) -> None:
        assert self.transport.script == [], "queued replies were never requested"


@pytest.fixture
def oslog(caplog: pytest.LogCaptureFixture) -> Iterator[pytest.LogCaptureFixture]:
    """Records of the tool's logger, whatever an earlier test did to its propagation."""
    logger = logging.getLogger("video_cleaner")
    logger.addHandler(caplog.handler)
    old_level = logger.level
    logger.setLevel(logging.DEBUG)
    try:
        yield caplog
    finally:
        logger.setLevel(old_level)
        logger.removeHandler(caplog.handler)


def assert_no_secret(text: str) -> None:
    for secret in SECRETS:
        assert secret not in text


def assert_logs_clean(caplog: pytest.LogCaptureFixture) -> None:
    assert caplog.records, "nothing was logged, so the check proves nothing"
    for record in caplog.records:
        assert_no_secret(record.getMessage())
        assert_no_secret(record.exc_text or "")
        assert SERVER_FILE_NAME not in record.getMessage()


def result(
    file_id: int = 1,
    *,
    hash_match: bool = False,
    fpo: bool = False,
    downloads: int = 0,
    hi: bool = False,
) -> SubtitleResult:
    return SubtitleResult(
        file_id=file_id,
        release="r",
        title="t",
        year=None,
        download_count=downloads,
        hash_match=hash_match,
        hearing_impaired=hi,
        foreign_parts_only=fpo,
    )


def search_ok(data: object | None = None) -> HttpResponse:
    return reply(200, {"data": data} if data is not None else fixture_json("search_hash.json"))


def item(file_id: object = 7, **attributes: object) -> dict[str, object]:
    attrs: dict[str, object] = {
        "language": "en",
        "download_count": 1,
        "release": "Some.Release",
        "files": [{"file_id": file_id}],
        "feature_details": {"title": "Title", "year": 2001},
    }
    attrs.update(attributes)
    return {"id": "1", "type": "subtitle", "attributes": attrs}


# --- rank_results -----------------------------------------------------------------------------


def test_rank_puts_hash_matches_first() -> None:
    ranked = rank_results([result(1, downloads=999), result(2, hash_match=True)])
    assert [r.file_id for r in ranked] == [2, 1]


def test_rank_prefers_not_foreign_parts_only_after_hash_match() -> None:
    ranked = rank_results([result(1, fpo=True, downloads=999), result(2, downloads=1)])
    assert [r.file_id for r in ranked] == [2, 1]
    ranked = rank_results([result(1, hash_match=True, fpo=True), result(2, hash_match=True)])
    assert [r.file_id for r in ranked] == [2, 1]


def test_rank_orders_by_download_count_descending() -> None:
    ranked = rank_results([result(1, downloads=5), result(2, downloads=50), result(3, downloads=9)])
    assert [r.file_id for r in ranked] == [2, 3, 1]


def test_rank_prefers_not_hearing_impaired_when_counts_tie() -> None:
    ranked = rank_results([result(1, downloads=5, hi=True), result(2, downloads=5)])
    assert [r.file_id for r in ranked] == [2, 1]


def test_rank_is_stable_and_leaves_its_input_alone() -> None:
    original = [result(1), result(2), result(3)]
    assert [r.file_id for r in rank_results(original)] == [1, 2, 3]
    assert [r.file_id for r in original] == [1, 2, 3]
    assert rank_results([]) == []


# --- host rule --------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "host",
    [
        "opensubtitles.com",
        "api.opensubtitles.com",
        "vip-api.opensubtitles.com",
        "A.OpenSubtitles.COM",
    ],
)
def test_host_rule_accepts_the_domain_and_its_subdomains(host: str) -> None:
    assert is_opensubtitles_host(host)


@pytest.mark.parametrize(
    "host",
    [
        "evilopensubtitles.com",
        "opensubtitles.com.evil.net",
        "opensubtitles.com.",
        "xn--opensubtitles.com",
        "api.opensubtitles.com@evil.net",
        "evil.net",
        "",
        None,
        ".opensubtitles.com",
        "api..opensubtitles.com",
        "opensubtitles.co",
        "127.0.0.1",
    ],
)
def test_host_rule_rejects_look_alikes(host: str | None) -> None:
    assert not is_opensubtitles_host(host)


# --- search -----------------------------------------------------------------------------------


def test_search_returns_only_usable_english_results_hash_matches_first() -> None:
    h = Harness(fx("login_ok.json"), search_ok())
    found = h.client.search(moviehash=HASH, title="Fake Movie", year=2001)
    # 5000002 hash match; then 5000001 and 5000005 (120 downloads, the hearing-impaired one
    # last); then 5000004 (foreign parts only). The French entry, the two-file entry and the
    # entry without a file are never returned.
    assert [r.file_id for r in found] == [5000002, 5000001, 5000005, 5000004]
    assert found[0].hash_match
    assert not any(r.hash_match for r in found[1:])
    h.done()


def test_search_maps_the_fields_and_survives_null_ones() -> None:
    h = Harness(fx("login_ok.json"), search_ok())
    by_id = {r.file_id: r for r in h.client.search(moviehash=HASH, title=None, year=None)}
    first = by_id[5000001]
    assert first == SubtitleResult(
        file_id=5000001,
        release="Fake.Movie.2001.1080p.FAKE.release-GROUPA",
        title="Fake Movie",
        year=2001,
        download_count=120,
        hash_match=False,
        hearing_impaired=False,
        foreign_parts_only=False,
    )
    nulls = by_id[5000005]
    assert nulls.year is None
    assert nulls.title == "Fake Movie"  # from movie_name, the title being null
    assert nulls.hearing_impaired
    assert by_id[5000004].foreign_parts_only


def test_search_asks_only_for_english_with_sorted_lowercase_parameters() -> None:
    h = Harness(fx("login_ok.json"), search_ok([]))
    assert h.client.search(moviehash=HASH, title="Fake Movie: The Sequel", year=2001) == []
    request = h.transport.requests[1]
    assert request.method == "GET"
    assert request.host == "api.opensubtitles.com"
    assert request.path == "/api/v1/subtitles"
    raw_query = urllib.parse.urlsplit(request.url).query
    names = [pair.split("=")[0] for pair in raw_query.split("&")]
    assert names == sorted(names)
    assert request.query == {
        "languages": ["en"],
        "moviehash": [HASH],
        "query": ["fake movie: the sequel"],
        "year": ["2001"],
    }


def test_search_by_title_only_and_by_hash_only() -> None:
    h = Harness(fx("login_ok.json"), search_ok([]), search_ok([]))
    h.client.search(moviehash=None, title="Fake Movie", year=None)
    h.client.search(moviehash=HASH, title=None, year=2001)
    assert h.transport.requests[1].query == {"languages": ["en"], "query": ["fake movie"]}
    # A year only narrows a title search.
    assert h.transport.requests[2].query == {"languages": ["en"], "moviehash": [HASH]}


@pytest.mark.parametrize(
    ("moviehash", "title"),
    [(None, None), (None, "   "), ("xyz", None), ("0123456789ABCDEF", None), (HASH + "0", None)],
)
def test_search_rejects_unusable_arguments(moviehash: str | None, title: str | None) -> None:
    h = Harness()
    with pytest.raises(ValueError):
        h.client.search(moviehash=moviehash, title=title, year=None)
    assert h.transport.requests == []


def test_search_filters_out_other_languages_even_when_the_server_sends_them() -> None:
    data = [item(1, language="fr"), item(2, language="EN"), item(3, language="en-GB"), item(4)]
    h = Harness(fx("login_ok.json"), search_ok(data))
    assert [r.file_id for r in h.client.search(moviehash=HASH, title=None, year=None)] == [2, 4]


def test_search_skips_malformed_items_but_keeps_the_good_ones(
    oslog: pytest.LogCaptureFixture,
) -> None:
    data: list[object] = [
        "not an object",
        None,
        {"attributes": "nope"},
        {"attributes": {"language": "en", "files": "nope"}},
        {"attributes": {"language": "en", "files": [None]}},
        {"attributes": {"language": "en", "files": [{"file_id": "5"}]}},
        {"attributes": {"language": "en", "files": [{"file_id": True}]}},
        {"attributes": {"language": "en", "files": [{"file_id": 10**30}]}},
        {"attributes": {"language": "en", "files": [{"file_id": 0}]}},
        {"attributes": {"language": None, "files": [{"file_id": 3}]}},
        item(9),
    ]
    h = Harness(fx("login_ok.json"), search_ok(data))
    assert [r.file_id for r in h.client.search(moviehash=HASH, title=None, year=None)] == [9]
    assert any("skipped 10" in r.getMessage() for r in oslog.records)


def test_search_with_only_malformed_items_is_a_provider_error() -> None:
    h = Harness(fx("login_ok.json"), search_ok(["x", 1, None]))
    with pytest.raises(ProviderError):
        h.client.search(moviehash=HASH, title=None, year=None)


def test_search_clamps_odd_numbers_and_cleans_text() -> None:
    nasty = "Rel\x1b[31mease\x00‮" + "x" * 1000
    data = [
        item(1, download_count=10**30, release=nasty),
        item(2, download_count=-5, feature_details={"year": 99999, "title": 5}),
        item(3, download_count=True, feature_details="nope"),
        item(4, download_count=1.5, release=7),
    ]
    h = Harness(fx("login_ok.json"), search_ok(data))
    by_id = {r.file_id: r for r in h.client.search(moviehash=HASH, title=None, year=None)}
    assert by_id[1].download_count <= 10**12
    assert "\x1b" not in by_id[1].release
    assert "\x00" not in by_id[1].release
    assert "‮" not in by_id[1].release
    assert len(by_id[1].release) <= 300
    assert by_id[2].download_count == 0
    assert by_id[2].year is None
    assert by_id[2].title == ""
    assert by_id[3].download_count == 0
    assert by_id[4].download_count == 0
    assert by_id[4].release == ""


@pytest.mark.parametrize(
    "body",
    [
        b"",
        b"not json",
        b"\xff\xfe\x00",
        b"[]",
        b'"text"',
        b"null",
        b"{}",
        b'{"data": null}',
        b'{"data": {"a": 1}}',
        b'{"data": "x"}',
        b'{"data": [NaN]}',
        b"[" * 100000,
        b'{"data": ' + b"[" * 100000,
        b"9" * 5000,
    ],
)
def test_search_hostile_bodies_raise_provider_error(body: bytes) -> None:
    h = Harness(fx("login_ok.json"), reply(200, body))
    with pytest.raises(ProviderError):
        h.client.search(moviehash=HASH, title=None, year=None)


def test_search_body_over_the_cap_is_a_provider_error() -> None:
    h = Harness(fx("login_ok.json"), reply(200, b" " * (MAX_JSON_BYTES + 1)))
    with pytest.raises(ProviderError):
        h.client.search(moviehash=HASH, title=None, year=None)
    # The transport was told the cap, so a real one never reads more.
    assert h.transport.requests[1].max_bytes == MAX_JSON_BYTES


def test_search_oversize_signalled_by_the_transport_is_a_provider_error() -> None:
    h = Harness(fx("login_ok.json"), ResponseTooLarge("too big"))
    with pytest.raises(ProviderError):
        h.client.search(moviehash=HASH, title=None, year=None)
    assert len(h.transport.requests) == 2  # not retried


def test_search_a_406_is_a_provider_error_not_a_quota_signal() -> None:
    h = Harness(fx("login_ok.json"), fx("download_quota_406.json", 406))
    with pytest.raises(ProviderError):
        h.client.search(moviehash=HASH, title=None, year=None)


@pytest.mark.parametrize("status", [400, 403, 404, 410, 422])
def test_other_client_errors_fail_at_once(status: int) -> None:
    h = Harness(fx("login_ok.json"), reply(status, {"message": "FAKE"}))
    with pytest.raises(ProviderError, match=str(status)):
        h.client.search(moviehash=HASH, title=None, year=None)
    assert len(h.transport.requests) == 2
    assert h.sleeps == []


# --- headers, session and secrets -------------------------------------------------------------


def test_every_request_carries_the_key_user_agent_and_bearer_token() -> None:
    h = Harness(
        fx("login_ok.json"),
        search_ok([item(7)]),
        fx("download_ok.json"),
        reply(200, b"1\n00:00:01,000 --> 00:00:02,000\nHi\n"),
    )
    found = h.client.search(moviehash=HASH, title=None, year=None)
    h.client.download(found[0], max_bytes=1000)
    login, search, info, link = h.transport.requests
    for request in (login, search, info, link):
        assert request.headers["User-Agent"] == f"video-cleaner v{__version__}"
        assert request.timeout_s > 0
    for request in (login, search, info):
        assert request.headers["Api-Key"] == API_KEY
        assert request.headers["Accept"] == "application/json"
    assert "Authorization" not in login.headers
    assert search.headers["Authorization"] == f"Bearer {SESSION_TOKEN}"
    assert info.headers["Authorization"] == f"Bearer {SESSION_TOKEN}"
    assert info.headers["Content-Type"] == "application/json"
    # The temporary download link is fetched with nothing but the User-Agent.
    assert set(link.headers) == {"User-Agent"}


def test_user_agent_carries_no_personal_data() -> None:
    h = Harness(fx("login_ok.json"), search_ok([]))
    h.client.search(moviehash=HASH, title=None, year=None)
    agent = h.transport.requests[0].headers["User-Agent"]
    assert agent == f"video-cleaner v{__version__}"
    assert USERNAME not in agent


def test_first_call_logs_in_and_stores_the_token_later_calls_do_not() -> None:
    h = Harness(fx("login_ok.json"), search_ok([]), search_ok([]))
    h.client.search(moviehash=HASH, title=None, year=None)
    h.client.search(moviehash=HASH, title=None, year=None)
    assert h.transport.paths() == ["/api/v1/login", "/api/v1/subtitles", "/api/v1/subtitles"]
    login = h.transport.requests[0]
    assert login.method == "POST"
    assert login.headers["Content-Type"] == "application/json"
    assert login.json_body() == {"username": USERNAME, "password": PASSWORD}
    assert h.store.values[TOKEN] == SESSION_TOKEN


def test_a_stored_token_is_used_without_logging_in() -> None:
    h = Harness(search_ok([]), token="FAKE-STORED-TOKEN-1111")
    h.client.search(moviehash=HASH, title=None, year=None)
    assert h.transport.paths() == ["/api/v1/subtitles"]
    assert h.transport.requests[0].headers["Authorization"] == "Bearer FAKE-STORED-TOKEN-1111"


def test_a_malformed_stored_token_is_ignored() -> None:
    h = Harness(fx("login_ok.json"), search_ok([]), token="has a space\r\nand a newline")
    h.client.search(moviehash=HASH, title=None, year=None)
    assert h.transport.paths() == ["/api/v1/login", "/api/v1/subtitles"]
    assert h.store.values[TOKEN] == SESSION_TOKEN


def test_the_credentials_are_registered_for_redaction_when_the_client_is_built() -> None:
    Harness()
    text = redact(f"{API_KEY} {USERNAME} {PASSWORD}")
    assert_no_secret(text)


def test_the_token_is_registered_the_moment_it_is_received() -> None:
    h = Harness(fx("login_ok.json"), search_ok([]))
    assert SESSION_TOKEN in redact(f"x {SESSION_TOKEN}")  # nothing registered it yet
    h.client.search(moviehash=HASH, title=None, year=None)
    assert SESSION_TOKEN not in redact(f"seen {SESSION_TOKEN}")


def test_the_stored_token_is_registered_when_it_is_read() -> None:
    h = Harness(search_ok([]), token="FAKE-STORED-TOKEN-1111")
    assert "FAKE-STORED-TOKEN-1111" in redact("x FAKE-STORED-TOKEN-1111")  # not read yet
    h.client.search(moviehash=HASH, title=None, year=None)
    assert "FAKE-STORED-TOKEN-1111" not in redact("seen FAKE-STORED-TOKEN-1111")


def test_no_secret_in_any_url_or_log_record_over_a_full_session(
    oslog: pytest.LogCaptureFixture,
) -> None:
    h = Harness(
        fx("login_ok.json"),
        search_ok(),
        fx("download_ok.json"),
        reply(200, b"subtitle text"),
        fx("login_ok.json"),
        fx("error_401.json", 401),
    )
    found = h.client.search(moviehash=HASH, title="Fake Movie", year=2001)
    h.client.download(found[0], max_bytes=100)
    for request in h.transport.requests:
        assert_no_secret(request.url)
    assert_logs_clean(oslog)


def test_no_secret_in_the_log_or_the_error_when_things_go_wrong(
    oslog: pytest.LogCaptureFixture,
) -> None:
    h = Harness(
        fx("login_ok.json"),
        fx("error_401.json", 401),
        fx("login_ok.json"),
        fx("error_401.json", 401),
    )
    with pytest.raises(AuthError) as excinfo:
        h.client.search(moviehash=HASH, title=None, year=None)
    assert_no_secret(str(excinfo.value))
    assert_no_secret(repr(excinfo.value))
    assert_logs_clean(oslog)


def test_a_401_triggers_exactly_one_relogin_and_a_retry() -> None:
    h = Harness(
        fx("error_401.json", 401),
        fx("login_ok.json"),
        search_ok([item(7)]),
        token="FAKE-STALE-TOKEN-2222",
    )
    found = h.client.search(moviehash=HASH, title=None, year=None)
    assert [r.file_id for r in found] == [7]
    assert h.transport.paths() == ["/api/v1/subtitles", "/api/v1/login", "/api/v1/subtitles"]
    assert h.transport.requests[0].headers["Authorization"] == "Bearer FAKE-STALE-TOKEN-2222"
    assert h.transport.requests[2].headers["Authorization"] == f"Bearer {SESSION_TOKEN}"
    assert h.store.values[TOKEN] == SESSION_TOKEN
    h.done()


def test_a_second_401_after_the_relogin_is_an_auth_error_without_another_login() -> None:
    h = Harness(
        fx("error_401.json", 401),
        fx("login_ok.json"),
        fx("error_401.json", 401),
        token="FAKE-STALE-TOKEN-2222",
    )
    with pytest.raises(AuthError):
        h.client.search(moviehash=HASH, title=None, year=None)
    assert h.transport.paths().count("/api/v1/login") == 1
    assert len(h.transport.requests) == 3


def test_a_401_straight_after_a_fresh_login_is_not_retried_with_another_login() -> None:
    h = Harness(fx("login_ok.json"), fx("error_401.json", 401))
    with pytest.raises(AuthError):
        h.client.search(moviehash=HASH, title=None, year=None)
    assert h.transport.paths() == ["/api/v1/login", "/api/v1/subtitles"]


def test_a_rejected_login_is_an_auth_error_and_leaves_no_token_behind() -> None:
    h = Harness(fx("error_401.json", 401), token=None)
    with pytest.raises(AuthError):
        h.client.search(moviehash=HASH, title=None, year=None)
    assert len(h.transport.requests) == 1
    assert TOKEN not in h.store.values


def test_a_stale_stored_token_is_dropped_when_the_relogin_fails() -> None:
    h = Harness(fx("error_401.json", 401), fx("error_401.json", 401), token="FAKE-STALE-TOKEN-2222")
    with pytest.raises(AuthError):
        h.client.search(moviehash=HASH, title=None, year=None)
    assert TOKEN not in h.store.values


def test_a_credential_store_that_fails_does_not_stop_the_search() -> None:
    h = Harness(fx("login_ok.json"), search_ok([item(7)]))
    h.store.fail_get = True
    h.store.fail_set = True
    assert [r.file_id for r in h.client.search(moviehash=HASH, title=None, year=None)] == [7]


def test_credentials_with_control_characters_are_refused_before_any_request() -> None:
    transport = FakeTransport()
    with pytest.raises(AuthError):
        OpenSubtitlesClient(
            OpenSubtitlesCredentials("key\r\nX-Evil: 1", USERNAME, PASSWORD),
            MemoryStore(),
            transport,
            RecordingReporter(),
        )
    assert transport.requests == []


# --- login response ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "base_url",
    [
        "http://api.opensubtitles.com",
        "evilopensubtitles.com",
        "https://evilopensubtitles.com/api/v1",
        "api.opensubtitles.com.evil.net",
        "https://api.opensubtitles.com.evil.net/api/v1",
        "https://evil.example",
        "https://api.opensubtitles.com@evil.example/",
        "https://evil.example@api.opensubtitles.com/",
        "https://api.opensubtitles.com:8443/api/v1",
        "ftp://api.opensubtitles.com",
        "file:///etc/passwd",
        "https://api.opensubtitles.com\\@evil.example",
        "https://api.opensubtitles.com/ evil",
        "",
        "   ",
        "https://",
        5,
        ["api.opensubtitles.com"],
        "a" * 5000,
    ],
)
def test_a_login_base_url_outside_opensubtitles_is_an_auth_error(base_url: object) -> None:
    body = dict(fixture_json("login_ok.json"), base_url=base_url)
    h = Harness(reply(200, body))
    with pytest.raises(AuthError):
        h.client.search(moviehash=HASH, title=None, year=None)
    assert len(h.transport.requests) == 1  # nothing was sent to the odd host
    assert TOKEN not in h.store.values  # and the token that came with it is not kept


@pytest.mark.parametrize(
    ("base_url", "expected_host"),
    [
        ("api.opensubtitles.com", "api.opensubtitles.com"),
        ("vip-api.opensubtitles.com", "vip-api.opensubtitles.com"),
        ("https://vip-api.opensubtitles.com/api/v1", "vip-api.opensubtitles.com"),
        ("https://VIP-API.opensubtitles.com:443", "vip-api.opensubtitles.com"),
        (None, "api.opensubtitles.com"),
    ],
)
def test_a_login_base_url_inside_opensubtitles_is_used_for_later_requests(
    base_url: str | None, expected_host: str
) -> None:
    body = dict(fixture_json("login_ok.json"), base_url=base_url)
    h = Harness(reply(200, body), search_ok([]))
    h.client.search(moviehash=HASH, title=None, year=None)
    assert h.transport.requests[0].host == "api.opensubtitles.com"  # the login itself
    assert h.transport.requests[1].host == expected_host
    assert h.transport.requests[1].path == "/api/v1/subtitles"
    assert urllib.parse.urlsplit(h.transport.requests[1].url).scheme == "https"


def test_a_login_without_base_url_uses_the_default_host() -> None:
    body = {k: v for k, v in fixture_json("login_ok.json").items() if k != "base_url"}
    h = Harness(reply(200, body), search_ok([]))
    h.client.search(moviehash=HASH, title=None, year=None)
    assert h.transport.requests[1].host == "api.opensubtitles.com"


@pytest.mark.parametrize(
    "body",
    [
        b"",
        b"not json",
        b"[]",
        b"{}",
        b'{"token": null}',
        b'{"token": 5}',
        b'{"token": ""}',
        b'{"token": "has space"}',
        b'{"token": "line\\nbreak"}',
        b'{"token": "' + b"a" * 5000 + b'"}',
        b'{"token": "\\u00e9"}',
    ],
)
def test_a_malformed_login_response_is_a_provider_error(body: bytes) -> None:
    h = Harness(reply(200, body))
    with pytest.raises(ProviderError):
        h.client.search(moviehash=HASH, title=None, year=None)
    assert TOKEN not in h.store.values


@pytest.mark.parametrize("status", [400, 401, 403, 406])
def test_a_login_refused_with_a_client_error_is_an_auth_error(status: int) -> None:
    h = Harness(reply(status, {"message": "FAKE"}))
    with pytest.raises(AuthError):
        h.client.search(moviehash=HASH, title=None, year=None)
    assert h.sleeps == []


def test_a_login_that_keeps_failing_with_5xx_is_a_provider_error() -> None:
    h = Harness(*[reply(503)] * (MAX_RETRIES + 1))
    with pytest.raises(ProviderError):
        h.client.search(moviehash=HASH, title=None, year=None)


# --- retries ----------------------------------------------------------------------------------


def test_429_sleeps_retry_after_and_retries_then_succeeds() -> None:
    h = Harness(
        fx("login_ok.json"),
        reply(429, b"", {"Retry-After": "3"}),
        reply(429, b"", {"retry-after": "2"}),
        search_ok([item(7)]),
    )
    assert [r.file_id for r in h.client.search(moviehash=HASH, title=None, year=None)] == [7]
    assert h.sleeps == [3.0, 2.0]
    h.done()


def test_429_gives_up_after_three_retries_with_a_provider_error() -> None:
    h = Harness(
        fx("login_ok.json"),
        *[reply(429, b"", {"Retry-After": "1"}) for _ in range(MAX_RETRIES + 1)],
    )
    with pytest.raises(ProviderError, match="429"):
        h.client.search(moviehash=HASH, title=None, year=None)
    assert len(h.transport.requests) == 1 + MAX_RETRIES + 1
    assert h.sleeps == [1.0] * MAX_RETRIES
    assert MAX_RETRIES == 3


def test_429_without_retry_after_backs_off_exponentially() -> None:
    h = Harness(
        fx("login_ok.json"),
        *[reply(429) for _ in range(MAX_RETRIES + 1)],
    )
    with pytest.raises(ProviderError):
        h.client.search(moviehash=HASH, title=None, year=None)
    assert h.sleeps == [1.0, 2.0, 4.0]


@pytest.mark.parametrize("value", ["soon", "-5", "nan", "inf", "", "1e999", "0x10"])
def test_an_unusable_retry_after_falls_back_to_the_backoff(value: str) -> None:
    h = Harness(fx("login_ok.json"), reply(429, b"", {"Retry-After": value}), search_ok([item(7)]))
    h.client.search(moviehash=HASH, title=None, year=None)
    assert h.sleeps == [1.0]


def test_a_huge_retry_after_is_clamped() -> None:
    h = Harness(
        fx("login_ok.json"), reply(429, b"", {"Retry-After": "999999"}), search_ok([item(7)])
    )
    h.client.search(moviehash=HASH, title=None, year=None)
    assert len(h.sleeps) == 1
    assert 0 < h.sleeps[0] <= 120


def test_retry_after_may_be_an_http_date() -> None:
    when = (NOW + timedelta(seconds=30)).strftime("%a, %d %b %Y %H:%M:%S GMT")
    h = Harness(fx("login_ok.json"), reply(429, b"", {"Retry-After": when}), search_ok([item(7)]))
    h.client.search(moviehash=HASH, title=None, year=None)
    assert h.sleeps == [30.0]


@pytest.mark.parametrize("status", [500, 502, 503, 504])
def test_5xx_is_retried_three_times_then_a_provider_error(status: int) -> None:
    h = Harness(fx("login_ok.json"), *[reply(status) for _ in range(MAX_RETRIES + 1)])
    with pytest.raises(ProviderError, match=str(status)):
        h.client.search(moviehash=HASH, title=None, year=None)
    assert len(h.transport.requests) == 1 + MAX_RETRIES + 1
    assert h.sleeps == [1.0, 2.0, 4.0]


def test_5xx_then_success_returns_the_results() -> None:
    h = Harness(fx("login_ok.json"), reply(503), reply(502), search_ok([item(7)]))
    assert [r.file_id for r in h.client.search(moviehash=HASH, title=None, year=None)] == [7]
    assert h.sleeps == [1.0, 2.0]


def test_timeouts_are_retried_three_times_then_a_provider_error() -> None:
    h = Harness(
        fx("login_ok.json"), *[TransportTimeout("timed out") for _ in range(MAX_RETRIES + 1)]
    )
    with pytest.raises(ProviderError, match="timed out"):
        h.client.search(moviehash=HASH, title=None, year=None)
    assert len(h.transport.requests) == 1 + MAX_RETRIES + 1
    assert h.sleeps == [1.0, 2.0, 4.0]


def test_a_timeout_then_success_returns_the_results() -> None:
    h = Harness(fx("login_ok.json"), TransportTimeout("t"), search_ok([item(7)]))
    assert [r.file_id for r in h.client.search(moviehash=HASH, title=None, year=None)] == [7]


def test_connection_errors_are_retried_like_timeouts() -> None:
    h = Harness(fx("login_ok.json"), TransportError("reset"), search_ok([item(7)]))
    assert [r.file_id for r in h.client.search(moviehash=HASH, title=None, year=None)] == [7]
    h2 = Harness(fx("login_ok.json"), *[TransportError("reset") for _ in range(MAX_RETRIES + 1)])
    with pytest.raises(ProviderError):
        h2.client.search(moviehash=HASH, title=None, year=None)


def test_every_request_has_a_timeout() -> None:
    h = Harness(fx("login_ok.json"), search_ok([]))
    h.client.search(moviehash=HASH, title=None, year=None)
    assert all(0 < r.timeout_s <= 120 for r in h.transport.requests)


# --- progress ---------------------------------------------------------------------------------


def test_each_network_call_is_a_reported_stage() -> None:
    h = Harness(
        fx("login_ok.json"), search_ok([item(7)]), fx("download_ok.json"), reply(200, b"data")
    )
    found = h.client.search(moviehash=HASH, title=None, year=None)
    h.client.download(found[0], max_bytes=100)
    assert h.reporter.stages == [
        "OpenSubtitles login",
        "OpenSubtitles search",
        "OpenSubtitles download",
        "OpenSubtitles download",
    ]


# --- download ---------------------------------------------------------------------------------


def a_result(file_id: int = 7) -> SubtitleResult:
    return result(file_id)


def test_download_returns_the_body_and_asks_for_the_file_id_only() -> None:
    h = Harness(fx("login_ok.json"), fx("download_ok.json"), reply(200, b"subtitle bytes"))
    assert h.client.download(a_result(42), max_bytes=1000) == b"subtitle bytes"
    _, info, link = h.transport.requests
    assert info.method == "POST"
    assert info.path == "/api/v1/download"
    assert info.json_body() == {"file_id": 42}
    assert link.method == "GET"
    assert link.host == "dl.opensubtitles.com"
    assert link.max_bytes == 1000
    assert link.url == fixture_json("download_ok.json")["link"]


def test_download_never_uses_the_server_file_name(oslog: pytest.LogCaptureFixture) -> None:
    assert SERVER_FILE_NAME in fixture_bytes("download_ok.json").decode()
    h = Harness(fx("login_ok.json"), fx("download_ok.json"), reply(200, b"data"))
    h.client.download(a_result(), max_bytes=100)
    for request in h.transport.requests:
        assert SERVER_FILE_NAME not in request.url
        assert SERVER_FILE_NAME not in (request.body or b"").decode()
    assert_logs_clean(oslog)
    assert all(SERVER_FILE_NAME not in stage for stage in h.reporter.stages)


def test_download_rejects_a_body_over_max_bytes_even_if_the_transport_did_not() -> None:
    h = Harness(fx("login_ok.json"), fx("download_ok.json"), reply(200, b"x" * 101))
    with pytest.raises(ProviderError, match="larger"):
        h.client.download(a_result(), max_bytes=100)


def test_download_rejects_a_body_the_transport_stopped_reading() -> None:
    h = Harness(fx("login_ok.json"), fx("download_ok.json"), ResponseTooLarge("cap"))
    with pytest.raises(ProviderError, match="larger"):
        h.client.download(a_result(), max_bytes=100)
    assert len(h.transport.requests) == 3  # no retry of an oversize body


def test_download_accepts_a_body_of_exactly_max_bytes() -> None:
    h = Harness(fx("login_ok.json"), fx("download_ok.json"), reply(200, b"x" * 100))
    assert len(h.client.download(a_result(), max_bytes=100)) == 100


def test_download_rejects_an_empty_body() -> None:
    h = Harness(fx("login_ok.json"), fx("download_ok.json"), reply(200, b""))
    with pytest.raises(ProviderError, match="empty"):
        h.client.download(a_result(), max_bytes=100)


@pytest.mark.parametrize("max_bytes", [0, -1])
def test_download_needs_a_positive_cap(max_bytes: int) -> None:
    h = Harness()
    with pytest.raises(ValueError):
        h.client.download(a_result(), max_bytes=max_bytes)
    assert h.transport.requests == []


@pytest.mark.parametrize(
    "link",
    [
        "http://dl.opensubtitles.com/x.srt",
        "https://evil.example/x.srt",
        "https://evilopensubtitles.com/x.srt",
        "https://opensubtitles.com.evil.net/x.srt",
        "https://dl.opensubtitles.com@evil.example/x.srt",
        "https://evil.example@dl.opensubtitles.com/x.srt",
        "https://dl.opensubtitles.com:8443/x.srt",
        "https://dl.opensubtitles.com\\@evil.example/x.srt",
        "ftp://dl.opensubtitles.com/x.srt",
        "file:///etc/passwd",
        "//dl.opensubtitles.com/x.srt",
        "/relative/x.srt",
        "https://dl.opensubtitles.com/x y.srt",
        "https://dl.opensubtitles.com/x\r\nHost: evil.example",
        "",
        "https://dl.opensubtitles.com/" + "a" * 5000,
        None,
        5,
        ["https://dl.opensubtitles.com/x.srt"],
    ],
)
def test_download_refuses_a_link_outside_opensubtitles(link: object) -> None:
    info = dict(fixture_json("download_ok.json"), link=link)
    h = Harness(fx("login_ok.json"), reply(200, info))
    with pytest.raises(ProviderError):
        h.client.download(a_result(), max_bytes=100)
    assert len(h.transport.requests) == 2  # the link was never requested


def test_download_without_a_link_key_is_a_provider_error() -> None:
    info = {k: v for k, v in fixture_json("download_ok.json").items() if k != "link"}
    h = Harness(fx("login_ok.json"), reply(200, info))
    with pytest.raises(ProviderError):
        h.client.download(a_result(), max_bytes=100)


@pytest.mark.parametrize("body", [b"", b"nope", b"[]", b"null", b'{"link": null}', b"[" * 100000])
def test_download_hostile_info_bodies_raise_provider_error(body: bytes) -> None:
    h = Harness(fx("login_ok.json"), reply(200, body))
    with pytest.raises(ProviderError):
        h.client.download(a_result(), max_bytes=100)


def test_download_link_failure_statuses() -> None:
    h = Harness(fx("login_ok.json"), fx("download_ok.json"), reply(404))
    with pytest.raises(ProviderError, match="404"):
        h.client.download(a_result(), max_bytes=100)


def test_download_link_is_retried_on_5xx_and_429() -> None:
    h = Harness(
        fx("login_ok.json"),
        fx("download_ok.json"),
        reply(503),
        reply(429, b"", {"Retry-After": "2"}),
        reply(200, b"data"),
    )
    assert h.client.download(a_result(), max_bytes=100) == b"data"
    assert h.sleeps == [1.0, 2.0]


def test_download_link_401_is_not_a_reason_to_log_in_again() -> None:
    h = Harness(fx("login_ok.json"), fx("download_ok.json"), reply(401))
    with pytest.raises(ProviderError, match="401"):
        h.client.download(a_result(), max_bytes=100)
    assert h.transport.paths().count("/api/v1/login") == 1


def test_download_info_401_relogs_in_once() -> None:
    h = Harness(
        fx("error_401.json", 401),
        fx("login_ok.json"),
        fx("download_ok.json"),
        reply(200, b"data"),
        token="FAKE-STALE-TOKEN-2222",
    )
    assert h.client.download(a_result(), max_bytes=100) == b"data"
    assert h.transport.paths()[:3] == ["/api/v1/download", "/api/v1/login", "/api/v1/download"]


# --- quota ------------------------------------------------------------------------------------


def test_last_quota_is_unknown_before_any_download() -> None:
    assert Harness().client.last_quota() is None


def test_last_quota_reflects_remaining_and_reset_of_the_last_download() -> None:
    h = Harness(fx("login_ok.json"), fx("download_ok.json"), reply(200, b"data"))
    h.client.download(a_result(), max_bytes=100)
    quota = h.client.last_quota()
    assert quota == QuotaInfo(97, datetime(2030, 1, 2, tzinfo=UTC))
    assert quota is not None
    assert quota.reset_at.tzinfo is not None


def test_last_quota_follows_the_most_recent_download() -> None:
    second = dict(fixture_json("download_ok.json"), remaining=96)
    h = Harness(
        fx("login_ok.json"),
        fx("download_ok.json"),
        reply(200, b"data"),
        reply(200, second),
        reply(200, b"data"),
    )
    h.client.download(a_result(), max_bytes=100)
    h.client.download(a_result(), max_bytes=100)
    quota = h.client.last_quota()
    assert quota is not None
    assert quota.remaining == 96


def test_a_406_raises_quota_exhausted_with_the_reset_time() -> None:
    h = Harness(fx("login_ok.json"), fx("download_quota_406.json", 406))
    with pytest.raises(QuotaExhausted) as excinfo:
        h.client.download(a_result(), max_bytes=100)
    assert excinfo.value.reset_at == datetime(2030, 1, 2, tzinfo=UTC)
    assert h.client.last_quota() == QuotaInfo(0, datetime(2030, 1, 2, tzinfo=UTC))
    assert len(h.transport.requests) == 2  # no retry, no link request
    assert h.sleeps == []


@pytest.mark.parametrize("body", [b"", b"<html>", b"[]", b'{"reset_time_utc": "soon"}', b"null"])
def test_a_406_with_an_unreadable_body_is_still_quota_exhausted(body: bytes) -> None:
    h = Harness(fx("login_ok.json"), reply(406, body))
    with pytest.raises(QuotaExhausted) as excinfo:
        h.client.download(a_result(), max_bytes=100)
    assert excinfo.value.reset_at is None
    assert h.client.last_quota() is None


def test_a_negative_remaining_raises_quota_exhausted_even_with_a_link() -> None:
    info = dict(fixture_json("download_ok.json"), remaining=-1)
    h = Harness(fx("login_ok.json"), reply(200, info))
    with pytest.raises(QuotaExhausted) as excinfo:
        h.client.download(a_result(), max_bytes=100)
    assert excinfo.value.reset_at == datetime(2030, 1, 2, tzinfo=UTC)
    assert len(h.transport.requests) == 2


def test_the_last_allowed_download_is_delivered_and_the_quota_reads_zero() -> None:
    # remaining counts what is left after this download, so 0 is the last one that works.
    info = dict(fixture_json("download_ok.json"), remaining=0)
    h = Harness(fx("login_ok.json"), reply(200, info), reply(200, b"data"))
    assert h.client.download(a_result(), max_bytes=100) == b"data"
    assert h.client.last_quota() == QuotaInfo(0, datetime(2030, 1, 2, tzinfo=UTC))


@pytest.mark.parametrize(
    "overrides",
    [
        {"remaining": "97"},
        {"remaining": None},
        {"remaining": 10**30},
        {"remaining": True},
        {"reset_time_utc": None},
        {"reset_time_utc": "not a time"},
        {"reset_time_utc": 5},
        {"reset_time_utc": "0000-00-00T00:00:00Z"},
    ],
)
def test_odd_quota_fields_never_break_a_download(overrides: dict[str, object]) -> None:
    info = dict(fixture_json("download_ok.json"), **overrides)
    h = Harness(fx("login_ok.json"), reply(200, info), reply(200, b"data"))
    assert h.client.download(a_result(), max_bytes=100) == b"data"
    assert h.client.last_quota() is None


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("2030-01-02T00:00:00.000Z", datetime(2030, 1, 2, tzinfo=UTC)),
        ("2030-01-02T00:00:00Z", datetime(2030, 1, 2, tzinfo=UTC)),
        ("2030-01-02 00:00:00 UTC", datetime(2030, 1, 2, tzinfo=UTC)),
        ("2030-01-02T00:00:00", datetime(2030, 1, 2, tzinfo=UTC)),
        ("2030-01-02T02:00:00+02:00", datetime(2030, 1, 2, tzinfo=UTC)),
    ],
)
def test_reset_time_formats(text: str, expected: datetime) -> None:
    info = dict(fixture_json("download_ok.json"), reset_time_utc=text)
    h = Harness(fx("login_ok.json"), reply(200, info), reply(200, b"data"))
    h.client.download(a_result(), max_bytes=100)
    quota = h.client.last_quota()
    assert quota is not None
    assert quota.reset_at == expected


# --- smaller paths ----------------------------------------------------------------------------


@pytest.mark.parametrize("url", ["https://api.opensubtitles.com:abc/x", "https://[::1/x"])
def test_unparsable_urls_are_refused(url: str) -> None:
    assert https_host(url) is None


def test_the_default_clock_is_timezone_aware_utc() -> None:
    assert module._utc_now().utcoffset() == timedelta(0)


def test_a_naive_retry_after_date_counts_as_utc() -> None:
    when = (NOW + timedelta(seconds=45)).strftime("%a, %d %b %Y %H:%M:%S -0000")
    h = Harness(fx("login_ok.json"), reply(429, b"", {"Retry-After": when}), search_ok([item(7)]))
    h.client.search(moviehash=HASH, title=None, year=None)
    assert h.sleeps == [45.0]


def test_a_past_retry_after_date_sleeps_zero() -> None:
    when = (NOW - timedelta(seconds=45)).strftime("%a, %d %b %Y %H:%M:%S GMT")
    h = Harness(fx("login_ok.json"), reply(429, b"", {"Retry-After": when}), search_ok([item(7)]))
    h.client.search(moviehash=HASH, title=None, year=None)
    assert h.sleeps == [0.0]


def test_a_redirect_status_from_the_login_is_a_provider_error() -> None:
    h = Harness(reply(302, b"", {"Location": "https://evil.example/"}))
    with pytest.raises(ProviderError, match="302"):
        h.client.search(moviehash=HASH, title=None, year=None)


def test_download_info_with_an_unexpected_status_is_a_provider_error() -> None:
    h = Harness(fx("login_ok.json"), reply(403, {"message": "FAKE"}))
    with pytest.raises(ProviderError, match="403"):
        h.client.download(a_result(), max_bytes=100)


def test_a_store_that_cannot_delete_the_token_does_not_hide_the_auth_error() -> None:
    class Stuck(MemoryStore):
        def delete(self, name: str) -> None:
            raise CredentialError("keyring delete failed")

    h = Harness(fx("error_401.json", 401), fx("login_ok.json"), search_ok([item(7)]))
    h.store = Stuck(**{TOKEN: "FAKE-STALE-TOKEN-2222"})
    h.client._store = h.store
    assert [r.file_id for r in h.client.search(moviehash=HASH, title=None, year=None)] == [7]


# --- a download that may have been counted is never retried -----------------------------------


@pytest.mark.parametrize("error", [TransportTimeout("read timed out"), TransportError("reset")])
def test_the_download_post_is_sent_once_when_the_reply_never_came(error: TransportError) -> None:
    h = Harness(fx("login_ok.json"), error)
    with pytest.raises(ProviderError, match="not retried"):
        h.client.download(a_result(), max_bytes=100)
    assert h.transport.paths() == ["/api/v1/login", "/api/v1/download"]
    assert h.sleeps == []


@pytest.mark.parametrize("status", [500, 502, 503, 504])
def test_the_download_post_is_sent_once_when_the_server_answers_5xx(status: int) -> None:
    h = Harness(fx("login_ok.json"), reply(status))
    with pytest.raises(ProviderError, match=f"{status}.*not retried"):
        h.client.download(a_result(), max_bytes=100)
    assert h.transport.paths() == ["/api/v1/login", "/api/v1/download"]
    assert h.sleeps == []


@pytest.mark.parametrize(
    "error", [RequestNotSent("connect failed"), ConnectTimeout("connect timed out")]
)
def test_the_download_post_is_retried_when_the_request_never_left(error: TransportError) -> None:
    h = Harness(fx("login_ok.json"), error, fx("download_ok.json"), reply(200, b"data"))
    assert h.client.download(a_result(), max_bytes=100) == b"data"
    assert h.transport.paths().count("/api/v1/download") == 2
    assert h.sleeps == [1.0]


def test_the_download_post_gives_up_after_the_retries_when_it_never_connects() -> None:
    h = Harness(
        fx("login_ok.json"), *[RequestNotSent("connect failed") for _ in range(MAX_RETRIES + 1)]
    )
    with pytest.raises(ProviderError):
        h.client.download(a_result(), max_bytes=100)
    assert h.transport.paths().count("/api/v1/download") == MAX_RETRIES + 1


def test_the_download_post_is_retried_on_429() -> None:
    h = Harness(
        fx("login_ok.json"),
        reply(429, b"", {"Retry-After": "2"}),
        fx("download_ok.json"),
        reply(200, b"data"),
    )
    assert h.client.download(a_result(), max_bytes=100) == b"data"
    assert h.transport.paths().count("/api/v1/download") == 2
    assert h.sleeps == [2.0]


def test_the_download_link_is_still_retried_after_a_lost_reply() -> None:
    # The link costs nothing ("the file can be downloaded more than once"), so a lost reply is
    # simply asked for again.
    h = Harness(
        fx("login_ok.json"),
        fx("download_ok.json"),
        TransportTimeout("t"),
        TransportError("reset"),
        reply(503),
        reply(200, b"data"),
    )
    assert h.client.download(a_result(), max_bytes=100) == b"data"
    assert h.transport.paths().count("/api/v1/download") == 1
    assert h.sleeps == [1.0, 2.0, 4.0]


# --- a redirect that cannot be followed is final ----------------------------------------------


def test_an_unfollowable_redirect_is_a_provider_error_without_a_retry_at_login() -> None:
    h = Harness(RedirectRefused("redirect refused"))
    with pytest.raises(ProviderError, match="redirect"):
        h.client.search(moviehash=HASH, title=None, year=None)
    assert len(h.transport.requests) == 1
    assert h.sleeps == []


@pytest.mark.parametrize("operation", ["search", "download", "link"])
def test_an_unfollowable_redirect_is_a_provider_error_without_a_retry(operation: str) -> None:
    refused = RedirectRefused("redirect refused")
    script: list[HttpResponse | Exception] = [fx("login_ok.json")]
    if operation == "link":
        script.append(fx("download_ok.json"))
    script.append(refused)
    h = Harness(*script)
    with pytest.raises(ProviderError, match="redirect"):
        if operation == "search":
            h.client.search(moviehash=HASH, title=None, year=None)
        else:
            h.client.download(a_result(), max_bytes=100)
    assert len(h.transport.requests) == len(script)
    assert h.sleeps == []


# --- a refused login is remembered ------------------------------------------------------------


def test_after_a_refused_login_later_calls_make_no_request() -> None:
    h = Harness(fx("error_401.json", 401))
    with pytest.raises(AuthError):
        h.client.search(moviehash=HASH, title=None, year=None)
    assert len(h.transport.requests) == 1
    with pytest.raises(AuthError):
        h.client.search(moviehash=HASH, title=None, year=None)
    with pytest.raises(AuthError):
        h.client.download(a_result(), max_bytes=100)
    assert len(h.transport.requests) == 1


def test_the_remembered_refusal_keeps_its_text_and_carries_no_secret() -> None:
    h = Harness(fx("error_401.json", 401))
    with pytest.raises(AuthError) as first:
        h.client.search(moviehash=HASH, title=None, year=None)
    with pytest.raises(AuthError) as second:
        h.client.search(moviehash=HASH, title=None, year=None)
    assert str(second.value) == str(first.value)
    assert_no_secret(str(second.value))


def test_a_new_client_starts_without_the_refusal() -> None:
    first = Harness(fx("error_401.json", 401))
    with pytest.raises(AuthError):
        first.client.search(moviehash=HASH, title=None, year=None)
    fresh = Harness(fx("login_ok.json"), search_ok([item(7)]))
    assert [r.file_id for r in fresh.client.search(moviehash=HASH, title=None, year=None)] == [7]


def test_an_untrusted_base_url_is_remembered_too() -> None:
    body = dict(fixture_json("login_ok.json"), base_url="https://evil.example")
    h = Harness(reply(200, body))
    for _ in range(2):
        with pytest.raises(AuthError):
            h.client.search(moviehash=HASH, title=None, year=None)
    assert len(h.transport.requests) == 1


def test_a_session_token_the_server_refuses_twice_is_remembered() -> None:
    h = Harness(
        fx("error_401.json", 401),
        fx("login_ok.json"),
        fx("error_401.json", 401),
        token="FAKE-STALE-TOKEN-2222",
    )
    for _ in range(2):
        with pytest.raises(AuthError):
            h.client.search(moviehash=HASH, title=None, year=None)
    assert len(h.transport.requests) == 3


def test_a_provider_error_at_login_is_not_remembered() -> None:
    h = Harness(reply(200, b"nope"), fx("login_ok.json"), search_ok([item(7)]))
    with pytest.raises(ProviderError):
        h.client.search(moviehash=HASH, title=None, year=None)
    assert [r.file_id for r in h.client.search(moviehash=HASH, title=None, year=None)] == [7]


# --- a configurable endpoint ------------------------------------------------------------------

MOVED = ApiEndpoint(
    domain="opensubtitles.example",
    host="api.opensubtitles.example",
    path="/api/v2",
    language="fr",
)


def test_the_default_endpoint_is_the_current_service() -> None:
    default = ApiEndpoint()
    assert (default.domain, default.host, default.path, default.language) == (
        "opensubtitles.com",
        "api.opensubtitles.com",
        "/api/v1",
        "en",
    )
    assert default.base == "https://api.opensubtitles.com/api/v1"


def test_an_endpoint_comes_from_the_settings() -> None:
    from video_cleaner.config import Settings

    settings = Settings(
        library_root=Path("L"),
        scratch_dir=Path("S"),
        completed_dir=Path("C"),
        state_db=Path("d.db"),
        log_dir=Path("logs"),
        opensubtitles_domain="opensubtitles.example",
        opensubtitles_api_host="api.opensubtitles.example",
        opensubtitles_api_path="/api/v2",
        subtitle_language="fr",
    )
    assert ApiEndpoint.from_settings(settings) == MOVED


def test_a_moved_service_gets_the_login_and_the_search_at_the_new_address() -> None:
    login = dict(fixture_json("login_ok.json"), base_url="api.opensubtitles.example")
    h = Harness(reply(200, login), search_ok([]), endpoint=MOVED)
    h.client.search(moviehash=HASH, title=None, year=None)
    first, second = h.transport.requests
    assert (first.host, first.path) == ("api.opensubtitles.example", "/api/v2/login")
    assert (second.host, second.path) == ("api.opensubtitles.example", "/api/v2/subtitles")
    assert second.query["languages"] == ["fr"]


def test_a_moved_service_without_base_url_in_the_login_uses_the_configured_host() -> None:
    body = {k: v for k, v in fixture_json("login_ok.json").items() if k != "base_url"}
    h = Harness(reply(200, body), search_ok([]), endpoint=MOVED)
    h.client.search(moviehash=HASH, title=None, year=None)
    assert h.transport.requests[1].host == "api.opensubtitles.example"


def test_the_language_setting_picks_the_results_and_ignores_case() -> None:
    portuguese = ApiEndpoint(language="pt-br")
    data = [item(1, language="pt-BR"), item(2, language="en"), item(3, language="pt-br")]
    h = Harness(fx("login_ok.json"), search_ok(data), endpoint=portuguese)
    found = h.client.search(moviehash=HASH, title=None, year=None)
    assert sorted(r.file_id for r in found) == [1, 3]
    assert h.transport.requests[1].query["languages"] == ["pt-br"]


@pytest.mark.parametrize("base_url", ["api.opensubtitles.com", "https://vip-api.opensubtitles.com"])
def test_a_moved_service_refuses_a_login_that_names_the_old_domain(base_url: str) -> None:
    body = dict(fixture_json("login_ok.json"), base_url=base_url)
    h = Harness(reply(200, body), endpoint=MOVED)
    with pytest.raises(AuthError):
        h.client.search(moviehash=HASH, title=None, year=None)
    assert len(h.transport.requests) == 1


def test_a_moved_service_uses_its_own_subdomains_and_path_from_the_login() -> None:
    body = dict(fixture_json("login_ok.json"), base_url="vip-api.opensubtitles.example")
    h = Harness(reply(200, body), search_ok([]), endpoint=MOVED)
    h.client.search(moviehash=HASH, title=None, year=None)
    assert h.transport.requests[1].host == "vip-api.opensubtitles.example"
    assert h.transport.requests[1].path == "/api/v2/subtitles"


def test_a_moved_service_accepts_a_download_link_on_its_domain_only() -> None:
    good = dict(fixture_json("download_ok.json"), link="https://dl.opensubtitles.example/x.srt")
    login = dict(fixture_json("login_ok.json"), base_url="api.opensubtitles.example")
    h = Harness(reply(200, login), reply(200, good), reply(200, b"1\n"), endpoint=MOVED)
    assert h.client.download(result(7), max_bytes=1000) == b"1\n"
    assert h.transport.requests[2].host == "dl.opensubtitles.example"
    old = dict(fixture_json("download_ok.json"))  # a link on the old domain
    h = Harness(reply(200, login), reply(200, old), endpoint=MOVED)
    with pytest.raises(ProviderError):
        h.client.download(result(7), max_bytes=1000)


def test_the_host_rule_follows_the_domain_it_is_given() -> None:
    assert is_opensubtitles_host("api.opensubtitles.example", "opensubtitles.example")
    assert is_opensubtitles_host("OPENSUBTITLES.EXAMPLE", "opensubtitles.example")
    assert not is_opensubtitles_host("api.opensubtitles.com", "opensubtitles.example")
    assert not is_opensubtitles_host("evilopensubtitles.example", "opensubtitles.example")
    assert not is_opensubtitles_host("opensubtitles.example.evil.net", "opensubtitles.example")
    assert not is_opensubtitles_host("api.opensubtitles.example", "opensubtitles.com")
    assert https_host("https://dl.opensubtitles.example/x", "opensubtitles.example") == (
        "dl.opensubtitles.example"
    )
    assert https_host("https://dl.opensubtitles.com/x", "opensubtitles.example") is None
    assert https_host("https://dl.opensubtitles.example/x") is None  # the default domain
