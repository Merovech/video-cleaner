"""`UrllibTransport` driven through a real opener whose HTTPS handler is replaced by a fake.

The real redirect and error handlers run; only the network is faked, so no test needs a socket
or TLS.
"""

from __future__ import annotations

import email.message
import http.client
import io
import urllib.error
import urllib.request
import urllib.response
from collections.abc import Callable
from typing import BinaryIO

import pytest

from video_cleaner.subtitles.opensubtitles import (
    HttpResponse,
    ResponseTooLarge,
    TransportError,
    TransportTimeout,
    UrllibTransport,
)

Answer = tuple[int, dict[str, str], bytes | BinaryIO]


class FakeHTTPS(urllib.request.HTTPSHandler):
    def __init__(self, responder: Callable[[urllib.request.Request], Answer]) -> None:
        super().__init__()
        self.responder = responder
        self.seen: list[urllib.request.Request] = []

    def https_open(  # type: ignore[override]
        self, req: urllib.request.Request
    ) -> urllib.response.addinfourl:
        self.seen.append(req)
        code, headers, body = self.responder(req)
        message = email.message.Message()
        for name, value in headers.items():
            message[name] = value
        stream = io.BytesIO(body) if isinstance(body, bytes) else body
        response = urllib.response.addinfourl(stream, message, req.full_url, code)
        response.msg = http.client.responses.get(code, "")  # type: ignore[attr-defined]
        return response


def transport(
    responder: Callable[[urllib.request.Request], Answer],
    **kwargs: object,
) -> tuple[UrllibTransport, FakeHTTPS]:
    handler = FakeHTTPS(responder)
    return UrllibTransport(https_handler=handler, **kwargs), handler  # type: ignore[arg-type]


def get(
    t: UrllibTransport, url: str, *, max_bytes: int = 1000, timeout_s: float = 7.0
) -> HttpResponse:
    return t.request(
        "GET",
        url,
        headers={"User-Agent": "ua"},
        body=None,
        timeout_s=timeout_s,
        max_bytes=max_bytes,
    )


URL = "https://api.opensubtitles.com/api/v1/subtitles"


def test_a_plain_response_is_returned_with_status_headers_and_body() -> None:
    t, handler = transport(
        lambda req: (200, {"Content-Type": "application/json", "X-A": "b"}, b'{"a":1}')
    )
    response = get(t, URL)
    assert response.status == 200
    assert response.body == b'{"a":1}'
    assert response.headers["content-type"] == "application/json"
    assert response.headers["x-a"] == "b"
    request = handler.seen[0]
    assert request.get_method() == "GET"
    assert request.get_header("User-agent") == "ua"
    assert request.timeout == 7.0


def test_a_post_sends_its_body() -> None:
    t, handler = transport(lambda req: (200, {}, b"{}"))
    t.request("POST", URL, headers={"A": "b"}, body=b'{"x":1}', timeout_s=3, max_bytes=100)
    assert handler.seen[0].get_method() == "POST"
    assert handler.seen[0].data == b'{"x":1}'


@pytest.mark.parametrize("status", [400, 401, 404, 406, 429, 500, 503])
def test_an_error_status_is_a_response_not_an_exception(status: int) -> None:
    t, _ = transport(lambda req: (status, {"Retry-After": "9"}, b'{"message":"x"}'))
    response = get(t, URL)
    assert response.status == status
    assert response.headers["retry-after"] == "9"
    assert response.body == b'{"message":"x"}'


@pytest.mark.parametrize(
    "url",
    [
        "http://api.opensubtitles.com/api/v1/subtitles",
        "https://evil.example/x",
        "https://evilopensubtitles.com/x",
        "https://opensubtitles.com.evil.net/x",
        "https://api.opensubtitles.com@evil.example/x",
        "file:///etc/passwd",
        "ftp://api.opensubtitles.com/x",
        "api.opensubtitles.com/x",
        "",
    ],
)
def test_it_refuses_any_url_that_is_not_https_on_opensubtitles(url: str) -> None:
    t, handler = transport(lambda req: (200, {}, b"x"))
    with pytest.raises(TransportError):
        get(t, url)
    assert handler.seen == []


def test_a_body_over_the_cap_is_refused_while_reading_not_after() -> None:
    reads: list[int] = []

    class Endless(io.RawIOBase):
        def readable(self) -> bool:
            return True

        def readinto(self, buffer: bytearray | memoryview) -> int:  # type: ignore[override]
            reads.append(len(buffer))
            buffer[:] = b"x" * len(buffer)
            return len(buffer)

    t, _ = transport(lambda req: (200, {}, io.BufferedReader(Endless())))
    with pytest.raises(ResponseTooLarge):
        get(t, URL, max_bytes=100)
    # Stopped by the cap: nothing like an unbounded read happened.
    assert sum(reads) < 1_000_000


def test_a_declared_content_length_over_the_cap_is_refused_without_reading() -> None:
    class NoRead(io.BytesIO):
        def read(self, size: int | None = -1) -> bytes:  # pragma: no cover - must not be called
            raise AssertionError("the body was read")

    t, _ = transport(lambda req: (200, {"Content-Length": "5000"}, NoRead(b"x")))
    with pytest.raises(ResponseTooLarge):
        get(t, URL, max_bytes=100)


def test_a_lying_content_length_does_not_get_past_the_cap() -> None:
    t, _ = transport(lambda req: (200, {"Content-Length": "5"}, b"x" * 500))
    with pytest.raises(ResponseTooLarge):
        get(t, URL, max_bytes=100)


def test_a_body_of_exactly_the_cap_is_accepted() -> None:
    t, _ = transport(lambda req: (200, {}, b"x" * 100))
    assert len(get(t, URL, max_bytes=100).body) == 100


def test_an_error_body_is_capped_too() -> None:
    t, _ = transport(lambda req: (500, {}, b"x" * 500))
    with pytest.raises(ResponseTooLarge):
        get(t, URL, max_bytes=100)


def test_a_redirect_to_another_host_is_refused() -> None:
    def responder(req: urllib.request.Request) -> Answer:
        return 302, {"Location": "https://evil.example/steal"}, b""

    t, handler = transport(responder)
    with pytest.raises(TransportError, match="redirect"):
        get(t, URL)
    assert len(handler.seen) == 1  # the other host was never contacted


@pytest.mark.parametrize(
    "location",
    [
        "http://api.opensubtitles.com/x",
        "https://evilopensubtitles.com/x",
        "https://opensubtitles.com.evil.net/x",
        "https://api.opensubtitles.com@evil.example/x",
        "file:///etc/passwd",
        "ftp://dl.opensubtitles.com/x",
    ],
)
def test_a_redirect_to_a_plain_http_or_look_alike_target_is_refused(location: str) -> None:
    t, handler = transport(lambda req: (301, {"Location": location}, b""))
    with pytest.raises(TransportError, match="redirect"):
        get(t, URL)
    assert len(handler.seen) == 1


def test_a_redirect_within_opensubtitles_is_followed() -> None:
    def responder(req: urllib.request.Request) -> Answer:
        if req.full_url.startswith("https://api."):
            return 302, {"Location": "https://dl.opensubtitles.com/final"}, b""
        return 200, {}, b"final body"

    t, handler = transport(responder)
    response = get(t, URL)
    assert response.body == b"final body"
    assert [r.full_url for r in handler.seen] == [URL, "https://dl.opensubtitles.com/final"]


def test_a_relative_redirect_stays_on_the_same_host() -> None:
    def responder(req: urllib.request.Request) -> Answer:
        if req.full_url == URL:
            return 302, {"Location": "/elsewhere"}, b""
        return 200, {}, b"ok"

    t, handler = transport(responder)
    assert get(t, URL).body == b"ok"
    assert handler.seen[1].full_url == "https://api.opensubtitles.com/elsewhere"


def test_a_redirect_loop_ends_in_an_error() -> None:
    t, _ = transport(lambda req: (302, {"Location": URL}, b""))
    with pytest.raises(TransportError):
        get(t, URL)


def test_a_socket_timeout_is_a_transport_timeout() -> None:
    def responder(req: urllib.request.Request) -> Answer:
        raise TimeoutError("timed out")

    t, _ = transport(responder)
    with pytest.raises(TransportTimeout):
        get(t, URL)


def test_a_url_error_wrapping_a_timeout_is_a_transport_timeout() -> None:
    def responder(req: urllib.request.Request) -> Answer:
        raise urllib.error.URLError(TimeoutError("timed out"))

    t, _ = transport(responder)
    with pytest.raises(TransportTimeout):
        get(t, URL)


@pytest.mark.parametrize(
    "error",
    [
        urllib.error.URLError("connection refused"),
        ConnectionResetError("reset"),
        OSError("network is down"),
        http.client.IncompleteRead(b"abc"),
        http.client.BadStatusLine("garbage"),
        ValueError("Invalid header value"),
    ],
)
def test_other_network_failures_are_transport_errors_without_the_url_in_the_message(
    error: Exception,
) -> None:
    def responder(req: urllib.request.Request) -> Answer:
        raise error

    t, _ = transport(responder)
    with pytest.raises(TransportError) as excinfo:
        get(t, URL)
    assert not isinstance(excinfo.value, TransportTimeout)
    assert "opensubtitles" not in str(excinfo.value)
    assert excinfo.value.__cause__ is None  # nothing that could carry a request detail


def test_a_slow_drip_that_never_ends_hits_the_total_time_limit() -> None:
    clock = {"now": 0.0}

    class Drip(io.RawIOBase):
        def readable(self) -> bool:
            return True

        def readinto(self, buffer: bytearray | memoryview) -> int:  # type: ignore[override]
            clock["now"] += 30.0  # each read takes 30 s, always inside the per-read timeout
            buffer[:1] = b"x"
            return 1

    t, _ = transport(
        lambda req: (200, {}, io.BufferedReader(Drip())),
        max_duration_s=100.0,
        monotonic=lambda: clock["now"],
    )
    with pytest.raises(TransportTimeout):
        get(t, URL, max_bytes=10**9)
    assert clock["now"] < 1000.0
