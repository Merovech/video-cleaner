from __future__ import annotations

import logging
import re
from collections.abc import Iterator
from pathlib import Path

import pytest

from video_cleaner.consolefmt import ConsoleLogHandler
from video_cleaner.logging_setup import (
    LOGGER_NAME,
    RedactingFilter,
    redact,
    register_secrets,
    setup_logging,
)


@pytest.fixture(autouse=True)
def _reset_logging() -> Iterator[None]:
    yield
    pkg = logging.getLogger(LOGGER_NAME)
    for handler in list(pkg.handlers):
        if type(handler) in (logging.FileHandler, logging.StreamHandler):
            pkg.removeHandler(handler)
            handler.close()


def test_redact_masks_explicit_secret_and_api_key_header() -> None:
    text = "login with hunter2-secret; headers: Api-Key: abc123XYZ; done"
    out = redact(text, secrets=["hunter2-secret"])
    assert "hunter2-secret" not in out
    assert "abc123XYZ" not in out
    assert "***" in out
    assert out.startswith("login with ")
    assert out.endswith("; done")


@pytest.mark.parametrize(
    ("text", "leaked"),
    [
        ("curl -H 'api-key: s3cr3t'", "s3cr3t"),
        ("API-KEY: s3cr3t", "s3cr3t"),
        ("url?password=p4ss&x=1", "p4ss"),
        ("PASSWORD=p4ss", "p4ss"),
        ("token=tok_123 next", "tok_123"),
        ("Authorization: Bearer eyJhbGciOi.payload.sig", "eyJhbGciOi"),
        ("authorization: bearer lowercase-token", "lowercase-token"),
    ],
)
def test_redact_generic_patterns_case_insensitive(text: str, leaked: str) -> None:
    out = redact(text)
    assert leaked not in out
    assert "***" in out


def test_redact_leaves_ordinary_text_alone() -> None:
    text = "ffmpeg -i in.mkv -c:v libx265 out.mkv (token budget unaffected)"
    assert redact(text) == text


def test_redact_ignores_empty_secrets_and_masks_longest_first() -> None:
    out = redact("a=abc abcdef", secrets=["", "abc", "abcdef"])
    assert "abc" not in out
    assert out == "a=*** ***"


def test_log_file_is_debug_even_when_quiet(tmp_path: Path) -> None:
    path = setup_logging(tmp_path / "logs", verbosity=-1)
    log = logging.getLogger(f"{LOGGER_NAME}.test")
    log.debug("debug-line-marker")
    log.info("info-line-marker")
    for handler in logging.getLogger(LOGGER_NAME).handlers:
        handler.flush()
    content = path.read_text(encoding="utf-8")
    assert "debug-line-marker" in content
    assert "info-line-marker" in content
    assert path.parent == tmp_path / "logs"
    assert re.fullmatch(r"run-\d{8}-\d{6}\.log", path.name)


@pytest.mark.parametrize(
    ("verbosity", "debug_shown", "info_shown", "warning_shown"),
    [(-1, False, False, True), (0, False, True, True), (1, True, True, True)],
)
def test_console_level_follows_verbosity(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    verbosity: int,
    debug_shown: bool,
    info_shown: bool,
    warning_shown: bool,
) -> None:
    setup_logging(tmp_path, verbosity=verbosity)
    log = logging.getLogger(f"{LOGGER_NAME}.console")
    log.debug("dbg-marker")
    log.info("inf-marker")
    log.warning("wrn-marker")
    err = capsys.readouterr().err
    assert ("dbg-marker" in err) is debug_shown
    assert ("inf-marker" in err) is info_shown
    assert ("wrn-marker" in err) is warning_shown


def test_secrets_never_reach_console_or_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = setup_logging(tmp_path, verbosity=1, secrets=["topsecretvalue"])
    log = logging.getLogger(f"{LOGGER_NAME}.sec")
    log.info("using %s and Api-Key: header-secret", "topsecretvalue")
    try:
        raise RuntimeError("failed with token=traceback-secret")
    except RuntimeError:
        log.exception("boom topsecretvalue")
    err = capsys.readouterr().err
    content = path.read_text(encoding="utf-8")
    for blob in (err, content):
        assert "topsecretvalue" not in blob
        assert "header-secret" not in blob
        assert "traceback-secret" not in blob
        assert "RuntimeError" in blob  # traceback still present, just masked


def test_setup_logging_twice_does_not_duplicate_handlers(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    setup_logging(tmp_path, verbosity=0)
    path = setup_logging(tmp_path, verbosity=0)
    ours = [
        h
        for h in logging.getLogger(LOGGER_NAME).handlers
        if type(h) in (logging.FileHandler, ConsoleLogHandler)
    ]
    assert len(ours) == 2  # one file + one console, not four
    logging.getLogger(f"{LOGGER_NAME}.twice").info("only-once-marker")
    assert capsys.readouterr().err.count("only-once-marker") == 1
    assert path.read_text(encoding="utf-8").count("only-once-marker") == 1


def test_setup_logging_creates_missing_log_dir(tmp_path: Path) -> None:
    target = tmp_path / "a" / "b"
    path = setup_logging(target, verbosity=0)
    assert path.exists()


@pytest.mark.parametrize(
    ("text", "leaked"),
    [
        ("{'Api-Key': 'abc123'}", "abc123"),
        ('{"password": "hunter2"}', "hunter2"),
        ('{"token":"tok999","x":1}', "tok999"),
        ("access_token=abc123", "abc123"),
        ("x_api_key=abc123", "abc123"),
        ("X_API_KEY: abc123", "abc123"),
        ('password="my pass phrase"', "pass phrase"),
        ("password='my pass phrase' next", "pass phrase"),
        ('{"password": "my pass phrase"}', "pass phrase"),
        ('api_key = "two words"', "two words"),
    ],
)
def test_redact_dict_underscore_and_quoted_forms(text: str, leaked: str) -> None:
    out = redact(text)
    assert leaked not in out
    assert "***" in out


def test_redact_quoted_value_keeps_surrounding_text() -> None:
    out = redact('before password="my pass phrase" after')
    assert out.startswith("before ")
    assert out.endswith(" after")


def test_bad_log_call_never_raises_and_still_redacts(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = setup_logging(tmp_path, verbosity=1, secrets=["topsecretvalue"])
    log = logging.getLogger(f"{LOGGER_NAME}.bad")
    logging.raiseExceptions = False  # keep the stdlib "Logging error" noise out of the test
    try:
        log.info("a %s %s", 1)  # too few args
        log.info("no placeholders", "topsecretvalue")  # too many args
        log.info("%d items", "topsecretvalue")  # wrong type
    finally:
        logging.raiseExceptions = True
    log.info("still-alive-marker")
    err = capsys.readouterr().err
    content = path.read_text(encoding="utf-8")
    assert "still-alive-marker" in content
    assert "topsecretvalue" not in err
    assert "topsecretvalue" not in content


def test_log_file_option_and_file_level(tmp_path: Path) -> None:
    target = tmp_path / "chosen" / "my.log"
    path = setup_logging(tmp_path / "logs", verbosity=0, log_file=target, file_level=logging.INFO)
    assert path == target
    log = logging.getLogger(f"{LOGGER_NAME}.test")
    log.debug("debug-line-marker")
    log.info("info-line-marker")
    for handler in logging.getLogger(LOGGER_NAME).handlers:
        handler.flush()
    content = target.read_text(encoding="utf-8")
    assert "info-line-marker" in content
    assert "debug-line-marker" not in content
    assert not (tmp_path / "logs").exists()


def test_console_lines_carry_a_level_tag_the_file_lines_do_not(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = setup_logging(tmp_path, verbosity=1)
    log = logging.getLogger(f"{LOGGER_NAME}.tags")
    log.debug("d-marker")
    log.info("i-marker")
    log.warning("w-marker")
    log.error("e-marker")
    err = capsys.readouterr().err.splitlines()
    assert err == ["[VERB] d-marker", "[INFO] i-marker", "[WARN] w-marker", "[FAIL] e-marker"]
    assert "[INFO]" not in path.read_text(encoding="utf-8")  # the file keeps its own format


def test_redact_masks_a_bearer_value_that_follows_a_masked_label() -> None:
    out = redact("token: Bearer abc123")
    assert "abc123" not in out
    assert "Bearer abc123" not in out


@pytest.mark.parametrize(
    ("text", "leaked"),
    [
        ("Authorization: Basic dXNlcjpwdw==", "dXNlcjpwdw"),
        ("authorization: basic dXNlcjpwdw==", "dXNlcjpwdw"),
        ('Authorization: Digest username="bob", realm="x", response="deadbeef"', "deadbeef"),
        ("password: Basic dXNlcjpwdw==", "dXNlcjpwdw"),
    ],
)
def test_redact_masks_basic_and_digest_credentials(text: str, leaked: str) -> None:
    out = redact(text)
    assert leaked not in out
    assert "***" in out


def test_redact_leaves_the_word_basic_in_prose_alone() -> None:
    text = "basic configuration loaded; digest of the file computed"
    assert redact(text) == text


@pytest.mark.parametrize(
    ("text", "leaked"),
    [
        ('{\\"password\\": \\"s3cr3t\\"}', "s3cr3t"),
        ('{\\"password\\": \\"two words\\", \\"x\\": 1}', "two words"),
        ('{"password": "pa\\"ss word"}', "ss word"),
    ],
)
def test_redact_handles_escaped_quotes_inside_json_strings(text: str, leaked: str) -> None:
    out = redact(text)
    assert leaked not in out
    assert "***" in out


def test_redact_escaped_json_keeps_the_text_after_the_value() -> None:
    out = redact('{\\"password\\": \\"s3cr3t\\", \\"user\\": \\"bob\\"}')
    assert "bob" in out


@pytest.fixture
def clean_registry() -> Iterator[None]:
    from video_cleaner import logging_setup

    logging_setup._reset_registered_secrets()
    yield
    logging_setup._reset_registered_secrets()


@pytest.mark.usefixtures("clean_registry")
def test_registered_secret_is_masked_by_redact() -> None:
    register_secrets(["hunter22"])
    out = redact("connect with hunter22 now")
    assert "hunter22" not in out
    assert out == "connect with *** now"


@pytest.mark.usefixtures("clean_registry")
def test_registered_secret_is_masked_by_the_filter() -> None:
    register_secrets(["hunter22"])
    record = logging.LogRecord(
        "video_cleaner.x", logging.INFO, __file__, 1, "got %s", ("hunter22",), None
    )
    assert RedactingFilter().filter(record)
    assert "hunter22" not in record.getMessage()


@pytest.mark.usefixtures("clean_registry")
def test_registered_secret_is_masked_in_the_log_file(tmp_path: Path) -> None:
    register_secrets(["hunter22"])
    path = setup_logging(tmp_path, verbosity=0)
    logging.getLogger(f"{LOGGER_NAME}.reg").info("value hunter22 here")
    for handler in logging.getLogger(LOGGER_NAME).handlers:
        handler.flush()
    assert "hunter22" not in path.read_text(encoding="utf-8")


@pytest.mark.usefixtures("clean_registry")
def test_registering_accumulates_and_ignores_values_under_four_characters() -> None:
    register_secrets(["a", "abc", ""])
    assert redact("a abc") == "a abc"
    register_secrets(["abcd"])
    register_secrets(["wxyz"])
    assert redact("abcd wxyz") == "*** ***"
