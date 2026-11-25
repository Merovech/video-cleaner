from __future__ import annotations

import os
import re
import stat
import subprocess
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import keyring
import keyring.backends.fail
import keyring.backends.null
import keyring.errors
import pytest
from typer.testing import CliRunner

from video_cleaner import credentials
from video_cleaner.cli import app
from video_cleaner.credentials import (
    ALL_NAMES,
    API_KEY,
    PASSWORD,
    SERVICE,
    TOKEN,
    USERNAME,
    CredentialError,
    FileStore,
    KeyringStore,
    OpenSubtitlesCredentials,
    credentials_file_path,
    current_identity,
    keyring_status,
    load_credentials,
    open_store,
    restrict_to_owner,
)
from video_cleaner.logging_setup import redact

API = "k3y-abcdef-123456"
USER = "someuser"
PHRASE = "hunter2-hunter2"
SID = "*S-1-5-21-111-222-333-1001"


class MemoryStore:
    """In-memory `CredentialStore`."""

    def __init__(self, values: Mapping[str, str] | None = None) -> None:
        self.values = dict(values or {})

    def get(self, name: str) -> str | None:
        return self.values.get(name)

    def set(self, name: str, value: str) -> None:
        self.values[name] = value

    def delete(self, name: str) -> None:
        self.values.pop(name, None)


class FakeBackend:
    """What `KeyringStore` needs from a keyring backend, with the real error types."""

    def __init__(self) -> None:
        self.passwords: dict[tuple[str, str], str] = {}

    def get_password(self, service: str, username: str) -> str | None:
        return self.passwords.get((service, username))

    def set_password(self, service: str, username: str, password: str) -> None:
        self.passwords[(service, username)] = password

    def delete_password(self, service: str, username: str) -> None:
        if (service, username) not in self.passwords:
            raise keyring.errors.PasswordDeleteError("not found")
        del self.passwords[(service, username)]


class BrokenBackend(FakeBackend):
    def get_password(self, service: str, username: str) -> str | None:
        raise keyring.errors.KeyringError(f"locked, was reading {PHRASE}")

    def set_password(self, service: str, username: str, password: str) -> None:
        raise keyring.errors.PasswordSetError("denied")

    def delete_password(self, service: str, username: str) -> None:
        raise keyring.errors.KeyringError("locked")


FULL = {API_KEY: API, USERNAME: USER, PASSWORD: PHRASE}


# --- FileStore ------------------------------------------------------------------------


def test_file_store_round_trip(tmp_path: Path) -> None:
    store = FileStore(tmp_path / "sub" / "credentials.json")
    assert store.get(API_KEY) is None  # no file yet
    store.set(API_KEY, API)
    store.set(USERNAME, USER)
    assert store.get(API_KEY) == API
    assert store.get(USERNAME) == USER
    assert FileStore(tmp_path / "sub" / "credentials.json").get(API_KEY) == API  # persisted
    store.set(API_KEY, "other-value")
    assert store.get(API_KEY) == "other-value"
    store.delete(API_KEY)
    assert store.get(API_KEY) is None
    assert store.get(USERNAME) == USER
    store.delete(API_KEY)  # deleting what is not there is fine


def test_file_store_delete_without_a_file_creates_nothing(tmp_path: Path) -> None:
    path = tmp_path / "credentials.json"
    FileStore(path).delete(API_KEY)
    assert not path.exists()


def test_file_store_leaves_no_temp_files(tmp_path: Path) -> None:
    store = FileStore(tmp_path / "credentials.json")
    store.set(API_KEY, API)
    store.set(API_KEY, "again")
    assert [p.name for p in tmp_path.iterdir()] == ["credentials.json"]


@pytest.mark.parametrize("content", ["not json", "[1, 2]", '{"a": 1}', ""])
def test_file_store_rejects_a_damaged_file_without_echoing_it(tmp_path: Path, content: str) -> None:
    path = tmp_path / "credentials.json"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(CredentialError, match=r"credentials.json"):
        FileStore(path).get(API_KEY)


def test_file_store_reports_an_unreadable_file(tmp_path: Path) -> None:
    path = tmp_path / "credentials.json"
    path.mkdir()  # a folder where the file should be
    with pytest.raises(CredentialError, match=r"cannot read"):
        FileStore(path).get(API_KEY)


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits")
def test_file_store_file_is_owner_only_on_posix(tmp_path: Path) -> None:
    path = tmp_path / "credentials.json"
    store = FileStore(path)
    store.set(API_KEY, API)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    store.set(API_KEY, "again")
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_file_store_restricts_before_it_writes_the_secret(tmp_path: Path) -> None:
    path = tmp_path / "credentials.json"
    seen: list[str] = []

    def restrict(target: Path) -> None:
        seen.append(target.read_text(encoding="utf-8"))  # what is in the file when it is locked

    FileStore(path, restrict=restrict).set(API_KEY, API)
    assert len(seen) == 1
    assert API not in seen[0]
    assert path.read_text(encoding="utf-8") != seen[0]


def test_file_store_failed_restriction_leaves_no_secret_behind(tmp_path: Path) -> None:
    def restrict(target: Path) -> None:
        raise CredentialError("icacls failed")

    path = tmp_path / "credentials.json"
    with pytest.raises(CredentialError, match="icacls"):
        FileStore(path, restrict=restrict).set(API_KEY, API)
    assert list(tmp_path.iterdir()) == []


def test_restrict_to_owner_runs_icacls_without_a_shell_on_windows(tmp_path: Path) -> None:
    calls: list[list[str]] = []

    def run(cmd: Sequence[str]) -> tuple[int, str]:
        calls.append(list(cmd))
        return 0, ""

    target = tmp_path / "credentials.json"
    restrict_to_owner(target, windows=True, identity=SID, system_root="C:/Win", run=run)
    assert calls == [
        [
            str(Path("C:/Win") / "System32" / "icacls.exe"),
            str(target),
            "/inheritance:r",
            "/grant:r",
            f"{SID}:F",
        ]
    ]


def test_restrict_to_owner_reports_an_icacls_failure(tmp_path: Path) -> None:
    def run(cmd: Sequence[str]) -> tuple[int, str]:
        return 5, "Access is denied.\n"

    with pytest.raises(CredentialError, match="Access is denied"):
        restrict_to_owner(tmp_path / "x", windows=True, identity=SID, system_root="C:/Win", run=run)


def test_restrict_to_owner_reports_a_missing_icacls(tmp_path: Path) -> None:
    def run(cmd: Sequence[str]) -> tuple[int, str]:
        raise FileNotFoundError("icacls")

    with pytest.raises(CredentialError, match="icacls"):
        restrict_to_owner(tmp_path / "x", windows=True, identity=SID, system_root="C:/Win", run=run)


WHOAMI_CSV = '"shiva\\ariro","S-1-5-21-111-222-333-1001"\r\n'


def test_current_identity_is_the_sid_of_the_process_token() -> None:
    calls: list[list[str]] = []

    def run(cmd: Sequence[str]) -> tuple[int, str]:
        calls.append(list(cmd))
        return 0, WHOAMI_CSV

    identity = current_identity(system_root="C:/Win", run=run, env={})
    assert identity == SID
    assert calls == [
        [str(Path("C:/Win") / "System32" / "whoami.exe"), "/user", "/fo", "csv", "/nh"]
    ]


@pytest.mark.parametrize(
    "result",
    [(1, ""), (0, ""), (0, "garbage"), (0, '"a\\b","not-a-sid"\r\n'), (0, '"only-one-field"\r\n')],
)
def test_current_identity_falls_back_to_the_domain_qualified_name(result: tuple[int, str]) -> None:
    env = {"USERDOMAIN": "SHIVA", "USERNAME": "ariro"}
    identity = current_identity(system_root="C:/Win", run=lambda cmd: result, env=env)
    assert identity == "SHIVA\\ariro"


def test_current_identity_falls_back_when_whoami_cannot_run() -> None:
    def run(cmd: Sequence[str]) -> tuple[int, str]:
        raise FileNotFoundError("whoami")

    env = {"USERDOMAIN": "SHIVA", "USERNAME": "ariro"}
    assert current_identity(system_root="C:/Win", run=run, env=env) == "SHIVA\\ariro"


@pytest.mark.parametrize("env", [{}, {"USERNAME": "ariro"}, {"USERDOMAIN": "SHIVA"}])
def test_current_identity_without_any_identity_is_an_error(env: dict[str, str]) -> None:
    with pytest.raises(CredentialError, match="identity"):
        current_identity(system_root="C:/Win", run=lambda cmd: (1, ""), env=env)


def test_restrict_to_owner_grants_the_process_identity_not_the_login_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("USERNAME", "someone-else")
    monkeypatch.setenv("USERDOMAIN", "ELSEWHERE")
    calls: list[list[str]] = []

    def run(cmd: Sequence[str]) -> tuple[int, str]:
        calls.append(list(cmd))
        return (0, WHOAMI_CSV) if Path(cmd[0]).stem == "whoami" else (0, "")

    restrict_to_owner(tmp_path / "c.json", windows=True, system_root="C:/Win", run=run)
    icacls = next(c for c in calls if Path(c[0]).stem == "icacls")
    assert icacls[-1] == SID + ":F"
    assert "someone-else" not in " ".join(icacls)


def test_restrict_to_owner_stops_when_no_identity_can_be_found(tmp_path: Path) -> None:
    calls: list[list[str]] = []

    def run(cmd: Sequence[str]) -> tuple[int, str]:
        calls.append(list(cmd))
        return 1, ""

    with pytest.raises(CredentialError, match="identity"):
        restrict_to_owner(tmp_path / "c.json", windows=True, system_root="C:/Win", run=run, env={})
    assert all(Path(c[0]).stem != "icacls" for c in calls)  # nothing was granted


@pytest.mark.skipif(os.name != "nt", reason="whoami and the token are Windows only")
def test_the_real_identity_is_a_sid() -> None:
    assert re.fullmatch(r"\*S-1-\d+(-\d+)+", current_identity())


def test_restrict_to_owner_uses_chmod_on_posix(tmp_path: Path) -> None:
    target = tmp_path / "credentials.json"
    target.write_text("{}", encoding="utf-8")
    calls: list[tuple[Path, int]] = []
    restrict_to_owner(target, windows=False, chmod=lambda p, m: calls.append((p, m)))
    assert calls == [(target, 0o600)]


@pytest.mark.skipif(os.name != "nt", reason="icacls is Windows only")
def test_file_store_is_owner_only_on_windows_with_the_real_icacls(tmp_path: Path) -> None:
    from video_cleaner.doctor import loose_acl_entries

    path = tmp_path / "credentials.json"
    store = FileStore(path)
    store.set(API_KEY, API)
    store.set(API_KEY, "again")  # the replace keeps the restriction
    root = os.environ.get("SYSTEMROOT", "C:/Windows")
    out = subprocess.run(  # noqa: S603  # nosec B603
        [str(Path(root) / "System32" / "icacls.exe"), str(path)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert loose_acl_entries(out) == []
    assert "(I)" not in out  # nothing inherited from the folder
    assert store.get(API_KEY) == "again"


# --- KeyringStore ---------------------------------------------------------------------


def test_keyring_store_round_trip_uses_the_service_name() -> None:
    backend = FakeBackend()
    store = KeyringStore(backend)
    assert store.get(API_KEY) is None
    store.set(API_KEY, API)
    assert backend.passwords == {(SERVICE, API_KEY): API}
    assert store.get(API_KEY) == API
    store.delete(API_KEY)
    assert store.get(API_KEY) is None
    store.delete(API_KEY)  # already gone: not an error


class PyWinError(Exception):
    """Stands in for `pywintypes.error`, which the Windows backend lets through unwrapped."""


class WinBackend(FakeBackend):
    def get_password(self, service: str, username: str) -> str | None:
        raise PyWinError(f"(1312, 'CredRead', {PHRASE})")

    def set_password(self, service: str, username: str, password: str) -> None:
        raise PyWinError(f"(1312, 'CredWrite', {password})")

    def delete_password(self, service: str, username: str) -> None:
        raise PyWinError(f"(1312, 'CredDelete', {PHRASE})")


def test_keyring_store_wraps_any_backend_error_with_the_type_name_only() -> None:
    store = KeyringStore(WinBackend())
    calls: list[Callable[[], object]] = [
        lambda: store.get(API_KEY),
        lambda: store.set(API_KEY, PHRASE),
        lambda: store.delete(API_KEY),
    ]
    for call in calls:
        with pytest.raises(CredentialError) as info:
            call()
        assert "PyWinError" in str(info.value)
        assert PHRASE not in str(info.value)
        assert info.value.__cause__ is None
        assert info.value.__suppress_context__


def test_file_store_wraps_a_failing_folder_creation(tmp_path: Path) -> None:
    blocker = tmp_path / "blocker"
    blocker.write_text("a file where a folder is needed", encoding="utf-8")
    with pytest.raises(CredentialError) as info:
        FileStore(blocker / "sub" / "credentials.json").set(API_KEY, PHRASE)
    assert PHRASE not in str(info.value)
    assert info.value.__cause__ is None


def test_file_store_wraps_a_failing_replace_and_cleans_up(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken_replace(src: object, dst: object) -> None:
        raise PermissionError(f"locked while writing {PHRASE}")

    monkeypatch.setattr(os, "replace", broken_replace)
    with pytest.raises(CredentialError) as info:
        FileStore(tmp_path / "credentials.json").set(API_KEY, PHRASE)
    assert "PermissionError" in str(info.value)
    assert PHRASE not in str(info.value)
    assert list(tmp_path.iterdir()) == []


def test_file_store_wraps_a_failing_temp_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import tempfile

    def broken_mkstemp(*args: object, **kwargs: object) -> tuple[int, str]:
        raise OSError(f"disk full {PHRASE}")

    monkeypatch.setattr(tempfile, "mkstemp", broken_mkstemp)
    with pytest.raises(CredentialError) as info:
        FileStore(tmp_path / "credentials.json").set(API_KEY, PHRASE)
    assert "OSError" in str(info.value)
    assert PHRASE not in str(info.value)


def test_keyring_store_wraps_backend_errors_without_the_secret() -> None:
    store = KeyringStore(BrokenBackend())
    with pytest.raises(CredentialError) as info:
        store.get(API_KEY)
    assert PHRASE not in str(info.value)
    with pytest.raises(CredentialError):
        store.set(API_KEY, API)
    with pytest.raises(CredentialError):
        store.delete(API_KEY)


def test_service_name() -> None:
    assert SERVICE == "video-cleaner"
    assert set(ALL_NAMES) == {
        "opensubtitles_api_key",
        "opensubtitles_username",
        "opensubtitles_password",
        "opensubtitles_token",
    }


# --- open_store -----------------------------------------------------------------------


def _env(tmp_path: Path) -> dict[str, str]:
    return {"LOCALAPPDATA": str(tmp_path / "local")}


def test_credentials_file_path_is_under_localappdata(tmp_path: Path) -> None:
    expected = tmp_path / "local" / "video-cleaner" / "credentials.json"
    assert credentials_file_path(_env(tmp_path)) == expected
    assert credentials_file_path({}) is None


def test_open_store_uses_a_usable_keyring(tmp_path: Path) -> None:
    backend = FakeBackend()
    store = open_store(_env(tmp_path), backend_factory=lambda: backend)
    assert isinstance(store, KeyringStore)
    store.set(API_KEY, API)
    assert backend.passwords[(SERVICE, API_KEY)] == API


@pytest.mark.parametrize(
    "backend",
    [
        None,
        keyring.backends.fail.Keyring(),  # type: ignore[no-untyped-call]
        keyring.backends.null.Keyring(),  # type: ignore[no-untyped-call]
    ],
    ids=["no keyring", "fail backend", "null backend"],
)
def test_open_store_falls_back_to_the_owner_only_file(tmp_path: Path, backend: object) -> None:
    store = open_store(_env(tmp_path), backend_factory=lambda: backend)
    assert isinstance(store, FileStore)
    assert store.path == tmp_path / "local" / "video-cleaner" / "credentials.json"


def test_open_store_falls_back_when_the_keyring_cannot_be_loaded(tmp_path: Path) -> None:
    def broken() -> object:
        raise RuntimeError("no dbus")

    assert isinstance(open_store(_env(tmp_path), backend_factory=broken), FileStore)


def test_keyring_status_names_a_usable_backend() -> None:
    backend = FakeBackend()
    name, usable = keyring_status(lambda: backend)
    assert name == f"{FakeBackend.__module__}.{FakeBackend.__qualname__}"
    assert usable is True


@pytest.mark.parametrize(
    ("backend", "name"),
    [
        (None, "none"),
        (keyring.backends.fail.Keyring(), "keyring.backends.fail.Keyring"),  # type: ignore[no-untyped-call]
        (keyring.backends.null.Keyring(), "keyring.backends.null.Keyring"),  # type: ignore[no-untyped-call]
    ],
)
def test_keyring_status_flags_the_backends_open_store_does_not_use(
    backend: object, name: str
) -> None:
    assert keyring_status(lambda: backend) == (name, False)


def test_keyring_status_never_repeats_the_text_of_a_loading_error() -> None:
    def broken() -> object:
        raise RuntimeError("secret-detail")

    assert keyring_status(broken) == ("unavailable: RuntimeError", False)


def test_open_store_without_keyring_or_localappdata_is_an_error() -> None:
    with pytest.raises(CredentialError, match="LOCALAPPDATA"):
        open_store({}, backend_factory=lambda: None)


def test_open_store_default_asks_the_keyring_library(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = FakeBackend()
    monkeypatch.setattr(keyring, "get_keyring", lambda: backend)
    assert isinstance(open_store(_env(tmp_path)), KeyringStore)
    monkeypatch.setattr(keyring, "get_keyring", keyring.backends.fail.Keyring)
    assert isinstance(open_store(_env(tmp_path)), FileStore)


def test_the_suite_never_reaches_the_real_keyring(tmp_path: Path) -> None:
    # conftest makes the keyring library report "no usable backend" for every test.
    assert isinstance(open_store(_env(tmp_path)), FileStore)


# --- load_credentials -----------------------------------------------------------------


def test_load_credentials_from_the_store() -> None:
    assert load_credentials(MemoryStore(FULL), {}) == OpenSubtitlesCredentials(API, USER, PHRASE)


@pytest.mark.parametrize("missing", [API_KEY, USERNAME, PASSWORD])
def test_load_credentials_is_none_when_any_one_is_missing(missing: str) -> None:
    values = {k: v for k, v in FULL.items() if k != missing}
    assert load_credentials(MemoryStore(values), {}) is None


def test_load_credentials_is_none_for_empty_values() -> None:
    assert load_credentials(MemoryStore({**FULL, API_KEY: ""}), {}) is None


def test_environment_beats_the_store() -> None:
    env = {
        "VC_OPENSUBTITLES_API_KEY": "env-api-key",
        "VC_OPENSUBTITLES_USERNAME": "env-user",
        "VC_OPENSUBTITLES_PASSWORD": "env-password",
    }
    creds = load_credentials(MemoryStore(FULL), env)
    assert creds == OpenSubtitlesCredentials("env-api-key", "env-user", "env-password")


def test_environment_and_store_can_each_supply_part() -> None:
    store = MemoryStore({USERNAME: USER, PASSWORD: PHRASE})
    creds = load_credentials(store, {"VC_OPENSUBTITLES_API_KEY": API})
    assert creds == OpenSubtitlesCredentials(API, USER, PHRASE)


def test_an_empty_environment_variable_does_not_hide_the_store() -> None:
    creds = load_credentials(MemoryStore(FULL), {"VC_OPENSUBTITLES_API_KEY": ""})
    assert creds is not None
    assert creds.api_key == API


def test_load_credentials_reads_os_environ_when_no_env_is_given(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VC_OPENSUBTITLES_API_KEY", "from-os-environ")
    creds = load_credentials(MemoryStore({USERNAME: USER, PASSWORD: PHRASE}))
    assert creds is not None
    assert creds.api_key == "from-os-environ"


def test_loaded_credentials_are_registered_for_redaction() -> None:
    text = f"GET /login key={API} user={USER} pw={PHRASE}"
    assert API in redact(text)  # nothing registered yet
    assert load_credentials(MemoryStore(FULL), {}) is not None
    masked = redact(text)
    assert API not in masked
    assert USER not in masked
    assert PHRASE not in masked


def test_environment_credentials_are_registered_for_redaction() -> None:
    env = {
        "VC_OPENSUBTITLES_API_KEY": "env-api-key",
        "VC_OPENSUBTITLES_USERNAME": "env-user",
        "VC_OPENSUBTITLES_PASSWORD": "env-password",
    }
    assert load_credentials(MemoryStore(), env) is not None
    assert "env-password" not in redact("sent env-password")


def test_credentials_repr_does_not_show_the_secret_values() -> None:
    creds = OpenSubtitlesCredentials(API, USER, PHRASE)
    assert API not in repr(creds)
    assert PHRASE not in repr(creds)


def test_missing_names_lists_what_is_not_set() -> None:
    store = MemoryStore({USERNAME: USER})
    assert credentials.missing_names(store, {"VC_OPENSUBTITLES_PASSWORD": PHRASE}) == [API_KEY]
    assert credentials.missing_names(MemoryStore(FULL), {}) == []


# --- `auth` command -------------------------------------------------------------------


@pytest.fixture
def cli_store(monkeypatch: pytest.MonkeyPatch) -> MemoryStore:
    store = MemoryStore()
    monkeypatch.setattr(credentials, "open_store", lambda env=None: store)
    return store


def _config(tmp_path: Path) -> Path:
    config = tmp_path / "config.toml"
    config.write_text(
        f"library_root = '{tmp_path / 'library'}'\n"
        f"scratch_dir = '{tmp_path / 'scratch'}'\n"
        f"completed_dir = '{tmp_path / 'completed'}'\n"
        f"state_db = '{tmp_path / 'state.db'}'\n"
        f"log_dir = '{tmp_path / 'logs'}'\n"
        f"subtitle_cache_dir = '{tmp_path / 'subs'}'\n",
        encoding="utf-8",
    )
    return config


def _auth(
    tmp_path: Path, *args: str, input_text: str | None = None, env: dict[str, str] | None = None
) -> tuple[int, str]:
    result = CliRunner().invoke(
        app, ["auth", "--config", str(_config(tmp_path)), *args], input=input_text, env=env
    )
    return result.exit_code, result.output


def test_auth_prompts_for_the_three_values_and_stores_them(
    tmp_path: Path, cli_store: MemoryStore
) -> None:
    code, output = _auth(tmp_path, input_text=f"{API}\n{USER}\n{PHRASE}\n")
    assert code == 0, output
    assert cli_store.values == FULL
    for typed in (API, USER, PHRASE):
        assert typed not in output  # a hidden prompt never echoes
    assert "[ OK ] stored opensubtitles_password" in output


def test_auth_never_echoes_the_typed_password(tmp_path: Path, cli_store: MemoryStore) -> None:
    _, output = _auth(tmp_path, input_text=f"{API}\n{USER}\n{PHRASE}\n")
    assert PHRASE not in output


def test_auth_prompts_through_typer_with_hidden_input(
    tmp_path: Path, cli_store: MemoryStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    import typer

    calls: list[bool] = []
    answers = iter([API, USER, PHRASE])

    def fake_prompt(text: str, **kwargs: object) -> str:
        calls.append(bool(kwargs.get("hide_input")))
        return next(answers)

    monkeypatch.setattr(typer, "prompt", fake_prompt)
    code, _ = _auth(tmp_path)
    assert code == 0
    assert calls == [True, True, True]


def test_auth_forgets_a_cached_login_token(tmp_path: Path, cli_store: MemoryStore) -> None:
    cli_store.values[TOKEN] = "stale-token-value"
    code, _ = _auth(tmp_path, input_text=f"{API}\n{USER}\n{PHRASE}\n")
    assert code == 0
    assert TOKEN not in cli_store.values


def test_auth_registers_what_it_was_given_for_redaction(
    tmp_path: Path, cli_store: MemoryStore
) -> None:
    _auth(tmp_path, input_text=f"{API}\n{USER}\n{PHRASE}\n")
    assert PHRASE not in redact(f"password was {PHRASE}")


def test_auth_strips_the_api_key_and_username_but_keeps_the_password_as_typed(
    tmp_path: Path, cli_store: MemoryStore
) -> None:
    code, _ = _auth(tmp_path, input_text=f"  {API} \n {USER}\t\n  {PHRASE} \n")
    assert code == 0
    assert cli_store.values == {API_KEY: API, USERNAME: USER, PASSWORD: f"  {PHRASE} "}


def test_auth_refuses_a_blank_password_but_keeps_inner_spaces(
    tmp_path: Path, cli_store: MemoryStore
) -> None:
    code, _ = _auth(tmp_path, input_text=f"{API}\n{USER}\n \t \n")
    assert code == 2
    assert cli_store.values == {}
    code, _ = _auth(tmp_path, input_text=f"{API}\n{USER}\na b  c\n")
    assert code == 0
    assert cli_store.values[PASSWORD] == "a b  c"


def test_auth_deletes_the_stale_token_before_it_writes_new_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class SecondSetFails(MemoryStore):
        sets = 0

        def set(self, name: str, value: str) -> None:
            self.sets += 1
            if self.sets == 2:
                raise CredentialError("keyring locked")
            super().set(name, value)

    store = SecondSetFails({TOKEN: "stale-token-1234", API_KEY: "old-key-1234"})
    monkeypatch.setattr(credentials, "open_store", lambda env=None: store)
    code, _ = _auth(tmp_path, input_text=f"{API}\n{USER}\n{PHRASE}\n")
    assert code == 2
    assert store.values[API_KEY] == API  # the first write went through
    assert TOKEN not in store.values  # and no old token sits beside the new key


def test_auth_stores_nothing_when_input_ends_early(tmp_path: Path, cli_store: MemoryStore) -> None:
    code, _ = _auth(tmp_path, input_text=f"{API}\n")
    assert code != 0
    assert cli_store.values == {}


def test_auth_rejects_a_blank_value_and_stores_nothing(
    tmp_path: Path, cli_store: MemoryStore
) -> None:
    code, output = _auth(tmp_path, input_text=f"{API}\n{USER}\n   \n")
    assert code == 2
    assert "[FAIL]" in output
    assert cli_store.values == {}


def test_auth_status_lists_names_and_never_values(tmp_path: Path, cli_store: MemoryStore) -> None:
    cli_store.values.update({API_KEY: API, PASSWORD: PHRASE})
    code, output = _auth(tmp_path, "--status")
    assert code == 0, output
    assert "[ OK ] opensubtitles_api_key: set" in output
    assert "[WARN] opensubtitles_username: not set" in output
    assert "[ OK ] opensubtitles_password: set" in output
    assert "[WARN] opensubtitles_token: not set" in output
    assert API not in output
    assert PHRASE not in output


def test_auth_status_counts_environment_values_as_set(
    tmp_path: Path, cli_store: MemoryStore
) -> None:
    code, output = _auth(
        tmp_path, "--status", env={"VC_OPENSUBTITLES_API_KEY": "env-only-secret-value"}
    )
    assert code == 0
    assert "opensubtitles_api_key: set" in output
    assert "env-only-secret-value" not in output
    assert "opensubtitles_username: not set" in output


def test_auth_clear_deletes_all_four(tmp_path: Path, cli_store: MemoryStore) -> None:
    cli_store.values.update({**FULL, TOKEN: "tok-value-1234"})
    code, output = _auth(tmp_path, "--clear")
    assert code == 0, output
    assert cli_store.values == {}
    assert "tok-value-1234" not in output


def test_auth_status_and_clear_together_are_refused(tmp_path: Path, cli_store: MemoryStore) -> None:
    cli_store.values.update(FULL)
    code, output = _auth(tmp_path, "--status", "--clear")
    assert code == 2
    assert "[FAIL]" in output
    assert cli_store.values == FULL


@pytest.mark.parametrize(
    "args",
    [["--api-key", "x"], ["--password", "x"], ["--username", "x"], ["some-secret"]],
)
def test_auth_accepts_no_value_as_an_option_or_argument(
    tmp_path: Path, cli_store: MemoryStore, args: list[str]
) -> None:
    code, _ = _auth(tmp_path, *args)
    assert code == 2
    assert cli_store.values == {}


def test_auth_reports_a_store_error_without_a_traceback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(env: object = None) -> MemoryStore:
        raise CredentialError("no usable credential store")

    monkeypatch.setattr(credentials, "open_store", broken)
    code, output = _auth(tmp_path, "--status")
    assert code == 2
    assert "[FAIL]" in output
    assert "no usable credential store" in output


def test_auth_reports_a_failing_write_without_the_secret(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Failing(MemoryStore):
        def set(self, name: str, value: str) -> None:
            raise CredentialError("keyring locked")

    monkeypatch.setattr(credentials, "open_store", lambda env=None: Failing())
    code, output = _auth(tmp_path, input_text=f"{API}\n{USER}\n{PHRASE}\n")
    assert code == 2
    assert "keyring locked" in output
    assert PHRASE not in output


def test_auth_names_the_file_store_when_it_is_the_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "credentials.json"
    monkeypatch.setattr(credentials, "open_store", lambda env=None: FileStore(path))
    code, output = _auth(tmp_path, "--status")
    assert code == 0
    assert str(path) in output


def test_auth_with_a_missing_config_exits_2(tmp_path: Path, cli_store: MemoryStore) -> None:
    result = CliRunner().invoke(app, ["auth", "--config", str(tmp_path / "missing.toml")])
    assert result.exit_code == 2
    assert "config" in result.output


def _fail_with(error: Exception) -> type[MemoryStore]:
    class Failing(MemoryStore):
        def get(self, name: str) -> str | None:
            raise error

        def set(self, name: str, value: str) -> None:
            raise error

        def delete(self, name: str) -> None:
            raise error

    return Failing


@pytest.mark.parametrize(
    "error",
    [
        OSError(f"disk full while writing {PHRASE} {API}"),
        PyWinError(f"(1312, 'CredWrite', {PHRASE})"),
        RuntimeError(f"{USER} {PHRASE}"),
    ],
    ids=["OSError", "PyWinError", "RuntimeError"],
)
@pytest.mark.parametrize("args", [(), ("--status",), ("--clear",)])
def test_auth_turns_any_store_error_into_a_fail_line_without_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error: Exception, args: tuple[str, ...]
) -> None:
    store = _fail_with(error)()
    monkeypatch.setattr(credentials, "open_store", lambda env=None: store)
    typed = os.linesep.join([API, USER, PHRASE]) + os.linesep
    code, output = _auth(tmp_path, *args, input_text=typed)
    assert code == 2
    assert "[FAIL]" in output
    assert "Traceback" not in output
    assert type(error).__name__ in output
    for secret in (API, USER, PHRASE):
        assert secret not in output


def test_auth_with_a_real_failing_file_store_names_no_value(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    blocker = tmp_path / "blocker"
    blocker.write_text("x", encoding="utf-8")
    store = FileStore(blocker / "credentials.json")
    monkeypatch.setattr(credentials, "open_store", lambda env=None: store)
    typed = os.linesep.join([API, USER, PHRASE]) + os.linesep
    result = CliRunner().invoke(
        app, ["auth", "--config", str(_config(tmp_path))], input=typed, catch_exceptions=False
    )
    assert result.exit_code == 2
    assert "[FAIL]" in result.output
    assert PHRASE not in result.output


def test_the_app_never_shows_local_variables_in_a_traceback() -> None:
    assert app.pretty_exceptions_show_locals is False


class PlaintextKeyring(FakeBackend):
    """Named like the keyrings.alt backend that keeps passwords in a plain file."""


class AltFileKeyring(FakeBackend):
    pass


AltFileKeyring.__module__ = "keyrings.alt.file"


@pytest.mark.parametrize("backend_class", [PlaintextKeyring, AltFileKeyring])
def test_a_plaintext_or_alt_keyring_is_not_used(tmp_path: Path, backend_class: type) -> None:
    backend = backend_class()
    store = open_store(_env(tmp_path), backend_factory=lambda: backend)
    assert isinstance(store, FileStore)  # the owner-only file, not the plain-text keyring
    name, usable = keyring_status(lambda: backend)
    assert usable is False
    assert backend_class.__name__ in name


def test_an_ordinary_third_party_keyring_is_still_used(tmp_path: Path) -> None:
    class SecretServiceLike(FakeBackend):
        pass

    store = open_store(_env(tmp_path), backend_factory=SecretServiceLike)
    assert isinstance(store, KeyringStore)


def test_the_credentials_helper_runner_never_hands_credentials_to_a_child(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys

    monkeypatch.setenv("VC_OPENSUBTITLES_API_KEY", "k")
    code = "import os; print(os.environ.get('VC_OPENSUBTITLES_API_KEY', 'absent'))"
    assert credentials._run_tool([sys.executable, "-c", code])[1].strip() == "absent"


def _chain(*members: object) -> object:
    import keyring.backends.chainer

    class Chain(keyring.backends.chainer.ChainerBackend):
        backends = tuple(members)

    return Chain()  # type: ignore[no-untyped-call]


def test_a_chain_with_a_plaintext_member_is_not_used(tmp_path: Path) -> None:
    chain = _chain(FakeBackend(), PlaintextKeyring())
    assert isinstance(open_store(_env(tmp_path), backend_factory=lambda: chain), FileStore)
    name, usable = keyring_status(lambda: chain)
    assert usable is False
    assert "PlaintextKeyring" in name  # the reason is visible


def test_a_chain_of_ordinary_backends_is_used(tmp_path: Path) -> None:
    class Vault(FakeBackend):
        pass

    chain = _chain(FakeBackend(), Vault())
    assert isinstance(open_store(_env(tmp_path), backend_factory=lambda: chain), KeyringStore)
    assert keyring_status(lambda: chain)[1] is True


def test_an_empty_chain_is_not_used(tmp_path: Path) -> None:
    chain = _chain()
    assert isinstance(open_store(_env(tmp_path), backend_factory=lambda: chain), FileStore)
