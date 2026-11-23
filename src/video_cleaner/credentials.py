"""OpenSubtitles credentials: where they are kept and how they are read.

A credential is stored by name in the operating system's keyring (Windows Credential Manager)
when it has a usable backend, otherwise in an owner-only JSON file beside the config
(``%LOCALAPPDATA%\\video-cleaner\\credentials.json``). The ``VC_OPENSUBTITLES_*`` environment
variables override what is stored. Nothing here prints or logs a value, and every value
`load_credentials` returns is registered with `logging_setup.register_secrets`, so a log line or
an error text that happens to contain one is masked.
"""

from __future__ import annotations

import contextlib
import getpass
import json
import os
import subprocess  # nosec B404
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, cast

import keyring
import keyring.backends.fail
import keyring.backends.null
import keyring.errors

from video_cleaner.config import APP_DIR_NAME, env_var
from video_cleaner.logging_setup import register_secrets

SERVICE = "video-cleaner"

API_KEY = "opensubtitles_api_key"
USERNAME = "opensubtitles_username"
PASSWORD = "opensubtitles_password"  # noqa: S105  # nosec B105  (a credential name, not a value)
TOKEN = "opensubtitles_token"  # noqa: S105  # nosec B105
LOGIN_NAMES = (API_KEY, USERNAME, PASSWORD)  # what the user supplies
ALL_NAMES = (*LOGIN_NAMES, TOKEN)  # `auth --clear` removes all of them (the token is cached)

CREDENTIALS_FILE_NAME = "credentials.json"
DEFAULT_SYSTEM_ROOT = "C:/Windows"
ICACLS_TIMEOUT_S = 20.0

OWNER_ONLY_MODE = 0o600


class CredentialError(Exception):
    """A credential store could not be opened, read or written. Never carries a value."""


class CredentialStore(Protocol):
    def get(self, name: str) -> str | None: ...

    def set(self, name: str, value: str) -> None: ...

    def delete(self, name: str) -> None: ...


class _KeyringBackend(Protocol):
    """The part of a ``keyring`` backend `KeyringStore` uses."""

    def get_password(self, service_name: str, username: str, /) -> str | None: ...

    def set_password(self, service_name: str, username: str, password: str, /) -> None: ...

    def delete_password(self, service_name: str, username: str, /) -> None: ...


class KeyringStore:
    """Credentials in the operating system's keyring, under the service `SERVICE`."""

    def __init__(self, backend: _KeyringBackend | None = None) -> None:
        self._backend: _KeyringBackend = keyring.get_keyring() if backend is None else backend

    def get(self, name: str) -> str | None:
        try:
            return self._backend.get_password(SERVICE, name)
        except keyring.errors.KeyringError as exc:
            raise CredentialError(f"keyring read of {name} failed: {type(exc).__name__}") from None

    def set(self, name: str, value: str) -> None:
        try:
            self._backend.set_password(SERVICE, name, value)
        except keyring.errors.KeyringError as exc:
            raise CredentialError(f"keyring write of {name} failed: {type(exc).__name__}") from None

    def delete(self, name: str) -> None:
        try:
            self._backend.delete_password(SERVICE, name)
        except keyring.errors.PasswordDeleteError:
            pass  # nothing stored under that name
        except keyring.errors.KeyringError as exc:
            raise CredentialError(
                f"keyring delete of {name} failed: {type(exc).__name__}"
            ) from None


Runner = Callable[[Sequence[str]], tuple[int, str]]


def _run_icacls(cmd: Sequence[str]) -> tuple[int, str]:
    # Fixed argument list, shell=False.
    proc = subprocess.run(  # noqa: S603  # nosec B603
        [str(part) for part in cmd],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=ICACLS_TIMEOUT_S,
        check=False,
        shell=False,
    )
    return proc.returncode, proc.stdout + proc.stderr


def restrict_to_owner(
    path: Path,
    *,
    windows: bool | None = None,
    user: str | None = None,
    system_root: str | None = None,
    run: Runner | None = None,
    chmod: Callable[[Path, int], None] | None = None,
) -> None:
    """Make *path* readable and writable by its owner only.

    POSIX: ``chmod 600``. Windows: ``icacls <path> /inheritance:r /grant:r <user>:F`` (the
    folder's inherited rights are removed, the user keeps full control), run from
    ``System32`` by full path and without a shell. The keyword arguments exist for tests.
    """
    on_windows = os.name == "nt" if windows is None else windows
    if not on_windows:
        (chmod or os.chmod)(path, OWNER_ONLY_MODE)
        return
    root = system_root or os.environ.get("SYSTEMROOT") or DEFAULT_SYSTEM_ROOT
    icacls = str(Path(root) / "System32" / "icacls.exe")
    grant = f"{user or getpass.getuser()}:F"
    try:
        code, output = (run or _run_icacls)(
            [icacls, str(path), "/inheritance:r", "/grant:r", grant]
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise CredentialError(f"cannot run icacls to restrict {path}: {exc}") from None
    if code != 0:
        first = next((line.strip() for line in output.splitlines() if line.strip()), "")
        raise CredentialError(f"icacls could not restrict {path} (exit {code}): {first}")


@dataclass(frozen=True)
class FileStore:
    """Credentials in a JSON file only the owner can open (the keyring-less fallback).

    Every write goes to a new temporary file beside the target, which is restricted *before* the
    secrets are written into it and then moved over the target, so the values are never in a file
    other users can read, and a crash never leaves a half-written file.
    """

    path: Path
    restrict: Callable[[Path], None] = field(default=restrict_to_owner, repr=False)

    def _read(self) -> dict[str, str]:
        try:
            text = self.path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return {}
        except OSError as exc:
            raise CredentialError(f"cannot read {self.path}: {type(exc).__name__}") from None
        try:
            data = json.loads(text)
        except ValueError:
            data = None
        if not isinstance(data, dict) or not all(
            isinstance(k, str) and isinstance(v, str) for k, v in data.items()
        ):
            raise CredentialError(
                f"{self.path} is not a credentials file (a JSON object of text values); "
                "delete it and run `auth` again"
            )
        return data

    def _write(self, data: Mapping[str, str]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(
            prefix=f".{self.path.name}.", suffix=".tmp", dir=self.path.parent
        )
        os.close(fd)
        try:
            self.restrict(Path(tmp))  # still empty here
            with open(tmp, "w", encoding="utf-8", newline="") as fh:
                json.dump(dict(data), fh, indent=2)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, self.path)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp)
            raise

    def get(self, name: str) -> str | None:
        return self._read().get(name)

    def set(self, name: str, value: str) -> None:
        data = self._read()
        data[name] = value
        self._write(data)

    def delete(self, name: str) -> None:
        data = self._read()
        if name in data:
            del data[name]
            self._write(data)


def credentials_file_path(env: Mapping[str, str] | None = None) -> Path | None:
    """The fallback credentials file, or None without ``LOCALAPPDATA``.

    The single source of this rule: `doctor` checks the permissions of the same file.
    """
    environ: Mapping[str, str] = os.environ if env is None else env
    local_appdata = environ.get("LOCALAPPDATA")
    return Path(local_appdata) / APP_DIR_NAME / CREDENTIALS_FILE_NAME if local_appdata else None


def _default_keyring_backend() -> object | None:
    return keyring.get_keyring()


def _usable(backend: object | None) -> bool:
    """False for no backend, the library's "no backend found" one and the null one."""
    return backend is not None and not isinstance(
        backend, (keyring.backends.fail.Keyring, keyring.backends.null.Keyring)
    )


def open_store(
    env: Mapping[str, str] | None = None,
    *,
    backend_factory: Callable[[], object | None] | None = None,
) -> CredentialStore:
    """The keyring when it has a usable backend, otherwise the owner-only `FileStore`.

    *backend_factory* (tests) returns the keyring backend to consider; it defaults to
    ``keyring.get_keyring``. A factory that raises counts as "no usable backend".
    """
    factory = backend_factory or _default_keyring_backend
    try:
        backend = factory()
    except Exception:
        backend = None
    if _usable(backend):
        return KeyringStore(cast("_KeyringBackend", backend))
    path = credentials_file_path(env)
    if path is None:
        raise CredentialError(
            "no usable keyring and LOCALAPPDATA is not set, so there is nowhere to keep "
            "credentials; use the VC_OPENSUBTITLES_* environment variables"
        )
    return FileStore(path)


@dataclass(frozen=True)
class OpenSubtitlesCredentials:
    api_key: str = field(repr=False)
    username: str = field(repr=False)
    password: str = field(repr=False)


def _from_env(env: Mapping[str, str], name: str) -> str | None:
    return env.get(env_var(name)) or None


def _lookup(store: CredentialStore, env: Mapping[str, str], name: str) -> str | None:
    """The environment's value of *name*, else the store's; None when empty or absent."""
    return _from_env(env, name) or store.get(name) or None


def source_of(store: CredentialStore, env: Mapping[str, str] | None, name: str) -> str | None:
    """Where *name* comes from: ``"environment"``, ``"store"`` or None when it is not set.

    Only the three login values can come from the environment; the cached token is stored only.
    """
    environ: Mapping[str, str] = os.environ if env is None else env
    if name in LOGIN_NAMES and _from_env(environ, name):
        return "environment"
    return "store" if store.get(name) else None


def missing_names(store: CredentialStore, env: Mapping[str, str] | None = None) -> list[str]:
    """The names of the three login values neither the environment nor *store* has."""
    environ: Mapping[str, str] = os.environ if env is None else env
    return [name for name in LOGIN_NAMES if _lookup(store, environ, name) is None]


def load_credentials(
    store: CredentialStore, env: Mapping[str, str] | None = None
) -> OpenSubtitlesCredentials | None:
    """The OpenSubtitles login, or None when any of the three values is missing.

    ``VC_OPENSUBTITLES_API_KEY`` / ``_USERNAME`` / ``_PASSWORD`` win over the store, value by
    value. What is returned is registered for redaction before it leaves this function.
    """
    environ: Mapping[str, str] = os.environ if env is None else env
    values = [_lookup(store, environ, name) for name in LOGIN_NAMES]
    api_key, username, password = values
    if api_key is None or username is None or password is None:
        return None
    register_secrets([api_key, username, password])
    return OpenSubtitlesCredentials(api_key, username, password)
