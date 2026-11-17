# video-cleaner

A Python command-line tool that re-encodes a video library to HEVC (x265) to reclaim disk space,
without ever modifying the source library.

Status: early development (Phase 1). `doctor`, `survey`, `plan`, `calibrate`, `run`, `status`
and `report` work: `run` encodes with x265 at a per-file CRF found by a VMAF sample search and
keeps an output only if it passes the VMAF guard and is at least 10% smaller. See
[docs/usage.md](docs/usage.md) for the commands, the run model, all settings and where logs go.

## Setup (Windows)

```powershell
powershell -ExecutionPolicy Bypass -File .\setup.ps1
```

Creates the Python 3.12 environment with `uv`, installs ffmpeg and MKVToolNix through winget,
creates a private `%APPDATA%\video-cleaner\config.toml` from `config.example.toml` and runs
`video-cleaner doctor`. Safe to rerun.

## Principles

- The library root is read-only: the tool never writes to, renames or deletes anything there.
- A re-encode is accepted only if the result is at least 10% smaller; otherwise the original stays.
- Audio, chapters, attachments and all audio/subtitle tracks are copied untouched.
- 4K, already-HEVC/AV1, low-bitrate and small files are left alone (not re-encoded).
- Encode candidates that are HDR, Dolby Vision, interlaced or have several video streams are
  flagged and skipped for review.

## Requirements

- Python 3.11 or 3.12
- [uv](https://docs.astral.sh/uv/)
- ffmpeg and ffprobe built with `libx265` and `libvmaf`
- MKVToolNix (`mkvmerge`)

## Development

```bash
uv sync
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run video-cleaner --help
```

Copy `config.example.toml` to `%APPDATA%\video-cleaner\config.toml` and edit it. Individual values
can be overridden with `VC_*` environment variables.

## Licence

MIT. See [LICENSE](LICENSE).
