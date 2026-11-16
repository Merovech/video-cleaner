# video-cleaner

A Python command-line tool that re-encodes a video library to HEVC (x265) to reclaim disk space,
without ever modifying the source library.

Status: early development (Phase 1 foundation). The commands `doctor`, `survey`, `plan`,
`calibrate`, `run`, `status` and `report` exist as stubs.

## Principles

- The library root is read-only: the tool never writes to, renames or deletes anything there.
- A re-encode is accepted only if the result is at least 10% smaller; otherwise the original stays.
- Audio, chapters, attachments and all audio/subtitle tracks are copied untouched.
- HDR, Dolby Vision, interlaced, 4K, already-HEVC/AV1 and low-bitrate files are flagged and skipped.

## Requirements

- Python 3.11 or 3.12
- [uv](https://docs.astral.sh/uv/)
- ffmpeg and ffprobe built with `libx265` and `libvmaf`

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
