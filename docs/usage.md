# Using video-cleaner

video-cleaner re-encodes a video library to HEVC (x265) to reclaim space. It never modifies the
library itself: it reads originals from the NAS and leaves verified results in a local
`completed` folder, and you replace the originals yourself.

## Setup

From the repository root on Windows:

```powershell
powershell -ExecutionPolicy Bypass -File .\setup.ps1
```

The script is safe to rerun. It creates a Python 3.12 virtual environment with `uv`, installs the
locked dependencies (`uv sync --frozen`), installs ffmpeg (`Gyan.FFmpeg`) and MKVToolNix
(`MoritzBunkus.MKVToolNix`) through winget only, copies `config.example.toml` to
`%APPDATA%\video-cleaner\config.toml` if it is not there yet, restricts that file to your user
with `icacls`, and finally runs `video-cleaner doctor`. Pass `-SkipTools` to skip the winget
installs.

Edit `%APPDATA%\video-cleaner\config.toml` before the first run (at least `library_root`,
`scratch_dir` and `completed_dir`), then rerun `uv run video-cleaner doctor`.

## Commands

All commands are run as `uv run video-cleaner <command>` (or `video-cleaner <command>` inside the
activated virtual environment). `--help` works on every command.

| Command | What it does |
|---|---|
| `doctor [--config PATH]` | Checks the environment and prints one line per check: `[ OK ]`, `[WARN]` (informational check failed) or `[FAIL]` (required check failed). Also prunes old run logs. Exit code 0 when all required checks pass, 1 otherwise, 2 for a configuration error. |
| `survey` | Scans the library read-only and probes every file (codec, resolution, bitrate, HDR, Dolby Vision, subtitles). |
| `plan` | Dry run: decides per file what will happen (re-encode, remux, leave alone, skip and flag) and projects the savings. The plan must be acknowledged before `run`. |
| `calibrate` | Runs sample encodes to set the CRF search and the quality guard thresholds. Its results must be approved before `run`. |
| `run [--until HH:MM] [--max-hours N] [--path X]` | Executes the acknowledged plan, one encode at a time, biggest expected saving first. Stops cleanly after the current file at `--until` or after `--max-hours`; rerunning resumes. |
| `status` | Shows the progress of the current or last run. |
| `report` | Summarises the results of a run. |

`doctor` checks: Python 3.11 or 3.12; ffmpeg found and built with `libx265` and `libvmaf`
(`hevc_nvenc` is informational); ffprobe and mkvmerge found; an NVIDIA GPU visible
(informational); free space on the scratch drive at least `scratch_cap_bytes`; `library_root`
readable; `completed_dir` and `scratch_dir` writable and not inside the library; the config
file not readable by other users (informational); `keyring` importable (informational, needed
from Phase 2). Tools are found through `VC_FFMPEG` / `VC_FFPROBE` / `VC_MKVMERGE`, then `PATH`,
then the default winget and MKVToolNix install folders.

## Run model

1. **Originals are read from the NAS** and copied to local scratch; the next file is prefetched
   while the current one encodes. The tool never writes to, renames or deletes anything under
   `library_root`.
2. **Finished files go to the local `completed` folder**, mirroring each original's relative
   path, together with `completed/manifest.csv`. The manifest lists, per finished file, the
   original path, the completed path, both sizes, the verification and VMAF results, and the
   original's size and modification time as recorded at copy time. A file is kept only if it is
   at least 10% smaller and passes the VMAF guard; otherwise the original stays as it is.
3. **You overwrite the originals on the NAS yourself**, normally at the end of each night, using
   the manifest: it says which originals to remove and which file to place (one-to-one is not
   guaranteed: an MP4/AVI that is re-encoded becomes `.mkv`, multi-part movies become one
   file). Before overwriting, check that the original still has the size and modification time
   in the manifest; if not, the result is stale and must not replace it.
4. **Take a TrueNAS snapshot before every third overwrite session.** Snapshots keep the
   replaced originals' space until they are destroyed, so the saving is realised only after
   old snapshots are deleted.
5. **`run` is refused until a `plan` has been acknowledged and the calibration approved.**

Reruns skip files that are already done: state is keyed by path, size and modification time,
so an overwritten original is seen as already HEVC and left alone.

A run can also decide to leave a file alone or flag it (projected saving below `min_saving`,
quality guard or verification failure, output name collision). A re-survey of the unchanged
file keeps that outcome, so the next run does not repeat the same hours-long encode. To try such
a file again deliberately, run `video-cleaner plan --retry <path>` with the path as surveyed:
the file goes back to the status survey gave it (pending for an encode or remux) and the next
`run` picks it up. A file whose size or modification time changed is planned afresh anyway.

## Settings

Settings come from the defaults, then `%APPDATA%\video-cleaner\config.toml`, then environment
variables named `VC_<SETTING IN UPPERCASE>` (for example `VC_VMAF_TARGET=94`). Unknown keys are
an error. Use TOML literal strings (single quotes) for Windows paths.

| Setting | Default | Meaning |
|---|---|---|
| `library_root` | required | Library root, read-only (for example `'\\nas\media'`). |
| `scratch_dir` | required | Local working area for copies and encodes. |
| `completed_dir` | required | Local folder for finished files and `manifest.csv`. |
| `state_db` | `%LOCALAPPDATA%\video-cleaner\state.db` | State database. |
| `log_dir` | `%LOCALAPPDATA%\video-cleaner\logs` | Per-run log files. |
| `scratch_cap_bytes` | `2000000000000` (2 TB) | Maximum scratch use; `doctor` requires this much free space. |
| `min_file_size` | `0` | Ignore files smaller than this many bytes. |
| `bitrate_floor_kbps` | `2500` | Skip files below this video bitrate (set by calibration). |
| `vmaf_target` | `95.0` | VMAF the per-file CRF search aims for. |
| `crf_min` / `crf_max` | `16` / `24` | Clamp for the chosen CRF. |
| `guard_mean_floor` | `93.0` | Post-encode guard: minimum mean VMAF over the sample windows. |
| `guard_worst_floor` | `88.0` | Post-encode guard: minimum worst-window VMAF (set by calibration). |
| `min_saving` | `0.10` | Keep the output only if it is at least this fraction smaller. |
| `preset` | `slow` | x265 preset. |
| `heartbeat_s` | `15.0` | The console is never silent longer than this. |
| `plain_progress_s` | `30.0` | Progress line interval when output is not a terminal. |

## Logs

Every run writes a DEBUG log file `run-YYYYMMDD-HHMMSS.log` to `log_dir`
(default `%LOCALAPPDATA%\video-cleaner\logs\`). Secrets are redacted. Failing ffmpeg and
mkvmerge invocations keep their stderr in a per-file log next to it. Run logs older than 30
days are deleted, then the oldest until the run logs total at most 200 MB; other files in the
folder are never touched. `doctor` applies this retention and prints how many logs it removed.
