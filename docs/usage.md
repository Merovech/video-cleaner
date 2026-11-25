<!-- TODO: when the project is complete, rewrite this document in the style of a man page
     (synopsis, options, exit codes) instead of a walkthrough. -->

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
`%LOCALAPPDATA%\video-cleaner\config.toml` if it is not there yet, restricts that file to your user
with `icacls`, and finally runs `video-cleaner doctor`. Pass `-SkipTools` to skip the winget
installs.

Edit `%LOCALAPPDATA%\video-cleaner\config.toml` before the first run (at least `library_root`,
`scratch_dir` and `completed_dir`), then rerun `uv run video-cleaner doctor`.

## Commands

All commands are run as `uv run video-cleaner <command>` (or `video-cleaner <command>` inside the
activated virtual environment). `--help` works on every command.

| Command | What it does |
|---|---|
| `doctor` | Checks the environment and prints one line per check: `[ OK ]`, `[WARN]` (informational check failed) or `[FAIL]` (required check failed). Also prunes old run logs. Exit code 0 when all required checks pass, 1 otherwise. |
| `survey [--root X]` | Scans the library (or only folder `X`) read-only and probes every file (codec, resolution, bitrate, HDR, Dolby Vision, subtitles) and stores its OpenSubtitles movie hash. Writes `survey.csv` beside the state database. |
| `plan [--ack] [--retry PATH]` | Dry run: decides per file what will happen (re-encode, remux, leave alone, skip and flag) and projects the savings. Writes `plan.csv` and `plan.hash` beside the state database. `--ack` acknowledges the plan last written (its hash must still match the current state). `--retry PATH` puts a file a run left alone or flagged back in the queue. |
| `calibrate --samples N` | Encodes N sample files at CRF 17/20/23/26, scores them with VMAF and writes `calibration.txt` beside the state database and side-by-side clips to `completed_dir\calibration`. |
| `calibrate --approve --vmaf-target X --crf-max Y` | Approves a VMAF target (`guard_mean_floor` to 100) and a CRF ceiling (`crf_min` to 51). A run uses these instead of the config's `vmaf_target` and `crf_max`. Refused until a `calibrate --samples` sweep has finished. |
| `run [--until HH:MM] [--max-hours N] [--path X]` | Executes the acknowledged plan, one encode at a time, biggest expected saving first. Stops cleanly after the current file at `--until` or after `--max-hours`; rerunning resumes. Only one `run` or `calibrate` runs at a time. |
| `status` | Shows the number of files per status, whether the plan is acknowledged (and still matches) and the calibration approved, and the last 10 events. |
| `auth [--status] [--clear]` | Stores the OpenSubtitles API key, username and password. It prompts for the three values with hidden input (no value is accepted as an option or an argument), keeps them in the Windows Credential Manager through `keyring` (or, when the system has no usable keyring, in `%LOCALAPPDATA%\video-cleaner\credentials.json`, which only your user can open) and forgets the cached login token. `--status` prints one `[ OK ]` / `[WARN]` line per name saying whether it is set, never a value. `--clear` deletes all four stored names. A store that cannot be read or written gives a `[FAIL]` line naming the kind of error (never a value) and exit code 2. `VC_OPENSUBTITLES_API_KEY`, `VC_OPENSUBTITLES_USERNAME` and `VC_OPENSUBTITLES_PASSWORD` override what is stored. Takes `--config`. |
| `report [--output FILE]` | Writes the results so far to `report.csv` beside the state database (or `FILE`) and prints a summary, including for every manifest entry whether its original(s) are still unchanged (run it before overwriting originals). |

**Current state (Phase 1):** every command works. `run` searches a CRF per file (3 sample
windows of 30 s encoded at CRF 20 and 24 and scored with VMAF, interpolated to the approved VMAF
target and clamped to `crf_min`..the approved `crf_max`), leaves the file alone when the projected
saving is under `min_saving`, encodes, muxes, scores 4 windows of 20 s against the original (the
quality guard) and verifies the output. `calibrate --samples` reads the sample files from the
library directly and read-only.

`survey`, `plan`, `calibrate` and `run` take `--config PATH` (default
`%LOCALAPPDATA%\video-cleaner\config.toml`), `-v`/`--verbose` (repeat for more console logging) and
`-q`/`--quiet` (warnings only) and `--log-file PATH` (this run's log goes there instead of
`log_dir`); `doctor`, `status`, `report` and `auth` take `--config` only.

`survey`, `calibrate` and `run` also take `--progress auto|rich|plain`. The default, `auto`, shows
a live in-place progress line (spinner, stage, bar, percent, fps and ETA) when the output is an
interactive terminal, and plain lines every `plain_progress_s` seconds otherwise (a pipe, a file,
a scheduled task). `rich` always draws the live line, even when the output is piped, which only
looks right if the program reading it is a terminal; `plain` always writes plain lines, which
suits logs. Git Bash's default console often is not detected as a terminal, so use `--progress
rich` there, or run from Windows Terminal or PowerShell.

Console lines the tool writes about its own work start with a level tag: `[ OK ]` (green),
`[INFO]` (white), `[WARN]` (yellow), `[FAIL]` (red) or `[VERB]` (grey, the detail `-v` adds);
only the text inside the brackets is coloured, and colour is dropped when the output is not a
terminal. A file's closing line is `[ OK ]` when it completed, `[WARN]` when it was flagged or must be retried and `[INFO]` when it
was left alone. While the live bar is up, these lines appear above it. The plain progress lines
carry the tag and a time (`[INFO] 07:30:05 encode ...`) and include a "still working" line
every `heartbeat_s` seconds for tools that report no percentage; the live bar has a spinner
instead and prints none. The run log file keeps its own, more detailed format.

Exit codes: 0 success; 1 a run that stopped (scratch full, library unreachable, ...) or a
failed required `doctor` check; 2 a refusal (plan not acknowledged, calibration not approved,
an output overlapping the library, another run active, no state database yet, a stored calibration
that cannot be read), a configuration error or a calibration that failed (nothing to sample, an
encode, probe or VMAF error, an unusable scratch folder); 130 interrupted with Ctrl+C.

`doctor` checks: Python 3.11 or 3.12; ffmpeg found and built with `libx265` and `libvmaf`
(`hevc_nvenc` is informational); ffprobe and mkvmerge found; an NVIDIA GPU visible
(informational); free space on the scratch drive at least `scratch_cap_bytes`; `library_root`
readable; the state database, log folder, subtitle cache, `completed_dir` and `scratch_dir` neither
inside the library nor containing it ("library overlap"); `completed_dir` and `scratch_dir` writable (a
missing one is reported as "does not exist (will be created)"; `doctor` creates nothing); the
config file not readable by other users (informational); `keyring` importable (informational);
the OpenSubtitles credentials set (a warning, not a failure, when they are missing); and, when the
fallback `credentials.json` exists, that other users cannot read it (informational). Tools are found through `VC_FFMPEG` / `VC_FFPROBE` / `VC_MKVMERGE`, then `PATH`,
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
   The acknowledgement covers the plan as it was reviewed: after any re-survey, or after a run
   found a changed (stale) original, run `plan`, review it and `plan --ack` again. The approved
   `vmaf_target` and `crf_max` override the config values for every run.
6. **One `run` or `calibrate` at a time.** A second one is refused (exit 2) with the process ID of
   the one holding `video-cleaner.lock` beside the state database. The lock is released by the
   operating system when the process ends, so a crash never leaves it stuck.

7. **Low impact.** ffmpeg, ffprobe and mkvmerge run at below-normal priority, so the PC stays
   usable, and Windows is kept from sleeping while `run` works (the normal power settings
   return when it ends).

### The manifest and the completed folder

`manifest.csv` accumulates: every run appends the files it delivered, and rows are never removed
except to repair an interrupted delivery (see below). `original_mtime` is the original's
modification time as a float of seconds since the epoch (what `os.stat` reports), so compare it
numerically, not as a date string. `video-cleaner report` does this check for you: it prints
`yes`/`no`/`unknown` per completed file and writes it to the `original_unchanged` column of
`report.csv`; only a `yes` file may replace its original(s). After you have overwritten the originals of some rows on the
NAS, you may move or delete those completed files; the rows can stay (a row whose file is gone is left as
history, even after you overwrite the original and re-survey it) or you can archive the manifest and start a
new one between runs. `completed_dir\calibration` holds calibration clips for viewing only:
never copy it to the NAS.

A file is delivered as `<name>.mkv.part`, its manifest row is written, and only then is it
renamed to `<name>.mkv`; a rename never replaces an existing file. If a run is killed in
between, the next run finishes the rename, drops a row that has no file at all only if that
original's record was left unfinished, removes a
`*.mkv.part` that has no row, and marks the file completed when its delivered output is complete.

### Scratch

All scratch work goes to `scratch_dir\.video-cleaner`, a folder the tool creates with a marker
file. Only that folder is swept and counted toward `scratch_cap_bytes`, so `scratch_dir` may be a
shared folder; a dedicated folder is still recommended.

Reruns skip files that are already done: state is keyed by path, size and modification time.
An overwritten original is seen as already HEVC and left alone only after a re-survey (which
also requires `plan` and `plan --ack` again); until then the old record stays completed.

A run can also decide to leave a file alone or flag it (projected saving below `min_saving`,
quality guard or verification failure, output name collision). A re-survey of the unchanged
file keeps that outcome, so the next run does not repeat the same hours-long encode. To try such
a file again deliberately, run `video-cleaner plan --retry <path>` with the path as surveyed:
the file goes back to the status survey gave it (pending for an encode or remux) and the next
`run` picks it up. A file whose size or modification time changed is planned afresh anyway.

A file that makes a run fail (a tool error or an I/O error on that file) is retried by the next
run; after the third such failure it is flagged instead, with the last error in its note.
`plan --retry` also resets that count. Stop conditions that are not the file's fault (scratch
full, a tool not found, the library or the manifest unreachable) do not count.

## Settings

Settings come from the defaults, then `%LOCALAPPDATA%\video-cleaner\config.toml`, then environment
variables named `VC_<SETTING IN UPPERCASE>` (for example `VC_VMAF_TARGET=94`). Unknown keys are
an error. Use TOML literal strings (single quotes) for Windows paths. Environment variables such
as `%LOCALAPPDATA%` are not expanded inside the file, so a path override is written out in full
(for example `state_db = 'C:\Users\you\AppData\Local\video-cleaner\state.db'`).

`config.example.toml` holds only the three required paths; every other setting below is optional
and has the default shown.

| Setting | Default | Meaning |
|---|---|---|
| `library_root` | required | Library root, read-only (for example `'\\nas\media'`). |
| `scratch_dir` | required | Local working area; the tool works only in its `.video-cleaner` subfolder. A dedicated folder is recommended. |
| `completed_dir` | required | Local folder for finished files and `manifest.csv`. |
| `state_db` | `%LOCALAPPDATA%\video-cleaner\state.db` | State database. |
| `log_dir` | `%LOCALAPPDATA%\video-cleaner\logs` | Per-run log files. |
| `scratch_cap_bytes` | `2000000000000` (2 TB) | Maximum scratch use; `doctor` requires this much free space. |
| `min_file_size` | `0` | Files smaller than this many bytes are not re-encoded (they can still be remuxed as part of a multi-part group). |
| `bitrate_floor_kbps` | `2500` | Files whose overall container bitrate (video plus audio) is below this are not re-encoded. A placeholder you set in the config; calibration does not change it. |
| `vmaf_target` | `95.0` | VMAF the per-file CRF search aims for. Overridden by the approved calibration. |
| `crf_min` / `crf_max` | `16` / `24` | Clamp for the chosen CRF. `crf_max` is overridden by the approved calibration. |
| `guard_mean_floor` | `93.0` | Post-encode guard: minimum mean VMAF over the sample windows. |
| `guard_worst_floor` | `88.0` | Post-encode guard: minimum worst-window VMAF. A placeholder you set in the config after reviewing the calibration table; calibration does not change it. |
| `min_saving` | `0.10` | Keep the output only if it is at least this fraction smaller. |
| `preset` | `slow` | x265 preset. |
| `heartbeat_s` | `15.0` | The console is never silent longer than this. |
| `plain_progress_s` | `30.0` | Progress line interval when output is not a terminal. |
| `log_file_level` | `DEBUG` | Level of the run log file (`DEBUG`, `INFO`, `WARNING`, `ERROR`). |
| `subtitle_cache_dir` | `%LOCALAPPDATA%\video-cleaner\subtitles` | Local folder for downloaded and prepared subtitles. Refused (like every other output) when it is inside `library_root` or contains it. |
| `local_subtitle_dirs` | `[]` | Folders searched for subtitles you already have: a TOML array of absolute paths (`['D:\subs', 'E:\more']`), or in `VC_LOCAL_SUBTITLE_DIRS` several paths separated by `;` on Windows. |
| `forced_full_cues_per_hour` | `250.0` | Cues per hour of video above which a subtitle counts as a full one rather than forced-only. A placeholder, calibrated against the real library. |
| `align_max_offset_s` | `120.0` | Largest shift in seconds the alignment may apply to a subtitle. |
| `align_scale_tolerance` | `0.05` | Largest speed change (fraction, `0` to below `1`) the alignment may apply. |
| `align_min_score` | `0.0` | Lowest alignment score accepted; `0` disables the check. Set after the review of the first results. |
| `subtitle_max_bytes` | `2000000` | A downloaded subtitle larger than this is rejected. |
| `subtitle_min_cues` | `20` | A subtitle with fewer cues is rejected. |
| `subtitle_candidates_per_file` | `3` | How many OpenSubtitles downloads one video may use in total, across lookups (`subs retry` resets the count). |
| `opensubtitles_domain` | `opensubtitles.com` | The service's domain. The credentials and every download go only to this domain and its subdomains; a login reply, a download link or a redirect that names another domain is refused. Change it only if the service moves, and only to a domain you trust: it is the trust boundary. Two or more DNS labels, no scheme, port or path. |
| `opensubtitles_api_host` | `api.opensubtitles.com` | The API host used for the login. Must be `opensubtitles_domain` or a host under it, so changing the domain means changing this too. |
| `opensubtitles_api_path` | `/api/v1` | Path prefix of the API on that host (no query, no `..`). |
| `subtitle_language` | `en` | Language code asked of the service and accepted in its results (`en`, `pt-br`, ...). Matching of embedded tracks and local files, and the alignment, are still written for English. |

## Logs

Every run writes a DEBUG log file `run-YYYYMMDD-HHMMSS.log` to `log_dir`
(default `%LOCALAPPDATA%\video-cleaner\logs\`). Secrets are redacted. Failing ffmpeg and
mkvmerge invocations keep their stderr in a per-file log next to it. Run logs older than 30
days are archived, then the oldest until the run logs total at most 200 MB. Nothing is deleted:
each old log is added to a zip named for its month (`archive\run-logs-YYYY-MM.zip` under
`log_dir`) and removed from the folder only after the copy in the zip has been read back and
matches. A log that cannot be archived stays where it is. The archive is not pruned; delete old
zips yourself when you no longer need them. Other files in the folder are never touched.
`doctor` and `run` apply this retention; `doctor` prints how many logs it archived.
