<!-- TODO: when the project is complete, rewrite this document in the style of a man page
     (synopsis, options, exit codes) instead of a walkthrough. -->

# Using video-cleaner

video-cleaner re-encodes a video library to HEVC (x265) to reclaim space. It never modifies the
library itself: it reads originals from the NAS and leaves verified results in a local
`completed` folder, and you replace the originals yourself. For files that have no English
subtitle it can also find one and add it as a new track (see [Subtitles](#subtitles)).

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

The locked dependencies include `ffsubsync` (subtitle alignment) and `keyring` (credential
storage). Nothing extra is needed for them on Windows: `uv.lock` has prebuilt wheels for every
compiled dependency on Python 3.12 (the two pure-Python ones without a wheel, `auditok` and `srt`,
are built by `uv` from source), and `ffsubsync` finds `ffmpeg` because the tool puts the folder
of the `ffmpeg` it located on the `PATH` of the aligner it starts.

Edit `%LOCALAPPDATA%\video-cleaner\config.toml` before the first run (at least `library_root`,
`scratch_dir` and `completed_dir`), then rerun `uv run video-cleaner doctor`. To use OpenSubtitles
run `uv run video-cleaner auth` once (see [`auth`](#commands)); sidecar files and your own
subtitle folders work without it.

## Daily routine

1. **`survey`** once, and again after you add or replace files. Besides probing each file it
   classifies the file's own subtitle tracks and stores one subtitle record per video. After
   upgrading from a version without subtitle support, survey again: every video that can be
   processed then gets a subtitle record, which changes the plan.
2. **`plan`, review `plan.csv`, then `plan --ack`.** The plan hash includes each file's subtitle
   status (and, for a found subtitle, which subtitle), so the plan changes whenever a subtitle is
   found, replaced, goes missing or ends up unavailable or unclear, and `run` is refused until you
   have reviewed and acknowledged the new plan. `plan.csv` has a `subtitle` column with each
   file's status.
3. **`subs resolve`** each day. It looks up a subtitle for the files that wait for one, biggest
   expected saving first, and stops cleanly when the day's OpenSubtitles downloads are used up;
   the next day it carries on. It keeps its own lock (`video-cleaner.subs.lock` beside the state
   database), so it may run beside a `run`; a file whose subtitle is found while a run is under
   way is left to the next run, which needs a new `plan` and `plan --ack` first.
4. **`run`** overnight (`run --until 07:00`, say). A file that is queued but whose subtitle is
   still wanted is deferred: the run skips it and says how many it skipped, so you do not have to
   wait for every lookup. `run --include-unresolved` runs those files too, without a subtitle,
   and such a file then never gets one from this tool (see [Statuses](#statuses)).

A normal day is therefore `subs resolve`, `plan`, review, `plan --ack`, `run`. Acknowledge again
after every `subs resolve` that changed the plan; it prints "The plan changed" when it did.

## Commands

All commands are run as `uv run video-cleaner <command>` (or `video-cleaner <command>` inside the
activated virtual environment). `--help` works on every command.

| Command | What it does |
|---|---|
| `doctor` | Checks the environment and prints one line per check: `[ OK ]`, `[WARN]` (informational check failed) or `[FAIL]` (required check failed). Also prunes old run logs. Exit code 0 when all required checks pass, 1 otherwise. |
| `survey [--root X]` | Scans the library (or only folder `X`) read-only and probes every file (codec, resolution, bitrate, HDR, Dolby Vision, subtitles) and stores its OpenSubtitles movie hash. It also classifies each file's own subtitle tracks and stores a subtitle record (see [Subtitles](#subtitles)); for a forced-only English text track it reads the track with ffmpeg to count its cues. Writes `survey.csv` beside the state database, with the extra columns `english_subtitle` (`embedded`: a full English track is there; `reclassified`: an English track marked forced is dense enough to count as a full one; `none`: no usable English track; `unclear`: see the `unclear` status below; empty: the file gets no subtitle record, for example it cannot be probed or has no video stream) and `subtitle_status` (the stored status, empty when there is no record). Refused while a `subs resolve` is running. |
| `plan [--ack] [--retry PATH]` | Dry run: decides per file what will happen (re-encode, remux, leave alone, skip and flag) and projects the savings. Writes `plan.csv` (with a `subtitle` column: each file's subtitle status) and `plan.hash` beside the state database. The hash covers the subtitle status, so it changes when a subtitle is found or a lookup ends. `--ack` acknowledges the plan last written (its hash must still match the current state). `--retry PATH` puts a file a run left alone or flagged back in the queue. |
| `calibrate --samples N` | Encodes N sample files at CRF 17/20/23/26, scores them with VMAF and writes `calibration.txt` beside the state database and side-by-side clips to `completed_dir\calibration`. |
| `calibrate --approve --vmaf-target X --crf-max Y` | Approves a VMAF target (`guard_mean_floor` to 100) and a CRF ceiling (`crf_min` to 51). A run uses these instead of the config's `vmaf_target` and `crf_max`. Refused until a `calibrate --samples` sweep has finished. |
| `run [--until HH:MM] [--max-hours N] [--path X] [--include-unresolved]` | Executes the acknowledged plan, one encode at a time, biggest expected saving first. Stops cleanly after the current file at `--until` or after `--max-hours`; rerunning resumes. Only one `run` or `calibrate` runs at a time. A queued file whose subtitle is still `wanted` is skipped (the summary says how many); `--include-unresolved` runs those files too, without a subtitle, and a file delivered that way is never looked up afterwards (see [Statuses](#statuses)). A file whose subtitle was found after the run started is also left for the next run (the summary says so; `plan` and `plan --ack` first). |
| `status` | Shows the number of files per status, how many files wait on subtitles, whether the plan is acknowledged (and still matches) and the calibration approved, and the last 10 events. |
| `auth [--status] [--clear]` | Stores the OpenSubtitles API key, username and password. It prompts for the three values with hidden input (no value is accepted as an option or an argument), keeps them in the Windows Credential Manager through `keyring` (or, when the system has no usable keyring, or only a `keyrings.alt` backend, one named as plain text, or a chain containing one, in `%LOCALAPPDATA%\video-cleaner\credentials.json`, which only your user can open) and forgets the cached login token. `--status` prints one `[ OK ]` / `[WARN]` line per name saying whether it is set, never a value. `--clear` deletes all four stored names. A store that cannot be read or written gives a `[FAIL]` line naming the kind of error (never a value) and exit code 2. `VC_OPENSUBTITLES_API_KEY`, `VC_OPENSUBTITLES_USERNAME` and `VC_OPENSUBTITLES_PASSWORD` override what is stored. `--status` and `--clear` together are refused. Takes `--config`. Without stored credentials `subs resolve` still uses sidecar files and your local folders and leaves the other files `wanted`. |
| `subs resolve [--limit N] [--until HH:MM] [--path X]` | Looks up a subtitle for every file whose subtitle is `wanted` (sidecar, local folders, then OpenSubtitles; see [Where subtitles come from](#where-subtitles-come-from)), biggest expected saving first, under its own lock (only one at a time). `--limit N` attempts at most N files this time; `--until HH:MM` starts no new file after that time of day (it rolls to tomorrow when the time has passed); `--path X` only looks at that file or folder (a part of a multi-part film selects the film). At the start it checks that the cached file of every found subtitle (of files no run has delivered, is working on or has decided on, and within `--path`) is still there and unchanged, and sends a missing or changed one back to `wanted` (see [The cache](#the-cache)). It prints one line per file, then `Subtitles: found N, unavailable N, deferred N (allowance), skipped N`, then `Finished` or `Stopped: <reason>`. Running out of the day's allowance, `--limit`, `--until` and Ctrl+C are normal stops; exit code 1 means a problem to fix (the cache folder is not writable or overlaps a video, or OpenSubtitles failed for three files in a row); 2 a refusal, before anything is looked up (no state database, another `subs resolve` active, a bad option or setting, a cache folder that overlaps the library, a credential store that cannot be read, ffmpeg, ffprobe or mkvmerge not found, or ffsubsync not usable: it is started with `--help` first, with or without credentials, so a broken install costs no download); 130 Ctrl+C. Takes `--config`, `-v`, `-q`, `--log-file` and `--progress` like `run`. |
| `subs status` | Shows the number of subtitle records per status, the OpenSubtitles downloads left and when they renew (UTC), the files waiting on subtitles and how many wanted subtitles are not looked up because a run has delivered, is working on or has decided on the file. It reads the database only: it does not look at the cache folder. Takes `--config`. |
| `subs retry --path X` | Makes the `unavailable` or `unclear` subtitle of file `X` (or of any part of a multi-part film) `wanted` again, forgetting which candidates were tried and resetting the per-file download count, so the next `subs resolve` starts afresh. It says how many of the remembered candidates are still kept in the cache and are re-checked without a download; it does not look anything up itself. Any other status, or a path with no surveyed file or no subtitle record, is refused (exit code 1, nothing changed); no state database or a configuration error is exit code 2. Takes `--config`. |
| `report [--output FILE]` | Writes the results so far to `report.csv` beside the state database (or `FILE`) and prints a summary, including how many subtitles were added and, for every manifest entry, whether its original(s) are still unchanged (run it before overwriting originals). `report.csv` and `manifest.csv` have a `subtitle` column: the track title of the subtitle muxed in (`English (OpenSubtitles)`, `English (local file)` or `English (sidecar)`), empty when none was added. |

**Current state:** every command works, including the subtitle commands (`auth`,
`subs resolve | status | retry`), the subtitle columns in the CSV files and the subtitle step of
`run`; see [Subtitles](#subtitles). `run` searches a CRF per file (3 sample
windows of 30 s encoded at CRF 20 and 24 and scored with VMAF, interpolated to the approved VMAF
target and clamped to `crf_min`..the approved `crf_max`), leaves the file alone when the projected
saving is under `min_saving`, encodes, muxes, scores 4 windows of 20 s against the original (the
quality guard) and verifies the output. `calibrate --samples` reads the sample files from the
library directly and read-only.

`survey`, `plan`, `calibrate`, `run` and `subs resolve` take `--config PATH` (default
`%LOCALAPPDATA%\video-cleaner\config.toml`), `-v`/`--verbose` (repeat for more console logging) and
`-q`/`--quiet` (warnings only) and `--log-file PATH` (this run's log goes there instead of
`log_dir`); `doctor`, `status`, `report`, `auth`, `subs status` and `subs retry` take `--config`
only.

`survey`, `calibrate`, `run` and `subs resolve` also take `--progress auto|rich|plain`. The default, `auto`, shows
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
encode, probe or VMAF error, an unusable scratch folder); 130 interrupted with Ctrl+C. `subs
resolve` and `subs retry` have their own meanings for 1 and 2; they are listed in the table above.

`doctor` checks: Python 3.11 or 3.12; ffmpeg found and built with `libx265` and `libvmaf`
(`hevc_nvenc` is informational); ffprobe and mkvmerge found; `ffsubsync` runs
(`python -m ffsubsync.ffsubsync --help` with this Python, started the way the aligner is, from a folder of its own; required, and a failure says to run
`uv sync` or `uv pip install ffsubsync`); an NVIDIA GPU visible
(informational); free space on the scratch drive at least `scratch_cap_bytes`; `library_root`
readable; the state database, log folder, subtitle cache, `completed_dir` and `scratch_dir` neither
inside the library nor containing it ("library overlap"); `completed_dir` and `scratch_dir` writable
(a missing one is reported as "does not exist (will be created)"; `doctor` creates nothing), and the
same for `subtitle_cache_dir`, which is informational because only `subs resolve` needs it; the
config file not readable by other users (informational; it looks for the well-known English
principal names `Everyone`, `BUILTIN\Users` and `NT AUTHORITY\Authenticated Users` in the `icacls`
output, so it misses a localized Windows or an ACE for another account, and it is no proof of
privacy; `setup.ps1` restricts the file with `icacls` regardless); the `keyring` backend in use
(informational): the line names the backend class, and when the library found no usable one (its
"fail" or "null" backend, a `keyrings.alt` one or one named as plain text, a chain containing
one, or none) it is a `[WARN]` saying that credentials are then kept in the owner-only
`credentials.json`, or, without `LOCALAPPDATA`, can only come from the `VC_OPENSUBTITLES_*`
variables (this check only asks which backend is in use; the credentials check below looks the
values up to see whether they are set and never prints one);
the OpenSubtitles credentials set (a warning, not a failure, when they are missing); and, when the
fallback `credentials.json` exists, that other users cannot read it (informational; the same
check as for the config file, with the same limits; the file is created owner-only regardless). Tools are found through `VC_FFMPEG` / `VC_FFPROBE` / `VC_MKVMERGE`, then `PATH`,
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
   The acknowledgement covers the plan as it was reviewed: after any re-survey, after a run
   found a changed (stale) original, or after a subtitle was found or went stale, run `plan`,
   review it and `plan --ack` again. The approved
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

## Subtitles

A file that has no English subtitle gets one: the tool finds a subtitle, checks it, and `run` adds
it to the MKV as a new track (original tracks never change). The work is split in two so that the
slow, quota-limited part does not hold up the encodes: `survey` decides which files need a
subtitle, `subs resolve` finds them (a few each day), and `run` muxes what has been found. See the
[daily routine](#daily-routine) for the order of the commands.

### Where subtitles come from

**In the file.** `survey` reads the file's own subtitle tracks. English is a track tagged `eng`,
`en` or `english`. A file that has one that is not marked forced needs nothing (`not_needed`), SDH
and hearing-impaired tracks included. A forced-only English track (only the signs and foreign
dialogue) does not count, unless it is a text track with at least `forced_full_cues_per_hour` cues
per hour of video: then it is a full track that was marked forced by mistake (`reclassified` in
`survey.csv`). A forced image track cannot be counted and stays forced-only. English only as
closed captions inside the video stream, or a subtitle track tagged `und` (undetermined language)
or with no language tag at all (when there is no English track that counts), makes the file
`unclear` (see below).

**Lookup order.** For a `wanted` file `subs resolve` tries these in turn and takes the first
candidate that passes the checks below:

1. **A sidecar file** next to the video: the same name (any case) plus an optional `.en` or `.eng`
   and one of `.srt`, `.ass`, `.vtt` (`Movie (2010).mkv` finds `Movie (2010).srt` and
   `Movie (2010).en.srt`). At most three are looked at, regular files only. It is only read.
2. **Your local folders** (`local_subtitle_dirs`, below).
3. **OpenSubtitles**, with the stored credentials: first a search by the video's movie hash (a hit
   says the subtitle was made for this very file), and only when that gives nothing usable a
   search by title and year. A result is skipped without a download when it is marked foreign
   parts only, was tried before, or (unless it is a hash match) does not match the file's title
   and year. A file whose name
   reads as a TV episode is searched by hash only. A multi-part film is not looked up at all (see
   `unclear`).

Every candidate, wherever it comes from, goes through the same checks: it must be plain-text
subtitles (SRT, WebVTT or ASS/SSA, recognised by content), at most `subtitle_max_bytes` and at
least `subtitle_min_cues` cues, and its text must look English. Then `ffsubsync` aligns it to the
video's first audio track (a hash-exact match is not aligned), and the aligned result is checked
again: still English, at least 95% of the cues kept, and not longer than the video allows (10% plus
two minutes). A doubtful candidate is rejected: a wrong subtitle is worse than none. What is kept
is always the tool's own plain SRT version of the accepted subtitle, never the file as received.

**Local folders.** Put subtitles you already have in one or more folders and list them:

```toml
local_subtitle_dirs = ['D:\subs', 'E:\more\subs']
```

`subs resolve` indexes them each time and prints how many files it found. They are searched to
8 folder levels (at most 100,000 entries), and links and junctions found while listing are skipped
(and a file that turns into a link before it is read is refused). A `.srt`, `.ass`, `.ssa` or
`.vtt` file is listed by its name. Downloaded subtitles often lose their extension to a release
tag (`Title (2010).en.264-GRP`), so a file with any other name is opened and listed if its content
is a subtitle in one of those formats. Never listed that way: unfinished downloads and backups
(ending `.part`, `.partial`, `.crdownload`, `.download`, `.!ut`, `.tmp`, `.bak`, `.old`, `.orig`,
`.disabled` or `~`), copies (`Title.srt.1`, `Title.srt.txt`, `Title.srt (1)`), pictures, archives,
`.nfo` and `.idx` files. If more than 2,000 files would have to be opened, none of them is used (a
warning says so), so point `local_subtitle_dirs` at folders that hold subtitles.

The subtitle is matched by name, and a library file's name and a subtitle's name are read the same
way. With the year in brackets, the title is everything before it, as written
(`Title - Second Part (2010)`; a leading site or picture-size tag such as `[1080p]` or
`www.site.com -` is dropped). An edition written before the year (`Title - Director's Cut (2010)`)
is therefore part of the title, and that subtitle does not match the plain `Title (2010)`.
Any other name is read as a release name (`Title.2010.1080p.BluRay.en`). For a library file, the
folder's name also fills in a missing year when its title is the same, or gives the whole title and
year of a generic name such as `movie.mkv`. A subtitle matches when its year is the same and its
title is near-identical, when the years differ by one and the title is closer still, or, when
either side has no year, when the title is near-exact. A match counts only if it is clearly the
best one; two similar candidates are ambiguous and both ignored (so are two copies of the same
film's subtitle in different folders). Ignored as well: names that read as TV episodes or give no
title, names tagged with a language other than English (`fr`, `spa`, `pt-BR`, `en.fr`; a `hi` tag
means hearing impaired, not Hindi), empty files, files above `subtitle_max_bytes`, links and anything that is not a regular file. A name
with no language tag is kept; its text is checked for English like any other.

### Statuses

Each video has one subtitle status (`subs status` counts them; `plan.csv` and `survey.csv` show
them):

| Status | Meaning | What you do |
|---|---|---|
| `wanted` | No English subtitle in the file; the lookup has not finished. Also the state of a file whose lookup stopped early: no credentials, login refused, allowance spent, OpenSubtitles failing, cache folder not writable. Its reason says which. | Run `subs resolve`; `auth` if credentials are missing. A queued `wanted` file is deferred by `run` unless you pass `--include-unresolved`; a file delivered that way is never looked up afterwards (`subs resolve` skips it and `subs status` counts it as already delivered, `plan --retry` refuses a completed file), so it only gets a subtitle after you have replaced the original with the result and surveyed again. |
| `found` | A subtitle was accepted and is in the cache. The file is planned as a remux (or an encode) that adds it. | Review the plan, `plan --ack`, `run`. |
| `not_needed` | The file has a full English track. Nothing is added. | Nothing. |
| `unavailable` | The lookup finished and nothing passed. The reason says what each source did, and a file that used up `subtitle_candidates_per_file` downloads says so. The file is not retried by itself and a re-survey keeps the status. | Nothing, if you can do without: the file is encoded or left alone as planned, with no subtitle. To try again put a subtitle beside the video or in a local folder, or loosen an alignment setting (`align_max_offset_s`, `align_scale_tolerance`, or `align_min_score` if you raised it), then `subs retry --path` and `subs resolve`. |
| `unclear` | The tool cannot tell whether the file needs one: a subtitle track tagged `und` or with no language tag, English only as closed captions, or a multi-part film. | See below. |

A track tagged **`und`**, or with no language tag at all, flags the file (unless the file has an
English track that counts): the survey cannot know whether the track is English, so it
puts the file in the plan as `flag` (never touched) instead of searching, and the survey may flag
more files than you expect. To see them, open `survey.csv` and filter `english_subtitle` or
`subtitle_status` on `unclear`; the `reasons` column says `subtitle track with no language tag
(stream N)` or `English only as closed captions (...)`, and `plan.csv` shows them with the action
`flag`. Look at the track in a player. If it is English, retag it yourself (the tool never changes
the library), then `survey` again: the file becomes `not_needed`. If you want an English track
anyway, run `subs retry --path` and `subs resolve` (before the next `survey`: a survey classifies
the tracks again and sets the file back to `unclear`). When a subtitle is found, the new English
track is added beside the `und` track and the file leaves the flagged state; if nothing is found
it stays flagged.

A film made of **several parts** that the plan joins into one file gets `unclear` from the lookup
with a reason that starts `multi-part film`: finding a subtitle for the joined film would need the
parts combined, which is not supported yet, so no lookup (and no download) is made. Its plan is
unchanged and retrying it gives the same answer.

### Quota

OpenSubtitles limits downloads: 20 a day on a free account at the time of writing (the tool uses
whatever allowance the service reports). A download counts against it when the request reaches the service, whether or
not the subtitle is then accepted; a download that fails in a way that may have reached the
service is counted too.
`subs resolve` checks the stored allowance before every file and stops as soon as it is used up
(`Stopped: quota exhausted until YYYY-MM-DD HH:MM UTC`); the files not reached stay `wanted` and the next day's
`subs resolve` resumes with them. `subs status` shows the downloads left and when they renew. The
allowance renews daily, at 00:00 UTC unless the service reports another time (the time stored is
never taken to be more than an hour past the next midnight UTC).

Besides the daily allowance each file has its own cap: `subtitle_candidates_per_file` (default 3)
downloads in total, counted across days. A file that reaches it becomes `unavailable` with the reason
"per-file download limit (N) reached"; `subs retry` resets the count and forgets which candidates
were tried. A download rejected at or after the alignment (its score, offset or frame-rate scale,
too little speech, too few cues kept, or its length) is kept on disk as a `.dl.srt` file, so after
changing an alignment setting and `subs retry` it is checked again without a new download; the
retry message says how many are kept. A download rejected for its content (not English, not a subtitle) is not kept.

A refused login (wrong credentials, key revoked) turns OpenSubtitles off for the rest of that
`subs resolve` only: sidecar and local files are still used, the other files stay `wanted` with
the reason "OpenSubtitles login was refused this run", and the next `subs resolve` tries again.
When OpenSubtitles fails for three files in a row (the service not answering) the job ends with
`Stopped: OpenSubtitles is not answering` and exit code 1.

### What gets added to the MKV

When a file's subtitle is `found`, `run` adds exactly one new track to its MKV: language `eng`,
title `English (OpenSubtitles)`, `English (local file)` or `English (sidecar)`, with the default
and forced flags off, so players keep choosing audio and subtitles as before. Every original audio and
subtitle track (including an `und` one), the chapters and the attachments stay as they were. The
output is verified to hold exactly that one new track more.

- A file that is also re-encoded gets the subtitle in the same step. If its encode is not worth
  keeping (the projected saving is below `min_saving`, the quality guard fails, or the output is
  too large), a subtitle-only remux of the original is delivered instead.
- A file that needs no re-encode (already HEVC or AV1, 4K, small, low bitrate) becomes a remux whose only
  change is the subtitle. It is delivered even though the new file is slightly larger.
- An HDR, Dolby Vision, interlaced or multi-video-stream file, which a plan would otherwise flag,
  is remuxed with its video copied untouched when a subtitle was found.
- Before the video is copied, `run` checks that the cached subtitle is still there and unchanged,
  and copies it to the file's scratch folder for mkvmerge. If it is gone or changed, the file goes
  to retry, its subtitle back to `wanted`, and the plan changes (see [The cache](#the-cache)).

### The cache

Subtitles are cached in `subtitle_cache_dir` (default `%LOCALAPPDATA%\video-cleaner\subtitles`),
a folder that should be on a local drive (not enforced) and must not be inside the library,
contain it, or overlap a video. File names are
`<key>-<source>-<id>.srt`: `<key>` is 16 hex digits made from the video's path, size and
modification time, `<source>` is `sidecar`, `local` or `opensubtitles`, and `<id>` is 12 hex
digits taken from the subtitle's text (sidecar and local) or the OpenSubtitles file number. Other
files you may see there:

| Name | What it is |
|---|---|
| `*.dl.srt` | A copy of a candidate while it is aligned (any source), or an OpenSubtitles download kept after a rejection at or after the alignment (see [Quota](#quota)), so a retry can check it again for free. The accepted candidate's copy is removed; other kept downloads of the video stay (at most `subtitle_candidates_per_file` are kept, the oldest removed when another is added) until you delete them. |
| `*.aligned.srt` | The aligner's output; removed after use. |
| `.align-*` (folders) | `ffsubsync` work folders; removed after each alignment, and any a killed run left are swept at the start of `subs resolve`. Only folders that hold this tool's marker file (`.video-cleaner-align`) are swept, so an `.align-*` folder of another tool in the same place is left alone. |

The tool never deletes a finished `.srt` itself, not even when the video it belongs to has been
replaced or removed; clean those up yourself when you no longer need them. It is safe to delete
them: the next `subs resolve` (at its start) or `run` (when it reaches the file, before copying
anything) notices that the cached file of a `found` subtitle is missing or changed, and puts the
file back to `wanted`, remembering what was tried and how many downloads it used. A sidecar or
local subtitle is then read again for free; an OpenSubtitles one has to be downloaded again, which
costs a download from the day's allowance and counts toward the file's cap: if the file has
already used `subtitle_candidates_per_file` downloads, the lookup cannot download it again and the
file becomes `unavailable` (per-file download limit); run `subs retry --path` and `subs resolve`
then. The plan changes, so `plan` and `plan --ack` again. `survey`, `plan` and `subs status` do not look in the cache folder, so they report `found`
until one of those two commands has noticed.

### Security and privacy

- The OpenSubtitles API key, username and password (and the login token the service returns) are
  kept in the operating system's keyring (the Windows Credential Manager). Where the system has no
  usable keyring (or only a `keyrings.alt` backend, one named as plain text, or a chain
  containing one) they go to
  `%LOCALAPPDATA%\video-cleaner\credentials.json`, restricted to your user. They are never written
  to the state database, the logs (every credential the tool loads is masked in log lines and in
  tool output; the messages the tool prints itself contain none by construction), the plan and
  report files, or a command line: `auth` reads them with hidden input. Every program the tool
  starts (ffmpeg, ffprobe, mkvmerge, ffsubsync, and the `--help` check of ffsubsync that `doctor`
  and `subs resolve` make) is started without any `VC_OPENSUBTITLES_*` variable or any variable
  whose name contains `API_KEY`, `PASSWORD`, `TOKEN` or `SECRET`.
- Subtitles are untrusted input. Their size is checked before anything is decoded, only plain-text
  formats are read (nothing is evaluated), formatting is stripped and the result is written as the
  tool's own SRT, and no name or text from a server or a subtitle ever becomes part of a path
  or a command line (cache names are built from a hash and a number). File names, tool output and
  server text are shown on the console and in the logs with control, bidi and zero-width
  characters escaped.
- The tool talks only to the domain in `opensubtitles_domain` (default `opensubtitles.com`) and
  its subdomains, over HTTPS, with a size cap on every reply. Each network operation has a
  timeout and a reply body has an overall limit of 120 s; name resolution (DNS) has no timeout of
  its own, and the connection and header phase is bounded only per operation. A login reply, a
  download link or a redirect that names any other domain is refused, so your credentials cannot
  be sent to another site. The system proxy (the Windows Internet settings, or `HTTPS_PROXY`) is
  used when one is set: the connection stays encrypted end to end, but a TLS-inspecting proxy
  whose certificate your machine trusts would see the credentials.
- Nothing is written to the library: sidecar and local subtitles are only read, and the cache
  folder is refused when it overlaps the library.

### Known limits

- TV is out of scope. A file whose name reads as an episode is never matched to a local subtitle
  and is searched on OpenSubtitles by hash only.
- A multi-part film gets no lookup (`unclear`).
- There is no speech-to-text fallback (Whisper) yet: a file whose status ends `unavailable` stays
  without a subtitle.
- English only: the language checks, the matching of embedded tracks and the alignment assume
  English (see `subtitle_language`).
- The alignment listens to the video's first audio track.
- `align_min_score` (default `0`, so only a score of zero or less is refused) is a starting value.
  Tune it against your own library: the run log (at its default `DEBUG` level) has the offset,
  frame-rate scale and score of every alignment and the reason for every rejection.
  `forced_full_cues_per_hour` (default `150`) comes from measurements of one library: forced
  tracks of signs and foreign dialogue stayed under about 100 cues per hour, a silent film's
  intertitles reached about 185, and spoken dialogue 300 or more.
- A hash-exact OpenSubtitles match is not aligned, since the service says it was made for this
  file; the length check still applies to it.
- A download that was counted but whose link fetch still fails after its retries is not tried
  again later: it costs one download and the candidate is remembered as tried.
- ffprobe, mkvmerge and the verification have no time limit, so a crafted file that makes one of
  them loop would hold `survey` or `run` until you stop it (Ctrl+C).
- Very long single-line output from a tool (for example an embedded subtitle track that is one
  huge cue) is held in memory before the output cap applies.
- With the `credentials.json` fallback, running `auth` while `subs resolve` is running can lose
  an update (the two read and write the whole file without a lock); run them one after the other.

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
| `min_file_size` | `0` | Files smaller than this many bytes are not re-encoded (they can still be remuxed: as part of a multi-part group, or to add a found subtitle). |
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
| `subtitle_cache_dir` | `%LOCALAPPDATA%\video-cleaner\subtitles` | Local folder for the subtitles found for your files (see [The cache](#the-cache)). Refused (like every other output) when it is inside `library_root` or contains it. |
| `local_subtitle_dirs` | `[]` | Folders searched for subtitles you already have (see [Where subtitles come from](#where-subtitles-come-from)): a TOML array of absolute paths (`['D:\subs', 'E:\more']`), or in `VC_LOCAL_SUBTITLE_DIRS` several paths separated by `;` on Windows. |
| `forced_full_cues_per_hour` | `150.0` | Cues per hour of video at or above which an English text track marked forced counts as a full subtitle rather than forced-only (more than 0). Tune it if your library has forced tracks with long foreign-language stretches. |
| `align_max_offset_s` | `120.0` | Largest shift in seconds the alignment may apply to a subtitle; an alignment that lands on this limit (within 0.05 s) is rejected as a cut-off search. |
| `align_scale_tolerance` | `0.05` | Largest frame-rate change the alignment may apply (the scale factor may differ from 1 by this fraction, `0` to below `1`). |
| `align_min_score` | `0.0` | Lowest alignment score accepted (not negative). `0` sets no minimum, but a score of zero or less is always refused. A starting value; raise it after reviewing the scores in the first run logs. |
| `subtitle_max_bytes` | `2000000` | A subtitle larger than this (a download, a sidecar, a local file, the aligner's output) is rejected; larger local files are not indexed. |
| `subtitle_min_cues` | `20` | A subtitle with fewer cues is rejected (`0` or more). |
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

## Third-party licences

video-cleaner itself is MIT. It runs on these Python packages, with the licences their package
metadata declares (read from the packages installed on Windows with Python 3.11, at the versions `uv.lock` pins for it; this is a
summary, not legal advice):

| Package | Licence | Note |
|---|---|---|
| `typer`, `rich`, `keyring`, `ffsubsync` | MIT | Direct dependencies. |
| `guessit` | LGPL-3.0-or-later | Direct dependency, **copyleft**. Used unmodified as a library (it reads titles and years out of file names); `uv sync` installs it from PyPI, nothing is copied into this repository. |
| `auditok` | GPL-3.0 | **Copyleft.** Installed because `ffsubsync` depends on it, but `ffsubsync` imports it only when run with an auditok voice detector; the tool always starts it with `--vad webrtc`, so it is never loaded. |
| `faust-cchardet` | MPL-1.1 OR GPL-2.0-or-later OR LGPL-2.1-or-later | **Copyleft** (a choice of licences). Installed by `ffsubsync`, unmodified. |
| `tqdm` | MPL-2.0 AND MIT | **Weak copyleft** (file-level). Installed by `ffsubsync`, unmodified. |
| `numpy` | BSD-3-Clause | Installed by `ffsubsync`. The Windows wheel bundles OpenBLAS and LAPACK (BSD) and the GCC runtime library (GPL-3.0-or-later with the GCC runtime library exception, which permits this use). |

The other packages in `uv.lock` declare MIT, BSD, Apache-2.0, ISC, PSF-2.0 or 0BSD licences. Those
that only other platforms install (the Linux keyring support: `secretstorage`, `jeepney`,
`cryptography`, `cffi`, `pycparser`) were not checked.

ffmpeg, ffprobe and MKVToolNix are separate programs that the tool starts as processes. It does
not bundle, link or modify them (`setup.ps1` installs them with winget); the licence of the build
you install applies to them.
