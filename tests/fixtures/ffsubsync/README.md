# ffsubsync log fixtures

Real ffsubsync output, captured from **synthetic media** only: a 60 second clip whose audio is a
generated voice-like buzz (`ffmpeg -f lavfi aevalsrc=...`) in seven invented speech windows, and
a synthetic SRT of invented lines whose cues lie three seconds after the sound. No real film,
subtitle or library file was used.

- ffsubsync 0.5.1, Python 3.11 on Windows, ffmpeg 9.0.2, webrtcvad voice detection
- command: `python -W ignore::RuntimeWarning:runpy -m ffsubsync.ffsubsync --srtin=in.srt
  --srtout=<out> --encoding utf-8 --max-offset-seconds 10.0 --reference-stream a:0
  --vad webrtc -- <media>`, run from a scratch folder with relative names
- everything was written to **stderr** (rich-formatted logging plus a tqdm progress bar); stdout
  was empty, because the result goes to the `--srtout` file
- the Windows output has CRLF line ends; the fixtures were normalised to LF, progress-bar redraws
  were reduced to the last one and local paths were replaced (`C:\venv`). The parser is also
  tested with CRLF.

| File | What it is |
| --- | --- |
| `success.txt` | a clean run: `score: 5256.000`, `offset seconds: -3.000`, `framerate scale factor: 1.000`; exit code 0 |
| `silent_audio.txt` | media with a silent audio track ("total of speech segments: 0.0"): ffsubsync still reports numbers (`score: 324.167`, `offset seconds: 4.780`, `framerate scale factor: 1.043`), all meaningless; exit code 0 |
| `no_audio_stream.txt` | media without an audio stream: a Python traceback ending in `ValueError: Unable to detect speech`, no numbers; exit code 1 |
| `failed_to_sync.txt` | the numbers were reported, then writing the output failed (missing folder): `ERROR failed to sync` with a traceback; **exit code 0** |

Nothing here was edited by hand apart from the normalisation above.
