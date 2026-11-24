# OpenSubtitles fixtures

These files are **synthesised from the public OpenSubtitles REST API (v1) documentation and from
the response shapes third-party clients publish. They were not recorded from a live account.**
Every key, token, id, file name and link in them is an obviously fake placeholder; none is, or
was ever, real data.

The shapes (field names, nesting, types) were checked against the official OpenAPI export
(<https://stoplight.io/api/v1/projects/opensubtitles/opensubtitles-api/nodes/open_api.json>) and the
Best Practices page
(<https://stoplight.io/api/v1/projects/opensubtitles/opensubtitles-api/nodes/docs/02-Best-Practices.md>);
the values are synthetic.

The tests replay them through a fake transport (no network).

At the local gate (the run that uses a real account) a recorded login / search / download exchange
replaces these files: scrub the key, user name, password, token, user ids and file names, keep the
field names and nesting exactly as the server sent them, and re-run `tests/subtitles/`. Any field
the real server sends that these files lack should be added here, with a test.

| File | What it stands for |
| --- | --- |
| `login_ok.json` | `POST /login` 200 |
| `search_hash.json` | `GET /subtitles` 200: hash match, other language, foreign-parts-only, hearing impaired, two-file and empty-file entries, null fields |
| `download_ok.json` | `POST /download` 200 |
| `download_quota_406.json` | `POST /download` 406, daily allowance used up |
| `error_401.json` | any request 401 |
