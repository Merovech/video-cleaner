from __future__ import annotations

import csv
import dataclasses
import errno
import os
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from video_cleaner.config import Settings
from video_cleaner.manifest import (
    MANIFEST_COLUMNS,
    ManifestEntry,
    ManifestError,
    check_unchanged,
    completed_path_for,
    read_manifest,
    write_manifest,
)

windows_only = pytest.mark.skipif(os.name != "nt", reason="Windows path semantics")

MTIME = 1_700_000_000.123456


def make_settings(library_root: Path, completed_dir: Path) -> Settings:
    return Settings(
        library_root=library_root,
        scratch_dir=completed_dir.parent / "scratch",
        completed_dir=completed_dir,
        state_db=completed_dir.parent / "state.db",
        log_dir=completed_dir.parent / "logs",
    )


def make_entry(
    originals: tuple[Path, ...],
    completed: Path,
    *,
    sizes: tuple[int, ...] | None = None,
    mtimes: tuple[float, ...] | None = None,
    completed_size: int = 700,
    vmaf_mean: float | None = 95.31,
    vmaf_worst: float | None = 90.5,
    crf: int | None = 21,
    verified: bool = True,
    action: str = "reencode",
) -> ManifestEntry:
    return ManifestEntry(
        original=originals,
        original_sizes=sizes
        if sizes is not None
        else tuple(1000 + i for i in range(len(originals))),
        original_mtimes=mtimes
        if mtimes is not None
        else tuple(MTIME + i for i in range(len(originals))),
        completed=completed,
        completed_size=completed_size,
        vmaf_mean=vmaf_mean,
        vmaf_worst=vmaf_worst,
        crf=crf,
        verified=verified,
        action=action,
    )


def make_file(path: Path, data: bytes = b"x" * 10, mtime: float = MTIME) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    os.utime(path, (mtime, mtime))
    return path


def entry_for_files(files: tuple[Path, ...], completed: Path) -> ManifestEntry:
    stats = [f.stat() for f in files]
    return make_entry(
        files,
        completed,
        sizes=tuple(s.st_size for s in stats),
        mtimes=tuple(s.st_mtime for s in stats),
    )


# --- completed_path_for -------------------------------------------------------------------


def test_completed_path_mirrors_relative_folder(tmp_path: Path) -> None:
    root = tmp_path / "library"
    completed = tmp_path / "completed"
    settings = make_settings(root, completed)
    original = root / "movies" / "Heat (1995)" / "Heat (1995).mkv"

    assert completed_path_for(settings, original) == (
        completed / "movies" / "Heat (1995)" / "Heat (1995).mkv"
    )


def test_extension_becomes_mkv_for_avi_input(tmp_path: Path) -> None:
    settings = make_settings(tmp_path / "library", tmp_path / "completed")
    original = tmp_path / "library" / "old" / "Clip.avi"

    assert completed_path_for(settings, original) == tmp_path / "completed" / "old" / "Clip.mkv"


def test_multipart_uses_combined_name(tmp_path: Path) -> None:
    settings = make_settings(tmp_path / "library", tmp_path / "completed")
    original = tmp_path / "library" / "movies" / "Epic (2001)" / "Epic (2001) - cd1.avi"

    result = completed_path_for(settings, original, new_name="Epic (2001).mkv")

    assert result == tmp_path / "completed" / "movies" / "Epic (2001)" / "Epic (2001).mkv"


def test_completed_path_for_file_directly_in_root(tmp_path: Path) -> None:
    settings = make_settings(tmp_path / "library", tmp_path / "completed")

    assert completed_path_for(settings, tmp_path / "library" / "a.mp4") == (
        tmp_path / "completed" / "a.mkv"
    )


def test_completed_path_rejects_original_outside_library_root(tmp_path: Path) -> None:
    settings = make_settings(tmp_path / "library", tmp_path / "completed")

    with pytest.raises(ValueError, match="not under"):
        completed_path_for(settings, tmp_path / "elsewhere" / "a.mkv")


def test_completed_path_rejects_sibling_with_shared_prefix(tmp_path: Path) -> None:
    # "library2" starts with "library" as a string but is not inside it.
    settings = make_settings(tmp_path / "library", tmp_path / "completed")

    with pytest.raises(ValueError, match="not under"):
        completed_path_for(settings, tmp_path / "library2" / "a.mkv")


def test_completed_path_rejects_the_root_itself(tmp_path: Path) -> None:
    settings = make_settings(tmp_path / "library", tmp_path / "completed")

    with pytest.raises(ValueError):
        completed_path_for(settings, tmp_path / "library")


def test_completed_path_rejects_parent_traversal(tmp_path: Path) -> None:
    settings = make_settings(tmp_path / "library", tmp_path / "completed")
    sneaky = tmp_path / "library" / "movies" / ".." / ".." / "outside" / "a.mkv"

    with pytest.raises(ValueError, match=r"\.\."):
        completed_path_for(settings, sneaky)


@pytest.mark.parametrize(
    "bad_name", ["", ".", "..", "sub/a.mkv", "sub\\a.mkv", "..\\a.mkv", "C:a.mkv"]
)
def test_completed_path_rejects_new_name_that_is_not_a_plain_file_name(
    tmp_path: Path, bad_name: str
) -> None:
    settings = make_settings(tmp_path / "library", tmp_path / "completed")

    with pytest.raises(ValueError, match="new_name"):
        completed_path_for(settings, tmp_path / "library" / "m" / "a.avi", new_name=bad_name)


@windows_only
def test_completed_path_compares_root_case_insensitively_and_keeps_casing(
    tmp_path: Path,
) -> None:
    settings = make_settings(tmp_path / "Library", tmp_path / "completed")
    original = Path(str(tmp_path / "Library").upper()) / "Movies" / "HeAt (1995).MKV"

    result = completed_path_for(settings, original)

    assert str(result) == str(tmp_path / "completed" / "Movies" / "HeAt (1995).mkv")


@windows_only
def test_completed_path_for_unc_root() -> None:
    settings = make_settings(Path(r"\\nas\media\movies"), Path(r"D:\completed"))
    original = Path(r"\\NAS\Media\movies\Heat (1995)\Heat (1995).mp4")

    result = completed_path_for(settings, original)

    assert str(result) == r"D:\completed\Heat (1995)\Heat (1995).mkv"


@windows_only
def test_completed_path_rejects_other_share_on_same_server() -> None:
    settings = make_settings(Path(r"\\nas\media\movies"), Path(r"D:\completed"))

    with pytest.raises(ValueError, match="not under"):
        completed_path_for(settings, Path(r"\\nas\other\movies\a.mkv"))


@windows_only
def test_completed_path_for_unc_path_longer_than_260_chars() -> None:
    settings = make_settings(Path(r"\\nas\media\movies"), Path(r"D:\completed"))
    folders = [f"Folder {i:02d} " + "x" * 40 for i in range(8)]
    original = Path("\\\\nas\\media\\movies", *folders, "Long (2020).avi")
    assert len(str(original)) > 260

    result = completed_path_for(settings, original)

    assert result == Path("D:\\completed", *folders, "Long (2020).mkv")


@pytest.mark.parametrize(
    ("folder", "name", "expected_name"),
    [
        # Review Focus 1: the name from the real library, which ends in a bracket.
        (
            "The First Omen (2024)",
            "The First Omen (2024).en.MX].mkv",
            "The First Omen (2024).en.MX].mkv",
        ),
        # Without a video extension only the last dotted piece is treated as the extension.
        (
            "The First Omen (2024)",
            "The First Omen (2024).en.MX]",
            "The First Omen (2024).en.mkv",
        ),
        (
            "[Group] Show [1080p]",
            "[Group] Show - 01 [ABCD1234].avi",
            "[Group] Show - 01 [ABCD1234].mkv",
        ),
        (
            "Amélie (2001)",
            "Le Fabuleux Destin d'Amélie Poulain.mp4",
            "Le Fabuleux Destin d'Amélie Poulain.mkv",
        ),
        ("千と千尋の神隠し (2001)", "千と千尋の神隠し.mkv", "千と千尋の神隠し.mkv"),
        ("Emoji 🎬", "Film 🎬.avi", "Film 🎬.mkv"),
        (
            "Mr. Nobody (2009)",
            "Mr. Nobody (2009).Extended.Cut.avi",
            "Mr. Nobody (2009).Extended.Cut.mkv",
        ),
        ("Comma, Quote ' and ; (1999)", "a,b;c 'd'.avi", "a,b;c 'd'.mkv"),
        ("noext", "README", "README.mkv"),
        ("hidden", ".hidden", ".hidden.mkv"),
    ],
)
def test_completed_path_handles_odd_names(
    tmp_path: Path, folder: str, name: str, expected_name: str
) -> None:
    settings = make_settings(tmp_path / "library", tmp_path / "completed")

    result = completed_path_for(settings, tmp_path / "library" / folder / name)

    assert result == tmp_path / "completed" / folder / expected_name


def test_completed_path_keeps_trailing_dots() -> None:
    # Pure path arithmetic: a trailing dot must neither crash nor be silently dropped.
    settings = make_settings(Path("/lib"), Path("/done"))

    result = completed_path_for(settings, Path("/lib") / "Etc." / "Vol. 1..avi")

    assert result.parts[-2:] == ("Etc.", "Vol. 1..mkv")
    assert completed_path_for(settings, Path("/lib") / "Etc." / "Trailing.").name == "Trailing..mkv"


# --- ManifestEntry ------------------------------------------------------------------------


def test_entry_requires_parallel_tuples(tmp_path: Path) -> None:
    files = (tmp_path / "a.avi", tmp_path / "b.avi")
    with pytest.raises(ValueError, match="original_sizes"):
        make_entry(files, tmp_path / "c.mkv", sizes=(1,))
    with pytest.raises(ValueError, match="original_mtimes"):
        make_entry(files, tmp_path / "c.mkv", mtimes=(1.0, 2.0, 3.0))


def test_entry_requires_at_least_one_original(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least one"):
        make_entry((), tmp_path / "c.mkv")


def test_entry_is_frozen(tmp_path: Path) -> None:
    entry = make_entry((tmp_path / "a.avi",), tmp_path / "a.mkv")
    with pytest.raises(AttributeError):
        entry.verified = False  # type: ignore[misc]


# --- write_manifest / read_manifest -------------------------------------------------------


def test_manifest_lists_every_original_for_multipart(tmp_path: Path) -> None:
    lib = tmp_path / "library" / "Epic (2001)"
    completed = tmp_path / "completed" / "Epic (2001)" / "Epic (2001).mkv"
    entry = make_entry(
        (lib / "Epic (2001) - cd1.avi", lib / "Epic (2001) - cd2.avi"),
        completed,
        sizes=(111, 222),
        mtimes=(MTIME, MTIME + 60.25),
        action="merge+reencode",
    )
    manifest = tmp_path / "completed" / "manifest.csv"

    write_manifest(manifest, [entry])

    with manifest.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        assert reader.fieldnames == list(MANIFEST_COLUMNS)
        rows = list(reader)
    assert len(rows) == 2
    assert [r["original"] for r in rows] == [str(p) for p in entry.original]
    assert {r["completed"] for r in rows} == {str(completed)}
    assert [r["original_size"] for r in rows] == ["111", "222"]
    assert [float(r["original_mtime"]) for r in rows] == [MTIME, MTIME + 60.25]
    assert {r["completed_size"] for r in rows} == {"700"}
    assert {r["action"] for r in rows} == {"merge+reencode"}
    assert read_manifest(manifest) == [entry]


def test_manifest_columns_are_the_documented_ones() -> None:
    assert MANIFEST_COLUMNS == (
        "original",
        "original_size",
        "original_mtime",
        "completed",
        "completed_size",
        "vmaf_mean",
        "vmaf_worst",
        "crf",
        "verified",
        "action",
    )


def test_manifest_roundtrip_unicode_and_brackets(tmp_path: Path) -> None:
    lib = tmp_path / "library"
    done = tmp_path / "completed"
    entries = [
        make_entry(
            (lib / "The First Omen (2024)" / "The First Omen (2024).en.MX].mkv",),
            done / "The First Omen (2024)" / "The First Omen (2024).en.MX].mkv",
        ),
        make_entry(
            (lib / "千と千尋の神隠し (2001)" / "千と千尋の神隠し 🎬 [1080p].avi",),
            done / "千と千尋の神隠し (2001)" / "千と千尋の神隠し 🎬 [1080p].mkv",
            vmaf_mean=None,
            vmaf_worst=None,
            crf=None,
            verified=False,
            action="remux",
        ),
        make_entry(
            (
                lib / 'Quote "it", then; (1999)' / "Amélie, part 1 [x264].avi",
                lib / 'Quote "it", then; (1999)' / "Amélie, part 2 [x264].avi",
            ),
            done / 'Quote "it", then; (1999)' / "Amélie [x264].mkv",
            mtimes=(0.1 + 0.2, 1_699_999_999.9999998),
            action="merge",
        ),
    ]
    manifest = done / "manifest.csv"

    write_manifest(manifest, entries)

    raw = manifest.read_bytes()
    assert not raw.startswith(b"\xef\xbb\xbf")  # no BOM
    text = raw.decode("utf-8")  # strictly valid UTF-8
    assert "千と千尋の神隠し 🎬 [1080p].avi" in text
    with manifest.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 4
    assert rows[0]["original"] == str(entries[0].original[0])
    assert rows[1]["original"] == str(entries[1].original[0])
    assert rows[1]["vmaf_mean"] == rows[1]["vmaf_worst"] == rows[1]["crf"] == ""
    assert rows[1]["verified"] == "false"
    assert rows[2]["original"] == str(entries[2].original[0])
    assert rows[2]["verified"] == "true"
    assert all(None not in r and len(r) == len(MANIFEST_COLUMNS) for r in rows)

    assert read_manifest(manifest) == entries


def test_manifest_roundtrips_names_with_line_breaks_and_leading_spaces(tmp_path: Path) -> None:
    # Not legal on NTFS, but legal on the NAS file system; the CSV must still not mis-split.
    entry = ManifestEntry(
        original=(Path("/lib") / " lead \r\n break" / 'a "b".avi',),
        original_sizes=(5,),
        original_mtimes=(MTIME,),
        completed=Path("/done") / " lead \r\n break" / 'a "b".mkv',
        completed_size=3,
        vmaf_mean=95.0,
        vmaf_worst=91.0,
        crf=20,
        verified=True,
        action="reencode",
    )
    manifest = tmp_path / "manifest.csv"

    write_manifest(manifest, [entry])

    assert read_manifest(manifest) == [entry]


def test_manifest_neutralises_formulas_and_still_roundtrips(tmp_path: Path) -> None:
    # Relative paths never reach the manifest in a run; they make every cell start with a
    # formula trigger (or an apostrophe) so the escaping is exercised both ways.
    entry = ManifestEntry(
        original=(Path("=HYPERLINK(1).avi"), Path("'quoted.avi")),
        original_sizes=(5, 6),
        original_mtimes=(MTIME, MTIME),
        completed=Path("@done.mkv"),
        completed_size=3,
        vmaf_mean=95.0,
        vmaf_worst=91.0,
        crf=20,
        verified=True,
        action="-reencode",
    )
    manifest = tmp_path / "manifest.csv"

    write_manifest(manifest, [entry])

    with manifest.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert [r["original"] for r in rows] == ["'=HYPERLINK(1).avi", "''quoted.avi"]
    assert rows[0]["completed"] == "'@done.mkv"
    assert rows[0]["action"] == "'-reencode"
    assert read_manifest(manifest) == [entry]


def test_manifest_preserves_exact_mtime_floats(tmp_path: Path) -> None:
    mtimes = (1_700_000_000.1234567, 0.1 + 0.2, 1e-7, 1_731_000_000.0)
    files = tuple(tmp_path / f"p{i}.avi" for i in range(len(mtimes)))
    entry = make_entry(files, tmp_path / "p.mkv", mtimes=mtimes)

    write_manifest(tmp_path / "manifest.csv", [entry])

    assert read_manifest(tmp_path / "manifest.csv")[0].original_mtimes == mtimes


def test_manifest_roundtrip_keeps_check_unchanged_true(tmp_path: Path) -> None:
    a = make_file(tmp_path / "lib" / "a.avi", mtime=1_700_000_000.1234567)
    b = make_file(tmp_path / "lib" / "b.avi", b"yy", mtime=1_700_000_100.7654321)
    entry = entry_for_files((a, b), tmp_path / "done" / "ab.mkv")
    manifest = tmp_path / "done" / "manifest.csv"

    write_manifest(manifest, [entry])
    (reread,) = read_manifest(manifest)

    assert check_unchanged(reread) is True


def test_write_manifest_empty_writes_header_only(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"

    write_manifest(manifest, [])

    assert manifest.read_text(encoding="utf-8").splitlines() == [",".join(MANIFEST_COLUMNS)]
    assert read_manifest(manifest) == []


def test_write_manifest_creates_parent_and_replaces_existing(tmp_path: Path) -> None:
    manifest = tmp_path / "completed" / "manifest.csv"
    first = make_entry((tmp_path / "a.avi",), tmp_path / "completed" / "a.mkv")
    second = make_entry((tmp_path / "b.avi",), tmp_path / "completed" / "b.mkv")

    write_manifest(manifest, [first])
    write_manifest(manifest, [first, second])

    assert read_manifest(manifest) == [first, second]
    assert [p.name for p in manifest.parent.iterdir()] == ["manifest.csv"]  # no temp left


def test_write_manifest_failure_keeps_previous_manifest_and_leaves_no_temp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = tmp_path / "manifest.csv"
    first = make_entry((tmp_path / "a.avi",), tmp_path / "a.mkv")
    write_manifest(manifest, [first])
    before = manifest.read_bytes()

    def boom(src: object, dst: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr("video_cleaner.manifest.os.replace", boom)
    with pytest.raises(OSError, match="disk full"):
        write_manifest(manifest, [make_entry((tmp_path / "b.avi",), tmp_path / "b.mkv")])

    assert manifest.read_bytes() == before
    assert [p.name for p in tmp_path.iterdir()] == ["manifest.csv"]


def test_write_manifest_rejects_two_entries_for_one_completed_path(tmp_path: Path) -> None:
    done = tmp_path / "x.mkv"
    entries = [make_entry((tmp_path / "a.avi",), done), make_entry((tmp_path / "b.avi",), done)]

    with pytest.raises(ManifestError, match="more than one entry"):
        write_manifest(tmp_path / "manifest.csv", entries)

    assert not (tmp_path / "manifest.csv").exists()


def test_read_manifest_regroups_rows_that_are_not_adjacent(tmp_path: Path) -> None:
    multi = make_entry((tmp_path / "a1.avi", tmp_path / "a2.avi"), tmp_path / "a.mkv")
    single = make_entry((tmp_path / "b.avi",), tmp_path / "b.mkv")
    manifest = tmp_path / "manifest.csv"
    write_manifest(manifest, [multi, single])
    with manifest.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.reader(fh))
    reordered = [rows[0], rows[1], rows[3], rows[2]]
    with manifest.open("w", encoding="utf-8", newline="") as fh:
        csv.writer(fh).writerows(reordered)

    assert read_manifest(manifest) == [multi, single]


def test_read_manifest_missing_file_raises_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_manifest(tmp_path / "nope.csv")


@pytest.mark.parametrize(
    ("column", "value", "match"),
    [
        ("original_size", "big", "original_size"),
        ("original_mtime", "yesterday", "original_mtime"),
        ("completed_size", "", "completed_size"),
        ("crf", "21.5", "crf"),
        ("vmaf_mean", "high", "vmaf_mean"),
        ("verified", "maybe", "verified"),
        ("original", "", "original"),
        ("completed", "", "completed"),
    ],
)
def test_read_manifest_rejects_bad_values_with_line_number(
    tmp_path: Path, column: str, value: str, match: str
) -> None:
    manifest = tmp_path / "manifest.csv"
    write_manifest(manifest, [make_entry((tmp_path / "a.avi",), tmp_path / "a.mkv")])
    with manifest.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    rows[0][column] = value
    with manifest.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(MANIFEST_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ManifestError, match=match) as excinfo:
        read_manifest(manifest)
    assert "line 2" in str(excinfo.value)


def test_read_manifest_rejects_missing_column(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("original,completed\na,b\n", encoding="utf-8")

    with pytest.raises(ManifestError, match="missing column"):
        read_manifest(manifest)


def test_read_manifest_rejects_short_row(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(",".join(MANIFEST_COLUMNS) + "\na.avi,10\n", encoding="utf-8")

    with pytest.raises(ManifestError, match="line 2"):
        read_manifest(manifest)


def test_read_manifest_rejects_parts_that_disagree_about_the_completed_file(
    tmp_path: Path,
) -> None:
    multi = make_entry((tmp_path / "a1.avi", tmp_path / "a2.avi"), tmp_path / "a.mkv")
    manifest = tmp_path / "manifest.csv"
    write_manifest(manifest, [multi])
    with manifest.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    rows[1]["completed_size"] = "701"
    with manifest.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(MANIFEST_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ManifestError, match="disagree"):
        read_manifest(manifest)


# --- check_unchanged ----------------------------------------------------------------------


def test_check_unchanged_true_for_untouched_originals(tmp_path: Path) -> None:
    a = make_file(tmp_path / "a.avi")
    b = make_file(tmp_path / "b [x].avi", b"yyyy", mtime=MTIME + 5)

    assert check_unchanged(entry_for_files((a, b), tmp_path / "ab.mkv")) is True


def test_check_unchanged_false_after_touch(tmp_path: Path) -> None:
    a = make_file(tmp_path / "a.avi")
    entry = entry_for_files((a,), tmp_path / "a.mkv")
    assert check_unchanged(entry) is True

    os.utime(a, (MTIME + 1, MTIME + 1))  # same bytes, newer mtime

    assert check_unchanged(entry) is False


def test_check_unchanged_false_for_sub_second_mtime_change(tmp_path: Path) -> None:
    a = make_file(tmp_path / "a.avi", mtime=1_700_000_000.0)
    entry = entry_for_files((a,), tmp_path / "a.mkv")

    os.utime(a, (1_700_000_000.001, 1_700_000_000.001))

    assert check_unchanged(entry) is False


def test_check_unchanged_false_when_size_differs_but_mtime_restored(tmp_path: Path) -> None:
    a = make_file(tmp_path / "a.avi")
    entry = entry_for_files((a,), tmp_path / "a.mkv")

    a.write_bytes(b"x" * 11)
    os.utime(a, (entry.original_mtimes[0], entry.original_mtimes[0]))

    assert check_unchanged(entry) is False


def test_check_unchanged_false_when_only_the_last_part_changed(tmp_path: Path) -> None:
    files = tuple(make_file(tmp_path / f"cd{i}.avi") for i in (1, 2, 3))
    entry = entry_for_files(files, tmp_path / "all.mkv")

    os.utime(files[2], (MTIME + 9, MTIME + 9))

    assert check_unchanged(entry) is False


def test_check_unchanged_false_when_an_original_vanished(tmp_path: Path) -> None:
    a = make_file(tmp_path / "a.avi")
    b = make_file(tmp_path / "b.avi")
    entry = entry_for_files((a, b), tmp_path / "ab.mkv")

    b.unlink()

    assert check_unchanged(entry) is False


def test_check_unchanged_false_when_parent_folder_vanished(tmp_path: Path) -> None:
    entry = make_entry((tmp_path / "gone" / "deeper" / "a.avi",), tmp_path / "a.mkv")

    assert check_unchanged(entry) is False


def test_check_unchanged_false_when_original_became_a_directory(tmp_path: Path) -> None:
    a = make_file(tmp_path / "a.avi")
    entry = entry_for_files((a,), tmp_path / "a.mkv")
    a.unlink()
    a.mkdir()
    st = a.stat()

    assert (
        check_unchanged(
            replace(entry, original_sizes=(st.st_size,), original_mtimes=(st.st_mtime,))
        )
        is False
    )


def test_check_unchanged_false_for_unusable_path(tmp_path: Path) -> None:
    # An embedded NUL makes os.stat raise ValueError rather than OSError.
    entry = make_entry((Path(str(tmp_path / "a") + "\x00.avi"),), tmp_path / "a.mkv")

    assert check_unchanged(entry) is False


@pytest.mark.parametrize(
    "error",
    [
        PermissionError(errno.EACCES, "access denied"),
        TimeoutError("share timed out"),
        OSError(errno.EIO, "I/O error"),
    ],
)
def test_check_unchanged_propagates_errors_that_are_not_a_missing_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error: OSError
) -> None:
    # "Could not check" (NAS drop, permissions) must never look like "original changed".
    a = make_file(tmp_path / "a.avi")
    entry = entry_for_files((a,), tmp_path / "a.mkv")

    def fail(*args: object, **kwargs: object) -> os.stat_result:
        raise error

    monkeypatch.setattr("video_cleaner.manifest.os.stat", fail)

    with pytest.raises(type(error)):
        check_unchanged(entry)


@pytest.mark.parametrize(
    "error",
    [
        FileNotFoundError(errno.ENOENT, "gone"),
        NotADirectoryError(errno.ENOTDIR, "parent is a file"),
    ],
)
def test_check_unchanged_false_for_missing_file_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error: OSError
) -> None:
    a = make_file(tmp_path / "a.avi")
    entry = entry_for_files((a,), tmp_path / "a.mkv")

    def fail(*args: object, **kwargs: object) -> os.stat_result:
        raise error

    monkeypatch.setattr("video_cleaner.manifest.os.stat", fail)

    assert check_unchanged(entry) is False


def test_check_unchanged_false_when_parent_is_a_file(tmp_path: Path) -> None:
    parent = make_file(tmp_path / "not_a_dir")
    entry = make_entry((parent / "a.avi",), tmp_path / "a.mkv")

    assert check_unchanged(entry) is False


# --- malformed input ----------------------------------------------------------------------


def test_write_manifest_rejects_path_not_representable_as_utf8(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    good = make_entry((tmp_path / "a.avi",), tmp_path / "a.mkv")
    write_manifest(manifest, [good])
    bad = make_entry((tmp_path / "lone \udc80 surrogate.avi",), tmp_path / "b.mkv")

    with pytest.raises(ManifestError, match="UTF-8"):
        write_manifest(manifest, [good, bad])

    assert read_manifest(manifest) == [good]  # previous manifest untouched
    assert [p.name for p in tmp_path.iterdir()] == ["manifest.csv"]


def test_read_manifest_rejects_non_finite_mtime(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        ",".join(MANIFEST_COLUMNS) + "\na.avi,10,nan,a.mkv,5,,,,true,remux\n", encoding="utf-8"
    )

    with pytest.raises(ManifestError, match="original_mtime"):
        read_manifest(manifest)


def test_read_manifest_rejects_broken_quoting(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        ",".join(MANIFEST_COLUMNS) + '\n"a.avi,10,1.0,a.mkv,5,,,,true,remux\n', encoding="utf-8"
    )

    with pytest.raises(ManifestError, match="line"):
        read_manifest(manifest)


def test_read_manifest_rejects_non_utf8_file(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    manifest.write_bytes(
        ",".join(MANIFEST_COLUMNS).encode() + b"\nAm\xe9lie.avi,10,1.0,a.mkv,5,,,,true,remux\n"
    )

    with pytest.raises(ManifestError, match="UTF-8"):
        read_manifest(manifest)


def test_write_manifest_flushes_to_disk(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    synced: list[int] = []
    real = os.fsync

    def spy(fd: int) -> None:
        synced.append(fd)
        real(fd)

    monkeypatch.setattr(os, "fsync", spy)
    write_manifest(
        tmp_path / "manifest.csv", [make_entry((tmp_path / "a.avi",), tmp_path / "a.mkv")]
    )
    assert len(synced) == 1


@pytest.mark.parametrize("field", ["original_mtimes", "vmaf_mean", "vmaf_worst"])
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_entry_rejects_non_finite_numbers(tmp_path: Path, field: str, bad: float) -> None:
    entry = make_entry((tmp_path / "a.avi",), tmp_path / "a.mkv")
    value: Any = (bad,) if field == "original_mtimes" else bad
    with pytest.raises(ValueError, match=field):
        dataclasses.replace(entry, **{field: value})
