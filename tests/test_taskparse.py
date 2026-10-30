"""Zone ranges in the task box (cleanproof/ui/taskparse.py): all the sentences listed in the known-issues list + exclusions, ranges and phrasings that are not recognised."""

import pytest

from cleanproof.ui import operator as op
from cleanproof.ui.taskparse import parse_task

ALL = [f"{r}{c}" for r in "ABC" for c in range(1, 6)]


def zs(text):
    return parse_task(text, 3, 5)


def test_exclusions_become_all_but():
    for t, ex in [("Do not touch B3, clean the rest", ["B3"]), ("clean everything except C2", ["C2"]), ("avoid B3", ["B3"]),
                  ("Don't touch B3", ["B3"]), ("don’t clean B3", ["B3"]), ("Clean the table without B3", ["B3"]), ("skip B3", ["B3"]),
                  ("clean all zones other than B3 and C2", ["B3", "C2"]), ("Clean all except for A1", ["A1"]),
                  ("do not wipe B3 or B4", ["B3", "B4"])]:
        r = zs(t)
        assert r["mode"] == "except", t
        assert sorted(r["excluded"]) == sorted(ex), t
        assert r["zones"] == [z for z in ALL if z not in ex], t
        assert r["label"].startswith("All zones except"), t
        assert not r["confirm"], t


def test_ranges():
    assert zs("Clean zones A1-A3")["zones"] == ["A1", "A2", "A3"]
    assert zs("Clean A1 to A3")["zones"] == ["A1", "A2", "A3"]
    assert zs("Clean A1–A3")["zones"] == ["A1", "A2", "A3"]      # en dash
    assert zs("Clean A1 — A3")["zones"] == ["A1", "A2", "A3"]    # em dash
    assert zs("clean A1-3")["zones"] == ["A1", "A2", "A3"]
    assert zs("clean B2 to C3")["zones"] == ["B2", "B3", "C2", "C3"]
    assert zs("Clean A1-A5 but not B3")["zones"] == ["A1", "A2", "A3", "A4", "A5"]
    r = zs("Clean A1 to C3 except B2")
    assert r["mode"] == "except" and "B2" not in r["zones"] and "A1" in r["zones"] and len(r["zones"]) == 8


def test_only_and_whole_table_keep_old_behaviour():
    assert zs("Wipe B2 and C3 only")["zones"] == ["B2", "C3"]
    assert zs("clean b2, b2 and zone C 5")["zones"] == ["B2", "C5"]
    assert zs("Clean the table. Then D1 and A9.")["mode"] == "all"       # not in this site's grid
    assert zs("Wipe a 2 cm spot near the mug")["mode"] == "all"          # the article a + a number, not zone A2
    assert zs("")["mode"] == "all" and zs(None)["mode"] == "all"
    assert zs("clean the dining table")["notes"] == []
    assert zs("clean B3 without leaving streaks")["zones"] == ["B3"]     # without after a zone: B3 is not excluded


def test_unrecognised_phrases_fall_back_to_whole_table_and_say_so():
    r = zs("Clean the left half")
    assert r["mode"] == "all" and r["zones"] == [] and r["notes"] and not r["confirm"]
    assert "left half" in r["notes"][0]
    for t in ("Don't touch the glass", "avoid the left side", "do not wipe row B"):
        r = zs(t)
        assert r["mode"] == "all" and r["confirm"] and r["notes"], t     # a negation without a recognised zone: confirmation is required before the job starts


def test_everything_excluded_is_flagged():
    r = parse_task("everything except A1", 1, 1)
    assert r["mode"] == "except" and r["zones"] == [] and r["notes"]


def test_zones_in_task_wrapper():
    assert op.zones_in_task("Wipe B2 and C3 only", 3, 5) == ["B2", "C3"]
    assert op.zones_in_task("A1-A3 are wet; skip E1", 3, 5) == ["A1", "A2", "A3"]
    assert op.zones_in_task("avoid B3", 3, 5) == [z for z in ALL if z != "B3"]
    assert op.zones_in_task("", 3, 5) == []
