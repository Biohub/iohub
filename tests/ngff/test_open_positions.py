import json
from pathlib import Path

import numpy as np
import pytest
import zarr

from iohub.ngff import NGFFNode, Plate, Position, open_ome_zarr, open_positions
from iohub.ngff.nodes import _store_dirpath

# Created in this order so that iohub's lexicographic plate order (A/1, A/10, A/2)
# differs from the natural order the positions must come in.
POSITIONS = [("B", "1", "0"), ("A", "10", "0"), ("A", "2", "0"), ("A", "1", "0")]
NATURAL = ["A/1/0", "A/2/0", "A/10/0", "B/1/0"]


def _plate(path: Path, version: str = "0.5") -> Path:
    with open_ome_zarr(path, mode="w", layout="hcs", channel_names=["c"], version=version) as plate:
        for key in POSITIONS:
            plate.create_position(*key).create_zeros("0", shape=(1, 1, 1, 2, 2), dtype=np.uint8)
    return path


def _fov(path: Path) -> Path:
    with open_ome_zarr(path, mode="w", layout="fov", channel_names=["c"]) as fov:
        fov.create_zeros("0", shape=(1, 1, 1, 2, 2), dtype=np.uint8)
    return path


def _paths(positions: list[Position]) -> list[Path]:
    return [_store_dirpath(position) for position in positions]


def _record_closes(monkeypatch) -> list[NGFFNode]:
    """The nodes closed from now on in the test."""
    calls: list[NGFFNode] = []
    close = NGFFNode.close

    def record(node):
        calls.append(node)
        close(node)

    monkeypatch.setattr(NGFFNode, "close", record)
    return calls


@pytest.mark.parametrize("version", ["0.4", "0.5"])
def test_plate_path_gives_positions_in_natural_order(tmp_path, monkeypatch, version):
    plate = _plate(tmp_path / "plate.zarr", version)
    closed = _record_closes(monkeypatch)
    with open_positions(plate) as positions:
        assert [p.zgroup.path for p in positions] == NATURAL
        assert all(type(p) is Position and p.zgroup.read_only for p in positions)
        assert _paths(positions) == [plate.resolve() / name for name in NATURAL]
        assert not closed
    assert len(closed) == 1
    assert isinstance(closed[0], Plate)


def test_glob_skips_files_and_sorts_naturally(tmp_path, monkeypatch):
    plate = _plate(tmp_path / "plate.zarr")
    closed = _record_closes(monkeypatch)
    # The pattern also matches each well's zarr.json.
    assert (plate / "A" / "1" / "zarr.json").is_file()
    with open_positions(str(plate / "*" / "*" / "*")) as positions:
        assert _paths(positions) == [plate.resolve() / name for name in NATURAL]
    assert len(closed) == len(NATURAL)


def test_shell_expanded_glob_with_files(tmp_path):
    """What an unquoted glob passes: positions and files, as separate items."""
    plate = _plate(tmp_path / "plate.zarr")
    items = [plate / "A" / "1" / "0", plate / "A" / "1" / "zarr.json", plate / "B" / "1" / "0"]
    with open_positions(items) as positions:
        assert _paths(positions) == [plate.resolve() / "A/1/0", plate.resolve() / "B/1/0"]


def test_items_keep_their_order_and_duplicates_are_dropped(tmp_path):
    plate = _plate(tmp_path / "plate.zarr")
    items = [plate / "B" / "1" / "0", str(plate), str(plate / "A" / "*" / "0"), plate / "A" / "2" / "0"]
    with open_positions(items) as positions:
        assert [p.zgroup.path for p in positions] == ["", "A/1/0", "A/2/0", "A/10/0"]
        assert _paths(positions) == [plate.resolve() / name for name in ["B/1/0", *NATURAL[:3]]]


def test_single_fov_store_and_mixed_layouts(tmp_path):
    fov = _fov(tmp_path / "fov.zarr")
    plate = _plate(tmp_path / "plate.zarr", version="0.4")
    with open_positions([fov, plate]) as positions:
        assert _paths(positions) == [fov.resolve()] + [plate.resolve() / name for name in NATURAL]
        assert [p.version for p in positions] == ["0.5"] + ["0.4"] * len(NATURAL)


def test_path_with_glob_characters_opens_as_named(tmp_path):
    fov = _fov(tmp_path / "fov[1].zarr")
    with open_positions(fov) as positions:
        assert _paths(positions) == [fov.resolve()]


def test_open_nodes_are_given_back_and_left_open(tmp_path, monkeypatch):
    plate_path = _plate(tmp_path / "plate.zarr")
    fov_path = _fov(tmp_path / "fov.zarr")
    closed = _record_closes(monkeypatch)
    with open_ome_zarr(plate_path) as plate, open_ome_zarr(fov_path) as fov:
        with open_positions([fov, plate, plate_path / "A" / "1" / "0"]) as positions:
            assert positions[0] is fov
            assert [p.zgroup.path for p in positions[1:]] == NATURAL
            assert all(p.zgroup.store is plate.zgroup.store for p in positions[1:])
        # The duplicate opened from the path is closed; the given nodes are not.
        assert len(closed) == 1
        assert closed[0] is not plate
        assert closed[0] is not fov


def test_nodes_are_closed_when_the_block_raises(tmp_path, monkeypatch):
    plate = _plate(tmp_path / "plate.zarr")
    closed = _record_closes(monkeypatch)
    with pytest.raises(RuntimeError), open_positions(plate):
        raise RuntimeError
    assert len(closed) == 1


def test_read_write_mode(tmp_path):
    plate = _plate(tmp_path / "plate.zarr")
    with open_positions(plate, mode="r+") as positions:
        assert not any(p.zgroup.read_only for p in positions)
        positions[0].zattrs["note"] = "written"
    with open_ome_zarr(plate / "A" / "1" / "0") as position:
        assert position.zattrs["note"] == "written"


def test_read_write_mode_refuses_a_read_only_node(tmp_path):
    fov_path = _fov(tmp_path / "fov.zarr")
    with open_ome_zarr(fov_path) as fov, pytest.raises(ValueError, match=r"read-only; open it with mode='r\+'"):
        with open_positions(fov, mode="r+"):
            pass


@pytest.mark.parametrize("mode", ["a", "w", "w-"])
def test_creating_modes_are_refused(tmp_path, mode):
    plate = _plate(tmp_path / "plate.zarr")
    with pytest.raises(ValueError, match="mode must be 'r' or 'r\\+'"), open_positions(plate, mode=mode):
        pass
    with open_positions(plate) as positions:
        assert len(positions) == len(NATURAL)


def test_pattern_matching_nothing(tmp_path, monkeypatch):
    plate = _plate(tmp_path / "plate.zarr")
    closed = _record_closes(monkeypatch)
    missing = str(tmp_path / "typo.zarr" / "*")
    with pytest.raises(FileNotFoundError, match=r"typo\.zarr"), open_positions([plate, missing]):
        pass
    # The plate opened before the failing pattern is closed.
    assert len(closed) == 1


@pytest.mark.parametrize("name", ["plain", "group.zarr", "plate.zarr/A", "plate.zarr/A/1", "plate.zarr/A/1/0/0"])
def test_directory_that_is_not_a_position_or_plate(tmp_path, name):
    _plate(tmp_path / "plate.zarr")
    (tmp_path / "plain").mkdir()
    zarr.open_group(tmp_path / "group.zarr", mode="w")
    with pytest.raises(ValueError, match="is not an OME-Zarr position or plate"), open_positions(tmp_path / name):
        pass


def test_glob_matching_a_directory_that_is_not_a_position(tmp_path):
    plate = _plate(tmp_path / "plate.zarr")
    (plate / "A" / "1" / "notes").mkdir()
    with pytest.raises(ValueError, match="notes is not an OME-Zarr"), open_positions(str(plate / "*" / "*" / "*")):
        pass


def test_bioformats2raw_collection_is_refused(tmp_path):
    root = tmp_path / "series.ome.zarr"
    root.mkdir()
    attributes = {"ome": {"version": "0.5", "bioformats2raw.layout": 3}}
    (root / "zarr.json").write_text(json.dumps({"zarr_format": 3, "node_type": "group", "attributes": attributes}))
    _fov(root / "0")
    with pytest.raises(ValueError, match="bioformats2raw collection"), open_positions(root):
        pass
    with open_positions(root / "0") as positions:
        assert _paths(positions) == [(root / "0").resolve()]


def test_no_positions(tmp_path):
    plate = _plate(tmp_path / "plate.zarr")
    for items in [[], plate / "zarr.json"]:
        with pytest.raises(ValueError, match="No OME-Zarr positions"), open_positions(items):
            pass
    with open_ome_zarr(tmp_path / "empty.zarr", mode="w", layout="hcs", channel_names=["c"]) as empty:
        with pytest.raises(ValueError, match=r"No OME-Zarr positions in \['Plate'\]"), open_positions(empty):
            pass


def test_unsupported_item(tmp_path):
    plate_path = _plate(tmp_path / "plate.zarr")
    with open_ome_zarr(plate_path) as plate:
        well = plate["A/1"]
        with pytest.raises(TypeError, match="got Well"), open_positions([well]):
            pass
