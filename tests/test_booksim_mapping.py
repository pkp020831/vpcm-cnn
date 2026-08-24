from pathlib import Path

from navcim_m4.booksim_mapping import FloorplanLayer, write_booksim_records
from navcim_m4.layers import LayerRecord


def test_booksim_records_use_neurosim_tile_locations(tmp_path: Path):
    records = (
        LayerRecord("conv0", 32, 32, 3, 3, 3, 8, 0, 1, 0, 1, 1),
        LayerRecord("conv1", 32, 32, 8, 3, 3, 16, 0, 1, 0, 1, 1),
    )
    floorplan = (
        FloorplanLayer(0, 2, 0, 2, 2),
        FloorplanLayer(1, 2, 2, 2, 2),
    )

    cluster, element = write_booksim_records(floorplan, records, tmp_path, 128, 1e3, 1e9)

    rows = cluster.read_text(encoding="ascii").splitlines()
    assert rows[0].endswith("tile_rows,tile_cols")
    assert len(rows) == 5
    assert ",2,,conv0,0-0,MAC," in rows[1]
    assert ",2,,conv0,0-1,MAC," in rows[2]
    assert ",,conv1,1-0,MAC," in rows[3]
    assert ",,conv1,1-1,MAC," in rows[4]
    assert element.read_text(encoding="ascii").strip() == "node,used,activation_size,injection_rate"
