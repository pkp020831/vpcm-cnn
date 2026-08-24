import csv

from navcim_m4.meta.contracts import file_reference, write_manifest
from navcim_m4.meta.models import create_split_stage


def test_meta_split_keeps_hardware_configs_disjoint(tmp_path):
    dataset = tmp_path / "dataset.csv"
    rows = [
        {
            "config_key": f"config-{config}",
            "predicted_booksim_latency_ns": 10 + layer,
            "neurosim_latency_ns": 20 + layer,
            "actual_overall_latency_ns": 30 + 2 * layer,
            "predicted_booksim_dynamic_power_w": 0.1,
            "neurosim_dynamic_energy_pj": 100,
            "actual_overall_dynamic_energy_pj": 200,
        }
        for config in range(4)
        for layer in range(3)
    ]
    with dataset.open("w", newline="", encoding="ascii") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    manifest_path = tmp_path / "dataset.json"
    write_manifest(
        manifest_path,
        "dataset",
        {"outputs": {"dataset": file_reference(dataset)}, "metrics": {"configs": 4, "rows": 12}},
        0.0,
    )

    split = create_split_stage(manifest_path, tmp_path / "split.json", seed=117, test_ratio=0.25)

    assert set(split["split"]["train"]).isdisjoint(split["split"]["test"])
    assert split["metrics"]["train_rows"] == 9
    assert split["metrics"]["test_rows"] == 3
