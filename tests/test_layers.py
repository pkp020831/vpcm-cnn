from navcim_m4.layers import create_layer_artifacts
from navcim_m4.models import create_model


def test_cifar_vgg11_has_expected_weight_layers(tmp_path):
    artifacts = create_layer_artifacts(create_model(), tmp_path)
    assert len(artifacts.records) == 9
    assert sum(record.is_fc == 0 for record in artifacts.records) == 8
    assert sum(record.is_fc == 1 for record in artifacts.records) == 1
    assert artifacts.records[0].csv_row() == (32, 32, 3, 3, 3, 64, 1, 1, 0, 1, 1)
    assert artifacts.records[-1].csv_row() == (1, 1, 512, 1, 1, 10, 0, 1, 1, 0, 0)
