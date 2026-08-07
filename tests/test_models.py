import torch

from navcim_m4.models import VGG11Cifar10, create_model, load_checkpoint, sample_input


def test_vgg11_cifar10_output_shape():
    with torch.inference_mode():
        output = create_model()(sample_input())
    assert tuple(output.shape) == (1, 10)


def test_checkpoint_round_trip(tmp_path):
    model = VGG11Cifar10()
    path = tmp_path / "model.pt"
    torch.save({
        "model_name": "VGG11_CIFAR10",
        "num_classes": 10,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": {},
        "epoch": 1,
        "seed": 117,
        "training_accuracy": 1.0,
        "validation_accuracy": 1.0,
        "preprocessing": {},
    }, path)
    restored, metadata = load_checkpoint(path)
    assert metadata["epoch"] == 1
    assert tuple(restored(sample_input()).shape) == (1, 10)
