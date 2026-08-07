import torch

from navcim_m4.models import create_model, sample_input


def test_vgg11_cifar10_output_shape():
    with torch.inference_mode():
        output = create_model()(sample_input())
    assert tuple(output.shape) == (1, 10)
