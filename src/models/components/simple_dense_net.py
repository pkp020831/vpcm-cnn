import torch
from torch import nn


class SimpleDenseNet(nn.Module):
    """A simple fully-connected neural net for computing predictions."""

    def __init__(
        self,
        input_size: int = 784,
        hidden_sizes: list[int] = [256, 256, 256],
        output_size: int = 10,
        initialization: str = "default",
        act_fn: str = "ReLU", # Added activation function parameter
    ) -> None:
        """Initialize a `SimpleDenseNet` module."""
        super().__init__()

        # Map string to activation function class
        act_map = {
            "ReLU": nn.ReLU,
            "Tanh": nn.Tanh,
            "Identity": nn.Identity, # Added for linear activation
        }
        if act_fn not in act_map:
            raise ValueError(f"Unsupported activation function: {act_fn}")
        activation_function = act_map[act_fn]

        layers = []
        current_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(current_size, hidden_size))
            layers.append(activation_function())
            current_size = hidden_size
        
        layers.append(nn.Linear(current_size, output_size))

        self.model = nn.Sequential(*layers)

        if initialization == "orthogonal":
            for m in self.model.modules():
                if isinstance(m, nn.Linear):
                    nn.init.orthogonal_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor, return_all_activities: bool = False) -> torch.Tensor:
        """Perform a single forward pass through the network.

        :param x: The input tensor.
        :return: A tensor of predictions.
        """
        batch_size, channels, width, height = x.size()

        # (batch, 1, width, height) -> (batch, 1*width*height)
        x = x.view(batch_size, -1)

        activities = [x] # Store input activity
        current_output = x
        for layer in self.model:
            current_output = layer(current_output)
            activities.append(current_output)
        
        if return_all_activities:
            return activities
        else:
            return activities[-1] # Return only the final output by default


if __name__ == "__main__":
    _ = SimpleDenseNet()