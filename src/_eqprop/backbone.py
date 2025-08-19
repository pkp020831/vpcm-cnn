from copy import deepcopy
from typing import Literal
import math
import torch

import torch.nn as nn

from src.core import eqprop
from src.core.eqprop import nn as enn
from src.utils import eqprop_utils


def mup_initialization(model, width, depth, variance=1.0):
    """
    Performs muP initialization on a model.
    Assumes model contains a sequence of nn.Linear layers.
    """
    std = math.sqrt(variance)
    
    # Extract linear layers from the model
    linear_layers = [m for m in model.modules() if isinstance(m, nn.Linear)]
    num_layers = len(linear_layers)

    if num_layers == 0:
        print("Warning: No linear layers found for muP initialization.")
        return

    for i, layer in enumerate(linear_layers):
        # Start with standard Gaussian initialization
        nn.init.normal_(layer.weight, mean=0.0, std=std)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)

        # Apply muP scaling
        with torch.no_grad():
            if i == 0:  # Input layer
                layer.weight.data *= math.sqrt(1 / 1568)
            elif i == num_layers - 1:  # Output layer
                layer.weight.data *= 1 / width
            else:  # Hidden layer
                layer.weight.data *= math.sqrt(1 / width / depth)


def goemup_initialization(model, width, depth, variance=1.0):
    """
    Performs GOE muP initialization on a model.
    Assumes model contains a sequence of nn.Linear layers.
    """
    
    linear_layers = [m for m in model.modules() if isinstance(m, nn.Linear)]
    num_layers = len(linear_layers)

    if num_layers == 0:
        print("Warning: No linear layers found for GOE muP initialization.")
        return

    for i, layer in enumerate(linear_layers):
        # GOE initialization
        with torch.no_grad():
            m, n = layer.weight.shape
            
            # 1. Create the base GOE matrix
            N = max(m, n)
            # Generate a matrix with elements from N(0, variance / 2)
            A = torch.randn(N, N, device=layer.weight.device, dtype=layer.weight.dtype) * math.sqrt(variance / 2.0)
            # Symmetrize
            X = torch.triu(A) + torch.triu(A, 1).T
            # Adjust diagonal variance to be N(0, variance)
            # Current diagonal variance is variance/2. We need to add another variance/2.
            diag_adjustment = torch.randn(N, device=layer.weight.device, dtype=layer.weight.dtype) * math.sqrt(variance / 2.0)
            X.diagonal().add_(diag_adjustment)

            # 2. Take a sub-block for the rectangular weight matrix
            layer.weight.data.copy_(X[:m, :n])

        if layer.bias is not None:
            nn.init.zeros_(layer.bias)

        # Apply muP scaling
        with torch.no_grad():
            if i == 0:  # Input layer
                layer.weight.data *= math.sqrt(1 / 1568)
            elif i == num_layers - 1:  # Output layer
                layer.weight.data *= 1 / width
            else:  # Hidden layer
                layer.weight.data *= math.sqrt(1 / width / depth)


def initialize_weights(model, initialization):
    """
    Initializes the weights of a model based on the provided configuration.

    Args:
        model (nn.Module): The model to initialize.
        initialization (dict | None): The initialization configuration.
    """
    if initialization is None:
        initialization = {"name": "default"}

    init_name = initialization.get("name", "default")

    if init_name == "default":
        pass
    elif init_name == "orthogonal":
        for m in model.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    elif init_name == "gaussian":
        init_variance = initialization.get("variance")
        if init_variance is None:
            raise ValueError("'variance' must be specified in initialization config for gaussian.")
        print(f"DEBUG: Initializing with gaussian, init_variance={init_variance}")
        for i, m in enumerate(model.modules()):
            if isinstance(m, nn.Linear):
                std = init_variance**0.5
                nn.init.normal_(m.weight, mean=0.0, std=std)
                print(f"  - Layer {i}: weight.std() = {m.weight.std():.4f} (target std: {std:.4f})")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    elif init_name == "mup":
        mup_width = initialization.get("width")
        mup_depth = initialization.get("depth")
        init_variance = initialization.get("variance", 1.0)

        if mup_width is None or mup_depth is None:
            raise ValueError("'width' and 'depth' must be specified in initialization config for muP.")
        
        mup_initialization(model, width=mup_width, depth=mup_depth, variance=init_variance)
    elif init_name == "goemup":
        mup_width = initialization.get("width")
        mup_depth = initialization.get("depth")
        init_variance = initialization.get("variance", 1.0)

        if mup_width is None or mup_depth is None:
            raise ValueError("'width' and 'depth' must be specified in initialization config for GOEmuP.")

        goemup_initialization(model, width=mup_width, depth=mup_depth, variance=init_variance)


class MultiplyActivation(nn.Module):
    """Multiply Activation layer."""

    def __init__(self, scale: float = 1.0):
        """Initialize MultiplyActivation layer.

        Args:
            scale (float, optional): _description_. Defaults to 1.0.
        """
        super().__init__()
        self.scale = scale

    def forward(self, x):
        return self.scale * x


class EqPropBackbone(nn.Module):
    """EqPropBackbone with Default EqPropLinear layers.

    Each EqPropLinear layer uses CenteredEqPropFunc as eqprop function
    and ProxQPStrategy as solver.
    """

    def __init__(
        self,
        cfg: list[int] = [784 * 2, 128, 10 * 2],
        beta: float = 0.1,
        bias: bool | list[bool] = [True, False],
        scale_input: int = 2,
        scale_output: int = 2,
        data_scale: float = 1.0,
        solver: eqprop.solvers.EqPropSolver | None = None,
        param_adjuster: eqprop_utils.AdjustParams | None = eqprop_utils.AdjustParams(),
        layer_scale: float = 4,
        initialization: dict | None = None,
        residual: bool = False,  # Add residual argument
        res_scale: float = 1.0,  # Add res_scale for scaling the residual
    ) -> None:
        """Initialize EqPropBackbone.

        Args:
            cfg (list[int], optional): Configuration of layers. Defaults to [784 * 2, 128, 10 * 2].
            bias (bool | list[bool], optional): Bias for each layer. Defaults to [True, False].
            scale_input (int, optional): Scale input. Defaults to 2.
            scale_output (int, optional): Scale output. Defaults to 2.
            data_scale (float, optional): Scale for input data. Defaults to 1.0.
            solver (Optional[EqPropSolver], optional): Solver for EqProp. Defaults to None.
            param_adjuster (Optional[eqprop_utils.AdjustParams], optional): Parameter adjuster for every forward call.
                Defaults to eqprop_utils.AdjustParams().
            layer_scale (float, optional): Scaling factor between eqprop layers. Defaults to 4.0.
            initialization (dict | None, optional): Weight initialization method. Defaults to None.
            residual (bool, optional): Whether to use identity shortcut connections. Defaults to False.
            res_scale (float, optional): Scaling factor for the residual connection. Defaults to 1.0.
        """
        super().__init__()
        self.residual = residual  # Store residual flag
        self.res_scale = res_scale
        self.data_scale = data_scale
        layers = self._make_layers(cfg, bias, solver, layer_scale)
        self.model = nn.Sequential(*layers)
        self.param_adjuster = param_adjuster
        eqprop_utils.interleave.set_num_input(scale_input)
        eqprop_utils.interleave.set_num_output(scale_output)

        initialize_weights(self.model, initialization)

    def _make_layers(self, cfg, bias, solver, layer_scale) -> list[nn.Module]:
        layers = []
        for idx in range(len(cfg) - 1):
            bias_idx = bias if isinstance(bias, bool) else bias[idx]
            solver_ = deepcopy(solver) if solver else None
            layers.append(
                enn.EqPropLinear(cfg[idx], cfg[idx + 1], bias=bias_idx, solver=solver_)
            )
            # layers.append(nn.Tanh())
            layers.append(MultiplyActivation(scale=layer_scale))
        return layers

    @eqprop_utils.interleave(type="both")
    def forward(self, x, return_all_activities: bool = False):
        if self.param_adjuster is not None:
            self.model.apply(self.param_adjuster)

        x = x * self.data_scale
        activities = [x] # Store input activity
        current_output = x
        
        # This variable will hold the input to the *current* EqPropLinear layer
        # It's used for the shortcut connection. It gets updated after each MultiplyActivation layer.
        input_for_shortcut = x 

        # Iterate through layers in self.model (which are [EqPropLinear, MultiplyActivation, ...])
        # The loop index 'i' refers to the index within self.model.
        for i, layer in enumerate(self.model):
            if isinstance(layer, enn.EqPropLinear):
                # This is an EqPropLinear layer
                linear_output = layer(current_output)
                
                # Apply shortcut if residual is True and this is a hidden EqPropLinear layer.
                # Hidden EqPropLinear layers are those not at index 0 (first)
                # AND not the last EqPropLinear (i = len(self.model) - 2).
                if self.residual and i > 0 and i < len(self.model) - 2:
                    # Check if dimensions match for shortcut connection
                    if linear_output.shape != input_for_shortcut.shape:
                        raise ValueError(
                            f"Dimension mismatch for residual connection at layer {i}: "
                            f"Linear output shape {linear_output.shape} vs shortcut input shape {input_for_shortcut.shape}"
                        )
                    current_output = self.res_scale * linear_output + input_for_shortcut
                else:
                    current_output = linear_output
                
            elif isinstance(layer, MultiplyActivation):
                # This is a MultiplyActivation layer, apply it to the current_output
                current_output = layer(current_output)
                # After MultiplyActivation, the output becomes the new input for the next potential shortcut
                input_for_shortcut = current_output
            
            activities.append(current_output)
        
        if return_all_activities:
            return activities
        else:
            return activities[-1] # Return only the final output by default


class EqPropSequentialBackbone(nn.Module):
    def __init__(
        self,
        cfg: list[int] = [784 * 2, 128, 10 * 2],
        beta: float = 0.1,
        bias: bool | list[bool] = [True, True],
        scale_input: int = 2,
        scale_output: int = 2,
        data_scale: float = 1.0,
        solver: eqprop.solvers.EqPropSolver | None = None,
        param_adjuster: eqprop_utils.AdjustParams | None = eqprop_utils.AdjustParams(),
        eqprop_fn: Literal["positive", "altered", "centered"] = "centered",
        initialization: dict | None = None,  # Changed back to 'initialization', type is dict
        residual: bool = False,
        res_scale: float = 1.0,
    ) -> None:
        """Initialize EqPropBackbone.

        Args:
            cfg (list[int], optional): Configuration of layers. Defaults to [784 * 2, 128, 10 * 2].
            bias (bool | list[bool], optional): Bias for each layer. Defaults to [True, False].
            scale_input (int, optional): Scale input. Defaults to 2.
            scale_output (int, optional): Scale output. Defaults to 2.
            data_scale (float, optional): Scale for input data. Defaults to 1.0.
            solver (Optional[EqPropSolver], optional): Solver for EqProp. Defaults to None.
            param_adjuster (Optional[eqprop_utils.AdjustParams], optional): Parameter adjuster for every forward call.
                Defaults to eqprop_utils.AdjustParams().
            eqprop_fn (str, optional): EqProp function type. Defaults to "centered".
            initialization (dict | None, optional): Initialization config dictionary. Defaults to None.
            residual (bool, optional): Whether to use identity shortcut connections. Defaults to False.
            res_scale (float, optional): Scaling factor for the residual connection. Defaults to 1.0.
        """
        super().__init__()
        self.data_scale = data_scale
        
        self.model = enn.EqPropSequential(
            *self._make_layers(cfg, bias),
            eqprop_fn=eqprop_fn,
            solver=solver,
        )
        self.param_adjuster = param_adjuster
        eqprop_utils.interleave.set_num_input(scale_input)
        eqprop_utils.interleave.set_num_output(scale_output)

        initialize_weights(self.model, initialization)

    @staticmethod
    def _make_layers(cfg, bias):
        layers = []
        for idx in range(len(cfg) - 1):
            bias_idx = bias if isinstance(bias, bool) else bias[idx]
            layers.append(
                enn.EqPropLinear(
                    cfg[idx],
                    cfg[idx + 1],
                    bias=bias_idx,
                )
            )
            # layers.append(nn.Tanh())
        return layers

    @eqprop_utils.interleave(type="both")
    def forward(self, x, return_all_activities: bool = False):
        if self.param_adjuster is not None:
            self.model.apply(self.param_adjuster)
        x = x * self.data_scale
        return self.model(x, return_all_activities=return_all_activities)


class HybridEqPropBackbone(EqPropBackbone):
    """Composite EqPropBackbone with Default EqPropLinear layers and Linear
    layers."""

    def _make_layers(self, cfg, bias, solver, layer_scale) -> list[nn.Module]:
        layers = []
        for idx in range(len(cfg) - 1):
            bias_idx = bias if isinstance(bias, bool) else bias[idx]
            solver_ = deepcopy(solver) if solver else None
            layers.extend(
                [
                    nn.Linear(cfg[idx], cfg[idx], bias=bias_idx),
                    nn.ReLU(),
                    enn.EqPropLinear(
                        cfg[idx], cfg[idx + 1], bias=bias_idx, solver=solver_
                    ),
                    MultiplyActivation(scale=layer_scale),
                ]
            )

        return layers


class GroupedHybridBackbone(EqPropBackbone):
    def __init__(
        self,
        cfg: list[int],  # 계층별 차원 리스트
        cfg_types: list[str],  # 계층별 블록 타입
        beta: float = 0.1,
        eqprop_fn_type_for_sequential: str = "centered",
        bias: bool | list[bool] = [True, True],
        # EqPropBackbone.__init__에 전달될 파라미터들
        scale_input: int = 2,
        scale_output: int = 2,
        solver: eqprop.solvers.EqPropSolver | None = None,
        param_adjuster: eqprop_utils.AdjustParams | None = eqprop_utils.AdjustParams(),
        layer_scale: float = 4,
    ) -> None:
        if len(cfg_types) != len(cfg) - 1:
            raise ValueError("cfg_types의 길이는 cfg 길이 - 1 이어야 합니다.")

        # `super().__init__`가 `self._make_layers`를 호출하므로,
        # `_make_layers`에서 사용할 인스턴스 변수들을 여기서 먼저 설정해야 합니다.
        self.cfg_types = cfg_types
        self.eqprop_fn_type_for_sequential = eqprop_fn_type_for_sequential

        # 부모 클래스(EqPropBackbone)의 __init__ 호출
        # 이 호출 과정에서 self._make_layers(cfg, bias_config, solver_config_template, layer_scale_config)가 실행됨
        super().__init__(
            cfg=cfg,
            bias=bias,  # bias_config를 'bias' 파라미터로 전달
            solver=solver,  # solver_config_template을 'solver' 파라미터로 전달
            layer_scale=layer_scale,  # layer_scale_config를 'layer_scale' 파라미터로 전달
            scale_input=scale_input,
            scale_output=scale_output,
            param_adjuster=param_adjuster,
        )

    # _make_layers 메소드를 재정의. EqPropBackbone.__init__에서 호출됨.
    # 시그니처는 EqPropBackbone.__init__이 _make_layers를 호출할 때 전달하는 인자들과 일치해야 함.
    def _make_layers(
        self,
        cfg,
        bias,
        solver,
        layer_scale,
    ) -> list[nn.Module]:
        overall_layers_list: list[nn.Module] = []

        cfg_type_idx = 0  # self.cfg_types 리스트를 순회하는 인덱스
        while cfg_type_idx < len(
            self.cfg_types
        ):  # self.cfg_types 사용 (생성자에서 설정됨)
            current_block_type = self.cfg_types[cfg_type_idx]
            block_start_cfg_type_idx = cfg_type_idx

            while (
                cfg_type_idx < len(self.cfg_types)
                and self.cfg_types[cfg_type_idx] == current_block_type
            ):
                cfg_type_idx += 1

            internal_block_sub_layers: list[nn.Module] = []
            for k_internal_layer_idx in range(block_start_cfg_type_idx, cfg_type_idx):
                layer_input_dim = cfg[k_internal_layer_idx]  # 전달받은 cfg_arg 사용
                layer_output_dim = cfg[k_internal_layer_idx + 1]

                current_layer_bias: bool
                if isinstance(bias, bool):  # 전달받은 bias_arg 사용
                    current_layer_bias = bias
                elif isinstance(bias, list) and len(bias) == len(self.cfg_types):
                    current_layer_bias = bias[k_internal_layer_idx]
                else:
                    current_layer_bias = True

                if current_block_type == "EP":
                    internal_block_sub_layers.append(
                        nn.Linear(
                            layer_input_dim, layer_output_dim, bias=current_layer_bias
                        )
                    )
                elif current_block_type == "BP":
                    internal_block_sub_layers.append(
                        nn.Linear(
                            layer_input_dim, layer_output_dim, bias=current_layer_bias
                        )
                    )
                    internal_block_sub_layers.append(nn.ReLU())

            if current_block_type == "EP":
                solver_instance_for_seq = solver

                ep_sequential_block = enn.EqPropSequential(
                    *internal_block_sub_layers,
                    eqprop_fn=self.eqprop_fn_type_for_sequential,  # self에서 가져옴
                    solver=solver,
                )
                overall_layers_list.append(ep_sequential_block)
                overall_layers_list.append(
                    MultiplyActivation(scale=layer_scale)
                )  # 전달받은 layer_scale_arg 사용

            elif current_block_type == "BP":
                overall_layers_list.append(nn.Sequential(*internal_block_sub_layers))

            else:
                raise ValueError(
                    f"알 수 없는 블록 타입: '{current_block_type}' (인덱스 {block_start_cfg_type_idx}). "
                    "'EP' 또는 'BP' 여야 합니다."
                )

        return overall_layers_list

class AdjacentShortcutBackbone(nn.Module):
    def __init__(
        self,
        cfg: list[int] = [784 * 2, 128, 10 * 2],
        beta: float = 0.1,
        bias: bool | list[bool] = [True, True],
        scale_input: int = 2,
        scale_output: int = 2,
        data_scale: float = 1.0,
        solver: eqprop.solvers.EqPropSolver | None = None,
        param_adjuster: eqprop_utils.AdjustParams | None = eqprop_utils.AdjustParams(),
        eqprop_fn: Literal["positive", "altered", "centered"] = "centered",
        initialization: dict | None = None,
        res_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.data_scale = data_scale

        if solver:
            # Automatically generate shortcuts for adjacent HIDDEN layers
            num_layers = len(cfg) - 1
            # We only want to connect hidden layers, so we loop up to the second to last layer.
            # The last connection (last hidden -> output) is excluded.
            num_hidden_connections = num_layers - 2
            adjacent_shortcuts = []
            if num_hidden_connections > 0:
                for i in range(num_hidden_connections):
                    adjacent_shortcuts.append((i, i + 1, res_scale))

            if hasattr(solver, "set_shortcuts"):
                solver.set_shortcuts(adjacent_shortcuts)
            else:
                raise AttributeError("The provided solver does not have a 'set_shortcuts' method.")

        self.model = enn.EqPropSequential(
            *self._make_layers(cfg, bias),
            eqprop_fn=eqprop_fn,
            solver=solver,
        )
        self.param_adjuster = param_adjuster
        eqprop_utils.interleave.set_num_input(scale_input)
        eqprop_utils.interleave.set_num_output(scale_output)

        initialize_weights(self.model, initialization)

    @staticmethod
    def _make_layers(cfg, bias):
        layers = []
        for idx in range(len(cfg) - 1):
            bias_idx = bias if isinstance(bias, bool) else bias[idx]
            layers.append(
                enn.EqPropLinear(
                    cfg[idx],
                    cfg[idx + 1],
                    bias=bias_idx,
                )
            )
        return layers

    @eqprop_utils.interleave(type="both")
    def forward(self, x, return_all_activities: bool = False):
        if self.param_adjuster is not None:
            self.model.apply(self.param_adjuster)
        x = x * self.data_scale
        return self.model(x, return_all_activities=return_all_activities)
