from typing import Any
from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.core.eqprop.solvers import AnalogEqPropSolver
from src.utils import eqprop_utils


class EP(nn.Module):
    def __init__(
        self,
        batch_size: int,
        beta: float = 1e-2,
        dims: list = [784, 500, 10],
        iters: tuple = (20, 4),
        activation=torch.sigmoid,
        epsilon: float = 0.5,
        criterion=nn.MSELoss(reduction="none"),
        L: Sequence = None,
        U: Sequence = None,
        *args,
        **kwargs,
    ):
        """Equilibrium Propagation (EP) model.

        Args:
            batch_size (_type_): _description_
            beta (float, optional): loss coupling strength. Defaults to 1e-2.
            dims (list, optional): dimensions of network architectures. Defaults to [784,500,10].
            iters (tuple, optional): Iterations for minimizing phases. Defaults to (200,4).
            activation (_type_, optional): nonlinear activation function for nodes. Defaults to torch.sigmoid.
            epsilon (float, optional): step size for minimizing Energy. Defaults to 0.5.
            criterion (_type_, optional): loss function. Defaults to nn.MSELoss(reduction='none').
        """
        super().__init__()

        self.batch_size = batch_size
        self.beta = beta
        self.bias: bool | int = kwargs.get("bias", True)
        self.pos_W = kwargs.get("pos_W", False)
        self.doubling = kwargs.get("doubling", False)
        self.num_classes = dims[-1]
        if self.doubling:
            # eqprop_utils.interleave.on()
            self.num_classes //= 2
        self.eps = epsilon
        self.dims = dims
        self.L = L
        self.U = U
        assert len(iters) == 2, ValueError("2 iteration steps(free, nudge) required")
        self.free_iters, self.nudge_iters = iters
        self.activation = activation

        self._Nodes = [
            torch.empty((batch_size, n)).normal_(0.5, 0.5).clamp(0, 1).requires_grad_(True)
            for n in dims[1:]
        ]

        self.criterion = criterion
        # self.inner_optimizer = torch.optim.SGD(self._Nodes, lr=self.eps)
        self._init_weights()
        # self.metric_handler = metricHandler().setup(num_layers=len(self.dims) - 1)

    def _init_weights(self):
        # pos_W, bias
        self.W = nn.ModuleList()
        bias = False if self.bias == 2 else self.bias

        for idx in range(len(self.dims) - 1):
            if idx == 0 & self.bias == 2:
                self.W.append(nn.Linear(self.dims[idx], self.dims[idx + 1], bias=True))
                nn.init.constant_(self.W[idx].bias, 1)
            else:
                self.W.append(nn.Linear(self.dims[idx], self.dims[idx + 1], bias=bias))
            if self.pos_W:
                assert self.L is not None, ValueError("L is required for pos_W")
                assert self.U is not None, ValueError("U is required for pos_W")
                if isinstance(self.L, float | int):
                    self.L = [self.L] * (len(self.dims) - 1)
                    self.U = [self.U] * (len(self.dims) - 1)

                assert len(self.L) == len(self.dims) - 1, ValueError(
                    "L and U must have the same length as dims"
                )
                nn.init.uniform_(self.W[idx].weight, self.L[idx], self.U[idx])
            else:
                nn.init.xavier_uniform_(self.W[idx].weight)

    # @eqprop_utils.type_as
    def forward(self, x, y=None, beta=0.0) -> list[torch.Tensor]:
        """Relax Nodes till converge."""
        self.W.requires_grad_(False)  # freeze weights
        if beta == 0.0:
            opt_nodes = self.minimize(x, y, beta=beta, iters=self.free_iters)
        else:
            opt_nodes = self.minimize(x, y, beta=beta, iters=self.nudge_iters)
        return opt_nodes

    # TODO: implement a better algorithm to find the optimal nodes (e.g. Newton's method)

    def minimize(
        self,
        x,
        y=None,
        Nodes: list[torch.Tensor] = None,
        beta=0.0,
        iters=None,
        **kwargs,
    ) -> list[torch.Tensor]:
        """Minimize the total energy function using torch.autograd."""
        # 1. 최적화할 뉴런(Nodes)과 반복 횟수(iters) 설정
        Nodes = self._Nodes if Nodes is None else Nodes
        iters = self.free_iters if iters is None else iters
        self.W.requires_grad_(False)  # 2. freeze weights
        # 3. 지정된 횟수만큼 에너지 최소화 단계를 반복
        for _ in range(iters):
            # print(Nodes)
            self.step(Nodes, x, y, beta)
        # 4. 최적화된 뉴런 상태를 복사하여 반환
        relaxedNodes1 = [nodes.clone().detach() for nodes in Nodes]
        return relaxedNodes1

    def step(self, Nodes: list[torch.Tensor], x, y=None, beta: float = 0.0) -> None:
        """Update Nodes one step.

        Args:
            Nodes (list[torch.Tensor]): _description_
            x (_type_): _description_
            y (_type_, optional): _description_. Defaults to None.
            beta (float, optional): _description_. Defaults to 0.0.
        """
        # compute grads(dE/du) # 현재 뉴런 상태(Nodes)에서의 총 에너지(E)를 계산
        E, __ = self.Tenergy(Nodes, x, y, beta=beta)
        # update Nodes
        grads = torch.autograd.grad(E.sum(), Nodes)
        # 계산된 그래디언트 사용, Nodes 업데이트
        with torch.no_grad():
            for idx, layergrads in enumerate(grads):
                # 3a. 경사 하강법 업데이트
                Nodes[idx] -= self.eps * layergrads
                # 3b. 뉴런 활성화 값의 범위를 [0,1]로 제한
                Nodes[idx] = torch.clamp(Nodes[idx], 0, 1)
                # 3c. 다음 스텝을 위해 그래디언트 추적 다시 활성화
                Nodes[idx].requires_grad_(True)

    # TODO: make net._Nodes gradient zero after step
    def step_(self, Nodes: list[torch.Tensor], x, y=None, beta: float = 0.0) -> None:
        """Alternative method to step. Use autograd.backward() instead.

        Args:
            Nodes (list[torch.Tensor]): _description_
            x (_type_): _description_
            y (_type_, optional): _description_. Defaults to None.
            beta (float, optional): _description_. Defaults to 0.0.
        """
        # 1. 현재 뉴런 상태에서의 총에너지(E)를 계산합니다. (step 함수와 동일)
        E, _ = self.Tenergy(Nodes, x, y, beta)
        # 2. autograd.backward(0)를 호출하여 그래디언트를 계산하고 저장합니다.(핵심 차이점 1)
        E.sum().backward()
        # 3. 계산된 그래디언트를 사용하여 뉴런 상태(Nodes)를 업데이트합니다.
        with torch.no_grad():
            for idx, nodes in enumerate(Nodes):
                # 3a. .grad 속성에 저장된 그래디언트를 사용 (핵심 차이점 2)
                nodes -= self.eps * nodes.grad
                # 3b. 뉴런 활성화 값 범위 [0,1]로 제한
                nodes = torch.clamp(nodes, 0, 1)
                # 3c. 그래디언트 추적을 다시 활성화
                nodes.requires_grad_(True)
                # 3d. 사용한 그래디언트를 초기화 (핵심 차이점 3)
                nodes.grad = None  # ...?

    def energy(self, Nodes: list[torch.Tensor], x) -> torch.Tensor: 
        """Energy function."""
        it = len(Nodes)
        act = self.activation
        assert it == len(self.dims) - 1, ValueError(
            "number of nodes must match the number of layers"
        )
        assert it == len(self.W), ValueError("number of nodes must match the number of layers")

        def layer_energy(n: torch.Tensor, w: nn.Module, m: torch.Tensor): # 1. 두 레이어 사이의 에너지
            """Energy function for a layer.

            Args:
                n (torch.Tensor):B x I
                w (torch.nn.Linear): O x I
                m (torch.Tensor): B x O

            Returns:
                E_layer = E_nodes - E_weights - E_biases
            """
            # 1-1. 뉴런 자체의 에너지 (Self-energy)
            nodes_energy = 0.5 * torch.sum(torch.pow(n, 2), dim=1)
            # 1-2. 가중치를 통한 뉴런 간 상호작용 에너지 (Interaction Energy)
            weights_energy = 0.5 * (torch.matmul(act(m), w.weight) * act(n)).sum(dim=1)
            # 1-3. 편향(bias)에 의한 에너지
            biases_energy = torch.matmul(act(m), w.bias) if getattr(w, "bias") is not None else 0.0
            return nodes_energy - weights_energy - biases_energy
        # 2. 메인 루프 : 총 에너지 계산
        for idx in range(it):
            if idx == 0:
                # 첫 번째 레이어 : 입력 x와 첫 번째 히든 레이어 Nodes[0] 사이의 에너지
                E = layer_energy(x, self.W[idx], Nodes[idx])
            else:
                # 중간 레이어들 : 이전 히든 레이어와 현재 히든 레이어 사이의 에너지
                E += layer_energy(Nodes[idx - 1], self.W[idx], Nodes[idx])
        # 마지막 출력 레이어의 자체 에너지를 더해줌
        E += 0.5 * torch.sum(torch.pow(Nodes[-1], 2), dim=1)  # add E_nodes of output layer
        return E

    def Tenergy(
        self, Nodes: list[torch.Tensor], x, y=None, beta: float = 0.0, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute Total Free Energy: Wsum rho(u_i)W_{ij}rho(u_j)"""
        # 1. 먼저, 순수한 물리적 에너지를 계산
        E = self.energy(Nodes, x)
        L = None
        # 2. 'beta'가 0이 아니면 (즉, Nudge phase이면) 손실 항을 추가
        if beta != 0:
            # 2a. y(정답 레이블)가 반드시 제공되어야 함을 확인
            assert y is not None, ValueError("y must be provided if beta != 0")
            # 2b. 출력 뉴런(Nodes[-1])과 정답(y) 사이 손실(L)을 계산
            L = self.loss(Nodes[-1], y)
            # 2c. 총 에너지 E에 (beta * 손실 L)을 더함
            E += beta * L
            # 2d. 로깅 등을 위해 손실 값의 평균을 계산하여 준비
            L = L.mean().detach()
        return (E, L)

    @eqprop_utils.interleave(type="in") # y -> y+, y- 즉, non-negative 두 개를 이용하여 음수를 표현할 수 있도록 하는 데코레이터
    def loss(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor: #loss 게산하는 함수
        """Compute loss."""
        # 1. 손실 함수의 종류를 확인(이름에 'MSE'가 포함되어 있는지)
        if self.criterion.__class__.__name__.find("MSE") != -1:
            # 2a. MSE 게열일 경우 : 정답 y를 one-hot 벡터로 변환
            y = F.one_hot(y, num_classes=self.num_classes)
            # 3a. 변환된 y와 에측 y_hat으로 손실 계산, 클래스 차원에 대해 합산
            L = self.criterion(y_hat.float(), y.float()).sum(dim=1).squeeze()
        else:
            #2b. 다른 손실 함수일 경우 (ex : CrossEntropyLoss) y를 변환 없이 그대로 사용하여 손실 계산
            L = self.criterion(y_hat.float(), y).squeeze()
        return L

    @torch.no_grad()
    def update(self, free_nodes: list[torch.Tensor], nudge_nodes: list[torch.Tensor], x) -> None:
        """Update weights with hardcoded gradients from theorm.

        dw_ij = (rho(un_i)rho(un_j) - rho(uf_i)rho(uf_j))/beta
        Annot.
          un<>: minimized nudge_nodes
          uf<>: minimized free_nodes

        Args:
            free_nodes (list[torch.Tensor]): _description_
            nudge_nodes (list[torch.Tensor]): _description_
            x (_type_): _description_
        """
        # lr = 1e-1
        act = self.activation
        # 1. 입력 x를 뉴런 리스트의 맨 앞에 추가 (0번째 레이어로 취급)
        free_nodes.insert(0, x)
        nudge_nodes.insert(0, x)
        # 2. 가중치(W)의 그래디언트를 계산할 준비
        self.W.requires_grad_(True)
        self.W.zero_grad()
        # 3. 각 레이어의 가중치(W)와 편항(bias)에 대해 그래디언트를 계산하고 설정
        for idx, W in enumerate(self.W):
            # 3a. 가중치(W)의 그래디언트 계산
            W.weight.grad = (
                torch.matmul(
                    act(nudge_nodes[idx + 1]).mean(dim=0, keepdim=True).T,
                    act(nudge_nodes[idx]).mean(dim=0, keepdim=True),
                )
                - torch.matmul(
                    act(free_nodes[idx + 1]).mean(dim=0, keepdim=True).T,
                    act(free_nodes[idx]).mean(dim=0, keepdim=True),
                )
            ) / (-self.beta) # 이론상 1/beta인데, 코드에서는 -1/beta. 이는 에너지 정의와 관련될 수 있음. (gemini 왈)
            # consider bias as synaptic weights with u_i = 1
            
            # 3b. 편향(bias)의 그래디언트 계산 
            if getattr(W, "bias") is not None:
                W.bias.grad = (
                    act(nudge_nodes[idx + 1]).mean(dim=0) - act(free_nodes[idx + 1]).mean(dim=0)
                ) / (-self.beta)

    def update_(self, free_nodes, nudge_nodes, x, y) -> tuple[torch.Tensor, torch.Tensor, Any]:
        """Update weights from optimized free_nodes & nudge_nodes using
        autograd.backward()

        Args:
            free_nodes (_type_): _description_
            nudge_nodes (_type_): _description_
            x (_type_): _description_
            y (_type_): _description_

        Returns:
            Tuple[torch.Tensor, torch.Tensor, Any]: Free Energy, Nudge Energy, (optional) loss
        """
        self.W.requires_grad_(True)
        self.W.zero_grad()
        # set W.grads
        Efs, _ = self.Tenergy(free_nodes, x, y, beta=0.0)
        Ef = Efs.mean()
        Ef.backward(retain_graph=True)
        Ens, loss = self.Tenergy(nudge_nodes, x, y, beta=self.beta)
        En = Ens.mean()
        (-En).backward()
        return (Ef.clone().detach(), En.clone().detach(), loss)

    @property
    def Nodes(self) -> list[torch.Tensor]:
        """Get nodes."""
        Nodes = [nodes.clone().detach().requires_grad_(True) for nodes in self._Nodes]
        return Nodes


class AnalogEP(EP):
    """Use slightly different energy(pseudopower)

    Attrs:
        _Nodes (list[torch.Tensor]): output node voltages of each layer
    """

    def __init__(
        self,
        batch_size: int,
        beta: float = 1e-2,
        dims: list = [784, 500, 10],
        activation=lambda x: x,
        epsilon: float = 0.2,
        criterion=nn.MSELoss(reduction="none"),
        *args,
        **kwargs,
    ):
        super().__init__(
            batch_size,
            beta,
            dims,
            (1, 1),
            activation,
            epsilon,
            criterion,
            *args,
            **kwargs,
        )
        DeprecationWarning("AnalogEP is deprecated. Use AnalogEP2 instead.")

    def _init_weights(self):
        super()._init_weights()

    def energy(self, Nodes, x) -> torch.Tensor:
        if not hasattr(self, "Is"):
            self.Is = 1e-6
            self.Vl = 0.1
            self.Vr = 0.9

        num_layers = len(Nodes)
        act = self.activation
        assert num_layers == len(self.dims) - 1, ValueError(
            "number of nodes must match the number of layers"
        )
        assert num_layers == len(self.W), ValueError(
            "number of nodes must match the number of layers"
        )

        def layer_power(n: torch.Tensor, w: nn.Module, m: torch.Tensor):
            r"""Energy function for a layer.

            Args:
                n (torch.Tensor):B x I
                w (torch.nn.Linear): O x I
                m (torch.Tensor): B x O

            Returns:
                E_layer = 0.5 * \Sum{G * (n_i - m_i)^2} : B

            """
            return 0.5 * torch.sum(w.weight * self.deltaV(n, m).pow(2), dim=(1, 2))

        def rectifier_power(x: torch.Tensor):  # , Is=1e-8, Vt1=0.1, Vt2=0.9, eta=1):
            def diode_power(V, Vt):
                return 0.026 * self.Is * (torch.exp((V - Vt) / (0.026)) - 1) - self.Is * (V - Vt)

            return torch.sum(diode_power(-x, -self.Vl) + diode_power(x, self.Vr), dim=-1)

        for idx in range(num_layers):
            if idx == 0:
                E = layer_power(x, self.W[idx], Nodes[idx]) + rectifier_power(Nodes[idx])
            elif idx != num_layers - 1:
                E += layer_power(act(Nodes[idx - 1]), self.W[idx], Nodes[idx]) + rectifier_power(
                    Nodes[idx]
                )
            else:
                E += layer_power(act(Nodes[idx - 1]), self.W[idx], Nodes[idx])
        return E

    @classmethod
    def deltaV(cls, n: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        """Compute deltaV matrix from 2 node voltages.

        Args:
            n (torch.Tensor): (B x) I
            m (torch.Tensor): (B x) O

        Returns:
            torch.Tensor: (B x) O x I
        """
        if len(n.shape) == 2: #배치 처리 (B : 배치 크기, I : 입력 뉴런 수, O : 출력 뉴런 수)
            # n의 shape : (B, I), m의 shape: (B, O)
            assert n.shape[0] == m.shape[0], ValueError("n and m must have the same batch size")
            # 1. 입력, 출력 텐서의 n, m을 브로드캐스팅 가능하게 변형
            N = n.clone().unsqueeze(dim=-1).repeat(1, 1, m.shape[-1]).transpose(1, 2)
            M = m.clone().unsqueeze(dim=-1).repeat(1, 1, n.shape[-1])
        elif len(n.shape) == 1:
            N = n.clone().unsqueeze(dim=-1).repeat(1, m.shape[-1]).T
            M = m.clone().unsqueeze(dim=-1).repeat(1, n.shape[-1])
        else:
            ValueError("n and m must be 1D or 2D")
        # 2. 두 텐서 차이를 계산하여 deltaV 행렬을 반환
        return N - M

    @torch.no_grad()
    def update(self, free_opt_Vout: list[torch.Tensor], nudge_opt_Vout: list[torch.Tensor], x):
        """Update weights from optimized Node Voltages (free_opt_Vout &
        nudge_opt_Vout)"""
        # 1. 그래디언트 계산 준비 (자동 미분 비활성화, 기존 그래디언트 초기화)
        self.W.requires_grad_(True)
        self.W.zero_grad()
        self.fdV.clear() # (로깅용 변수 초기화)
        self.ndV.clear()
        # 2. 각 레이어(W)를 순회하며 그래디언트 계산
        for idx, W in enumerate(self.W):
            # 2a. 현재 레이어의 입력 전압을 결정 (첫 레이어는 x, 나머지는 이전 레이어의 출력)
            free_opt_vin = self.activation(free_opt_Vout[idx - 1]) if idx != 0 else x
            # 2b. deltaV 함수로 자유/유도 단계의 전압 차이(delta V) 행렬을 각각 계산
            fdV = self.deltaV(free_opt_vin, free_opt_Vout[idx])
            # (로깅을 위해 배치 평균 저장)
            self.fdV.append(fdV.mean(dim=0))
            nudge_opt_vin = self.activation(nudge_opt_Vout[idx - 1]) if idx != 0 else x
            ndV = self.deltaV(nudge_opt_vin, nudge_opt_Vout[idx])
            self.ndV.append(ndV.mean(dim=0))
            # 2c. 학습 규칙에 따라 그래디언트를 직접 계산하여 .grad 속성에 할당
            W.weight.grad = (1 / self.beta) * (ndV.pow(2).mean(dim=0) - fdV.pow(2).mean(dim=0))

            # TODO: bias?


class AnalogEP2(nn.Module):
    """Direct implementation of analog eqprop.

    Args:
        batch_size (int): batch size
        solver (AnalogEqPropSolver): solver class
        cfg (list[int], optional): network architecture. Defaults to [784*2, 128, 10*2].
        beta (float, optional): loss coupling strength. Defaults to 0.1.
        bias (bool, optional): use bias. Defaults to False.
        positive_w (bool, optional): positive weights. Defaults to True.
        min_w (float, optional): minimum weight. Defaults to 1e-6.
        max_w (Optional[float], optional): maximum weight. Defaults to None.
        max_w_gain (Optional[float], optional): maximum weight gain. Defaults to 0.28.
        scale_input (int, optional): input scaling. Defaults to 2.
        scale_output (int, optional): output scaling. Defaults to 2.
    """

    def __init__(
        self,
        batch_size: int,
        solver: AnalogEqPropSolver,
        cfg: list[int] = [784 * 2, 128, 10 * 2],
        beta: float = 0.1,
        bias: bool = False,
        positive_w: bool = True,
        min_w: float = 1e-6,
        max_w: float | None = None,
        max_w_gain: float | None = 0.28,
        scale_input: int = 2,
        scale_output: int = 2,
    ) -> None:
        super().__init__()
        self.beta = beta
        layers = []
        for idx in range(len(cfg) - 1):
            layers.append(nn.Linear(cfg[idx], cfg[idx + 1], bias=bias))
        self.model = nn.Sequential(*layers)

        # init weights
        if positive_w:
            self.model.apply(
                eqprop_utils.positive_param_init(
                    min_w,
                    max_w,
                    max_w_gain,
                )
            )

        # Add free/nudge nodes per layer as buffers
        self.init_nodes(batch_size)
        self.register_buffer("ypred", torch.empty(batch_size, cfg[-1]))

        # instiantiate solver
        solver.set_model(self.model)
        self.solver = solver
        self.dims = self.solver.strategy.dims
        eqprop_utils.interleave.set_num_input(scale_input)
        eqprop_utils.interleave.set_num_output(scale_output)

        FutureWarning("AnalogEP2 will be replaced by src.core.eqprop.nn.EqPropLinear")

    @eqprop_utils.interleave(type="both")
    @torch.no_grad()
    def forward(self, x):
        """Forward propagation.

        Args:
            x (_type_): _description_

        Returns:
            _type_: _description_
        """
        # assert self.training is False
        self.reset_nodes()
        vouts = self.solver(x)
        self.set_nodes(vouts, positive_phase=True)
        logits = self.model[-1].get_buffer("positive_node")
        self.ypred = logits.detach().clone().requires_grad_(True)
        return self.ypred

    @eqprop_utils.interleave(type="in")
    @torch.no_grad()
    def eqprop(self, x: torch.Tensor):
        """Nudge phase & grad calculation."""
        assert self.training
        vout = self.solver(x, grad=self.ypred.grad)
        self.set_nodes(vout, positive_phase=False)
        self.prev_positive = self.prev_negative = x
        self.model.apply(self._update)

    def _update(self, submodule: nn.Module):
        """Set gradients of parameters manually.

        dL/dw = (nudge_dV^2 - free_dV^2)/beta
        = [prev_negative^2 + n_node^2
        - (prev_positive^2 + p_node^2)
        - 2(prev_negative.T@n_node - prev_positive@p_node)]/beta

        Args:
            submodule (nn.Module): submodule of self.model
        """
        if hasattr(submodule, "weight"):
            if submodule.weight.grad is None:
                submodule.weight.grad = torch.zeros_like(submodule.weight)
            p_node = submodule.get_buffer("positive_node")
            n_node = submodule.get_buffer("negative_node")
            res = 2 * (
                torch.bmm(p_node.unsqueeze(2), self.prev_positive.unsqueeze(1))
                .reshape((p_node.size(0), *submodule.weight.shape))
                .mean(dim=0)
                - torch.bmm(n_node.unsqueeze(2), self.prev_negative.unsqueeze(1))
                .reshape((p_node.size(0), *submodule.weight.shape))
                .mean(dim=0)
            )
            # broadcast to 2D
            res += self.prev_negative.pow(2).mean(dim=0) - self.prev_positive.pow(2).mean(dim=0)
            res += (n_node.pow(2).mean(dim=0) - p_node.pow(2).mean(dim=0)).unsqueeze(1)
            submodule.weight.grad += res / self.solver.beta
            if submodule.bias is not None:
                if submodule.bias.grad is None:
                    submodule.bias.grad = torch.zeros_like(submodule.bias)
                submodule.bias.grad += (
                    (
                        (n_node - p_node)
                        * (
                            n_node + p_node - 2 * torch.ones_like(p_node)
                        )  # (n-1)^2-(f-1)^2=2(n-f)(n+f-2)
                    ).mean(dim=0)
                    * 2
                    / self.solver.beta
                )

            self.prev_positive = p_node
            self.prev_negative = n_node
            amp_factor = self.solver.amp_factor
            if amp_factor != 1.0:
                self.prev_positive *= amp_factor
                self.prev_negative *= amp_factor

    def init_nodes(self, batch_size) -> None:
        """Initialize free/nudge nodes."""

        def _init_nodes(submodule: nn.Module):
            if hasattr(submodule, "weight"):
                assert submodule._get_name() in ["Linear"], "Only Linear layer is supported"
                output_size = submodule.weight.shape[0]
                positive_node = torch.zeros((batch_size, output_size))
                negative_node = torch.zeros((batch_size, output_size))
                submodule.register_buffer("positive_node", positive_node)
                submodule.register_buffer("negative_node", negative_node)

        self.model.apply(_init_nodes)

    def reset_nodes(self):
        """Reset free/nudge nodes to zero."""
        for buf in self.model.buffers():
            buf.zero_()

    def set_nodes(self, vout: tuple[torch.Tensor], positive_phase: bool) -> None:
        """Set free/nudge nodes to each layer.

        Args:
            vout (torch.Tensor): concatenated output of each layer
            positive_phase (bool): True if positive phase, False otherwise
        """

        def _set_nodes_layer(submodule: nn.Module):
            nonlocal nodes
            if hasattr(submodule, "positive_node"):
                if positive_phase:
                    submodule.positive_node = nodes.pop()
                else:
                    submodule.negative_node = nodes.pop()

        nodes = list(reversed(vout))
        self.model.apply(_set_nodes_layer)
        del nodes

    def state_dict(self, *args, **kwargs):
        """Override state_dict to exclude node buffers from checkpointing."""
        state_dict = super().state_dict(*args, **kwargs)

        # Remove positive_node, negative_node, and ypred buffers from state_dict
        keys_to_remove = []
        for key in list(state_dict.keys()):
            if any(pattern in key for pattern in ["positive_node", "negative_node", "ypred"]):
                keys_to_remove.append(key)

        for key in keys_to_remove:
            if key in state_dict:
                del state_dict[key]

        return state_dict

    def load_state_dict(self, state_dict, strict: bool = True):
        """Override load_state_dict to handle missing node buffers and batch size mismatch."""
        # Create a copy to avoid modifying the original
        state_dict_copy = state_dict.copy()

        # Remove any remaining node buffers and ypred from the state_dict
        # to avoid size mismatch issues when batch size changes
        keys_to_remove = []
        for key in list(state_dict_copy.keys()):
            if any(pattern in key for pattern in ["positive_node", "negative_node", "ypred"]):
                keys_to_remove.append(key)

        for key in keys_to_remove:
            if key in state_dict_copy:
                del state_dict_copy[key]

        # The node buffers are not in the state_dict (by design)
        # so we need to use strict=False to avoid errors
        return super().load_state_dict(state_dict_copy, strict=False)


class AnalogEPSym(AnalogEP2):

    #Use 3rd nudge phase to compute gradients.
    """

    @eqprop_utils.interleave(type="both")
    @torch.no_grad()
    def forward(self, x):
        Forward propagation.
        # assert self.training is False
        self.reset_nodes()
        vout = self.solver(x)
        # nodes = list(vout.split(self.dims, dim=1))
        logits = vout[:, -self.dims[-1] :]
        self.ypred = logits.clone().detach().requires_grad_(True)
        return self.ypred

    @eqprop_utils.interleave(type="in")
    @torch.no_grad()
    def eqprop(self, x: torch.Tensor):
        Nudge phases & grad calculation.
        vout = self.solver(x, grad=self.ypred.grad)
        self.set_nodes(vout, positive_phase=True)
        self.solver.flip_beta()
        vout = self.solver(x, grad=self.ypred.grad)
        self.set_nodes(vout, positive_phase=False)
        self.prev_positive = self.prev_negative = x
        self.model.apply(self._update)
        self.solver.flip_beta()


class DummyAnalogEP2(AnalogEP2):
    Dummy AnalogEP2 for testing purposes.

    def __init__(
        self,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.model.insert(1, nn.ReLU())

    @eqprop_utils.interleave(type="both")
    def forward(self, x):
        return self.model(x)

    def eqprop(self, x: torch.Tensor):
        pass
"""