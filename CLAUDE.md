# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Environment Setup
```bash
# Create conda environment with uv
conda env create -f environment.yaml -n myenv
conda activate myenv

# Alternative: uv install (recommended)
uv sync

# Development dependencies
uv sync --extra dev
```

### Training and Evaluation
```bash
# Train model with default configuration
python src/train.py

# Train on specific hardware
python src/train.py trainer=cpu
python src/train.py trainer=gpu

# Train with experiment configurations
python src/train.py experiment=eqprop.yaml
python src/train.py experiment=direct_eqprop.yaml

# Evaluate model
python src/eval.py
```

### Testing
```bash
# Run fast tests (excluding slow tests)
make test
# Or: pytest -k "not slow"

# Run all tests including slow ones
make test-full
# Or: pytest

# Run specific test modules
pytest tests/eqprop/test_strategies.py
pytest tests/test_train.py
```

### Code Quality
```bash
# Run pre-commit hooks (formatting, linting)
make format
# Or: pre-commit run -a

# Clean generated files
make clean

# Clean logs
make clean-logs
```

## Architecture Overview

This is a **PyTorch Lightning + Hydra** machine learning research project focused on **Equilibrium Propagation (EqProp)** - a biologically-inspired learning algorithm for neural networks.

### Key Components

**Core EqProp Implementation** (`src/core/eqprop/`):
- `solvers.py`: Main EqPropSolver class that manages equilibrium point solving
- `strategy/strategies.py`: Various solving strategies (GradientDescent, ProxQP, Xyce, etc.)
- `activation.py`: Activation functions for EqProp networks
- `nn/module.py`: EqProp-specific neural network modules

**Models** (`src/models/`, `src/_eqprop/`):
- `mnist_module.py`: Lightning module for MNIST classification
- `backbone.py`: EqPropBackbone with configurable layers
- `direct_backbone.py`: Direct EqProp implementation

**Data** (`src/data/`):
- `mnist_datamodule.py`: Lightning data module for MNIST
- `xor_datamodule.py`: Data module for XOR problem

**Configuration System** (`configs/`):
- Hierarchical Hydra configs for experiments, models, solvers, strategies
- `experiment/`: Pre-configured experiments (eqprop.yaml, direct_eqprop.yaml)
- `eqprop/solver/strategy/`: Different solving strategies (proxqp, newton, gd, etc.)
- `model/net/`: Network architectures for different tasks

### EqProp Solver Strategies

The framework supports multiple equilibrium solving strategies:
- **ProxQPStrategy**: Quadratic programming using proxsuite
- **GradientDescentStrategy**: Iterative gradient descent
- **XyceStrategy**: SPICE circuit simulation (requires Xyce)
- **NewtonStrategy**: Newton's method for faster convergence

### Training Flow

1. **Data Loading**: MNIST/XOR data via Lightning DataModules
2. **Model Setup**: EqProp networks with configurable solvers/strategies
3. **Equilibrium Solving**: Find stable states using chosen strategy
4. **Learning**: Update weights based on equilibrium differences (free vs nudged phases)

### Configuration Override Pattern

Use Hydra's override syntax to customize experiments:
```bash
# Override trainer settings
python src/train.py trainer.max_epochs=50 trainer.accelerator=gpu

# Override model configuration
python src/train.py model.net.bias=false data.batch_size=128

# Override solver strategy
python src/train.py model.solver.strategy=proxqp
```

### Testing Structure

- `tests/eqprop/`: EqProp-specific tests (strategies, circuits, core functionality)
- `tests/test_*.py`: Integration tests for training, evaluation, configs
- Slow tests marked with `@pytest.mark.slow` decorator

### Key Dependencies

- PyTorch Lightning for training orchestration
- Hydra for configuration management
- Optional: proxsuite, qpsolvers, Xyce for advanced solving strategies
- Rich for terminal formatting, pre-commit for code quality