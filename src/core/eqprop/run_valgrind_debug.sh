#!/bin/bash

# Valgrind debugging script for ProxSuite nanobind memory leaks

echo "=== Valgrind ProxSuite Memory Leak Analysis ==="

# Check if valgrind is available
if ! command -v valgrind &> /dev/null; then
    echo "❌ Valgrind not found. Installing..."
    sudo apt-get update && sudo apt-get install -y valgrind
fi

# Create a simple test script
cat > /tmp/simple_proxqp_test.py << 'EOF'
import torch
import sys
import os
sys.path.insert(0, '/home/sinnce/workspace/ml')

from src.core.eqprop.strategy.strategies import ProxQPStrategy
from src.core.eqprop import activation

def simple_test():
    # Create strategy
    ots_activation = activation.SymmetricReLU(Vl=-1.0, Vr=1.0)
    strategy = ProxQPStrategy(activation=ots_activation, max_iter=5, atol=1e-6)

    # Set up small network
    strategy.dims = [8, 4, 2]
    strategy.W = [
        torch.randn(8, 16) * 0.1,
        torch.randn(4, 8) * 0.1,
        torch.randn(2, 4) * 0.1,
    ]
    strategy.B = [torch.zeros(dim) for dim in strategy.dims]

    # Test with small batch
    x = torch.randn(1, 16)
    i_ext = torch.randn(1, 2) * 0.1

    print("Solving free phase...")
    v_free = strategy.solve(x, None)

    print("Solving nudged phase...")
    v_nudged = strategy.solve(x, i_ext)

    print("Cleaning up...")
    if hasattr(strategy, 'qps') and strategy.qps is not None:
        for qp in strategy.qps:
            if hasattr(qp, 'clear'):
                qp.clear()
        del strategy.qps
        strategy.qps = None

    strategy.reset()
    del strategy
    print("Test completed")

if __name__ == "__main__":
    simple_test()
EOF

echo "Running simple ProxQP test with Valgrind..."

# Run with Valgrind memory leak detection
valgrind \
    --tool=memcheck \
    --leak-check=full \
    --show-leak-kinds=all \
    --track-origins=yes \
    --verbose \
    --log-file=/tmp/valgrind_proxqp.log \
    python /tmp/simple_proxqp_test.py

echo "Valgrind analysis completed. Results:"
echo "=========================================="
cat /tmp/valgrind_proxqp.log

echo ""
echo "=== Memory Leak Summary ==="
grep -A 10 -B 2 "LEAK SUMMARY" /tmp/valgrind_proxqp.log || echo "No leak summary found"

echo ""
echo "=== Detailed Leak Analysis ==="
grep -A 20 "definitely lost\|indirectly lost\|possibly lost" /tmp/valgrind_proxqp.log || echo "No detailed leaks found"

# Clean up
rm -f /tmp/simple_proxqp_test.py
echo ""
echo "Full valgrind log saved to: /tmp/valgrind_proxqp.log"
