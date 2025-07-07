#!/usr/bin/env python3
"""
Minimal reproduction script for ProxSuite nanobind memory leak.

This script reproduces the exact pattern causing nanobind memory leaks:
1. Create VectorQP()
2. Solve problems
3. Don't delete objects
4. Use WARM_START for nudged phase
"""

import torch
import numpy as np
import proxsuite


def reproduce_nanobind_leak():
    """Reproduce the nanobind memory leak with minimal code."""
    print("=== Reproducing nanobind memory leak ===")

    # Problem dimensions
    n = 50  # problem size
    batch_size = 2

    # Create dummy QP problem: minimize 0.5 * x^T * H * x + g^T * x
    # subject to lb <= x <= ub
    H = np.eye(n, dtype=np.float64)  # Identity matrix
    g = np.random.randn(batch_size, n).astype(np.float64)
    lb = -np.ones(n, dtype=np.float64)
    ub = np.ones(n, dtype=np.float64)

    print(f"Problem size: {n}")
    print(f"Batch size: {batch_size}")

    # Step 1: Create VectorQP and solve (free phase)
    print("\n--- Step 1: Create VectorQP and solve (free phase) ---")
    qps = proxsuite.proxqp.dense.VectorQP()

    for i in range(batch_size):
        qp = proxsuite.proxqp.dense.QP(n, 0, n, True)  # n, n_eq, n_ineq, box_constrained
        qp.init(H, g[i], None, None, None, None, None, lb, ub)
        qps.append(qp)

    # Solve in parallel
    proxsuite.proxqp.dense.solve_in_parallel(qps, 1)

    print("Free phase solved")
    for i, qp in enumerate(qps):
        print(f"  QP {i} status: {qp.results.info.status}")
        print(f"  QP {i} solution norm: {np.linalg.norm(qp.results.x):.4f}")

    # Step 2: Modify problem for nudged phase WITHOUT cleaning up
    print("\n--- Step 2: Modify for nudged phase (NO cleanup) ---")

    # Modify g vector for nudged phase (simulate external current)
    g_nudged = g + 0.1 * np.random.randn(batch_size, n).astype(np.float64)

    # Update QP problems with WARM_START (this is where leak happens)
    for i, qp in enumerate(qps):
        qp.settings.initial_guess = proxsuite.proxqp.InitialGuess.WARM_START_WITH_PREVIOUS_RESULT
        qp.update(g=g_nudged[i], l_box=lb, u_box=ub)

    # Solve again in parallel
    proxsuite.proxqp.dense.solve_in_parallel(qps, 1)

    print("Nudged phase solved")
    for i, qp in enumerate(qps):
        print(f"  QP {i} status: {qp.results.info.status}")
        print(f"  QP {i} solution norm: {np.linalg.norm(qp.results.x):.4f}")

    # Step 3: Don't clean up properly (this causes the leak)
    print("\n--- Step 3: Improper cleanup (causing leak) ---")
    print("NOT calling qp.clear() on individual QP objects...")
    print("NOT calling qps.clear() on VectorQP...")
    print("Just deleting variables without proper cleanup...")

    # This is the problematic pattern - just delete without calling clear()
    del qps
    del g, g_nudged, H, lb, ub

    print("Variables deleted (leak should occur now)")


def reproduce_with_repeated_calls():
    """Reproduce with multiple iterations to amplify the leak."""
    print("\n\n=== Reproducing with multiple iterations ===")

    for iteration in range(3):
        print(f"\n>>> Iteration {iteration + 1} <<<")

        # Problem dimensions
        n = 20
        batch_size = 2

        # Create problem
        H = np.eye(n, dtype=np.float64)
        g = np.random.randn(batch_size, n).astype(np.float64)
        lb = -np.ones(n, dtype=np.float64)
        ub = np.ones(n, dtype=np.float64)

        # Create VectorQP
        qps = proxsuite.proxqp.dense.VectorQP()

        for i in range(batch_size):
            qp = proxsuite.proxqp.dense.QP(n, 0, n, True)
            qp.init(H, g[i], None, None, None, None, None, lb, ub)
            qps.append(qp)

        # Free phase
        proxsuite.proxqp.dense.solve_in_parallel(qps, 1)

        # Nudged phase with WARM_START
        g_nudged = g + 0.1 * np.random.randn(batch_size, n).astype(np.float64)

        for i, qp in enumerate(qps):
            qp.settings.initial_guess = (
                proxsuite.proxqp.InitialGuess.WARM_START_WITH_PREVIOUS_RESULT
            )
            qp.update(g=g_nudged[i], l_box=lb, u_box=ub)

        proxsuite.proxqp.dense.solve_in_parallel(qps, 1)

        # IMPROPER cleanup - just delete without clear()
        del qps, g, g_nudged, H, lb, ub

        print(f"Iteration {iteration + 1} completed (leak accumulating)")


def demonstrate_proper_cleanup():
    """Demonstrate proper cleanup to prevent leaks."""
    print("\n\n=== Demonstrating PROPER cleanup ===")

    n = 30
    batch_size = 2

    # Create problem
    H = np.eye(n, dtype=np.float64)
    g = np.random.randn(batch_size, n).astype(np.float64)
    lb = -np.ones(n, dtype=np.float64)
    ub = np.ones(n, dtype=np.float64)

    # Create VectorQP
    qps = proxsuite.proxqp.dense.VectorQP()

    for i in range(batch_size):
        qp = proxsuite.proxqp.dense.QP(n, 0, n, True)
        qp.init(H, g[i], None, None, None, None, None, lb, ub)
        qps.append(qp)

    # Free phase
    proxsuite.proxqp.dense.solve_in_parallel(qps, 1)

    # Nudged phase
    g_nudged = g + 0.1 * np.random.randn(batch_size, n).astype(np.float64)

    for i, qp in enumerate(qps):
        qp.settings.initial_guess = proxsuite.proxqp.InitialGuess.WARM_START_WITH_PREVIOUS_RESULT
        qp.update(g=g_nudged[i], l_box=lb, u_box=ub)

    proxsuite.proxqp.dense.solve_in_parallel(qps, 1)

    # PROPER cleanup
    print("Performing PROPER cleanup...")
    for i, qp in enumerate(qps):
        print(f"  Clearing QP {i}")
        qp.clear()

    print("Clearing VectorQP container")
    qps.clear()

    del qps, g, g_nudged, H, lb, ub

    print("Proper cleanup completed (no leak should occur)")


if __name__ == "__main__":
    print("ProxSuite nanobind memory leak reproduction")
    print(f"ProxSuite version: {proxsuite.__version__}")

    # Reproduce the leak pattern
    reproduce_nanobind_leak()

    # Amplify with multiple iterations
    reproduce_with_repeated_calls()

    # Show proper cleanup
    demonstrate_proper_cleanup()

    print("\n=== Script completed ===")
    print("Check for nanobind leak messages above...")
