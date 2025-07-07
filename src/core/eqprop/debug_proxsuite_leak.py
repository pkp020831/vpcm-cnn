#!/usr/bin/env python3
"""
Debug script for ProxSuite nanobind memory leak analysis.

This script provides comprehensive debugging tools to track ProxSuite object lifecycle
and identify nanobind memory leaks through reference counting analysis.
"""

import gc
import sys
import weakref
import traceback
from typing import Any
import torch

# Enable garbage collection debugging
gc.set_debug(gc.DEBUG_STATS | gc.DEBUG_LEAK)

# Import the problematic module
from src.core.eqprop.strategy.strategies import ProxQPStrategy
from src.core.eqprop import activation


class ProxSuiteLeakDebugger:
    """Debug class to track ProxSuite object lifecycle and references."""

    def __init__(self):
        self.object_refs: list[weakref.ref] = []
        self.gc_stats_before = None
        self.gc_stats_after = None

    def track_object(self, obj: Any, name: str = "unknown") -> None:
        """Track an object with weak references."""

        def callback(ref):
            print(f"Object {name} (id: {id(ref)}) was garbage collected")

        weak_ref = weakref.ref(obj, callback)
        self.object_refs.append(weak_ref)
        print(f"Tracking object {name} (id: {id(obj)}, refs: {sys.getrefcount(obj)})")

    def print_reference_counts(self, obj: Any, name: str = "object") -> None:
        """Print reference count information for an object."""
        print(f"\n=== Reference Analysis for {name} ===")
        print(f"Object ID: {id(obj)}")
        print(f"Reference count: {sys.getrefcount(obj)}")
        print(f"Object type: {type(obj)}")

        # Try to get referrers (objects that reference this object)
        referrers = gc.get_referrers(obj)
        print(f"Number of referrers: {len(referrers)}")

        for i, ref in enumerate(referrers):
            if ref is not locals() and ref is not globals():
                print(f"  Referrer {i}: {type(ref)} (id: {id(ref)})")
                if hasattr(ref, "__dict__"):
                    # Check if our object is in the referrer's __dict__
                    for attr_name, attr_value in ref.__dict__.items():
                        if attr_value is obj:
                            print(f"    -> Found in attribute '{attr_name}'")

    def start_gc_monitoring(self) -> None:
        """Start monitoring garbage collection."""
        self.gc_stats_before = gc.get_stats()
        print("\n=== GC Stats Before ===")
        for i, stats in enumerate(self.gc_stats_before):
            print(f"Generation {i}: {stats}")

    def end_gc_monitoring(self) -> None:
        """End monitoring and show GC statistics."""
        self.gc_stats_after = gc.get_stats()
        print("\n=== GC Stats After ===")
        for i, stats in enumerate(self.gc_stats_after):
            print(f"Generation {i}: {stats}")

        print("\n=== GC Stats Diff ===")
        for i in range(len(self.gc_stats_before)):
            before = self.gc_stats_before[i]
            after = self.gc_stats_after[i]
            print(f"Generation {i}:")
            print(f"  Collections: {after['collections'] - before['collections']}")
            print(f"  Collected: {after['collected'] - before['collected']}")
            print(f"  Uncollectable: {after['uncollectable'] - before['uncollectable']}")

    def force_gc_and_analyze(self) -> None:
        """Force garbage collection and analyze results."""
        print("\n=== Forcing Garbage Collection ===")
        collected = gc.collect()
        print(f"Objects collected: {collected}")

        # Check for uncollectable objects
        uncollectable = gc.garbage
        if uncollectable:
            print(f"Uncollectable objects found: {len(uncollectable)}")
            for i, obj in enumerate(uncollectable):
                print(f"  {i}: {type(obj)} (id: {id(obj)})")
        else:
            print("No uncollectable objects found")

    def analyze_proxsuite_objects(self) -> None:
        """Analyze all objects containing 'proxsuite' in their type name."""
        print("\n=== ProxSuite Objects Analysis ===")
        all_objects = gc.get_objects()
        proxsuite_objects = []

        for obj in all_objects:
            obj_type = str(type(obj))
            if "proxsuite" in obj_type.lower() or "proxqp" in obj_type.lower():
                proxsuite_objects.append(obj)

        print(f"Found {len(proxsuite_objects)} ProxSuite-related objects")
        for i, obj in enumerate(proxsuite_objects):
            print(f"  {i}: {type(obj)} (id: {id(obj)}, refs: {sys.getrefcount(obj)})")
            if hasattr(obj, "__dict__"):
                attrs = [k for k in obj.__dict__.keys() if not k.startswith("_")]
                if attrs:
                    print(f"     Attributes: {attrs}")


def test_proxqp_strategy_basic():
    """Basic test to reproduce the nanobind memory leak."""
    print("=== Basic ProxQP Strategy Test ===")

    debugger = ProxSuiteLeakDebugger()
    debugger.start_gc_monitoring()

    # Create strategy
    ots_activation = activation.SymmetricReLU(Vl=-1.0, Vr=1.0)
    strategy = ProxQPStrategy(activation=ots_activation, max_iter=10, atol=1e-6)
    debugger.track_object(strategy, "ProxQPStrategy")

    # Set up network parameters
    strategy.dims = [128, 64, 10]
    strategy.W = [
        torch.randn(128, 784) * 0.1,
        torch.randn(64, 128) * 0.1,
        torch.randn(10, 64) * 0.1,
    ]
    strategy.B = [torch.zeros(dim) for dim in strategy.dims]

    # Create input data
    batch_size = 4
    x = torch.randn(batch_size, 784)
    i_ext = torch.randn(batch_size, 10) * 0.1

    print(f"Input shape: {x.shape}")
    print(f"External current shape: {i_ext.shape}")

    # Analyze objects before solving
    debugger.analyze_proxsuite_objects()

    # Free phase
    print("\n--- Free Phase ---")
    v_free = strategy.solve(x, None)
    print(f"Free solution shape: {v_free.shape}")

    # Track QP objects if they exist
    if hasattr(strategy, "qps") and strategy.qps is not None:
        debugger.track_object(strategy.qps, "VectorQP")
        debugger.print_reference_counts(strategy.qps, "VectorQP")

    # Nudged phase
    print("\n--- Nudged Phase ---")
    v_nudged = strategy.solve(x, i_ext)
    print(f"Nudged solution shape: {v_nudged.shape}")

    # Analyze objects after solving
    debugger.analyze_proxsuite_objects()
    debugger.print_reference_counts(strategy, "ProxQPStrategy")

    # Manual cleanup
    print("\n--- Manual Cleanup ---")
    if hasattr(strategy, "qps") and strategy.qps is not None:
        print("Manually clearing QP problems...")
        for i, qp in enumerate(strategy.qps):
            print(f"  Clearing QP {i}")
            if hasattr(qp, "clear"):
                qp.clear()
        del strategy.qps
        strategy.qps = None

    # Reset strategy
    strategy.reset()

    # Clean up strategy
    del strategy
    del v_free
    del v_nudged

    # Force garbage collection and analyze
    debugger.force_gc_and_analyze()
    debugger.analyze_proxsuite_objects()
    debugger.end_gc_monitoring()


def test_repeated_solving():
    """Test repeated solving to reproduce memory leaks."""
    print("\n\n=== Repeated Solving Test ===")

    debugger = ProxSuiteLeakDebugger()
    debugger.start_gc_monitoring()

    # Create strategy
    ots_activation = activation.SymmetricReLU(Vl=-1.0, Vr=1.0)
    strategy = ProxQPStrategy(activation=ots_activation, max_iter=10, atol=1e-6)

    # Set up network parameters
    strategy.dims = [64, 32, 10]
    strategy.W = [
        torch.randn(64, 512).clamp(min=1e-5) * 0.1,
        torch.randn(32, 64).clamp(min=1e-5) * 0.1,
        torch.randn(10, 32).clamp(min=1e-5) * 0.1,
    ]
    strategy.B = [torch.zeros(dim) for dim in strategy.dims]

    # Test multiple iterations
    num_iterations = 2
    batch_size = 2

    for iteration in range(num_iterations):
        print(f"\n--- Iteration {iteration + 1}/{num_iterations} ---")

        # Create new input data for each iteration
        x = torch.randn(batch_size, 512)
        i_ext = torch.randn(batch_size, 10) * 0.1

        # Solve
        v_free = strategy.solve(x, None)
        v_nudged = strategy.solve(x, i_ext)
        v_nudged_2 = strategy.solve(x, -i_ext)

        # Check object counts
        debugger.analyze_proxsuite_objects()

        # Clean up iteration variables
        del x, i_ext, v_free, v_nudged

        # Force GC between iterations
        collected = gc.collect()
        print(f"Objects collected after iteration {iteration + 1}: {collected}")

    # Final cleanup
    print("\n--- Final Cleanup ---")
    strategy.reset()
    del strategy

    debugger.force_gc_and_analyze()
    debugger.analyze_proxsuite_objects()
    debugger.end_gc_monitoring()


def test_with_context_manager():
    """Test ProxQP strategy with proper context management."""
    print("\n\n=== Context Manager Test ===")

    class ProxQPContextManager:
        def __init__(self, strategy):
            self.strategy = strategy

        def __enter__(self):
            return self.strategy

        def __exit__(self, exc_type, exc_val, exc_tb):
            print("Context manager cleanup...")
            if hasattr(self.strategy, "qps") and self.strategy.qps is not None:
                for qp in self.strategy.qps:
                    if hasattr(qp, "clear"):
                        qp.clear()
                del self.strategy.qps
                self.strategy.qps = None
            self.strategy.reset()

    debugger = ProxSuiteLeakDebugger()
    debugger.start_gc_monitoring()

    # Create strategy
    ots_activation = activation.SymmetricReLU(Vl=-1.0, Vr=1.0)
    strategy = ProxQPStrategy(activation=ots_activation, max_iter=10, atol=1e-6)

    # Set up network parameters
    strategy.dims = [32, 16, 5]
    strategy.W = [
        torch.randn(32, 256) * 0.1,
        torch.randn(16, 32) * 0.1,
        torch.randn(5, 16) * 0.1,
    ]
    strategy.B = [torch.zeros(dim) for dim in strategy.dims]

    # Use context manager
    with ProxQPContextManager(strategy) as ctx_strategy:
        x = torch.randn(2, 256)
        i_ext = torch.randn(2, 5) * 0.1

        v_free = ctx_strategy.solve(x, None)
        v_nudged = ctx_strategy.solve(x, i_ext)

        print(f"Free solution: {v_free.shape}")
        print(f"Nudged solution: {v_nudged.shape}")

        debugger.analyze_proxsuite_objects()

    # After context manager
    print("After context manager exit...")
    debugger.force_gc_and_analyze()
    debugger.analyze_proxsuite_objects()
    debugger.end_gc_monitoring()


if __name__ == "__main__":
    print("Starting ProxSuite nanobind memory leak debugging...")
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")

    try:
        import proxsuite

        print(f"ProxSuite version: {proxsuite.__version__}")
    except (ImportError, AttributeError):
        print("ProxSuite version info not available")

    # Run tests
    test_proxqp_strategy_basic()
    test_repeated_solving()
    test_with_context_manager()

    print("\n=== Final System State ===")
    final_collected = gc.collect()
    print(f"Final GC collection: {final_collected} objects")

    if gc.garbage:
        print(f"Remaining uncollectable objects: {len(gc.garbage)}")
        for obj in gc.garbage:
            print(f"  {type(obj)}")
    else:
        print("No uncollectable objects remaining")
