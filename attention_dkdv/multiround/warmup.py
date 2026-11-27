"""Reusable warmup-only graph with actual CUDA-event duration accounting."""

import math

import torch


class GraphWarmup:
    def __init__(self, fn, launches=32):
        self.fn = fn
        self.launches = launches
        self.graph = torch.cuda.CUDAGraph()
        torch.cuda.synchronize()
        with torch.cuda.graph(self.graph):
            for _ in range(launches):
                fn()
        torch.cuda.synchronize()
        self.estimate_ms = None

    def batch(self, count):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(count):
            self.graph.replay()
        end.record()
        end.synchronize()
        return start.elapsed_time(end)

    def run(self, target_ms):
        elapsed = 0.0
        replays = 0
        if self.estimate_ms is None:
            elapsed = self.batch(1)
            replays = 1
            self.estimate_ms = max(0.001, elapsed)
        while elapsed < target_ms:
            count = max(1, math.ceil((target_ms - elapsed) / self.estimate_ms))
            duration = self.batch(count)
            elapsed += duration
            replays += count
            self.estimate_ms = max(0.001, duration / count)
        assert elapsed >= target_ms
        return {
            "target_ms": target_ms,
            "actual_event_ms": elapsed,
            "graph_replays": replays,
            "kernel_launches": replays * self.launches,
        }
