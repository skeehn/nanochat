"""
Progress monitoring and logging utilities for nanochat.
"""

import time
from typing import Optional, Dict, Any
from contextlib import contextmanager


class ProgressTracker:
    """Track progress for training and inference tasks.

    Attributes:
        name: Name of the task being tracked
        total: Total number of items/steps
        current: Current progress count
        start_time: Start time of the task
    """

    def __init__(self, name: str, total: Optional[int] = None):
        """Initialize progress tracker.

        Args:
            name: Name of the task
            total: Total number of items (None for unknown)
        """
        self.name = name
        self.total = total
        self.current = 0
        self.start_time = time.time()
        self.metrics: Dict[str, Any] = {}

    def update(self, n: int = 1, **metrics: Any) -> None:
        """Update progress by n steps and optionally log metrics.

        Args:
            n: Number of steps to increment
            **metrics: Additional metrics to track
        """
        self.current += n
        if metrics:
            self.metrics.update(metrics)

    def get_elapsed_time(self) -> float:
        """Get elapsed time in seconds since start.

        Returns:
            Elapsed time in seconds
        """
        return time.time() - self.start_time

    def get_eta(self) -> Optional[float]:
        """Estimate time remaining based on current progress.

        Returns:
            Estimated seconds remaining, or None if unknown
        """
        if self.total is None or self.current == 0:
            return None

        elapsed = self.get_elapsed_time()
        rate = self.current / elapsed
        remaining = self.total - self.current
        return remaining / rate if rate > 0 else None

    def get_rate(self) -> float:
        """Get current processing rate (items per second).

        Returns:
            Items per second
        """
        elapsed = self.get_elapsed_time()
        return self.current / elapsed if elapsed > 0 else 0.0

    def format_time(self, seconds: Optional[float]) -> str:
        """Format time duration as human-readable string.

        Args:
            seconds: Duration in seconds

        Returns:
            Formatted time string (e.g., "1h 23m 45s")
        """
        if seconds is None:
            return "unknown"

        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)

        if hours > 0:
            return f"{hours}h {minutes}m {secs}s"
        elif minutes > 0:
            return f"{minutes}m {secs}s"
        else:
            return f"{secs}s"

    def __str__(self) -> str:
        """Get progress string representation.

        Returns:
            Progress string with name, progress, rate, and ETA
        """
        parts = [self.name]

        if self.total is not None:
            percentage = (self.current / self.total) * 100
            parts.append(f"{self.current}/{self.total} ({percentage:.1f}%)")
        else:
            parts.append(f"{self.current}")

        rate = self.get_rate()
        parts.append(f"{rate:.2f} it/s")

        eta = self.get_eta()
        if eta is not None:
            parts.append(f"ETA: {self.format_time(eta)}")

        elapsed = self.get_elapsed_time()
        parts.append(f"Elapsed: {self.format_time(elapsed)}")

        if self.metrics:
            metric_str = ", ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                                  for k, v in self.metrics.items())
            parts.append(f"[{metric_str}]")

        return " | ".join(parts)


@contextmanager
def timed_operation(name: str, verbose: bool = True):
    """Context manager to time an operation and optionally print duration.

    Args:
        name: Name of the operation
        verbose: Whether to print timing information

    Yields:
        Dictionary that will contain 'duration' key after context exits

    Example:
        with timed_operation("model training") as timer:
            # ... training code ...
            pass
        print(f"Training took {timer['duration']:.2f}s")
    """
    start_time = time.time()
    timer = {}

    try:
        yield timer
    finally:
        duration = time.time() - start_time
        timer['duration'] = duration
        if verbose:
            print(f"{name} completed in {duration:.2f}s")


class MetricsLogger:
    """Simple metrics logger for tracking training metrics.

    Attributes:
        metrics: Dictionary mapping metric names to lists of values
    """

    def __init__(self):
        """Initialize metrics logger."""
        self.metrics: Dict[str, list] = {}
        self.step = 0

    def log(self, **metrics: Any) -> None:
        """Log metrics for the current step.

        Args:
            **metrics: Metrics to log as keyword arguments
        """
        for name, value in metrics.items():
            if name not in self.metrics:
                self.metrics[name] = []
            self.metrics[name].append(value)
        self.step += 1

    def get_metric(self, name: str) -> list:
        """Get all values for a specific metric.

        Args:
            name: Metric name

        Returns:
            List of all logged values for this metric
        """
        return self.metrics.get(name, [])

    def get_latest(self, name: str) -> Optional[Any]:
        """Get the most recent value for a metric.

        Args:
            name: Metric name

        Returns:
            Most recent value or None if not logged
        """
        values = self.metrics.get(name, [])
        return values[-1] if values else None

    def get_average(self, name: str, last_n: Optional[int] = None) -> Optional[float]:
        """Get average value for a numeric metric.

        Args:
            name: Metric name
            last_n: If specified, average over last N values only

        Returns:
            Average value or None if not available
        """
        values = self.metrics.get(name, [])
        if not values:
            return None

        if last_n is not None:
            values = values[-last_n:]

        try:
            return sum(values) / len(values)
        except (TypeError, ZeroDivisionError):
            return None

    def clear(self) -> None:
        """Clear all logged metrics."""
        self.metrics.clear()
        self.step = 0

    def summary(self) -> str:
        """Get a summary string of all metrics.

        Returns:
            Multi-line summary of all metrics
        """
        lines = [f"Metrics Summary (Step {self.step}):"]
        for name, values in self.metrics.items():
            if values:
                latest = values[-1]
                avg = self.get_average(name)
                if isinstance(latest, float):
                    lines.append(f"  {name}: latest={latest:.4f}, avg={avg:.4f}, count={len(values)}")
                else:
                    lines.append(f"  {name}: latest={latest}, count={len(values)}")
        return "\n".join(lines)
