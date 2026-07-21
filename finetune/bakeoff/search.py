"""Pure binary search for the largest workable sequence length.

Kept free of torch/unsloth imports so it is unit-testable on CPU with no
model present. The caller injects `fits`, which does the expensive thing.
"""
from typing import Callable


def find_max_seqlen(
    fits: Callable[[int], bool],
    lo: int = 512,
    hi: int = 16384,
    step: int = 512,
) -> int:
    """Largest multiple of `step` in [lo, hi] for which `fits` is True.

    Assumes `fits` is monotonic: if n fits, everything below n fits. Returns
    0 when even `lo` fails. Probes O(log n) times, since each probe may cost
    minutes of GPU time, and never probes the same value twice.

    Raises `ValueError` if `lo` is not a multiple of `step` — the search
    invariant (and the "largest multiple of `step`" contract) only holds
    for an aligned `lo`, and silently rounding would risk wasting hours of
    GPU time on a mis-specified sweep.
    """
    if lo % step != 0:
        raise ValueError(f"lo={lo} must be a multiple of step={step}")

    cache: dict[int, bool] = {}

    def probe(n: int) -> bool:
        if n not in cache:
            cache[n] = fits(n)
        return cache[n]

    if not probe(lo):
        return 0

    best = lo
    low, high = lo, hi
    while low <= high:
        mid = ((low + high) // 2 // step) * step
        if probe(mid):
            best = max(best, mid)
            low = mid + step
        else:
            high = mid - step
    return best
