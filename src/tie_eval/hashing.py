from __future__ import annotations

import numpy as np

GOLDEN = np.uint64(0x9E3779B97F4A7C15)
FINALIZER_1 = np.uint64(0xFF51AFD7ED558CCD)
FINALIZER_2 = np.uint64(0xC4CEB9FE1A85EC53)


def mixed_uint64_keys(
    candidates: np.ndarray,
    user_indices: np.ndarray,
    seed: int,
) -> np.ndarray:
    """Return the archived 64-bit mixer output before floating-point conversion.

    Arithmetic intentionally wraps modulo 2**64.  The function is independent of
    candidate position: its inputs are only the declared seed, user identity, and
    item identity.
    """
    candidates = np.asarray(candidates, dtype=np.uint64)
    users = np.asarray(user_indices, dtype=np.uint64)
    if candidates.ndim != 2:
        raise ValueError("candidates must have shape [rows, candidates]")
    if users.ndim != 1 or len(users) != len(candidates):
        raise ValueError("user_indices must have one entry per candidate row")

    with np.errstate(over="ignore"):
        z = (
            np.uint64(seed)
            ^ (users[:, None] * GOLDEN)
            ^ (candidates * (GOLDEN >> np.uint64(1)))
        )
        z = (z ^ (z >> np.uint64(33))) * FINALIZER_1
        z = (z ^ (z >> np.uint64(33))) * FINALIZER_2
        z = z ^ (z >> np.uint64(33))
    return z.astype(np.uint64, copy=False)


def archived_float32_keys(
    candidates: np.ndarray,
    user_indices: np.ndarray,
    seed: int,
) -> np.ndarray:
    """Reproduce the paper run's historical float32 secondary keys exactly.

    Converting 64-bit hash values to float32 can create secondary-key collisions.
    The artifact retains this function only to regenerate the archived table.
    New evaluations should use :func:`mixed_uint64_keys` with an item-ID fallback.
    """
    keys = mixed_uint64_keys(candidates, user_indices, seed)
    return (keys.astype(np.float64) / np.float64(2**64)).astype(np.float32)
