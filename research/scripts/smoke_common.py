"""Small shared helpers for the separately budgeted Claude07 smoke jobs."""
import hashlib
import importlib.metadata
import json
import os
import resource
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SMOKE = ROOT / "training_smoke"
MODEL_SHA = "d5773bf9440e96d6eb8c598f84bd0e6369fcfa432f626a87f890e07da3c651c9"


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def versions(names):
    result = {}
    for name in names:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def require_cuda_job(torch):
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("GPU work requires a Slurm job; login-node/local runs are forbidden")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Smoke jobs require exactly one visible CUDA GPU")
    torch.set_num_threads(min(6, int(os.environ.get("SLURM_CPUS_PER_TASK", "6"))))
    return {"job_id": os.environ["SLURM_JOB_ID"], "hostname": os.uname().nodename,
            "gpu": torch.cuda.get_device_name(0), "torch_cuda_build": torch.version.cuda,
            "cpus": torch.get_num_threads(), "visible_gpus": torch.cuda.device_count()}


def resources(torch):
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return {"cpu_user_seconds": usage.ru_utime, "cpu_system_seconds": usage.ru_stime,
            "peak_cpu_rss_KiB_linux": usage.ru_maxrss,
            "cuda_allocated_bytes": torch.cuda.memory_allocated(),
            "cuda_reserved_bytes": torch.cuda.memory_reserved(),
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved()}


def distribution(values):
    import numpy as np
    return {"n": len(values), "p50_seconds": float(np.percentile(values, 50)),
            "p95_seconds": float(np.percentile(values, 95)),
            "total_seconds": float(sum(values))} if values else {"n": 0}


class Budget:
    def __init__(self, seconds, raw_ef_cap=None):
        self.start = time.monotonic()
        self.seconds = seconds
        self.raw_ef_cap = raw_ef_cap
        self.raw_ef_calls = 0
        self.forward_calls = 0

    def check(self, n=0):
        if time.monotonic() - self.start >= self.seconds:
            raise RuntimeError("Internal wall-time cap reached")
        if self.raw_ef_cap is not None and self.raw_ef_calls + n > self.raw_ef_cap:
            raise RuntimeError("Raw E/F cap reached before next forward")

    def charge(self, n):
        self.check(n)
        self.raw_ef_calls += n
        self.forward_calls += 1


def map_final(atoms, initial):
    """Periodic Hungarian assignment; Y and Zr compete on the same sublattice."""
    import numpy as np
    from ase.geometry import find_mic
    from scipy.optimize import linear_sum_assignment
    from calibration import canonical_key, reference
    parent = reference(3)
    assigned = np.full(len(atoms), -1, dtype=int)
    distances = np.full(len(atoms), np.nan)
    for mask, sites in [(np.isin(atoms.numbers, [39, 40]), np.flatnonzero(parent.numbers == 40)),
                        (atoms.numbers == 8, np.flatnonzero(parent.numbers == 8))]:
        ai = np.flatnonzero(mask)
        delta = atoms.positions[ai, None, :] - parent.positions[sites][None, :, :]
        cost = find_mic(delta.reshape(-1, 3), parent.cell, pbc=True)[1].reshape(len(ai), len(sites))
        ii, jj = linear_sum_assignment(cost ** 2)
        assigned[ai[ii]] = sites[jj]
        distances[ai[ii]] = cost[ii, jj]
    if np.any(assigned < 0) or len(set(assigned.tolist())) != 316:
        raise ValueError("Final occupation mapping is not bijective")
    mapped = parent.copy()
    mapped.numbers[:] = 0
    mapped.numbers[assigned] = atoms.numbers
    return {"canonical_key": canonical_key(mapped, 3),
            "initial_canonical_key": canonical_key(initial, 3),
            "vacancy_site_ids": np.flatnonzero(mapped.numbers == 0).tolist(),
            "Y_site_ids": np.flatnonzero(mapped.numbers == 39).tolist(),
            "max_assignment_distance_A": float(distances.max()),
            "assigned_site_ids": assigned.tolist()}
