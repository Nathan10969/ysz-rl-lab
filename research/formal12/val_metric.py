"""Validation metric for Review12.3: mean over starts of the best-of-4 MACE energy drop (eV), with a bootstrap SE.

Usage: python val_metric.py JOB_NAME ...   (reads jobs/<name>/out/result.json; works for LLM evals and SA search)
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent


def drops(result):
    if 'groups' in result:
        out = []
        for g in result['groups']:
            best = g['best_energy_eV']; best = min(best) if isinstance(best, list) else best
            out.append(g['start_energy_eV'] - best)
        return np.array(out)
    by_start = {}
    for c in result['chains']:
        by_start.setdefault(c['start'], []).append(c['initial_energy_eV'] - c['best_energy_eV'])
    return np.array([max(v) for _, v in sorted(by_start.items())])


def metric(name):
    r = json.loads((ROOT / 'jobs' / name / 'out' / 'result.json').read_text())
    d = drops(r)
    rng = np.random.default_rng(0)
    boot = [rng.choice(d, len(d)).mean() for _ in range(2000)]
    return dict(job=name, status=r['status'], n=len(d), mean_drop_eV=round(float(d.mean()), 3),
                se_eV=round(float(np.std(boot)), 3), per_start=[round(float(x), 3) for x in d],
                lowest_eV=round(r['terminals'][0]['energy_eV'], 3) if r.get('terminals') else None,
                config=r.get('config'))


def paired(candidate, reference):
    """Paired per-start comparison on the same val starts (common random numbers). Accept rule (META_LOG):
    mean improvement >= max(0.10 eV, 1 paired bootstrap SE)."""
    a = np.array(metric(candidate)['per_start']); b = np.array(metric(reference)['per_start'])
    d = a - b
    rng = np.random.default_rng(1)
    se = float(np.std([rng.choice(d, len(d)).mean() for _ in range(4000)]))
    threshold = max(.10, se)
    return dict(candidate=candidate, reference=reference, mean_improvement_eV=round(float(d.mean()), 3),
                paired_se_eV=round(se, 3), threshold_eV=round(threshold, 3), accept=bool(d.mean() >= threshold),
                per_start_diff=[round(float(x), 3) for x in d])


if __name__ == '__main__':
    args = sys.argv[1:]
    if args and args[0] == '--paired':
        print(json.dumps(paired(args[1], args[2])))
    else:
        for n in args:
            print(json.dumps(metric(n)))
