"""CPU/MPS feasibility, not a DFT accuracy benchmark. Uses one frozen checkpoint."""
import argparse, csv, hashlib, json, platform, time, traceback
from pathlib import Path
import numpy as np
import torch
from ase.io import write
from ase.optimize import BFGS
from mace.calculators import MACECalculator
from lattice import make_state, physical_atoms, canonical_key, permutations

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results'
MODEL = ROOT/'models/mace-small-density-agnesi-stress.model'

def sync(device):
    if device == 'mps':
        torch.mps.synchronize()


def run_device(device, states):
    result = {'device':device, 'dtype':'float32', 'cpu_threads':4}
    t0 = time.perf_counter()
    # Stored checkpoint is float64; Apple MPS needs conversion on CPU before transfer.
    model_arg = {'models': torch.load(MODEL, map_location='cpu', weights_only=False).float()} if device == 'mps' else {'model_paths': str(MODEL)}
    calc = MACECalculator(**model_arg, device=device, default_dtype='float32',
                          enable_cueq=False, enable_oeq=False)
    sync(device)
    result['load_seconds'] = time.perf_counter()-t0
    atom = physical_atoms(states[0]); atom.calc = calc
    t0 = time.perf_counter(); e = float(atom.get_potential_energy());sync(device)
    result['cold_evaluation_seconds'] = time.perf_counter()-t0
    result['cold_energy_eV'] = e
    repeat = []
    for _ in range(5):
        calc.reset();sync(device); t0=time.perf_counter()
        atom.get_potential_energy();sync(device)
        repeat.append(time.perf_counter()-t0)
    result['warm_single_point_seconds'] = repeat
    result['warm_median_seconds'] = float(np.median(repeat))
    rows = []
    for i,state in enumerate(states):
        atoms = physical_atoms(state);atoms.calc = calc
        calc.reset();sync(device);t0 = time.perf_counter()
        en=float(atoms.get_potential_energy());forces=atoms.get_forces();sync(device)
        rows.append({'id':i,'seed':20261022+i,'device':device,'energy_eV':en,
                     'fmax_eV_A':float(np.linalg.norm(forces,axis=1).max()),
                     'seconds':time.perf_counter()-t0})
        if i % 20 == 0: print(device,'random',i,flush=True)
    with (OUT/f'random_energies_{device}.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    energy=np.array([x['energy_eV'] for x in rows])
    result['random_100']={'n':len(rows),'seconds':sum(x['seconds'] for x in rows),
                          'min_eV':float(energy.min()),'max_eV':float(energy.max()),
                          'mean_eV':float(energy.mean()),'std_eV':float(energy.std()),
                          'quantiles_eV':np.quantile(energy,[0,.25,.5,.75,1]).tolist(),
                          'geometry':'unrelaxed ideal lattice; all compositions identical'}
    atom = physical_atoms(states[0]);atom.calc=calc;calc.reset()
    dyn=BFGS(atom,logfile=str(OUT/f'relax_{device}.log'),trajectory=str(OUT/f'relax_{device}.traj'),maxstep=.1)
    sync(device);t0=time.perf_counter()
    converged=dyn.run(fmax=.05,steps=200)
    sync(device);elapsed=time.perf_counter()-t0
    result['relax']={'seconds':elapsed,'converged':bool(converged),'steps':dyn.nsteps,
                     'fmax_eV_A':float(np.linalg.norm(atom.get_forces(),axis=1).max()),
                     'energy_eV':float(atom.get_potential_energy()),'cell':'fixed',
                     'max_steps':200,'target_fmax_eV_A':.05}
    write(OUT/f'relaxed_{device}.extxyz',atom)
    print(device,'done',result['warm_median_seconds'],result['relax'],flush=True)
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--device',choices=['cpu','mps'],required=True)
    args=parser.parse_args();OUT.mkdir(exist_ok=True)
    torch.set_num_threads(4);torch.set_default_dtype(torch.float32)
    states=[make_state(20261022+i) for i in range(100)]
    keys=[canonical_key(s) for s in states]
    write(OUT/'random_100_lattice.extxyz',states)
    write(OUT/'random_100_physical.extxyz',[physical_atoms(s) for s in states])
    meta={'model_sha256':hashlib.sha256(MODEL.read_bytes()).hexdigest(),
          'torch':torch.__version__,'platform':platform.platform(),
          'mps_available':torch.backends.mps.is_available(),
          'symmetry_operations':len(permutations()),'symmetry_unique_random':len(set(keys)),
          'n_lattice_sites':96,'n_real_atoms':94,'formula':'Zr28Y4O62',
          'a_conventional_A':5.14,'seed_base':20261022}
    (OUT/'probe_metadata.json').write_text(json.dumps(meta,indent=2))
    try:result=run_device(args.device,states)
    except Exception as e:
        result={'device':args.device,'error':repr(e),'traceback':traceback.format_exc()}
        print(result['traceback'],flush=True)
    (OUT/f'probe_{args.device}.json').write_text(json.dumps(result,indent=2))
    if 'error' in result:raise SystemExit(1)

if __name__ == '__main__':main()
