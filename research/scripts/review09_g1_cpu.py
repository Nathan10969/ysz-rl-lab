"""Explicitly authorized Review09 G1 CPU diagnostic; no Slurm or GPU calls."""
import json
import time
from pathlib import Path
import numpy as np
import torch
from ase.io import read, write
from ase.optimize import BFGS
from mace.calculators import MACECalculator
from smoke_common import MODEL_SHA, map_final, save_json, sha256

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'calibration/claude07/evidence/review09_G1_CPU'


def main():
    OUT.mkdir(exist_ok=False)
    torch.set_num_threads(2);torch.set_default_dtype(torch.float64)
    model=ROOT/'models/mace-small-density-agnesi-stress.model'
    if sha256(model)!=MODEL_SHA:raise ValueError('Frozen physical model changed')
    sources={'DFT_final':'calibration_gpu/claude07/outputs/c07_G1_L00_continue108/vasprun.xml',
             'MACE_final':'calibration/geometries/L00_relaxed.extxyz',
             'ideal':'calibration/geometries/L00_initial.extxyz'}
    ideal=read(ROOT/sources['ideal']);mace=read(ROOT/sources['MACE_final'])
    dft=read(ROOT/sources['DFT_final'],index=-1)
    calc=MACECalculator(model_paths=str(model),device='cpu',default_dtype='float64',enable_cueq=False,enable_oeq=False)
    count={'raw_ef':0}
    def hook(module,args):count['raw_ef']+=1
    token=calc.models[0].register_forward_pre_hook(hook)
    result={'device':'cpu','GPU_work_performed':False,'no_GPU_pool_charge':True,
            'model_sha256':MODEL_SHA,'source_sha256':{k:sha256(ROOT/v) for k,v in sources.items()},
            'protocol':{'optimizer':'BFGS','fmax':.03,'steps':300,'maxstep':.1,'dtype':'float64'},'SP':{},'relaxations':{}}
    started=time.monotonic()
    try:
        for name,cell in [('MACE_final',mace),('DFT_final',dft)]:
            a=cell.copy();a.calc=calc
            f=a.get_forces();e=a.get_potential_energy()
            if not np.isfinite(f).all() or not np.isfinite(e):raise RuntimeError('Nonfinite diagnostic SP')
            result['SP'][name]={'energy_eV':float(e),'fmax_eV_A':float(np.linalg.norm(f,axis=1).max()),'mapping':map_final(a,ideal)}
        save_json(OUT/'progress.json',result)
        for name,cell in [('from_DFT_final',dft),('from_ideal',ideal[ideal.numbers!=0])]:
            a=cell.copy();a.calc=calc;before=count['raw_ef'];t=time.monotonic()
            opt=BFGS(a,logfile=str(OUT/(name+'.log')),trajectory=str(OUT/(name+'.traj')),maxstep=.1)
            converged=bool(opt.run(fmax=.03,steps=300))
            f=a.get_forces();e=float(a.get_potential_energy())
            if not np.isfinite(f).all() or not np.isfinite(e):raise RuntimeError('Nonfinite diagnostic relaxation')
            write(OUT/(name+'.extxyz'),a)
            result['relaxations'][name]={'converged':converged,'steps':opt.nsteps,'energy_eV':e,
                'fmax_eV_A':float(np.linalg.norm(f,axis=1).max()),'raw_ef':count['raw_ef']-before,
                'wall_seconds':time.monotonic()-t,'mapping':map_final(a,ideal)}
            save_json(OUT/'progress.json',result)
        result['status']='completed'
    except Exception as error:
        result.update(status='failed',error=str(error));raise
    finally:
        token.remove();result.update(raw_ef=count['raw_ef'],wall_seconds=time.monotonic()-started)
        save_json(OUT/'result.json',result)
    print(json.dumps({'status':result['status'],'raw_ef':result['raw_ef'],'wall_seconds':result['wall_seconds']}))


if __name__=='__main__':main()
