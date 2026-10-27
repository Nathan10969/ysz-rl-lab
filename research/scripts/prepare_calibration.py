"""Prepare local review inputs and cheap MLIP geometries. Never connects or submits."""
import csv,hashlib,json,sys,time
from pathlib import Path
import numpy as np
import torch
from ase.io import read,write
from ase.optimize import BFGS
from mace.calculators import MACECalculator
from calibration import reference,make_state,canonical_key,shell_counts,symmetry_permutations,job_specs

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'calibration';MODEL=ROOT/'models/mace-small-density-agnesi-stress.model'
REMOTE='/online1/ycsc_hzw/hd100371/Jony/runs/01_calib'
POT_HASH={'Zr_sv':'25aed69cb10325f9d37c5c68912b61a17387d1f8e4f1d804860ffa10c8a4bf76',
          'Y_sv':'5e7f7496c6fa99024fec326bcd60dcee8ff7e886f9613d31da35c685a538a036',
          'O':'818f92134a0a090dccd8ba1447fa70422a3b330e708bb4f08108d8ae51209ddf'}
BINARY_HASH='f51016d290697d9c63e27feb71a278f7127309f6e23696cd6d68c3f1c43a4159'


def physical(state):
    a=state[state.numbers!=0];a.set_array('site_id',np.flatnonzero(state.numbers!=0))
    order=sorted(range(len(a)),key=lambda i:{'Zr':0,'Y':1,'O':2}[a[i].symbol])
    return a[order]


def make_calc(device,dtype):
    if device=='mps':
        model=torch.load(MODEL,map_location='cpu',weights_only=False).float()
        return MACECalculator(models=model,device=device,default_dtype=dtype)
    return MACECalculator(model_paths=str(MODEL),device=device,default_dtype=dtype)


def relax(state,calc,cid,save=True):
    a=physical(state);a.calc=calc;calc.reset();t=time.perf_counter()
    initial=float(a.get_potential_energy())
    dyn=BFGS(a,logfile=str(OUT/'geometries'/f'{cid}.log') if save else None,maxstep=.1)
    converged=dyn.run(fmax=.03,steps=300)
    elapsed=time.perf_counter()-t
    if not converged:raise RuntimeError(f'{cid}: MLIP relaxation did not converge; no job inputs emitted')
    meta={'id':cid,'n_atoms':len(a),'n_sites':len(state),'n_y':int(np.sum(state.numbers==39)),
          'canonical_key':canonical_key(state,2 if len(state)==96 else 3),
          'initial_energy_eV':initial,'relaxed_energy_eV':float(a.get_potential_energy()),
          'relax_seconds':elapsed,'steps':dyn.nsteps,'converged':bool(converged),
          'max_force_eV_A':float(np.linalg.norm(a.get_forces(),axis=1).max()),
          **shell_counts(state)}
    if save:
        write(OUT/'geometries'/f'{cid}_initial.extxyz',state)
        write(OUT/'geometries'/f'{cid}_relaxed.extxyz',a)
    print(cid,'MLIP relaxed',dyn.nsteps,'steps',round(elapsed,2),'s',flush=True)
    return a,meta


def batch_script(j):
    seconds=round(j['limit_hours']*3600);hh=seconds//3600;mm=(seconds%3600)//60
    checks='\n'.join(f"printf '%s  %s\\n' '{h}' \"$P/{p}/POTCAR\" | sha256sum -c -" for p,h in POT_HASH.items())
    return f'''#!/bin/bash
# Use only within the latest user-relayed phase approval; PMI2 required on this platform.
#SBATCH --partition=q_amd_share
#SBATCH --job-name=ysz_{j['id']}
#SBATCH --nodes=1
#SBATCH --ntasks={j['cores']}
#SBATCH --cpus-per-task=1
#SBATCH --mem={j['cores']*1500}M
#SBATCH --time={hh:02d}:{mm:02d}:00
#SBATCH --no-requeue
#SBATCH --chdir={REMOTE}/{j['id']}
#SBATCH --output=slurm-%j.out
set -euo pipefail
module purge
module load amd/intel_compiler/2022u2
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
VASP="$HOME/online1/Jony/software/vasp.6.5.1/bin/vasp_std"
P="$HOME/online1/Jony/software/potpaw_vasp/PBE"
printf '%s  %s\\n' '{BINARY_HASH}' "$VASP" | sha256sum -c -
{checks}
cat "$P/Zr_sv/POTCAR" "$P/Y_sv/POTCAR" "$P/O/POTCAR" > POTCAR
sha256sum INCAR KPOINTS POSCAR POTCAR > runtime_input_hashes.sha256
date -Is
export I_MPI_PMI_LIBRARY=/opt/hpc/slurm/21.08.6/lib/libpmi2.so
test -r "$I_MPI_PMI_LIBRARY"
srun --mpi=pmi2 --ntasks="$SLURM_NTASKS" --cpu-bind=cores "$VASP"
date -Is
# COMPLETED / exit 0 is not a substitute for SCF and ionic convergence checks.
'''


def main():
    t0=time.perf_counter();torch.set_num_threads(4)
    reuse='--reuse-geometries' in sys.argv
    cached=json.loads((OUT/'configurations.json').read_text()) if reuse else []
    (OUT/'geometries').mkdir(parents=True,exist_ok=True)
    (OUT/'01_calib').mkdir(exist_ok=True)
    pool=read(ROOT/'results/random_100_lattice.extxyz',index=':')
    energies=np.genfromtxt(ROOT/'results/random_energies_cpu.csv',delimiter=',',names=True)['energy_eV']
    shell=[shell_counts(s) for s in pool]
    selected=list(range(4))
    selected += [int(i) for i in np.argsort(energies) if i not in selected][:4]
    # NN/NNN boundary group, not an unmeasured Ewald-disagreement claim.
    for reverse in [False,True]:
        ranks=sorted(range(100),key=lambda i:(shell[i]['Y_vac_NN'],i),reverse=reverse)
        selected += [i for i in ranks if i not in selected][:2]
    assert len(selected)==len(set(selected))==12
    states={f'S{i:02d}':pool[idx] for i,idx in enumerate(selected)}
    states.update({'L00':make_state(20261023,3,16),'L01':make_state(20261024,3,16)})
    assert len({canonical_key(states[f'S{i:02d}'],2) for i in range(12)})==12
    cpu=make_calc('cpu','float64');geometry={};records=[]
    for cid,state in states.items():
        if reuse:
            a=read(OUT/'geometries'/f'{cid}_relaxed.extxyz')
            meta=next(row.copy() for row in cached if row['id']==cid)
        else:
            a,meta=relax(state,cpu,cid)
        meta['preparation_device']='cpu';meta['preparation_dtype']='float64'
        if cid.startswith('S'):
            ii=int(cid[1:]);meta.update(pool_index=selected[ii],seed=20261022+selected[ii],
                selection_group=['random','unrelaxed_MACE_low','NN_boundary'][ii//4],
                pool_unrelaxed_MACE_energy_eV=float(energies[selected[ii]]))
        else:meta.update(seed=20261023+int(cid[1:]),selection_group='maincell_random')
        geometry[cid]=a;records.append(meta)
        (OUT/'configurations.json').write_text(json.dumps(records,indent=2))
    jobs=job_specs()
    for j in jobs:
        d=OUT/'01_calib'/j['id'];d.mkdir(exist_ok=True)
        a=(physical(states[j['config']]) if j['input_geometry']=='ideal_initial_occupation'
           else geometry[j['config']])
        write(d/'POSCAR',a,format='vasp',direct=True,sort=False,vasp5=True)
        nbands=480 if len(a)==94 else 1536
        kpar=1 if j['kmesh']==[1,1,1] else 2
        incar=f'''SYSTEM = YSZ {j['id']} review draft
GGA = PE
PREC = Accurate
ENCUT = {j['encut']}
EDIFF = 1E-6
ALGO = Normal
NELM = 120
NBANDS = {nbands}
ISMEAR = 0
SIGMA = 0.05
LASPH = .TRUE.
LREAL = {j['lreal']}
ISPIN = 1
ISYM = 0
NCORE = 4
KPAR = {kpar}
ISTART = 0
ICHARG = 2
LWAVE = .FALSE.
LCHARG = .FALSE.
NSW = {200 if j['kind']=='relax' else 8 if j['kind']=='pilot_relax' else 0}
'''
        if j['kind'] in ['relax','pilot_relax']:incar+='IBRION = 2\nISIF = 2\nEDIFFG = -0.02\nPOTIM = 0.3\n'
        (d/'INCAR').write_text(incar)
        k=' '.join(map(str,j['kmesh']));(d/'KPOINTS').write_text(f'Gamma centered\n0\nGamma\n{k}\n0 0 0\n')
        (d/'sub.sh').write_text(batch_script(j))
        mapping={'order':['Zr','Y','O'],'counts':[a.get_chemical_symbols().count(s) for s in ['Zr','Y','O']],
                 'vasp_atom_index_0based_to_parent_site':a.arrays['site_id'].tolist()}
        (d/'atom_mapping.json').write_text(json.dumps(mapping,indent=2))
        j.update(n_atoms=len(a),nbands=nbands,ncore=4,kpar=kpar,
                 review_status='APPROVED_P0' if j['phase']=='P0_cost' else 'HELD_FOR_POST_P0_REVIEW',
                 remote_directory=f'{REMOTE}/{j["id"]}',
                 input_hashes={n:hashlib.sha256((d/n).read_bytes()).hexdigest() for n in ['INCAR','KPOINTS','POSCAR','sub.sh']})
        (d/'job.json').write_text(json.dumps(j,indent=2))
    (OUT/'jobs.json').write_text(json.dumps(jobs,indent=2))
    columns=['id','phase','config','kind','n_atoms','cores','limit_hours','hard_core_hours',
             'estimated_core_hours_low','estimated_core_hours_high','encut','kmesh','lreal','input_geometry','nbands','ncore','kpar','review_status']
    with (OUT/'JOB_LIST.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore');writer.writeheader();writer.writerows(jobs)
    if not reuse:
        # Main-cell CPU and MPS development-only timing; cloud CUDA is not measured here.
        timing={}
        for device,dtype in [('cpu','float64'),('mps','float32')]:
            calc=make_calc(device,dtype);a=physical(states['L00']);a.calc=calc;calc.reset()
            t=time.perf_counter();a.get_potential_energy()
            if device=='mps':torch.mps.synchronize()
            cold=time.perf_counter()-t;warm=[]
            for _ in range(5):
                calc.reset();t=time.perf_counter();a.get_potential_energy()
                if device=='mps':torch.mps.synchronize()
                warm.append(time.perf_counter()-t)
            b,meta=relax(states['L00'],calc,f'L00_timing_{device}',save=False)
            if device=='mps':torch.mps.synchronize()
            timing[device]={'dtype':dtype,'cold_sp_seconds':cold,'warm_sp_seconds':warm,
                           'warm_median_seconds':float(np.median(warm)),'relax':meta,
                           'scope':'local development timing only; cloud production uses original CUDA MACE'}
            (OUT/'maincell_timing.json').write_text(json.dumps(timing,indent=2))
    summary={'jobs':len(jobs),'n_sp':sum(j['kind']=='sp' for j in jobs),'n_relax':sum(j['kind']=='relax' for j in jobs),'n_pilot_relax':sum(j['kind']=='pilot_relax' for j in jobs),
             'hard_core_hours':sum(j['hard_core_hours'] for j in jobs),
             'estimated_core_hours_low':sum(j['estimated_core_hours_low'] for j in jobs),
             'estimated_core_hours_high':sum(j['estimated_core_hours_high'] for j in jobs),
             'smoke_used_core_hours':16*23/3600,'authorized_core_hours':3000,
             'reserve_after_all_time_limits':3000-16*23/3600-sum(j['hard_core_hours'] for j in jobs),
             'local_preparation_seconds':time.perf_counter()-t0,'reused_cached_geometries':reuse,'cloud_jobs_submitted_by_this_script':0,
             'potcar_files_in_package':False,'n_maincell_symmetry_operations':len(symmetry_permutations(3))}
    (OUT/'BUDGET.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()
