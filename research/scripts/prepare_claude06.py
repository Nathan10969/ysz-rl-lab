"""Build exactly the newly released inputs. Never submits or modifies old inputs."""
import argparse,datetime,hashlib,json,shutil,subprocess
from pathlib import Path
import numpy as np
from ase.io import read,write
from prepare_calibration import POT_HASH,BINARY_HASH

ROOT=Path(__file__).resolve().parents[1];C=ROOT/'calibration';G=ROOT/'calibration_gpu'
REF='REVIEW_CLAUDE_06_release.md'
CPU_REMOTE='/online1/ycsc_hzw/hd100371/Jony/runs/01_calib'
GPU_REMOTE='/CLUSTER_HOME/runs/01_calib_gpu'
GPU_BINARY_HASH={'vasp_gam':'deb79c3ffcef4faa1e05df3d9c704ff532ac9de9cdde581a6911a5b2ee3ee326',
                 'vasp_std':'e5dd34c611648a60e4dc8542d92c0e461c935ed38765138147ef060a8ee864bf'}

def save(path,data):path.write_text(json.dumps(data,indent=2)+'\n')
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def write_job(name,config,phase,source,gpu=False,relax=False,lreal='Auto',encut=520,kmesh=None,ideal=False):
    home=G if gpu else C;d=home/'execution_claude06'/name
    assert not d.exists(),'Never overwrite existing job inputs: '+str(d)
    d.mkdir(parents=True)
    for fn in ['POSCAR','atom_mapping.json']:shutil.copy2(source/fn,d/fn)
    atoms=read(d/'POSCAR');n=len(atoms);assert n in [94,316]
    counts=[28,4,62] if n==94 else [92,16,208]
    assert atoms.get_chemical_symbols()==['Zr']*counts[0]+['Y']*counts[1]+['O']*counts[2]
    assert atoms.get_all_distances(mic=True)[np.triu_indices(n,1)].min()>1.5
    kmesh=kmesh or ([2,2,2] if n==94 else [1,1,1])
    binary='vasp_gam' if gpu and kmesh==[1,1,1] else 'vasp_std'
    (d/'KPOINTS').write_text('Gamma centered\n0\nGamma\n'+' '.join(map(str,kmesh))+'\n0 0 0\n')
    incar=f'''SYSTEM = YSZ {name} approved Claude06
GGA = PE
PREC = Accurate
ENCUT = {encut}
EDIFF = 1E-5
ALGO = Normal
NELM = 120
NBANDS = {480 if n==94 else 1536}
ISMEAR = 0
SIGMA = 0.05
LASPH = .TRUE.
LREAL = {lreal}
ISPIN = 1
ISYM = 0
NCORE = {1 if gpu else 4}
KPAR = {1 if gpu else 2}
ISTART = 0
ICHARG = 2
LWAVE = .FALSE.
LCHARG = .FALSE.
AMIX = 0.2
BMIX = 0.0001
NSW = {200 if relax else 0}
'''
    if relax:incar+='IBRION = 2\nISIF = 2\nEDIFFG = -0.02\nPOTIM = 0.3\n'
    (d/'INCAR').write_text(incar)
    shutil.copy2(ROOT/'scripts/claude06_safety.py',d/'claude06_safety.py')
    (d/'check_result.py').write_text('''import json,sys
from pathlib import Path
from claude06_safety import validate_output
r=validate_output(Path('vasprun.xml').read_text(errors='replace'),Path('OUTCAR').read_text(errors='replace'),bool(int(sys.argv[1])))
Path('RESULT_CHECK.json').write_text(json.dumps(r,indent=2)+'\\n')
print(json.dumps(r))
if not r['valid']:raise SystemExit('SCF or ionic convergence failed; no automatic retry')
''')
    remote=(GPU_REMOTE if gpu else CPU_REMOTE)+'/'+name
    pot='$HOME/Jony/software/potpaw_vasp/PBE' if gpu else '$HOME/online1/Jony/software/potpaw_vasp/PBE'
    vasp='/data/software/app/vasp.6.5.1/bin/'+binary if gpu else '$HOME/online1/Jony/software/vasp.6.5.1/bin/vasp_std'
    hashbin=GPU_BINARY_HASH[binary] if gpu else BINARY_HASH
    checks='\n'.join("printf '%s  %s\\n' '"+h+"' \"$P/"+p+"/POTCAR\" | sha256sum -c -" for p,h in POT_HASH.items())
    module='''set -eo pipefail
ulimit -s unlimited
ulimit -l unlimited
source /etc/profile
module purge
module load mkl/latest nvhpc/25.3
set -u
export YSZ_EXPECTED_GPUS=1
export OMP_NUM_THREADS=6
export OMP_PROC_BIND=spread
export OMP_PLACES=threads
export MKL_NUM_THREADS=6
''' if gpu else '''set -euo pipefail
module purge
module load amd/intel_compiler/2022u2
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export I_MPI_PMI_LIBRARY=/opt/hpc/slurm/21.08.6/lib/libpmi2.so
test -r "$I_MPI_PMI_LIBRARY"
'''
    if gpu:shutil.copy2(G/'execution_claude05/07r_L01_1g_driverfix/select_real_driver.sh',d/'select_real_driver.sh')
    run='#!/bin/bash\n'+module+f'''test ! -e ../CLAUDE06_STOP.json
sha256sum -c REVIEWED_INPUTS.sha256
'''+('source ./select_real_driver.sh\n' if gpu else '')+f'''
VASP="{vasp}"
P="{pot}"
printf '%s  %s\\n' '{hashbin}' "$VASP" | sha256sum -c -
{checks}
cat "$P/Zr_sv/POTCAR" "$P/Y_sv/POTCAR" "$P/O/POTCAR" > POTCAR
sha256sum INCAR KPOINTS POSCAR POTCAR > runtime_input_hashes.sha256
date -Is
scontrol show job "$SLURM_JOB_ID"
'''
    if gpu:
        run+='''test -n "${CUDA_VISIBLE_DEVICES:-}"
nvidia-smi --id="$CUDA_VISIBLE_DEVICES" --query-gpu=timestamp,index,uuid,memory.used,memory.total,utilization.gpu --format=csv -l 10 > gpu_samples.csv &
SAMPLE_PID=$!
trap 'kill "$SAMPLE_PID" 2>/dev/null || true; wait "$SAMPLE_PID" 2>/dev/null || true' EXIT
mpirun -np 1 --host "$(hostname):1" --map-by slot:PE=6 --bind-to core --report-bindings "$VASP" > log.txt 2>&1
'''
    else:run+='srun --mpi=pmi2 --ntasks="$SLURM_NTASKS" --cpu-bind=cores "$VASP" > log.txt 2>&1\n'
    run+=f'python3 check_result.py {int(relax)}\ndate -Is\n';(d/'run.sh').write_text(run)
    seconds=14340 if gpu and relax else 900 if gpu else 1800
    hh,rem=divmod(seconds,3600);mm,ss=divmod(rem,60)
    resources='''#SBATCH --partition=a100
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=90G
''' if gpu else '''#SBATCH --partition=q_amd_share
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --mem=48000M
'''
    batch=f'''#!/bin/bash
# Claude06 release only. No retries, no training, no dual GPU.
#SBATCH --nodes=1
{resources}#SBATCH --time={hh:02d}:{mm:02d}:{ss:02d}
#SBATCH --no-requeue
#SBATCH --job-name=ysz_{name}
#SBATCH --chdir={remote}
#SBATCH --output=slurm-%j.out
set -eu
if test -e ../CLAUDE06_STOP.json; then echo 'Batch halted; do not run VASP';exit 90;fi
set +e
timeout --signal=TERM --kill-after=10s {seconds-60}s bash run.sh
YSZ_EXIT_CODE=$?
if test "$YSZ_EXIT_CODE" -ne 0; then
  printf '{{"job_id":"%s","job_name":"%s","exit_code":%s}}\\n' "$SLURM_JOB_ID" '{name}' "$YSZ_EXIT_CODE" > ../CLAUDE06_STOP.json
fi
exit "$YSZ_EXIT_CODE"
''';(d/'sub.sh').write_text(batch)
    names=['POSCAR','INCAR','KPOINTS','atom_mapping.json','run.sh','claude06_safety.py','check_result.py']
    if gpu:names.append('select_real_driver.sh')
    (d/'REVIEWED_INPUTS.sha256').write_text(''.join(sha(d/fn)+'  '+fn+'\n' for fn in names))
    for fn in ['sub.sh','run.sh']:
        subprocess.run(['/bin/bash','-n',str(d/fn)],check=True)
    job=dict(name=name,config=config,phase=phase,approval_reference=REF,n_atoms=n,
             kind='relax' if relax else 'sp',input_geometry='ideal_initial_occupation' if ideal else 'frozen_MACE_CPU_float64_relaxed',
             source_directory=str(source.relative_to(ROOT)),remote_directory=remote,host='your-cluster' if gpu else 'hzw-cpu',
             gpus=int(gpu),cores=6 if gpu else 32,mpi_ranks=1 if gpu else 32,kmesh=kmesh,
             encut=encut,lreal=lreal,ediff=1e-5,binary=binary,binary_sha256=hashbin,
             limit_seconds=seconds,limit_hours=seconds/3600,cap=seconds/3600 if gpu else 32*seconds/3600,
             input_hashes={f.name:sha(f) for f in d.iterdir()})
    save(d/'job.json',job)
    return job

def base():
    assert (ROOT/REF).exists()
    assert not (C/'execution_claude06').exists() and not (G/'execution_claude06').exists()
    archive=ROOT/'archive/Claude06_before_execution';archive.mkdir()
    for src in [C/'BUDGET.json',C/'REVIEW_STATUS.json',G/'GPU_BUDGET.json',ROOT/'PLAN.md',ROOT/'CLAUDE05_RESULTS.md']:
        shutil.copy2(src,archive/src.name)
    cpu=[]
    for i in range(1,12):
        cid=f'S{i:02d}';old=next((C/'01_calib').glob(f'*_{cid}_sp16'))
        cpu.append(write_job(f'c06_{i+4:02d}_{cid}_sp',cid,'P1_small',old))
    for cid in ['S01','S04','S08']:
        old=next((C/'01_calib').glob(f'*_{cid}_sp16'))
        cpu.append(write_job(f'c06_{cid}_lreal_false',cid,'P1_lreal_pair',old,lreal='.FALSE.'))
    for cid in ['S00','S04','S08']:
        for variant in ['encut650','k333']:
            old=next((C/'01_calib').glob(f'*_{cid}_{variant}'))
            cpu.append(write_job(f'c06_{cid}_{variant}',cid,'P2_convergence',old,
                                 encut=650 if variant=='encut650' else 520,kmesh=[3,3,3] if variant=='k333' else [2,2,2]))
    assert len(cpu)==20 and sum(j['cap'] for j in cpu)==320
    save(C/'CLAUDE06_JOB_LIST.json',cpu)
    gpu=[]
    for jid in ['31_S02_relax_fromideal','32_S06_relax_fromideal']:
        cid=jid.split('_')[1]
        gpu.append(write_job('c06_'+jid,cid,'crosssite',C/'01_calib'/jid,gpu=True,relax=True,ideal=True))
    for cid in ['L00','L01']:
        src=G/'claude06_ideal_sources'/cid;src.mkdir(parents=True)
        state=read(C/'geometries'/f'{cid}_initial.extxyz');a=state[state.numbers!=0]
        ids=np.flatnonzero(state.numbers!=0);order=sorted(range(len(a)),key=lambda i:{'Zr':0,'Y':1,'O':2}[a[i].symbol]);a=a[order]
        write(src/'POSCAR',a,format='vasp',direct=True,sort=False)
        save(src/'atom_mapping.json',dict(order=['Zr','Y','O'],counts=[92,16,208],vasp_atom_index_0based_to_parent_site=ids[order].tolist()))
        gpu.append(write_job('c06_'+cid+'_relax_fromideal',cid,'crosssite',src,gpu=True,relax=True,ideal=True))
    for jid in ['24_S00_relax','25_S01_relax','26_S04_relax','28_S08_relax','30_L00_pilot8']:
        cid=jid.split('_')[1];name='c06_'+jid.replace('pilot8','fullrelax')
        gpu.append(write_job(name,cid,'full_relax_cost',C/'01_calib'/jid,gpu=True,relax=True))
    assert len(gpu)==9
    save(G/'CLAUDE06_JOB_LIST.json',gpu)
    save(C/'CLAUDE06_PLAN.json',dict(approval_reference=REF,cpu_job_count=20,cpu_batch_cap_core_hours=320,
         cpu_historical_core_hours=253.68,gpu_historical_card_hours=.07416666666666666,
         gpu_job_count=25,gpu_sp_count=16,gpu_sp_group_cap=4,gpu_relax_count=9,gpu_relax_each_cap=239/60,
         gpu_total_projected=.07416666666666666+4+9*239/60,cpu_max_concurrent=2,gpu_max_concurrent=8,
         note='CPU P2 uses latest explicit32cores30min6jobs=96coreh, replacing older192 estimate; GPU relaxation239min leaves history/headroom inside40. No training authorized.'))
    status=json.loads((C/'REVIEW_STATUS.json').read_text());status.update(approval_reference=REF,
         claude_review='CLAUDE06_RELEASED_CPU_AND_40_GPU_CARD_HOURS',p0_state='CLAUDE06_PREPARING',
         p1_state='APPROVED_CLAUDE06',p2_and_later='P2_P2b_CROSSSITE_FULLRELAX_APPROVED_ONLY',
         gpu_authorized_card_hours=40,gpu_further_submissions_allowed=True,further_submissions_allowed=True,
         further_submissions_scope='Exactly20CPUand25singleGPUClaude06jobs; no08s,noLLMtraining',
         max_concurrent_cpu=2,max_concurrent_gpu=8,dual_gpu_state='CANCELLED_DRAFT_NEVER_SUBMITTED_CLAUDE06',
         recovery_release_required=False,awaiting='Prepare16independentmaincellMACEgeometries;runreleasedClaude06only',
         updated_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    save(C/'REVIEW_STATUS.json',status)
    print(json.dumps(dict(CPU=len(cpu),GPU_relax=len(gpu),GPU_SP_pending=16,submitted=False)))

def pool():
    existing=json.loads((G/'CLAUDE06_JOB_LIST.json').read_text());assert len(existing)==9
    records=json.loads((C/'claude06_pool/configurations.json').read_text());assert len(records)==16
    added=[]
    for record in records:
        cid=record['id'];a=read(C/'claude06_pool'/f'{cid}_relaxed.extxyz')
        src=G/'claude06_pool_sources'/cid;src.mkdir(parents=True)
        write(src/'POSCAR',a,format='vasp',direct=True,sort=False)
        save(src/'atom_mapping.json',dict(order=['Zr','Y','O'],counts=[92,16,208],
              vasp_atom_index_0based_to_parent_site=a.arrays['site_id'].tolist()))
        added.append(write_job('c06_P2b_'+cid,cid,'P2b_maincell',src,gpu=True))
    save(G/'CLAUDE06_JOB_LIST.json',added+existing)
    print(json.dumps(dict(GPU_total=len(added+existing),P2b=len(added),submitted=False)))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--base',action='store_true');parser.add_argument('--pool',action='store_true');args=parser.parse_args()
    if args.base:base()
    elif args.pool:pool()
    else:parser.error('Choose--baseor--pool')
