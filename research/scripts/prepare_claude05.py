"""Prepare Claude05's exact CPU reference and serial GPU recovery, without submission."""
import datetime,hashlib,json,shutil,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];C=ROOT/'calibration';G=ROOT/'calibration_gpu'
REF='REVIEW_CLAUDE_05_fast_gpu.md'

def hashes(d,names):return {name:hashlib.sha256((d/name).read_bytes()).hexdigest() for name in names}
def finish(d,meta):
    for name in ['sub.sh','run.sh','select_real_driver.sh']:
        if (d/name).exists():subprocess.run(['/bin/bash','-n',str(d/name)],check=True)
    meta['input_hashes']=hashes(d,[p.name for p in d.iterdir() if p.is_file()])
    (d/'job.json').write_text(json.dumps(meta,indent=2)+'\n')
    return meta

def main():
    assert (ROOT/REF).is_file()
    assert not (C/'execution_claude05').exists() and not (G/'execution_claude05').exists(),'Existing inputs: reconcile, never overwrite'
    archive=ROOT/'archive/Claude05_before_execution';archive.mkdir()
    for name in ['PLAN.md','P0_RESULTS.md','GPU_PILOT_RESULTS.md','BUDGET_REPLAN.md']:
        shutil.copy2(ROOT/name,archive/name)
    for name in ['BUDGET.json','REVIEW_STATUS.json','P0_ANALYSIS.json']:
        shutil.copy2(C/name,archive/name)
    shutil.copy2(G/'GPU_SUBMISSIONS.json',archive/'GPU_SUBMISSIONS.json')
    src=C/'execution_claude03/03q_L00_gamma64';d=C/'execution_claude05/09_L00_ref';d.mkdir(parents=True)
    oldmeta=json.loads((src/'job.json').read_text())
    for name,want in oldmeta['input_hashes'].items():assert hashlib.sha256((src/name).read_bytes()).hexdigest()==want
    for name in ['POSCAR','KPOINTS','atom_mapping.json']:shutil.copy2(src/name,d/name)
    (d/'INCAR').write_text((src/'INCAR').read_text()+'AMIX = 0.2\nBMIX = 0.0001\n')
    batch=(src/'sub.sh').read_text().replace('03q_L00_gamma64','09_L00_ref').replace('--ntasks=64','--ntasks=128').replace('--mem=96000M','--mem=192000M').replace('--time=01:00:00','--time=01:30:00')
    batch=batch.replace('# Claude03: remaining P0 approved; P1 conditional on converged 03/04 each <=40 core-hours. P2+ held.',
                        '# Claude05: one CPU L00 reference only; P1/P2 remain held.')
    batch=batch.replace('srun --mpi=pmi2','timeout --signal=TERM --kill-after=10s 5300s srun --mpi=pmi2')
    batch=batch.replace('set -euo pipefail\n','set -euo pipefail\nsha256sum -c REVIEWED_INPUTS.sha256\n')
    (d/'sub.sh').write_text(batch)
    reviewed=hashes(d,['POSCAR','INCAR','KPOINTS','atom_mapping.json'])
    (d/'REVIEWED_INPUTS.sha256').write_text(''.join(f'{v}  {k}\n' for k,v in reviewed.items()))
    cpu=finish(d,dict(oldmeta,name='09_L00_ref',id='ref_L00',logical_job='ref_L00',config='L00',
                    phase='CLAUDE05_reference',cores=128,limit_hours=1.5,hard_core_hours=192,
                    approval_reference=REF,remote_directory='/online1/ycsc_hzw/hd100371/Jony/runs/01_calib/09_L00_ref',
                    lreal='.FALSE.',ediff=1e-6,amix=.2,bmix=.0001,binary='vasp_std'))
    (C/'CLAUDE05_JOB_LIST.json').write_text(json.dumps([cpu],indent=2)+'\n')
    helper=r'''#!/bin/bash
# Claude05: filter first; use LD_PRELOAD only if the filtered-driver probe fails.
set -euo pipefail
export LD_LIBRARY_PATH=$(python3 - <<'FILTER'
import os,re
parts=os.environ.get('LD_LIBRARY_PATH','').split(':')
keep=[p for p in parts if p and 'stubs' not in p.lower() and not
      ('hpc_sdk' in p and re.search(r'/cuda(?:/[^/]+)?/lib64(?:/|$)',p))]
print(':'.join(keep))
FILTER
)
yszt_probe_driver() {
  python3 - "$1" "$2" <<'PROBE'
import ctypes,json,os,sys
destination=sys.argv[1];expected=int(sys.argv[2]);result={}
try:
    driver=ctypes.CDLL('libcuda.so.1')
    code=driver.cuInit(0);result['cuInit_code']=code
    if code:raise RuntimeError('cuInit code %d'%code)
    count=ctypes.c_int();code=driver.cuDeviceGetCount(ctypes.byref(count))
    if code:raise RuntimeError('cuDeviceGetCount code %d'%code)
    result['visible_GPU_count']=count.value
    paths=sorted({line.split()[-1] for line in open('/proc/self/maps') if '/libcuda.so' in line})
    result['loaded_driver_paths']=paths
    if not paths or any('stub' in p.lower() for p in paths):raise RuntimeError('No verified non-stub driver mapping')
    if count.value!=expected:raise RuntimeError('GPU count differs from the approved allocation')
    result['passed']=True
except Exception as error:
    result.update(passed=False,error=str(error))
with open(destination,'w') as f:json.dump(result,f,indent=2)
print(json.dumps(result))
sys.exit(0 if result['passed'] else 1)
PROBE
}
if yszt_probe_driver cuda_preflight_filtered.json "$YSZ_EXPECTED_GPUS"; then
  # Preserve the SDK's MPI/math libraries; the real driver directory still precedes
  # the executable RUNPATH, which may contain the SDK's stub-bearing CUDA directory.
  YSZ_DRIVER_DIRECTORY=$(python3 - <<'DIR'
import json,os
print(os.path.dirname(json.load(open('cuda_preflight_filtered.json'))['loaded_driver_paths'][0]))
DIR
)
  export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}$YSZ_DRIVER_DIRECTORY"
  printf 'Filtered CUDA driver precheck passed; no LD_PRELOAD used\n'
else
  YSZ_REAL_DRIVER=""
  YSZ_CUDA_CANDIDATES=(/usr/lib/x86_64-linux-gnu/libcuda.so.1 /lib/x86_64-linux-gnu/libcuda.so.1 /usr/lib64/libcuda.so.1)
  while IFS= read -r yszt_candidate; do
    if test -n "$yszt_candidate"; then YSZ_CUDA_CANDIDATES+=("$yszt_candidate"); fi
  done < <(ldconfig -p 2>/dev/null | awk '/libcuda.so.1 / {print $NF}')
  for yszt_candidate in "${YSZ_CUDA_CANDIDATES[@]}"; do
    if ! test -r "$yszt_candidate"; then continue; fi
    yszt_resolved=$(readlink -f "$yszt_candidate")
    case "$yszt_resolved" in *stub*|*STUB*) continue ;; esac
    YSZ_REAL_DRIVER="$yszt_resolved";break
  done
  if test -z "$YSZ_REAL_DRIVER"; then echo 'No real driver; exit before VASP' >&2;exit 70;fi
  export LD_PRELOAD="$YSZ_REAL_DRIVER${LD_PRELOAD:+:$LD_PRELOAD}"
  if ! yszt_probe_driver cuda_preflight_preload.json "$YSZ_EXPECTED_GPUS"; then
    echo 'CUDA prechecks failed; exit before VASP, no automatic retry' >&2;exit 70
  fi
fi
'''
    gpu=[]
    originals=json.loads((G/'JOB_LIST.json').read_text())
    for meta,name in zip(originals,['07r_L01_1g_driverfix','08r_L01_2g_driverfix']):
        src=G/'inputs'/meta['name'];d=G/'execution_claude05'/name;d.mkdir(parents=True)
        for n in ['POSCAR','INCAR','KPOINTS','atom_mapping.json']:shutil.copy2(src/n,d/n)
        (d/'select_real_driver.sh').write_text(helper)
        run=(src/'run.sh').read_text().replace('set -u\n',f'set -u\nexport YSZ_EXPECTED_GPUS={meta["gpus"]}\n',1)
        run=run.replace('sha256sum -c REVIEWED_INPUTS.sha256\n','sha256sum -c REVIEWED_INPUTS.sha256\nsource ./select_real_driver.sh\n',1)
        (d/'run.sh').write_text(run)
        batch=(src/'sub.sh').read_text().replace(meta['name'],name).replace('--time=01:00:00','--time=00:59:00').replace('3540s','3480s')
        (d/'sub.sh').write_text(batch)
        reviewed=hashes(d,['POSCAR','INCAR','KPOINTS','atom_mapping.json','run.sh','select_real_driver.sh'])
        (d/'REVIEWED_INPUTS.sha256').write_text(''.join(f'{v}  {k}\n' for k,v in reviewed.items()))
        job=finish(d,dict(meta,name=name,approval_reference=REF,limit_hours=59/60,
                         hard_card_hours=meta['gpus']*59/60,
                         guarded_runtime_seconds=3480,
                         remote_directory='/CLUSTER_HOME/runs/01_calib_gpu/'+name,
                         submission_rule='07r now;08r only after07r COMPLETED and independent SCF/GPU checks pass'))
        gpu.append(job)
    (G/'CLAUDE05_JOB_LIST.json').write_text(json.dumps(gpu,indent=2)+'\n')
    now=datetime.datetime.now(datetime.timezone.utc).isoformat()
    b=json.loads((C/'BUDGET.json').read_text());b.update(approval_reference=REF,reference_hard_core_hours=192,
         reference_actual_core_hours=0,remaining_reference_core_hour_caps=192,
         actual_plus_remaining_caps=b['actual_plus_remaining_caps']+192)
    assert b['actual_plus_remaining_caps']<=3000
    (C/'BUDGET.json').write_text(json.dumps(b,indent=2)+'\n')
    gb=json.loads((G/'GPU_BUDGET.json').read_text());gb.update(approval_reference=REF,
        claude05_rounded_cumulative_cap_card_hours=2.9511,
        claude05_nominal_caps_exact_card_hours=gb['actual_card_hours']+sum(j['hard_card_hours'] for j in gpu),
        recovery_reserve_card_hours=sum(j['hard_card_hours'] for j in gpu),
        remaining_card_hour_caps=sum(j['hard_card_hours'] for j in gpu),
        actual_plus_remaining_caps=gb['actual_card_hours']+sum(j['hard_card_hours'] for j in gpu),
        review_rounding_note='2.951111111 nominal is2.9511 rounded; internal3480s watchdog keeps actual strictly below2.9511 including prior4seconds',
        recovery_authorized=True,no_retries=True,updated_at_utc=now)
    assert gb['actual_plus_remaining_caps']<=3
    (G/'GPU_BUDGET.json').write_text(json.dumps(gb,indent=2)+'\n')
    s=json.loads((C/'REVIEW_STATUS.json').read_text());s.update(approval_reference=REF,
        claude_review='CLAUDE05_REFERENCE_AND_GPU_RECOVERY_APPROVED',p0_state='CLAUDE05_PREPARED',
        relative_energy_tolerance_eV_cell=.02,max_force_tolerance_eV_A=.02,
        gpu_energy_tolerance_eV_cell=.01,gpu_force_tolerance_eV_A=.02,
        cpu_ref_allowed=True,gpu_07r_allowed=True,gpu_08r_requires_completed_07r_SCF=True,
        further_submissions_scope='Exactly09_L00_ref,07r;08r onlyafter07r passes; P1/P2 held',
        later_phases='P1_P2_AND_LATER_HELD_CLAUDE05',p1_state='HELD_CLAUDE05',p2_and_later='HELD',
        recovery_release_required=False,awaiting='Submit reviewed09 and07r; later08r conditional only',updated_at_utc=now)
    (C/'REVIEW_STATUS.json').write_text(json.dumps(s,indent=2)+'\n')
    print(json.dumps({'CPU':[cpu['name']],'GPU':[j['name'] for j in gpu],'CPU_projected_core_hours':b['actual_plus_remaining_caps'],'GPU_nominal_cumulative_card_hour_cap':gb['actual_plus_remaining_caps'],'submitted':False},indent=2))

if __name__=='__main__':main()
