"""Prepare the nine approved Claude07 DFT inputs locally. No network/submission.

Existing inputs, outputs, budgets, and Claude06 ledgers are read-only. A repeated
invocation verifies byte-identical preparations and refuses any submitted/output
directory. New roots retain the existing 40 card-hour and 3000 core-hour pools.
"""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

import numpy as np
from ase.io import read, write

from claude07_safety import budget_status, last_complete_calculation, verify_hash_manifest

ROOT = Path(__file__).resolve().parents[1]
REF = 'REVIEW_CLAUDE_07_gates.md'
REMOTE = {'CPU': '/online1/ycsc_hzw/hd100371/Jony/runs/02_calib_c07',
          'GPU': '/CLUSTER_HOME/runs/02_calib_c07'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, data):
    Path(path).write_text(json.dumps(data, indent=2, sort_keys=True)+'\n')


def literal_constant(path, name):
    for node in ast.parse(Path(path).read_text()).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError('Missing verified constant: '+name)


def incar_values(text):
    return {line.split('=', 1)[0].strip().upper(): line.split('=', 1)[1].strip()
            for line in text.splitlines() if '=' in line}


def old_job(root, relative):
    directory = root/relative
    meta = json.loads((directory/'job.json').read_text())
    for name, digest in meta['input_hashes'].items():
        if sha(directory/name) != digest:
            raise ValueError('Previously reviewed C6 input changed: '+str(directory/name))
    return directory, meta


def source_hashes(root, files):
    return {str(Path(p).relative_to(root)): sha(p) for p in files}


def validate_atoms(atoms, n):
    expected = ['Zr']*(92 if n == 316 else 28)+['Y']*(16 if n == 316 else 4)+['O']*(208 if n == 316 else 62)
    if atoms.get_chemical_symbols() != expected:
        raise ValueError('Geometry no longer has the original reviewed VASP atom order')
    distances = atoms.get_all_distances(mic=True)[np.triu_indices(n, 1)]
    if len(atoms) != n or not np.isfinite(atoms.positions).all() or distances.min() <= 1.5:
        raise ValueError('Invalid atom count, nonfinite geometry, or overlapping atoms')


def runtime_script(meta, pot_hashes, binary_path, pot_path):
    gpu = bool(meta['gpus'])
    module = '''set -eo pipefail
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
    checks = '\n'.join("printf '%s  %s\\n' '"+digest+"' \"$P/"+name+"/POTCAR\" | sha256sum -c -"
                       for name, digest in pot_hashes.items())
    run = '#!/bin/bash\n'+module+'sha256sum -c REVIEWED_INPUTS.sha256\n'
    if gpu:
        run += 'source ./select_real_driver.sh\n'
    run += '''VASP="%s"
P="%s"
printf '%%s  %%s\\n' '%s' "$VASP" | sha256sum -c -
%s
cat "$P/Zr_sv/POTCAR" "$P/Y_sv/POTCAR" "$P/O/POTCAR" > POTCAR
printf '%%s  %%s\\n' '%s' POTCAR | sha256sum -c -
sha256sum INCAR KPOINTS POSCAR POTCAR > runtime_input_hashes.sha256
python3 - "$VASP" <<'FINGERPRINT'
import hashlib,json,os,socket,sys
from pathlib import Path
from claude07_safety import atomic_json,validate_allocation
meta=json.loads(Path('job.json').read_text())
validate_allocation(meta,os.environ)
digest=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
result=dict(binary_path=sys.argv[1],binary_sha256=digest(sys.argv[1]),
            input_sha256={n:digest(n) for n in ['INCAR','KPOINTS','POSCAR','POTCAR']},
            potcar_component_sha256=meta['potcar_component_sha256'],
            mpi_ranks=meta['mpi_ranks'],cores=meta['cores'],gpus=meta['gpus'],
            mpi_command=meta['mpi_command'],hostname=socket.gethostname(),
            slurm_job_id=os.environ['SLURM_JOB_ID'],
            cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
            allocation={k:v for k,v in os.environ.items() if k.startswith('SLURM_')})
if result['binary_sha256']!=meta['binary_sha256']:raise SystemExit('Binary fingerprint changed')
if result['input_sha256']['POTCAR']!=meta['combined_potcar_sha256']:raise SystemExit('POTCAR fingerprint changed')
atomic_json(Path('RUNTIME_FINGERPRINT.json'),result)
FINGERPRINT
date -Is
scontrol show job "$SLURM_JOB_ID" > allocation.txt
''' % (binary_path, pot_path, meta['binary_sha256'], checks, meta['combined_potcar_sha256'])
    if gpu:
        run += '''test -n "${CUDA_VISIBLE_DEVICES:-}"
nvidia-smi --id="$CUDA_VISIBLE_DEVICES" --query-gpu=timestamp,index,uuid,memory.used,memory.total,utilization.gpu --format=csv -l 10 > gpu_samples.csv &
YSZ_SAMPLE_PID=$!
trap 'kill "$YSZ_SAMPLE_PID" 2>/dev/null || true; wait "$YSZ_SAMPLE_PID" 2>/dev/null || true' EXIT
mpirun -np 1 --host "$(hostname):1" --map-by slot:PE=6 --bind-to core --report-bindings "$VASP" > log.txt 2>&1
'''
    else:
        run += 'srun --mpi=pmi2 --ntasks="$SLURM_NTASKS" --cpu-bind=cores "$VASP" > log.txt 2>&1\n'
    return run+'python3 claude07_safety.py check-result .\ndate -Is\n'


def batch_script(meta):
    seconds = meta['limit_seconds']
    hours, rem = divmod(seconds, 3600)
    minutes, sec = divmod(rem, 60)
    resources = '''#SBATCH --partition=a100
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=90G
''' if meta['gpus'] else '''#SBATCH --partition=q_amd_share
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --mem=48000M
'''
    return '''#!/bin/bash
# Claude07 approved DFT work only; per-task failure, no retries.
# Resource afterany lanes are supplied by the future submission step.
#SBATCH --nodes=1
%s#SBATCH --time=%02d:%02d:%02d
#SBATCH --no-requeue
#SBATCH --job-name=ysz_%s
#SBATCH --chdir=%s
#SBATCH --output=slurm-%%j.out
set -euo pipefail
python3 claude07_safety.py claim .
yszt_finish() {
  YSZ_EXIT_CODE=$?
  trap - EXIT
  python3 claude07_safety.py finish . "$YSZ_EXIT_CODE" || true
  exit "$YSZ_EXIT_CODE"
}
trap yszt_finish EXIT
timeout --signal=TERM --kill-after=10s %ss bash run.sh
''' % (resources, hours, minutes, sec, meta['name'], meta['remote_directory'], seconds-60)


def make_job(root, pool_root, spec, context):
    pool, group, config, name, seconds = spec
    gpu = pool == 'GPU'
    relax = group in {'G1', 'G2'}
    previous_name = 'c06_L00_relax_fromideal' if group == 'G1' else 'c06_P2b_'+config if gpu else 'c06_S08_k333'
    previous_rel = ('calibration_gpu' if gpu else 'calibration')+'/execution_claude06/'+previous_name
    previous, old = old_job(root, previous_rel)
    directory = pool_root/'execution'/name
    directory.mkdir(parents=True)
    provenance_files = [previous/'POSCAR', previous/'INCAR', previous/'KPOINTS', previous/'atom_mapping.json', previous/'job.json']
    parent = ('calibration/claude06_pool/'+config+'_initial.extxyz' if config.startswith('B')
              else 'calibration/geometries/'+config+'_initial.extxyz')
    mace = ('calibration/claude06_pool/'+config+'_relaxed.extxyz' if config.startswith('B')
            else 'calibration/geometries/'+config+'_relaxed.extxyz')
    if group == 'G1':
        source = root/'calibration_gpu/claude06_outputs/c06_L00_relax_fromideal/vasprun.xml'
        frame = context['frame108']
        atoms = read(previous/'POSCAR')
        if atoms.get_chemical_symbols() != frame['symbols']:
            raise ValueError('XML atominfo no longer matches original VASP order')
        atoms.set_cell(frame['basis'], scale_atoms=False)
        atoms.set_scaled_positions(frame['positions'])
        write(directory/'POSCAR', atoms, format='vasp', direct=True, sort=False)
        shutil.copyfile(previous/'atom_mapping.json', directory/'atom_mapping.json')
        provenance_files.append(source)
        source_kind = 'last_complete_XML_calculation_108_no_CONTCAR_fallback'
    elif gpu:
        source = root/mace
        atoms = read(source)
        old_atoms = read(previous/'POSCAR')
        validate_atoms(atoms, 316)
        if not (np.allclose(atoms.positions, old_atoms.positions, atol=1e-10, rtol=0)
                and np.allclose(atoms.cell.array, old_atoms.cell.array, atol=1e-12, rtol=0)):
            raise ValueError('MACE file differs from the original Gamma geometry: '+config)
        mapping = json.loads((previous/'atom_mapping.json').read_text())
        if atoms.arrays['site_id'].tolist() != mapping['vasp_atom_index_0based_to_parent_site']:
            raise ValueError('MACE/VASP atom mapping changed: '+config)
        # Copy original Gamma POSCAR bytes only after verifying the extxyz source.
        for filename in ['POSCAR', 'atom_mapping.json']:
            shutil.copyfile(previous/filename, directory/filename)
        provenance_files.append(source)
        source_kind = 'frozen_MACE_CPU_float64_relaxed_original_VASP_order'
    else:
        source = previous/'POSCAR'
        atoms = read(source)
        for filename in ['POSCAR', 'atom_mapping.json']:
            shutil.copyfile(previous/filename, directory/filename)
        source_kind = 'exact_C6_S08_k333_geometry_and_physics'
    validate_atoms(atoms, 316 if gpu else 94)
    kmesh = [2, 2, 2] if group == 'G3' else old['kmesh']
    binary = 'vasp_std' if group == 'G3' or not gpu else 'vasp_gam'
    binary_hash = context['gpu_binary_hashes'][binary] if gpu else old['binary_sha256']
    incar = incar_values((previous/'INCAR').read_text())
    old_physics = dict(incar)
    incar['SYSTEM'] = 'YSZ '+name+' approved Claude07'
    if relax:
        incar.update(NSW='200', ISIF='2', IBRION='2', POTIM='0.3', EDIFFG='-0.02')
    else:
        incar['NSW'] = '0'
    (directory/'INCAR').write_text(''.join(k+' = '+v+'\n' for k, v in incar.items()))
    if group == 'G3':
        (directory/'KPOINTS').write_text('Gamma centered\n0\nGamma\n2 2 2\n0 0 0\n')
    else:
        shutil.copyfile(previous/'KPOINTS', directory/'KPOINTS')
    differences = {k: dict(before=old_physics.get(k), after=v) for k, v in incar.items()
                   if k != 'SYSTEM' and old_physics.get(k) != v}
    expected = {'NSW', 'ISIF', 'IBRION', 'POTIM', 'EDIFFG'} if group == 'G2' else set()
    if set(differences) != expected:
        raise ValueError('Unapproved INCAR physics changes: '+str(differences))
    if not gpu and binary_hash != context['cpu_binary_hash']:
        raise ValueError('CPU binary hash differs from verified C6')
    index = len(context['jobs'][pool])
    meta = dict(name=name, pool=pool, group=group, phase=group, config=config,
                approval_reference=REF, n_atoms=len(atoms), kind='relax' if relax else 'sp',
                source_directory=str(source.parent.relative_to(root)), geometry_reference=str(source.relative_to(root)),
                input_geometry=source_kind, parent_geometry=parent, mace_geometry=mace,
                previous_job=previous_name, previous_input_directory=previous_rel,
                local_directory=('calibration_gpu' if gpu else 'calibration')+'/claude07/execution/'+name,
                remote_directory=REMOTE[pool]+'/'+name, host='your-cluster' if gpu else 'hzw-cpu',
                gpus=int(gpu), cores=6 if gpu else 32, mpi_ranks=1 if gpu else 32,
                memory='90G' if gpu else '48000M', kmesh=kmesh, binary=binary, binary_sha256=binary_hash,
                encut=520, lreal='Auto', ediff=1e-5, limit_seconds=seconds,
                watchdog_seconds=seconds-60, cleanup_window_seconds=60, limit_hours=seconds/3600,
                cap=seconds/3600 if gpu else 32*seconds/3600,
                pool_authorization='existing_DFT_40_card_hours' if gpu else 'existing_CPU_3000_core_hours',
                depends_on=[], lane=index % 2, resource_afterany=[context['jobs'][pool][index-2]['name']] if index >= 2 else [],
                dependency_policy='afterany_for_resource_lanes_only; afterok_for_true_science_dependencies_only',
                no_retries=True, combined_potcar_sha256=context['combined_potcar_sha256'],
                potcar_component_sha256=context['pot_hashes'],
                mpi_command='mpirun -np 1 --host "$(hostname):1" --map-by slot:PE=6 --bind-to core --report-bindings' if gpu
                else 'srun --mpi=pmi2 --ntasks=32 --cpu-bind=cores',
                physical_differences_from_C6=dict(INCAR=differences, kmesh=dict(before=old['kmesh'], after=kmesh),
                                                  binary=dict(before=old['binary'], after=binary),
                                                  geometry='XML_calculation_108' if group == 'G1' else 'identical'))
    if group == 'G1':
        meta.update(restart_frame_1based=108, restart_source_job_id='596', previous_completed_ionic_steps=108,
                    previous_elapsed_seconds=context['previous_elapsed_seconds'],
                    previous_card_hours=context['previous_elapsed_seconds']/3600,
                    restart_is_geometry_only=True, electronic_restart='ISTART0_ICHARG2_no_WAVECAR_or_CHGCAR')
    shutil.copyfile(root/'scripts/claude07_safety.py', directory/'claude07_safety.py')
    if gpu:
        shutil.copyfile(previous/'select_real_driver.sh', directory/'select_real_driver.sh')
    provenance_files += [root/parent, root/mace, root/REF, root/'scripts/claude07_safety.py', root/'scripts/prepare_claude07.py']
    save(directory/'PROVENANCE.json', dict(source_sha256=source_hashes(root, provenance_files),
                                          source_kind=source_kind, original_VASP_atom_order_preserved=True,
                                          CONTCAR_used=False))
    binary_path = '/data/software/app/vasp.6.5.1/bin/'+binary if gpu else '$HOME/online1/Jony/software/vasp.6.5.1/bin/vasp_std'
    pot_path = '$HOME/Jony/software/potpaw_vasp/PBE' if gpu else '$HOME/online1/Jony/software/potpaw_vasp/PBE'
    (directory/'run.sh').write_text(runtime_script(meta, context['pot_hashes'], binary_path, pot_path))
    (directory/'sub.sh').write_text(batch_script(meta))
    for filename in ['run.sh', 'sub.sh']:
        subprocess.run(['/bin/bash', '-n', str(directory/filename)], check=True, capture_output=True)
    meta['input_hashes'] = {p.name: sha(p) for p in sorted(directory.iterdir())}
    save(directory/'job.json', meta)
    (directory/'REVIEWED_INPUTS.sha256').write_text(''.join(sha(p)+'  '+p.name+'\n' for p in sorted(directory.iterdir())))
    verify_hash_manifest(directory)
    entry = dict(meta, input_hashes={p.name: sha(p) for p in sorted(directory.iterdir())})
    context['jobs'][pool].append(entry)


def ensure_pristine(root):
    for path in root.rglob('*'):
        if path.name in {'submitted_job_id.txt', 'RUN_CLAIM.json', 'TASK_STATUS.json', 'OUTCAR', 'CONTCAR',
                          'vasprun.xml', 'OSZICAR', 'log.txt', 'RESULT_CHECK.json', 'runtime_input_hashes.sha256',
                          'RUNTIME_FINGERPRINT.json', 'CLAUDE07_SUBMISSIONS.json', 'CLAUDE07_POOL_STOP.json'} or path.name.startswith('slurm-'):
            raise ValueError('Existing submission/output/control state; reconcile without preparing again: '+str(path))
    ledger = json.loads((root/'PREPARED_LEDGER.json').read_text())
    if ledger.get('submitted_jobs') or ledger.get('pending_submission'):
        raise ValueError('Preparation ledger already has submission state')


def install_or_verify(stage, destination):
    if destination.exists():
        ensure_pristine(destination)
        existing = {str(p.relative_to(destination)): sha(p) for p in destination.rglob('*') if p.is_file()}
        expected = {str(p.relative_to(stage)): sha(p) for p in stage.rglob('*') if p.is_file()}
        if existing != expected:
            raise ValueError('Existing preparation differs; refusing overwrite: '+str(destination))
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=str(destination.parent), prefix='.claude07-prepare-') as temporary:
        payload = Path(temporary)/'payload'
        shutil.copytree(stage, payload)
        os.rename(payload, destination)


def prepare(source_root=ROOT, output_root=None):
    root = Path(source_root).resolve()
    output = Path(output_root).resolve() if output_root is not None else root
    gpu_budget = json.loads((root/'calibration_gpu/GPU_BUDGET.json').read_text())
    cpu_budget = json.loads((root/'calibration/BUDGET.json').read_text())
    old_lists = {pool: json.loads((root/relative/'CLAUDE06_JOB_LIST.json').read_text())
                 for pool, relative in [('CPU', 'calibration'), ('GPU', 'calibration_gpu')]}
    if len(old_lists['GPU']) != 25 or len(old_lists['CPU']) != 20:
        raise ValueError('Unexpected C6 manifest; reconcile first')
    previous = root/'calibration_gpu/claude06_outputs/c06_L00_relax_fromideal'
    if (previous/'submitted_job_id.txt').read_text().strip() != '596':
        raise ValueError('L00 source is not reviewed job 596')
    frame = last_complete_calculation((previous/'vasprun.xml').read_text(), 108)
    diagnosis = json.loads((root/'calibration/evidence/Claude06_L00_timeout_diagnosis.json').read_text())
    if diagnosis['job_id'] != '596' or diagnosis['completed_ionic_outputs_for_diagnosis_only'] != 108:
        raise ValueError('L00 historical accounting/source mismatch')
    old_hashes = {line.split()[1]: line.split()[0]
                  for line in (previous/'runtime_input_hashes.sha256').read_text().splitlines()}
    context = dict(frame108=frame, previous_elapsed_seconds=diagnosis['elapsed_seconds'],
                   gpu_binary_hashes=literal_constant(root/'scripts/prepare_claude06.py', 'GPU_BINARY_HASH'),
                   cpu_binary_hash=literal_constant(root/'scripts/prepare_calibration.py', 'BINARY_HASH'),
                   pot_hashes=literal_constant(root/'scripts/prepare_calibration.py', 'POT_HASH'),
                   combined_potcar_sha256=old_hashes['POTCAR'], jobs={'CPU': [], 'GPU': []})
    specs = [('GPU', 'G1', 'L00', 'c07_G1_L00_continue108', 10800)]
    specs += [('GPU', 'G2', c, 'c07_G2_'+c+'_relax', 18000) for c in ['B10', 'B11', 'B15', 'B00']]
    specs += [('GPU', 'G3', c, 'c07_G3_'+c+'_k222', 5400) for c in ['B10', 'B07', 'B00']]
    specs += [('CPU', 'C1', 'S08', 'c07_C1_S08_k333', 3600)]
    with tempfile.TemporaryDirectory(prefix='claude07-local-') as temporary:
        stage = Path(temporary)
        for pool, group, config, name, seconds in specs:
            make_job(root, stage/pool, (pool, group, config, name, seconds), context)
        snapshots = {'GPU': (gpu_budget['actual_card_hours'], 40, 27.5, 'calibration_gpu/GPU_BUDGET.json'),
                     'CPU': (cpu_budget['cumulative_actual_core_hours'], 3000, 32, 'calibration/BUDGET.json')}
        for pool, (historical, limit, round_limit, budget_file) in snapshots.items():
            jobs = context['jobs'][pool]
            status = budget_status(jobs, {}, historical, limit, round_limit)
            if not status['safe']:
                raise ValueError('Budget reservation rejected: '+str(status))
            home = stage/pool
            save(home/'JOB_LIST.json', jobs)
            save(home/'BUDGET_RESERVATION.json', dict(status, resource_pool=pool,
                 existing_authorization='DFT40_card_hours' if pool == 'GPU' else 'CPU3000_core_hours',
                 new_authorization_added=0, historical_actual=historical, budget_snapshot=budget_file,
                 budget_snapshot_sha256=sha(root/budget_file), max_concurrent=2,
                 no_borrowing_from_training_smoke_pool=True, local_preparation_only=True))
            save(home/'PREPARED_LEDGER.json', dict(approval_reference=REF, resource_pool=pool,
                 state='PREPARED_NOT_SUBMITTED', submitted_jobs=[], prepared_jobs=[j['name'] for j in jobs],
                 max_concurrent=2, resource_dependency='afterany', no_retries=True))
            save(home/'PHYSICS_AUDIT.json', {j['name']: j['physical_differences_from_C6'] for j in jobs})
        frame_report = {k: v for k, v in frame.items() if k not in {'positions', 'forces', 'basis', 'calculation_xml', 'symbols'}}
        frame_report.update(source='calibration_gpu/claude06_outputs/c06_L00_relax_fromideal/vasprun.xml',
                            source_sha256=sha(previous/'vasprun.xml'), source_job_id='596', CONTCAR_used=False,
                            max_force_eV_A=float(np.linalg.norm(frame['forces'], axis=1).max()))
        save(stage/'GPU/L00_RESTART_SOURCE.json', frame_report)
        (stage/'GPU/L00_calculation108.xml').write_text(frame['calculation_xml']+'\n')
        # Verify every existing root before installing either new root.
        for pool, relative in [('CPU', 'calibration'), ('GPU', 'calibration_gpu')]:
            destination = output/relative/'claude07'
            if destination.exists():
                ensure_pristine(destination)
        for pool, relative in [('CPU', 'calibration'), ('GPU', 'calibration_gpu')]:
            install_or_verify(stage/pool, output/relative/'claude07')
    return dict(cpu_jobs=1, gpu_jobs=8, cpu_cap_core_hours=32, gpu_cap_card_hours=27.5,
                submitted=False, uploaded=False, network_used=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, help='Alternative local output root for tests/review')
    args = parser.parse_args()
    print(json.dumps(prepare(ROOT, args.output_root), indent=2))
