"""Remote, one-shot v2.2 teacher submission. Standard library only.

Run only in a newly staged recovery_v22/teacher directory after local durable
intent, independent source reviews, real CPU execution, and fee verification.
No preparation, reconciliation, retry, cancellation, or model execution occurs.
An uncertain actual sbatch consumes the exclusive claim permanently.
"""
from __future__ import annotations

import argparse
import copy
from collections import Counter
import datetime
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import uuid


JOB = dict(id='c10_v22_teacher', name='ysz_c10v22_teacher', gpus=1, cpu_cores=6,
    memory_GiB=90, hard_gpu_hours=2, slurm_wall_seconds=7200, outer_timeout_seconds=7140,
    python_work_seconds=6600, raw_ef_call_cap=72240, actual_relaxation_cap=240,
    action_budget=224, seed=2026102301, teacher_trajectories=16, steps_per_trajectory=8)
CAPS = {'teacher':2, 'sft':1.85, 'grpo':2}
BASE_SECONDS = {'633':35, '635':9, '648':36, '650':72, '652':507, '656':153}
CORE = {'claude10_physics.py', 'claude08_env.py', 'smoke_mace.py', 'smoke_grpo08.py'}
PROTOCOL = 'YSZ22_terminal_local2_lbfgs_noise_v2.2'
SELECTED = {'mapping': {'ambiguous_nearest_two_gap_A_strict': 0.1,
             'assignment_tie_break': 'canonical_decorated_geometry_frame_global_feasible_lexicographic_sites',
             'assignment_tie_tolerance_A2': 1e-08,
             'canonical_frame_fractional_decimals': 12,
             'cation_guard': 'unchanged_strict_half_nearest_neighbor_and_occupation_ambiguity',
             'distance_roundoff_A': 1e-14,
             'global_tie_count_field': 'occupation_tie_count',
             'global_tie_metadata_field': 'TIE',
             'near_threshold_assigned_distance_A_strict': 1.1,
             'oxygen_assignment': 'global_Hungarian_squared_MIC_distance',
             'oxygen_equal_cost_occupation': 'quality_metadata_only',
             'oxygen_max_assigned_distance_A_inclusive': 1.8,
             'unmap_stop_denominator': 'physical_action_attempts_excluding_initial_illegal_cache',
             'unmap_stop_min_attempts': 40,
             'unmap_stop_strict_fraction': [1, 10]},
 'protocol': 'YSZ22_terminal_local2_lbfgs_noise_v2.2',
 'relaxation': {'alpha': 70.0,
                'damping': 1.0,
                'displacement_master_seed': 2026102301,
                'displacement_sigma': 0.02,
                'dtype': 'float64',
                'fmax': 0.03,
                'graph_version': 'native_smoke_mace_build_batch_v1',
                'maxstep': 0.1,
                'memory': 100,
                'model_sha': 'd5773bf9440e96d6eb8c598f84bd0e6369fcfa432f626a87f890e07da3c651c9',
                'optimizer': 'LBFGS',
                'steps': 300,
                'use_line_search': False}}
PROTOCOL_SHA = '51c0715ab6565c07b553505d935cee492321b1c7d5df30401500cd3b3f45556b'
CHECKPOINT_SHA256 = 'd5773bf9440e96d6eb8c598f84bd0e6369fcfa432f626a87f890e07da3c651c9'
DRIVER_SHA256 = '03958bf622d142bd00f800356141b34fc0ae861bcb6784567f6bcb468e237142'
HISTORICAL_CLAIM_IDS = {'a348fed9b6234ccfb4c84594377a1daf','eec7f3fcde1044babe12656c56d13bf0'}
PRODUCTION_SOURCES = {'calibration.py','claude08_data.py','claude08_env.py','claude08_grpo.py',
    'claude10_data.py','claude10_physics.py','lattice.py','smoke_common.py','smoke_grpo08.py',
    'smoke_grpo10.py','smoke_mace.py','smoke_sft10.py','smoke_teacher10.py'}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda:handle.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError('Duplicate JSON field: '+key)
        value[key] = item
    return value


def read_json(path):
    return json.loads(Path(path).read_text(), object_pairs_hook=_object,
                      parse_constant=lambda value:(_ for _ in ()).throw(ValueError('Nonfinite JSON value')))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def number(value):
    require(type(value) in (int, float) and math.isfinite(value) and value >= 0, 'Invalid numeric fee')
    return value


def same_number(value, expected):
    return abs(number(value)-expected) <= 1e-12


def relative(root, name):
    require(isinstance(name, str) and re.fullmatch(r'[A-Za-z0-9_./-]+',name), 'Unsafe staged path')
    parts = PurePosixPath(name)
    require(not parts.is_absolute() and '..' not in parts.parts and str(parts) == name, 'Noncanonical staged path')
    result = root/name
    require(result.resolve().is_relative_to(root) and result.is_file() and not result.is_symlink(),
            'Missing or escaped staged file: '+name)
    return result


def checked(root, reference):
    require(isinstance(reference, dict) and set(reference) == {'path','sha256'}, 'A path/hash binding is required')
    path = relative(root, reference['path'])
    require(re.fullmatch(r'[0-9a-f]{64}',str(reference['sha256'])) is not None
            and sha256(path) == reference['sha256'], 'Staged evidence hash changed: '+str(reference['path']))
    return path


def inventory(root):
    """The final list covers every immutable staged file, without self hashes."""
    hashes = {}
    for line in (root/'INPUTS.sha256').read_text().splitlines():
        match = re.fullmatch(r'([0-9a-f]{64}) [ *]([A-Za-z0-9_./-]+)', line)
        require(match is not None, 'Malformed INPUTS.sha256 row')
        digest, name = match.groups()
        require(name not in hashes and name != 'INPUTS.sha256', 'Duplicate or circular input hash')
        require(sha256(relative(root,name)) == digest, 'Staged input changed: '+name)
        hashes[name] = digest
    dynamic = {'SUBMISSION_CLAIM.json','SUBMISSIONS.json','submitted_job_id.txt'}
    actual = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()
              and p.name != 'INPUTS.sha256' and p.relative_to(root).parts[0] != 'submission_checks'
              and p.relative_to(root).as_posix() not in dynamic}
    require(set(hashes) == actual, 'Every staged input must be listed, with no extra files')
    return hashes


def no_previous(root):
    require(root.parts[-4:] == ('training_smoke','claude10','recovery_v22','teacher'),
            'Only the new independent recovery_v22 teacher root is allowed')
    if (root/'SUBMISSIONS.json').exists() or (root/'SUBMISSION_CLAIM.json').exists():
        raise RuntimeError('Existing remote ledger/claim: read-only reconciliation only, never resubmit')
    if (root.parents[2]/'STOP_SMOKE.json').exists():
        raise RuntimeError('Existing smoke pool stop marker; do not clear it')


def verify_fees(root):
    pool = read_json(root/'ROOT_POOL_PRE_SUBMISSION.json')
    package = read_json(root/'PACKAGE_PRE_SUBMISSION.json')
    for budget, cap, actual in ((pool,10,812/3600),(package,6.5,660/3600)):
        require(budget.get('accounting_complete') is True and budget.get('accounting_unknown_job_ids') == []
                and not budget.get('spending_frozen') and not budget.get('collection_risks')
                and budget.get('collection_status') != 'DEFERRED', 'Accounting is not closed')
        require(same_number(budget.get('authorized_card_hours'),cap)
                and same_number(budget.get('actual_card_hours'),actual)
                and same_number(budget.get('remaining_caps'),5.85)
                and same_number(budget.get('actual_plus_remaining_caps'),actual+5.85)
                and actual+5.85 <= cap, 'All original fees and full recovery reservations must remain')
    require(same_number(pool.get('review10_reservation_card_hours'),5.85)
            and package.get('jobs') == [] and same_number(package.get('new_actual_card_hours'),0)
            and package.get('stage_caps') == CAPS
            and same_number(package.get('frozen652_actual_card_hours'),507/3600)
            and same_number(package.get('frozen656_actual_card_hours'),153/3600)
            and same_number(package.get('historical_package_actual_card_hours'),660/3600), 'Package reservation differs before its unique teacher')
    rows = pool.get('jobs')
    require(isinstance(rows,list) and len(rows)==6, 'Exactly six historical parents required')
    seen=set()
    for row in rows:
        sid=str(row.get('slurm_job_id'))
        require(sid in BASE_SECONDS and sid not in seen and row.get('allocation_gpu_count')==1
                and row.get('state') in ('COMPLETED','FAILED','TIMEOUT','CANCELLED','OUT_OF_MEMORY')
                and same_number(row.get('actual_card_hours'),BASE_SECONDS[sid]/3600), 'Historical parent fee changed')
        seen.add(sid)


def verify_scripts(root, manifest):
    expected = dict(nodes='1', partition='a100', ntasks='1', **{'ntasks-per-node':'1',
        'cpus-per-task':'6','gres':'gpu:a100:1','mem':'90G','time':'02:00:00',
        'job-name':JOB['name'],'chdir':str(root),'output':'slurm-%j.out','no-requeue':None})
    found={}
    text=(root/'sub.sh').read_text()
    for line in text.splitlines():
        line=line.lstrip()
        if line.startswith('#SBATCH'):
            option=line[len('#SBATCH'):].strip()
            match=re.fullmatch(r'--([a-z-]+)(?:=(\S+))?',option)
            require(match is not None and match[1] not in found,'Unexpected or duplicate Slurm directive')
            found[match[1]]=match[2]
    require(found==expected,'Actual Slurm script resources differ from frozen teacher')
    commands=[line.strip() for line in text.splitlines() if line.strip() and not line.lstrip().startswith('#')]
    entry='set -euo pipefail'
    timeout='exec timeout --signal=TERM --kill-after=30s 7140s bash run.sh'
    safety=['test -n "${SLURM_JOB_ID:-}"','test ! -e ../../../STOP_SMOKE.json']
    require(text.startswith('#!/bin/bash\n') and commands in ([entry,timeout],[entry]+safety+[timeout]),
            'Only strict shell, exact allocation/stop safety checks and outer timeout may execute')
    require(set(manifest.get('stage_scripts_sha256',{}))=={'sub.sh','run.sh'}, 'Reviewed stage scripts required')
    for name,digest in manifest['stage_scripts_sha256'].items():
        require(sha256(relative(root,name))==digest, 'Stage script differs from reviewed bytes')


def geometry_count(path, frames=1, parent_occupancy=False):
    lines=path.read_text().splitlines()
    cursor=0
    composition={'Zr':92,'Y':16,'O':208}
    if parent_occupancy:
        composition['X']=8
    count=sum(composition.values())
    for _ in range(frames):
        require(cursor<len(lines) and lines[cursor]==str(count) and cursor+count+2<=len(lines),
                'CPU geometry must contain316 physical atoms, plus8empty parent slots when applicable')
        symbols=[row.split()[0] for row in lines[cursor+2:cursor+count+2] if row.split()]
        require(Counter(symbols)==composition,'CPU geometry composition changed')
        cursor+=count+2


def verify_cpu(root, manifest, spec):
    proof=read_json(root/'CPU_PREFLIGHT_VERIFIED.json')
    require(proof.get('independently_verified') is True and proof.get('CPU_only') is True
            and proof.get('same_production_code_path') is True and proof.get('batch_sizes')==[1,4]
            and type(proof.get('raw_ef')) is int and proof['raw_ef']==22
            and type(proof.get('forward_calls')) is int and proof['forward_calls']==13,
            'Real same-path CPU1/4 execution requires22E/F and13forwards')
    require(proof.get('source_sha256')==manifest['source_sha256']
            and proof.get('checkpoint_sha256')==manifest['checkpoint']['sha256']
            and proof.get('protocol_sha256')==manifest['protocol']['sha256']
            and proof.get('input_sha256')==manifest['preflight_input']['sha256']
            and proof.get('driver_sha256')==manifest['driver_sha256']
            and proof.get('driver_binding_mode')=='static_reviewed_bytes_not_executed_in_CPU_preflight',
            'CPU proof bindings changed; CUDA driver is statically reviewed, not CPU-executed')
    evidence=proof.get('evidence_sha256')
    require(isinstance(evidence,dict), 'Full CPU raw evidence hashes required')
    required={'cpu_preflight/result.json','cpu_preflight/actual_model_forwards.jsonl',
              'cpu_preflight/physics_receipts.jsonl'} | {
        f'cpu_preflight/lbfgs_one_step_b{size}_lane{lane}.extxyz' for size in (1,4) for lane in range(size)}
    require(required<=set(evidence), 'Missing real CPU result, audit, or short-relaxation geometries')
    for name,digest in evidence.items():
        checked(root,dict(path=name,sha256=digest))
    raw=read_json(root/'cpu_preflight/result.json')
    core={Path(name).name:digest for name,digest in manifest['source_sha256'].items() if Path(name).name in CORE}
    require(raw.get('status')=='cpu_execution_preflight_passed' and raw.get('production_code_path') is True
            and raw.get('runtime',{}).get('device')=='cpu' and raw['runtime'].get('GPU_work_performed') is False
            and raw.get('checkpoint_sha256')==proof['checkpoint_sha256'] and raw.get('source_hashes')==core
            and raw.get('input_sha256')==proof['input_sha256'] and raw.get('protocol')==PROTOCOL
            and raw.get('relaxation_spec')==spec, 'CPU result is not the frozen production path')
    counters=raw.get('counters',{})
    require(raw.get('short_receipts_for_training') is False
            and type(counters.get('attempts')) is int and counters['attempts']==0
            and type(counters.get('relax_actual')) is int and counters['relax_actual']==0,
            'CPU preflight cannot perform training or issue formal training receipts')
    audit=raw.get('actual_forward_audit',{})
    require(type(counters.get('raw_ef')) is int and counters['raw_ef']==22
            and type(counters.get('forward_calls')) is int and counters['forward_calls']==13
            and audit==dict(observed_raw_ef=22,observed_forward_calls=13,all_precharged=True,matches_counter=True),
            'Raw CPU counts disagree with independent forward audit')
    observed=[json.loads(line) for line in (root/'cpu_preflight/actual_model_forwards.jsonl').read_text().splitlines()]
    sizes=[1]*4+[4]+[1]*4+[1]*2+[4]*2
    require(len(observed)==13,'Exactly13real model forwards required')
    cumulative=0
    for index,(row,size) in enumerate(zip(observed,sizes),1):
        cumulative+=size
        require(type(row.get('batch_size')) is int and row['batch_size']==size
                and row.get('forward_call')==index and row.get('raw_ef_cumulative')==cumulative
                and row.get('precharged') is True,'Raw CPU model forward sequence changed')
    parity=raw.get('parity')
    fields={'batch_single_energy_difference_eV','batch_single_force_difference_eV_A',
            'batch_native_energy_difference_eV','batch_native_force_difference_eV_A'}
    require(isinstance(parity,list) and len(parity)==4,'Four SP/native/batch parity lanes required')
    for lane,row in enumerate(parity):
        require(set(row)==fields|{'lane'} and row['lane']==lane
                and all(number(row[key])<=1e-6 for key in fields),'CPU native/batch1/4 SP parity failed')
    geometry_count(checked(root,manifest['preflight_input']),frames=4,parent_occupancy=True)
    protocol_sha=hashlib.sha256(json.dumps(SELECTED,sort_keys=True,
                                         separators=(',',':'),allow_nan=False).encode()).hexdigest()
    receipts=[json.loads(line,object_pairs_hook=_object) for line in
              (root/'cpu_preflight/physics_receipts.jsonl').read_text().splitlines()]
    require(len(receipts)==13, 'Full thirteen backend SP/short-LBFGS receipts required')
    seen=set()
    for receipt in receipts:
        identity=(receipt.get('operation'),receipt.get('lane'))
        require(identity not in seen and type(identity[0]) is int and type(identity[1]) is int,
                'Unique CPU operation/lane receipt identities required')
        seen.add(identity)
        require(receipt.get('formal_receipt') is False and receipt.get('device')=='cpu'
                and receipt.get('protocol')==PROTOCOL and receipt.get('protocol_sha')==protocol_sha
                and receipt.get('checkpoint_sha256')==proof['checkpoint_sha256']
                and receipt.get('source_hashes')==core and receipt.get('relaxation_spec')==spec,
                'Every raw CPU receipt must bind the current nontraining production protocol')
    singles=[row for row in receipts if row.get('execution_max_steps')==0]
    short=[row for row in receipts if row.get('execution_max_steps')==1]
    require(len(singles)==8 and len(short)==5
            and all(row.get('status')=='single_point' and row.get('nsteps')==0
                    and row.get('raw_ef_attempts')==1 for row in singles)
            and all(row.get('nsteps')==1 and row.get('raw_ef_attempts')==2 for row in short),
            'Eight SP receipts and five genuine one-step LBFGS receipts required')
    for size in (1,4):
        rows=raw.get('relaxations',{}).get(str(size))
        require(isinstance(rows,list) and len(rows)==size,'CPU short LBFGS1/4 runs required')
        for lane,row in enumerate(rows):
            require(row in short and row.get('lane')==lane and row.get('nsteps')==1 and row.get('optimizer')=='LBFGS' and row.get('dtype')=='float64'
                    and row.get('device')=='cpu' and row.get('execution_max_steps')==1
                    and row.get('formal_receipt') is False and row.get('relaxation_spec')==spec
                    and row.get('protocol')==PROTOCOL and row.get('protocol_sha')==protocol_sha
                    and row.get('checkpoint_sha256')==proof['checkpoint_sha256']
                    and row.get('source_hashes')==core,'CPU relaxation is not one real frozen LBFGS step')
            geometry_count(root/f'cpu_preflight/lbfgs_one_step_b{size}_lane{lane}.extxyz')
    return proof


def precheck(root):
    no_previous(root)
    inventory(root)
    manifest=read_json(root/'JOB_MANIFEST.json')
    require(manifest.get('schema')=='review10-v22-teacher-stage-v1' and manifest.get('stage')=='teacher'
            and manifest.get('remote_smoke_root')==str(root) and manifest.get('jobs')==[JOB], 'Wrong fixed teacher manifest')
    auth=read_json(root/'AUTHORIZATION.json')
    require(auth.get('schema')=='review10-v22-authorization-v1'
            and auth.get('direct_user_confirmation') is True
            and auth.get('teacher_submission_released') is True and auth.get('automatic_retry') is False
            and auth.get('formal_training_allowed') is False and auth.get('new_DFT_allowed') is False
            and same_number(auth.get('package_total_cap_including652_656_card_hours'),6.5)
            and type(auth.get('root_pool_card_hours')) is int and auth['root_pool_card_hours']==10
            and type(auth.get('max_concurrent_GPU_jobs')) is int and auth['max_concurrent_GPU_jobs']==1
            and type(auth.get('max_new_teacher_submissions')) is int and auth['max_new_teacher_submissions']==1
            and type(auth.get('additional_GPU_submissions_allowed')) is int and auth['additional_GPU_submissions_allowed']==1
            and type(auth.get('teacher_submissions_used')) is int and auth['teacher_submissions_used']==0
            and same_number(auth.get('historical_package_actual_card_hours'),660/3600)
            and same_number(auth.get('historical_root_actual_card_hours'),812/3600)
            and auth.get('stage_caps_card_hours')==CAPS and auth.get('selected_protocol')==SELECTED
            and auth.get('protocol_sha')==PROTOCOL_SHA,
            'Explicit bounded v2.2 teacher release is required')
    model=manifest.get('checkpoint',{})
    require(set(model)=={'path','sha256'} and isinstance(model.get('path'),str)
            and Path(model['path']).is_absolute() and Path(model['path']).is_file()
            and model.get('sha256')==CHECKPOINT_SHA256 and sha256(model['path'])==model.get('sha256'),'Frozen readable checkpoint hash changed')
    spec=dict(model_sha=model['sha256'],dtype='float64',optimizer='LBFGS',fmax=.03,steps=300,maxstep=.1,
        memory=100,alpha=70.,damping=1.,use_line_search=False,displacement_sigma=.02,
        displacement_master_seed=2026102301,graph_version='native_smoke_mace_build_batch_v1')
    require(read_json(checked(root,manifest.get('protocol')))==SELECTED,
            'Actual v2.2 protocol changed')
    source=manifest.get('source_sha256')
    require(isinstance(source,dict) and set(source)=={'project/scripts/'+name for name in PRODUCTION_SOURCES}
            and auth.get('core_source_sha256')=={Path(p).name:digest for p,digest in source.items()},
            'All thirteen reviewed v22 production sources must match authorization')
    require(set(source)=={p.relative_to(root).as_posix() for p in (root/'project/scripts').rglob('*.py')},
            'Every staged Python source must be independently bound')
    for name,digest in source.items():
        require(name.startswith('project/scripts/') and sha256(relative(root,name))==digest,'Reviewed source changed')
    driver=manifest.get('driver_sha256')
    require(driver=={'select_real_driver.sh':DRIVER_SHA256}
            and sha256(relative(root,'select_real_driver.sh'))==DRIVER_SHA256,
            'Frozen static CUDA driver bytes changed')
    verify_scripts(root,manifest)
    verify_fees(root)
    verify_cpu(root,manifest,spec)
    gate=read_json(root/'PRE_SUBMIT_GATE.json')
    bindings=dict(source_sha256=source,driver_sha256=driver,stage_scripts_sha256=manifest['stage_scripts_sha256'],
        cpu_preflight_verification_sha256=sha256(root/'CPU_PREFLIGHT_VERIFIED.json'),
        checkpoint_sha256=model['sha256'],protocol_sha256=manifest['protocol']['sha256'],wrapper_sha256=sha256(__file__))
    require(gate.get('schema')=='review10-v22-teacher-pre-submit-v1' and gate.get('independently_verified') is True
            and all(gate.get(k)==v for k,v in bindings.items())
            and sha256(root/'submit_once.py')==bindings['wrapper_sha256']
            and gate.get('cpu_preflight_verification_sha256')==sha256(root/'CPU_PREFLIGHT_VERIFIED.json'),
            'Pre-submit independent source/model/protocol/wrapper/CPU binding changed')
    reviews=gate.get('review_evidence',{})
    require(set(reviews)=={'spec','quality'},'Both independent source reviews required')
    for kind,verdict in (('spec','PASS'),('quality','APPROVE')):
        review=read_json(checked(root,reviews[kind]))
        require(review.get('independently_reviewed') is True and review.get('verdict')==verdict
                and all(review.get(k)==v for k,v in bindings.items()),'Independent '+kind+' source review missing or stale')
    intent=read_json(root/'SUBMISSION_INTENT_ID.json')
    require(intent.get('schema')=='review10-v22-local-intent-v1' and intent.get('stage')=='teacher'
            and intent.get('max_submissions')==1 and intent.get('durable_locally') is True
            and re.fullmatch(r'[0-9a-f]{32}',str(intent.get('claim_id'))) is not None
            and intent['claim_id'] not in HISTORICAL_CLAIM_IDS
            and intent['claim_id']==manifest.get('submission_claim_id')
            and intent.get('manifest_sha256')==sha256(root/'JOB_MANIFEST.json'),'Same durable local unique intent required')
    return manifest,intent


def fsync_directory(path):
    descriptor=os.open(path,os.O_RDONLY)
    try: os.fsync(descriptor)
    finally: os.close(descriptor)


def exclusive_json(path,value):
    descriptor=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    with os.fdopen(descriptor,'w') as handle:
        json.dump(value,handle,sort_keys=True,indent=2,allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    fsync_directory(path.parent)


def evidence(root,kind,value):
    directory=root/'submission_checks'
    directory.mkdir(exist_ok=True)
    exclusive_json(directory/(kind+'_'+uuid.uuid4().hex+'.json'),value)


def dependencies(stdout):
    parents=[]
    for line in stdout.splitlines():
        fields=line.split('|')
        require(len(fields)==3,'Malformed project queue observation')
        sid,name,state=fields
        if name.startswith('ysz_'):
            require(re.fullmatch(r'[1-9][0-9]*',sid) is not None and re.fullmatch(r'[A-Z_]+',state) is not None,
                    'Ambiguous project queue parent identity/state')
            require(name!=JOB['name'],'The approved teacher is already in the project queue')
            parents.append(sid)
    return sorted(set(parents),key=int)


def submit_once(root):
    manifest,intent=precheck(root)
    # SBATCH_* can override reviewed directives. Drop all inherited SLURM_*
    # too: CPU/GPU/task counts, dependencies, partition/cluster/configuration,
    # and enclosing-job context must not affect this new allocation. Ordinary
    # PATH/HOME/authentication variables remain; the caller's env is untouched.
    # Query and both submissions share this snapshot to observe the same cluster.
    scheduler_environment={key:value for key,value in os.environ.items()
                           if not key.startswith(('SBATCH_','SLURM_'))}
    environment_filter=dict(removed_prefixes=['SBATCH_','SLURM_'],
                            remaining_variable_names=sorted(scheduler_environment))
    queue=subprocess.run(['squeue','-u','CLUSTER_USER','-h','-o','%i|%j|%T'],capture_output=True,text=True,
                         check=True,timeout=60,env=scheduler_environment)
    parents=dependencies(queue.stdout)
    arguments=(['--dependency=afterany:'+':'.join(parents),'--kill-on-invalid-dep=yes'] if parents else [])+[str(root/'sub.sh')]
    test=subprocess.run(['sbatch','--test-only']+arguments,cwd=root,capture_output=True,text=True,
                        timeout=60,env=scheduler_environment)
    evidence(root,'scheduler_test',dict(code=test.returncode,stdout=test.stdout,stderr=test.stderr,
                                     queue=queue.stdout,dependencies=parents,environment_filter=environment_filter))
    if test.returncode:
        raise RuntimeError('Scheduler test failed; no actual submission claim consumed')
    # The scheduler test grants no permission to alter staged bytes or bills.
    require(precheck(root)==(manifest,intent),
            'Immutable manifest and durable local intent changed during scheduler validation')
    exclusive_json(root/'SUBMISSION_CLAIM.json',dict(claim_id=intent['claim_id'],stage='teacher',
        max_submissions=1,manifest_sha256=sha256(root/'JOB_MANIFEST.json')))
    ledger=dict(schema='review10-v22-stage-submissions-v1',stage='teacher',max_submissions=1,
        claim_id=intent['claim_id'],jobs=[],pending_submission=dict(id=JOB['id'],claim_id=intent['claim_id']))
    exclusive_json(root/'SUBMISSIONS.json',ledger)
    try:
        response=subprocess.run(['sbatch','--parsable']+arguments,cwd=root,capture_output=True,text=True,
                                timeout=60,env=scheduler_environment)
    except (OSError,subprocess.SubprocessError) as error:
        evidence(root,'actual_response',dict(state='SUBMISSION_UNCERTAIN',error=type(error).__name__+': '+str(error)))
        raise RuntimeError('Actual submission uncertain; claim consumed; reconcile read-only, never retry') from error
    evidence(root,'actual_response',dict(code=response.returncode,stdout=response.stdout,stderr=response.stderr))
    match=re.fullmatch(r'([1-9][0-9]*)(?:;[A-Za-z0-9_.-]+)?',response.stdout.strip())
    if response.returncode or match is None:
        raise RuntimeError('Actual submission rejected or ambiguous; claim consumed; never retry')
    sid=match[1]
    require(sid not in BASE_SECONDS,'Returned identity conflicts with a historical parent; reconcile only')
    require(read_json(root/'SUBMISSIONS.json')==ledger,'Own pending submission ledger changed; never overwrite it')
    job=copy.deepcopy(manifest['jobs'][0])
    job.update(slurm_job_id=sid,dependency_parents=parents,remote_output=str(root/'outputs'/('teacher_'+sid)),
        remote_log=str(root/('slurm-'+sid+'.out')),submitted_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    ledger.pop('pending_submission')
    ledger['jobs']=[job]
    temporary=root/('.SUBMISSIONS_'+intent['claim_id']+'.tmp')
    exclusive_json(temporary,ledger)
    os.replace(temporary,root/'SUBMISSIONS.json')
    fsync_directory(root)
    descriptor=os.open(root/'submitted_job_id.txt',os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    with os.fdopen(descriptor,'w') as handle:
        handle.write(sid+'\n');handle.flush();os.fsync(handle.fileno())
    fsync_directory(root)
    return ledger


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parent)
    args=parser.parse_args(argv)
    result=submit_once(args.root.resolve())
    print(json.dumps(result,sort_keys=True))
    return result


if __name__=='__main__': main()
