"""One-shot Review10 v2.2 SFT/GRPO submission guard; standard library only.

This validates an independently prepared stage, never prepares data or executes
validators/models. A real sbatch attempt permanently consumes its stage claim.
The parent must independently run the frozen semantic validators and preserve
their raw reports before releasing a stage. See fixture tests for the interface.
"""
from __future__ import annotations

import argparse
import copy
import datetime
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import types


HELPER_SHA256 = '671cb999d2940fdaab6f69cb2757b00d1f6ee51aabcbfce5bef343c01f5bfeb5'
BASELINE_SHA256 = '7ff5adde926387658519eac73e86fb90cea7ba0814162ecdf18c8cad416c9ac6'
QWEN_WEIGHT_SHA = 'f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b'


def load_helper(path=None):
    """Authenticate reviewed helper bytes before executing any imported code."""
    path = Path(path) if path is not None else Path(__file__).with_name('review10_v22_teacher_submit.py')
    payload=path.read_bytes()
    if path.is_symlink() or hashlib.sha256(payload).hexdigest() != HELPER_SHA256:
        raise ValueError('Reviewed teacher helper hash changed before import')
    # Do not create an unlisted __pycache__ in the staged immutable inventory.
    sys.dont_write_bytecode = True
    module=types.ModuleType('review10_v22_bound_teacher_helper')
    module.__file__=str(path)
    # Execute exactly the bytes hashed above; do not reopen the file via an
    # import loader after authenticating it.
    exec(compile(payload,str(path),'exec'),module.__dict__)
    return module


shared = load_helper()
sha256 = shared.sha256
require = shared.require
number = shared.number
same_number = shared.same_number
relative = shared.relative
DATA_SOURCES = {'claude08_env.py','claude08_data.py','claude10_physics.py','claude10_data.py',
                'smoke_teacher10.py','smoke_sft10.py','smoke_mace.py','calibration.py','smoke_grpo08.py'}
FROZEN_SOURCES = {
    'calibration.py':'04eec85b96cc37f83cd45cbba21ccb032ef59038458c1a30aee4d4ec2c7bd4da',
    'claude08_data.py':'e09ae7060f9bb8a64dd9b27b0f9bce112bc9902aaf0c41f563a16c5e482d4941',
    'claude08_env.py':'e5057da4de8b11bd376d06762efa5a933924d937bbedf7d82d52c96fc85e43d9',
    'claude08_grpo.py':'29c7ec68fc5fd539745dfc4d26316acc8b627776f2a44c1ada05e2f8a87f5a9a',
    'claude10_data.py':'d9c890b58dad68f4f9b934b60d17569264ec4ee1901fbaccfb7a76a27efe57d6',
    'claude10_physics.py':'47a726cef4e8ad11b4c2d66a474795ed4b4a0cfd6e71851220fd3507b7c48730',
    'lattice.py':'ddda5be210b59e4f2d263ff30f0353f31d2a1f21254e38b994509d235723b034',
    'smoke_common.py':'1adcacf9915d4604533c97006f952a9076c9e752e2c91617382152f260833ebd',
    'smoke_grpo08.py':'b0da45518e6aafe09d905326d6ae5dd3d06d251f37bfdcc581f2f535d8718674',
    'smoke_grpo10.py':'c0a083aba8d9366918692670e649690c916c77add04b141fd444e8c295db7eed',
    'smoke_mace.py':'fb7663c09bc566b04ed0bc26656f57182ff757d54b8717515404d9f6c02da43c',
    'smoke_sft10.py':'a17f97b6fce9d419aa09104b5a045413637f55d0dc6a94ca16f6f5d5dceb845b',
    'smoke_teacher10.py':'e3e84d34a07340d75b9b9e690d00b14267ad1138344401f496313cce89fd7a0b'}
SFT_CONFIG = {'epochs':2,'minibatch':4,'gradient_accumulation':2,'learning_rate':1e-4,
    'clip_norm':1.,'lora_r':16,'lora_alpha':32,'lora_dropout':.05,'target_modules':['q_proj','v_proj'],
    'teacher_oversample':8,'max_prompt_tokens':1984,'max_sequence_tokens':2048,'max_new_tokens':64,
    'validation_denominator':80,'required_legal':76,'wall_seconds':6000,'outer_seconds':6600,'slurm_seconds':6660,
    'base_dtype':'bfloat16','adapter_dtype':'float32','temperature':1.,'top_p':1.,'top_k':0,'constrained_decoding':False}
JOBS = {
    'sft':dict(id='c10_v22_sft',name='ysz_c10v22_sft',gpus=1,cpu_cores=6,memory_GiB=90,
        hard_gpu_hours=1.85,slurm_wall_seconds=6660,outer_timeout_seconds=6600,python_work_seconds=6000,
        seed=2026102301,validation_denominator=80,required_legal=76),
    'grpo':dict(id='c10_v22_grpo',name='ysz_c10v22_grpo',gpus=1,cpu_cores=6,memory_GiB=90,
        hard_gpu_hours=2,slurm_wall_seconds=7200,outer_timeout_seconds=7140,python_work_seconds=6600,
        seed=2026102301,groups=4,group_size=4,action_budget=8,actual_relaxation_cap=132,raw_ef_call_cap=50000,
        max_generated_tokens=9472,max_prefill_tokens=293632,max_seen_tokens=262144,
        validation_denominator=20,required_legal=19)}


def checked(root, reference):
    path=shared.checked(root,reference)
    require(path.relative_to(root).parts[0]!='submission_checks' and path.name not in
            {'INPUTS.sha256','SUBMISSION_CLAIM.json','SUBMISSIONS.json','submitted_job_id.txt'},
            'Static evidence cannot use dynamic inventory exclusions')
    return path


def read_json(path):
    # Semantic evidence must use an explicit JSON filename, so every file this
    # guard parses is included in the recursive JSON reference closure.
    require(Path(path).suffix=='.json','Semantic JSON evidence requires a .json suffix')
    return shared.read_json(path)


def directory(root, name):
    require(isinstance(name,str) and re.fullmatch(r'[A-Za-z0-9_./-]+',name), 'Unsafe staged directory')
    p = PurePosixPath(name)
    require(not p.is_absolute() and '..' not in p.parts and str(p)==name and p.parts[0]!='submission_checks',
            'Noncanonical staged directory')
    target = root/name
    require(target.is_dir() and not target.is_symlink() and target.resolve().is_relative_to(root),
            'Missing or escaped staged directory')
    return target


def check_references(root, item, visited=None):
    """Hash the reference closure, opening referenced JSON evidence once.

    Evidence references are stage-relative. The official Qwen manifest's files
    are relative to its model directory, as required by that existing schema.
    """
    visited=set() if visited is None else visited
    if isinstance(item,dict):
        if 'path' in item and 'sha256' in item and (
                isinstance(item['path'],str) or isinstance(item['sha256'],str)):
            path=checked(root,item)
            if path.suffix=='.json' and path not in visited:
                visited.add(path)
                payload=read_json(path)
                if isinstance(payload,dict) and payload.get('repository')=='Qwen/Qwen3-0.6B':
                    for entry in payload.get('files',[]):
                        target=relative(path.parent,entry.get('path'))
                        check_references(root,dict(path=target.relative_to(root).as_posix(),
                            sha256=entry.get('sha256')),visited)
                    payload={key:value for key,value in payload.items() if key!='files'}
                check_references(root,payload,visited)
        else:
            for value in item.values(): check_references(root,value,visited)
    elif isinstance(item,list):
        for value in item: check_references(root,value,visited)


def safe_root(root, stage):
    require(isinstance(root,Path) and root.is_absolute() and root==root.resolve()
            and re.fullmatch(r'/[A-Za-z0-9_./-]+',str(root)) is not None
            and root.parts[-4:]==('training_smoke','claude10','recovery_v22',stage),
            'Only a canonical shell-safe fresh recovery_v22 stage root is allowed')


def verify_static_documents(root, hashes):
    """Apply one reference-closure policy to all immutable JSON and JSONL.

    This includes raw CPU evidence subsequently read by the authenticated
    teacher helper, rather than relying on individual validator entry points.
    """
    visited=set()
    def nonfinite(value):
        raise ValueError('Nonfinite JSONL value: '+value)
    for name,digest in hashes.items():
        path=relative(root,name)
        if path.suffix=='.json':
            check_references(root,dict(path=name,sha256=digest),visited)
        elif path.suffix=='.jsonl':
            with path.open() as handle:
                for line in handle:
                    item=json.loads(line,object_pairs_hook=shared._object,parse_constant=nonfinite)
                    check_references(root,item,visited)


def validation(root, reference, method, inputs):
    proof = read_json(checked(root,reference))
    require(proof.get('schema')=='review10-v22-independent-validation-v1'
            and proof.get('independently_verified') is True and bool(proof.get('reviewer'))
            and proof.get('method')==method and proof.get('input_sha256')==inputs,
            'Independent complete validation is missing or stale: '+method)
    source = 'project/scripts/'+method.split('.')[0]+'.py'
    require(proof.get('validator')==dict(path=source,sha256=sha256(relative(root,source))),
            'Independent validation must name its frozen production validator')
    checked(root,proof['validator'])
    raw = read_json(checked(root,proof.get('raw_validation')))
    check_references(root,raw)
    require(raw.get('status')=='passed' and raw.get('method')==method and raw.get('input_sha256')==inputs,
            'Raw independent validator result is missing or stale: '+method)
    return raw


def data_sources(manifest):
    return {Path(k).name:v for k,v in manifest['source_sha256'].items() if Path(k).name in DATA_SOURCES}


def protocol_bound(item):
    return item.get('protocol')==shared.PROTOCOL and item.get('protocol_sha')==shared.PROTOCOL_SHA


def verify_teacher(root, manifest):
    entry = manifest['teacher']
    sid = str(entry.get('job_id'))
    require(re.fullmatch(r'[1-9][0-9]*',sid) is not None and sid not in shared.BASE_SECONDS
            and entry.get('terminal')=='COMPLETED' and entry.get('recovered') is True,
            'New recovered successful teacher required')
    result = read_json(checked(root,entry['result']))
    require(result.get('schema')=='review10-v22-teacher-v1' and result.get('status')=='completed'
            and protocol_bound(result) and result.get('source_hashes')==data_sources(manifest)
            and result.get('checkpoint_sha256')==shared.CHECKPOINT_SHA256 and result.get('seed')==2026102301
            and result.get('diagnostic') is False and result.get('formal_batch_lbfgs_verified') is True
            and result.get('all_240_evaluations_verified') is True
            and str(result.get('runtime',{}).get('slurm_job_id'))==sid
            and result['runtime'].get('GPU_work_performed') is True
            and len(result.get('records',[]))==240 and len(result.get('trajectories',[]))==16,
            'Full completed v2.2 teacher evidence required')
    raw = validation(root,entry['verification'],'claude10_data.verify_teacher_result',
        dict(result=entry['result']['sha256'],protocol=shared.PROTOCOL_SHA,source_sha256=manifest['source_sha256']))
    counters = result.get('counters',{})
    require(raw.get('all_240_evaluations_verified') is True and raw.get('complete_trajectories')==16
            and raw.get('attempts_per_trajectory')==8 and str(raw.get('job_id'))==sid
            and raw.get('parent_exit_code')=='0:0'
            and counters.get('relax_actual')==240 and counters.get('physical_action_attempts')==224
            and counters.get('attempts')==224 and number(counters.get('raw_ef'))<=72240
            and all(raw.get(k)==counters.get(k) for k in
                ('relax_actual','raw_ef','forward_calls','attempts','physical_action_attempts','unmappable_terminals')),
            'All charged raw teacher outcomes and trajectories must be independently replayed')
    require(raw.get('verified_record_ids')==list(range(240))
            and raw.get('verified_trajectory_ids')==list(range(16)), 'Complete independent per-record/trajectory audit required')
    review = read_json(checked(root,entry['claude_review']))
    raw_review = read_json(checked(root,review.get('raw_review')))
    bindings = dict(teacher_result_sha256=entry['result']['sha256'],
                    teacher_verification_sha256=entry['verification']['sha256'])
    require(review.get('independently_reviewed') is True and review.get('reviewer')=='Claude'
            and review.get('verdict')=='APPROVE' and raw_review.get('reviewer')=='Claude'
            and raw_review.get('decision')=='APPROVE'
            and all(review.get(k)==v and raw_review.get(k)==v for k,v in bindings.items()),
            'Independent Claude technical review of the actual new teacher is required')
    return sid


def verify_upstream_claim(root, manifest, stage):
    entry=manifest[stage]
    ledger=read_json(checked(root,entry['submission_ledger']))
    claim=entry.get('submission_claim_id')
    require(re.fullmatch('[0-9a-f]{32}',str(claim)) is not None
            and ledger.get('schema')=='review10-v22-stage-submissions-v1' and ledger.get('stage')==stage
            and ledger.get('max_submissions')==1 and ledger.get('claim_id')==claim
            and not ledger.get('pending_submission') and len(ledger.get('jobs',[]))==1
            and str(ledger['jobs'][0].get('slurm_job_id'))==str(entry['job_id']),
            'Recovered upstream once-only claim/ledger binding required')
    return claim


def verify_qwen_data(root, manifest):
    model = directory(root,manifest['qwen_model_dir'])
    qwen_path = checked(root,manifest['qwen_manifest'])
    qwen = read_json(qwen_path)
    require(qwen_path.parent==model and qwen.get('repository')=='Qwen/Qwen3-0.6B'
            and qwen.get('status')=='verified_complete', 'Official verified Qwen base manifest required')
    names = set()
    for item in qwen.get('files',[]):
        path = relative(model,item.get('path'))
        require(item['path'] not in names and sha256(path)==item.get('sha256'), 'Qwen base file changed')
        require(not path.name.startswith('adapter_'), 'Old adapters are prohibited in the official base')
        names.add(item['path'])
    require('model.safetensors' in names and sha256(model/'model.safetensors')==QWEN_WEIGHT_SHA
            and {'tokenizer.json','tokenizer_config.json','config.json'}<=names,
            'Complete official base weights and tokenizer files required')
    require(names=={p.relative_to(model).as_posix() for p in model.rglob('*') if p.is_file() and p!=qwen_path},
            'Official base manifest must cover every model/tokenizer file')
    data_path = checked(root,manifest['data_manifest'])
    data = read_json(data_path)
    require(data_path.name=='manifest.json' and data.get('schema')=='review10-v22-data-v1'
            and protocol_bound(data) and data.get('seed')==2026102301 and data.get('teacher_oversample')==8
            and data.get('source_hashes')==data_sources(manifest)
            and data.get('checkpoint_sha256')==shared.CHECKPOINT_SHA256
            and data.get('format_physical_calls')==0 and data.get('DFT_labels_used') is False,
            'Frozen v2.2 data and unchanged source/protocol required')
    expected={'format_train.jsonl','teacher_train.jsonl','teacher_validation.jsonl',
              'legality_sft.jsonl','legality_grpo.jsonl','starts.jsonl','teacher_result.json'}
    require(set(data.get('files',{}))==expected, 'Full unchanged frozen data file set required')
    for name,digest in data['files'].items():
        require(sha256(relative(data_path.parent,name))==digest, 'Frozen data bytes changed')
    require(data['files']['teacher_result.json']==manifest['teacher']['result']['sha256'],
            'Frozen data must use exactly the recovered new teacher')
    counts={}
    used=set()
    for name in expected-{'teacher_result.json'}:
        rows=[json.loads(line,object_pairs_hook=shared._object) for line in (data_path.parent/name).read_text().splitlines()]
        check_references(root,rows)
        counts[name]=len(rows)
        if name in ('teacher_train.jsonl','teacher_validation.jsonl'): continue
        for row in rows:
            key=row.get('canonical_key')
            require(isinstance(key,str) and key not in used,'Frozen format/80/20/start families must be disjoint')
            used.add(key)
    require(counts==data.get('counts') and counts['format_train.jsonl']==4096
            and counts['legality_sft.jsonl']==80 and counts['legality_grpo.jsonl']==20 and counts['starts.jsonl']==4,
            'Fixed 4096/80/20/4 counts required')
    raw=validation(root,manifest['data_verification'],'claude10_data.load_frozen_data',
        dict(data_manifest=manifest['data_manifest']['sha256'],teacher_result=manifest['teacher']['result']['sha256'],
             source_sha256=manifest['source_sha256']))
    require(raw.get('semantic_recovery') is True and raw.get('disjoint_canonical_families') is True
            and raw.get('counts')==counts, 'Complete independent semantic data-freeze verification required')
    return counts


def verify_tokenizer(root, manifest, counts):
    refs=manifest['tokenizer_preflight']
    result=read_json(checked(root,refs['result']))
    plan=read_json(checked(root,refs['plan']))
    require(result.get('status')=='tokenizer_preflight_passed' and result.get('GPU_work_performed') is False
            and result.get('physical_calls')==0 and result.get('config')==SFT_CONFIG
            and result.get('data_manifest_sha256')==manifest['data_manifest']['sha256']
            and result.get('qwen_manifest_sha256')==manifest['qwen_manifest']['sha256']
            and result.get('frozen_schedule_sha256')==refs['plan']['sha256'], 'Actual CPU tokenizer preflight required')
    require(protocol_bound(plan) and plan.get('config')==SFT_CONFIG and plan.get('source_hashes')==data_sources(manifest)
            and plan.get('data_manifest_sha256')==manifest['data_manifest']['sha256']
            and plan.get('base')=='frozen_official_Qwen3-0.6B_without_633_adapter'
            and plan.get('physical_calls')==0 and plan.get('unique_format_examples')==4096
            and plan.get('independent_accepted_teacher_train_examples')==counts['teacher_train.jsonl']
            and plan.get('teacher_repeated_examples')==counts['teacher_train.jsonl']*8
            and plan.get('teacher_repeat_interpretation')=='eight_repeats_not_independent_data'
            and 0<number(plan.get('max_sequence_tokens'))<=2048 and 0<number(plan.get('max_prompt_tokens'))<=1984
            and len(plan.get('validation_prompt_tokens',[]))==80
            and all(0<number(v)<=1984 for v in plan['validation_prompt_tokens'])
            and plan.get('max_validation_generated_tokens')==5120,
            'No truncation, exact fixed SFT tokenizer plan required')
    raw=validation(root,refs['verification'],'smoke_sft10.tokenizer_preflight',
        dict(data_manifest=manifest['data_manifest']['sha256'],qwen_manifest=manifest['qwen_manifest']['sha256'],
             result=refs['result']['sha256'],plan=refs['plan']['sha256'],source_sha256=manifest['source_sha256']))
    require(all(raw.get(k) is True for k in ('no_truncation','exact_schedule','official_base_only')),
            'Independent full tokenizer/schedule validation required')


def verify_sft_release(root, manifest, teacher_id):
    entry=manifest['sft']; sid=str(entry.get('job_id'))
    require(re.fullmatch(r'[1-9][0-9]*',sid) is not None and sid not in set(shared.BASE_SECONDS)|{teacher_id}
            and entry.get('terminal')=='COMPLETED' and entry.get('recovered') is True,'New completed SFT required')
    result=read_json(checked(root,entry['result']))
    require(result.get('schema')=='review10-v22-sft-v1' and result.get('status')=='completed' and protocol_bound(result)
            and result.get('config')==SFT_CONFIG and result.get('seed')==2026102301
            and str(result.get('runtime',{}).get('slurm_job_id'))==sid
            and result.get('GPU_work_performed') is True and result.get('physical_calls')==0
            and result.get('training_runs')==1 and result.get('validation_retries')==0
            and result.get('adapter_release_allowed') is True and result.get('base_weight_sha256')==QWEN_WEIGHT_SHA
            and result.get('teacher_checkpoint_sha256')==shared.CHECKPOINT_SHA256
            and result.get('source_hashes')==data_sources(manifest)
            and result.get('data_manifest_sha256')==manifest['data_manifest']['sha256']
            and result.get('qwen_manifest_sha256')==manifest['qwen_manifest']['sha256'], 'Actual new fixed successful SFT required')
    raw=validation(root,entry['verification'],'smoke_grpo10.verify_release_manifest',
        dict(result=entry['result']['sha256'],data_manifest=manifest['data_manifest']['sha256'],source_sha256=manifest['source_sha256']))
    train=result.get('training',{}); updates=train.get('updates',[]); reload=result.get('reload',{})
    require(raw.get('finite_gradients') is True and raw.get('parameters_changed') is True
            and raw.get('legality_replayed') is True and train.get('parameters_changed') is True
            and updates and type(train.get('optimizer_updates')) is int and train['optimizer_updates']==len(updates)
            and all(number(row.get('gradient_norm'))>0 for row in updates)
            and re.fullmatch('[0-9a-f]{64}',str(train.get('adapter_hash_before'))) is not None
            and re.fullmatch('[0-9a-f]{64}',str(train.get('adapter_hash_after'))) is not None
            and train['adapter_hash_before']!=train['adapter_hash_after'], 'Finite gradients and actual SFT parameter changes required')
    require(reload.get('save_unload_reload') is True and all(reload.get(k) is True
            and raw.get('adapter_reload',{}).get(k) is True for k in
            ('unloaded_without_merge','reloaded_from_disk','state_exact_match')), 'Exact SFT disk reload proof required')
    adapter=directory(root,manifest['adapter_dir'])
    files={}
    for item in entry.get('adapter_files',[]):
        path=checked(root,item)
        require(path.parent==adapter and path.name not in files, 'Adapter files must be the new SFT directory')
        files[path.name]=item['sha256']
    require(set(files)=={'adapter_config.json','adapter_model.safetensors'}
            and reload.get('adapter_sha256')==files['adapter_model.safetensors']
            and reload.get('adapter_config_sha256')==files['adapter_config.json'], 'New SFT adapter hashes must match reload')
    config=read_json(adapter/'adapter_config.json')
    require(config.get('r')==16 and config.get('lora_alpha')==32 and config.get('lora_dropout')==.05
            and set(config.get('target_modules',[]))=={'q_proj','v_proj'}, 'Fixed new SFT LoRA configuration required')
    legality=result.get('legality',{}); records=legality.get('records',[])
    require(legality.get('status')=='passed' and legality.get('denominator')==80 and legality.get('attempts')==80
            and len(records)==80 and type(legality.get('legal')) is int and 76<=legality['legal']<=80
            and sum(r.get('legal') is True for r in records)==legality['legal']
            and all(r.get('index')==i and r.get('status')=='completed' and type(r.get('legal')) is bool
                    and isinstance(r.get('text'),str) for i,r in enumerate(records)), 'At least76/80 exactly once, independently replayed SFT legality required')
    release=read_json(checked(root,manifest['release_manifest']))
    check_references(root,release)
    require(release.get('schema')=='review10-v22-release-v1' and protocol_bound(release)
            and release.get('source_hashes')=={Path(k).name:v for k,v in manifest['source_sha256'].items()}
            and all(release.get(k)==manifest[k] for k in ('teacher','sft','checkpoint','data_manifest','qwen_manifest')),
            'Full production release must bind this exact stage and new adapter')
    require(release.get('seed')==2026102301 and release.get('authorization')==dict(review=10,recovery='v2.2',
                retry_of='650',authorized_retry_index=1,max_card_hours=2,additional_card_hours=6.5)
            and set(release.get('cpu_preflight',{}))=={'result','verification'}
            and set(release.get('root_fees',{}))=={'verification','root_budget','package_budget','baseline_budget'}
            and release.get('mace648',{}).get('job_id')=='648'
            and release['root_fees']['root_budget']==dict(path='ROOT_POOL_PRE_SUBMISSION.json',sha256=sha256(root/'ROOT_POOL_PRE_SUBMISSION.json'))
            and release['root_fees']['package_budget']==dict(path='PACKAGE_PRE_SUBMISSION.json',sha256=sha256(root/'PACKAGE_PRE_SUBMISSION.json'))
            and release['root_fees']['baseline_budget']==manifest['baseline_budget']
            and release['cpu_preflight']['result']==dict(path='cpu_preflight/result.json',sha256=sha256(root/'cpu_preflight/result.json'))
            and release['cpu_preflight']['verification']==dict(path='CPU_PREFLIGHT_VERIFIED.json',sha256=sha256(root/'CPU_PREFLIGHT_VERIFIED.json')),
            'Every prerequisite of the full unchanged production release is required')
    verified=validation(root,manifest['release_verification'],'smoke_grpo10.verify_release_manifest',
        dict(release_manifest=manifest['release_manifest']['sha256'],source_sha256=manifest['source_sha256']))
    require(verified.get('verified') is True and verified.get('full_release_verified') is True
            and verified.get('all_evidence_hashes_checked') is True
            and verified.get('manifest_sha256')==manifest['release_manifest']['sha256']
            and verified.get('source_hashes')==release['source_hashes'], 'Independent complete production release verification required')
    actual=verified.get('verification_result',{})
    require(actual.get('verified') is True and protocol_bound(actual)
            and actual.get('manifest_sha256')==manifest['release_manifest']['sha256']
            and actual.get('source_hashes')==release['source_hashes'] and actual.get('adapter_hashes')==files
            and actual.get('evidence')=={name:release[name] for name in ('sft','teacher','mace648','cpu_preflight','root_fees')}
            and all(actual.get(key+'_sha256')==manifest[key]['sha256'] for key in ('data_manifest','qwen_manifest','checkpoint')),
            'Preserve the actual full production verify_release_manifest return value')
    return sid


def verify_fees(root, manifest, stage, upstream):
    pool=read_json(root/'ROOT_POOL_PRE_SUBMISSION.json'); package=read_json(root/'PACKAGE_PRE_SUBMISSION.json')
    baseline=read_json(checked(root,manifest['baseline_budget']))
    check_references(root,[pool,package,baseline])
    require(manifest['baseline_budget']['sha256']==BASELINE_SHA256, 'Frozen six historical bills changed')
    historical={str(row['slurm_job_id']):row for row in baseline['jobs']}
    require(set(historical)==set(shared.BASE_SECONDS), 'All six historical parent allocations required')
    for sid,seconds in shared.BASE_SECONDS.items():
        require(same_number(historical[sid].get('actual_card_hours'),seconds/3600), 'Historical fee changed')
    new={}
    for prior,sid in upstream.items():
        budget=read_json(checked(root,manifest[prior]['budget']))
        require(budget.get('accounting_complete') is True and budget.get('accounting_unknown_job_ids')==[]
                and not budget.get('collection_risks') and not budget.get('collection_error') and not budget.get('pending_submission')
                and same_number(budget.get('authorized_card_hours'),shared.CAPS[prior])
                and same_number(budget.get('remaining_caps'),0) and len(budget.get('jobs',[]))==1,
                'Prior stage fee must be closed and known')
        job=budget['jobs'][0]; account=job.get('sacct',{}); seconds=account.get('elapsed_seconds')
        require(str(job.get('slurm_job_id'))==sid and job.get('state')=='COMPLETED'
                and str(account.get('job_id'))==sid and account.get('state')=='COMPLETED' and account.get('exit_code')=='0:0'
                and type(seconds) is int and seconds>0 and job.get('gpus')==1 and job.get('allocation_gpu_count')==1
                and job.get('cpu_cores')==6 and job.get('memory_GiB')==90 and not job.get('hard_cap_exceeded')
                and not job.get('unexpected_gpu_allocation') and same_number(job.get('hard_gpu_hours'),shared.CAPS[prior])
                and same_number(job.get('actual_card_hours'),seconds/3600) and seconds/3600<=shared.CAPS[prior]
                and same_number(budget.get('actual_card_hours'),seconds/3600)
                and same_number(budget.get('actual_plus_remaining_caps'),seconds/3600), 'Actual successful parent allocation fee required')
        observed=account.get('alloc_tres')
        require(isinstance(observed,str) and bool(observed),'Missing observed allocation TRES')
        parts=observed.split(',')
        require(all(re.fullmatch(r'[A-Za-z0-9_/:.-]+=[A-Za-z0-9_.:-]+',part) is not None for part in parts),
                'Malformed observed allocation TRES')
        pairs=[part.split('=',1) for part in parts]
        require(len({key for key,_ in pairs})==len(pairs),'Duplicate or ambiguous observed allocation TRES')
        tres=dict(pairs)
        require(tres.get('cpu')=='6' and tres.get('mem') in ('90G','92160M') and tres.get('node')=='1'
                and tres.get('gres/gpu')=='1' and tres.get('gres/gpu:a100')=='1'
                and not any(k.startswith('gres/gpu:') and k!='gres/gpu:a100' for k in tres), 'Observed1A100/6CPU/90GiB allocation required')
        new[sid]=job
    spent=sum(row['actual_card_hours'] for row in new.values()); remaining=3.85 if stage=='sft' else 2
    for budget,cap,history,rows in ((pool,10,812/3600,{**historical,**new}),(package,6.5,660/3600,new)):
        actual_rows=budget.get('jobs',[])
        require(len(actual_rows)==len(rows) and {str(row.get('slurm_job_id')):row for row in actual_rows}==rows,
                'Root/package must preserve exact historical and new parent rows once')
        require(budget.get('accounting_complete') is True and budget.get('accounting_unknown_job_ids')==[]
                and not budget.get('spending_frozen') and not budget.get('collection_risks')
                and not budget.get('collection_error') and not budget.get('pending_submission')
                and budget.get('collection_status')!='DEFERRED'
                and same_number(budget.get('authorized_card_hours'),cap)
                and same_number(budget.get('actual_card_hours'),history+spent)
                and same_number(budget.get('remaining_caps'),remaining)
                and same_number(budget.get('actual_plus_remaining_caps'),history+spent+remaining)
                and history+spent+remaining<=cap, 'Closed actual fees plus all remaining stage caps required')
    require(package.get('stage_caps')==shared.CAPS and same_number(package.get('frozen652_actual_card_hours'),507/3600)
            and same_number(package.get('frozen656_actual_card_hours'),153/3600)
            and same_number(package.get('historical_package_actual_card_hours'),660/3600)
            and same_number(package.get('new_actual_card_hours'),spent), '652/656 must remain within package6.5')
    if stage=='grpo':
        require(660/3600+spent+remaining<=6, 'Keep the production GRPO release additional conservative6 gate')


def runner_scripts(root, stage, manifest):
    """Return exact shell bytes; caller only writes these after upstream release."""
    require(stage in JOBS, 'Unknown stage')
    safe_root(root,stage)
    job=JOBS[stage]
    # Manifest paths are validated again at submission; only shell-safe relative
    # names can enter these templates. Interpreter location is fixed from root.
    paths=[manifest['qwen_model_dir'],manifest['qwen_manifest']['path'],manifest['data_manifest']['path']]
    if stage=='grpo': paths += [manifest['adapter_dir'],manifest['checkpoint']['path'],manifest['release_manifest']['path']]
    for name in paths:
        p=PurePosixPath(name)
        require(isinstance(name,str) and re.fullmatch('[A-Za-z0-9_./-]+',name) is not None
                and not p.is_absolute() and '..' not in p.parts and str(p)==name, 'Unsafe runner input path')
    command='"$YSZ_PYTHON" project/scripts/smoke_'+stage+'10.py --model '+manifest['qwen_model_dir']
    command+=' --qwen-manifest '+manifest['qwen_manifest']['path']+' --data '+str(PurePosixPath(manifest['data_manifest']['path']).parent)
    if stage=='grpo':
        command+=' --adapter '+manifest['adapter_dir']+' --checkpoint '+manifest['checkpoint']['path']+' --release-manifest '+manifest['release_manifest']['path']
    command+=' --output "$YSZ_OUT"'
    directives={'nodes':'1','partition':'a100','ntasks':'1','ntasks-per-node':'1','cpus-per-task':'6',
        'gres':'gpu:a100:1','mem':'90G','time':'01:51:00' if stage=='sft' else '02:00:00',
        'job-name':job['name'],'chdir':str(root),'output':'slurm-%j.out'}
    sub='#!/bin/bash\n'+''.join('#SBATCH --'+k+'='+v+'\n' for k,v in directives.items())
    sub+='#SBATCH --no-requeue\nset -euo pipefail\ntest -n "${SLURM_JOB_ID:-}"\ntest ! -e ../../../STOP_SMOKE.json\n'
    sub+='exec timeout --signal=TERM --kill-after=30s '+str(job['outer_timeout_seconds'])+'s bash run.sh\n'
    run='''#!/bin/bash
set -eo pipefail
source /etc/profile
module purge
set -u
test -n "${SLURM_JOB_ID:-}"
YSZ_STAGE_ROOT=$(cd "$(dirname "$0")" && pwd)
YSZ_SMOKE_ROOT=$(cd "$YSZ_STAGE_ROOT/../../.." && pwd)
YSZ_PYTHON="$YSZ_SMOKE_ROOT/claude08_grpo/venv/bin/python"
export PATH="$(dirname "$YSZ_PYTHON"):$PATH"
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 NUMEXPR_NUM_THREADS=6 PYTHONDONTWRITEBYTECODE=1
export YSZ_EXPECTED_GPUS=1 XDG_CACHE_HOME="$YSZ_SMOKE_ROOT/cache"
test ! -e "$YSZ_SMOKE_ROOT/STOP_SMOKE.json"
cd "$YSZ_STAGE_ROOT"
sha256sum -c INPUTS.sha256
'''
    run+='YSZ_OUT="$YSZ_STAGE_ROOT/outputs/'+stage+'_$SLURM_JOB_ID"\n'
    run+='''YSZ_RUNTIME="$YSZ_STAGE_ROOT/runtime/$SLURM_JOB_ID"
test ! -e "$YSZ_OUT"
test ! -e "$YSZ_RUNTIME"
mkdir -p "$YSZ_RUNTIME"
date -Is > "$YSZ_RUNTIME/start.txt"
trap 'kill "${YSZ_SAMPLE_PID:-}" 2>/dev/null || true; if test -n "${YSZ_SAMPLE_PID:-}"; then wait "$YSZ_SAMPLE_PID" 2>/dev/null || true; fi; date -Is > "$YSZ_RUNTIME/end.txt"; mkdir -p "$YSZ_OUT"; cp "$YSZ_RUNTIME"/* "$YSZ_OUT"/; for yszt_proof in cuda_preflight_*.json; do if test -f "$yszt_proof"; then cp "$yszt_proof" "$YSZ_OUT"/; fi; done' EXIT
scontrol show job "$SLURM_JOB_ID" > "$YSZ_RUNTIME/slurm_allocation.txt"
source ./select_real_driver.sh
nvidia-smi --id="$CUDA_VISIBLE_DEVICES" --query-gpu=timestamp,index,uuid,memory.used,memory.total,utilization.gpu,utilization.memory,power.draw --format=csv -l 5 > "$YSZ_RUNTIME/gpu_samples.csv" &
YSZ_SAMPLE_PID=$!
'''
    return {'sub.sh':sub,'run.sh':run+command+'\n'}


def gate_bindings(root, manifest):
    return dict(manifest_sha256=sha256(root/'JOB_MANIFEST.json'),authorization_sha256=sha256(root/'AUTHORIZATION.json'),
        root_budget_sha256=sha256(root/'ROOT_POOL_PRE_SUBMISSION.json'),package_budget_sha256=sha256(root/'PACKAGE_PRE_SUBMISSION.json'),
        source_sha256=manifest['source_sha256'],stage_scripts_sha256=manifest['stage_scripts_sha256'],
        cpu_preflight_verification_sha256=sha256(root/'CPU_PREFLIGHT_VERIFIED.json'),
        wrapper_sha256=sha256(__file__),helper_sha256=HELPER_SHA256)


def precheck(root, stage):
    require(stage in JOBS,'Unknown stage')
    safe_root(root,stage)
    if any((root/name).exists() for name in ('SUBMISSIONS.json','SUBMISSION_CLAIM.json','submitted_job_id.txt')):
        raise RuntimeError('Existing remote claim/ledger: reconcile read-only, never resubmit')
    if (root.parents[2]/'STOP_SMOKE.json').exists(): raise RuntimeError('Global STOP marker; never override')
    require(not root.is_symlink() and not any(p.is_symlink() for p in root.rglob('*')), 'Stage symlinks are prohibited')
    hashes=shared.inventory(root)
    verify_static_documents(root,hashes)
    require(sha256(relative(root,'review10_v22_teacher_submit.py'))==HELPER_SHA256,'Bound helper changed')
    manifest=read_json(root/'JOB_MANIFEST.json')
    require(manifest.get('schema')=='review10-v22-downstream-stage-v1' and manifest.get('stage')==stage
            and manifest.get('remote_smoke_root')==str(root) and manifest.get('jobs')==[JOBS[stage]]
            and all(type(manifest['jobs'][0].get(k)) is type(v) for k,v in JOBS[stage].items()), 'Wrong fixed stage manifest')
    check_references(root,manifest)
    auth=read_json(root/'AUTHORIZATION.json')
    check_references(root,auth)
    require(auth.get('schema')=='review10-v22-authorization-v1' and auth.get('direct_user_confirmation') is True
            and auth.get(stage+'_submission_released') is True and auth.get('automatic_retry') is False
            and auth.get('formal_training_allowed') is False and auth.get('new_DFT_allowed') is False
            and auth.get('teacher_submissions_used')==1 and auth.get('additional_GPU_submissions_allowed')==0
            and auth.get('max_new_teacher_submissions')==1 and auth.get('max_concurrent_GPU_jobs')==1
            and all(type(auth.get(k)) is int for k in ('teacher_submissions_used','additional_GPU_submissions_allowed',
                                                        'max_new_teacher_submissions','max_concurrent_GPU_jobs'))
            and same_number(auth.get('package_total_cap_including652_656_card_hours'),6.5)
            and same_number(auth.get('root_pool_card_hours'),10) and auth.get('stage_caps_card_hours')==shared.CAPS
            and auth.get('selected_protocol')==shared.SELECTED and auth.get('protocol_sha')==shared.PROTOCOL_SHA,
            'Explicit released downstream stage and unchanged bounded authorization required')
    source=manifest.get('source_sha256')
    expected={'project/scripts/'+name:digest for name,digest in FROZEN_SOURCES.items()}
    require(source==expected and auth.get('core_source_sha256')==FROZEN_SOURCES
            and set(source)=={p.relative_to(root).as_posix() for p in (root/'project/scripts').rglob('*') if p.is_file()},
            'Exactly thirteen frozen reviewed production sources required')
    for name,digest in source.items(): require(sha256(relative(root,name))==digest,'Frozen production source changed')
    require(read_json(checked(root,manifest['protocol']))==shared.SELECTED
            and manifest['checkpoint']['sha256']==shared.CHECKPOINT_SHA256
            and manifest.get('driver_sha256')=={'select_real_driver.sh':shared.DRIVER_SHA256}
            and sha256(relative(root,'select_real_driver.sh'))==shared.DRIVER_SHA256, 'Frozen protocol/model/driver required')
    cpu_proof=read_json(root/'CPU_PREFLIGHT_VERIFIED.json')
    check_references(root,cpu_proof)
    for name,digest in cpu_proof.get('evidence_sha256',{}).items():
        checked(root,dict(path=name,sha256=digest))
    shared.verify_cpu(root,manifest,shared.SELECTED['relaxation'])
    teacher_id=verify_teacher(root,manifest)
    verify_upstream_claim(root,manifest,'teacher')
    counts=verify_qwen_data(root,manifest)
    verify_tokenizer(root,manifest,counts)
    upstream={'teacher':teacher_id}
    if stage=='grpo':
        upstream['sft']=verify_sft_release(root,manifest,teacher_id)
        verify_upstream_claim(root,manifest,'sft')
    verify_fees(root,manifest,stage,upstream)
    require(set(manifest.get('stage_scripts_sha256',{}))=={'sub.sh','run.sh'}, 'Both exact runner scripts required')
    for name,text in runner_scripts(root,stage,manifest).items():
        require(relative(root,name).read_text()==text and sha256(root/name)==manifest['stage_scripts_sha256'][name],
                'Only exact fixed resource/safety/entry commands may execute')
    bindings=gate_bindings(root,manifest)
    require(sha256(relative(root,'submit_once.py'))==bindings['wrapper_sha256'],'Staged wrapper differs from executing reviewed guard')
    gate=read_json(root/'PRE_SUBMIT_GATE.json')
    check_references(root,gate)
    require(gate.get('schema')=='review10-v22-downstream-pre-submit-v1' and gate.get('independently_verified') is True
            and all(gate.get(k)==v for k,v in bindings.items()), 'Independent pre-submit bindings changed')
    require(set(gate.get('review_evidence',{}))=={'spec','quality'},'Independent SPEC then quality reviews required')
    for kind,verdict in (('spec','PASS'),('quality','APPROVE')):
        review=read_json(checked(root,gate['review_evidence'][kind]))
        require(review.get('independently_reviewed') is True and bool(review.get('reviewer')) and review.get('verdict')==verdict
                and all(review.get(k)==v for k,v in bindings.items()),'Independent '+kind+' review missing or stale')
    intent=read_json(root/'SUBMISSION_INTENT_ID.json')
    check_references(root,intent)
    prior_claims=shared.HISTORICAL_CLAIM_IDS|{'79a380067c944847b10b80b587433239'}
    prior_claims|={manifest[name]['submission_claim_id'] for name in upstream}
    require(intent.get('schema')=='review10-v22-local-intent-v1' and intent.get('stage')==stage
            and intent.get('max_submissions')==1 and intent.get('durable_locally') is True
            and re.fullmatch('[0-9a-f]{32}',str(intent.get('claim_id'))) is not None
            and intent['claim_id'] not in prior_claims and intent['claim_id']==manifest.get('submission_claim_id')
            and intent.get('manifest_sha256')==sha256(root/'JOB_MANIFEST.json'), 'Fresh unique durable local stage intent required')
    return manifest,intent


def dependencies(stdout, stage):
    parents=[]
    for line in stdout.splitlines():
        fields=line.split('|')
        require(len(fields)==3,'Malformed project queue observation')
        sid,name,state=fields
        if name.startswith('ysz_'):
            require(re.fullmatch('[1-9][0-9]*',sid) is not None and re.fullmatch('[A-Z_]+',state) is not None,
                    'Ambiguous project queue identity/state')
            require(name!=JOBS[stage]['name'],'This unique stage is already queued')
            parents.append(sid)
    return sorted(set(parents),key=int)


def submit_once(root, stage):
    root=Path(root)
    safe_root(root,stage)
    manifest,intent=precheck(root,stage)
    environment={k:v for k,v in os.environ.items() if not k.startswith(('SBATCH_','SLURM_'))}
    queue=subprocess.run(['squeue','-u','CLUSTER_USER','-h','-o','%i|%j|%T'],capture_output=True,text=True,
        check=True,timeout=60,env=environment)
    parents=dependencies(queue.stdout,stage)
    arguments=(['--dependency=afterany:'+':'.join(parents),'--kill-on-invalid-dep=yes'] if parents else [])+[str(root/'sub.sh')]
    test=subprocess.run(['sbatch','--test-only']+arguments,cwd=root,capture_output=True,text=True,timeout=60,env=environment)
    shared.evidence(root,'scheduler_test',dict(code=test.returncode,stdout=test.stdout,stderr=test.stderr,
        queue=queue.stdout,dependencies=parents,environment_filter=dict(removed_prefixes=['SBATCH_','SLURM_'],
        remaining_variable_names=sorted(environment))))
    if test.returncode: raise RuntimeError('Scheduler test failed; no actual claim consumed')
    require(precheck(root,stage)==(manifest,intent),'Staged inputs/intent changed during scheduler test')
    shared.exclusive_json(root/'SUBMISSION_CLAIM.json',dict(claim_id=intent['claim_id'],stage=stage,
        max_submissions=1,manifest_sha256=sha256(root/'JOB_MANIFEST.json')))
    ledger=dict(schema='review10-v22-stage-submissions-v1',stage=stage,max_submissions=1,
        claim_id=intent['claim_id'],jobs=[],pending_submission=dict(id=JOBS[stage]['id'],claim_id=intent['claim_id']))
    shared.exclusive_json(root/'SUBMISSIONS.json',ledger)
    try:
        response=subprocess.run(['sbatch','--parsable']+arguments,cwd=root,capture_output=True,text=True,timeout=60,env=environment)
    except (OSError,subprocess.SubprocessError) as exc:
        shared.evidence(root,'actual_response',dict(state='SUBMISSION_UNCERTAIN',error=type(exc).__name__+': '+str(exc)))
        raise RuntimeError('Actual submission uncertain; claim consumed; reconcile read-only, never retry') from exc
    shared.evidence(root,'actual_response',dict(code=response.returncode,stdout=response.stdout,stderr=response.stderr))
    match=re.fullmatch(r'([1-9][0-9]*)(?:;[A-Za-z0-9_.-]+)?',response.stdout.strip())
    if response.returncode or match is None: raise RuntimeError('Actual result rejected or ambiguous; claim consumed; never retry')
    sid=match[1]
    upstream={str(manifest[name]['job_id']) for name in ('teacher','sft') if name in manifest}
    require(sid not in set(shared.BASE_SECONDS)|upstream|set(parents),'Returned job identity already exists; never retry')
    require(read_json(root/'SUBMISSIONS.json')==ledger,'Own pending ledger changed; never overwrite it')
    job=copy.deepcopy(manifest['jobs'][0])
    job.update(slurm_job_id=sid,dependency_parents=parents,remote_output=str(root/'outputs'/(stage+'_'+sid)),
        remote_log=str(root/('slurm-'+sid+'.out')),submitted_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    ledger.pop('pending_submission');ledger['jobs']=[job]
    temp=root/('.SUBMISSIONS_'+intent['claim_id']+'.tmp')
    shared.exclusive_json(temp,ledger);os.replace(temp,root/'SUBMISSIONS.json');shared.fsync_directory(root)
    descriptor=os.open(root/'submitted_job_id.txt',os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    with os.fdopen(descriptor,'w') as handle:
        handle.write(sid+'\n');handle.flush();os.fsync(handle.fileno())
    shared.fsync_directory(root)
    return ledger


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=tuple(JOBS),required=True)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parent)
    args=parser.parse_args(argv)
    result=submit_once(args.root,args.stage)
    print(json.dumps(result,sort_keys=True))
    return result


if __name__=='__main__': main()
