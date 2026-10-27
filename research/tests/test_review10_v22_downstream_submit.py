"""Temporary-only downstream guard tests; real processes and network are denied."""
import ast
import copy
import hashlib
import importlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(HERE/'scripts'), str(HERE/'tests')]
from test_review10_v22_teacher_submit import fixture as cpu_fixture

try:
    submit = importlib.import_module('review10_v22_downstream_submit')
except ModuleNotFoundError as exc:
    if exc.name != 'review10_v22_downstream_submit':
        raise
    submit = None

CANONICAL_SOURCE_PINS = copy.deepcopy(submit.FROZEN_SOURCES) if submit is not None else None

_BLOCK_EXTERNAL = []


def deny_external(event, _args):
    if _BLOCK_EXTERNAL and event in {'subprocess.Popen', 'socket.connect', 'os.system'}:
        raise AssertionError('Forbidden real external operation in fixture test: '+event)


sys.addaudithook(deny_external)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True))


def load(path):
    return json.loads(path.read_text())


def digest(path):
    path = Path(path)
    if path.name == 'shared.model':
        return submit.shared.CHECKPOINT_SHA256
    if path.name == 'model.safetensors':
        return submit.QWEN_WEIGHT_SHA
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ref(root, path):
    return dict(path=path.relative_to(root).as_posix(), sha256=digest(path))


def checksum(root):
    excluded = {'INPUTS.sha256', 'SUBMISSION_CLAIM.json', 'SUBMISSIONS.json', 'submitted_job_id.txt'}
    files = sorted(p for p in root.rglob('*') if p.is_file() and p.name not in excluded
                   and p.relative_to(root).parts[0] != 'submission_checks')
    (root/'INPUTS.sha256').write_text(''.join(digest(p)+'  '+p.relative_to(root).as_posix()+'\n' for p in files))


def audit(root, name, method, inputs, values):
    raw = root/f'evidence/{name}_raw.json'
    save(raw, dict(status='passed', method=method, input_sha256=inputs, **values))
    proof = root/f'evidence/{name}_verified.json'
    validator = method.split('.')[0]
    save(proof, dict(schema='review10-v22-independent-validation-v1', independently_verified=True,
        reviewer='independent-fixture-reviewer', method=method, input_sha256=inputs,
        validator=ref(root, root/'project/scripts'/f'{validator}.py'), raw_validation=ref(root, raw)))
    return ref(root, proof)


def reseal(root):
    manifest = load(root/'JOB_MANIFEST.json')
    scripts = submit.runner_scripts(root, manifest['stage'], manifest)
    for name, contents in scripts.items():
        (root/name).write_text(contents)
    manifest['stage_scripts_sha256'] = {name:digest(root/name) for name in scripts}
    save(root/'JOB_MANIFEST.json', manifest)
    bindings = submit.gate_bindings(root, manifest)
    reviews = {}
    for kind, verdict in [('spec', 'PASS'), ('quality', 'APPROVE')]:
        path = root/f'reviews/downstream_{kind}.json'
        save(path, dict(independently_reviewed=True, reviewer='independent-'+kind,
                        verdict=verdict, **bindings))
        reviews[kind] = ref(root, path)
    save(root/'PRE_SUBMIT_GATE.json', dict(schema='review10-v22-downstream-pre-submit-v1',
         independently_verified=True, **bindings, review_evidence=reviews))
    save(root/'SUBMISSION_INTENT_ID.json', dict(schema='review10-v22-local-intent-v1',
         stage=manifest['stage'], claim_id=manifest['submission_claim_id'], max_submissions=1,
         durable_locally=True, manifest_sha256=digest(root/'JOB_MANIFEST.json')))
    checksum(root)


def fixture(base, stage='sft'):
    teacher_root = cpu_fixture(base)
    root = teacher_root.with_name(stage)
    teacher_root.rename(root)
    source = load(root/'JOB_MANIFEST.json')['source_sha256']
    model = root/'shared.model'
    shutil.copyfile(base/'shared.model', model)
    shutil.copyfile(HERE/'scripts/review10_v22_teacher_submit.py', root/'review10_v22_teacher_submit.py')
    shutil.copyfile(HERE/'scripts/review10_v22_downstream_submit.py', root/'submit_once.py')
    (root/'qwen').mkdir()
    (root/'qwen/model.safetensors').write_bytes(b'fixture only; never loaded')
    (root/'qwen/tokenizer.json').write_text('{}')
    (root/'qwen/tokenizer_config.json').write_text('{}')
    (root/'qwen/config.json').write_text('{}')
    save(root/'qwen/manifest.json', dict(repository='Qwen/Qwen3-0.6B', status='verified_complete',
        files=[dict(path=p.name, sha256=digest(p)) for p in sorted((root/'qwen').iterdir())]))
    teacher = dict(schema='review10-v22-teacher-v1', status='completed', protocol=submit.shared.PROTOCOL,
        protocol_sha=submit.shared.PROTOCOL_SHA, checkpoint_sha256=submit.shared.CHECKPOINT_SHA256,
        source_hashes={Path(k).name:v for k,v in source.items() if Path(k).name in submit.DATA_SOURCES},
        diagnostic=False, formal_batch_lbfgs_verified=True, all_240_evaluations_verified=True,
        runtime=dict(slurm_job_id='659', GPU_work_performed=True), seed=2026102301,
        records=[dict(id=i) for i in range(240)], trajectories=[dict(id=i, complete=True, steps=[{}]*8) for i in range(16)],
        counters=dict(relax_actual=240, physical_action_attempts=224, attempts=224,
                      raw_ef=480, forward_calls=120, unmappable_terminals=0))
    save(root/'evidence/teacher_result.json', teacher)
    teacher_ref = ref(root, root/'evidence/teacher_result.json')
    teacher_inputs = dict(result=teacher_ref['sha256'], protocol=submit.shared.PROTOCOL_SHA,
                         source_sha256=source)
    teacher_proof = audit(root, 'teacher', 'claude10_data.verify_teacher_result', teacher_inputs,
        dict(job_id='659', parent_exit_code='0:0', all_240_evaluations_verified=True,
             complete_trajectories=16, attempts_per_trajectory=8, verified_record_ids=list(range(240)),
             verified_trajectory_ids=list(range(16)), **teacher['counters']))
    save(root/'evidence/claude_transcript.json', dict(decision='APPROVE', reviewer='Claude',
         teacher_result_sha256=teacher_ref['sha256'], teacher_verification_sha256=teacher_proof['sha256']))
    save(root/'evidence/claude_review.json', dict(independently_reviewed=True, reviewer='Claude', verdict='APPROVE',
        teacher_result_sha256=teacher_ref['sha256'], teacher_verification_sha256=teacher_proof['sha256'],
        raw_review=ref(root, root/'evidence/claude_transcript.json')))
    counts = {'format_train.jsonl':4096, 'legality_sft.jsonl':80, 'legality_grpo.jsonl':20,
              'starts.jsonl':4, 'teacher_train.jsonl':1, 'teacher_validation.jsonl':1}
    for name, n in counts.items():
        p = root/'data'/name
        p.parent.mkdir(exist_ok=True)
        p.write_text(''.join(json.dumps(dict(canonical_key=name+str(i), occupancy=[i], unscored=True))+'\n'
                            for i in range(n)))
    shutil.copyfile(root/'evidence/teacher_result.json', root/'data/teacher_result.json')
    save(root/'data/manifest.json', dict(schema='review10-v22-data-v1', protocol=submit.shared.PROTOCOL,
        protocol_sha=submit.shared.PROTOCOL_SHA, seed=2026102301, checkpoint_sha256=submit.shared.CHECKPOINT_SHA256,
        source_hashes=teacher['source_hashes'], teacher_oversample=8, format_physical_calls=0, DFT_labels_used=False,
        counts=counts, files={p.name:digest(p) for p in (root/'data').iterdir()},
        teacher_train_trajectory_ids=[0], teacher_validation_trajectory_ids=[1]))
    data_ref = ref(root, root/'data/manifest.json')
    data_proof = audit(root, 'data', 'claude10_data.load_frozen_data',
        dict(data_manifest=data_ref['sha256'], teacher_result=teacher_ref['sha256'], source_sha256=source),
        dict(semantic_recovery=True, disjoint_canonical_families=True, counts=counts))
    qwen_ref = ref(root, root/'qwen/manifest.json')
    plan = dict(config=submit.SFT_CONFIG, protocol=submit.shared.PROTOCOL, protocol_sha=submit.shared.PROTOCOL_SHA,
        source_hashes=teacher['source_hashes'], data_manifest_sha256=data_ref['sha256'], unique_format_examples=4096,
        independent_accepted_teacher_train_examples=1, teacher_repeated_examples=8,
        teacher_repeat_interpretation='eight_repeats_not_independent_data', max_sequence_tokens=100,
        max_prompt_tokens=80, validation_prompt_tokens=[80]*80, max_validation_generated_tokens=5120,
        seen_token_ceiling=20000, schedule=dict(microbatches=2052, optimizer_updates=1026, seen_padded_tokens=8480),
        physical_calls=0, base='frozen_official_Qwen3-0.6B_without_633_adapter')
    save(root/'evidence/tokenizer_plan.json', plan)
    save(root/'evidence/tokenizer_result.json', dict(status='tokenizer_preflight_passed', GPU_work_performed=False,
         physical_calls=0, config=submit.SFT_CONFIG, data_manifest_sha256=data_ref['sha256'],
         qwen_manifest_sha256=qwen_ref['sha256'], frozen_schedule_sha256=digest(root/'evidence/tokenizer_plan.json')))
    token_refs = dict(result=ref(root, root/'evidence/tokenizer_result.json'),
                      plan=ref(root, root/'evidence/tokenizer_plan.json'))
    token_refs['verification'] = audit(root, 'tokenizer', 'smoke_sft10.tokenizer_preflight',
        dict(data_manifest=data_ref['sha256'], qwen_manifest=qwen_ref['sha256'],
             result=token_refs['result']['sha256'], plan=token_refs['plan']['sha256'], source_sha256=source),
        dict(no_truncation=True, exact_schedule=True, official_base_only=True))
    auth = load(root/'AUTHORIZATION.json')
    auth.update(teacher_submission_released=False, teacher_submissions_used=1, additional_GPU_submissions_allowed=0,
                sft_submission_released=stage=='sft', grpo_submission_released=stage=='grpo')
    save(root/'AUTHORIZATION.json', auth)
    rows = []
    for name, sid, seconds in [('teacher', '659', 1500)] + ([('sft', '670', 4000)] if stage=='grpo' else []):
        row = dict(slurm_job_id=sid, stage=name, state='COMPLETED', gpus=1, allocation_gpu_count=1,
            cpu_cores=6, memory_GiB=90, hard_gpu_hours=submit.shared.CAPS[name], actual_card_hours=seconds/3600,
            sacct=dict(job_id=sid, state='COMPLETED', exit_code='0:0', elapsed_seconds=seconds,
                       alloc_tres='billing=6,cpu=6,gres/gpu:a100=1,gres/gpu=1,mem=90G,node=1'))
        rows.append(row)
        save(root/f'evidence/{name}_submissions.json', dict(schema='review10-v22-stage-submissions-v1',
            stage=name, max_submissions=1,claim_id=('c' if name=='teacher' else 'd')*32,jobs=[dict(slurm_job_id=sid)]))
        save(root/f'evidence/{name}_budget.json', dict(authorized_card_hours=submit.shared.CAPS[name],
            actual_card_hours=seconds/3600, actual_plus_remaining_caps=seconds/3600, remaining_caps=0,
            accounting_complete=True, accounting_unknown_job_ids=[], jobs=[row]))
    shutil.copyfile(HERE/'training_smoke/claude10/recovery_v22/BASELINE_BUDGET.json',root/'BASELINE_BUDGET.json')
    old = load(root/'BASELINE_BUDGET.json')['jobs']
    spent = sum(r['actual_card_hours'] for r in rows)
    remaining = 3.85 if stage=='sft' else 2
    for filename, cap, history, jobs in [('ROOT_POOL_PRE_SUBMISSION.json',10,812/3600,old+rows),
                                        ('PACKAGE_PRE_SUBMISSION.json',6.5,660/3600,rows)]:
        value = load(root/filename)
        value.update(authorized_card_hours=cap, actual_card_hours=history+spent,
                     actual_plus_remaining_caps=history+spent+remaining, remaining_caps=remaining,
                     new_actual_card_hours=spent, jobs=jobs, accounting_complete=True, accounting_unknown_job_ids=[])
        save(root/filename, value)
    manifest = dict(schema='review10-v22-downstream-stage-v1', stage=stage, remote_smoke_root=str(root),
        jobs=[submit.JOBS[stage]], source_sha256=source, driver_sha256={'select_real_driver.sh':submit.shared.DRIVER_SHA256},
        checkpoint=ref(root, model), protocol=ref(root, root/'PROTOCOL.json'),
        preflight_input=ref(root, root/'project/training_smoke/data/mace_initial_lattice.extxyz'),
        qwen_manifest=qwen_ref, qwen_model_dir='qwen', data_manifest=data_ref, data_verification=data_proof,
        baseline_budget=ref(root,root/'BASELINE_BUDGET.json'),
        teacher=dict(job_id='659', terminal='COMPLETED', recovered=True, result=teacher_ref,
                     verification=teacher_proof, claude_review=ref(root,root/'evidence/claude_review.json'),
                     submission_claim_id='c'*32,submission_ledger=ref(root,root/'evidence/teacher_submissions.json'),
                     budget=ref(root,root/'evidence/teacher_budget.json')),
        tokenizer_preflight=token_refs, submission_claim_id='b'*32)
    if stage == 'grpo':
        adapter = root/'evidence/sft_adapter'
        adapter.mkdir()
        save(adapter/'adapter_config.json', dict(r=16,lora_alpha=32,lora_dropout=.05,target_modules=['q_proj','v_proj']))
        (adapter/'adapter_model.safetensors').write_bytes(b'new fixture adapter, not 633')
        sft = dict(schema='review10-v22-sft-v1', status='completed', protocol=submit.shared.PROTOCOL,
            protocol_sha=submit.shared.PROTOCOL_SHA, seed=2026102301, config=submit.SFT_CONFIG,
            runtime=dict(slurm_job_id='670'), GPU_work_performed=True, physical_calls=0, training_runs=1,
            validation_retries=0, adapter_release_allowed=True, base_weight_sha256=submit.QWEN_WEIGHT_SHA,
            teacher_checkpoint_sha256=submit.shared.CHECKPOINT_SHA256, source_hashes=teacher['source_hashes'],
            data_manifest_sha256=data_ref['sha256'], qwen_manifest_sha256=qwen_ref['sha256'],
            training=dict(parameters_changed=True, adapter_hash_before='a'*64, adapter_hash_after='b'*64,
                optimizer_updates=1, updates=[dict(gradient_norm=1)]),
            reload=dict(save_unload_reload=True, unloaded_without_merge=True,reloaded_from_disk=True,state_exact_match=True,
                adapter_sha256=digest(adapter/'adapter_model.safetensors'), adapter_config_sha256=digest(adapter/'adapter_config.json')),
            legality=dict(status='passed', denominator=80, attempts=80, legal=76,
                records=[dict(index=i,status='completed',legal=i<76,text='fixture action') for i in range(80)]))
        save(root/'evidence/sft_result.json', sft)
        sft_ref = ref(root, root/'evidence/sft_result.json')
        sft_proof = audit(root, 'sft', 'smoke_grpo10.verify_release_manifest',
            dict(result=sft_ref['sha256'], data_manifest=data_ref['sha256'], source_sha256=source),
            dict(finite_gradients=True, parameters_changed=True, legality_replayed=True, adapter_reload=sft['reload']))
        manifest['sft'] = dict(job_id='670', terminal='COMPLETED', recovered=True, result=sft_ref,
            verification=sft_proof, budget=ref(root,root/'evidence/sft_budget.json'),
            submission_claim_id='d'*32,submission_ledger=ref(root,root/'evidence/sft_submissions.json'),
            adapter_files=[ref(root,p) for p in sorted(adapter.iterdir())])
        manifest['adapter_dir'] = 'evidence/sft_adapter'
        release = dict(schema='review10-v22-release-v1', seed=2026102301, protocol=submit.shared.PROTOCOL,
            protocol_sha=submit.shared.PROTOCOL_SHA, source_hashes={Path(k).name:v for k,v in source.items()},
            data_manifest=data_ref, qwen_manifest=qwen_ref, checkpoint=manifest['checkpoint'],
            teacher=manifest['teacher'], sft=manifest['sft'])
        save(root/'evidence/mace648_result.json',dict(status='completed'))
        save(root/'evidence/mace648_verification.json',dict(independently_verified=True))
        save(root/'evidence/fees_verification.json',dict(independently_verified=True))
        release.update(authorization=dict(review=10,recovery='v2.2',retry_of='650',authorized_retry_index=1,
            max_card_hours=2,additional_card_hours=6.5),
            mace648=dict(job_id='648',terminal='COMPLETED',recovered=True,
                result=ref(root,root/'evidence/mace648_result.json'),verification=ref(root,root/'evidence/mace648_verification.json')),
            cpu_preflight=dict(result=ref(root,root/'cpu_preflight/result.json'),verification=ref(root,root/'CPU_PREFLIGHT_VERIFIED.json')),
            root_fees=dict(verification=ref(root,root/'evidence/fees_verification.json'),
                root_budget=ref(root,root/'ROOT_POOL_PRE_SUBMISSION.json'),package_budget=ref(root,root/'PACKAGE_PRE_SUBMISSION.json'),
                baseline_budget=manifest['baseline_budget']))
        save(root/'RELEASE_MANIFEST.json', release)
        manifest['release_manifest'] = ref(root, root/'RELEASE_MANIFEST.json')
        manifest['release_verification'] = audit(root, 'release', 'smoke_grpo10.verify_release_manifest',
            dict(release_manifest=manifest['release_manifest']['sha256'], source_sha256=source),
            dict(verified=True, full_release_verified=True, manifest_sha256=manifest['release_manifest']['sha256'],
                 source_hashes=release['source_hashes'], all_evidence_hashes_checked=True,
                 verification_result=dict(verified=True,protocol=submit.shared.PROTOCOL,protocol_sha=submit.shared.PROTOCOL_SHA,
                    manifest_sha256=manifest['release_manifest']['sha256'],source_hashes=release['source_hashes'],
                    adapter_hashes={p.name:digest(p) for p in adapter.iterdir()},data_manifest_sha256=data_ref['sha256'],
                    qwen_manifest_sha256=qwen_ref['sha256'],checkpoint_sha256=manifest['checkpoint']['sha256'],
                    evidence={name:release[name] for name in ('sft','teacher','mace648','cpu_preflight','root_fees')})))
    save(root/'JOB_MANIFEST.json',manifest)
    reseal(root)
    return root


class DownstreamTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(submit, 'New standard-library downstream guard is required')
        _BLOCK_EXTERNAL.append(True)
        self.addCleanup(_BLOCK_EXTERNAL.pop)
        for owner in (submit, submit.shared):
            patcher = patch.object(owner,'sha256',side_effect=digest)
            patcher.start(); self.addCleanup(patcher.stop)
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = fixture(Path(self.tmp.name))
        self.pins = patch.object(submit,'FROZEN_SOURCES',
            {Path(k).name:v for k,v in load(self.root/'JOB_MANIFEST.json')['source_sha256'].items()})
        self.pins.start(); self.addCleanup(self.pins.stop)

    def scheduler(self, response='701;cluster\n', code=0, test_code=0, mutate=None):
        def run(args, **kwargs):
            if args[0]=='squeue': return subprocess.CompletedProcess(args,0,'901|ysz_other|RUNNING\n','')
            if args[:2]==['sbatch','--test-only']:
                if mutate: mutate()
                return subprocess.CompletedProcess(args,test_code,'checked','')
            if args[:2]==['sbatch','--parsable']:
                self.assertTrue(load(self.root/'SUBMISSIONS.json')['pending_submission'])
                self.assertTrue((self.root/'SUBMISSION_CLAIM.json').exists())
                if isinstance(response,Exception): raise response
                return subprocess.CompletedProcess(args,code,response,'')
            self.fail('Unexpected external call '+str(args))
        return run

    def reject(self, mutate, stage='sft'):
        mutate(); checksum(self.root)
        with patch.object(submit.subprocess,'run') as run:
            with self.assertRaises((ValueError,RuntimeError,FileNotFoundError)):
                submit.submit_once(self.root, stage)
            run.assert_not_called()
        self.assertFalse((self.root/'SUBMISSION_CLAIM.json').exists())

    def test_success_both_stages_controller_ledger_and_env(self):
        for stage in ('sft','grpo'):
            with self.subTest(stage=stage), tempfile.TemporaryDirectory() as tmp:
                self.root=fixture(Path(tmp),stage)
                with patch.dict(submit.os.environ, {'PATH':'/fixture/bin','SBATCH_GRES':'gpu:4','SLURM_CONF':'bad'},clear=True), \
                     patch('builtins.print'), \
                     patch.object(submit.subprocess,'run',side_effect=self.scheduler()) as run:
                    result=submit.main(['--stage',stage,'--root',str(self.root)])
                self.assertEqual(len(run.call_args_list),3)
                envs=[c.kwargs['env'] for c in run.call_args_list]
                self.assertTrue(all(env is envs[0] for env in envs))
                self.assertEqual(envs[0],{'PATH':'/fixture/bin'})
                calls=[c.args[0] for c in run.call_args_list]
                self.assertEqual(calls[1][2:],calls[2][2:])
                self.assertIn('--dependency=afterany:901',calls[2])
                job=result['jobs'][0]
                self.assertEqual(job['remote_output'],str(self.root/f'outputs/{stage}_701'))
                self.assertEqual(job['remote_log'],str(self.root/'slurm-701.out'))
                self.assertEqual((job['gpus'],job['cpu_cores'],job['memory_GiB']),(1,6,90))
                with patch.object(submit.subprocess,'run') as again:
                    with self.assertRaises(RuntimeError): submit.submit_once(self.root,stage)
                    again.assert_not_called()

    def test_consumes_claim_for_every_uncertain_actual_outcome(self):
        for response,code in [('garbage',0),('701\nextra',0),('701',1),('652',0),
                              ('659',0),(subprocess.TimeoutExpired(['sbatch'],60),0)]:
            with self.subTest(response=str(response)), tempfile.TemporaryDirectory() as tmp:
                self.root=fixture(Path(tmp))
                with patch.object(submit.subprocess,'run',side_effect=self.scheduler(response,code)):
                    with self.assertRaises((ValueError,RuntimeError)): submit.submit_once(self.root,'sft')
                self.assertTrue((self.root/'SUBMISSION_CLAIM.json').exists())
                self.assertTrue(load(self.root/'SUBMISSIONS.json')['pending_submission'])

    def test_scheduler_test_failure_and_hash_race_do_not_claim(self):
        for race in (False,True):
            with self.subTest(race=race), tempfile.TemporaryDirectory() as tmp:
                self.root=fixture(Path(tmp))
                mutate=(lambda:(self.root/'data/starts.jsonl').write_text('changed')) if race else None
                with patch.object(submit.subprocess,'run',side_effect=self.scheduler(test_code=0 if race else 1,mutate=mutate)):
                    with self.assertRaises((ValueError,RuntimeError)): submit.submit_once(self.root,'sft')
                self.assertFalse((self.root/'SUBMISSION_CLAIM.json').exists())

    def test_unreleased_stage_cannot_use_teacher_additional_zero_as_permission(self):
        def change():
            auth=load(self.root/'AUTHORIZATION.json');auth['sft_submission_released']=False
            save(self.root/'AUTHORIZATION.json',auth);reseal(self.root)
        self.reject(change)

    def test_incomplete_teacher_audit_claude_and_data_semantics_rejected(self):
        for file,key,value in [('teacher_raw.json','all_240_evaluations_verified',False),
            ('teacher_raw.json','complete_trajectories',15),('data_raw.json','disjoint_canonical_families',False),
            ('claude_review.json','verdict','REJECT')]:
            with self.subTest(file=file,key=key), tempfile.TemporaryDirectory() as tmp:
                self.root=fixture(Path(tmp))
                def change():
                    p=self.root/'evidence'/file;v=load(p);v[key]=value;save(p,v)
                self.reject(change)

    def test_path_escape_symlink_unlisted_file_old_claim_and_stop_rejected(self):
        cases=[lambda:(self.root/'unexpected.py').write_text('extra'),
               lambda:(self.root/'SUBMISSION_CLAIM.json').write_text('{}'),
               lambda:(self.root.parents[2]/'STOP_SMOKE.json').write_text('{}'),
               lambda:(self.root/'link').symlink_to(self.root/'data/manifest.json')]
        for index,change in enumerate(cases):
            with self.subTest(case=index), tempfile.TemporaryDirectory() as tmp:
                self.root=fixture(Path(tmp));change()
                with patch.object(submit.subprocess,'run') as run:
                    with self.assertRaises((ValueError,RuntimeError)): submit.submit_once(self.root,'sft')
                    run.assert_not_called()
        with tempfile.TemporaryDirectory() as tmp:
            self.root=fixture(Path(tmp))
            manifest=load(self.root/'JOB_MANIFEST.json');manifest['data_manifest']['path']='../escaped.json'
            save(self.root/'JOB_MANIFEST.json',manifest);checksum(self.root)
            with patch.object(submit.subprocess,'run') as run:
                with self.assertRaises((ValueError,RuntimeError)): submit.submit_once(self.root,'sft')
                run.assert_not_called()

    def test_unknown_bill_incomplete_fee_or_changed_historical_cost_rejected(self):
        for key,value in [('accounting_complete',False),('accounting_unknown_job_ids',['659']),
                          ('actual_plus_remaining_caps',100),('remaining_caps',2)]:
            with self.subTest(key=key), tempfile.TemporaryDirectory() as tmp:
                self.root=fixture(Path(tmp))
                def change():
                    budget=load(self.root/'ROOT_POOL_PRE_SUBMISSION.json');budget[key]=value
                    save(self.root/'ROOT_POOL_PRE_SUBMISSION.json',budget);reseal(self.root)
                self.reject(change)

    def test_exact_runner_commands_and_limits(self):
        m=load(self.root/'JOB_MANIFEST.json')
        script=submit.runner_scripts(self.root,'sft',m)
        self.assertIn('--time=01:51:00',script['sub.sh'])
        self.assertIn('6600s bash run.sh',script['sub.sh'])
        self.assertIn('smoke_sft10.py',script['run.sh'])
        self.assertNotIn('--adapter',script['run.sh'])
        self.assertIn('STOP_SMOKE.json',script['run.sh'])
        self.assertIn('--no-requeue',script['sub.sh'])
        self.assertEqual(submit.JOBS['grpo']['slurm_wall_seconds'],7200)
        self.assertEqual(submit.JOBS['grpo']['python_work_seconds'],6600)
        def change():
            (self.root/'run.sh').write_text(script['run.sh']+'echo unreviewed\n')
            m['stage_scripts_sha256']['run.sh']=digest(self.root/'run.sh');save(self.root/'JOB_MANIFEST.json',m)
        self.reject(change)

    def test_helper_is_hashed_before_import_and_sources_are_pinned(self):
        p=Path(self.tmp.name)/'untrusted.py'
        p.write_text('raise AssertionError("untrusted module executed")\n')
        with self.assertRaisesRegex(ValueError,'helper'):
            submit.load_helper(p)
        self.assertEqual(set(submit.FROZEN_SOURCES),submit.shared.PRODUCTION_SOURCES)
        imported={n.names[0].name.split('.')[0] for n in ast.walk(ast.parse(Path(submit.__file__).read_text()))
                  if isinstance(n,ast.Import)}
        self.assertFalse({'torch','numpy','paramiko'} & imported)

    def test_static_evidence_cannot_hide_in_dynamic_inventory_exclusions(self):
        p=self.root/'submission_checks/hidden.json';save(p,{'independently_verified':True})
        with self.assertRaises(ValueError): submit.checked(self.root,ref(self.root,p))

    def test_nested_json_reference_closure_rejects_escaped_or_dynamic_evidence(self):
        for target in ('../outside.json','submission_checks/hidden.json'):
            with self.subTest(target=target):
                p=self.root/target;save(p,{'unsafe':True})
                outer=self.root/'evidence/outer.json'
                save(outer,dict(inner=dict(path=target,sha256=digest(p))))
                with self.assertRaises(ValueError):
                    submit.check_references(self.root,dict(evidence=ref(self.root,outer)))

    def test_root_cannot_inject_a_slurm_directive(self):
        manifest=load(self.root/'JOB_MANIFEST.json')
        root=Path('/tmp/unsafe\n#SBATCH --gres=gpu:a100:4/training_smoke/claude10/recovery_v22/sft')
        with self.assertRaises(ValueError): submit.runner_scripts(root,'sft',manifest)
        with patch.object(submit.subprocess,'run') as run:
            with self.assertRaises(ValueError): submit.submit_once(root,'sft')
            run.assert_not_called()

    def test_semantic_json_cannot_use_an_extension_to_bypass_the_closure(self):
        m=load(self.root/'JOB_MANIFEST.json')
        raw=load(self.root/'evidence/claude_transcript.json')
        raw['nested_evidence']=dict(path='../outside.json',sha256='0'*64)
        save(self.root/'evidence/claude_transcript.dat',raw)
        review=load(self.root/'evidence/claude_review.json')
        review['raw_review']=ref(self.root,self.root/'evidence/claude_transcript.dat')
        save(self.root/'evidence/claude_review.dat',review)
        m['teacher']['claude_review']=ref(self.root,self.root/'evidence/claude_review.dat')
        save(self.root/'JOB_MANIFEST.json',m);reseal(self.root)
        with self.assertRaisesRegex(ValueError,'JSON'):
            submit.precheck(self.root,'sft')

    def test_jsonl_rows_cannot_hide_extra_escaped_references(self):
        m=load(self.root/'JOB_MANIFEST.json')
        p=self.root/'data/starts.jsonl';rows=p.read_text().splitlines();row=json.loads(rows[0])
        row['nested_evidence']=dict(path='../outside.json',sha256='0'*64)
        rows[0]=json.dumps(row);p.write_text('\n'.join(rows)+'\n')
        data=load(self.root/'data/manifest.json');data['files']['starts.jsonl']=digest(p)
        save(self.root/'data/manifest.json',data);m['data_manifest']=ref(self.root,self.root/'data/manifest.json')
        with self.assertRaisesRegex(ValueError,'Noncanonical|escaped'):
            submit.verify_qwen_data(self.root,m)

    def test_tokenizer_vocabulary_keys_are_not_evidence_references(self):
        p=self.root/'qwen/tokenizer.json'
        save(p,dict(model=dict(vocab={'path':100,'sha256':101,'ordinary':102})))
        submit.check_references(self.root,ref(self.root,p))

    def test_helper_cpu_json_and_jsonl_are_inside_the_reference_closure(self):
        for name in ('result.json','physics_receipts.jsonl'):
            with self.subTest(name=name),tempfile.TemporaryDirectory() as tmp:
                self.root=fixture(Path(tmp))
                p=self.root/'cpu_preflight'/name
                rows=p.read_text().splitlines() if name.endswith('.jsonl') else [p.read_text()]
                first=json.loads(rows[0]);first['nested_evidence']=dict(path='../outside.json',sha256='0'*64)
                rows[0]=json.dumps(first);p.write_text('\n'.join(rows)+'\n')
                cpu=load(self.root/'CPU_PREFLIGHT_VERIFIED.json');cpu['evidence_sha256']['cpu_preflight/'+name]=digest(p)
                save(self.root/'CPU_PREFLIGHT_VERIFIED.json',cpu);reseal(self.root)
                with self.assertRaisesRegex(ValueError,'Noncanonical|escaped'):
                    submit.precheck(self.root,'sft')

    def test_frozen_source_directory_cannot_contain_unbound_native_modules(self):
        p=self.root/'project/scripts/claude08_env.so';p.write_bytes(b'unbound module placeholder')
        checksum(self.root)
        with self.assertRaisesRegex(ValueError,'thirteen'):
            submit.precheck(self.root,'sft')

    def test_closed_snapshots_cannot_have_pending_or_collection_error(self):
        for key,value in [('pending_submission',True),('collection_error','unresolved')]:
            with self.subTest(key=key),tempfile.TemporaryDirectory() as tmp:
                self.root=fixture(Path(tmp))
                p=self.root/'ROOT_POOL_PRE_SUBMISSION.json';budget=load(p);budget[key]=value
                save(p,budget);reseal(self.root)
                with self.assertRaisesRegex(ValueError,'Closed'):
                    submit.precheck(self.root,'sft')

    def test_duplicate_or_malformed_tres_is_rejected_before_dictionary_conversion(self):
        for prefix in ('cpu=12,gres/gpu=4,gres/gpu:a100=4,mem=180G,','cpu=6,','cpu=6=12,'):
            with self.subTest(prefix=prefix),tempfile.TemporaryDirectory() as tmp:
                self.root=fixture(Path(tmp))
                m=load(self.root/'JOB_MANIFEST.json')
                p=self.root/'evidence/teacher_budget.json';budget=load(p);row=budget['jobs'][0]
                row['sacct']['alloc_tres']=prefix+row['sacct']['alloc_tres'];save(p,budget)
                m['teacher']['budget']=ref(self.root,p)
                for name in ('ROOT_POOL_PRE_SUBMISSION.json','PACKAGE_PRE_SUBMISSION.json'):
                    p=self.root/name;value=load(p)
                    value['jobs']=[copy.deepcopy(row) if str(r['slurm_job_id'])=='659' else r for r in value['jobs']]
                    save(p,value)
                with self.assertRaisesRegex(ValueError,'TRES'):
                    submit.verify_fees(self.root,m,'sft',{'teacher':'659'})

    def test_existing_frozen_cpu_proof_metadata_is_compatible_without_rewriting(self):
        original=HERE/'training_smoke/claude10/recovery_v22/teacher/input_bundle'
        originals=[original/'CPU_PREFLIGHT_VERIFIED.json']+list((original/'cpu_preflight').rglob('*'))+list((original/'reviews').rglob('*'))
        before={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in originals if p.is_file()}
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve()/'training_smoke/claude10/recovery_v22/sft'
            root.mkdir(parents=True)
            for directory in ('cpu_preflight','reviews'):
                shutil.copytree(original/directory,root/directory)
            for name in ('CPU_PREFLIGHT_VERIFIED.json','PROTOCOL.json','select_real_driver.sh'):
                shutil.copyfile(original/name,root/name)
            manifest=load(original/'JOB_MANIFEST.json')
            name=manifest['preflight_input']['path']
            (root/name).parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(original/name,root/name)
            # This is only a read-only CPU-evidence compatibility probe. No
            # teacher result, training rows, official base, claim, or job input
            # bundle is prepared, and the checkpoint/model is never opened.
            hashes={p.relative_to(root).as_posix():digest(p) for p in root.rglob('*') if p.is_file()}
            submit.verify_static_documents(root,hashes)
            proof=submit.shared.verify_cpu(root,manifest,submit.shared.SELECTED['relaxation'])
            self.assertEqual((proof['raw_ef'],proof['forward_calls']),(22,13))
        self.assertEqual(before,{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in originals if p.is_file()})

    def test_boolean_resource_counters_do_not_equal_integer_one(self):
        def change():
            m=load(self.root/'JOB_MANIFEST.json');m['jobs'][0]['gpus']=True
            save(self.root/'JOB_MANIFEST.json',m);reseal(self.root)
        self.reject(change)

    def test_semantically_bad_raw_teacher_report_is_rejected_after_rehash(self):
        m=load(self.root/'JOB_MANIFEST.json')
        raw=self.root/'evidence/teacher_raw.json';value=load(raw);value['verified_record_ids']=list(range(239));save(raw,value)
        p=self.root/'evidence/teacher_verified.json';proof=load(p);proof['raw_validation']=ref(self.root,raw);save(p,proof)
        m['teacher']['verification']=ref(self.root,p)
        with self.assertRaisesRegex(ValueError,'per-record'):
            submit.verify_teacher(self.root,m)

    def test_raw_data_semantics_and_tokenizer_schedule_rejected_after_rehash(self):
        m=load(self.root/'JOB_MANIFEST.json')
        raw=self.root/'evidence/data_raw.json';value=load(raw);value['disjoint_canonical_families']=False;save(raw,value)
        p=self.root/'evidence/data_verified.json';proof=load(p);proof['raw_validation']=ref(self.root,raw);save(p,proof)
        m['data_verification']=ref(self.root,p)
        with self.assertRaisesRegex(ValueError,'semantic'):
            submit.verify_qwen_data(self.root,m)

    def test_reviewed_source_config_and_limits_match_actual_production(self):
        # Parse literals without importing a model/scientific dependency.
        tree=ast.parse((HERE/'review10_v22/scripts/smoke_sft10.py').read_text())
        config=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign)
                    and any(isinstance(t,ast.Name) and t.id=='CONFIG' for t in n.targets))
        self.assertEqual(submit.SFT_CONFIG,config)
        actual={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (HERE/'review10_v22/scripts').glob('*.py')
                if p.name in submit.shared.PRODUCTION_SOURCES}
        self.assertEqual(actual,CANONICAL_SOURCE_PINS)
        self.assertEqual(actual,load(HERE/'training_smoke/claude10/recovery_v22/AUTHORIZATION.json')['core_source_sha256'])

    def test_grpo_failed_or_old_sft_and_missing_release_proof_fail(self):
        for target in ('sft_result','release_raw'):
            with self.subTest(target=target), tempfile.TemporaryDirectory() as tmp:
                self.root=fixture(Path(tmp),'grpo')
                def change():
                    p=self.root/f'evidence/{target}.json';v=load(p)
                    if target=='sft_result': v['legality']['legal']=75
                    else: v['full_release_verified']=False
                    save(p,v)
                self.reject(change,'grpo')


if __name__=='__main__': unittest.main()
