"""Temporary fixtures only: no scheduler, network, or model is invoked."""
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
try:
    submit = importlib.import_module('review10_v21_teacher_submit')
except ModuleNotFoundError as error:
    if error.name != 'review10_v21_teacher_submit':
        raise
    submit = None


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True))


def load(path):
    return json.loads(path.read_text())


def checksum(root):
    excluded = {'INPUTS.sha256'}
    files = sorted(p for p in root.rglob('*') if p.is_file() and p.name not in excluded)
    (root/'INPUTS.sha256').write_text(''.join(digest(p)+'  '+p.relative_to(root).as_posix()+'\n' for p in files))


def fixture(base):
    base=base.resolve()
    root = base/'training_smoke/claude10/recovery_v21/teacher'
    root.mkdir(parents=True)
    source_names = ['claude10_physics.py','claude08_env.py','smoke_mace.py',
                    'smoke_grpo08.py','smoke_teacher10.py','claude10_data.py']
    for name in source_names:
        p = root/'project/scripts'/name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text('# frozen fixture '+name+'\n')
    source = {'project/scripts/'+name:digest(root/'project/scripts'/name) for name in source_names}
    model = base/'shared.model'
    model.write_bytes(b'frozen model fixture; never loaded')
    spec = dict(model_sha=digest(model), dtype='float64', optimizer='LBFGS', fmax=.03,
        steps=300, maxstep=.1, memory=100, alpha=70., damping=1., use_line_search=False,
        displacement_sigma=.02, displacement_master_seed=2026102301,
        graph_version='native_smoke_mace_build_batch_v1')
    protocol = dict(protocol='YSZ21_terminal_local2_lbfgs_noise_v2.1', relaxation=spec)
    save(root/'PROTOCOL.json', protocol)
    input_path = root/'project/training_smoke/data/mace_initial_lattice.extxyz'
    input_path.parent.mkdir(parents=True)
    species=('Zr 0 0 0\n'*92)+('Y 0 0 0\n'*16)+('O 0 0 0\n'*208)
    geometry = '316\nfixture geometry\n'+species
    input_path.write_text(('324\nfixture parent occupancy\n'+species+('X 0 0 0\n'*8))*4)
    for size in (1,4):
        for lane in range(size):
            p = root/f'cpu_preflight/lbfgs_one_step_b{size}_lane{lane}.extxyz'
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(geometry)
    core = {name:source['project/scripts/'+name] for name in source_names[:4]}
    receipt = dict(nsteps=1, optimizer='LBFGS', dtype='float64', protocol=protocol['protocol'],
        protocol_sha=hashlib.sha256(json.dumps(protocol,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
        checkpoint_sha256=digest(model), source_hashes=core, device='cpu',
        execution_max_steps=1, formal_receipt=False, relaxation_spec=spec)
    raw = dict(status='cpu_execution_preflight_passed', production_code_path=True,
        runtime=dict(device='cpu',GPU_work_performed=False), source_hashes=core,
        checkpoint_sha256=digest(model), input_sha256=digest(input_path),
        protocol=protocol['protocol'], relaxation_spec=spec,
        counters=dict(raw_ef=22,forward_calls=13), actual_forward_audit=dict(observed_raw_ef=22,
            observed_forward_calls=13,all_precharged=True,matches_counter=True),
        parity=[dict(lane=i,batch_single_energy_difference_eV=0.,batch_single_force_difference_eV_A=0.,
            batch_native_energy_difference_eV=0.,batch_native_force_difference_eV_A=0.) for i in range(4)],
        singleton_energies_eV=[0.]*4,native_energies_eV=[0.]*4,
        relaxations={'1':[copy.deepcopy(receipt)],'4':[copy.deepcopy(receipt) for _ in range(4)]})
    save(root/'cpu_preflight/result.json',raw)
    audit=[]
    cumulative=0
    for i,size in enumerate([1]*4+[4]+[1]*4+[1]*2+[4]*2,1):
        cumulative+=size
        audit.append(dict(batch_size=size,forward_call=i,raw_ef_cumulative=cumulative,precharged=True))
    (root/'cpu_preflight/actual_model_forwards.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in audit))
    evidence={p.relative_to(root).as_posix():digest(p) for p in (root/'cpu_preflight').iterdir()}
    cpu=dict(independently_verified=True,CPU_only=True,same_production_code_path=True,
        batch_sizes=[1,4],raw_ef=22,forward_calls=13,source_sha256=source,
        checkpoint_sha256=digest(model),protocol_sha256=digest(root/'PROTOCOL.json'),
        input_sha256=digest(input_path),evidence_sha256=evidence)
    save(root/'CPU_PREFLIGHT_VERIFIED.json',cpu)
    auth=dict(schema='review10-652-recovery-authorization-v1',direct_user_confirmation=True,
        Claude_protocol_approved=True,teacher_submission_released=True,
        package_total_cap_including652_card_hours=6,root_pool_card_hours=10,
        max_concurrent_GPU_jobs=1,max_new_teacher_submissions=1,automatic_retry=False,
        formal_training_allowed=False,new_DFT_allowed=False,
        stage_caps_card_hours={'teacher':2,'sft':1.85,'grpo':2},
        selected_protocol=dict(version='v2.1',optimizer='LBFGS',memory=100,alpha=70,damping=1,
            use_line_search=False,fmax=.03,max_steps=300,maxstep=.1,dtype='float64',
            perturbation_sigma_A=.02,master_seed=2026102301,graph='native',
            seed_derivation='master+numberedrequest+protocolsha withoutarm',
            unmappable_max_fraction=.05,near_threshold_O_distance_A=1.1))
    save(root/'AUTHORIZATION.json',auth)
    old={'633':35,'635':9,'648':36,'650':72,'652':507}
    root_budget=dict(authorized_card_hours=10,actual_card_hours=659/3600,remaining_caps=5.85,
        actual_plus_remaining_caps=659/3600+5.85,accounting_complete=True,accounting_unknown_job_ids=[],
        review10_reservation_card_hours=5.85,jobs=[dict(slurm_job_id=sid,actual_card_hours=seconds/3600,
            allocation_gpu_count=1,state='COMPLETED' if sid not in ('635','652') else 'FAILED')
            for sid,seconds in old.items()])
    save(root/'ROOT_POOL_PRE_SUBMISSION.json',root_budget)
    save(root/'PACKAGE_PRE_SUBMISSION.json',dict(authorized_card_hours=6,actual_card_hours=507/3600,
        actual_plus_remaining_caps=507/3600+5.85,remaining_caps=5.85,new_actual_card_hours=0,
        accounting_complete=True,accounting_unknown_job_ids=[],jobs=[],stage_caps=auth['stage_caps_card_hours']))
    job=dict(id='c10_v21_teacher',name='ysz_c10v21_teacher',gpus=1,cpu_cores=6,memory_GiB=90,
        hard_gpu_hours=2,slurm_wall_seconds=7200,outer_timeout_seconds=7140,python_work_seconds=6600,
        raw_ef_call_cap=72240,actual_relaxation_cap=240,action_budget=224,seed=2026102301,
        teacher_trajectories=16,steps_per_trajectory=8)
    directives=dict(nodes='1',partition='a100',ntasks='1',**{'ntasks-per-node':'1','cpus-per-task':'6',
        'gres':'gpu:a100:1','mem':'90G','time':'02:00:00','job-name':job['name'],'chdir':str(root),
        'output':'slurm-%j.out'})
    (root/'sub.sh').write_text('#!/bin/bash\n'+''.join(f'#SBATCH --{k}={v}\n' for k,v in directives.items())+
        '#SBATCH --no-requeue\nset -euo pipefail\nexec timeout --signal=TERM --kill-after=30s 7140s bash run.sh\n')
    (root/'run.sh').write_text('#!/bin/bash\n# independently reviewed runner fixture\n')
    shutil.copyfile(Path(submit.__file__),root/'submit_once.py')
    scripts={name:digest(root/name) for name in ('sub.sh','run.sh')}
    bindings=dict(source_sha256=source,stage_scripts_sha256=scripts,checkpoint_sha256=digest(model),
        protocol_sha256=digest(root/'PROTOCOL.json'),wrapper_sha256=digest(root/'submit_once.py'))
    reviews={}
    for kind,verdict in (('spec','PASS'),('quality','APPROVE')):
        p=root/f'reviews/{kind}.json'
        save(p,dict(independently_reviewed=True,verdict=verdict,**bindings))
        reviews[kind]=dict(path=p.relative_to(root).as_posix(),sha256=digest(p))
    save(root/'PRE_SUBMIT_GATE.json',dict(schema='review10-v21-teacher-pre-submit-v1',
        independently_verified=True,**bindings,review_evidence=reviews,
        cpu_preflight_verification_sha256=digest(root/'CPU_PREFLIGHT_VERIFIED.json')))
    manifest=dict(schema='review10-v21-teacher-stage-v1',stage='teacher',remote_smoke_root=str(root),jobs=[job],
        source_sha256=source,stage_scripts_sha256=scripts,
        checkpoint=dict(path=str(model),sha256=digest(model)),
        protocol=dict(path='PROTOCOL.json',sha256=digest(root/'PROTOCOL.json')),
        preflight_input=dict(path=input_path.relative_to(root).as_posix(),sha256=digest(input_path)),
        submission_claim_id='a'*32)
    save(root/'JOB_MANIFEST.json',manifest)
    save(root/'SUBMISSION_INTENT_ID.json',dict(schema='review10-v21-local-intent-v1',claim_id='a'*32,
        stage='teacher',max_submissions=1,durable_locally=True,manifest_sha256=digest(root/'JOB_MANIFEST.json')))
    checksum(root)
    return root


class TeacherSubmitTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(submit,'New pure-standard-library submission wrapper required')
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root=fixture(Path(self.tmp.name))

    def scheduler(self, queue='901|ysz_other|RUNNING\n902|foreign|RUNNING\n903|ysz_pending|PENDING\n',
                  response='701;cluster\n', code=0, test_code=0):
        def run(args,**kwargs):
            if args[0]=='squeue':
                return subprocess.CompletedProcess(args,0,queue,'')
            if args[:2]==['sbatch','--test-only']:
                return subprocess.CompletedProcess(args,test_code,'validation','')
            if args[:2]==['sbatch','--parsable']:
                self.assertTrue((self.root/'SUBMISSION_CLAIM.json').is_file())
                pending=load(self.root/'SUBMISSIONS.json')
                self.assertEqual(pending['claim_id'],'a'*32)
                self.assertTrue(pending['pending_submission'])
                return subprocess.CompletedProcess(args,code,response,'scheduler stderr')
            self.fail('Unexpected external command: '+repr(args))
        return run

    def test_success_uses_identical_serial_dependencies_and_controller_compatible_ledger(self):
        with patch.object(submit.subprocess,'run',side_effect=self.scheduler()) as run:
            submit.main(['--root',str(self.root)])
        commands=[c.args[0] for c in run.call_args_list]
        self.assertEqual(len(commands),3)
        self.assertEqual(commands[1][2:],commands[2][2:])
        self.assertIn('--dependency=afterany:901:903',commands[2])
        self.assertIn('--kill-on-invalid-dep=yes',commands[2])
        ledger=load(self.root/'SUBMISSIONS.json')
        self.assertNotIn('pending_submission',ledger)
        self.assertEqual(ledger['claim_id'],'a'*32)
        job=ledger['jobs'][0]
        self.assertEqual(job['slurm_job_id'],'701')
        self.assertEqual(job['remote_output'],str(self.root/'outputs/teacher_701'))
        self.assertEqual(job['remote_log'],str(self.root/'slurm-701.out'))
        self.assertEqual(job['hard_gpu_hours'],2)
        with patch.object(submit.subprocess,'run') as run:
            with self.assertRaises(RuntimeError): submit.main(['--root',str(self.root)])
            run.assert_not_called()

    def test_any_existing_remote_ledger_is_never_overwritten(self):
        original='{"jobs": []}\n'
        (self.root/'SUBMISSIONS.json').write_text(original)
        with patch.object(submit.subprocess,'run') as run:
            with self.assertRaises(RuntimeError): submit.main(['--root',str(self.root)])
            run.assert_not_called()
        self.assertEqual((self.root/'SUBMISSIONS.json').read_text(),original)

    def test_queue_and_both_sbatch_calls_share_filtered_environment_and_real_dependencies(self):
        polluted={'PATH':'/approved/bin','HOME':'/fixture/home','SSH_AUTH_SOCK':'/fixture/agent',
            'SBATCH_GRES':'gpu:a100:2','SBATCH_CPUS_PER_TASK':'12','SBATCH_DEPENDENCY':'afterok:99',
            'SBATCH_EXPORT':'ALL','SLURM_JOB_ID':'99','SLURM_NTASKS':'12','SLURM_CPUS_PER_TASK':'12',
            'SLURM_JOB_GPUS':'0,1','SLURM_DEPENDENCY':'afterok:99','SLURM_CLUSTERS':'other',
            'SLURM_CONF':'/unreviewed/slurm.conf'}
        normal=self.scheduler()
        def scheduler_with_environment_change(args,**kwargs):
            if args[0]=='squeue':
                query_env=kwargs.get('env',submit.os.environ)
                if 'SLURM_CLUSTERS' in query_env or 'SLURM_CONF' in query_env:
                    return subprocess.CompletedProcess(args,0,'','')
            response=normal(args,**kwargs)
            if args[:2]==['sbatch','--test-only']:
                submit.os.environ['SBATCH_GRES']='gpu:a100:4'
                submit.os.environ['SLURM_CPUS_PER_TASK']='24'
            return response
        with patch.dict(submit.os.environ,polluted,clear=True), \
             patch.object(submit.subprocess,'run',side_effect=scheduler_with_environment_change) as run:
            submit.main(['--root',str(self.root)])
            sbatch=[call for call in run.call_args_list if call.args[0][0]=='sbatch']
            self.assertEqual(len(sbatch),2)
            test_env=sbatch[0].kwargs.get('env')
            actual_env=sbatch[1].kwargs.get('env')
            self.assertIsInstance(test_env,dict)
            self.assertIs(test_env,actual_env)
            self.assertTrue(all(call.kwargs.get('env') is test_env for call in run.call_args_list))
            self.assertIn('--dependency=afterany:901:903',sbatch[1].args[0])
            self.assertFalse(any(key.startswith(('SBATCH_','SLURM_')) for key in test_env))
            self.assertEqual(test_env,{'PATH':'/approved/bin','HOME':'/fixture/home','SSH_AUTH_SOCK':'/fixture/agent'})
            self.assertEqual(submit.os.environ['SBATCH_GRES'],'gpu:a100:4')
            self.assertEqual(submit.os.environ['SLURM_CPUS_PER_TASK'],'24')
            check=load(next((self.root/'submission_checks').glob('scheduler_test_*.json')))
            self.assertEqual(check['environment_filter'],dict(removed_prefixes=['SBATCH_','SLURM_'],
                             remaining_variable_names=['HOME','PATH','SSH_AUTH_SOCK']))
            self.assertNotIn('/fixture/agent',json.dumps(check))

    def test_ambiguous_or_nonzero_response_consumes_claim_without_second_submission(self):
        for output,code in (('not an ID',0),('701\nextra',0),('701',1)):
            with self.subTest(output=output,code=code):
                with tempfile.TemporaryDirectory() as tmp:
                    self.root=fixture(Path(tmp))
                    with patch.object(submit.subprocess,'run',side_effect=self.scheduler(response=output,code=code)) as run:
                        with self.assertRaises(RuntimeError): submit.main(['--root',str(self.root)])
                        self.assertEqual(sum(c.args[0][:2]==['sbatch','--parsable'] for c in run.call_args_list),1)
                    self.assertTrue((self.root/'SUBMISSION_CLAIM.json').exists())
                    self.assertTrue(load(self.root/'SUBMISSIONS.json')['pending_submission'])
                    with patch.object(submit.subprocess,'run') as run:
                        with self.assertRaises(RuntimeError): submit.main(['--root',str(self.root)])
                        run.assert_not_called()

    def test_actual_timeout_consumes_claim_and_persists_pending_intent(self):
        normal=self.scheduler()
        def timed_out(args,**kwargs):
            if args[:2]==['sbatch','--parsable']:
                raise subprocess.TimeoutExpired(args,60)
            return normal(args,**kwargs)
        with patch.object(submit.subprocess,'run',side_effect=timed_out) as run:
            with self.assertRaises(RuntimeError): submit.main(['--root',str(self.root)])
        self.assertEqual(len(run.call_args_list),3)
        self.assertTrue((self.root/'SUBMISSION_CLAIM.json').exists())
        self.assertTrue(load(self.root/'SUBMISSIONS.json')['pending_submission'])
        self.assertTrue(any(load(p).get('state')=='SUBMISSION_UNCERTAIN'
                            for p in (self.root/'submission_checks').glob('actual_response_*.json')))
        with patch.object(submit.subprocess,'run') as run:
            with self.assertRaises(RuntimeError): submit.main(['--root',str(self.root)])
            run.assert_not_called()

    def test_queue_failure_prevents_test_and_claim(self):
        with patch.object(submit.subprocess,'run',side_effect=subprocess.CalledProcessError(1,['squeue'])) as run:
            with self.assertRaises(subprocess.CalledProcessError): submit.main(['--root',str(self.root)])
        run.assert_called_once()
        self.assertFalse((self.root/'SUBMISSION_CLAIM.json').exists())

    def test_empty_queue_submits_without_dependency_arguments(self):
        with patch.object(submit.subprocess,'run',side_effect=self.scheduler(queue='')) as run:
            submit.main(['--root',str(self.root)])
        self.assertEqual(run.call_args_list[-1].args[0],['sbatch','--parsable',str(self.root/'sub.sh')])

    def test_exclusive_claim_race_never_runs_actual_sbatch(self):
        normal=self.scheduler()
        def competing_claim(args,**kwargs):
            result=normal(args,**kwargs)
            if args[:2]==['sbatch','--test-only']:
                save(self.root/'SUBMISSION_CLAIM.json',dict(claim_id='b'*32))
            return result
        with patch.object(submit.subprocess,'run',side_effect=competing_claim) as run:
            with self.assertRaises(RuntimeError): submit.main(['--root',str(self.root)])
        self.assertEqual(len(run.call_args_list),2)
        self.assertEqual(load(self.root/'SUBMISSION_CLAIM.json')['claim_id'],'b'*32)
        self.assertFalse((self.root/'SUBMISSIONS.json').exists())

    def test_native_and_LBFGS_facts_checked_even_when_all_evidence_hashes_are_refreshed(self):
        cases=[lambda raw:raw['actual_forward_audit'].update(observed_raw_ef=0),
               lambda raw:raw['counters'].update(raw_ef=0),
               lambda raw:raw['parity'][0].update(batch_native_force_difference_eV_A=.1),
               lambda raw:raw['relaxations']['4'][2].update(nsteps=0)]
        for mutate in cases:
            with tempfile.TemporaryDirectory() as tmp:
                self.root=fixture(Path(tmp))
                raw=load(self.root/'cpu_preflight/result.json');mutate(raw);save(self.root/'cpu_preflight/result.json',raw)
                proof=load(self.root/'CPU_PREFLIGHT_VERIFIED.json')
                proof['evidence_sha256']['cpu_preflight/result.json']=digest(self.root/'cpu_preflight/result.json')
                save(self.root/'CPU_PREFLIGHT_VERIFIED.json',proof)
                gate=load(self.root/'PRE_SUBMIT_GATE.json')
                gate['cpu_preflight_verification_sha256']=digest(self.root/'CPU_PREFLIGHT_VERIFIED.json')
                save(self.root/'PRE_SUBMIT_GATE.json',gate);checksum(self.root)
                with patch.object(submit.subprocess,'run') as run:
                    with self.assertRaises(ValueError): submit.main(['--root',str(self.root)])
                    run.assert_not_called()

    def test_indented_Slurm_override_cannot_bypass_fixed_resources(self):
        script=self.root/'sub.sh'
        script.write_text(script.read_text().replace('#SBATCH --gres=gpu:a100:1',
                                                   '#SBATCH --gres=gpu:a100:1\n  #SBATCH --gres=gpu:a100:2'))
        manifest=load(self.root/'JOB_MANIFEST.json')
        manifest['stage_scripts_sha256']['sub.sh']=digest(script)
        with self.assertRaises(ValueError): submit.verify_scripts(self.root,manifest)

    def test_test_only_failure_creates_no_actual_submission_claim(self):
        with patch.object(submit.subprocess,'run',side_effect=self.scheduler(test_code=1)) as run:
            with self.assertRaises(RuntimeError): submit.main(['--root',str(self.root)])
        self.assertEqual(len(run.call_args_list),2)
        self.assertFalse((self.root/'SUBMISSION_CLAIM.json').exists())
        self.assertFalse((self.root/'SUBMISSIONS.json').exists())

    def test_malformed_or_existing_target_queue_refuses_before_test_or_claim(self):
        for queue in ('901_2|ysz_other|PENDING\n','901|ysz_c10v21_teacher|RUNNING\n','malformed\n'):
            with self.subTest(queue=queue), patch.object(submit.subprocess,'run',side_effect=self.scheduler(queue=queue)) as run:
                with self.assertRaises(ValueError): submit.main(['--root',str(self.root)])
                self.assertEqual(len(run.call_args_list),1)
                self.assertFalse((self.root/'SUBMISSION_CLAIM.json').exists())

    def test_precheck_rejects_stop_source_hash_unlisted_file_and_resource_tamper(self):
        for kind in ('stop','source','unlisted','resource','wrapper'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as tmp:
                self.root=fixture(Path(tmp))
                if kind=='stop': (self.root.parents[2]/'STOP_SMOKE.json').write_text('{}')
                if kind=='source': (self.root/'project/scripts/smoke_teacher10.py').write_text('# tampered\n')
                if kind=='unlisted': (self.root/'extra.py').write_text('# unreviewed\n')
                if kind=='wrapper': (self.root/'submit_once.py').write_text('# wrong wrapper\n')
                if kind=='resource':
                    p=self.root/'sub.sh';p.write_text(p.read_text().replace('--gres=gpu:a100:1','--gres=gpu:a100:2'))
                    checksum(self.root)
                with patch.object(submit.subprocess,'run') as run:
                    with self.assertRaises((ValueError,RuntimeError)): submit.main(['--root',str(self.root)])
                    run.assert_not_called()

    def test_rehashing_source_does_not_bypass_independent_review_bindings(self):
        p=self.root/'project/scripts/smoke_teacher10.py';p.write_text('# tampered\n')
        checksum(self.root)
        with patch.object(submit.subprocess,'run') as run:
            with self.assertRaises(ValueError): submit.main(['--root',str(self.root)])
            run.assert_not_called()

    def test_auth_review_cpu_facts_and_full_fee_reservations_are_required(self):
        cases=[('AUTHORIZATION.json',lambda j:j.update(teacher_submission_released=False)),
               ('PRE_SUBMIT_GATE.json',lambda j:j['review_evidence']['spec'].update(sha256='0'*64)),
               ('CPU_PREFLIGHT_VERIFIED.json',lambda j:j.update(raw_ef=0)),
               ('cpu_preflight/result.json',lambda j:j['counters'].update(raw_ef=0)),
               ('ROOT_POOL_PRE_SUBMISSION.json',lambda j:j.update(accounting_complete=False)),
               ('ROOT_POOL_PRE_SUBMISSION.json',lambda j:j.update(remaining_caps=3.85)),
               ('PACKAGE_PRE_SUBMISSION.json',lambda j:j.update(actual_card_hours=0))]
        for name,mutate in cases:
            with self.subTest(name=name),tempfile.TemporaryDirectory() as tmp:
                self.root=fixture(Path(tmp));p=self.root/name;j=load(p);mutate(j);save(p,j);checksum(self.root)
                with patch.object(submit.subprocess,'run') as run:
                    with self.assertRaises(ValueError): submit.main(['--root',str(self.root)])
                    run.assert_not_called()


if __name__=='__main__': unittest.main()
