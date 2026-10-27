"""Fixed Review08 one-update smoke; CPU injectable, real backends lazy.

There is no submission/preparation command. The CLI is a future *in-job* runner
and refuses model loading outside one Slurm CUDA allocation or without hashed,
independently recovered 633/635 evidence. No GPU dependency imports on import.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass, replace
from contextlib import contextmanager
import importlib.metadata
import sys
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import time
import traceback
from typing import Callable
import numpy as np
import claude08_env as env
from claude08_data import sha256, unscored_state
from claude08_grpo import ActionSample, group_advantages, one_update, causal_logprobs, state_fingerprint

SEED = 2026102301
GROUPS = 4
GROUP_SIZE = 4
ATTEMPTS = 8
MAX_NEW_TOKENS = 64
MAX_PROMPT_TOKENS = 1984
MAX_TOTAL_TOKENS = 2048
INTERNAL_SECONDS = 6600
FUTURE_OUTER_SECONDS = 7140
FUTURE_SLURM_SECONDS = 7200
QWEN_WEIGHT_SHA = 'f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b'
LIMITS = env.BudgetLimits(max_attempts=8, max_relax_actual=132, max_raw_ef=50000,
        max_generated_tokens=9472, max_prefill_tokens=293632, max_seen_tokens=262144,
        wall_seconds=INTERNAL_SECONDS)


def save_json(path, value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def _append(path, row):
    if path is not None:
        with Path(path).open('a') as handle: handle.write(json.dumps(row,allow_nan=False)+'\n')


class TokenLedger:
    """Shared actual-cost ledger; no reservations or refunds after failed calls.

    A full-context generation/reference/current forward charges all input tokens
    as prefill AND seen before execution. Each generated token charges new+seen
    before sampling/decoding, including EOS. KV decode does not charge that
    token twice. Environments receive zero token deltas and own attempt/physics
    charging only. All caps and elapsed time include swaps, loading and saving.
    """
    def __init__(self,counters,limits=LIMITS,*,started_at=None,clock=time.monotonic,log_path=None):
        self.counters=counters; self.limits=limits; self.clock=clock
        self.started_at=clock() if started_at is None else started_at
        self.events=[]; self.phase_timings=[]; self.log_path=log_path
        self.raw=env.RawEFCounter(counters,limits,started_at=self.started_at,clock=clock)

    def check(self):
        self.raw.check_time()
        for name in ('relax_actual','raw_ef','generated_tokens','prefill_tokens','seen_tokens'):
            bound=getattr(self.limits,'max_'+name)
            if bound is not None and getattr(self.counters,name)>=bound:
                raise env.BudgetExceeded(name+' hard cap reached')

    def _charge(self,delta,phase):
        self.check()
        for name,n in delta.items():
            if type(n) is not int or n<0: raise ValueError('Token charges must be nonnegative integers')
            bound=getattr(self.limits,'max_'+name)
            if bound is not None and getattr(self.counters,name)+n>bound:
                raise env.BudgetExceeded(name+' cap before '+phase)
        for name,n in delta.items(): setattr(self.counters,name,getattr(self.counters,name)+n)
        row={'phase':phase,'delta':delta,'counts':asdict(self.counters),'elapsed_seconds':self.clock()-self.started_at}
        self.events.append(row); _append(self.log_path,row)

    def before_prefill(self,n,*,phase): self._charge({'prefill_tokens':n,'seen_tokens':n},phase)
    def before_generated_token(self,*,phase): self._charge({'generated_tokens':1,'seen_tokens':1},phase)


@dataclass(frozen=True)
class GeneratedAction:
    text: str
    input_ids: tuple[int,...]
    action_mask: tuple[bool,...]
    old_logprobs: tuple[float,...]
    reference_logprobs: tuple[float,...]

    def sample(self):
        return ActionSample(self.input_ids,self.action_mask,self.old_logprobs,self.reference_logprobs)


def decode_action_tokens(tokenizer,generated):
    """Only final EOS is a delimiter; every other sampled control token stays."""
    tokens=list(generated)
    if tokens and tokens[-1]==tokenizer.eos_token_id: tokens=tokens[:-1]
    return tokenizer.decode(tokens,skip_special_tokens=False).strip()


def require_single_gpu_job(*,environ=None,torch_module=None):
    """Must be called before any real model load, including CPU model loads."""
    environ=os.environ if environ is None else environ
    visible=environ.get('CUDA_VISIBLE_DEVICES','')
    devices=[x.strip() for x in visible.split(',') if x.strip()]
    if not environ.get('SLURM_JOB_ID') or len(devices)!=1 or devices[0] in ('-1','NoDevFiles'):
        raise RuntimeError('Real Review08 model loading requires Slurm and exactly one visible CUDA device')
    if torch_module is None:
        import torch as torch_module
    if not torch_module.cuda.is_available() or torch_module.cuda.device_count()!=1:
        raise RuntimeError('Expected exactly one usable CUDA device in this Slurm job')
    return {'slurm_job_id':environ['SLURM_JOB_ID'],'cuda_visible_devices':visible,
            'device_count':1,'device_name':torch_module.cuda.get_device_name(0)}


def verify_release_manifest(path):
    """Validate hash-bound recovered evidence; Slurm dependencies are insufficient.

    Schema ``review08-release-v1``: ``sft633`` and ``mace635`` each contain
    terminal (COMPLETED), recovered (true), result {path,sha256}, verification
    {path,sha256}. Both verification files must say independently_verified=true.
    SFT verification additionally proves finite_gradients, adapter_reload with
    unloaded_without_merge/reloaded_from_disk/state_exact_match all true, and
    adapter_files nonempty [{path,sha256}]. MACE verification requires raw_ef
    >0 and singleton_batch_parity {'4':{passed:true},'16':{passed:true}}.
    File paths are relative to manifest. All listed files are read and hashed.
    """
    path=Path(path); root=path.parent
    manifest=json.loads(path.read_text())
    schema=manifest.get('schema')
    if schema not in ('review08-release-v1','review08-release-v2'): raise ValueError('Unknown release manifest schema')
    receipts={}
    adapter_hashes={}
    def checked(entry):
        target=root/entry['path']
        if sha256(target)!=entry['sha256']: raise ValueError('Release evidence hash mismatch: '+str(target))
        return target
    for name in ('sft633','mace635'):
        item=manifest.get(name,{})
        replacement=(schema=='review08-release-v2' and name=='mace635')
        valid_id=(str(item.get('job_id'))==name[-3:])
        if replacement:
            valid_id=(str(item.get('job_id','')).isdigit() and str(item.get('job_id'))!='635'
                      and item.get('retry_of')=='635' and item.get('authorized_replacement_index')==1)
        if not valid_id or item.get('terminal')!='COMPLETED' or item.get('recovered') is not True:
            raise ValueError('Recovered terminal evidence required for '+name)
        result=json.loads(checked(item['result']).read_text())
        evidence=json.loads(checked(item['verification']).read_text())
        if result.get('status')!='completed' or evidence.get('independently_verified') is not True:
            raise ValueError('Independent successful verification required for '+name)
        if replacement and (str(result.get('runtime',{}).get('job_id'))!=str(item['job_id'])
                            or evidence.get('retry_of')!='635' or evidence.get('authorized_replacement_index')!=1):
            raise ValueError('Authorized635 replacement identity is not independently bound')
        if name=='sft633':
            reload=evidence.get('adapter_reload',{})
            fields=('unloaded_without_merge','reloaded_from_disk','state_exact_match')
            if evidence.get('finite_gradients') is not True or not all(reload.get(k) is True for k in fields):
                raise ValueError('633 reload and finite-gradient evidence required')
            files=evidence.get('adapter_files',[])
            if not files: raise ValueError('633 checkpoint files must be hash-bound')
            for entry in files:
                checked(entry)
                name_in_adapter=Path(entry['path']).name
                if name_in_adapter in adapter_hashes: raise ValueError('Duplicate adapter evidence filename')
                adapter_hashes[name_in_adapter]=entry['sha256']
            if not {'adapter_config.json','adapter_model.safetensors'} <= set(adapter_hashes):
                raise ValueError('Both adapter config and safe tensor evidence are required')
            # Cross-check independent summary against recovered source records.
            if not all(result.get('adapter_reload',{}).get(k) is True for k in fields):
                raise ValueError('633 recovered result does not prove exact reload')
            if result.get('optimizer_steps',result.get('optimization_steps',0))<1:
                raise ValueError('633 recovered result contains no optimizer step')
        else:
            parity=evidence.get('singleton_batch_parity',{})
            if not all(parity.get(k,{}).get('passed') is True for k in ('4','16')):
                raise ValueError('635 singleton/batch4/batch16 parity evidence required')
            raw=evidence.get('raw_ef')
            if type(raw) is not int or raw<1 or raw!=result.get('raw_ef_calls'):
                raise ValueError('635 raw cost evidence does not match recovered result')
            source=result.get('batch_equivalence',{}).get('per_batch_size',{})
            if not all(source.get(k,{}).get('passed') is True for k in ('4','16')):
                raise ValueError('635 recovered result lacks parity proof')
        receipts[name]={'result_sha256':item['result']['sha256'],'verification_sha256':item['verification']['sha256']}
    if schema=='review08-release-v2':
        cpu=json.loads(checked(manifest['grpo_cpu_preflight']).read_text())
        if not all(cpu.get(k) is True for k in ('independently_verified','CPU_only','same_production_code_path')) or cpu.get('batch_sizes')!=[1,4]:
            raise ValueError('Real batch1/4 CPU SP and short relaxation preflight required')
        for filename in ('smoke_grpo08.py','smoke_mace.py'):
            if cpu.get('source_sha256',{}).get(filename)!=sha256(Path(__file__).parent/filename):
                raise ValueError('GRPO CPU preflight source changed: '+filename)
    return {'verified':True,'manifest_sha256':sha256(path),'evidence':receipts,'adapter_hashes':adapter_hashes}


def verify_adapter_binding(adapter_path,release):
    hashes=release.get('adapter_hashes',{})
    if not release.get('verified') or not {'adapter_config.json','adapter_model.safetensors'} <= set(hashes):
        raise ValueError('Release lacks hash-bound reloaded 633 adapter files')
    for filename,digest in hashes.items():
        if Path(filename).name != filename or sha256(Path(adapter_path)/filename)!=digest:
            raise ValueError('Requested reference/policy adapter differs from verified 633: '+filename)


def attach_raw_ef_hook(model,context_provider,log_path,*,allowed_batch_sizes=(1,)):
    """Attach an actual forward hook, independently CPU-testable with Torch."""
    def before_forward(module,args,kwargs):
        active,request=context_provider()
        payload=args[0] if args else kwargs.get('data')
        batch=payload.get('batch') if isinstance(payload,dict) else None
        size=int(batch.max().item())+1 if batch is not None and batch.numel() else 1
        if size not in allowed_batch_sizes: raise RuntimeError('Unexpected physical forward batch size')
        active.before_forward(size)
        _append(log_path,{'request_key':request.key,'raw_ef':active.counters.raw_ef,
            'forward_call':active.counters.forward_calls,'batch_size':size,
            'elapsed_seconds':active.clock()-active.started_at})
    return model.register_forward_pre_hook(before_forward,with_kwargs=True)


def _resources():
    import resource
    row={'host_maxrss_native':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
         'gpu_allocated_bytes':None,'gpu_reserved_bytes':None,
         'gpu_peak_allocated_bytes':None,'gpu_peak_reserved_bytes':None,'device_name':None}
    torch=sys.modules.get('torch')
    if torch is not None and torch.cuda.is_initialized():
        row.update(gpu_allocated_bytes=torch.cuda.memory_allocated(),gpu_reserved_bytes=torch.cuda.memory_reserved(),
                   gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                   gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved(),device_name=torch.cuda.get_device_name(0))
    return row


def runtime_versions():
    versions={}
    for name in ('torch','transformers','peft','mace-torch','ase','numpy','scipy'):
        try: versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: versions[name]=None
    torch=sys.modules.get('torch')
    versions['torch_cuda_build']=None if torch is None else torch.version.cuda
    return versions


@contextmanager
def track_phase(ledger,name,**metadata):
    begin=ledger.clock(); before=asdict(ledger.counters)
    error=None
    try: yield
    except BaseException as exc:
        error=type(exc).__name__+': '+str(exc)
        raise
    finally:
        # Synchronize only an already initialized runtime, never initialize CUDA
        # from the fake CPU path. Include GPU completion in phase wall time.
        torch=sys.modules.get('torch')
        if torch is not None and torch.cuda.is_initialized(): torch.cuda.synchronize()
        row={'phase':name,'begin_elapsed_seconds':begin-ledger.started_at,
             'end_elapsed_seconds':ledger.clock()-ledger.started_at,'seconds':ledger.clock()-begin,
             'counts_before':before,'counts_after':asdict(ledger.counters),'resources':_resources(),**metadata}
        if error is not None: row['error']=error
        ledger.phase_timings.append(row)
        _append(None if ledger.log_path is None else Path(ledger.log_path).parent/'phase_timings.jsonl',row)


def run_smoke(actor,physics,starts,heldout,*,output=None,limits=LIMITS,started_at=None,clock=time.monotonic,runtime_context=None):
    """Execute the fixed 20 gate + 4 starts x G4 x K8 protocol with injected IO.

    ``actor`` implements generate, score_reference, update, finalize, close.
    Generation must charge ledger operations before each actual call. The real
    adapter implements this contract, while tests use tiny deterministic actors.
    Non-default limits are restricted to tighter bounds for CPU cap tests.
    """
    for name,value in asdict(limits).items():
        bound=getattr(LIMITS,name)
        if value is None or (bound is not None and value>bound):
            raise ValueError('Injected limits may only tighten the frozen protocol')
    if limits.max_attempts!=8: raise ValueError('Review08 always uses K8')
    if len(starts)!=4 or len(heldout)!=20: raise ValueError('Exactly four starts and twenty frozen legality states are required')
    output=None if output is None else Path(output)
    if output is not None: output.mkdir(parents=True,exist_ok=True)
    counters=env.Counters()
    ledger=TokenLedger(counters,limits,started_at=started_at,clock=clock,
                       log_path=None if output is None else output/'cost_attempts.jsonl')
    result={'status':'STARTED','seed':SEED,'limits':asdict(limits),'legality':{'denominator':20,'legal':0,'generations':[]},
            'initial_relaxations':0,'physics_receipts':[],'runtime_versions':runtime_versions(),'groups':[],'partial_episodes':[],'optimizer_updates':0,
            'usable_for_workflow':False,'scientific_efficacy_claim':False,
            'count_definition':'full-context generation/reference/current input = prefill+seen; sampled token = generated+seen; raw E/F charged in model prehook'}
    if runtime_context is not None:
        if set(runtime_context)-{'runtime','release'}: raise ValueError('Unknown runtime evidence key')
        result.update(runtime_context)
    result['backstops']={'future_outer_seconds':FUTURE_OUTER_SECONDS,'future_slurm_seconds':FUTURE_SLURM_SECONDS,
                         'OS_IO_hard_preemption':False}
    cache=env.RequestCache(); history=set(); initial_keys=set(); episodes=[]; returns=[]
    physics_phase={'name':'initial'}
    def audited_physics(request,counter):
        counter.phase_ledger=ledger
        index=len(result['physics_receipts'])
        filename='physics/receipt_'+str(index).zfill(4)+'_'+request.key+'.json'
        record={'index':index,'phase':physics_phase['name'],'protocol':env.PROTOCOL,
                'request_key':request.key,'relaxation_spec':asdict(request.spec),
                'requested_occupancy':list(request.occupancy),'raw_ef_begin':counter.counters.raw_ef,
                'relax_actual_at_entry':counter.counters.relax_actual,'status':'started',
                'begin_elapsed_seconds':clock()-ledger.started_at}
        _append(None if output is None else output/'relax_attempts.jsonl',record)
        try:
            with track_phase(ledger,'physical_relaxation',request_key=request.key,kind=physics_phase['name']):
                physical_result=physics(request,counter)
                terminal=env.verify_result(physical_result,request.spec)
                if counter.counters.raw_ef<=record['raw_ef_begin']:
                    raise RuntimeError('Physical callback returned without any charged model forward')
                record['status']='verified_converged'
                record['terminal']={'energy_eV':terminal.result.energy,
                    'forces_eV_A':terminal.result.forces.tolist(),'converged':True,
                    'fmax_eV_A':float(np.linalg.norm(terminal.result.forces,axis=1).max()),
                    'mapped_occupancy':list(terminal.mapping.occupancy),
                    'canonical_key':terminal.mapping.canonical_key,'geometry_hash':terminal.mapping.geometry_hash,
                    'assigned_site_ids':list(terminal.mapping.assigned_site_ids),
                    'assignment_distances_A':list(terminal.mapping.assignment_distances),
                    'shell_counts':terminal.mapping.shell_counts,'positions_A':terminal.result.atoms.positions.tolist(),
                    'numbers':terminal.result.atoms.numbers.tolist(),'cell_A':terminal.result.atoms.cell.array.tolist(),
                    'pbc':terminal.result.atoms.pbc.tolist()}
                return physical_result
        except BaseException as error:
            record.update(status='failed',error=type(error).__name__+': '+str(error),traceback=traceback.format_exc())
            raise
        finally:
            record.update(raw_ef_end=counter.counters.raw_ef,forward_calls_end=counter.counters.forward_calls,
                          end_elapsed_seconds=clock()-ledger.started_at,resources=_resources())
            result['physics_receipts'].append({'index':index,'phase':record['phase'],'status':record['status'],
                'request_key':request.key,'raw_ef_begin':record['raw_ef_begin'],'raw_ef_end':record['raw_ef_end'],
                'receipt_file':filename if output is not None else None})
            if output is not None: save_json(output/filename,record)
    try:
        # No physics callback, reference scoring or gradient code occurs here.
        for index,row in enumerate(heldout):
            ledger.check()
            occupancy=env.validate_occupancy(row['occupancy'])
            messages=row.get('messages')
            if messages is None: messages=env.encode_prompt_v2(unscored_state(occupancy,limits=limits,counters=counters))
            with track_phase(ledger,'generation_legality',index=index):
                action=actor.generate(messages,ledger,phase='legality',occupancy=occupancy)
            try:
                env.apply_action(occupancy,env.parse_action(action.text)); legal=True
            except env.ActionError: legal=False
            result['legality']['legal']+=int(legal)
            record={'index':index,'text':action.text,'legal':legal,'new_tokens':sum(action.action_mask)}
            result['legality']['generations'].append(record)
            _append(None if output is None else output/'generations.jsonl',{'phase':'legality',**record})
        if result['legality']['legal']<19:
            result['status']='NOT_RELEASED_LEGALITY_GATE'
            return result
        initial=[]
        # Four charged verified initial relaxations, each reused by its G4 group.
        for occupancy in starts:
            ledger.check()
            request=env.build_request(occupancy)
            if counters.relax_actual>=limits.max_relax_actual: raise env.BudgetExceeded('Initial relaxation cap')
            counters.relax_actual+=1; result['initial_relaxations']+=1
            terminal=cache.record(('review08',SEED),request,audited_physics(request,ledger.raw))
            initial.append(terminal.result)
            initial_keys.add(terminal.mapping.canonical_key)
        history.update(initial_keys)
        physics_phase['name']='rollout'
        for group_id,start in enumerate(initial):
            group=env.NoveltyGroup(history)
            group_episodes=[]; group_returns=[]; details=[]
            for rollout in range(4):
                environment=env.Environment(start,audited_physics,arm='review08',seed=SEED,limits=limits,counters=counters,
                        cache=cache,frozen_history=group.frozen_history,started_at=ledger.started_at,clock=clock)
                samples=[]; steps=[]; energy=novelty=invalid=illegal=physical_failure=0.
                for step in range(8):
                    ledger.check()
                    messages=env.encode_prompt_v2(environment.state)
                    with track_phase(ledger,'generation_rollout',group=group_id,rollout=rollout,step=step):
                        action=actor.generate(messages,ledger,phase='rollout',occupancy=environment.state.occupancy)
                    with track_phase(ledger,'reference_scoring',group=group_id,rollout=rollout,step=step):
                        action=actor.score_reference(action,ledger)
                    sample=action.sample(); samples.append(sample)
                    result['partial_episodes']=[{'group':group_id,'rollout':rollout,'steps':steps,
                                                 'samples':[asdict(s)|{'sha256':s.sha256} for s in samples]}]
                    outcome=environment.step(action.text)  # ledger already charged token work
                    energy+=outcome.energy_reward; novelty+=outcome.novelty_reward
                    if not outcome.success:
                        invalid+=outcome.reward
                        if outcome.status=='illegal': illegal+=outcome.reward
                        elif outcome.status=='physical_failure': physical_failure+=outcome.reward
                    record={'group':group_id,'rollout':rollout,'step':step,'text':action.text,
                            'outcome':asdict(outcome),'sample':asdict(sample),'record_sha256':sample.sha256,
                            'terminal_key_after':environment.state.terminal_key,'counts':asdict(counters)}
                    steps.append(record)
                    _append(None if output is None else output/'rollouts.jsonl',record)
                total=energy+novelty+invalid
                group_returns.append(total); group_episodes.append(samples)
                details.append({'return':total,'energy_return':energy,'novelty_return':novelty,
                                'invalid_return':invalid,'illegal_return':illegal,
                                'physical_failure_return':physical_failure,'steps':steps,'novelty_events':sum(s['outcome']['novelty_reward']>0 for s in steps)})
                group.register(environment.state)
            group.merge()
            episodes.extend(group_episodes); returns.append(group_returns)
            result['groups'].append({'group':group_id,'returns':group_returns,'episodes':details,
                                    'advantages':group_advantages([group_returns])[0].tolist()})
            result['partial_episodes']=[]
        advantages=group_advantages(returns)
        result['N_unique']=len(history-initial_keys)
        result['initial_terminal_keys']=sorted(initial_keys)
        result['merged_terminal_keys']=sorted(history)
        if not np.count_nonzero(advantages.numpy()):
            result['status']='NO_UPDATE_ZERO_GROUP_VARIANCE'
            result['update']={'optimizer_updates':0,'advantages':advantages.tolist(),
                              'gradient_status':'not_measured_no_update'}
        else:
            ledger.check()
            with track_phase(ledger,'policy_update'):
                update=actor.update(episodes,returns,ledger)
            if update.get('optimizer_updates')!=1 or update.get('finite_gradients') is not True or update.get('parameter_delta_l2',0)<=0:
                raise RuntimeError('One-update finite-gradient/actual-parameter-change check failed')
            update['gradient_status']='measured_finite'
            result['update']=update; result['optimizer_updates']=1; result['status']='COMPLETED_ONE_UPDATE'
        ledger.check()
        with track_phase(ledger,'adapter_save_unload_reload'):
            result['adapter_check']=actor.finalize(ledger,output)
            if result['adapter_check'].get('save_unload_reload') is not True:
                raise RuntimeError('Policy adapter save/unload/disk-reload evidence failed')
        ledger.check()
        result['usable_for_workflow']=True
    except env.BudgetExceeded as error:
        result['status']='BUDGET_EXHAUSTED'; result['error']=str(error)
    except Exception as error:
        result['status']='FAILED'; result['error']=str(error); result['traceback']=traceback.format_exc()
    finally:
        try:
            with track_phase(ledger,'actor_close'): actor.close()
        except Exception as error:
            result['close_error']=str(error); result['usable_for_workflow']=False; result['status']='FAILED'
        try:
            if hasattr(physics,'close'):
                with track_phase(ledger,'physics_close'): physics.close()
        except Exception as error:
            result['physics_close_error']=str(error); result['usable_for_workflow']=False; result['status']='FAILED'
        if limits.wall_seconds is not None and clock()-ledger.started_at >= limits.wall_seconds:
            result['status']='BUDGET_EXHAUSTED'; result['usable_for_workflow']=False
        result.update(counters=asdict(counters),wall_seconds=clock()-ledger.started_at,
                      resources=_resources(),cache_diagnostics=cache.diagnostics,
                      N_unique=len(history-initial_keys),cost_events=ledger.events,phase_timings=ledger.phase_timings)
        if output is not None:
            # Measure the actual owned result persistence after it completes.
            # OS IO is cooperative, not preemptible: a late overrun is corrected
            # on disk and is never returned usable/completed. The last compact
            # metadata write cannot contain its own exact finish timestamp; the
            # returned/printed boundary and Slurm parent cover that final IO.
            result['result_persistence']={'started_elapsed_seconds':clock()-ledger.started_at,
                'timing_boundary':'post-write measured in returned/printed summary; last metadata write is cooperative IO'}
            save_json(output/'result.json',result)
            result['wall_seconds']=clock()-ledger.started_at
            result['result_persistence']['first_write_finished_elapsed_seconds']=result['wall_seconds']
            if limits.wall_seconds is not None and result['wall_seconds']>=limits.wall_seconds:
                result['status']='BUDGET_EXHAUSTED'; result['usable_for_workflow']=False
                result['error']='Wall-time cap reached during final result persistence'
                save_json(output/'result.json',result)
                result['wall_seconds']=clock()-ledger.started_at
            result['result_persistence']['last_write_finished_elapsed_seconds']=result['wall_seconds']
    return result


class HFActor:
    """Qwen3-0.6B with two copies of the *reloaded 633* adapter on one base.

    Reference adapter stays fixed. set_adapter can alter requires_grad, therefore
    every switch explicitly freezes base/reference and enables only policy LoRA.
    All generation/scoring is eval mode (dropout off), including gradient scoring.
    """
    def __init__(self,model_path,adapter_path,qwen_manifest,*,release,output):
        if not release.get('verified'): raise ValueError('Verified prerequisites required')
        verify_adapter_binding(adapter_path,release)
        self.model_path=Path(model_path); self.adapter_path=Path(adapter_path)
        self.qwen_manifest=Path(qwen_manifest); self.output=Path(output)
        self.model=None; self.tokenizer=None; self.device='cpu'; self.generator=None
        self.initial_reference_hash=None; self.initial_policy_hash=None; self.updated=False

    def _ensure(self,ledger):
        if self.model is None:
            with track_phase(ledger,'HF_load'): self._load(ledger)
        else: self.to_device('cuda',ledger)

    def _load(self,ledger):
        if self.model is not None:
            self.to_device('cuda',ledger); return
        ledger.check(); require_single_gpu_job()
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM
            from peft import PeftModel
        except ImportError as error:
            raise RuntimeError('Review08 requires one future isolated environment containing torch, transformers, peft, MACE, ASE and SciPy; existing 633/635 environments are separate') from error
        provenance=json.loads(self.qwen_manifest.read_text())
        if provenance.get('repository')!='Qwen/Qwen3-0.6B' or provenance.get('status')!='verified_complete':
            raise ValueError('Only verified official Qwen3-0.6B is approved')
        if not any(f['path']=='model.safetensors' and f['sha256']==QWEN_WEIGHT_SHA for f in provenance['files']):
            raise ValueError('Qwen base weight SHA differs from approved manifest')
        for file in provenance['files']:
            ledger.check()
            if sha256(self.model_path/file['path'])!=file['sha256']: raise ValueError('Qwen file hash changed: '+file['path'])
        configuration=json.loads((self.adapter_path/'adapter_config.json').read_text())
        if configuration.get('r')!=16 or configuration.get('lora_alpha')!=32 or set(configuration.get('target_modules',[]))!={'q_proj','v_proj'}:
            raise ValueError('633 adapter is not the fixed r16 alpha32 q/v LoRA')
        self.torch=torch
        self.tokenizer=AutoTokenizer.from_pretrained(self.model_path,local_files_only=True,trust_remote_code=False)
        base=AutoModelForCausalLM.from_pretrained(self.model_path,local_files_only=True,trust_remote_code=False,
                torch_dtype=torch.bfloat16,attn_implementation='sdpa')
        ledger.check()
        self.model=PeftModel.from_pretrained(base,self.adapter_path,adapter_name='reference',is_trainable=False,local_files_only=True)
        self.model.load_adapter(self.adapter_path,adapter_name='policy',is_trainable=True,local_files_only=True)
        self._switch('policy')
        self.initial_reference_hash=self._reference_hash()
        self.initial_policy_hash=self._policy_hash()
        # Hash fixed base and reference as well as immutable action records.
        self.fixed_hash=state_fingerprint({n:p for n,p in self.model.named_parameters() if '.policy.' not in n})
        self.to_device('cuda',ledger)
        self.generator=torch.Generator(device='cuda').manual_seed(SEED)

    def _switch(self,name):
        self.model.set_adapter(name)
        for n,p in self.model.named_parameters(): p.requires_grad_('.policy.' in n and 'lora_' in n and name=='policy')
        self.model.eval()

    def _reference_hash(self):
        return state_fingerprint({n:p for n,p in self.model.named_parameters() if '.reference.' in n})
    def _policy_hash(self):
        return state_fingerprint({n:p for n,p in self.model.named_parameters() if '.policy.' in n})

    def to_device(self,device,ledger):
        ledger.check()
        if self.model is not None and self.device!=device:
            with track_phase(ledger,'HF_swap_'+device):
                self.model.to(device); self.device=device
                if device=='cpu': self.torch.cuda.empty_cache()
        ledger.check()

    def generate(self,messages,ledger,*,phase,occupancy):
        self._ensure(ledger); self._switch('policy')
        if self._policy_hash()!=self.initial_policy_hash and not self.updated:
            raise RuntimeError('Old generation policy changed before the single update')
        ids=self.tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,enable_thinking=False)
        if hasattr(ids,'tolist'): ids=ids.tolist()
        if not ids or len(ids)>MAX_PROMPT_TOKENS: raise ValueError('Prompt exceeds 1984; no silent truncation')
        torch=self.torch
        inputs=torch.tensor([ids],device='cuda')
        ledger.before_prefill(len(ids),phase=phase+'_generation_prefill')
        generated=[]; generated_lp=[]
        with torch.no_grad():
            output=self.model(input_ids=inputs,attention_mask=torch.ones_like(inputs),use_cache=True)
            prompt_lp=causal_logprobs(output.logits,inputs)[0].cpu().tolist() if len(ids)>1 else []
            cache=output.past_key_values
            for index in range(MAX_NEW_TOKENS):
                # No generate(), constrained decoder, processors, top-k/p or renormalization.
                scores=output.logits[0,-1].float()
                if not torch.isfinite(scores).all(): raise FloatingPointError('Nonfinite generation logits')
                logprobs=scores.log_softmax(-1)
                ledger.before_generated_token(phase=phase+'_sample')
                token=int(torch.multinomial(logprobs.exp(),1,generator=self.generator).item())
                generated.append(token); generated_lp.append(float(logprobs[token]))
                if token==self.tokenizer.eos_token_id: break
                if index+1<MAX_NEW_TOKENS:
                    ledger.check()
                    attention=torch.ones((1,len(ids)+len(generated)),device='cuda',dtype=torch.long)
                    output=self.model(input_ids=torch.tensor([[token]],device='cuda'),attention_mask=attention,
                                      past_key_values=cache,use_cache=True)
                    cache=output.past_key_values
        full=tuple(ids+generated)
        if len(full)>MAX_TOTAL_TOKENS: raise ValueError('Total sequence exceeds 2048')
        lp=tuple(prompt_lp+generated_lp)
        text=decode_action_tokens(self.tokenizer,generated)
        return GeneratedAction(text,full,tuple([False]*len(ids)+[True]*len(generated)),lp,lp)

    def score_reference(self,action,ledger):
        self._ensure(ledger); self._switch('reference')
        try:
            ids=self.torch.tensor([action.input_ids],device='cuda')
            ledger.before_prefill(len(action.input_ids),phase='reference_full_context')
            with self.torch.no_grad():
                output=self.model(input_ids=ids,attention_mask=self.torch.ones_like(ids),use_cache=False)
                lp=causal_logprobs(output.logits,ids)[0].cpu().tolist()
            if self._reference_hash()!=self.initial_reference_hash: raise RuntimeError('Reference adapter changed')
            return replace(action,reference_logprobs=tuple(lp))
        finally: self._switch('policy')

    def update(self,episodes,returns,ledger):
        self._ensure(ledger); self._switch('policy')
        old_ref=self._reference_hash(); old_policy=self._policy_hash()
        if old_policy!=self.initial_policy_hash: raise RuntimeError('Old policy changed before update')
        def score(sample):
            ledger.before_prefill(len(sample.input_ids),phase='current_training_full_context')
            ids=self.torch.tensor([sample.input_ids],device='cuda')
            output=self.model(input_ids=ids,attention_mask=self.torch.ones_like(ids),use_cache=False)
            return causal_logprobs(output.logits,ids)[0]
        result=one_update(self.model.parameters(),episodes,returns,score,before_step=ledger.check)
        self.updated=result['optimizer_updates']==1
        if self._reference_hash()!=old_ref: raise RuntimeError('Reference adapter changed during update')
        if state_fingerprint({n:p for n,p in self.model.named_parameters() if '.policy.' not in n})!=self.fixed_hash:
            raise RuntimeError('Frozen base/reference parameters changed')
        result.update(reference_hash_before=old_ref,reference_hash_after=self._reference_hash(),
                      old_policy_hash=old_policy,current_policy_hash=self._policy_hash())
        return result

    def finalize(self,ledger,output):
        self._ensure(ledger); self._switch('policy')
        from peft import PeftModel, get_peft_model_state_dict
        reference_final=self._reference_hash()
        policy_final=self._policy_hash()
        if reference_final!=self.initial_reference_hash: raise RuntimeError('Reference changed before save')
        if not self.updated and policy_final!=self.initial_policy_hash: raise RuntimeError('Old policy changed without update')
        if state_fingerprint({n:p for n,p in self.model.named_parameters() if '.policy.' not in n})!=self.fixed_hash:
            raise RuntimeError('Frozen base/reference changed before save')
        path=self.output/'policy_adapter'
        self.model.save_pretrained(path,safe_serialization=True,selected_adapters=['policy'])
        # Non-default PEFT adapters are saved in a named subdirectory.
        saved=path/'policy'
        if not (saved/'adapter_config.json').exists(): saved=path
        expected={k:v.detach().cpu().clone() for k,v in get_peft_model_state_dict(self.model,adapter_name='policy').items()}
        ledger.check()
        base=self.model.unload()  # explicitly no merge
        self.model=None
        if any('lora_' in n for n,_ in base.named_parameters()): raise RuntimeError('Unloaded base retains LoRA')
        reloaded=PeftModel.from_pretrained(base,saved,is_trainable=False,local_files_only=True)
        actual=get_peft_model_state_dict(reloaded)
        exact=set(actual)==set(expected) and all(self.torch.equal(actual[k].detach().cpu(),expected[k]) for k in expected)
        self.model=reloaded
        if not exact: raise RuntimeError('Saved policy adapter failed exact tensor disk reload')
        ledger.check()
        return {'save_unload_reload':True,'unloaded_without_merge':True,'state_exact_match':True,
                'reloaded_from_disk':True,'tensor_count':len(actual),'sha256':sha256(saved/'adapter_model.safetensors'),
                'old_policy_hash_initial':self.initial_policy_hash,'policy_hash_before_save':policy_final,
                'reference_hash_initial':self.initial_reference_hash,'reference_hash_final':reference_final,
                'reference_unchanged':True,'old_policy_unchanged_before_update':True,
                'gradient_status':'measured_finite' if self.updated else 'not_measured_no_update'}

    def close(self):
        self.model=None; self.tokenizer=None
        gc.collect()
        if hasattr(self,'torch'): self.torch.cuda.empty_cache()


class MACEPhysics:
    """Lazy fixed-cell BFGS callback; raw charges at actual model pre-forward.

    ASE's energy/force cache may avoid a second invocation. Counting calculate()
    would wrongly count graph preparation; the model prehook counts forwards,
    including failures. No final-frame energy/forces are fabricated.
    """
    def __init__(self,checkpoint,actor,ledger_output,*,cpu_preflight=False):
        self.checkpoint=Path(checkpoint); self.actor=actor; self.output=Path(ledger_output)
        self.calc=None; self.context=None; self.hook=None
        self.cpu_preflight=cpu_preflight
        self.device='cpu' if cpu_preflight else 'cuda'
        self.max_steps=1 if cpu_preflight else 300

    def _ensure(self,counter):
        counter.check_time()
        if not self.cpu_preflight: require_single_gpu_job()
        if sha256(self.checkpoint)!=env.MODEL_SHA: raise ValueError('Frozen MACE-MP0b2 small SHA mismatch')
        if self.calc is None:
            try:
                from mace.calculators import MACECalculator
                import torch
            except ImportError as error:
                raise RuntimeError('Combined Review08 environment needs mace-torch, ASE and SciPy alongside transformers/peft; do not reuse or modify the running environments') from error
            self.torch=torch
            self.calc=MACECalculator(model_paths=str(self.checkpoint),device=self.device,default_dtype='float64',enable_cueq=False,enable_oeq=False)
            if len(self.calc.models)!=1 or next(self.calc.models[0].parameters()).dtype!=torch.float64:
                raise RuntimeError('Exactly one float64 physical model required')
            self.hook=attach_raw_ef_hook(self.calc.models[0],lambda:self.context,self.output/'raw_ef_attempts.jsonl',
                                       allowed_batch_sizes=(1,4) if self.cpu_preflight else (1,))
        else:
            self.calc.models[0].to(self.device)
        counter.check_time()

    def __call__(self,request,counter):
        # Offloading HF occurs before MACE allocation; both share the one GPU.
        ledger=getattr(counter,'phase_ledger',None)
        if ledger is None:
            ledger=TokenLedger(counter.counters,counter.limits,started_at=counter.started_at,clock=counter.clock,
                               log_path=self.output/'cost_attempts.jsonl')
        self.actor.to_device('cpu',ledger)
        self.context=(counter,request)
        try:
            with track_phase(ledger,'MACE_load_or_swap_'+self.device):
                self._ensure(counter)
            from ase.optimize import BFGS
            atoms=request.atoms.copy(); atoms.calc=self.calc
            optimizer=BFGS(atoms,logfile=None,maxstep=.1)
            converged=bool(optimizer.run(fmax=.03,steps=self.max_steps))
            self.last_steps=optimizer.nsteps
            counter.check_time()
            forces=np.asarray(atoms.get_forces(),dtype=float)
            energy=float(atoms.get_potential_energy())
            result=env.PhysicsResult(atoms.copy(),energy,forces,converged)
            if self.cpu_preflight:
                if not np.isfinite(forces).all() or not np.isfinite(energy):
                    raise RuntimeError('Nonfinite CPU preflight physics')
            else:
                env.verify_result(result,request.spec)
            return result
        finally:
            self.context=None
            if self.calc is not None:
                with track_phase(ledger,'MACE_swap_cpu'):
                    self.calc.models[0].to('cpu'); self.torch.cuda.empty_cache()
            counter.check_time()

    def close(self):
        if self.hook is not None:
            self.hook.remove(); self.hook=None
        self.calc=None
        gc.collect()


def real_cpu_preflight(checkpoint,output):
    """Actual batch1/4 graph forwards plus short production BFGS callbacks.

    GRPO relaxes one request per callback. A four-request preflight exercises
    four such callbacks, while also testing an actual four-graph forward.
    Short, unconverged results are diagnostics only, never training receipts.
    """
    import torch
    from calibration import make_state
    from smoke_mace import build_batch
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);torch.set_default_dtype(torch.float64)
    class NoLLM:
        def to_device(self,device,ledger):
            if device!='cpu': raise RuntimeError('CPU preflight must not use a GPU')
            ledger.check()
    counters=env.Counters();counter=env.RawEFCounter(counters,env.BudgetLimits(max_raw_ef=64,wall_seconds=1800))
    physics=MACEPhysics(checkpoint,NoLLM(),output,cpu_preflight=True)
    groups=[]
    try:
        physics._ensure(counter)
        for size in (1,4):
            requests=[env.build_request(tuple(map(int,make_state(SEED+i,3,16).numbers))) for i in range(size)]
            physics.context=(counter,requests[0])
            batch=build_batch(physics.calc,[r.atoms for r in requests],'cpu')
            values=physics.calc.models[0](batch.to_dict(),compute_stress=False,training=False,
                    compute_edge_forces=False,compute_atomic_stresses=False)
            if not all(torch.isfinite(values[k]).all() for k in ('energy','forces')):
                raise RuntimeError('CPU batch E/F is nonfinite')
            steps=[]
            for request in requests:
                result=physics(request,counter);steps.append(physics.last_steps)
            groups.append({'batch_size':size,'real_SP_graphs':size,'BFGS_callbacks':size,'actual_steps':steps})
        result={'status':'CPU_REAL_PATH_PREFLIGHT_PASSED','independently_verified':True,'CPU_only':True,
                'same_production_code_path':True,'batch_sizes':[1,4],'groups':groups,'raw_ef':counters.raw_ef,
                'short_relaxation_convergence_not_required':True,'formal_relaxation_unchanged':'BFGS/.03/300/maxstep.1',
                'source_sha256':{name:sha256(Path(__file__).parent/name) for name in ('smoke_grpo08.py','smoke_mace.py')}}
        save_json(output/'result.json',result)
        print(json.dumps(result))
    finally:
        physics.close()


def main():
    started=time.monotonic()
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cpu-execution-preflight',action='store_true')
    for name in ('model','adapter','qwen-manifest','checkpoint','release-manifest','data','output'):
        parser.add_argument('--'+name,type=Path,required=name in ('checkpoint','output'))
    args=parser.parse_args()
    if args.cpu_execution_preflight:
        real_cpu_preflight(args.checkpoint,args.output)
        return 0
    if any(getattr(args,name) is None for name in ('model','adapter','qwen_manifest','release_manifest','data')):
        parser.error('All frozen production paths are required')
    # No CLI hyperparameter overrides; fail before any model loading.
    release=verify_release_manifest(args.release_manifest)
    runtime=require_single_gpu_job()
    manifest=json.loads((args.data/'manifest.json').read_text())
    if manifest.get('seed')!=SEED or manifest.get('starts')!=4 or manifest.get('legality_states')!=20:
        raise ValueError('Review08 frozen data counts/seed differ')
    for filename,digest in manifest['files'].items():
        if sha256(args.data/filename)!=digest: raise ValueError('Frozen data file hash differs')
    starts=[json.loads(line)['occupancy'] for line in (args.data/'starts.jsonl').read_text().splitlines()]
    heldout=[json.loads(line) for line in (args.data/'legality_v2.jsonl').read_text().splitlines()]
    actor=HFActor(args.model,args.adapter,args.qwen_manifest,release=release,output=args.output)
    physics=MACEPhysics(args.checkpoint,actor,args.output)
    result=run_smoke(actor,physics,starts,heldout,output=args.output,started_at=started,
                     runtime_context={'runtime':runtime,'release':release})
    print(json.dumps({'status':result['status'],'optimizer_updates':result['optimizer_updates'],
                      'counters':result['counters'],'wall_seconds':result['wall_seconds']}),flush=True)
    return 0 if result['status'] in ('COMPLETED_ONE_UPDATE','NO_UPDATE_ZERO_GROUP_VARIANCE','NOT_RELEASED_LEGALITY_GATE') else 1


if __name__=='__main__': raise SystemExit(main())
