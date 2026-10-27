"""CPU-only D0 contracts and later callable data freeze; no output on import.

The existing 64 unscored synthetic rows are format examples, never D0 physics
labels. ``PhysicalReceipt.verify`` only accepts an actual converged result and
explicit nonzero charged costs. The bounded collector executes only callbacks
provided by a later authorized caller; it supplies no SA, GA or CE claims.
"""
from __future__ import annotations
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
import numpy as np
import claude08_env as env
from calibration import make_state

SEED = 2026102301


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(1024*1024), b''): digest.update(block)
    return digest.hexdigest()


def seed_streams(seed):
    """Separate reproducible streams for teacher proposals, splits, and rollout."""
    return tuple(int(s.generate_state(1, dtype=np.uint64)[0]) for s in np.random.SeedSequence(seed).spawn(3))


def action_sft_row(tokenizer, row, *, max_tokens=2048):
    """Encode prompt + exact JSON action + EOS, without silently truncating."""
    action = env.parse_action(json.dumps(row['action'], separators=(',', ':'))).to_json()
    prefix = tokenizer.apply_chat_template(row['messages'], tokenize=False,
                    add_generation_prompt=True, enable_thinking=False)
    prompt = list(tokenizer(prefix, add_special_tokens=False)['input_ids'])
    full = list(tokenizer(prefix+action, add_special_tokens=False)['input_ids'])
    if not prompt or full[:len(prompt)] != prompt:
        raise ValueError('Unstable prompt/action token boundary')
    if type(tokenizer.eos_token_id) is not int:
        raise ValueError('A single explicit EOS token is required')
    full.append(tokenizer.eos_token_id)
    if len(full) > max_tokens:
        raise ValueError('SFT context exceeds fixed cap; truncation is forbidden')
    mask = [False]*len(prompt) + [True]*(len(full)-len(prompt))
    return {'input_ids':full, 'labels':[-100]*len(prompt)+full[len(prompt):],
            'action_mask':mask,'prompt_tokens':len(prompt),'action_tokens':sum(mask)}


@dataclass(frozen=True)
class PhysicalReceipt:
    request_key: str
    terminal_key: str
    energy: float
    geometry_hash: str
    costs: dict
    model_sha: str
    converged: bool
    _request: object
    _result: object

    def __post_init__(self):
        object.__setattr__(self, 'costs', MappingProxyType(dict(self.costs)))

    @classmethod
    def verify(cls, request, result, *, relax_actual, raw_ef):
        if (type(relax_actual) is not int or relax_actual < 1 or type(raw_ef) is not int or raw_ef < 1):
            raise ValueError('Verified physical receipts require recorded positive actual costs')
        if request.spec.model_sha != env.MODEL_SHA:
            raise ValueError('Teacher receipt uses an unfrozen physical model')
        terminal = env.verify_result(result, request.spec)
        return cls(request.key, terminal.mapping.canonical_key, terminal.result.energy,
                   terminal.mapping.geometry_hash, {'relax_actual':relax_actual,'raw_ef':raw_ef},
                   request.spec.model_sha, True, request, terminal.result)

    def reverify(self):
        fresh = self.verify(self._request, self._result, **self.costs)
        if (fresh.request_key,fresh.terminal_key,fresh.energy,fresh.geometry_hash,fresh.model_sha,fresh.converged,dict(fresh.costs)) != (self.request_key,self.terminal_key,self.energy,self.geometry_hash,self.model_sha,self.converged,dict(self.costs)):
            raise ValueError('Physical receipt has changed since verification')
        return fresh

    def summary(self):
        self.reverify()
        return {'request_key':self.request_key,'terminal_key':self.terminal_key,'energy_eV':self.energy,
                'geometry_hash':self.geometry_hash,'costs':dict(self.costs),'model_sha':self.model_sha,
                'converged':True,'DFT_labels_used':False}


def validate_teacher_trajectory(trajectory):
    """Require complete connected transitions carrying independently verifiable receipts."""
    if not isinstance(trajectory, dict) or trajectory.get('complete') is not True:
        raise ValueError('A complete physically verified teacher trajectory is required')
    initial = trajectory.get('initial_receipt')
    transitions = trajectory.get('transitions')
    if not isinstance(initial, PhysicalReceipt) or not isinstance(transitions, list) or not transitions:
        raise ValueError('Initial and per-transition physical receipts are required')
    previous = initial.reverify()
    limits = trajectory.get('limits', env.BudgetLimits())
    if not isinstance(limits, env.BudgetLimits) or len(transitions)>limits.max_attempts:
        raise ValueError('Teacher trajectory exceeds its bounded attempt budget')
    # Receipts do not record wall-clock time. A frozen teacher prompt therefore
    # carries no invented remaining seconds; recorded physical costs and actual
    # action history are reconstructed exactly from verified preceding receipts.
    limits = replace(limits, wall_seconds=None)
    counters = env.Counters(relax_actual=previous.costs['relax_actual'],raw_ef=previous.costs['raw_ef'],
                            forward_calls=previous.costs['raw_ef'])
    families = {previous.terminal_key}
    rows = []
    history = []
    best_energy = previous.energy
    for index, transition in enumerate(transitions):
        receipt = transition.get('receipt')
        if not isinstance(receipt, PhysicalReceipt):
            raise ValueError('Synthetic or merely asserted physical receipts are forbidden')
        receipt.reverify()
        mapping = env.map_terminal(previous._result.atoms)
        before = mapping.occupancy
        state = env.EpisodeState(before,previous._result.atoms.copy(),previous.energy,best_energy,
            mapping.canonical_key,mapping.geometry_hash,frozenset(),set(families),limits,counters,
            ('teacher',int(trajectory.get('seed',0))),0.,lambda:0.,steps_taken=index,history=list(history))
        messages = env.encode_prompt_v2(state)
        action = env.parse_action(json.dumps(transition['action'], separators=(',', ':')))
        expected = env.build_request(env.apply_action(before, action), previous._request.spec)
        if receipt.request_key != expected.key:
            raise ValueError('Teacher action does not connect the previous relaxed terminal to this request')
        families.add(receipt.terminal_key)
        rows.append({'messages':messages,'action':json.loads(action.to_json()),
                     'physical_receipt':receipt.summary()})
        history.append(env.HistoryEntry(action.to_json(),'success',receipt.energy-previous.energy,
                                         receipt.request_key,receipt.terminal_key,0.))
        counters.attempts += 1
        counters.relax_actual += receipt.costs['relax_actual']
        counters.raw_ef += receipt.costs['raw_ef']
        counters.forward_calls += receipt.costs['raw_ef']
        best_energy = min(best_energy,receipt.energy)
        previous = receipt
    return {'rows':rows,'terminal_families':sorted(families),'complete':True,'DFT_labels_used':False}


def split_terminal_families(trajectories, *, seed, validation_fraction=.2):
    """Keep connected terminal families together, including shared intermediates."""
    if not 0 < validation_fraction < 1: raise ValueError('Validation fraction must be between zero and one')
    components = []
    for trajectory in trajectories:
        families = set(trajectory.get('terminal_families', []))
        if not families: raise ValueError('Complete terminal family keys are required')
        touching = [i for i,(keys,_) in enumerate(components) if keys & families]
        rows = [trajectory]
        for i in reversed(touching):
            keys, group = components.pop(i); families.update(keys); rows.extend(group)
        components.append((families,rows))
    if len(components) < 2: raise ValueError('At least two disjoint terminal families are required for a split')
    order = np.random.default_rng(seed).permutation(len(components))
    valid_components = set(map(int,order[:min(len(components)-1,max(1,round(len(components)*validation_fraction)))]))
    train, valid = [], []
    for i,(_, rows) in enumerate(components): (valid if i in valid_components else train).extend(rows)
    return train, valid


def collect_teacher(initial_receipt, propose, callback, *, max_steps, counters, limits, seed):
    """Bounded injectable teacher collector, never called by this module itself.

    A proposal returns (messages, JSON action). Costs are charged before callback;
    result receipts are accepted only after exact environment verification.
    """
    if type(max_steps) is not int or not 1 <= max_steps <= 8:
        raise ValueError('Collector requires one to eight bounded steps')
    initial_receipt.reverify()
    current = initial_receipt
    raw = env.RawEFCounter(counters, limits)
    transitions = []
    for step in range(max_steps):
        raw.check_time()
        occupancy = env.map_terminal(current._result.atoms).occupancy
        messages, text = propose(occupancy, step, seed)
        request = env.build_request(env.apply_action(occupancy,env.parse_action(text)))
        if limits.max_relax_actual is not None and counters.relax_actual >= limits.max_relax_actual:
            raise env.BudgetExceeded('Teacher relaxation cap')
        before = counters.raw_ef
        counters.relax_actual += 1
        result = callback(request,raw)
        current = PhysicalReceipt.verify(request,result,relax_actual=1,raw_ef=counters.raw_ef-before)
        transitions.append({'messages':messages,'action':json.loads(text),'receipt':current})
    trajectory = {'initial_receipt':initial_receipt,'transitions':transitions,'complete':True,'seed':seed,
                  'limits':replace(limits,wall_seconds=None)}
    validated = validate_teacher_trajectory(trajectory)
    for transition,row in zip(transitions,validated['rows']):
        transition['messages'] = row['messages']
    return trajectory


def unscored_state(occupancy, *, limits=None, counters=None):
    """A legality observation explicitly carries unknown energy, never a fake score."""
    occupancy = env.validate_occupancy(occupancy)
    limits = env.BudgetLimits(max_attempts=8) if limits is None else limits
    counters = env.Counters() if counters is None else counters
    return env.EpisodeState(occupancy,env.ideal_atoms(occupancy),None,None,env.occupancy_key(occupancy),'unscored',
            frozenset(),set(),limits,counters,('review08',SEED),0.,lambda:0.)


def freeze_smoke_data(output, ideal_states, excluded_families, *, prerequisites, seed=SEED):
    """Later CPU preparation only: freeze starts/heldout with file hashes.

    ``prerequisites`` must be the verified result returned by the runner's release
    manifest checker. This callable deliberately creates nothing until invoked.
    It does not authorize or submit a GPU job and performs no teacher physics.
    """
    if not isinstance(prerequisites,dict) or prerequisites.get('verified') is not True:
        raise ValueError('Independently verified 633/635 prerequisites are required before data freeze')
    if len(ideal_states) != 8: raise ValueError('Exactly eight existing ideal states are required')
    if Path(output).exists(): raise FileExistsError('Freeze output already exists')
    exclusions = set(excluded_families)
    rng = np.random.default_rng(seed)
    eligible = []
    start_keys = set()
    for i in rng.permutation(8):
        values = env.validate_occupancy(ideal_states[int(i)])
        key = env.occupancy_key(values)
        if key not in start_keys:
            start_keys.add(key)
            eligible.append({'source_index':int(i),'occupancy':list(values),'canonical_key':key})
            exclusions.add(key)
        if len(eligible)==4: break
    if len(eligible)!=4: raise ValueError('Four distinct existing starts could not be selected')
    heldout=[]
    for _ in range(1000):
        state_seed = int(rng.integers(0,2**31))
        occupancy = env.validate_occupancy(make_state(state_seed,3,16).numbers)
        key=env.occupancy_key(occupancy)
        if key in exclusions: continue
        exclusions.add(key)
        heldout.append({'occupancy':list(occupancy),'canonical_key':key,'seed':state_seed,'unscored':True,
                        'messages':env.encode_prompt_v2(unscored_state(occupancy))})
        if len(heldout)==20: break
    if len(heldout)!=20: raise ValueError('Could not freeze twenty distinct legality families')
    output=Path(output); output.mkdir(parents=True)
    for filename,rows in [('starts.jsonl',eligible),('legality_v2.jsonl',heldout)]:
        (output/filename).write_text(''.join(json.dumps(r,sort_keys=True,allow_nan=False)+'\n' for r in rows))
    manifest={'seed':seed,'starts':4,'legality_states':20,'DFT_labels_used':False,'physical_calls':0,
              'excluded_633_families':sorted(excluded_families),
              'files':{name:sha256(output/name) for name in ['starts.jsonl','legality_v2.jsonl']}}
    (output/'manifest.json').write_text(json.dumps(manifest,sort_keys=True,indent=2)+'\n')
    return manifest
