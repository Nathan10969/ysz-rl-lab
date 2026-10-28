"""Complete replay log of LLM training batches (2026-10-27; spec: rsi/LOGGING_SPEC.md, replay v1).

Every training batch is written and fsynced before the optimizer step that consumes it. Per sample it keeps the
messages and full token ids, the menu and the chosen action, the allowed token set of every action token, the
old (sampling-policy) and reference log-probabilities exactly as the update uses them, and the group/step/lane;
per batch the returns and advantages, the per-branch reward components of step-level groups, and the policy and
reference versions. Logging only: nothing here changes what is sampled, scored or updated. Any inconsistency raises:
before the optimizer step when it can be seen then (sample mapping, tokens, mask, allowed sets, log-probs, policy
version, reward components and totals), right after it otherwise (advantages used by the optimizer); the job then fails.
"""
from __future__ import annotations
from collections import defaultdict, deque
import hashlib
import json
import os
from pathlib import Path

SCHEMA = 'formal12-replay-v1'


def _key(input_ids):
    return hashlib.sha256(json.dumps([int(i) for i in input_ids]).encode()).hexdigest()


def _write(path, value):
    """Atomic, fsynced JSON; returns the sha256 of the written bytes."""
    data = (json.dumps(value, sort_keys=True, allow_nan=False) + '\n').encode()
    temporary = Path(str(path) + '.tmp')
    with open(temporary, 'wb') as handle:
        handle.write(data); handle.flush(); os.fsync(handle.fileno())
    os.replace(temporary, path)
    return hashlib.sha256(data).hexdigest()


def _prompt_length(action):
    return list(action.action_mask).index(True)


class ReplayLog:
    def __init__(self, directory, *, job):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True)
        self.job, self.pending, self.noted, self.batches = dict(job), defaultdict(deque), 0, []
        self.last_policy_hash, self.outcomes = None, []

    def note(self, action, *, messages, env_text, menu, choice, group, step, lane, sampling):
        """One generated action, after reference scoring (old and reference log-probs are final)."""
        p = _prompt_length(action)
        record = dict(sample_index=self.noted, group=group, step=step, lane=lane, messages=messages,
                      input_ids=[int(i) for i in action.input_ids], prompt_tokens=p, action_text=action.text,
                      env_text=env_text, choice=choice,
                      menu=None if menu is None else [dict(k=int(m['k']), action=m['action'].to_json(), dNN=int(m['dNN']),
                                                           dNNN=int(m['dNNN'])) for m in menu],
                      allowed=[[int(i) for i in row] for row in action.allowed[p - 1:]],
                      old_logprobs=[float(x) for x in action.old_logprobs[p - 1:]],
                      reference_logprobs=[float(x) for x in action.reference_logprobs[p - 1:]],
                      sampling_vs_recompute_max_abs=sampling.get('sampling_vs_recompute_max_abs'))
        self.pending[_key(action.input_ids)].append(record)
        self.noted += 1

    def note_outcomes(self, *, group, step, branches):
        """Step-level group: per-branch reward components; `total` is the reward the update uses."""
        self.outcomes.append(dict(group=group, step=step, branches=list(branches)))

    def write_batch(self, episodes, returns, *, policy_hash, reference_hash, initial_policy_hash):
        """Before the update: match every episode sample to its noted record and persist the batch."""
        from claude08_grpo import group_advantages
        advantages = group_advantages(returns).flatten().tolist()
        if len(advantages) != len(episodes): raise RuntimeError('Replay: advantages and episodes differ in length')
        out = []
        for index, (episode, advantage) in enumerate(zip(episodes, advantages)):
            samples = []
            for sample in episode:
                queue = self.pending.get(_key(sample.input_ids))
                if not queue: raise RuntimeError('Replay: training sample was never noted')
                record = queue.popleft()
                p = record['prompt_tokens']
                if (record['input_ids'] != [int(i) for i in sample.input_ids]
                        or list(sample.action_mask) != [False] * p + [True] * (len(sample.input_ids) - p)
                        or record['allowed'] != [[int(i) for i in row] for row in sample.allowed[p - 1:]]
                        or any(row for row in sample.allowed[:p - 1])
                        or record['old_logprobs'] != [float(x) for x in sample.old_logprobs[p - 1:]]
                        or record['reference_logprobs'] != [float(x) for x in sample.reference_logprobs[p - 1:]]):
                    raise RuntimeError('Replay: noted record differs from the training sample')
                samples.append(record)
            out.append(dict(episode=index, advantage=advantage, samples=samples))
        if any(self.pending.values()): raise RuntimeError('Replay: noted samples missing from the training batch')
        expected = self.last_policy_hash or initial_policy_hash
        if policy_hash != expected: raise RuntimeError('Replay: sampling policy is not the previous update\'s policy')
        outcomes, self.outcomes = self.outcomes, []
        if self.job.get('arm') == 'grpo_step' and len(outcomes) != len(returns):
            raise RuntimeError('Replay: step-level batch needs one reward record per group')
        # summed in the order step_group uses, so equality is exact
        if any(b['immediate'] + b['novelty'] + b['illegal'] != b['total'] for o in outcomes for b in o['branches']):
            raise RuntimeError('Replay: reward components do not add up to the recorded total')
        if outcomes and [[b['total'] for b in o['branches']] for o in outcomes] != [[float(x) for x in row] for row in returns]:
            raise RuntimeError('Replay: recorded reward totals differ from the returns used')
        batch = dict(schema=SCHEMA, job=self.job, batch_index=len(self.batches), policy_hash=policy_hash,
                     policy_hash_matches_previous_update=True, reference_hash=reference_hash,
                     returns=[[float(x) for x in row] for row in returns], advantages=advantages, episodes=out,
                     step_outcomes=outcomes)
        path = self.directory / f'batch_{len(self.batches):03d}.json'
        digest = _write(path, batch)
        self.batches.append(dict(batch_index=batch['batch_index'], file=path.name, sha256=digest,
                                 samples=sum(len(e['samples']) for e in out), episodes=len(out)))
        return self.batches[-1]

    def record_update(self, result):
        """After the update: link the optimizer result to the batch it consumed."""
        entry = self.batches[-1]
        entry.update(status=result.get('status'), policy_hash_after=result.get('policy_hash_after'),
                     advantages_match=result.get('advantages') == json.loads(
                         (self.directory / entry['file']).read_text())['advantages'])
        self.last_policy_hash = result.get('policy_hash_after')
        with open(self.directory / 'updates.jsonl', 'a') as handle:
            handle.write(json.dumps(entry, sort_keys=True, allow_nan=False) + '\n'); handle.flush(); os.fsync(handle.fileno())
        if not entry['advantages_match']:   # the step already happened; the evidence is on disk, the job fails
            raise RuntimeError('Replay: optimizer advantages differ from the persisted batch')

    def close(self):
        """Samples noted after the last update were never trained on; keep them, marked as unused."""
        left = [r for queue in self.pending.values() for r in queue]
        if left: _write(self.directory / 'unused_samples.json', dict(schema=SCHEMA, job=self.job, samples=left,
                                                                    step_outcomes=self.outcomes))
        return dict(schema=SCHEMA, directory=self.directory.name, noted=self.noted, batches=self.batches,
                    unused_samples=len(left))
