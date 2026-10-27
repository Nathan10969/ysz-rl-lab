"""Review11 legal-action token trie and prefix-masked GRPO math.

The trie is built from env.enumerate_legal_actions (the only physical legal
set) and the canonical Action.to_json text, tokenized exactly like SFT targets
(text tokens, then one EOS). The policy is the per-token prefix-constrained
policy: at each generated position the softmax is renormalized over the token
IDs the trie allows after the already generated prefix. This is NOT a global
conditioning of the unconstrained model on the legal string set.

Sampling, old-policy, reference and current-policy log-probabilities all use
the same allowed sets, stored immutably in each sample. Forced positions (one
allowed token) have log-prob exactly 0 and carry no gradient. PG and KL are
averaged over branching positions only (allowed set larger than one).
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import json
import math
import torch

import claude08_env as env
from claude08_grpo import (ActionSample, group_advantages, CLIP, KL_COEFFICIENT,
                           LEARNING_RATE, GRAD_CLIP, _finite)

POLICY_PROTOCOL = 'review11_prefix_trie_masked_v1'
# Before the single update policy and reference adapters hold identical
# weights and run identical full-context forwards; any difference is a bug.
OLD_REFERENCE_TOLERANCE = 1e-3
# KV-cached sampling logits vs full-context recompute in BF16: diagnostic.
# Alignment is enforced structurally (sampled token in its allowed set); this
# bound only catches gross errors and must not stop on BF16 numerical noise.
SAMPLING_RECOMPUTE_STOP = 5.0


class ConstraintError(RuntimeError):
    pass


class ActionTrie:
    """Token trie over every legal action of one occupancy."""

    def __init__(self, tokenizer, occupancy, *, max_new_tokens=64):
        self.occupancy = env.validate_occupancy(occupancy)
        eos = tokenizer.eos_token_id
        if type(eos) is not int:
            raise ConstraintError('A single integer EOS token is required')
        actions = env.enumerate_legal_actions(self.occupancy)
        if not actions:
            raise ConstraintError('Empty legal action set')
        children, leaves = {}, {}
        for action in actions:
            text = action.to_json()
            ids = tuple(tokenizer(text, add_special_tokens=False)['input_ids'])
            if not ids or eos in ids:
                raise ConstraintError('Action text tokenization is empty or contains EOS')
            if tokenizer.decode(list(ids), skip_special_tokens=False).strip() != text:
                raise ConstraintError('Action tokenization does not round-trip: ' + text)
            sequence = ids + (eos,)
            if len(sequence) > max_new_tokens:
                raise ConstraintError('Legal action exceeds the generation cap')
            if sequence in leaves:
                raise ConstraintError('Two legal actions share one token sequence')
            leaves[sequence] = action
            for i in range(len(sequence)):
                children.setdefault(sequence[:i], set()).add(sequence[i])
        # EOS is the only terminator, so no complete sequence is a proper
        # prefix of another and every leaf ends exactly at its EOS.
        if any(sequence in children for sequence in leaves):
            raise ConstraintError('A complete action sequence is a prefix of another')
        self.eos = eos
        self.children = {prefix: tuple(sorted(tokens)) for prefix, tokens in children.items()}
        self.leaves = leaves
        payload = sorted((list(seq), action.to_json()) for seq, action in leaves.items())
        self.digest = hashlib.sha256(json.dumps(payload, separators=(',', ':')).encode()).hexdigest()

    def __len__(self):
        return len(self.leaves)

    def allowed(self, prefix):
        try:
            return self.children[tuple(prefix)]
        except KeyError:
            raise ConstraintError('Dead prefix outside the legal trie') from None

    def action(self, sequence):
        try:
            return self.leaves[tuple(sequence)]
        except KeyError:
            raise ConstraintError('Generated sequence is not a complete legal action') from None


@dataclass(frozen=True)
class ConstrainedActionSample(ActionSample):
    """allowed[t] is the allowed set for causal target t+1; () for context."""
    allowed: tuple[tuple[int, ...], ...] = ()

    def __post_init__(self):
        super().__post_init__()
        allowed = tuple(tuple(int(i) for i in row) for row in self.allowed)
        object.__setattr__(self, 'allowed', allowed)
        if len(allowed) != len(self.input_ids) - 1:
            raise ValueError('Allowed sets must align with causal targets')
        for t, row in enumerate(allowed):
            target = self.action_mask[t + 1]
            if bool(row) != target:
                raise ValueError('Allowed sets must cover exactly the action targets')
            if row and (self.input_ids[t + 1] not in row or list(row) != sorted(set(row))):
                raise ValueError('Sampled token outside its sorted allowed set')
        if not any(len(row) > 1 for row in allowed):
            raise ValueError('An action needs at least one branching token')

    @property
    def branch_mask(self):
        return tuple(len(row) > 1 for row in self.allowed)


def masked_logprob(row_logits, allowed, token):
    """log p(token) under softmax renormalized over the allowed IDs."""
    index = torch.as_tensor(allowed, device=row_logits.device, dtype=torch.long)
    values = row_logits.float()
    return values[token] - torch.logsumexp(values.index_select(0, index), dim=0)


def masked_causal_logprobs(logits, input_ids, allowed):
    """Causal log-probs: full softmax on context targets, masked on actions.

    Context positions keep the full softmax only so the vector stays aligned
    with input tokens; losses never read them.
    """
    if logits.ndim != 3 or logits.shape[0] != 1 or input_ids.shape != logits.shape[:2]:
        raise ValueError('Expected 1,T,V logits and matching token IDs')
    if len(allowed) != input_ids.shape[1] - 1:
        raise ValueError('Allowed sets must align with causal targets')
    _finite(logits, 'logits')
    shifted = logits[0, :-1].float()
    targets = input_ids[0, 1:]
    full = shifted.log_softmax(-1).gather(-1, targets[:, None]).squeeze(-1)
    positions = [t for t, row in enumerate(allowed) if row]
    if not positions:
        return full
    masked = torch.stack([masked_logprob(shifted[t], allowed[t], int(targets[t])) for t in positions])
    index = torch.as_tensor(positions, device=full.device, dtype=torch.long)
    return full.index_put((index,), masked)


def branch_loss(current, old, reference, branch, advantage):
    """Clipped PG plus k3 KL, averaged over branching positions only."""
    if not (current.shape == old.shape == reference.shape == branch.shape) or current.ndim != 1:
        raise ValueError('Logprob vectors and branch mask must match')
    if branch.dtype != torch.bool or not branch.any().item():
        raise ValueError('Every trainable action requires branching tokens')
    for value, name in ((current, 'current logprobs'), (old, 'old logprobs'), (reference, 'reference logprobs')):
        _finite(value, name)
    branch = branch.to(current.device)
    now = current[branch]
    before = old.detach().to(current.device)[branch]
    ref = reference.detach().to(current.device)[branch]
    advantage = torch.as_tensor(advantage, device=current.device, dtype=now.dtype).detach()
    _finite(advantage, 'advantage')
    ratio = (now - before).exp()
    pg = -torch.minimum(ratio * advantage, ratio.clamp(1 - CLIP, 1 + CLIP) * advantage)
    difference = ref - now
    kl = difference.exp() - difference - 1
    loss = pg.mean() + KL_COEFFICIENT * kl.mean()
    _finite(loss, 'GRPO loss')
    return loss, dict(pg=float(pg.detach().mean()), kl=float(kl.detach().mean()),
                      max_abs_current_minus_old=float((now - before).detach().abs().max()))


def one_update_masked(parameters, episodes, returns, score_current, *, before_step=None, lr=LEARNING_RATE, optimizer=None):
    """At most one AdamW update over prefix-masked action log-probs.

    Same contract as claude08_grpo.one_update, except each episode is one mean
    over its branching action tokens and samples are ConstrainedActionSample.
    """
    advantages = group_advantages(returns).flatten()
    if len(episodes) != len(advantages):
        raise ValueError('Episodes must match complete G4 return groups')
    if not torch.count_nonzero(advantages).item():
        return {'status': 'NO_UPDATE_ZERO_GROUP_VARIANCE', 'optimizer_updates': 0,
                'advantages': advantages.tolist(), 'finite_gradients': True, 'parameter_delta_l2': 0.}
    parameters = [p for p in parameters if p.requires_grad]
    if not parameters or any(not episode for episode in episodes):
        raise ValueError('Trainable policy and nonempty episodes are required')
    if any(not isinstance(s, ConstrainedActionSample) for e in episodes for s in e):
        raise TypeError('Review11 update requires constrained samples')
    initial = [p.detach().clone() for p in parameters]
    hashes = [[sample.sha256 for sample in episode] for episode in episodes]
    # Review12.3: an optional persistent optimizer keeps Adam moments across updates within a job.
    optimizer = torch.optim.AdamW(parameters, lr=lr) if optimizer is None else optimizer
    optimizer.zero_grad(set_to_none=True)
    logs, max_ratio_gap = [], 0.
    try:
        for episode, advantage in zip(episodes, advantages):
            episode_tokens = sum(sum(sample.branch_mask) for sample in episode)
            stats = {'loss': 0., 'pg': 0., 'kl': 0., 'branch_tokens': episode_tokens, 'episodes': 1}
            for sample in episode:
                current = score_current(sample)
                old = torch.tensor(sample.old_logprobs, device=current.device)
                ref = torch.tensor(sample.reference_logprobs, device=current.device)
                branch = torch.tensor(sample.branch_mask, device=current.device)
                loss, part = branch_loss(current, old, ref, branch, advantage)
                weight = int(branch.sum()) / episode_tokens
                (loss * weight / len(episodes)).backward()
                stats['loss'] += float(loss.detach()) * weight
                stats['pg'] += part['pg'] * weight
                stats['kl'] += part['kl'] * weight
                max_ratio_gap = max(max_ratio_gap, part['max_abs_current_minus_old'])
                del current, loss
            logs.append(stats)
        for p in parameters:
            if p.grad is not None:
                _finite(p.grad, 'policy gradient')
        norm = torch.nn.utils.clip_grad_norm_(parameters, GRAD_CLIP, error_if_nonfinite=True)
        _finite(norm, 'gradient norm')
        if before_step is not None:
            before_step()
        optimizer.step()
        for p in parameters:
            _finite(p, 'updated parameter')
        delta = math.sqrt(sum(float((p.detach().double() - o.double()).square().sum())
                              for p, o in zip(parameters, initial)))
        if delta == 0:
            raise RuntimeError('One update produced no parameter change')
        if hashes != [[sample.sha256 for sample in episode] for episode in episodes]:
            raise RuntimeError('Immutable old/reference/allowed records changed')
        return {'status': 'UPDATED', 'optimizer_updates': 1, 'advantages': advantages.tolist(),
                'finite_gradients': True, 'gradient_norm_before_clip': float(norm),
                'parameter_delta_l2': delta, 'episodes': logs, 'record_hashes': hashes,
                'loss_normalization': 'mean over branching action tokens per episode',
                'max_abs_current_minus_old_before_step': max_ratio_gap}
    except Exception:
        with torch.no_grad():
            for p, o in zip(parameters, initial):
                p.copy_(o)
        raise
    finally:
        optimizer.zero_grad(set_to_none=True)
