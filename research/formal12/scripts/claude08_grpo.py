"""Review08 GRPO math; Torch only, no model loading or execution on import.

Masks align with input tokens. Causal logits at position t predict token t+1,
so the mask is shifted once, alongside labels. Every episode contributes one
mean over *all* its action/EOS tokens, regardless of action length.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import hashlib
import json
import math
import torch

GROUP_SIZE = 4
CLIP = .2
KL_COEFFICIENT = .03
LEARNING_RATE = 1e-5
GRAD_CLIP = 1.
EPSILON = 1e-8


def _finite(tensor, name):
    if not torch.isfinite(tensor).all().item():
        raise FloatingPointError('Nonfinite ' + name)


def group_advantages(returns):
    values = torch.as_tensor(returns)
    if not values.is_floating_point():
        values = values.to(torch.float64)
    if values.ndim != 2 or values.shape[1] != GROUP_SIZE or not values.shape[0]:
        raise ValueError('Returns must contain complete groups of four episodes')
    _finite(values, 'returns')
    centered = values - values.mean(dim=1, keepdim=True)
    std = values.std(dim=1, unbiased=False, keepdim=True)
    return torch.where(std <= EPSILON, torch.zeros_like(centered), centered / std.clamp_min(EPSILON))


def causal_logprobs(logits, input_ids):
    """Full vocabulary softmax, no filtering, temperature exactly one."""
    if logits.ndim != 3 or input_ids.shape != logits.shape[:2] or input_ids.shape[1] < 2:
        raise ValueError('Expected B,T,V logits and matching B,T token IDs')
    _finite(logits, 'logits')
    return logits[:, :-1].float().log_softmax(dim=-1).gather(-1, input_ids[:, 1:, None]).squeeze(-1)


@dataclass(frozen=True)
class ActionSample:
    input_ids: tuple[int, ...]
    action_mask: tuple[bool, ...]
    old_logprobs: tuple[float, ...]
    reference_logprobs: tuple[float, ...]

    def __post_init__(self):
        for name in ('input_ids', 'action_mask', 'old_logprobs', 'reference_logprobs'):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        n = len(self.input_ids)
        if n < 2 or len(self.action_mask) != n or len(self.old_logprobs) != n-1 or len(self.reference_logprobs) != n-1:
            raise ValueError('Token IDs/mask and causal logprob lengths differ')
        if any(type(x) is not int or x < 0 for x in self.input_ids):
            raise ValueError('Token IDs must be nonnegative integers')
        if any(type(x) is not bool for x in self.action_mask) or self.action_mask[0] or not any(self.action_mask):
            raise ValueError('Action mask must exclude the initial context and include action targets')
        if not all(math.isfinite(float(x)) for x in self.old_logprobs+self.reference_logprobs):
            raise FloatingPointError('Nonfinite immutable logprobs')

    @property
    def sha256(self):
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def state_fingerprint(state):
    """Stable byte digest also supports BF16 tensors without NumPy BF16."""
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        cpu = tensor.detach().cpu().contiguous()
        digest.update(name.encode()); digest.update(str(cpu.dtype).encode())
        digest.update(str(tuple(cpu.shape)).encode())
        digest.update(cpu.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def episode_loss(new_logprobs, old_logprobs, reference_logprobs, masks, advantages):
    n = len(new_logprobs)
    if not n or any(len(x) != n for x in (old_logprobs, reference_logprobs, masks)) or len(advantages) != n:
        raise ValueError('Exactly one advantage and mask per episode is required')
    losses, pg_values, kl_values, token_count = [], [], [], 0
    for new, old, ref, mask, advantage in zip(new_logprobs, old_logprobs, reference_logprobs, masks, advantages):
        if new.ndim != 1 or old.shape != new.shape or ref.shape != new.shape or mask.shape != new.shape:
            raise ValueError('Episode logprobs and mask must be matching vectors')
        if mask.dtype != torch.bool or not mask.any().item():
            raise ValueError('Every trainable episode requires action/EOS tokens')
        for value, name in ((new,'current logprobs'), (old,'old logprobs'), (ref,'reference logprobs')):
            _finite(value, name)
        mask = mask.to(new.device)
        current = new[mask]
        previous = old.detach().to(new.device)[mask]
        reference = ref.detach().to(new.device)[mask]
        advantage = torch.as_tensor(advantage, device=new.device, dtype=current.dtype).detach()
        _finite(advantage, 'advantage')
        ratio = (current-previous).exp()
        pg = -torch.minimum(ratio*advantage, ratio.clamp(1-CLIP, 1+CLIP)*advantage)
        difference = reference-current
        kl = difference.exp()-difference-1
        loss = pg.mean() + KL_COEFFICIENT*kl.mean()
        _finite(loss, 'GRPO loss')
        losses.append(loss); pg_values.append(float(pg.detach().mean())); kl_values.append(float(kl.detach().mean()))
        token_count += int(mask.sum())
    return torch.stack(losses).mean(), {'pg':sum(pg_values)/n, 'kl':sum(kl_values)/n,
                                     'episodes':n, 'action_tokens':token_count}


def one_update(parameters, episodes, returns, score_current, *, before_step=None):
    """At most one AdamW update; scoring callback keeps model dependencies out.

    ``episodes`` is group-major flattened, each a nonempty list of ActionSamples.
    ``score_current`` returns causal logprobs for one complete action context.
    The callback must charge its full-context forward before performing it.
    """
    advantages = group_advantages(returns).flatten()
    if len(episodes) != len(advantages):
        raise ValueError('Episodes must match complete G4 return groups')
    if not torch.count_nonzero(advantages).item():
        return {'status':'NO_UPDATE_ZERO_GROUP_VARIANCE','optimizer_updates':0,
                'advantages':advantages.tolist(), 'finite_gradients':True, 'parameter_delta_l2':0.}
    parameters = [p for p in parameters if p.requires_grad]
    if not parameters or any(not episode for episode in episodes):
        raise ValueError('Trainable policy and nonempty episodes are required')
    initial = [p.detach().clone() for p in parameters]
    hashes = [[sample.sha256 for sample in episode] for episode in episodes]
    optimizer = torch.optim.AdamW(parameters, lr=LEARNING_RATE)
    optimizer.zero_grad(set_to_none=True)
    logs = []
    try:
        for episode, advantage in zip(episodes, advantages):
            # Algebraically the episode mean, but only one action graph is live.
            # The sum of token_count/episode_tokens weighted action means equals
            # a single mean over every action/EOS target in this trajectory.
            episode_tokens = sum(sum(sample.action_mask[1:]) for sample in episode)
            stats = {'loss':0.,'pg':0.,'kl':0.,'action_tokens':episode_tokens,
                     'episodes':1,'current_logprobs':[]}
            for sample in episode:
                current = score_current(sample)
                old = torch.tensor(sample.old_logprobs, device=current.device)
                ref = torch.tensor(sample.reference_logprobs, device=current.device)
                mask = torch.tensor(sample.action_mask[1:], device=current.device)
                loss, part = episode_loss([current], [old], [ref], [mask], [advantage])
                weight = int(mask.sum()) / episode_tokens
                (loss * weight / len(episodes)).backward()
                stats['loss'] += float(loss.detach()) * weight
                stats['pg'] += part['pg'] * weight
                stats['kl'] += part['kl'] * weight
                stats['current_logprobs'].extend(current.detach().cpu().tolist())
                del current, loss
            logs.append(stats)
        for p in parameters:
            if p.grad is not None: _finite(p.grad, 'policy gradient')
        norm = torch.nn.utils.clip_grad_norm_(parameters, GRAD_CLIP, error_if_nonfinite=True)
        _finite(norm, 'gradient norm')
        if before_step is not None: before_step()
        optimizer.step()
        for p in parameters: _finite(p, 'updated parameter')
        delta = math.sqrt(sum(float((p.detach().double()-old.double()).square().sum()) for p, old in zip(parameters, initial)))
        if delta == 0: raise RuntimeError('One update produced no parameter change')
        if hashes != [[sample.sha256 for sample in episode] for episode in episodes]:
            raise RuntimeError('Immutable old/reference/mask records changed')
        return {'status':'UPDATED','optimizer_updates':1,'advantages':advantages.tolist(),
                'finite_gradients':True,'gradient_norm_before_clip':float(norm),
                'parameter_delta_l2':delta, 'episodes':logs, 'record_hashes':hashes}
    except Exception:
        # A partial/nonfinite step is never exposed as a usable policy.
        with torch.no_grad():
            for p, old in zip(parameters, initial): p.copy_(old)
        raise
    finally:
        optimizer.zero_grad(set_to_none=True)
