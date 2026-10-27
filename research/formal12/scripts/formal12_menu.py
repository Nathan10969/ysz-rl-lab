"""Review12.3 meta step 6: candidate-menu agent (prompt/action v4).

At every step the policy sees M legal candidate moves sampled deterministically from the current state
(same state -> same menu, i.e. common random numbers across policies), each annotated with neutral local
features of the requested occupation: change in Y-vacancy nearest / next-nearest cation-anion pairs.
The policy outputs {"choice":k}; constrained decoding restricts k to the menu. The environment receives
the chosen candidate's canonical action text, so physics, rewards and accounting are unchanged.
"""
from __future__ import annotations
import hashlib
import json

import numpy as np

import claude08_env as env
from constrained import ConstraintError
from formal12_prompt import local_features, to_v3

MENU_SIZE = 16
NOTE = (' candidates lists the moves you may choose from: k, op, sites, and dNN/dNNN = the change in the number of'
        ' Y-vacancy pairs at the nearest / next-nearest cation-anion distance if that move is applied (before relaxation).'
        ' Return exactly one JSON object {"choice":k} with k the index of the chosen candidate, no other text.')


def _pairs(occupancy):
    occupancy = np.asarray(occupancy)
    return local_features(np.flatnonzero(occupancy == 39), np.flatnonzero(occupancy == 0))['pairs']


def make_menu(occupancy, salt, size=MENU_SIZE):
    """Deterministic sample of legal actions with feature deltas; salt = (seed, terminal key, step)."""
    legal = env.enumerate_legal_actions(occupancy)
    digest = hashlib.sha256(json.dumps(salt, sort_keys=True, default=str).encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], 'little'))
    picks = rng.choice(len(legal), min(size, len(legal)), replace=False)
    before = _pairs(occupancy)
    menu = []
    for k, index in enumerate(picks):
        action = legal[int(index)]
        after = _pairs(env.apply_action(occupancy, action))
        menu.append(dict(k=k, action=action, op=action.op, sites=list(action.sites),
                         dNN=after['Y_vac_NN'] - before['Y_vac_NN'], dNNN=after['Y_vac_NNN'] - before['Y_vac_NNN']))
    return menu


def to_v4(messages_v2, menu):
    system, user = to_v3(messages_v2)
    observation = json.loads(user['content'])
    observation['candidates'] = [dict(k=m['k'], op=m['op'], sites=m['sites'], dNN=m['dNN'], dNNN=m['dNNN']) for m in menu]
    system = dict(system, content=system['content'].replace(
        'Return exactly one JSON object with keys op and sites, no other text. ', '') + NOTE)
    return [system, dict(user, content=json.dumps(observation, separators=(',', ':'), allow_nan=False))]


def encode_prompt_v4(state, menu):
    return to_v4(env.encode_prompt_v2(state), menu)


def choice_text(k):
    return json.dumps({'choice': int(k)}, separators=(',', ':'))


class MenuTrie:
    """Same interface as constrained.ActionTrie, over the menu's {"choice":k} strings."""

    def __init__(self, tokenizer, menu, *, max_new_tokens=64):
        eos = tokenizer.eos_token_id
        children, leaves = {}, {}
        for m in menu:
            text = choice_text(m['k'])
            ids = tuple(tokenizer(text, add_special_tokens=False)['input_ids'])
            if not ids or eos in ids or tokenizer.decode(list(ids), skip_special_tokens=False).strip() != text:
                raise ConstraintError('Choice tokenization does not round-trip: ' + text)
            sequence = ids + (eos,)
            if len(sequence) > max_new_tokens or sequence in leaves: raise ConstraintError('Bad choice sequence')
            leaves[sequence] = m
            for i in range(len(sequence)): children.setdefault(sequence[:i], set()).add(sequence[i])
        if any(sequence in children for sequence in leaves): raise ConstraintError('Choice sequence is a prefix')
        self.eos, self.leaves = eos, leaves
        self.children = {p: tuple(sorted(t)) for p, t in children.items()}
        self.digest = hashlib.sha256(json.dumps(sorted((list(s), m['k']) for s, m in leaves.items())).encode()).hexdigest()

    def __len__(self):
        return len(self.leaves)

    def allowed(self, prefix):
        try: return self.children[tuple(prefix)]
        except KeyError: raise ConstraintError('Dead prefix outside the menu trie') from None

    def action(self, sequence):
        try: return self.leaves[tuple(sequence)]
        except KeyError: raise ConstraintError('Not a complete menu choice') from None


# Frozen before any greedy evaluation: linear fit of step dE on (dNN, dNNN) over the seed-1 D0 random/SA chains
# (training data only; formal12/greedy_weights.json, n=348, R2=0.083).
GREEDY_WEIGHTS = (0.07666600110740183, -0.08017777486080024)


def greedy_choice(menu, rng):
    """Non-LLM feature heuristic baseline: minimise the D0-fitted predicted dE; random tie-break."""
    score = lambda m: GREEDY_WEIGHTS[0] * m['dNN'] + GREEDY_WEIGHTS[1] * m['dNNN']
    best = min(score(m) for m in menu)
    tied = [m for m in menu if abs(score(m) - best) < 1e-12]
    return tied[int(rng.integers(len(tied)))]


def sft_rows_v4(rows_v2, *, seed, copies=4):
    """Teacher rows -> menu rows: the teacher action plus MENU_SIZE-1 other legal moves, shuffled; target = its index."""
    out = []
    for index, row in enumerate(rows_v2):
        observation = json.loads(row['messages'][1]['content'])
        occupancy = env.decode_occupancy(observation['occupancy'])
        teacher = env.parse_action(json.dumps(row['action']))
        legal = [a for a in env.enumerate_legal_actions(occupancy) if a != teacher]
        for copy in range(copies):
            rng = np.random.default_rng([seed, index, copy])
            others = [legal[int(i)] for i in rng.choice(len(legal), MENU_SIZE - 1, replace=False)]
            actions = others + [teacher]
            order = rng.permutation(len(actions))
            before = _pairs(occupancy)
            menu = []
            for k, j in enumerate(order):
                a = actions[int(j)]; after = _pairs(env.apply_action(occupancy, a))
                menu.append(dict(k=k, action=a, op=a.op, sites=list(a.sites), dNN=after['Y_vac_NN'] - before['Y_vac_NN'],
                                 dNNN=after['Y_vac_NNN'] - before['Y_vac_NNN']))
            target = next(m['k'] for m in menu if m['action'] == teacher)
            out.append(dict(messages=to_v4(row['messages'], menu), choice=target))
    return out
