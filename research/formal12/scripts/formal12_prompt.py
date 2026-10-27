"""Review12.3 meta step 5: prompt v3 = prompt v2 + neutral local Y-vacancy features.

The features are deterministic functions of the occupancy already present in the v2 prompt; they add no
energy information and no hint about which arrangement is favourable. Actions, trie and decoding are unchanged.
"""
from __future__ import annotations
import json

import numpy as np

import claude08_env as env

NOTE = (" local gives, from the mother lattice: vacancy_NN_Y = for each vacancy site, how many of its 4 nearest"
        " cation sites hold Y; Y_NN_vacancy = for each Y site, how many of its 8 nearest anion sites are vacant;"
        " pairs = total Y-vacancy pairs at the nearest and next-nearest cation-anion distances.")


def _shells():
    parent = env._parent()
    distances = env._distances()
    cations = np.flatnonzero(parent.numbers == 40); anions = np.flatnonzero(parent.numbers == 8)
    shells = np.unique(np.round(distances[np.ix_(cations, anions)], 6))
    return distances, shells[0], shells[1]


_CACHE = {}


def local_features(y_sites, vacancy_sites):
    if not _CACHE: _CACHE['d'] = _shells()
    distances, nn, nnn = _CACHE['d']
    y = np.array(y_sites, dtype=int); v = np.array(vacancy_sites, dtype=int)
    block = distances[np.ix_(v, y)]
    near = np.isclose(block, nn, atol=1e-5); next_near = np.isclose(block, nnn, atol=1e-5)
    return {'vacancy_NN_Y': {str(int(s)): int(c) for s, c in zip(v, near.sum(1))},
            'Y_NN_vacancy': {str(int(s)): int(c) for s, c in zip(y, near.sum(0))},
            'pairs': {'Y_vac_NN': int(near.sum()), 'Y_vac_NNN': int(next_near.sum())}}


def to_v3(messages):
    """Convert prompt-v2 messages (compact occupancy) to v3; pure and deterministic."""
    system, user = messages[0], messages[1]
    observation = json.loads(user['content'])
    occupancy = observation['occupancy']
    if not isinstance(occupancy, dict): raise ValueError('v3 conversion needs the compact occupancy encoding')
    observation['local'] = local_features(occupancy['Y_sites'], occupancy['vacancy_sites'])
    return [dict(system, content=system['content'] + NOTE),
            dict(user, content=json.dumps(observation, separators=(',', ':'), allow_nan=False))]


def encode_prompt_v3(state):
    return to_v3(env.encode_prompt_v2(state))
