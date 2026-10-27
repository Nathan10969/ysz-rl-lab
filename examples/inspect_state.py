#!/usr/bin/env python3
"""Inspect one actual YSZ lattice state using the research geometry code. No model is loaded."""
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research/formal12/scripts"))
from calibration import make_state
import claude08_env as env
from formal12_prompt import local_features

state = make_state(2026102301, 3, 16)
occupancy = env.validate_occupancy(state.numbers)
features = local_features(np.flatnonzero(state.numbers == 39), np.flatnonzero(state.numbers == 0))
actions = env.enumerate_legal_actions(occupancy)
print(json.dumps({"mother_sites": len(state), "real_atoms": int((state.numbers != 0).sum()),
    "composition": {name: int((state.numbers == number).sum()) for name, number in (("Zr", 40), ("Y", 39), ("O", 8), ("vacancy", 0))},
    "legal_action_count": len(actions), "local_pair_counts": features["pairs"],
    "one_legal_action": json.loads(actions[0].to_json()), "physical_model_calls": 0}, indent=2))
