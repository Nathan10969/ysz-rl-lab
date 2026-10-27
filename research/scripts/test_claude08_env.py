"""CPU-only contract tests: synthetic results never call a physical model."""
import importlib
import itertools
import json
import unittest

import numpy as np

from calibration import make_state, reference

try:
    env = importlib.import_module("claude08_env")
except ModuleNotFoundError as exc:
    if exc.name != "claude08_env":
        raise
    env = None


class EnvironmentContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.parent = reference(3)
        cls.occupation = tuple(map(int, make_state(31, 3, 16).numbers))

    def setUp(self):
        self.assertIsNotNone(env, "CPU Review08 environment implementation is required")

    def result(self, occupancy=None, energy=100.0, offset=0.0):
        atoms = env.ideal_atoms(self.occupation if occupancy is None else occupancy)
        atoms.positions += offset
        return env.PhysicsResult(atoms, energy, np.zeros((len(atoms), 3)), True)

    def environment(self, callback, **kwargs):
        return env.Environment(self.result(), callback, arm="test", seed=31, **kwargs)

    def action(self, occupancy=None, op=None):
        actions = env.enumerate_legal_actions(self.occupation if occupancy is None else occupancy)
        return next(action for action in actions if op is None or action.op == op)

    def test_strict_json_rejects_duplicates_nonintegers_and_extra_text(self):
        good = '{"op":"move_vacancy","sites":[1,2]}'
        self.assertEqual(env.parse_action(good), env.Action("move_vacancy", (1, 2)))
        bad = [good + " trailing", "```json\n" + good + "\n```", good.replace("1,2", "1.0,2"),
               good.replace("1,2", "true,2"), good.replace("1,2", "1,1"),
               good.replace("1,2", "-1,2"), good.replace("1,2", "1,324"),
               '{"op":"move_vacancy","op":"move_vacancy","sites":[1,2]}',
               '{"op":"move_vacancy","sites":[1,2],"extra":0}',
               '{"op":"bad","sites":[1,2]}', '{"op":"move_vacancy","sites":[1,2,3]}']
        for text in bad:
            with self.subTest(text=text), self.assertRaises(env.ActionError):
                env.parse_action(text)

    def test_sublattice_species_distance_endpoints_and_composition(self):
        numbers = np.asarray(self.occupation)
        cats = np.flatnonzero(self.parent.numbers == 40)
        oxy = np.flatnonzero(self.parent.numbers == 8)
        self.assertFalse(np.array_equal(cats, np.arange(108)))
        self.assertEqual(len(self.parent), 324)
        self.assertEqual(np.unique(numbers, return_counts=True)[1].tolist(), [8, 208, 16, 92])
        distances = self.parent.get_all_distances(mic=True)
        for op, sites, species, cutoff in [("swap_cations", cats, (39, 40), 5.14),
                                          ("move_vacancy", oxy, (0, 8), 3.63452886)]:
            i = int(sites[0])
            for target_distance, legal in [(cutoff, True), (float(distances[i, sites].max()), False)]:
                j = int(sites[np.argmin(abs(distances[i, sites] - target_distance))])
                occupied = numbers.copy()
                donor = int(np.flatnonzero((occupied == species[0]) & (np.arange(324) != j))[0])
                occupied[i], occupied[donor] = occupied[donor], occupied[i]
                if occupied[j] != species[1]:
                    donor = int(np.flatnonzero((occupied == species[1]) & (np.arange(324) != i))[0])
                    occupied[j], occupied[donor] = occupied[donor], occupied[j]
                action = env.Action(op, (i, j))
                if legal:
                    after = env.apply_action(occupied, action)
                    self.assertEqual(sorted(after), sorted(occupied))
                    self.assertEqual((after[i], after[j]), (species[1], species[0]))
                else:
                    with self.assertRaises(env.ActionError):
                        env.apply_action(occupied, action)
        with self.assertRaises(env.ActionError):
            env.apply_action(numbers, env.Action("move_vacancy", (int(cats[0]), int(oxy[0]))))
        corrupt = numbers.copy(); corrupt[cats[0]] = 8
        with self.assertRaises(ValueError):
            env.validate_occupancy(corrupt)

    def test_invalid_action_costs_attempt_but_zero_physics(self):
        def forbidden(*args):
            self.fail("Illegal output reached callback")
        instance = self.environment(forbidden)
        before = instance.state.occupancy
        outcome = instance.step('{"op":"move_vacancy","sites":[true,1]}')
        self.assertEqual((outcome.reward, outcome.energy_reward, outcome.novelty_reward), (-1, 0, 0))
        self.assertEqual(instance.state.occupancy, before)
        self.assertEqual((instance.counters.attempts, instance.counters.relax_actual, instance.counters.raw_ef), (1, 0, 0))

    def test_current_terminal_drives_next_action_and_reconstruction_is_ideal(self):
        first = self.action(op="move_vacancy")
        request_occupation = env.apply_action(self.occupation, first)
        second = next(a for a in env.enumerate_legal_actions(request_occupation)
                      if a.op == "move_vacancy" and first.sites[1] in a.sites and a != first)
        terminal_occupation = env.apply_action(request_occupation, second)
        callbacks = []
        def callback(request, counter):
            callbacks.append(request)
            counter.before_forward(1)
            expected = env.ideal_atoms(request.occupancy)
            np.testing.assert_allclose(request.atoms.positions, expected.positions)
            return self.result(terminal_occupation if len(callbacks) == 1 else request.occupancy, 99, .01)
        instance = self.environment(callback)
        self.assertTrue(instance.step(first.to_json()).success)
        third = next(a for a in env.enumerate_legal_actions(terminal_occupation)
                     if a not in env.enumerate_legal_actions(request_occupation))
        self.assertTrue(instance.step(third.to_json()).success)
        self.assertEqual(callbacks[1].occupancy, tuple(env.apply_action(terminal_occupation, third)))
        self.assertEqual((instance.counters.relax_actual, instance.counters.raw_ef), (2, 2))

    def test_request_cache_is_exact_verified_and_isolated(self):
        cache = env.RequestCache()
        calls = []
        def callback(request, counter):
            calls.append(request.key)
            counter.before_forward(2)
            return self.result(request.occupancy, 99)
        action = self.action()
        one = self.environment(callback, cache=cache)
        first = one.step(action.to_json())
        two = self.environment(callback, cache=cache)
        second = two.step(action.to_json())
        self.assertEqual((first.cache_hit, second.cache_hit), (False, True))
        self.assertEqual((two.counters.relax_actual, two.counters.raw_ef, two.counters.cache_hits), (0, 0, 1))
        other = env.Environment(self.result(), callback, arm="other", seed=31, cache=cache)
        other.step(action.to_json())
        seed = env.Environment(self.result(), callback, arm="test", seed=32, cache=cache)
        seed.step(action.to_json())
        self.assertEqual(len(calls), 3)
        request = env.build_request(env.apply_action(self.occupation, action), env.RelaxationSpec())
        changed = env.build_request(request.occupancy, env.RelaxationSpec(model_sha="alternate-model-identity"))
        self.assertNotEqual(request.key, changed.key)
        renumbered = list(request.occupancy)
        extra = next(a for a in env.enumerate_legal_actions(renumbered) if a != action)
        self.assertNotEqual(request.key, env.build_request(env.apply_action(renumbered, extra), env.RelaxationSpec()).key)

    def test_posthoc_terminal_duplicate_still_costs_and_preserves_geometry_variants(self):
        cache = env.RequestCache()
        calls = []
        def callback(request, counter):
            calls.append(request.key)
            return self.result(energy=99 if len(calls) == 1 else 99.02, offset=.001 * len(calls))
        instance = self.environment(callback, cache=cache)
        actions = env.enumerate_legal_actions(self.occupation)
        instance.step(actions[0].to_json())
        outcome = instance.step(actions[1].to_json())
        self.assertEqual(instance.counters.relax_actual, 2)
        self.assertFalse(outcome.cache_hit)
        self.assertEqual(outcome.novelty_reward, 0)
        records = cache.terminal_records(("test", 31), env.occupancy_key(self.occupation))
        self.assertEqual(len(records), 2)
        self.assertNotEqual(records[0].mapping.geometry_hash, records[1].mapping.geometry_hash)
        self.assertTrue(any(d["kind"] == "terminal_energy_conflict" for d in cache.diagnostics))

    def test_cache_returned_shell_metadata_cannot_mutate_either_cache_store(self):
        cache = env.RequestCache()
        namespace = ("test", 31)
        request = env.build_request(self.occupation)
        recorded = cache.record(namespace, request, self.result())
        terminal_key = recorded.mapping.canonical_key
        expected = dict(recorded.mapping.shell_counts)
        def assert_stores_unchanged():
            self.assertEqual(cache.get(namespace, request).mapping.shell_counts, expected)
            self.assertEqual(cache.terminal_records(namespace, terminal_key)[0].mapping.shell_counts, expected)
        recorded.mapping.shell_counts["Y_vac_NN"] += 100
        assert_stores_unchanged()
        fetched = cache.get(namespace, request)
        fetched.mapping.shell_counts["Y_vac_NNN"] += 200
        assert_stores_unchanged()
        terminal_records = cache.terminal_records(namespace, terminal_key)
        terminal_records[0].mapping.shell_counts.clear()
        assert_stores_unchanged()

    def test_failure_unconverged_nonfinite_and_raw_budget_are_charged(self):
        action = self.action()
        def crash(request, counter):
            self.assertEqual(instance.counters.relax_actual, 1)
            counter.before_forward(1)
            raise RuntimeError("synthetic failure")
        instance = self.environment(crash)
        self.assertEqual(instance.step(action.to_json()).reward, -1)
        self.assertEqual((instance.counters.relax_actual, instance.counters.raw_ef), (1, 1))
        for change in ["converged", "energy", "forces", "fmax"]:
            def invalid(request, counter, change=change):
                result = self.result(request.occupancy)
                if change == "converged": result.converged = False
                if change == "energy": result.energy = np.nan
                if change == "forces": result.forces[0, 0] = np.inf
                if change == "fmax": result.forces[0, 0] = .03001
                return result
            trial = self.environment(invalid)
            with self.subTest(change=change):
                self.assertEqual(trial.step(action.to_json()).reward, -1)
                self.assertEqual(trial.state.energy, 100)
                self.assertEqual(trial.counters.relax_actual, 1)
        raw = env.RawEFCounter(env.Counters(), env.BudgetLimits(max_raw_ef=3))
        raw.before_forward(2)
        with self.assertRaises(env.BudgetExceeded): raw.before_forward(2)
        self.assertEqual((raw.counters.raw_ef, raw.counters.forward_calls), (2, 1))
        exhausted = self.environment(crash, limits=env.BudgetLimits(max_relax_actual=0))
        with self.assertRaises(env.BudgetExceeded): exhausted.step(action.to_json())
        self.assertEqual(exhausted.counters.relax_actual, 0)

    def test_energy_reward_uses_historical_best_and_has_no_cycle_gain(self):
        energies = iter([102, 95, 100])
        def callback(request, counter): return self.result(request.occupancy, next(energies))
        instance = self.environment(callback)
        terms = []
        for _ in range(3):
            # A new exact request each time avoids a cache changing the supplied energy sequence.
            action = next(a for a in env.enumerate_legal_actions(instance.state.occupancy)
                          if env.build_request(env.apply_action(instance.state.occupancy, a), instance.spec).key
                          not in {h.request_key for h in instance.state.history})
            terms.append(instance.step(action.to_json()).energy_reward)
        self.assertEqual(terms, [0, 25, 0])
        self.assertEqual(instance.state.best_energy, 95)

    def test_group_frozen_history_makes_rollout_order_irrelevant(self):
        def run(order):
            history = {env.occupancy_key(self.occupation)}
            group = env.NoveltyGroup(history, group_size=4)
            rewards = {}
            for index in order:
                instance = self.environment(lambda request, counter: self.result(request.occupancy, 100),
                                            frozen_history=group.frozen_history)
                outcome = instance.step(self.action().to_json())
                rewards[index] = outcome.novelty_reward
                group.register(instance.state)
                self.assertEqual(len(history), 1)
            group.merge()
            return rewards, history
        forward, history = run([0, 1, 2, 3])
        reverse, reversed_history = run([3, 2, 1, 0])
        self.assertEqual(forward, reverse)
        self.assertEqual(list(forward.values()), [.05] * 4)
        self.assertEqual(history, reversed_history)

    def test_group_duplicate_novelty_events_are_not_global_unique_count(self):
        history = {env.occupancy_key(self.occupation)}
        group = env.NoveltyGroup(history)
        counters = env.Counters()
        for _ in range(4):
            instance = self.environment(lambda request, counter: self.result(request.occupancy),
                                        counters=counters, frozen_history=group.frozen_history)
            self.assertEqual(instance.step(self.action().to_json()).novelty_reward, .05)
            group.register(instance.state)
        self.assertEqual(counters.novelty_events, 4)
        self.assertEqual(len(group.merge()) - 1, 1)

    def test_mapping_uses_geometry_not_site_id_and_detects_vacancy_crossing(self):
        original = env.ideal_atoms(self.occupation)
        original.set_array("site_id", np.arange(len(original)))
        mapped = env.map_terminal(original)
        action = self.action(op="move_vacancy")
        moved = env.apply_action(self.occupation, action)
        terminal = env.ideal_atoms(moved)
        terminal.set_array("site_id", np.full(len(terminal), -999))
        terminal.positions += self.parent.cell.array[0]
        result = env.map_terminal(terminal)
        self.assertEqual(result.occupancy, tuple(moved))
        self.assertNotEqual(result.occupancy, mapped.occupancy)
        self.assertEqual(len(set(result.assigned_site_ids)), 316)
        self.assertLess(max(result.assignment_distances), 1e-10)
        far = original.copy()
        cat = int(np.flatnonzero(np.isin(far.numbers, [39, 40]))[0])
        far.positions[cat] += [0, 0, 1.82]
        with self.assertRaises(env.MappingError): env.map_terminal(far)

    def test_assignment_ambiguity_distinguishes_occupation_from_same_species(self):
        flat = np.array([[1., 1.], [1., 1.]])
        self.assertTrue(env.assignment_occupation_ambiguous(flat, [39, 40], [0, 1]))
        self.assertFalse(env.assignment_occupation_ambiguous(flat, [40, 40], [0, 1]))
        self.assertTrue(env.assignment_occupation_ambiguous(np.array([[1., 1.]]), [8], [0]))
        self.assertFalse(env.assignment_occupation_ambiguous(np.array([[1., 1. + 2e-8]]), [8], [0]))
        from scipy.optimize import linear_sum_assignment
        rng = np.random.default_rng(4)
        for rows, columns in [(3, 3), (3, 4)]:
            for _ in range(20):
                cost = rng.integers(0, 4, (rows, columns)).astype(float)
                species = [39, 40, 40] if rows == columns else [8, 8, 8]
                row, col = linear_sum_assignment(cost)
                optimal = cost[row, col].sum()
                occupations = set()
                for candidate in itertools.permutations(range(columns), rows):
                    if cost[np.arange(rows), candidate].sum() == optimal:
                        occupancy = [0] * columns
                        for i, j in enumerate(candidate): occupancy[j] = species[i]
                        occupations.add(tuple(occupancy))
                self.assertEqual(env.assignment_occupation_ambiguous(cost, species, col), len(occupations) > 1)

    def test_d0_helpers_are_deterministic_and_prompt_retains_all_occupancy(self):
        one = env.sample_legal_actions(self.occupation, 12, seed=77)
        two = env.sample_legal_actions(self.occupation, 12, seed=77)
        self.assertEqual(one, two)
        self.assertEqual(len(set(one)), 12)
        for action in one: env.apply_action(self.occupation, action)
        instance = self.environment(lambda *args: self.fail("Prompt encoding called physics"))
        messages = env.encode_prompt_v2(instance.state)
        observation = json.loads(messages[-1]["content"])
        self.assertEqual(observation["protocol"], "YSZ08_terminal_local2_v1")
        self.assertEqual(env.decode_occupancy(observation["occupancy"]), self.occupation)
        self.assertEqual(len(observation["occupancy"]["Y_sites"]), 16)
        self.assertEqual(len(observation["occupancy"]["vacancy_sites"]), 8)
        basis = np.asarray(observation["parent"]["basis_quarter_a"])
        for site in range(324):
            block, local = divmod(site, 12)
            block_xyz = [block // 9, (block // 3) % 3, block % 3]
            position = 1.285 * (4 * np.asarray(block_xyz) + basis[local])
            np.testing.assert_allclose(position, self.parent.positions[site], atol=1e-12)
        dense = json.loads(env.encode_prompt_v2(instance.state, dense_occupancy=True)[-1]["content"])
        self.assertEqual(dense["occupancy"], list(self.occupation))
        for bad in [{"Y_sites": [True] * 16, "vacancy_sites": [4] * 8},
                    {"Y_sites": [0] * 16, "vacancy_sites": [4] * 8}]:
            with self.assertRaises(ValueError): env.decode_occupancy(bad)

    def test_shared_counters_keep_the_eight_attempt_cap_per_episode(self):
        counters = env.Counters()
        limits = env.BudgetLimits(max_attempts=1, max_raw_ef=2)
        def callback(request, counter):
            counter.before_forward(1)
            return self.result(request.occupancy)
        one = self.environment(callback, counters=counters, limits=limits)
        two = self.environment(callback, counters=counters, limits=limits)
        self.assertTrue(one.step(self.action().to_json()).success)
        self.assertTrue(two.step(self.action().to_json()).success)
        with self.assertRaises(env.BudgetExceeded): one.step(self.action().to_json())
        self.assertEqual((counters.attempts, counters.raw_ef), (2, 2))
        self.assertEqual(one.state.steps_taken, 1)

    def test_attempt_token_and_wall_limits_fail_before_forwards(self):
        def forbidden(*args): self.fail("Exhausted budget reached physics")
        instance = self.environment(forbidden, limits=env.BudgetLimits(max_generated_tokens=1))
        with self.assertRaises(env.BudgetExceeded): instance.step(self.action().to_json(), generated_tokens=2)
        self.assertEqual(instance.counters.attempts, 0)
        now = [10.]
        instance = self.environment(forbidden, limits=env.BudgetLimits(wall_seconds=2), clock=lambda: now[0])
        now[0] = 12.
        with self.assertRaises(env.BudgetExceeded): instance.step(self.action().to_json())
        self.assertEqual(instance.counters.relax_actual, 0)

    def test_reached_physics_cap_stops_before_a_second_callback(self):
        for limits in [env.BudgetLimits(max_raw_ef=1), env.BudgetLimits(max_relax_actual=1)]:
            calls = []
            def callback(request, counter):
                calls.append(request.key)
                counter.before_forward(1)
                return self.result(request.occupancy)
            instance = self.environment(callback, limits=limits)
            instance.step(self.action().to_json())
            with self.assertRaises(env.BudgetExceeded): instance.step(self.action().to_json())
            self.assertEqual(len(calls), 1)
            self.assertEqual(instance.counters.attempts, 1)

    def test_mid_callback_budget_exception_stops_with_cost_retained(self):
        def callback(request, counter):
            counter.before_forward(1)
            counter.before_forward(2)
            self.fail("Over-budget forward was permitted")
        instance = self.environment(callback, limits=env.BudgetLimits(max_raw_ef=2))
        with self.assertRaises(env.BudgetExceeded): instance.step(self.action().to_json())
        self.assertEqual((instance.counters.attempts, instance.counters.relax_actual, instance.counters.raw_ef), (1, 1, 1))
        self.assertEqual(instance.state.energy, 100)

    def test_protocol_attempt_limit_cannot_be_disabled_or_extended(self):
        for limit in [None, 9, True, 1.0]:
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                env.BudgetLimits(max_attempts=limit)

    def test_protocol_relaxation_dtype_is_exactly_float64(self):
        self.assertEqual(env.RelaxationSpec().dtype, "float64")
        self.assertEqual(env.RelaxationSpec(dtype="float64").dtype, "float64")
        for dtype in ["float32", "float16", "bf16", "FLOAT64", "double", "", True, False, None, 64]:
            with self.subTest(dtype=dtype), self.assertRaises(ValueError):
                env.RelaxationSpec(dtype=dtype)

    def test_prompt_reports_actual_remaining_shared_wall_time(self):
        now = [20.]
        instance = self.environment(lambda *args: self.fail("Encoding called physics"),
                                    limits=env.BudgetLimits(wall_seconds=30),
                                    started_at=10., clock=lambda: now[0])
        first = json.loads(env.encode_prompt_v2(instance.state)[-1]["content"])
        self.assertEqual(first.get("remaining_wall_seconds"), 20.)
        now[0] = 39.
        second = json.loads(env.encode_prompt_v2(instance.state)[-1]["content"])
        self.assertEqual(second.get("remaining_wall_seconds"), 1.)
        self.assertNotEqual(first, second)
        later = self.environment(lambda *args: self.fail("Encoding called physics"),
                                 limits=env.BudgetLimits(wall_seconds=30),
                                 started_at=10., clock=lambda: now[0])
        self.assertEqual(json.loads(env.encode_prompt_v2(later.state)[-1]["content"])["remaining_wall_seconds"], 1.)
        now[0] = 41.
        expired = json.loads(env.encode_prompt_v2(instance.state)[-1]["content"])
        self.assertEqual(expired["remaining_wall_seconds"], 0.)
        self.assertTrue(np.isfinite(expired["remaining_wall_seconds"]))
        unlimited = self.environment(lambda *args: self.fail("Encoding called physics"))
        observation = json.loads(env.encode_prompt_v2(unlimited.state)[-1]["content"])
        self.assertIn("remaining_wall_seconds", observation)
        self.assertIsNone(observation["remaining_wall_seconds"])
        for nonfinite in [np.nan, np.inf]:
            now[0] = nonfinite
            with self.assertRaises(ValueError): env.encode_prompt_v2(instance.state)

    def test_protocol_group_size_is_exactly_integer_four(self):
        for size in [None, 0, 1, 2, 3, 5, True, False, 4., "4"]:
            with self.subTest(size=size), self.assertRaises(ValueError):
                env.NoveltyGroup(set(), group_size=size)
        self.assertEqual(env.NoveltyGroup(set(), group_size=4).group_size, 4)

    def test_group_rejects_mixed_arm_seed_histories_and_early_merge(self):
        group = env.NoveltyGroup(set(), group_size=4)
        with self.assertRaises(ValueError): group.merge()
        one = self.environment(lambda request, counter: self.result(), frozen_history=group.frozen_history)
        two = env.Environment(self.result(), lambda request, counter: self.result(), arm="different", seed=31,
                              frozen_history=group.frozen_history)
        group.register(one.state)
        with self.assertRaises(ValueError): group.register(two.state)


if __name__ == "__main__":
    unittest.main()
