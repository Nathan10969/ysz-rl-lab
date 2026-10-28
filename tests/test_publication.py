"""Meaningful offline checks for export integrity and paired result interpretation."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from publish_snapshot import clean, compact_result, budget_summary
import publish_snapshot
import results
from results import load, paired


class ExportIntegrity(unittest.TestCase):
    def test_menu_diagnostics_keep_metrics_without_private_model_inputs(self):
        item = compact_result('grid_readout', {
            'schema': 'formal12-imitate-v1', 'status': 'COMPLETED',
            'data': {'test_rows': 208, 'target_histogram': [0] * 16},
            'eval': {'candidate': {'test': {'n': 208, 'captured_fraction': .4,
                'optimal_mass': .24, 'chance_mass': .087, 'argmax_accuracy': .27,
                'messages': 'private prompts'}}},
            'inputs': {'model': 'private model location'}})
        self.assertEqual(item['menu_preference']['candidate']['test']['captured_fraction'], .4)
        self.assertEqual(item['diagnostic_data']['test_rows'], 208)
        self.assertIsNone(item['mean_best_of_4_drop_eV'])
        self.assertNotIn('private', json.dumps(item))

    def test_inconsistent_ledger_is_flagged_without_overwriting_reported_values(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp)
            (source / 'training_smoke').mkdir()
            (source / 'training_smoke/FORMAL_TRAINING_POOL.json').write_text(json.dumps({
                'actual_card_hours': 3., 'reserved_card_hours': 2., 'actual_plus_reserved': 7.}))
            budgets, _ = budget_summary(source)
            item = budgets['FORMAL_TRAINING_POOL.json']
            self.assertEqual(item['actual_plus_reserved'], 7.)
            self.assertEqual(item['header_sum_card_hours'], 5.)
            self.assertFalse(item['header_totals_consistent'])

    def test_timestamp_only_refresh_is_noop_and_secret_failure_preserves_public_files(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            source, target = base / "source", base / "public"
            (source / "training_smoke").mkdir(parents=True)
            (target / "tools").mkdir(parents=True)
            (source / "PROJECT_RECORD_RSI_YSZ.md").write_text("Recorded work\n")
            config = {"files": [{"source": "PROJECT_RECORD_RSI_YSZ.md", "destination": "PROJECT_RECORD_RSI_YSZ.md"}]}
            (target / "tools/publication_sources.json").write_text(json.dumps(config))
            budget = source / "training_smoke/FORMAL_TRAINING_POOL.json"
            budget.write_text(json.dumps({"actual_card_hours": 2., "updated_at_utc": "first check"}))
            with patch.object(publish_snapshot, "ROOT", target):
                self.assertTrue(publish_snapshot.build(source, "2026-10-26"))
                manifest = (target / "provenance/manifest.json").read_bytes()
                budget.write_text(json.dumps({"actual_card_hours": 2., "updated_at_utc": "second check"}))
                self.assertFalse(publish_snapshot.build(source, "2026-10-27"))
                self.assertEqual((target / "provenance/manifest.json").read_bytes(), manifest)
                (source / "PROJECT_RECORD_RSI_YSZ.md").write_text("ghp_" + "b" * 36)
                with self.assertRaisesRegex(ValueError, "Credential-like"):
                    publish_snapshot.build(source, "2026-10-27")
                self.assertEqual((target / "provenance/manifest.json").read_bytes(), manifest)

    def test_group_metric_is_best_of_four_before_mean(self):
        result = compact_result("example", {"status": "COMPLETED", "groups": [
            {"group": 0, "start_energy_eV": 10., "best_energy_eV": [9., 7., 8., 9.]},
            {"group": 1, "start_energy_eV": 20., "best_energy_eV": [19., 18., 19., 19.]}]})
        self.assertEqual([row["drop_eV"] for row in result["per_start"]], [3., 2.])
        self.assertEqual(result["mean_best_of_4_drop_eV"], 2.5)

    def test_search_chains_preserve_start_identity(self):
        result = compact_result("example", {"chains": [
            {"start": 1, "initial_energy_eV": 20., "best_energy_eV": 17.},
            {"start": 0, "initial_energy_eV": 10., "best_energy_eV": 8.},
            {"start": 1, "initial_energy_eV": 20., "best_energy_eV": 16.}]})
        self.assertEqual([row["start_index"] for row in result["per_start"]], [0, 1])
        self.assertEqual([row["drop_eV"] for row in result["per_start"]], [2., 4.])

    def test_secret_pattern_stops_export_without_echoing_value(self):
        with self.assertRaisesRegex(ValueError, "Credential-like"):
            clean("ghp_" + "a" * 36)

    def test_source_and_published_files_have_real_hashes(self):
        manifest = json.loads((ROOT / "provenance/manifest.json").read_text())
        for item in manifest["files"]:
            with self.subTest(file=item["published"]):
                self.assertEqual(len(item["source_sha256"]), 64)
                path = ROOT / item["published"]
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), item["published_sha256"])

    def test_saved_scores_equal_per_start_mean(self):
        for path in (ROOT / "experiments/results").glob("*.json"):
            if path.stem == "index":
                continue
            item = json.loads(path.read_text())
            if item["per_start"]:
                self.assertAlmostEqual(item["mean_best_of_4_drop_eV"],
                    sum(row["initial_energy_eV"] - row["best_energy_eV"] for row in item["per_start"]) / len(item["per_start"]), places=10)


class PairedEvidence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.candidate = load("meta_s6_greedy")
        cls.reference = load("meta_s6_menurandom")

    def test_reordering_starts_does_not_change_comparison(self):
        reversed_reference = copy.deepcopy(self.reference)
        reversed_reference["per_start"].reverse()
        normal = paired(self.candidate, self.reference)
        reordered = paired(self.candidate, reversed_reference)
        self.assertEqual(normal, reordered)
        self.assertEqual(normal["n"], 24)

    def test_different_initial_geometry_energy_is_not_silently_paired(self):
        wrong = copy.deepcopy(self.reference)
        wrong["per_start"][0]["initial_energy_eV"] += 1.
        with self.assertRaisesRegex(ValueError, "Initial energies differ"):
            paired(self.candidate, wrong)

    def test_failure_and_different_pools_are_not_treated_as_finished_evidence(self):
        failed = copy.deepcopy(self.reference)
        failed["status"] = "FAILED"
        with self.assertRaisesRegex(ValueError, "complete jobs"):
            paired(self.candidate, failed)
        other_pool = copy.deepcopy(self.reference)
        other_pool["config"]["pool"] = "another_pool"
        with self.assertRaisesRegex(ValueError, "Different start pools"):
            paired(self.candidate, other_pool)


class MenuGridEvidence(unittest.TestCase):
    def test_grid_selection_replays_the_recorded_rule_without_calling_it_a_significance_test(self):
        record = {'status': 'COMPLETED', 'diagnostic_data': {'test_rows': 208},
                  'menu_preference': {name: {'test': {'n': 208, 'captured_fraction': c}}
                     for name, c in [('step_a', .02), ('grid_lr2em5_u2', .03),
                                     ('grid_lr1em4_u1', .4)]}}
        with patch.object(results, 'load', return_value=record):
            summary = results.menu_grid_summary()
        self.assertEqual(summary['selected_adapter'], 'grid_lr1em4_u1')
        self.assertAlmostEqual(summary['observed_repeat_difference'], .01)
        self.assertAlmostEqual(summary['exploratory_threshold'], .05)
        self.assertTrue(summary['exploratory_rule_pass'])
        self.assertEqual(summary['data_role'], 'development')
        self.assertFalse(summary['formal_significance_test'])


if __name__ == "__main__":
    unittest.main()
