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
from publish_snapshot import clean, compact_result
import publish_snapshot
from results import load, paired


class ExportIntegrity(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
