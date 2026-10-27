"""Account only the independent smoke pool from a saved parent-provided sacct TSV."""
import argparse
import csv
import json
import re
from pathlib import Path
from smoke_common import SMOKE, save_json

ACTIVE = {"PENDING", "RUNNING", "CONFIGURING", "COMPLETING", "SUSPENDED", "REQUEUED", "RESIZING"}


def gpu_count(tres, fallback):
    fields = dict(field.split("=", 1) for field in tres.split(",") if "=" in field)
    if "gres/gpu" in fields:
        return int(fields["gres/gpu"])
    specific = [int(value) for name, value in fields.items() if name.startswith("gres/gpu:")]
    return max(specific) if specific else fallback


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-home", type=Path, default=SMOKE)
    parser.add_argument("--sacct-tsv", type=Path, required=True,
        help="sacct -X -n -P --format=JobIDRaw,State,ElapsedRaw,AllocTRES,ExitCode (parent executes query)")
    args = parser.parse_args()
    home = args.smoke_home
    budget = json.loads((home / "BUDGET.json").read_text())
    submissions_path = home / "SUBMISSIONS.json"
    submissions = json.loads(submissions_path.read_text()) if submissions_path.exists() else {"jobs": []}
    submitted = {row["id"]: row for row in submissions["jobs"]}
    if len(submitted) != len(submissions["jobs"]):
        raise ValueError("Duplicate smoke IDs would hide billed runs; record each approved run separately")
    records = {}
    for row in csv.reader(args.sacct_tsv.read_text().splitlines(), delimiter="|"):
        if len(row) >= 5 and re.fullmatch(r"\d+", row[0]):
            records[row[0]] = dict(job_id=row[0], state=row[1].split()[0].rstrip("+"),
                                   elapsed_seconds=int(row[2]), alloc_tres=row[3], exit_code=row[4])
    actual = 0.0
    remaining_caps = 0.0
    unknown = []
    for job in budget["jobs"]:
        source = submitted.get(job["id"])
        if source is None:
            job.update(state="prepared", slurm_job_id=None, actual_card_hours=0.0)
            remaining_caps += job["hard_gpu_hours"]
            continue
        job_id = str(source["slurm_job_id"])
        job["slurm_job_id"] = job_id
        row = records.get(job_id)
        if row is None:
            job["state"] = "ACCOUNTING_UNKNOWN"
            unknown.append(job_id)
            # Preserve known usage plus the remaining reservation; never infer free budget.
            known_cost = job.get("actual_card_hours", 0.0)
            actual += known_cost
            remaining_caps += max(0.0, job["hard_gpu_hours"] - known_cost)
            continue
        gpus = gpu_count(row["alloc_tres"], job["gpus"])
        cost = gpus * row["elapsed_seconds"] / 3600
        job.update(state=row["state"], actual_card_hours=cost, sacct=row,
                   hard_cap_exceeded=cost > job["hard_gpu_hours"] + 1e-6,
                   allocation_gpu_count=gpus, unexpected_gpu_allocation=gpus != job["gpus"])
        actual += cost
        if row["state"] in ACTIVE:
            remaining_caps += max(0.0, job["hard_gpu_hours"] - cost)
    budget.update(actual_card_hours=actual, actual_gpu_hours=actual,
                  actual_plus_remaining_caps=actual + remaining_caps,
                  accounting_unknown_job_ids=unknown,
                  unallocated_gpu_hours=budget["authorized_card_hours"] - actual - remaining_caps,
                  actual_status="sacct reconciled" if not unknown else "incomplete sacct; reservations retained",
                  pool_cap_exceeded=actual + remaining_caps > budget["authorized_card_hours"] + 1e-6,
                  last_sacct_source=str(args.sacct_tsv))
    save_json(home / "BUDGET.json", budget)
    print(json.dumps({key: budget[key] for key in ["actual_card_hours", "actual_plus_remaining_caps", "unallocated_gpu_hours", "actual_status"]}))


if __name__ == "__main__":
    main()
