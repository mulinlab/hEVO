#!/usr/bin/env python3
"""Roll GPN-Star ancestry array batches under the cluster job-record ceiling."""

from __future__ import annotations

import getpass
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path


SLURM_USER = os.environ.get("HEVO_SLURM_USER") or getpass.getuser()

MODELS = ("GPN-Star-P", "GPN-Star-V", "GPN-Star-M")
TOTAL_SAMPLES = 1500
EXPECTED_ROWS_PER_SAMPLE = 10000
EXPECTED_EMBEDDING_WIDTH = 3076
CHUNK_SIZE = 200
ARRAY_PART_SIZE = 100
POLL_SECONDS = 60
MIN_JOB_AGE_SECONDS = 310
TASK_TIME = "04:00:00"
MAX_EXEC_FORMAT_RETRIES = 2
MAX_SUBMIT_RETRIES = 5
MAX_SUBMIT_BACKOFF_SECONDS = 900
GPU_BUDGET = 40
M_GPU_BUDGET = 26
P_GPU_BUDGET = 6
PERMANENT_SBATCH_MARKERS = (
    "unable to open file",
    "invalid account",
    "invalid partition",
    "invalid qos",
    "invalid generic resource",
    "permission denied",
    "no such file or directory",
    "job violates accounting/qos policy",
)
TRANSIENT_SBATCH_MARKERS = (
    "slurmctld is not responding",
    "slurm controller is not responding",
    "unable to contact slurm controller",
    "unable to establish communication with slurm controller",
    "socket timed out",
    "connection timed out",
    "connection refused",
    "network is unreachable",
    "temporary failure in name resolution",
    "resource temporarily unavailable",
    "try again later",
)


def run(
    command: list[str], *, capture: bool = True
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        check=False,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.STDOUT if capture else None,
    )


def record(log_path: Path, message: str) -> None:
    text = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(text, flush=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(text + "\n")


def save_state(path: Path, state: dict) -> None:
    temp = path.with_suffix(".json.partial")
    temp.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, path)


def save_submission_event(root: Path, event: dict) -> None:
    path = root / "config/submission_events.json"
    events = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    events.append(event)
    temp = path.with_suffix(".json.partial")
    temp.write_text(json.dumps(events, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, path)


def classify_sbatch_error(output: str) -> str:
    lowered = output.lower()
    if any(marker in lowered for marker in PERMANENT_SBATCH_MARKERS):
        return "permanent"
    if any(marker in lowered for marker in TRANSIENT_SBATCH_MARKERS):
        return "transient_controller"
    return "permanent_or_unclassified"


def input_metadata(path: Path) -> dict:
    """Read the immutable identity/hash needed to validate one array output."""
    digest = hashlib.sha256()
    row_count = 0
    sample_id = population = None
    with path.open("rb") as handle:
        for line in handle:
            digest.update(line)
            if not line.strip():
                continue
            row_count += 1
            if sample_id is None:
                first = json.loads(line)
                sample_id = str(first["sample"])
                population = str(first["sample_popu"])
    if sample_id is None:
        raise RuntimeError(f"Empty ancestry input: {path}")
    if row_count != EXPECTED_ROWS_PER_SAMPLE:
        raise RuntimeError(
            f"{path}: expected {EXPECTED_ROWS_PER_SAMPLE} input rows, found {row_count}"
        )
    return {
        "input_file": str(path.resolve()),
        "sample_id": sample_id,
        "population": population,
        "input_rows": row_count,
        "input_sha256": digest.hexdigest(),
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def active_jobs(job_ids: list[str]) -> bool:
    result = run(
        ["squeue", "--noheader", "--jobs", ",".join(job_ids), "--format", "%T"]
    )
    if result.returncode != 0:
        raise RuntimeError(f"squeue failed: {result.stdout}")
    return bool(result.stdout.strip())


def retry_records(root: Path) -> list[dict]:
    path = root / "config/retries.json"
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def add_retry_record(root: Path, entry: dict) -> None:
    path = root / "config/retries.json"
    rows = retry_records(root)
    rows.append(entry)
    temp = path.with_suffix(".json.partial")
    temp.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, path)


def locate_exec_format_failure(
    root: Path, failed_job_id: str, model: str, array_jobs: dict
) -> tuple[int, str, int] | None:
    """Map a Slurm exec-format failure back to its sample index."""
    history = retry_records(root)
    retry_parent_to_sample = {
        str(row.get("retry_job_id")): int(row["sample_index"])
        for row in history
        if row.get("model") == model and row.get("retry_job_id") is not None
    }
    for log_path in (root / "logs").glob("array_*_*.out"):
        match = re.fullmatch(r"array_(\d+)_(\d+)\.out", log_path.name)
        if not match:
            continue
        try:
            contents = log_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if (
            f"job{failed_job_id}/slurm_script" not in contents
            or "Exec format error" not in contents
        ):
            continue
        parent_id, task_id = match.group(1), int(match.group(2))
        if parent_id in retry_parent_to_sample:
            return retry_parent_to_sample[parent_id], parent_id, task_id
        for model_state in array_jobs.get(model, {}).get("arrays", []):
            if str(model_state.get("job_id")) == parent_id:
                sample_index = int(model_state["start"]) + task_id
                return sample_index, parent_id, task_id
    return None


def parse_array_job_identity(job_id: str) -> tuple[str, int | None] | None:
    """Split Slurm's array job/task form, ignoring optional step suffixes."""
    normalized = job_id.split(".", 1)[0]
    match = re.fullmatch(r"(\d+)(?:_(\d+))?", normalized)
    if not match:
        return None
    task_id = int(match.group(2)) if match.group(2) is not None else None
    return match.group(1), task_id


def expected_array_identity(
    model: str,
    sample_index: int,
    job_ids: list[str],
    array_jobs: dict,
    retry_history: list[dict],
) -> tuple[str, int] | None:
    """Find the submitted array parent/task that owns this input row."""
    for retry in reversed(retry_history):
        if (
            retry.get("model") == model
            and int(retry.get("sample_index", -1)) == sample_index
            and str(retry.get("retry_job_id")) in job_ids
        ):
            parsed = parse_array_job_identity(str(retry["retry_job_id"]))
            if parsed is not None:
                parent_id, task_id = parsed
                return parent_id, 0 if task_id is None else task_id

    matches = []
    for entry in array_jobs.get(model, {}).get("arrays", []):
        parent_id = str(entry.get("job_id", ""))
        if parent_id not in job_ids:
            continue
        start = int(entry["start"])
        count = int(entry["count"])
        if start <= sample_index < start + count:
            matches.append((parent_id, sample_index - start))
    if len(matches) != 1:
        return None
    return matches[0]


def check_finished(
    root: Path,
    model: str,
    job_ids: list[str],
    expected_artifacts: list[tuple[Path, Path, dict]],
    log_path: Path,
    batch_start: int,
    *,
    max_submit_retries: int,
    poll_seconds: int,
) -> bool:
    array_jobs = json.loads(
        (root / "config/array_jobs.json").read_text(encoding="utf-8")
    )
    retry_history = retry_records(root)
    # Include any separately submitted recovery task for this sample range so
    # a coordinator restart cannot overlook a retry that is still in flight.
    for row in retry_history:
        index = int(row.get("sample_index", -1))
        retry_id = row.get("retry_job_id")
        if (
            row.get("model") == model
            and batch_start <= index < batch_start + len(expected_artifacts)
            and retry_id
        ):
            if str(retry_id) not in job_ids:
                job_ids.append(str(retry_id))
    if active_jobs(job_ids):
        return False
    result = run(
        [
            "sacct",
            "--noheader",
            "--parsable2",
            "--allocations",
            "--jobs",
            ",".join(job_ids),
            "--format=JobIDRaw,State,ExitCode",
        ]
    )
    if result.returncode != 0:
        raise RuntimeError(f"sacct failed: {result.stdout}")
    rows = [line.split("|") for line in result.stdout.splitlines() if "|" in line]
    if not rows:
        return False
    states = [(row[0], row[1].split()[0], row[2]) for row in rows if len(row) >= 3]
    failures = [row for row in states if row[1] != "COMPLETED" or row[2] != "0:0"]
    import numpy as np

    missing_or_invalid: list[tuple[int, Path]] = []
    for offset, (embedding_path, manifest_path, expected) in enumerate(
        expected_artifacts
    ):
        expected_job = expected_array_identity(
            model,
            int(expected["sample_index"]),
            job_ids,
            array_jobs,
            retry_history,
        )
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            embedding = np.load(embedding_path, mmap_mode="r")
            model_path = payload.get("model_path")
            msa_stores = payload.get("msa_stores")
            msa_species = payload.get("msa_species")
            has_array_identity = (
                payload.get("slurm_array_job_id") is not None
                or payload.get("slurm_array_task_id") is not None
            )
            array_identity_valid = (
                (
                    payload.get("sample_index") == expected["sample_index"]
                    and expected_job is not None
                    and payload.get("slurm_array_job_id") == expected_job[0]
                    and payload.get("slurm_array_task_id") == expected_job[1]
                )
                if has_array_identity
                else payload.get("input_file") == expected["input_file"]
            )
            valid = (
                payload.get("status") == "complete"
                and payload.get("model") == model
                and isinstance(model_path, str)
                and Path(model_path).is_absolute()
                and payload.get("input_file") == expected["input_file"]
                and payload.get("input_sha256") == expected["input_sha256"]
                and payload.get("sample_id") == expected["sample_id"]
                and payload.get("population") == expected["population"]
                and array_identity_valid
                and payload.get("input_rows") == expected["input_rows"]
                and payload.get("input_rows") == EXPECTED_ROWS_PER_SAMPLE
                and list(embedding.shape) == [EXPECTED_EMBEDDING_WIDTH]
                and bool(np.isfinite(embedding).all())
                and payload.get("embedding_shape") == [EXPECTED_EMBEDDING_WIDTH]
                and isinstance(msa_stores, list)
                and bool(msa_stores)
                and all(
                    isinstance(path, str) and Path(path).is_absolute()
                    for path in msa_stores
                )
                and isinstance(msa_species, list)
                and len(msa_species) == len(msa_stores)
                and all(isinstance(count, int) and count > 0 for count in msa_species)
                and (
                    payload.get("output_sha256") is None
                    or payload.get("output_sha256") == sha256_file(embedding_path)
                )
            )
        except Exception:
            valid = False
        if not valid:
            missing_or_invalid.append((batch_start + offset, embedding_path))

    if missing_or_invalid:
        failure_rows = [row for row in failures if row[2] == "8:0"]
        failed_samples: dict[int, tuple[str, int]] = {}
        for failed_job_id, _, _ in failure_rows:
            location = locate_exec_format_failure(
                root, failed_job_id, model, array_jobs
            )
            if location:
                sample_index, parent_id, task_id = location
                failed_samples[sample_index] = (parent_id, task_id)
        retryable = {index for index, _ in missing_or_invalid} & set(failed_samples)
        if retryable != {index for index, _ in missing_or_invalid}:
            raise RuntimeError(
                f"Array batch has failed/incomplete artifacts that are not confirmed "
                f"Slurm exec-format errors: failures={failures[:20]}, "
                f"missing/invalid={[str(path) for _, path in missing_or_invalid[:20]]}"
            )

        # Retry one sample at a time so transient failures can never push GPU
        # use above the configured model allocation.
        sample_index, embedding_path = missing_or_invalid[0]
        attempts = [
            row
            for row in retry_records(root)
            if row.get("model") == model
            and int(row.get("sample_index", -1)) == sample_index
        ]
        if len(attempts) >= MAX_EXEC_FORMAT_RETRIES:
            raise RuntimeError(
                f"Sample {sample_index} ({embedding_path.stem}) still lacks a valid "
                f"embedding after {len(attempts)} exec-format retries"
            )
        parent_id, task_id = failed_samples[sample_index]
        gpus = 2 if model == "GPN-Star-M" else 1
        retry_id = submit_part(
            root,
            model,
            sample_index,
            1,
            1,
            gpus=gpus,
            max_submit_retries=max_submit_retries,
            poll_seconds=poll_seconds,
        )
        job_ids.append(retry_id)
        entry = {
            "failed_array_parent": parent_id,
            "failed_array_task_index": task_id,
            "model": model,
            "sample_index": sample_index,
            "sample_id": embedding_path.stem,
            "failure": "slurmstepd Exec format error before Python extraction started",
            "retry_job_id": retry_id,
            "attempt": len(attempts) + 1,
        }
        add_retry_record(root, entry)
        record(
            log_path,
            f"auto-retrying {model} sample index {sample_index} ({embedding_path.stem}) "
            f"after Slurm exec-format failure {parent_id}_{task_id}; retry job {retry_id}",
        )
        return False

    if failures:
        record(
            log_path,
            f"array accounting reported failed task(s) {failures[:20]}, "
            "while all expected sample embeddings/manifests match the submitted task identities",
        )
    return True


def submit_part(
    root: Path,
    model: str,
    start: int,
    count: int,
    throttle: int,
    *,
    gpus: int,
    max_submit_retries: int = MAX_SUBMIT_RETRIES,
    poll_seconds: int = POLL_SECONDS,
    tranche_state: str = "submission_pending",
    prior_job_ids: list[str] | None = None,
) -> str:
    end = start + count - 1
    script = root / "arrays/gpn_star_ancestry_array.sbatch"
    name = {
        "GPN-Star-P": "GPNStarP_anc",
        "GPN-Star-V": "GPNStarV_anc",
        "GPN-Star-M": "GPNStarM_anc",
    }[model]
    command = [
        "sbatch",
        "--parsable",
        f"--job-name={name}_{start}",
        f"--array=0-{count - 1}%{throttle}",
        f"--export=ALL,MODEL_NAME={model},BATCH_SIZE=16,INDEX_OFFSET={start}",
        f"--time={TASK_TIME}",
    ]
    if gpus == 2:
        command.append("--gres=gpu:2")
    command.append(str(script))

    for attempt in range(1, max_submit_retries + 2):
        try:
            result = run(command)
        except OSError as exc:
            output = f"{type(exc).__name__}: {exc}"
            event = {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "status": "failed",
                "classification": "permanent_or_unclassified",
                "model": model,
                "tranche_start": start,
                "tranche_count": count,
                "tranche_state": tranche_state,
                "prior_submitted_job_ids": prior_job_ids or [],
                "attempt": attempt,
                "max_submit_retries": max_submit_retries,
                "command": command,
                "exit_code": None,
                "output": output,
            }
            save_submission_event(root, event)
            record(root / "logs/roll_submitter.log", f"sbatch launch failed: {output}")
            raise RuntimeError(
                f"Could not launch sbatch for {model} rows [{start},{start + count}): {output}"
            ) from exc
        if result.returncode == 0:
            response_lines = (result.stdout or "").strip().split(";")[0].splitlines()
            job_id = response_lines[-1].strip() if response_lines else ""
            if not job_id.isdigit():
                event = {
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    "status": "failed",
                    "classification": "invalid_success_response",
                    "model": model,
                    "tranche_start": start,
                    "tranche_count": count,
                    "tranche_state": tranche_state,
                    "prior_submitted_job_ids": prior_job_ids or [],
                    "attempt": attempt,
                    "max_submit_retries": max_submit_retries,
                    "command": command,
                    "exit_code": result.returncode,
                    "output": result.stdout or "",
                }
                save_submission_event(root, event)
                raise RuntimeError(f"Could not parse sbatch job ID: {result.stdout!r}")
            if attempt > 1:
                save_submission_event(
                    root,
                    {
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                        "status": "submitted_after_retry",
                        "classification": "success",
                        "model": model,
                        "tranche_start": start,
                        "tranche_count": count,
                        "tranche_state": tranche_state,
                        "prior_submitted_job_ids": prior_job_ids or [],
                        "attempt": attempt,
                        "max_submit_retries": max_submit_retries,
                        "command": command,
                        "exit_code": result.returncode,
                        "output": result.stdout or "",
                        "job_id": job_id,
                    },
                )
            return job_id
        record_path = root / "logs/roll_submitter.log"
        output = result.stdout or ""
        classification = classify_sbatch_error(output)
        will_retry = (
            classification == "transient_controller" and attempt <= max_submit_retries
        )
        event = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "status": "retrying" if will_retry else "failed",
            "classification": classification,
            "model": model,
            "tranche_start": start,
            "tranche_count": count,
            "tranche_state": tranche_state,
            "prior_submitted_job_ids": prior_job_ids or [],
            "attempt": attempt,
            "max_submit_retries": max_submit_retries,
            "command": command,
            "exit_code": result.returncode,
            "output": output,
        }
        save_submission_event(root, event)
        if not will_retry:
            record(
                record_path,
                f"sbatch submission failed for {model} rows [{start},{end}] "
                f"({classification}, exit={result.returncode}): {output.strip()}",
            )
            raise RuntimeError(
                f"sbatch failed for {model} rows [{start},{end}] after {attempt} attempt(s); "
                f"details saved to {root / 'config/submission_events.json'}: {output.strip()}"
            )
        delay = min(poll_seconds * (2 ** (attempt - 1)), MAX_SUBMIT_BACKOFF_SECONDS)
        record(
            record_path,
            f"transient sbatch failure for {model} rows [{start},{end}] "
            f"(attempt {attempt}/{max_submit_retries + 1}, exit={result.returncode}); "
            f"retrying in {delay}s: {output.strip()}",
        )
        time.sleep(delay)
    raise AssertionError("bounded sbatch retry loop exited without a result")


def wait_until_finished(
    root: Path,
    model: str,
    job_ids: list[str],
    expected_artifacts: list[tuple[Path, Path, dict]],
    batch_start: int,
    *,
    max_submit_retries: int,
    poll_seconds: int,
) -> None:
    log_path = root / "logs/roll_submitter.log"
    record(log_path, f"waiting for {model} arrays {','.join(job_ids)}")
    while not check_finished(
        root,
        model,
        job_ids,
        expected_artifacts,
        log_path,
        batch_start,
        max_submit_retries=max_submit_retries,
        poll_seconds=poll_seconds,
    ):
        state = json.loads(
            (root / "config/array_jobs.json").read_text(encoding="utf-8")
        )
        maybe_raise_v_throttle(root, state)
        maybe_adjust_m_throttle(root, state)
        time.sleep(poll_seconds)
    record(log_path, f"{model} arrays completed successfully")
    # Slurm retains completed array task records for MinJobAge. Let them expire
    # before adding another tranche near the controller's MaxJobCount ceiling.
    time.sleep(MIN_JOB_AGE_SECONDS)


def update_model_state(
    state: dict, model: str, job_id: str, start: int, count: int, throttle: int
) -> None:
    model_state = state[model]
    model_state["arrays"].append(
        {
            "job_id": job_id,
            "start": start,
            "count": count,
            "throttle": throttle,
            "gpus_per_task": 2 if model == "GPN-Star-M" else 1,
        }
    )
    model_state["submitted_rows"] = max(model_state["submitted_rows"], start + count)


def running_gpu_budget() -> tuple[dict[str, int], int]:
    """Return running GPU allocations for this account, split by model family."""
    result = run(
        [
            "squeue",
            "--noheader",
            "--user",
            SLURM_USER,
            "--states=RUNNING",
            "--format=%j|%b",
        ]
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"squeue failed while planning GPU allocation: {result.stdout}"
        )
    model_gpu = {"GPN-Star-M": 0, "GPN-Star-P": 0, "GPN-Star-V": 0}
    other_gpu = 0
    for line in result.stdout.splitlines():
        fields = line.split("|", 1)
        if len(fields) != 2:
            continue
        name, tres = fields
        match = re.search(r"(?:^|,)gpu:(\d+)", tres)
        if not match:
            continue
        gpu_count = int(match.group(1))
        if name.startswith("GPNStarM"):
            model_gpu["GPN-Star-M"] += gpu_count
        elif name.startswith("GPNStarP"):
            model_gpu["GPN-Star-P"] += gpu_count
        elif name.startswith("GPNStarV"):
            model_gpu["GPN-Star-V"] += gpu_count
        else:
            other_gpu += gpu_count
    return model_gpu, other_gpu


def active_array_entries(state: dict, model: str) -> list[dict]:
    """Return submitted arrays that still have queued or running tasks."""
    active = []
    for array in state.get(model, {}).get("arrays", []):
        job_id = str(array.get("job_id", ""))
        if not job_id:
            continue
        result = run(
            ["squeue", "--array", "--noheader", "--jobs", job_id, "--format=%i"]
        )
        active_ids = [
            line.strip()
            for line in result.stdout.splitlines()
            if re.fullmatch(r"\d+(?:_\d+)?", line.strip())
        ]
        if result.returncode == 0 and active_ids:
            active.append(array)
    return active


def array_running_gpu_count(arrays: list[dict]) -> int:
    """Count GPUs currently held by tasks in the given arrays."""
    job_ids = [str(array["job_id"]) for array in arrays]
    if not job_ids:
        return 0
    result = run(
        [
            "squeue",
            "--noheader",
            "--user",
            SLURM_USER,
            "--states=RUNNING",
            "--jobs",
            ",".join(job_ids),
            "--format=%b",
        ]
    )
    if result.returncode != 0:
        return 0
    total = 0
    for line in result.stdout.splitlines():
        match = re.search(r"(?:^|,)gpu:(\d+)", line)
        if match:
            total += int(match.group(1))
    return total


def maybe_adjust_m_throttle(root: Path, state: dict) -> None:
    """Use GPUs released by P/V for active M arrays, while staying under 40."""
    arrays = active_array_entries(state, "GPN-Star-M")
    if not arrays:
        return

    model_gpu, other_gpu = running_gpu_budget()
    p_pending = (
        int(state.get("GPN-Star-P", {}).get("submitted_rows", 0)) < TOTAL_SAMPLES
    )
    p_budget = max(P_GPU_BUDGET if p_pending else 0, model_gpu["GPN-Star-P"])
    m_array_gpu = array_running_gpu_count(arrays)
    m_retry_gpu = max(0, model_gpu["GPN-Star-M"] - m_array_gpu)
    v_arrays = active_array_entries(state, "GPN-Star-V")
    v_array_budget = sum(int(array.get("throttle", 0)) for array in v_arrays)
    v_budget = max(model_gpu["GPN-Star-V"], v_array_budget)
    available = GPU_BUDGET - p_budget - v_budget - other_gpu - m_retry_gpu
    concurrency = min(
        sum(int(array.get("count", 0)) for array in arrays),
        max(0, available // 2),
    )
    if concurrency < 1:
        return

    remaining = concurrency
    targets = []
    for index, array in enumerate(arrays):
        arrays_left = len(arrays) - index
        target = min(
            int(array.get("count", 0)),
            GPU_BUDGET // 2,
            (remaining + arrays_left - 1) // arrays_left,
        )
        targets.append(target)
        remaining -= target

    changed = False
    log_path = root / "logs/roll_submitter.log"
    for array, target in zip(arrays, targets):
        if target < 1 or target == int(array.get("throttle", 0)):
            continue
        job_id = str(array["job_id"])
        updated = run(
            ["scontrol", "update", f"JobId={job_id}", f"ArrayTaskThrottle={target}"]
        )
        partial_success = updated.returncode != 0 and "No error" in updated.stdout
        if updated.returncode != 0 and not partial_success:
            record(
                log_path,
                f"could not adjust M array {job_id} throttle to {target}: {updated.stdout.strip()}",
            )
            continue
        array["throttle"] = target
        changed = True
        if partial_success:
            record(
                log_path,
                f"M array {job_id}: completed task ranges ignored; active ranges updated",
            )
        record(
            log_path,
            f"adjusted GPN-Star-M array {job_id} throttle to {target}; "
            f"reserved P/V/other/retry GPUs={p_budget}/{model_gpu['GPN-Star-V']}/"
            f"{other_gpu}/{m_retry_gpu}",
        )
    if changed:
        plan = state.setdefault("allocation_plan", {})
        current = plan.setdefault("current_max_array_gpu_tasks", {})
        current["GPN-Star-M"] = sum(targets)
        plan["active_array_gpu_max"] = GPU_BUDGET
        plan["max_total_gpu_with_reserved_jobs"] = GPU_BUDGET
        plan["total_gpu_max"] = GPU_BUDGET
        save_state(root / "config/array_jobs.json", state)


def plan_v_throttles(root: Path, count: int, poll_seconds: int) -> tuple[int, int]:
    """Allocate V array concurrency from the remaining 40-GPU budget."""
    while True:
        model_gpu, other_gpu = running_gpu_budget()
        state = json.loads(
            (root / "config/array_jobs.json").read_text(encoding="utf-8")
        )
        # Keep capacity for models with unsubmitted rows. Once P is complete,
        # release its reservation so V can use the full remaining allocation.
        m_pending = (
            int(state.get("GPN-Star-M", {}).get("submitted_rows", 0)) < TOTAL_SAMPLES
        )
        p_pending = (
            int(state.get("GPN-Star-P", {}).get("submitted_rows", 0)) < TOTAL_SAMPLES
        )
        m_budget = max(M_GPU_BUDGET if m_pending else 0, model_gpu["GPN-Star-M"])
        p_budget = max(P_GPU_BUDGET if p_pending else 0, model_gpu["GPN-Star-P"])
        v_retries = model_gpu["GPN-Star-V"]
        other_budget = other_gpu
        available = GPU_BUDGET - m_budget - p_budget - v_retries - other_budget
        minimum = 1 if count <= ARRAY_PART_SIZE else 2
        if available >= minimum:
            break
        record(
            root / "logs/roll_submitter.log",
            f"waiting for V GPU slots: M/P/V-retry/other budgets="
            f"{m_budget}/{p_budget}/{v_retries}/{other_budget}, available={available}",
        )
        time.sleep(poll_seconds)
    allocation = min(GPU_BUDGET, available, count)
    if count > ARRAY_PART_SIZE:
        first = min(GPU_BUDGET // 2, (allocation + 1) // 2)
        second = min(GPU_BUDGET // 2, allocation - first)
    else:
        first = allocation
        second = 0
    if count > ARRAY_PART_SIZE and second < 1:
        raise RuntimeError(
            f"V batch needs two nonzero array throttles, got {allocation}"
        )
    record(
        root / "logs/roll_submitter.log",
        f"planned GPN-Star-V concurrency {first}+{second}; reserved budget "
        f"M/P/V-retry/other={m_budget}/{p_budget}/{v_retries}/{other_budget}",
    )
    return first, second


def submit_next_batch(
    root: Path,
    state: dict,
    model: str,
    start: int,
    *,
    max_submit_retries: int,
    poll_seconds: int,
) -> list[str]:
    count = min(CHUNK_SIZE, TOTAL_SAMPLES - start)
    first = min(ARRAY_PART_SIZE, count)
    second = count - first
    if model == "GPN-Star-M":
        throttles = (7, 6) if second else (M_GPU_BUDGET // 2, 0)
    elif model == "GPN-Star-P":
        throttles = (3, 3) if second else (P_GPU_BUDGET, 0)
    else:
        throttles = plan_v_throttles(root, count, poll_seconds)
    gpus = 2 if model == "GPN-Star-M" else 1
    ids = []
    parts = [(start, first, throttles[0])]
    if second:
        parts.append((start + first, second, throttles[1]))
    for part_start, part_count, throttle in parts:
        job_id = submit_part(
            root,
            model,
            part_start,
            part_count,
            throttle,
            gpus=gpus,
            max_submit_retries=max_submit_retries,
            poll_seconds=poll_seconds,
            tranche_state="partial_submission" if ids else "not_submitted",
            prior_job_ids=ids,
        )
        ids.append(job_id)
        # Persist every accepted array immediately. If a later part is rejected,
        # a restarted coordinator can resume after these rows without resubmitting.
        update_model_state(state, model, job_id, part_start, part_count, throttle)
        save_state(root / "config/array_jobs.json", state)
    record(
        root / "logs/roll_submitter.log",
        f"submitted {model} rows [{start},{start + count}) as arrays {','.join(ids)}",
    )
    return ids


def maybe_raise_v_throttle(root: Path, state: dict) -> None:
    """Raise active V array throttles to use capacity released by completed models."""
    result = run(
        [
            "squeue",
            "--noheader",
            "--user",
            SLURM_USER,
            "--states=RUNNING",
            "--format=%j|%b",
        ]
    )
    if result.returncode != 0:
        return
    model_gpu = {"GPN-Star-M": 0, "GPN-Star-P": 0, "GPN-Star-V": 0}
    other_gpu = 0
    for line in result.stdout.splitlines():
        fields = line.split("|", 1)
        if len(fields) != 2:
            continue
        name, tres = fields
        match = re.search(r"(?:^|,)gpu:(\d+)", tres)
        if not match:
            continue
        gpu_count = int(match.group(1))
        if name.startswith("GPNStarM"):
            model_gpu["GPN-Star-M"] += gpu_count
        elif name.startswith("GPNStarP"):
            model_gpu["GPN-Star-P"] += gpu_count
        elif name.startswith("GPNStarV"):
            model_gpu["GPN-Star-V"] += gpu_count
        else:
            other_gpu += gpu_count
    m_pending = (
        int(state.get("GPN-Star-M", {}).get("submitted_rows", 0)) < TOTAL_SAMPLES
    )
    p_pending = (
        int(state.get("GPN-Star-P", {}).get("submitted_rows", 0)) < TOTAL_SAMPLES
    )
    m_arrays = active_array_entries(state, "GPN-Star-M")
    p_arrays = active_array_entries(state, "GPN-Star-P")
    v_arrays = [
        array
        for array in active_array_entries(state, "GPN-Star-V")
        if int(array.get("start", -1)) >= 600 and int(array.get("count", 0)) > 0
    ]
    m_array_budget = 2 * sum(int(array.get("throttle", 0)) for array in m_arrays)
    p_array_budget = sum(int(array.get("throttle", 0)) for array in p_arrays)
    m_budget = max(
        M_GPU_BUDGET if m_pending or m_arrays else 0,
        model_gpu["GPN-Star-M"],
        m_array_budget,
    )
    p_budget = max(
        P_GPU_BUDGET if p_pending else 0,
        model_gpu["GPN-Star-P"],
        p_array_budget,
    )
    available = GPU_BUDGET - m_budget - p_budget - other_gpu
    if available < 1:
        return
    if not v_arrays:
        return

    allocation = min(
        GPU_BUDGET, available, sum(int(array["count"]) for array in v_arrays)
    )
    if len(v_arrays) > 1:
        per_array_cap = GPU_BUDGET // 2
        base, remainder = divmod(allocation, len(v_arrays))
        targets = [
            min(per_array_cap, base + (i < remainder)) for i in range(len(v_arrays))
        ]
    else:
        targets = [allocation]
    changed = False
    log_path = root / "logs/roll_submitter.log"
    for array, target in zip(v_arrays, targets):
        target = min(int(array["count"]), int(target))
        if target == int(array.get("throttle", 0)):
            continue
        job_id = str(array["job_id"])
        updated = run(
            ["scontrol", "update", f"JobId={job_id}", f"ArrayTaskThrottle={target}"]
        )
        partial_success = updated.returncode != 0 and "No error" in updated.stdout
        if updated.returncode != 0 and not partial_success:
            record(
                log_path,
                f"could not raise V array {job_id} throttle: {updated.stdout.strip()}",
            )
            continue
        array["throttle"] = target
        changed = True
        if partial_success:
            record(
                log_path,
                f"V array {job_id}: completed task ranges ignored; active ranges updated",
            )
        record(
            log_path,
            f"set GPN-Star-V array {job_id} throttle to {target}; "
            f"reserved M/P/other GPUs={m_budget}/{p_budget}/{other_gpu}, "
            f"V allocation cap={allocation}",
        )
    if changed:
        save_state(root / "config/array_jobs.json", state)


def run_finalization(
    root: Path,
    repo: Path,
    benchmark: Path,
    python: Path,
    regression_script: Path | None,
) -> None:
    repo = repo.expanduser().resolve()
    benchmark = benchmark.expanduser().resolve()
    tables = repo / "tables"
    python = python.expanduser().resolve()
    log_path = root / "logs/roll_submitter.log"
    finalizer = repo / "scripts/population_classify/finalize_gpn_star_ancestry.py"
    command = [
        str(python),
        str(finalizer),
        "--run-root",
        str(root),
        "--benchmark-root",
        str(benchmark),
        "--repo-root",
        str(repo),
        "--tables-dir",
        str(tables),
    ]
    if regression_script is not None:
        command.extend(
            ["--regression-script", str(regression_script.expanduser().resolve())]
        )
    record(log_path, "starting S33-S35 validation and table update")
    result = run(command, capture=False)
    if result.returncode != 0:
        raise RuntimeError("finalize_gpn_star_ancestry.py failed")

    figure_root = root / "figures"
    outputs = {
        "Fig1": [
            python,
            repo / "figures/scripts/Fig1/A4_integrated.py",
            "--tables-dir",
            tables,
            "--output-dir",
            figure_root / "Fig1",
            "--architecture-pdf",
            repo / "figures/Fig1/Fig1g_120mm.pdf",
        ],
        "Fig4": [
            python,
            repo / "figures/scripts/Fig4/A4_integrated.py",
            "--tables-dir",
            tables,
            "--output-dir",
            figure_root / "Fig4",
        ],
        "Fig5": [
            python,
            repo / "figures/scripts/Fig5/A4_integrated.py",
            "--tables-dir",
            tables,
            "--output-dir",
            figure_root / "Fig5",
        ],
    }
    for figure, argv in outputs.items():
        record(log_path, f"redrawing {figure}")
        command = [str(value) if isinstance(value, Path) else value for value in argv]
        result = run(command, capture=False)
        if result.returncode != 0:
            raise RuntimeError(f"Figure redraw failed: {figure}")

    import shutil
    from datetime import datetime

    backup_root = root / "backups/figures" / datetime.now().strftime("%Y%m%d_%H%M%S")
    copied = []
    for figure in ("Fig1", "Fig4", "Fig5"):
        source_root = figure_root / figure
        destination_root = repo / "figures" / figure
        for source in sorted(path for path in source_root.rglob("*") if path.is_file()):
            relative = source.relative_to(source_root)
            destination = destination_root / relative
            if destination.exists():
                backup = backup_root / figure / relative
                backup.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(destination, backup)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            copied.append(str(destination))
    report = {
        "status": "complete",
        "figures": list(outputs),
        "copied_to_repository": copied,
        "overwritten_file_backups": str(backup_root),
    }
    (root / "results/figure_refresh_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    record(log_path, "table update and figure refresh completed")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="hEvo_benchmark checkout (defaults to this script's checkout)",
    )
    parser.add_argument(
        "--benchmark-root",
        type=Path,
        default=Path(os.environ["DNAMODELS_BENCHMARK_ROOT"]).expanduser()
        if os.environ.get("DNAMODELS_BENCHMARK_ROOT")
        else None,
        help="DNAModelsBenchmark checkout (or DNAMODELS_BENCHMARK_ROOT)",
    )
    parser.add_argument(
        "--python-executable",
        type=Path,
        default=Path(sys.executable),
        help="Python used for finalization and figure refresh (defaults to this interpreter)",
    )
    parser.add_argument(
        "--regression-script",
        type=Path,
        default=Path(os.environ["HEVO_REGRESSION_SCRIPT"]).expanduser()
        if os.environ.get("HEVO_REGRESSION_SCRIPT")
        else None,
        help="Optional override; otherwise resolved from --benchmark-root",
    )
    parser.add_argument(
        "--max-submit-retries",
        type=int,
        default=MAX_SUBMIT_RETRIES,
        help="Maximum retries for recognized transient sbatch/controller failures (default: 5)",
    )
    parser.add_argument(
        "--poll-seconds",
        type=int,
        default=POLL_SECONDS,
        help="Base polling and retry-backoff delay in seconds (default: 60)",
    )
    args = parser.parse_args()
    if args.benchmark_root is None:
        parser.error("--benchmark-root or DNAMODELS_BENCHMARK_ROOT is required")
    if args.max_submit_retries < 0:
        parser.error("--max-submit-retries must be zero or greater")
    if args.poll_seconds < 1:
        parser.error("--poll-seconds must be a positive integer")
    root = args.run_root.expanduser().resolve()
    state_path = root / "config/array_jobs.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    if slurm_job_id:
        state["coordinator_job_id"] = str(slurm_job_id)
        save_state(state_path, state)
    log_path = root / "logs/roll_submitter.log"
    ready_after: dict[str, float | None] = {model: None for model in MODELS}
    next_start = {model: int(state[model]["submitted_rows"]) for model in MODELS}
    current_start = {model: 0 for model in MODELS}
    current_count = {model: int(state[model]["submitted_rows"]) for model in MODELS}
    current_ids = {
        model: [str(row["job_id"]) for row in state[model]["arrays"]]
        for model in MODELS
    }
    record(log_path, "rolling array coordinator started")

    input_files = [
        Path(line)
        for line in (root / "config/input_files.txt")
        .read_text(encoding="utf-8")
        .splitlines()
        if line
    ]
    if len(input_files) != TOTAL_SAMPLES:
        raise RuntimeError(f"Expected {TOTAL_SAMPLES} inputs, found {len(input_files)}")
    expected_input_metadata = [
        {"sample_index": index, **input_metadata(path)}
        for index, path in enumerate(input_files)
    ]
    expected_names = [
        f"{row['population']}_{row['sample_id']}.npy" for row in expected_input_metadata
    ]
    if len(set(expected_names)) != TOTAL_SAMPLES:
        raise RuntimeError(
            "Input roster contains duplicate sample/population identities"
        )

    def expected_artifacts(
        model: str, start: int, count: int
    ) -> list[tuple[Path, Path, dict]]:
        result_root = root / "results" / model
        pairs = []
        for metadata in expected_input_metadata[start : start + count]:
            artifact_stem = f"{metadata['population']}_{metadata['sample_id']}"
            pairs.append(
                (
                    result_root / "embeddings" / f"{artifact_stem}.npy",
                    result_root / "manifests" / f"{artifact_stem}.json",
                    metadata,
                )
            )
        if len(pairs) != count:
            raise RuntimeError(
                f"Input roster does not cover {model} rows [{start},{start + count})"
            )
        return pairs

    while any(next_start[model] < TOTAL_SAMPLES for model in MODELS):
        maybe_raise_v_throttle(root, state)
        maybe_adjust_m_throttle(root, state)
        for model in MODELS:
            if next_start[model] >= TOTAL_SAMPLES:
                continue
            artifacts = expected_artifacts(
                model, current_start[model], current_count[model]
            )
            if not check_finished(
                root,
                model,
                current_ids[model],
                artifacts,
                log_path,
                current_start[model],
                max_submit_retries=args.max_submit_retries,
                poll_seconds=args.poll_seconds,
            ):
                ready_after[model] = None
                continue
            if ready_after[model] is None:
                ready_after[model] = time.time() + MIN_JOB_AGE_SECONDS
                record(
                    log_path,
                    f"{model} tranche complete; waiting for task records to expire",
                )
                continue
            if time.time() < ready_after[model]:
                continue
            start = next_start[model]
            current_ids[model] = submit_next_batch(
                root,
                state,
                model,
                start,
                max_submit_retries=args.max_submit_retries,
                poll_seconds=args.poll_seconds,
            )
            current_start[model] = start
            current_count[model] = min(CHUNK_SIZE, TOTAL_SAMPLES - start)
            next_start[model] += current_count[model]
            ready_after[model] = None
        if any(next_start[model] < TOTAL_SAMPLES for model in MODELS):
            time.sleep(args.poll_seconds)

    for model in MODELS:
        artifacts = expected_artifacts(
            model, current_start[model], current_count[model]
        )
        if not check_finished(
            root,
            model,
            current_ids[model],
            artifacts,
            log_path,
            current_start[model],
            max_submit_retries=args.max_submit_retries,
            poll_seconds=args.poll_seconds,
        ):
            wait_until_finished(
                root,
                model,
                current_ids[model],
                artifacts,
                current_start[model],
                max_submit_retries=args.max_submit_retries,
                poll_seconds=args.poll_seconds,
            )
        else:
            record(log_path, f"final {model} tranche completed")
    record(log_path, "all 1500-sample model arrays completed; starting finalization")
    run_finalization(
        root,
        args.repo_root,
        args.benchmark_root,
        args.python_executable,
        args.regression_script,
    )


if __name__ == "__main__":
    main()
