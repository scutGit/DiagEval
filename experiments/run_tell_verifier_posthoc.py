#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Post-hoc TellVerifier script for round1 results.

For each completed task in a work_dir, reads the Tell action and screenshots,
runs TellVerifier, and writes verification results back to the result JSON.

Usage:
    cd <PROJECT_ROOT>
    python experiments/run_tell_verifier_posthoc.py \
        --work_dir work_dirs/webdevjudge_41-100_round1_gemini-3-flash-preview \
        --config test/webdevjudge_dev/run_config.yaml \
        --workers 8
"""

import argparse
import asyncio
import json
import os
import re
import sys
from pathlib import Path
from typing import List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from metagpt.logs import logger

from appeval.actions.tell_verifier import TellVerifier, VerificationResult


def find_task_dirs(work_dir: Path) -> List[Path]:
    """Find all task directories that have a result JSON."""
    result_dirs = []
    for task_dir in sorted(work_dir.iterdir()):
        if not task_dir.is_dir():
            continue
        inner = task_dir / task_dir.name
        result_json = inner / f"{task_dir.name}.json"
        if result_json.exists():
            result_dirs.append(task_dir)
    return result_dirs


def load_checkpoint(task_dir: Path) -> Optional[dict]:
    """Load resume_checkpoint.json."""
    cp_path = task_dir / "resume_checkpoint.json"
    if not cp_path.exists():
        return None
    with open(cp_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_tell_content(checkpoint: dict) -> Optional[str]:
    """Extract the last Tell action from checkpoint result_history."""
    result_history = checkpoint.get("decision_payload", {}).get("result_history", [])
    for entry in reversed(result_history):
        action = entry.get("action", "")
        if action.startswith("Tell"):
            return action
    return None


def get_screenshot_dir(checkpoint: dict, base_dir: Path) -> Optional[Path]:
    """Derive screenshot directory from checkpoint."""
    last_screenshot = checkpoint.get("replay_core", {}).get("last_screenshot_path", "")
    if last_screenshot:
        # Path is relative to repo root
        screenshot_path = base_dir / last_screenshot
        if screenshot_path.exists():
            return screenshot_path.parent
        # Try as relative from work_dirs parent
        parts = Path(last_screenshot).parts
        if "work_dirs" in parts:
            idx = list(parts).index("work_dirs")
            rel = Path(*parts[idx:])
            candidate = base_dir / rel
            if candidate.parent.exists():
                return candidate.parent
    return None


def get_last_iter(checkpoint: dict) -> int:
    """Get last completed iteration from checkpoint."""
    return checkpoint.get("replay_core", {}).get("last_completed_iter", 0)


def parse_action_reflection_from_log(info_txt: Path) -> Tuple[List[str], List[str]]:
    """Parse action_history and reflection_history from info.txt log."""
    if not info_txt.exists():
        return [], []

    action_history = []
    reflection_history = []

    try:
        content = info_txt.read_text(encoding="utf-8", errors="ignore")
        # Each agent step is wrapped in output_action blocks
        blocks = re.findall(
            r"######################## output_action.*?:\n(.*?)######################## output_action end",
            content,
            re.DOTALL,
        )
        for block in blocks:
            lines = block.strip().splitlines()
            # Last non-empty line is the action (Run/Tell/Stop)
            action_line = ""
            for line in reversed(lines):
                stripped = line.strip()
                if stripped and (
                    stripped.startswith("Run (")
                    or stripped.startswith("Tell (")
                    or stripped.startswith("Stop")
                    or stripped.startswith("Run(")
                    or stripped.startswith("Tell(")
                ):
                    action_line = stripped
                    break

            # Extract reflection from ### Thought ### section
            thought_match = re.search(
                r"### Thought ###\n(.*?)(?=\n### Action ###|\n### Operation ###|$)",
                block,
                re.DOTALL,
            )
            reflection = thought_match.group(1).strip() if thought_match else ""

            action_history.append(action_line)
            reflection_history.append(reflection)
    except Exception as e:
        logger.warning(f"Failed to parse info.txt: {e}")

    return action_history, reflection_history


def get_test_cases_str(result_json_path: Path) -> str:
    """Get test cases description string from result JSON."""
    try:
        with open(result_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        cases = data.get("test_cases", [])
        case_dict = {c["test_id"]: c["case_desc"] for c in cases if "case_desc" in c}
        return json.dumps(case_dict, ensure_ascii=False)
    except Exception:
        return ""


def already_verified(result_json_path: Path) -> bool:
    """Check if this result already has verification data."""
    try:
        with open(result_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        cases = data.get("test_cases", [])
        return any("verification_status" in c for c in cases)
    except Exception:
        return False


def update_result_json(
    result_json_path: Path,
    verification: VerificationResult,
    tell_content: str,
) -> None:
    """Write verification results back to the result JSON."""
    with open(result_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Parse corrections per test_id from the verification result
    corrections = verification.corrections  # dict: {test_id: {...}}

    for case in data.get("test_cases", []):
        tid = str(case.get("test_id", "0"))
        case["verification_status"] = verification.verification_status
        case["verification_reasoning"] = verification.reasoning[:500]
        case["tell_verifier_valid"] = verification.is_valid

        # Apply per-case result correction if available
        if tid in corrections:
            corr = corrections[tid]
            if "result" in corr:
                corrected_result = corr["result"]
                # Normalize: "Pass"/"Fail" -> True/False
                if isinstance(corrected_result, str):
                    case["result_original"] = case.get("result")
                    case["result"] = corrected_result.lower() == "pass"
                elif isinstance(corrected_result, bool):
                    case["result_original"] = case.get("result")
                    case["result"] = corrected_result
            if "evidence" in corr:
                case["evidence_original"] = case.get("evidence")
                case["evidence"] = corr["evidence"]

    with open(result_json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


async def process_task(
    task_dir: Path,
    verifier: TellVerifier,
    base_dir: Path,
    semaphore: asyncio.Semaphore,
) -> Tuple[str, str]:
    """Process a single task directory. Returns (task_name, status)."""
    task_name = task_dir.name
    inner_dir = task_dir / task_name
    result_json = inner_dir / f"{task_name}.json"

    async with semaphore:
        try:
            # Skip if already verified
            if already_verified(result_json):
                return task_name, "skipped"

            # Load checkpoint
            checkpoint = load_checkpoint(task_dir)
            if not checkpoint:
                return task_name, "no_checkpoint"

            # Get Tell content
            tell_content = get_tell_content(checkpoint)
            if not tell_content:
                return task_name, "no_tell_action"

            # Get screenshot directory
            screenshot_dir = get_screenshot_dir(checkpoint, base_dir)
            if not screenshot_dir:
                # Try to find it by scanning inner_dir
                ts_dirs = [d for d in inner_dir.iterdir() if d.is_dir() and re.match(r"\d{12}", d.name)]
                if ts_dirs:
                    screenshot_dir = sorted(ts_dirs)[-1]
                else:
                    return task_name, "no_screenshot_dir"

            last_iter = get_last_iter(checkpoint)

            # Parse action/reflection history from info.txt
            info_txt = screenshot_dir / "info.txt"
            action_history, reflection_history = parse_action_reflection_from_log(info_txt)

            # Get test cases description
            test_cases_str = get_test_cases_str(result_json)

            # Run verifier
            result = await verifier.run(
                tell_content=tell_content,
                action_history=action_history,
                reflection_history=reflection_history,
                screenshot_dir=str(screenshot_dir),
                current_iter=last_iter,
                test_cases=test_cases_str,
            )

            # Save results
            update_result_json(result_json, result, tell_content)

            status = f"verified:{result.verification_status}"
            logger.info(f"[{task_name}] {status}")
            return task_name, status

        except Exception as e:
            logger.error(f"[{task_name}] Error: {e}")
            return task_name, f"error:{e}"


async def main(work_dir: str, config_path: str, workers: int):
    work_dir_path = Path(work_dir)
    base_dir = Path(__file__).parent.parent

    logger.info(f"Scanning {work_dir_path} ...")
    task_dirs = find_task_dirs(work_dir_path)
    logger.info(f"Found {len(task_dirs)} task directories with result JSONs")

    # Initialize verifier (shared across all tasks)
    verifier = TellVerifier(config_path=config_path)

    semaphore = asyncio.Semaphore(workers)
    tasks = [
        process_task(td, verifier, base_dir, semaphore)
        for td in task_dirs
    ]

    results = await asyncio.gather(*tasks)

    # Summary
    from collections import Counter
    status_counts = Counter(s for _, s in results)
    logger.info("\n=== Summary ===")
    for status, count in sorted(status_counts.items()):
        logger.info(f"  {status}: {count}")
    logger.info(f"Total: {len(results)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Post-hoc TellVerifier for round1 results")
    parser.add_argument(
        "--work_dir",
        default="work_dirs/webdevjudge_41-100_round1_gemini-3-flash-preview",
        help="Work directory to process",
    )
    parser.add_argument(
        "--config",
        default="test/webdevjudge_dev/run_config.yaml",
        help="Path to run_config.yaml with tell_verifier section",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of concurrent workers",
    )
    args = parser.parse_args()

    os.chdir(Path(__file__).parent.parent)
    asyncio.run(main(args.work_dir, args.config, args.workers))
