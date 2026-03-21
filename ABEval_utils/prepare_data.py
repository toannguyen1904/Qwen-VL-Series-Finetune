#!/usr/bin/env python3
"""Convert ABEval evaluation sessions into multi-video SFT JSON for Qwen-VL finetuning."""

import json
import os
import random
import re
import sys
from pathlib import Path

import yaml

DATA_ROOT = Path("/data/tientoan/ABEval_data/evaluation_sessions")
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data"
CAMERAS = ("left", "right")  # wrist excluded
TRAIN_RATIO = 0.95
SEED = 42

HUMAN_TEMPLATE = (
    "<video>\n<video>\n"
    "You are given two robot policy rollout videos (Video A and Video B) "
    "performing the same task: \"{instruction}\".\n"
    "Evaluate which policy performed better. Provide:\n"
    "1. Your preference (A, B, or TIE).\n"
    "2. A free-form explanation of your preference.\n"
    "3. A progress score from 0 to 100 for each rollout "
    "(0 = no progress, 100 = task completed)."
)

GPT_TEMPLATE = (
    "Preference: {preference}\n\n"
    "Explanation: {explanation}\n\n"
    "Progress Score A: {score_a}\n"
    "Progress Score B: {score_b}"
)


def find_video(rollout_dir: Path, camera: str) -> str | None:
    """Return the filename of a video matching the given camera in rollout_dir."""
    suffix = f"_video_{camera}.mp4"
    for entry in os.scandir(rollout_dir):
        if entry.is_file() and entry.name.endswith(suffix):
            return entry.name
    return None


def find_rollout_dir(session_dir: Path, letter: str) -> Path | None:
    """Return the rollout directory for a given letter (A or B), or None."""
    prefix = f"{letter}_"
    for entry in os.scandir(session_dir):
        if entry.is_dir() and entry.name.startswith(prefix):
            return Path(entry.path)
    return None


def build_samples(session_dir: Path) -> list[dict]:
    meta_path = session_dir / "metadata.yaml"
    if not meta_path.exists():
        return []

    with open(meta_path) as f:
        meta = yaml.safe_load(f)

    policies = meta.get("policies", {})
    if "A" not in policies or "B" not in policies:
        return []

    instruction = meta.get("language_instruction", "").strip()
    preference = str(meta.get("preference", "")).strip()
    explanation = meta.get("longform_feedback", "").strip()
    score_a = int(round(float(policies["A"].get("partial_success", 0)) * 100))
    score_b = int(round(float(policies["B"].get("partial_success", 0)) * 100))

    if not instruction or not preference:
        return []

    dir_a = find_rollout_dir(session_dir, "A")
    dir_b = find_rollout_dir(session_dir, "B")
    if dir_a is None or dir_b is None:
        return []

    session_id = session_dir.name
    samples = []

    for camera in CAMERAS:
        vid_a = find_video(dir_a, camera)
        vid_b = find_video(dir_b, camera)
        if vid_a is None or vid_b is None:
            continue

        path_a = f"{session_id}/{dir_a.name}/{vid_a}"
        path_b = f"{session_id}/{dir_b.name}/{vid_b}"

        human_msg = HUMAN_TEMPLATE.format(instruction=instruction)
        gpt_msg = GPT_TEMPLATE.format(
            preference=preference,
            explanation=explanation,
            score_a=score_a,
            score_b=score_b,
        )

        samples.append({
            "id": f"{session_id}_{camera}",
            "video": [path_a, path_b],
            "conversations": [
                {"from": "human", "value": human_msg},
                {"from": "gpt", "value": gpt_msg},
            ],
        })

    return samples


def main():
    if not DATA_ROOT.is_dir():
        print(f"ERROR: data root not found: {DATA_ROOT}", file=sys.stderr)
        sys.exit(1)

    sessions = sorted(
        p for p in DATA_ROOT.iterdir() if p.is_dir() and not p.name.startswith(".")
    )
    print(f"Found {len(sessions)} evaluation sessions")

    all_samples: list[dict] = []
    skipped = 0
    for i, session_dir in enumerate(sessions):
        samples = build_samples(session_dir)
        if not samples:
            skipped += 1
        all_samples.extend(samples)
        if (i + 1) % 500 == 0:
            print(f"  processed {i + 1}/{len(sessions)} sessions, {len(all_samples)} samples so far")

    print(f"Total samples: {len(all_samples)}  (skipped {skipped} sessions)")

    random.seed(SEED)
    random.shuffle(all_samples)

    split_idx = int(len(all_samples) * TRAIN_RATIO)
    train_data = all_samples[:split_idx]
    test_data = all_samples[split_idx:]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train_path = OUTPUT_DIR / "abeval_train.json"
    test_path = OUTPUT_DIR / "abeval_test.json"

    with open(train_path, "w") as f:
        json.dump(train_data, f, indent=2)
    with open(test_path, "w") as f:
        json.dump(test_data, f, indent=2)

    print(f"Wrote {len(train_data)} train samples -> {train_path}")
    print(f"Wrote {len(test_data)} test samples  -> {test_path}")


if __name__ == "__main__":
    main()
