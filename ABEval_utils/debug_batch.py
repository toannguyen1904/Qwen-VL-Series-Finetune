#!/usr/bin/env python3
"""Debug: load a small batch from the dataset and check grid vs token counts."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch, itertools, ujson as json
from types import SimpleNamespace
from transformers import AutoProcessor

MODEL = "Qwen/Qwen3-VL-4B-Instruct"
DATA_PATH = "data/abeval_train.json"
IMAGE_FOLDER = "/data/tientoan/ABEval_data/evaluation_sessions"

processor = AutoProcessor.from_pretrained(MODEL)

data_args = SimpleNamespace(
    image_folder=IMAGE_FOLDER,
    image_min_pixels=3136,
    image_max_pixels=12845056,
    video_min_pixels=100352,
    video_max_pixels=147456,
    image_resized_width=None,
    image_resized_height=None,
    video_resized_width=None,
    video_resized_height=None,
    fps=1.0,
    nframes=None,
    enable_reasoning=False,
    eval_path=None,
    data_path=DATA_PATH,
)

from dataset.sft_dataset import SupervisedDataset, DataCollatorForSupervisedDataset

ds = SupervisedDataset(
    data_path=DATA_PATH,
    processor=processor,
    data_args=data_args,
    model_id=MODEL,
)

collator = DataCollatorForSupervisedDataset(pad_token_id=processor.tokenizer.pad_token_id)

print(f"Dataset length: {len(ds)}")
print("Loading first sample...")
sample = ds[0]
print(f"  Keys: {list(sample.keys())}")
if "video_grid_thw" in sample:
    print(f"  video_grid_thw shape: {sample['video_grid_thw'].shape}")
    print(f"  video_grid_thw:\n{sample['video_grid_thw']}")
if "mm_token_type_ids" in sample:
    types = sample["mm_token_type_ids"]
    for t in [0, 1, 2]:
        count = (types == t).sum().item()
        if count > 0:
            print(f"  mm_token_type_ids type={t} count: {count}")
    groups = [(k, len(list(g))) for k, g in itertools.groupby(types.tolist())]
    video_groups = [(k, l) for k, l in groups if k == 2]
    print(f"  Video groups (type=2): {len(video_groups)}, sizes: {[l for _, l in video_groups]}")

if "mm_token_type_ids" in sample:
    types = sample["mm_token_type_ids"]
    ids = sample["input_ids"]
    groups = []
    for k, g in itertools.groupby(enumerate(zip(types.tolist(), ids.tolist())), lambda x: x[1][0]):
        g = list(g)
        start = g[0][0]
        end = g[-1][0]
        tok_ids = [x[1][1] for x in g]
        groups.append((k, start, end+1, len(g), tok_ids[:5]))
    print(f"\n  Token groups (first 10):")
    for k, s, e, l, toks in groups[:10]:
        decoded = [processor.tokenizer.decode([t]) for t in toks]
        print(f"    type={k} range=[{s}:{e}] len={l} first_tokens={decoded}")
    print(f"  Total groups: {len(groups)}")

print(f"\nTesting _expand_video_grid_to_frames fix...")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'train'))
from monkey_patch_forward import _expand_video_grid_to_frames

expanded = _expand_video_grid_to_frames(sample["video_grid_thw"])
print(f"  Original grid: {sample['video_grid_thw'].shape} -> {sample['video_grid_thw'].tolist()}")
print(f"  Expanded grid: {expanded.shape} -> first 3: {expanded[:3].tolist()}")
video_groups = sum(1 for k, _ in itertools.groupby(sample["mm_token_type_ids"].tolist()) if k == 2)
print(f"  Video groups in mm_token_type_ids: {video_groups}")
print(f"  Expanded grid rows: {expanded.shape[0]}")
if video_groups == expanded.shape[0]:
    print(f"  OK: counts match after expansion")
else:
    print(f"  *** STILL MISMATCHED ***")

print(f"\nLoading 8-sample batch...")
samples = [ds[i] for i in range(8)]
batch = collator(samples)
print(f"Batch keys: {list(batch.keys())}")
print(f"  input_ids shape: {batch['input_ids'].shape}")
if "video_grid_thw" in batch and "mm_token_type_ids" in batch:
    total_video_groups = 0
    for idx in range(batch["input_ids"].shape[0]):
        attn = batch["attention_mask"][idx].bool()
        tt = batch["mm_token_type_ids"][idx][attn]
        vgroups = sum(1 for k, _ in itertools.groupby(tt.tolist()) if k == 2)
        total_video_groups += vgroups
    expanded_batch = _expand_video_grid_to_frames(batch["video_grid_thw"])
    print(f"  Total video groups across batch: {total_video_groups}")
    print(f"  Original grid rows: {batch['video_grid_thw'].shape[0]}")
    print(f"  Expanded grid rows: {expanded_batch.shape[0]}")
    if total_video_groups == expanded_batch.shape[0]:
        print(f"  OK: batch counts match after expansion")
    else:
        print(f"  *** STILL MISMATCHED: {total_video_groups} groups vs {expanded_batch.shape[0]} rows ***")
