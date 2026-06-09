#!/usr/bin/env python3
"""Initialize 6-layer students from the two staged 12-layer teachers."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Tuple

import torch
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

from experiment_config import INIT_ROOT, SIX_LAYER_INIT_MODELS, TEACHER_MODELS
from shared import copy_custom_model_files, require_dir, setup_logging, validate_all_teachers, write_json


LAYER_MAPPING = {1: 0, 3: 1, 5: 2, 7: 3, 9: 4, 11: 5}
LAYER_RE = re.compile(r"^(.*\.(?:layer|layers|block|blocks|h)\.)(\d+)(\..*)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Initialize Realm-Rank v4 6L student models")
    parser.add_argument("--output-root", default=str(INIT_ROOT))
    parser.add_argument("--force", action="store_true", help="Overwrite existing 6L model files")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def count_parameters(model: torch.nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def set_num_layers(config, layers: int) -> None:
    if hasattr(config, "num_hidden_layers"):
        config.num_hidden_layers = layers
    elif hasattr(config, "n_layers"):
        config.n_layers = layers
    elif hasattr(config, "n_layer"):
        config.n_layer = layers
    else:
        config.num_hidden_layers = layers


def build_student_state(teacher_state: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], int, int]:
    student_state: Dict[str, torch.Tensor] = {}
    copied = 0
    skipped = 0
    for key, tensor in teacher_state.items():
        match = LAYER_RE.match(key)
        if not match:
            student_state[key] = tensor.detach().clone()
            copied += 1
            continue
        prefix, idx_str, suffix = match.groups()
        teacher_idx = int(idx_str)
        if teacher_idx not in LAYER_MAPPING:
            skipped += 1
            continue
        student_key = f"{prefix}{LAYER_MAPPING[teacher_idx]}{suffix}"
        student_state[student_key] = tensor.detach().clone()
        copied += 1
    return student_state, copied, skipped


def init_one(name: str, teacher_dir: Path, output_dir: Path, force: bool, dry_run: bool) -> Dict[str, object]:
    if output_dir.exists() and not force and (output_dir / "pytorch_model.bin").is_file():
        config = json.load(open(output_dir / "config.json"))
        return {
            "name": name,
            "teacher_dir": str(teacher_dir),
            "output_dir": str(output_dir),
            "status": "exists",
            "num_hidden_layers": config.get("num_hidden_layers"),
        }

    config = AutoConfig.from_pretrained(str(teacher_dir), trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(str(teacher_dir), trust_remote_code=True)
    teacher = AutoModelForSequenceClassification.from_pretrained(
        str(teacher_dir),
        config=config,
        trust_remote_code=True,
        torch_dtype=torch.float32,
    )
    teacher_params = count_parameters(teacher)

    student_config = AutoConfig.from_pretrained(str(teacher_dir), trust_remote_code=True)
    set_num_layers(student_config, 6)
    student = AutoModelForSequenceClassification.from_config(student_config, trust_remote_code=True)
    student_state, copied, skipped = build_student_state(teacher.state_dict())
    missing, unexpected = student.load_state_dict(student_state, strict=False)
    student_params = count_parameters(student)

    report = {
        "name": name,
        "teacher_dir": str(teacher_dir),
        "output_dir": str(output_dir),
        "status": "dry_run" if dry_run else "written",
        "num_hidden_layers": int(getattr(student.config, "num_hidden_layers", 0)),
        "layer_mapping": {str(k): v for k, v in LAYER_MAPPING.items()},
        "teacher_parameters": teacher_params,
        "student_parameters": student_params,
        "parameter_ratio": student_params / teacher_params if teacher_params else None,
        "copied_tensors": copied,
        "skipped_tensors": skipped,
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
    }

    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        student.save_pretrained(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))
        copy_custom_model_files(teacher_dir, output_dir)
        write_json(output_dir / "init_report.json", report)

    del teacher, student
    return report


def main() -> None:
    setup_logging()
    args = parse_args()
    validate_all_teachers(TEACHER_MODELS)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    reports = []
    for name, teacher_dir in TEACHER_MODELS.items():
        output_dir = Path(args.output_root) / SIX_LAYER_INIT_MODELS[name].name
        report = init_one(name, Path(teacher_dir), output_dir, args.force, args.dry_run)
        reports.append(report)
        print(json.dumps(report, indent=2, sort_keys=True))

    if not args.dry_run:
        write_json(Path(args.output_root) / "six_layer_init_summary.json", {"models": reports})


if __name__ == "__main__":
    main()
