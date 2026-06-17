#!/usr/bin/env python3
"""Path and model-name defaults for the Realm-Rank six-model experiment."""

import os
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
GAMIL_ROOT = Path(os.environ.get("GAMIL_ROOT", SCRIPT_DIR.parents[2])).resolve()
RAW_DATA_ROOT = Path(os.environ.get("RAW_DATA_ROOT", GAMIL_ROOT / "raw_data" / "local_sources")).resolve()
PROCESSED_DATA_ROOT = Path(os.environ.get("PROCESSED_DATA_ROOT", GAMIL_ROOT / "processed_data")).resolve()
CHECKPOINT_ROOT = Path(os.environ.get("CHECKPOINT_ROOT", GAMIL_ROOT / "checkpoint" / "local_checkpoints")).resolve()

DATA_ROOT = PROCESSED_DATA_ROOT / "realm_rank"
TRAIN_CSV_DIR = DATA_ROOT / "train_csv"
TRAIN_CSV = TRAIN_CSV_DIR / "train.csv"
DEV_CSV = TRAIN_CSV_DIR / "dev.csv"
TEST_FASTA = DATA_ROOT / "test.fasta.gz"

OUTPUT_ROOT = Path(os.environ.get("OUTPUT_ROOT", CHECKPOINT_ROOT / "gamil_six_model")).resolve()
MODEL_ROOT = OUTPUT_ROOT / "models"
INIT_ROOT = OUTPUT_ROOT / "init"
LOG_ROOT = OUTPUT_ROOT / "logs"
STATE_ROOT = OUTPUT_ROOT / "state"
BENCHMARK_ROOT = OUTPUT_ROOT / "benchmark"
REFERENCE_ROOT = OUTPUT_ROOT / "reference"

STAGED_ROOT = Path(os.environ.get("STAGED_MODEL_ROOT", GAMIL_ROOT / "model" / "local_models" / "staged_models")).resolve()

TEACHER_MODELS = {
    "viralm_o": STAGED_ROOT / "viralm-o",
    "viralm_r": STAGED_ROOT / "viralm-r",
}

SIX_LAYER_INIT_MODELS = {
    "viralm_o": INIT_ROOT / "viralm_o_6l",
    "viralm_r": INIT_ROOT / "viralm_r_6l",
}

TRAINING_TASKS = [
    {
        "name": "viralm_o_6l_meanpool_kd",
        "entry": "train_meanpool_kd.py",
        "student": SIX_LAYER_INIT_MODELS["viralm_o"],
        "teacher": TEACHER_MODELS["viralm_o"],
        "kind": "meanpool_kd",
    },
    {
        "name": "viralm_r_6l_meanpool_kd",
        "entry": "train_meanpool_kd.py",
        "student": SIX_LAYER_INIT_MODELS["viralm_r"],
        "teacher": TEACHER_MODELS["viralm_r"],
        "kind": "meanpool_kd",
    },
    {
        "name": "viralm_o_6l_gated_mil_kd",
        "entry": "train_gated_mil_kd.py",
        "student": SIX_LAYER_INIT_MODELS["viralm_o"],
        "teacher": TEACHER_MODELS["viralm_o"],
        "kind": "gated_mil_kd",
    },
    {
        "name": "viralm_r_6l_gated_mil_kd",
        "entry": "train_gated_mil_kd.py",
        "student": SIX_LAYER_INIT_MODELS["viralm_r"],
        "teacher": TEACHER_MODELS["viralm_r"],
        "kind": "gated_mil_kd",
    },
    {
        "name": "viralm_o_12l_gated_mil",
        "entry": "train_gated_mil_supervised.py",
        "student": TEACHER_MODELS["viralm_o"],
        "teacher": None,
        "kind": "gated_mil_supervised",
    },
    {
        "name": "viralm_r_12l_gated_mil",
        "entry": "train_gated_mil_supervised.py",
        "student": TEACHER_MODELS["viralm_r"],
        "teacher": None,
        "kind": "gated_mil_supervised",
    },
]

BENCHMARK_MODEL_ORDER = [
    "viralm_o",
    "viralm_r",
    "viralm_o_6l_meanpool_kd",
    "viralm_r_6l_meanpool_kd",
    "viralm_o_6l_gated_mil_kd",
    "viralm_r_6l_gated_mil_kd",
    "viralm_o_12l_gated_mil",
    "viralm_r_12l_gated_mil",
]

REQUIRED_STAGED_FILES = (
    "config.json",
    "tokenizer.json",
    "pytorch_model.bin",
    "configuration_bert.py",
    "bert_layers.py",
    "bert_padding.py",
    "flash_attn_triton.py",
)

TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
)

CUSTOM_CODE_FILES = (
    "configuration_bert.py",
    "bert_layers.py",
    "bert_padding.py",
    "flash_attn_triton.py",
)

EUK_PRO_EVAL_ROOT = Path(os.environ.get("EUK_PRO_EVAL_ROOT", GAMIL_ROOT / "benchmark" / "results" / "euk_pro_o_vs_r")).resolve()
EUK_PRO_REFERENCE_FILES = {
    "summary.md": "euk_pro_summary.md",
    "sequence_o_vs_r_comparison.csv": "euk_pro_sequence_o_vs_r_comparison.csv",
    "sequence_metrics_with_auc.csv": "euk_pro_sequence_metrics_with_auc.csv",
    "fragment_metrics_with_auc.csv": "euk_pro_fragment_metrics_with_auc.csv",
}
