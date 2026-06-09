#!/usr/bin/env python3
"""Shard-aware ViraLM sequence classification inference.

This script is intentionally single-process/single-GPU for model inference.
Parallelism is provided by the outer shard queue, while tokenization can use
DataLoader worker processes.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import re
import shutil
import sys
import time
from collections import OrderedDict
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from Bio import SeqIO
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer


VALID_DNA_RE = re.compile(r"[^ACGT]")


@dataclass(frozen=True)
class Fragment:
    seq_name: str
    fragment_name: str
    record_index: int
    fragment_index: int
    start: int
    end: int
    length: int
    sequence: str


class FragmentDataset(Dataset):
    def __init__(self, fragments: Sequence[Fragment]):
        self.fragments = list(fragments)

    def __len__(self) -> int:
        return len(self.fragments)

    def __getitem__(self, index: int) -> Fragment:
        return self.fragments[index]


class TokenizeCollator:
    def __init__(self, tokenizer, model_max_length: int):
        self.tokenizer = tokenizer
        self.model_max_length = model_max_length

    def __call__(self, instances: Sequence[Fragment]) -> Dict[str, object]:
        sequences = [item.sequence for item in instances]
        tokenized = self.tokenizer(
            sequences,
            return_tensors="pt",
            padding="longest",
            max_length=self.model_max_length,
            truncation=True,
        )
        tokenized["meta"] = [
            {
                "seq_name": item.seq_name,
                "fragment_name": item.fragment_name,
                "record_index": item.record_index,
                "fragment_index": item.fragment_index,
                "start": item.start,
                "end": item.end,
                "length": item.length,
            }
            for item in instances
        ]
        return tokenized


class FlashCallCounter:
    def __init__(self) -> None:
        self.count = 0
        self.wrapped = []

    def wrap_transformers_modules(self) -> None:
        for module_name, module in list(sys.modules.items()):
            if "transformers_modules" not in module_name:
                continue
            if not module_name.endswith("bert_layers"):
                continue
            for attr_name in ("flash_attn_varlen_qkvpacked_func", "flash_attn_qkvpacked_func"):
                fn = getattr(module, attr_name, None)
                if fn is None or not callable(fn):
                    continue
                if getattr(fn, "_viralm_counter_wrapped", False):
                    continue

                def wrapped(*args, _fn=fn, **kwargs):
                    self.count += 1
                    return _fn(*args, **kwargs)

                wrapped._viralm_counter_wrapped = True  # type: ignore[attr-defined]
                setattr(module, attr_name, wrapped)
                self.wrapped.append(f"{module_name}.{attr_name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Shard-aware ViraLM inference")
    parser.add_argument("--input", "-i", required=True, help="Input FASTA or FASTA.GZ")
    parser.add_argument("--output", "-o", required=True, help="Output directory")
    parser.add_argument("--database", "-d", required=True, help="Model directory")
    parser.add_argument("--filename", "-n", default=None, help="Output name stem")
    parser.add_argument("--record-start", type=int, default=0, help="0-based inclusive FASTA record start")
    parser.add_argument("--record-end", type=int, default=None, help="0-based exclusive FASTA record end")
    parser.add_argument("--shard-id", default=None, help="Shard id used in output file suffix")
    parser.add_argument("--len", type=int, default=500, dest="min_len", help="Minimum sequence length")
    parser.add_argument("--fragment-len", type=int, default=2000, help="Internal fragment length")
    parser.add_argument("--min-tail-len", type=int, default=500, help="Minimum tail fragment length")
    parser.add_argument("--threshold", type=float, default=0.5, help="Virus score threshold")
    parser.add_argument("--batch_size", type=int, default=256, help="Inference batch size")
    parser.add_argument("--dataloader_workers", type=int, default=4, help="DataLoader worker processes")
    parser.add_argument("--prefetch_factor", type=int, default=4, help="Prefetch factor when workers > 0")
    parser.add_argument("--model-max-length", type=int, default=512, help="Tokenizer model_max_length")
    parser.add_argument("--infer-fp16", type=int, default=1, help="Use CUDA fp16 autocast, 1/0")
    parser.add_argument("--mil-sub-chunk-size", type=int, default=16, help="MIL fragment sub-chunk size")
    parser.add_argument("--mil-fast-path", type=int, default=1, help="Use flat-batch MIL inference, 1/0")
    parser.add_argument(
        "--mil-backbone-batch-size",
        type=int,
        default=0,
        help="Max flattened fragments per MIL backbone forward, 0=all fragments in sequence batch",
    )
    parser.add_argument("--force", "-f", action="store_true", help="Overwrite output directory")
    parser.add_argument("--keep-cache", action="store_true", help="Accepted for compatibility; no cache is written")
    parser.add_argument("--require-flash-attn", action="store_true", help="Require FlashAttention calls during warmup")
    parser.add_argument("--warmup-batches", type=int, default=1, help="Warmup batches before recorded inference")
    parser.add_argument("--threads", type=int, default=1, help="CPU torch threads")
    return parser.parse_args()


def open_text(path: str):
    if path.endswith(".gz"):
        return gzip.open(path, "rt")
    return open(path, "r")


def default_filename(input_path: str) -> str:
    name = Path(input_path).name
    if name.endswith(".fasta.gz"):
        return name[: -len(".fasta.gz")]
    if name.endswith(".fa.gz"):
        return name[: -len(".fa.gz")]
    return Path(name).stem


def is_valid_dna(sequence: str) -> bool:
    return VALID_DNA_RE.search(sequence) is None


def iter_record_fragments(
    input_path: str,
    record_start: int,
    record_end: Optional[int],
    min_len: int,
    fragment_len: int,
    min_tail_len: int,
) -> Tuple[List[Fragment], Dict[str, int]]:
    fragments: List[Fragment] = []
    stats = {
        "records_seen": 0,
        "records_in_range": 0,
        "records_too_short": 0,
        "records_without_valid_fragments": 0,
        "fragments_total": 0,
        "fragments_invalid": 0,
    }

    with open_text(input_path) as handle:
        for record_index, record in enumerate(SeqIO.parse(handle, "fasta")):
            stats["records_seen"] += 1
            if record_index < record_start:
                continue
            if record_end is not None and record_index >= record_end:
                break

            stats["records_in_range"] += 1
            seq_name = str(record.id)
            sequence = str(record.seq).upper()
            seq_len = len(sequence)

            if seq_len < min_len:
                stats["records_too_short"] += 1
                continue

            before = len(fragments)
            fragment_index = 0
            if seq_len >= fragment_len:
                last_pos = 0
                for start in range(0, seq_len - fragment_len + 1, fragment_len):
                    end = start + fragment_len
                    piece = sequence[start:end]
                    if is_valid_dna(piece):
                        fragments.append(
                            Fragment(
                                seq_name=seq_name,
                                fragment_name=f"{seq_name}_{start}_{end}",
                                record_index=record_index,
                                fragment_index=fragment_index,
                                start=start,
                                end=end,
                                length=len(piece),
                                sequence=piece,
                            )
                        )
                        fragment_index += 1
                    else:
                        stats["fragments_invalid"] += 1
                    last_pos = end
                if seq_len - last_pos >= min_tail_len:
                    piece = sequence[last_pos:]
                    if is_valid_dna(piece):
                        fragments.append(
                            Fragment(
                                seq_name=seq_name,
                                fragment_name=f"{seq_name}_{last_pos}_{seq_len}",
                                record_index=record_index,
                                fragment_index=fragment_index,
                                start=last_pos,
                                end=seq_len,
                                length=len(piece),
                                sequence=piece,
                            )
                        )
                    else:
                        stats["fragments_invalid"] += 1
            else:
                if is_valid_dna(sequence):
                    fragments.append(
                        Fragment(
                            seq_name=seq_name,
                            fragment_name=f"{seq_name}_0_{seq_len}",
                            record_index=record_index,
                            fragment_index=0,
                            start=0,
                            end=seq_len,
                            length=seq_len,
                            sequence=sequence,
                        )
                    )
                else:
                    stats["fragments_invalid"] += 1

            if len(fragments) == before:
                stats["records_without_valid_fragments"] += 1

    stats["fragments_total"] = len(fragments)
    return fragments, stats


def output_paths(output_dir: Path, filename: str, shard_id: Optional[str]) -> Dict[str, Path]:
    suffix = f".shard_{shard_id}" if shard_id is not None else ""
    return {
        "result": output_dir / f"result_{filename}{suffix}.csv",
        "fragment": output_dir / f"fragment_result_{filename}{suffix}.csv",
        "virus": output_dir / f"virus_{filename}{suffix}.fasta",
        "info": output_dir / f"run_info_{filename}{suffix}.json",
    }


def prepare_output_dir(output_dir: Path, force: bool) -> None:
    if output_dir.exists():
        if not output_dir.is_dir():
            raise SystemExit(f"output path exists and is not a directory: {output_dir}")
        if force:
            shutil.rmtree(output_dir)
            output_dir.mkdir(parents=True)
    else:
        output_dir.mkdir(parents=True)


def make_loader(
    fragments: Sequence[Fragment],
    tokenizer,
    batch_size: int,
    dataloader_workers: int,
    prefetch_factor: int,
    model_max_length: int,
    use_cuda: bool,
) -> DataLoader:
    collator = TokenizeCollator(tokenizer=tokenizer, model_max_length=model_max_length)
    kwargs = {
        "batch_size": batch_size,
        "shuffle": False,
        "collate_fn": collator,
        "num_workers": max(0, dataloader_workers),
        "pin_memory": bool(use_cuda),
    }
    if kwargs["num_workers"] > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = max(1, prefetch_factor)
    return DataLoader(FragmentDataset(fragments), **kwargs)


def predict_batch(model, batch: Dict[str, object], device: torch.device, use_fp16: bool):
    meta = batch["meta"]
    tensor_batch = {
        key: value.to(device, non_blocking=True)
        for key, value in batch.items()
        if key != "meta" and torch.is_tensor(value)
    }
    if device.type == "cuda" and use_fp16:
        context = torch.cuda.amp.autocast(dtype=torch.float16)
    else:
        context = nullcontext()
    with context:
        outputs = model(**tensor_batch)
    logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
    probs = torch.softmax(logits.float(), dim=-1)[:, 1].detach().cpu().tolist()
    return meta, probs


def is_mil_checkpoint(model_dir: str) -> bool:
    return (Path(model_dir) / "best_mil_model.pt").is_file()


def load_mil_state_dict(checkpoint_path: Path) -> Dict[str, torch.Tensor]:
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    if not isinstance(state_dict, dict):
        raise SystemExit(f"MIL checkpoint does not contain a state dict: {checkpoint_path}")

    for prefix in ("module.", "_orig_mod.", "model."):
        if any(str(key).startswith(prefix) for key in state_dict.keys()):
            state_dict = {
                str(key).removeprefix(prefix): value for key, value in state_dict.items()
            }
    return state_dict


def get_mil_layer_count(state_dict: Dict[str, torch.Tensor]) -> int:
    layers = set()
    prefix = "backbone.bert.encoder.layer."
    for key in state_dict.keys():
        key = str(key)
        if not key.startswith(prefix):
            continue
        try:
            layers.add(int(key[len(prefix) :].split(".", 1)[0]))
        except ValueError:
            pass
    return len(layers)


def load_mil_model(model_dir: str, device: torch.device):
    gamil_root = Path(os.environ.get("GAMIL_ROOT", Path(__file__).resolve().parents[2])).resolve()
    mil_code_dir = Path(
        os.environ.get("GAMIL_MODEL_CODE_DIR", gamil_root / "model" / "code" / "realm_rank_v4_six_model")
    ).resolve()
    if str(mil_code_dir) not in sys.path:
        sys.path.insert(0, str(mil_code_dir))
    from mil_model import ViraLM_MIL_Gated

    checkpoint_path = Path(model_dir) / "best_mil_model.pt"
    state_dict = load_mil_state_dict(checkpoint_path)
    layer_count = get_mil_layer_count(state_dict)
    if layer_count <= 0:
        raise SystemExit(f"Could not infer backbone layer count from {checkpoint_path}")

    config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    config.num_hidden_layers = layer_count
    config.num_labels = 2
    backbone = AutoModelForSequenceClassification.from_config(config, trust_remote_code=True)
    model = ViraLM_MIL_Gated(
        backbone,
        hidden_size=int(getattr(config, "hidden_size", 768)),
        num_classes=2,
    )
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise SystemExit(
            "Failed to load MIL checkpoint strictly: "
            f"missing={list(missing)[:10]} unexpected={list(unexpected)[:10]}"
        )

    model.to(device)
    model.eval()
    return model, {"mil_checkpoint_path": str(checkpoint_path), "mil_layer_count": layer_count}


def group_fragments_by_sequence(
    fragments: Sequence[Fragment],
) -> List[Tuple[str, List[Fragment]]]:
    groups: "OrderedDict[str, List[Fragment]]" = OrderedDict()
    for fragment in fragments:
        groups.setdefault(fragment.seq_name, []).append(fragment)
    return list(groups.items())


def iter_group_batches(
    groups: Sequence[Tuple[str, List[Fragment]]],
    batch_size: int,
) -> Iterable[Sequence[Tuple[str, List[Fragment]]]]:
    batch_size = max(1, int(batch_size))
    for start in range(0, len(groups), batch_size):
        yield groups[start : start + batch_size]


def tokenize_mil_fragments(
    tokenizer,
    fragments: Sequence[Fragment],
    model_max_length: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    encoded = tokenizer(
        [fragment.sequence for fragment in fragments],
        return_tensors="pt",
        padding="longest",
        max_length=model_max_length,
        truncation=True,
    )
    return encoded["input_ids"], encoded["attention_mask"]


def logits_to_virus_probs(logits: torch.Tensor) -> torch.Tensor:
    if logits.shape[-1] == 2:
        return torch.softmax(logits.float(), dim=-1)[:, 1]
    return torch.sigmoid(logits.float().squeeze(-1))


def extract_cls_hidden(outputs) -> torch.Tensor:
    hidden_states = outputs.hidden_states
    if isinstance(hidden_states, (tuple, list)):
        return hidden_states[-1][:, 0, :]
    return hidden_states[:, 0, :]


def run_mil_backbone_flat(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    device: torch.device,
    mil_backbone_batch_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    chunk_size = int(mil_backbone_batch_size)
    if chunk_size <= 0:
        chunk_size = int(input_ids.size(0))

    hidden_chunks = []
    logit_chunks = []
    for start in range(0, int(input_ids.size(0)), chunk_size):
        end = start + chunk_size
        chunk_ids = input_ids[start:end].to(device, non_blocking=True)
        chunk_mask = attention_mask[start:end].to(device, non_blocking=True)
        outputs = model.backbone(
            input_ids=chunk_ids,
            attention_mask=chunk_mask,
            output_hidden_states=True,
        )
        hidden_chunks.append(extract_cls_hidden(outputs))
        logit_chunks.append(outputs.logits)
    return torch.cat(hidden_chunks, dim=0), torch.cat(logit_chunks, dim=0)


def predict_mil_group_batch_fast(
    model,
    tokenizer,
    group_batch: Sequence[Tuple[str, List[Fragment]]],
    device: torch.device,
    use_fp16: bool,
    model_max_length: int,
    mil_backbone_batch_size: int,
    return_frag_logits: bool,
):
    flat_sequences = []
    group_sizes = []
    for _, fragments in group_batch:
        group_sizes.append(len(fragments))
        flat_sequences.extend(fragment.sequence for fragment in fragments)

    encoded = tokenizer(
        flat_sequences,
        return_tensors="pt",
        padding="longest",
        max_length=model_max_length,
        truncation=True,
    )

    if device.type == "cuda" and use_fp16:
        context = torch.cuda.amp.autocast(dtype=torch.float16)
    else:
        context = nullcontext()
    with context:
        hidden, frag_logits = run_mil_backbone_flat(
            model=model,
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            device=device,
            mil_backbone_batch_size=mil_backbone_batch_size,
        )

        seq_logits = []
        frag_logits_grouped = []
        offset = 0
        for group_size in group_sizes:
            next_offset = offset + group_size
            h_i = hidden[offset:next_offset]
            z_i = frag_logits[offset:next_offset]
            a_v = torch.tanh(model.attention_V(h_i))
            a_u = torch.sigmoid(model.attention_U(h_i))
            e_i = model.attention_w(a_v * a_u)
            a_i = torch.softmax(e_i, dim=0)
            h_seq = torch.mm(a_i.t(), h_i)
            seq_logits.append(model.seq_classifier(h_seq))
            if return_frag_logits:
                frag_logits_grouped.append(z_i.detach().cpu())
            offset = next_offset

        seq_logits = torch.cat(seq_logits, dim=0)

    seq_probs = logits_to_virus_probs(seq_logits).detach().cpu().tolist()
    if not return_frag_logits:
        return seq_probs, None

    frag_probs_grouped = [
        logits_to_virus_probs(logits).detach().cpu().tolist()
        for logits in frag_logits_grouped
    ]
    return seq_probs, frag_probs_grouped


def predict_mil_group_batch(
    model,
    tokenizer,
    group_batch: Sequence[Tuple[str, List[Fragment]]],
    device: torch.device,
    use_fp16: bool,
    model_max_length: int,
    mil_sub_chunk_size: int,
    mil_fast_path: bool,
    mil_backbone_batch_size: int,
    return_frag_logits: bool,
):
    if mil_fast_path:
        return predict_mil_group_batch_fast(
            model=model,
            tokenizer=tokenizer,
            group_batch=group_batch,
            device=device,
            use_fp16=use_fp16,
            model_max_length=model_max_length,
            mil_backbone_batch_size=mil_backbone_batch_size,
            return_frag_logits=return_frag_logits,
        )

    input_ids_list = []
    attention_mask_list = []
    for _, fragments in group_batch:
        input_ids, attention_mask = tokenize_mil_fragments(
            tokenizer=tokenizer,
            fragments=fragments,
            model_max_length=model_max_length,
        )
        input_ids_list.append(input_ids)
        attention_mask_list.append(attention_mask)

    if device.type == "cuda" and use_fp16:
        context = torch.cuda.amp.autocast(dtype=torch.float16)
    else:
        context = nullcontext()
    with context:
        seq_logits, _, frag_logits_grouped, _ = model(
            input_ids_list,
            attention_mask_list,
            sub_chunk_size=max(1, int(mil_sub_chunk_size)),
            return_frag_logits=return_frag_logits,
        )

    seq_probs = logits_to_virus_probs(seq_logits).detach().cpu().tolist()
    if not return_frag_logits:
        return seq_probs, None

    frag_probs_grouped = [
        logits_to_virus_probs(logits).detach().cpu().tolist()
        for logits in frag_logits_grouped
    ]
    return seq_probs, frag_probs_grouped


def run_mil_warmup(
    model,
    tokenizer,
    groups: Sequence[Tuple[str, List[Fragment]]],
    device: torch.device,
    use_fp16: bool,
    batch_size: int,
    model_max_length: int,
    mil_sub_chunk_size: int,
    mil_fast_path: bool,
    mil_backbone_batch_size: int,
    warmup_batches: int,
) -> int:
    batches = 0
    if warmup_batches <= 0:
        return 0
    with torch.inference_mode():
        for group_batch in iter_group_batches(groups, batch_size):
            predict_mil_group_batch(
                model=model,
                tokenizer=tokenizer,
                group_batch=group_batch,
                device=device,
                use_fp16=use_fp16,
                model_max_length=model_max_length,
                mil_sub_chunk_size=mil_sub_chunk_size,
                mil_fast_path=mil_fast_path,
                mil_backbone_batch_size=mil_backbone_batch_size,
                return_frag_logits=False,
            )
            batches += 1
            if batches >= warmup_batches:
                break
    if device.type == "cuda":
        torch.cuda.synchronize()
    return batches


def predict_mil_groups(
    model,
    tokenizer,
    groups: Sequence[Tuple[str, List[Fragment]]],
    device: torch.device,
    use_fp16: bool,
    batch_size: int,
    model_max_length: int,
    mil_sub_chunk_size: int,
    mil_fast_path: bool,
    mil_backbone_batch_size: int,
    threshold: float,
    flash_counter: FlashCallCounter,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    sequence_rows: List[Dict[str, object]] = []
    fragment_rows: List[Dict[str, object]] = []
    groups_done = 0

    with torch.inference_mode():
        for batch_index, group_batch in enumerate(iter_group_batches(groups, batch_size)):
            seq_probs, frag_probs_grouped = predict_mil_group_batch(
                model=model,
                tokenizer=tokenizer,
                group_batch=group_batch,
                device=device,
                use_fp16=use_fp16,
                model_max_length=model_max_length,
                mil_sub_chunk_size=mil_sub_chunk_size,
                mil_fast_path=mil_fast_path,
                mil_backbone_batch_size=mil_backbone_batch_size,
                return_frag_logits=True,
            )
            assert frag_probs_grouped is not None

            for (seq_name, fragments), seq_score, frag_probs in zip(
                group_batch, seq_probs, frag_probs_grouped
            ):
                if len(frag_probs) != len(fragments):
                    raise RuntimeError(
                        f"MIL fragment logit count mismatch for {seq_name}: "
                        f"{len(frag_probs)} != {len(fragments)}"
                    )
                sequence_rows.append(
                    {
                        "seq_name": seq_name,
                        "prediction": "virus" if float(seq_score) > threshold else "non-virus",
                        "virus_score": float(seq_score),
                        "fragment_count": len(fragments),
                    }
                )
                for fragment, frag_score in zip(fragments, frag_probs):
                    fragment_rows.append(
                        {
                            "seq_name": fragment.seq_name,
                            "fragment_name": fragment.fragment_name,
                            "record_index": fragment.record_index,
                            "fragment_index": fragment.fragment_index,
                            "start": fragment.start,
                            "end": fragment.end,
                            "length": fragment.length,
                            "prediction": "virus"
                            if float(frag_score) > threshold
                            else "non-virus",
                            "virus_score": float(frag_score),
                        }
                    )
            groups_done += len(group_batch)
            if batch_index % 50 == 0:
                print(
                    json.dumps(
                        {
                            "event": "mil_batch_done",
                            "batch_index": batch_index,
                            "sequences_done": groups_done,
                            "sequences_total": len(groups),
                            "fragments_done": len(fragment_rows),
                            "flash_attn_calls": flash_counter.count,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

    if device.type == "cuda":
        torch.cuda.synchronize()
    return sequence_rows, fragment_rows


def run_warmup(
    model,
    loader: DataLoader,
    device: torch.device,
    use_fp16: bool,
    warmup_batches: int,
) -> int:
    batches = 0
    if warmup_batches <= 0:
        return 0
    with torch.inference_mode():
        for batch in loader:
            predict_batch(model, batch, device, use_fp16)
            batches += 1
            if batches >= warmup_batches:
                break
    if device.type == "cuda":
        torch.cuda.synchronize()
    return batches


def write_empty_outputs(paths: Dict[str, Path]) -> None:
    with open(paths["fragment"], "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "seq_name",
                "fragment_name",
                "record_index",
                "fragment_index",
                "start",
                "end",
                "length",
                "prediction",
                "virus_score",
            ]
        )
    with open(paths["result"], "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["seq_name", "prediction", "virus_score", "fragment_count"])
    paths["virus"].write_text("")


def write_outputs(
    paths: Dict[str, Path],
    fragment_rows: Sequence[Dict[str, object]],
    threshold: float,
) -> Dict[str, int]:
    seq_sum: "OrderedDict[str, float]" = OrderedDict()
    seq_count: "OrderedDict[str, int]" = OrderedDict()

    with open(paths["fragment"], "w", newline="") as handle:
        fieldnames = [
            "seq_name",
            "fragment_name",
            "record_index",
            "fragment_index",
            "start",
            "end",
            "length",
            "prediction",
            "virus_score",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in fragment_rows:
            seq_name = str(row["seq_name"])
            score = float(row["virus_score"])
            seq_sum[seq_name] = seq_sum.get(seq_name, 0.0) + score
            seq_count[seq_name] = seq_count.get(seq_name, 0) + 1
            writer.writerow(row)

    result_rows = []
    for seq_name, total in seq_sum.items():
        count = seq_count[seq_name]
        score = total / max(1, count)
        result_rows.append(
            {
                "seq_name": seq_name,
                "prediction": "virus" if score > threshold else "non-virus",
                "virus_score": score,
                "fragment_count": count,
            }
        )
    result_rows.sort(key=lambda row: float(row["virus_score"]), reverse=True)

    with open(paths["result"], "w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["seq_name", "prediction", "virus_score", "fragment_count"],
        )
        writer.writeheader()
        writer.writerows(result_rows)

    paths["virus"].write_text("")
    return {"sequence_results": len(result_rows), "fragment_results": len(fragment_rows)}


def write_mil_outputs(
    paths: Dict[str, Path],
    sequence_rows: Sequence[Dict[str, object]],
    fragment_rows: Sequence[Dict[str, object]],
) -> Dict[str, int]:
    with open(paths["fragment"], "w", newline="") as handle:
        fieldnames = [
            "seq_name",
            "fragment_name",
            "record_index",
            "fragment_index",
            "start",
            "end",
            "length",
            "prediction",
            "virus_score",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(fragment_rows)

    ordered_sequence_rows = sorted(
        sequence_rows, key=lambda row: float(row["virus_score"]), reverse=True
    )
    with open(paths["result"], "w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["seq_name", "prediction", "virus_score", "fragment_count"],
        )
        writer.writeheader()
        writer.writerows(ordered_sequence_rows)

    paths["virus"].write_text("")
    return {
        "sequence_results": len(ordered_sequence_rows),
        "fragment_results": len(fragment_rows),
    }


def main() -> None:
    args = parse_args()
    if args.threshold < 0.5:
        raise SystemExit("threshold must be >= 0.5")
    if args.record_start < 0:
        raise SystemExit("--record-start must be >= 0")
    if args.record_end is not None and args.record_end < args.record_start:
        raise SystemExit("--record-end must be >= --record-start")
    if not os.path.isdir(args.database):
        raise SystemExit(f"model directory is missing or unreadable: {args.database}")

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.set_num_threads(max(1, int(args.threads)))
    if torch.cuda.is_available() and hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = True

    filename = args.filename or default_filename(args.input)
    output_dir = Path(args.output)
    prepare_output_dir(output_dir, args.force)
    paths = output_paths(output_dir, filename, args.shard_id)

    start_time = time.time()
    fragments, prep_stats = iter_record_fragments(
        input_path=args.input,
        record_start=args.record_start,
        record_end=args.record_end,
        min_len=args.min_len,
        fragment_len=args.fragment_len,
        min_tail_len=args.min_tail_len,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        json.dumps(
            {
                "event": "prepared_fragments",
                "input": args.input,
                "record_start": args.record_start,
                "record_end": args.record_end,
                "fragments": len(fragments),
                "device": str(device),
            },
            sort_keys=True,
        ),
        flush=True,
    )

    if not fragments:
        write_empty_outputs(paths)
        info = dict(prep_stats)
        info.update(
            {
                "filename": filename,
                "shard_id": args.shard_id,
                "device": str(device),
                "flash_attn_calls": 0,
                "elapsed_sec": round(time.time() - start_time, 3),
            }
        )
        paths["info"].write_text(json.dumps(info, indent=2, sort_keys=True) + "\n")
        return

    tokenizer = AutoTokenizer.from_pretrained(
        args.database,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=True,
        trust_remote_code=True,
    )
    mil_checkpoint = is_mil_checkpoint(args.database)
    mil_model_info: Dict[str, object] = {}
    if mil_checkpoint:
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token or "[PAD]"
        model, mil_model_info = load_mil_model(args.database, device)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            args.database,
            num_labels=2,
            trust_remote_code=True,
        )
        model.to(device)
        model.eval()

    flash_counter = FlashCallCounter()
    flash_counter.wrap_transformers_modules()
    if args.require_flash_attn and not flash_counter.wrapped:
        raise SystemExit("FlashAttention was required, but no callable FlashAttention function was found in remote code")

    use_fp16 = bool(int(args.infer_fp16)) and device.type == "cuda"
    if mil_checkpoint:
        groups = group_fragments_by_sequence(fragments)
        warmup_done = run_mil_warmup(
            model=model,
            tokenizer=tokenizer,
            groups=groups,
            device=device,
            use_fp16=use_fp16,
            batch_size=max(1, int(args.batch_size)),
            model_max_length=max(1, int(args.model_max_length)),
            mil_sub_chunk_size=max(1, int(args.mil_sub_chunk_size)),
            mil_fast_path=bool(int(args.mil_fast_path)),
            mil_backbone_batch_size=max(0, int(args.mil_backbone_batch_size)),
            warmup_batches=max(0, int(args.warmup_batches)),
        )
        if args.require_flash_attn and flash_counter.count <= 0:
            raise SystemExit(
                "FlashAttention was required, but warmup completed without any FlashAttention calls"
            )

        sequence_rows, fragment_rows = predict_mil_groups(
            model=model,
            tokenizer=tokenizer,
            groups=groups,
            device=device,
            use_fp16=use_fp16,
            batch_size=max(1, int(args.batch_size)),
            model_max_length=max(1, int(args.model_max_length)),
            mil_sub_chunk_size=max(1, int(args.mil_sub_chunk_size)),
            mil_fast_path=bool(int(args.mil_fast_path)),
            mil_backbone_batch_size=max(0, int(args.mil_backbone_batch_size)),
            threshold=args.threshold,
            flash_counter=flash_counter,
        )
        output_stats = write_mil_outputs(paths, sequence_rows, fragment_rows)
    else:
        loader = make_loader(
            fragments=fragments,
            tokenizer=tokenizer,
            batch_size=max(1, int(args.batch_size)),
            dataloader_workers=max(0, int(args.dataloader_workers)),
            prefetch_factor=max(1, int(args.prefetch_factor)),
            model_max_length=max(1, int(args.model_max_length)),
            use_cuda=device.type == "cuda",
        )

        warmup_done = run_warmup(
            model=model,
            loader=loader,
            device=device,
            use_fp16=use_fp16,
            warmup_batches=max(0, int(args.warmup_batches)),
        )
        if args.require_flash_attn and flash_counter.count <= 0:
            raise SystemExit(
                "FlashAttention was required, but warmup completed without any FlashAttention calls"
            )

        fragment_rows: List[Dict[str, object]] = []
        with torch.inference_mode():
            for batch_index, batch in enumerate(loader):
                meta_rows, probs = predict_batch(model, batch, device, use_fp16)
                for meta, score in zip(meta_rows, probs):
                    prediction = "virus" if float(score) > args.threshold else "non-virus"
                    row = dict(meta)
                    row["prediction"] = prediction
                    row["virus_score"] = float(score)
                    fragment_rows.append(row)
                if batch_index % 50 == 0:
                    print(
                        json.dumps(
                            {
                                "event": "batch_done",
                                "batch_index": batch_index,
                                "fragments_done": len(fragment_rows),
                                "fragments_total": len(fragments),
                                "flash_attn_calls": flash_counter.count,
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )

        if device.type == "cuda":
            torch.cuda.synchronize()

        output_stats = write_outputs(paths, fragment_rows, args.threshold)
    info = dict(prep_stats)
    info.update(output_stats)
    info.update(
        {
            "filename": filename,
            "shard_id": args.shard_id,
            "input": args.input,
            "model": args.database,
            "inference_mode": "gated_mil" if mil_checkpoint else "fragment_mean",
            "mil_checkpoint": mil_checkpoint,
            "mil_sub_chunk_size": int(args.mil_sub_chunk_size) if mil_checkpoint else None,
            "mil_fast_path": bool(int(args.mil_fast_path)) if mil_checkpoint else None,
            "mil_backbone_batch_size": max(0, int(args.mil_backbone_batch_size))
            if mil_checkpoint
            else None,
            "record_start": args.record_start,
            "record_end": args.record_end,
            "batch_size": int(args.batch_size),
            "dataloader_workers": int(args.dataloader_workers),
            "model_max_length": int(args.model_max_length),
            "device": str(device),
            "use_fp16": use_fp16,
            "warmup_batches": warmup_done,
            "flash_attn_wrapped": flash_counter.wrapped,
            "flash_attn_calls": flash_counter.count,
            "elapsed_sec": round(time.time() - start_time, 3),
        }
    )
    info.update(mil_model_info)
    paths["info"].write_text(json.dumps(info, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"event": "finished", **info}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
