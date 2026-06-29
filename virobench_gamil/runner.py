from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

from .heads import GatedMILHead, MultiTaskMLP
from .metrics import compute_multiclass_metrics, summarize_task_metrics
from .posthoc import evaluate_window_posthoc


TASK_LABELS = {
    "taxon": ["kingdom", "phylum", "class", "order", "family"],
    "host": ["host_label"],
}


def write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def parse_dataset_name(dataset_name: str) -> Tuple[str, str, str]:
    parts = dataset_name.split("-")
    if len(parts) != 3:
        raise ValueError("dataset_name must be {ALL|DNA|RNA}-{taxon|host}-{genus|times}")
    na_type, label_kind, split_kind = parts
    if na_type not in {"ALL", "DNA", "RNA"}:
        raise ValueError(f"invalid NA type: {na_type}")
    if label_kind not in TASK_LABELS:
        raise ValueError(f"invalid label kind: {label_kind}")
    if split_kind not in {"genus", "times"}:
        raise ValueError(f"invalid split kind: {split_kind}")
    return na_type, label_kind, split_kind


def setup_virobench_imports(virobench_root: Path) -> None:
    root = str(virobench_root.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)


def load_virobench_splits(
    virobench_root: Path,
    dataset_name: str,
    window_len: int,
    train_num_windows: int,
    eval_num_windows: int,
    use_small_dataset: bool,
    seed: int = 42,
):
    # ViroBench's datasets/__init__.py may import optional files that are not
    # present in the released source snapshot, so load virus_datasets.py
    # directly instead of importing the package.
    module_path = virobench_root / "datasets" / "virus_datasets.py"
    spec = importlib.util.spec_from_file_location("virobench_virus_datasets", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load ViroBench dataset module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    VirusSplitDatasets = module.VirusSplitDatasets

    na_type, label_kind, split_kind = parse_dataset_name(dataset_name)
    labels = TASK_LABELS[label_kind]
    cls_root = "cls_data_min_consistent" if use_small_dataset else "cls_data"
    split_dir = virobench_root / "data" / "all_viral" / cls_root / na_type / label_kind / split_kind
    if not split_dir.exists():
        raise FileNotFoundError(
            f"ViroBench split not found: {split_dir}. "
            "Download/unpack ViroBench data under external/ViroBench/data/all_viral first."
        )
    base = VirusSplitDatasets(
        split_dir,
        label_cols=labels,
        return_format="dict",
        attach_sequences=True,
    )
    win = base.make_windowed(
        window_len=window_len,
        train_num_windows=train_num_windows,
        eval_num_windows=eval_num_windows,
        seed=seed,
        return_format="dict",
    )
    task_dims = {name: len(base.label2id[name]) for name in labels}
    return labels, task_dims, base, win


def limit_window_dataset(dataset, max_sequences: int):
    if max_sequences is None or int(max_sequences) <= 0:
        return dataset
    max_sequences = int(max_sequences)
    if hasattr(dataset, "_flat_index"):
        kept = set()
        flat = []
        for seq_index, window_index in dataset._flat_index:
            seq_index = int(seq_index)
            if seq_index in kept or len(kept) < max_sequences:
                kept.add(seq_index)
                flat.append((seq_index, int(window_index)))
        dataset._flat_index = flat
        return dataset
    return dataset


def limit_windowed_splits(win, max_train_sequences: int, max_val_sequences: int, max_test_sequences: int):
    win.train = limit_window_dataset(win.train, max_train_sequences)
    win.val = limit_window_dataset(win.val, max_val_sequences)
    win.test = limit_window_dataset(win.test, max_test_sequences)
    return win


def load_embedder(
    virobench_root: Path,
    model_name: str,
    model_dir: Optional[str],
    device: str,
):
    setup_virobench_imports(virobench_root)
    parent = virobench_root.parent
    model_weight_root = parent / "model_weight"

    if model_name in {"DNABERT2-virobench", "DNABERT-2-117M", "dnabert2"}:
        from models import DNABERT2Model

        path = Path(model_dir) if model_dir else model_weight_root / "DNABERT-2-117M"
        if not path.exists():
            raise FileNotFoundError(
                f"DNABERT-2 model path not found: {path}. Pass --model-dir or place weights at {path}."
            )
        embedder = DNABERT2Model(
            model_name="DNABERT2-virobench",
            model_path=str(path),
            hf_home=str(model_weight_root / "cache"),
            device=device,
            use_mlm_head=False,
        )
        return embedder, {"hidden_size": 768, "emb_pool": "mean", "emb_layer_name": None, "emb_batch_size": 16}

    if model_name in {"OmniReg-GPT", "OmniReg-large", "omnireg-gpt"}:
        from models.omnireg_model import OmniRegGPTModel

        repo = Path(os.environ.get("OMNIREG_REPO_DIR", parent / "official" / "OmniReg-GPT"))
        assets = Path(os.environ.get("OMNIREG_ASSET_DIR", model_weight_root / "OmniReg-GPT"))
        if model_dir:
            candidate = Path(model_dir)
            if (candidate / "pytorch_model.bin").exists():
                assets = candidate
            elif (candidate / "OmniReg-GPT" / "pytorch_model.bin").exists():
                assets = candidate / "OmniReg-GPT"
            elif (candidate / "hybrid_transformer.py").exists():
                repo = candidate
        ckpt = assets / "pytorch_model.bin"
        tokenizer = assets / "gena-lm-bert-large-t2t"
        if not ckpt.exists():
            raise FileNotFoundError(f"OmniReg-GPT checkpoint not found: {ckpt}")
        if not tokenizer.exists():
            raise FileNotFoundError(f"OmniReg-GPT tokenizer not found: {tokenizer}")
        embedder = OmniRegGPTModel(
            model_name="OmniReg-GPT",
            model_path=str(ckpt),
            tokenizer_path=str(tokenizer),
            omnireg_repo_path=str(repo),
            hf_home=str(model_weight_root / "cache"),
            device=device,
            max_length=2048,
        )
        return embedder, {"hidden_size": 1024, "emb_pool": "mean", "emb_layer_name": None, "emb_batch_size": 4}

    if model_name in {"LucaVirus-default-step3.8M", "LucaVirus"}:
        from models.lucavirus import LucaVirusModel

        path = Path(model_dir) if model_dir else model_weight_root / "LucaVirus-default-step3.8M"
        legacy_path = model_weight_root / "LucaVirus-default-step3_8M"
        if model_dir is None and not path.exists() and legacy_path.exists():
            path = legacy_path
        if not path.exists():
            raise FileNotFoundError(f"LucaVirus model path not found: {path}. Pass --model-dir or download LucaGroup/LucaVirus-default-step3.8M.")
        embedder = LucaVirusModel(
            model_name="LucaVirus-default-step3.8M",
            model_path=str(path),
            device=device,
            torch_dtype="auto",
            force_download=False,
        )
        sample = embedder.get_embedding(["ACGT" * 128], batch_size=1, pool="mean", return_numpy=True)
        hidden_size = int(np.asarray(sample[0] if isinstance(sample, list) else sample).reshape(1, -1).shape[-1])
        return embedder, {"hidden_size": hidden_size, "emb_pool": "mean", "emb_layer_name": None, "emb_batch_size": 8}

    if model_name in {"ViroHyena-253m", "ViroHyena-253M"}:
        from models.hyenadna_local import HyenaDNALocal

        pretrain_root = virobench_root / "pretrain" / "hyena-dna"
        path = Path(model_dir) if model_dir else pretrain_root / "ViroHyena-253m"
        if not path.exists():
            raise FileNotFoundError(f"ViroHyena-253m model path not found: {path}. Pass --model-dir or download YDXX/ViroHyena-253m.")
        embedder = HyenaDNALocal(
            model_dir=str(path),
            device=device,
            pretrain_root=str(pretrain_root),
        )
        hidden_size = int(getattr(embedder, "d_model", 0) or 256)
        return embedder, {"hidden_size": hidden_size, "emb_pool": "final", "emb_layer_name": None, "emb_batch_size": 8}

    if model_name in {"DNABERT-S", "dnabert-s"}:
        from models import DNABERTSModel

        path = Path(model_dir) if model_dir else model_weight_root / "DNABERT-S"
        if not path.exists():
            raise FileNotFoundError(f"DNABERT-S model path not found: {path}. Pass --model-dir.")
        embedder = DNABERTSModel(
            model_name="DNABERT-S",
            model_path=str(path),
            hf_home=str(model_weight_root / "cache"),
            device=device,
        )
        return embedder, {"hidden_size": 768, "emb_pool": "mean", "emb_layer_name": None, "emb_batch_size": 16}

    raise ValueError(
        f"Unsupported model_name={model_name}. This extension supports DNABERT2-virobench, "
        "DNABERT-S, OmniReg-GPT, LucaVirus-default-step3.8M, and ViroHyena-253m."
    )


def normalize_embeddings(embs: Any, pool: str) -> np.ndarray:
    if isinstance(embs, torch.Tensor):
        arr = embs.detach().cpu().numpy()
    elif isinstance(embs, list):
        arrs = []
        for item in embs:
            a = np.asarray(item)
            if a.ndim == 1:
                a = a.reshape(1, -1)
            elif a.ndim == 3:
                if pool == "mean":
                    a = a.mean(axis=1)
                elif pool in {"cls", "final"}:
                    a = a[:, 0 if pool == "cls" else -1, :]
                else:
                    raise ValueError(f"unsupported pool for 3D embedding: {pool}")
            arrs.append(a)
        arr = np.concatenate(arrs, axis=0)
    else:
        arr = np.asarray(embs)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim == 3:
        if pool == "mean":
            arr = arr.mean(axis=1)
        elif pool in {"cls", "final"}:
            arr = arr[:, 0 if pool == "cls" else -1, :]
    if arr.ndim != 2:
        raise ValueError(f"expected 2D embeddings, got {arr.shape}")
    return arr.astype(np.float32, copy=False)


def labels_to_array(label_value: Any, num_tasks: int) -> np.ndarray:
    if isinstance(label_value, np.ndarray):
        arr = label_value.astype(np.int64).reshape(-1)
    elif isinstance(label_value, (list, tuple)):
        arr = np.asarray(label_value, dtype=np.int64).reshape(-1)
    else:
        arr = np.asarray([int(label_value)], dtype=np.int64)
    if arr.size != num_tasks:
        raise ValueError(f"expected {num_tasks} labels, got {arr}")
    return arr


def extract_embeddings(
    dataset,
    embedder,
    split: str,
    cache_dir: Path,
    task_names: Sequence[str],
    emb_pool: str,
    emb_batch_size: int,
    emb_layer_name: Optional[str],
    force: bool,
) -> Dict[str, torch.Tensor]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{split}_window_embeddings.pt"
    if cache_path.exists() and not force:
        return torch.load(cache_path, map_location="cpu")

    feats: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    groups: List[int] = []
    window_indices: List[int] = []
    num_windows: List[int] = []
    taxids: List[Any] = []
    seqs: List[str] = []
    meta: List[Dict[str, Any]] = []

    def flush() -> None:
        if not seqs:
            return
        kwargs = {"batch_size": emb_batch_size, "pool": emb_pool, "return_numpy": True}
        if emb_layer_name is not None:
            kwargs["layer_name"] = emb_layer_name
        emb = embedder.get_embedding(seqs, **kwargs)
        feats.append(normalize_embeddings(emb, emb_pool))
        seqs.clear()

    for idx in range(len(dataset)):
        item = dataset[idx]
        seqs.append(str(item["sequence"]))
        labels.append(labels_to_array(item["labels"], len(task_names)))
        groups.append(int(item.get("seq_index", idx)))
        window_indices.append(int(item.get("window_index", 0)))
        num_windows.append(int(item.get("num_windows", 1)))
        taxids.append(item.get("taxid", ""))
        meta.append(
            {
                "taxid": item.get("taxid", ""),
                "seq_index": int(item.get("seq_index", idx)),
                "window_index": int(item.get("window_index", 0)),
                "num_windows": int(item.get("num_windows", 1)),
                "window_start": int(item.get("window_start", 0)),
            }
        )
        if len(seqs) >= emb_batch_size:
            flush()
    flush()

    out = {
        "feats": torch.from_numpy(np.concatenate(feats, axis=0)).float(),
        "labels": torch.from_numpy(np.vstack(labels)).long(),
        "groups": torch.tensor(groups, dtype=torch.long),
        "window_indices": torch.tensor(window_indices, dtype=torch.long),
        "num_windows": torch.tensor(num_windows, dtype=torch.long),
        "meta": meta,
    }
    torch.save(out, cache_path)
    return out


def build_class_weights(labels: torch.Tensor, task_dims: Mapping[str, int], task_names: Sequence[str]) -> Dict[str, torch.Tensor]:
    weights: Dict[str, torch.Tensor] = {}
    for ti, task in enumerate(task_names):
        y = labels[:, ti].numpy()
        valid = y[(y >= 0) & (y < int(task_dims[task]))]
        if valid.size == 0:
            continue
        counts = np.bincount(valid, minlength=int(task_dims[task])).astype(np.float64)
        counts[counts == 0] = 1.0
        total = counts.sum()
        w = total / (len(counts) * counts)
        weights[task] = torch.tensor(w, dtype=torch.float32)
    return weights


def multitask_loss(
    logits_by_task: Mapping[str, torch.Tensor],
    labels: torch.Tensor,
    task_names: Sequence[str],
    class_weights: Optional[Mapping[str, torch.Tensor]] = None,
) -> torch.Tensor:
    losses = []
    for ti, task in enumerate(task_names):
        logits = logits_by_task[task]
        y = labels[:, ti]
        valid = (y >= 0) & (y < logits.size(1))
        if valid.any():
            weight = None
            if class_weights and task in class_weights:
                weight = class_weights[task].to(logits.device)
            losses.append(nn.CrossEntropyLoss(weight=weight)(logits[valid], y[valid]))
    if not losses:
        return torch.zeros((), device=next(iter(logits_by_task.values())).device)
    return sum(losses) / len(losses)


def logits_to_numpy(logits_by_task: Mapping[str, List[np.ndarray] | torch.Tensor]) -> Dict[str, np.ndarray]:
    out = {}
    for task, values in logits_by_task.items():
        if isinstance(values, torch.Tensor):
            out[task] = values.detach().cpu().numpy()
        else:
            out[task] = np.concatenate(values, axis=0)
    return out


def evaluate_logits(
    logits_by_task: Mapping[str, np.ndarray],
    labels: np.ndarray,
    task_names: Sequence[str],
    task_dims: Mapping[str, int],
) -> Dict[str, Any]:
    by_task = {}
    for ti, task in enumerate(task_names):
        by_task[task] = compute_multiclass_metrics(
            logits_by_task[task],
            labels[:, ti],
            num_classes=int(task_dims[task]),
        )
    return summarize_task_metrics(by_task)


def macro_f1_from_logits(
    logits_by_task: Mapping[str, np.ndarray],
    labels: np.ndarray,
    task_names: Sequence[str],
    task_dims: Mapping[str, int],
) -> float:
    vals: List[float] = []
    for ti, task in enumerate(task_names):
        logits = logits_by_task[task]
        y = labels[:, ti]
        num_classes = int(task_dims[task])
        valid = (y >= 0) & (y < num_classes)
        if not np.any(valid):
            continue
        yv = y[valid]
        pred = logits[valid].argmax(axis=1)
        classes = np.union1d(np.unique(yv), np.unique(pred))
        per_class = []
        for cls in classes:
            tp = np.sum((pred == cls) & (yv == cls))
            fp = np.sum((pred == cls) & (yv != cls))
            fn = np.sum((pred != cls) & (yv == cls))
            denom = (2 * tp + fp + fn)
            per_class.append(float(2 * tp / denom) if denom > 0 else 0.0)
        vals.append(float(np.mean(per_class)) if per_class else 0.0)
    return float(np.mean(vals)) if vals else -1.0


def train_window_mlp(
    train: Dict[str, torch.Tensor],
    val: Dict[str, torch.Tensor],
    test: Dict[str, torch.Tensor],
    task_names: Sequence[str],
    task_dims: Mapping[str, int],
    output_dir: Path,
    epochs: int,
    batch_size: int,
    lr: float,
    patience: int,
    seed: int,
    device: str,
) -> Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, Any]]:
    torch.manual_seed(seed)
    model = MultiTaskMLP(train["feats"].shape[1], task_dims).to(device)
    weights = build_class_weights(train["labels"], task_dims, task_names)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    train_loader = DataLoader(TensorDataset(train["feats"], train["labels"]), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(val["feats"], val["labels"]), batch_size=batch_size)
    test_loader = DataLoader(TensorDataset(test["feats"], test["labels"]), batch_size=batch_size)

    best_state = None
    best_f1 = -1.0
    bad = 0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        seen = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = multitask_loss(logits, yb, task_names, weights)
            loss.backward()
            opt.step()
            total += float(loss.detach().cpu()) * xb.size(0)
            seen += xb.size(0)
        val_logits = predict_window_model(model, val_loader, task_names, device)
        val_f1 = macro_f1_from_logits(val_logits, val["labels"].numpy(), task_names, task_dims)
        history.append({"epoch": epoch, "train_loss": total / max(seen, 1), "val_f1_macro": val_f1})
        if val_f1 > best_f1:
            best_f1 = val_f1
            bad = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if bad >= patience:
            break
    if best_state:
        model.load_state_dict(best_state)
    logits = {
        "train": predict_window_model(model, train_loader, task_names, device),
        "val": predict_window_model(model, val_loader, task_names, device),
        "test": predict_window_model(model, test_loader, task_names, device),
    }
    torch.save(best_state or model.state_dict(), output_dir / "window_mlp_best.pt")
    write_json(output_dir / "window_mlp_history.json", {"history": history, "best_val_f1_macro": best_f1})
    return logits, {"history": history, "best_val_f1_macro": best_f1}


def predict_window_model(model: nn.Module, loader: DataLoader, task_names: Sequence[str], device: str) -> Dict[str, np.ndarray]:
    model.eval()
    buckets = {task: [] for task in task_names}
    with torch.inference_mode():
        for xb, _ in loader:
            xb = xb.to(device)
            logits = model(xb)
            for task in task_names:
                buckets[task].append(logits[task].detach().cpu().numpy())
    return {task: np.concatenate(parts, axis=0) for task, parts in buckets.items()}


class BagDataset(Dataset):
    def __init__(self, data: Dict[str, torch.Tensor]):
        groups = data["groups"].numpy()
        self.feats = data["feats"]
        self.labels = data["labels"]
        order = np.argsort(groups, kind="mergesort")
        sorted_groups = groups[order]
        self.unique_groups, start = np.unique(sorted_groups, return_index=True)
        end = np.append(start[1:], len(sorted_groups))
        self.indices = [order[a:b] for a, b in zip(start, end)]

    def __len__(self) -> int:
        return len(self.unique_groups)

    def __getitem__(self, idx: int):
        inds = self.indices[idx]
        first = int(inds[0])
        return self.feats[inds], self.labels[first], int(self.unique_groups[idx])


def bag_collate(batch):
    max_len = max(x[0].size(0) for x in batch)
    dim = batch[0][0].size(1)
    feats = torch.zeros(len(batch), max_len, dim, dtype=torch.float32)
    mask = torch.zeros(len(batch), max_len, dtype=torch.bool)
    labels = []
    groups = []
    for i, (x, y, g) in enumerate(batch):
        n = x.size(0)
        feats[i, :n] = x
        mask[i, :n] = True
        labels.append(y)
        groups.append(g)
    return feats, mask, torch.stack(labels).long(), torch.tensor(groups, dtype=torch.long)


def train_bag_head(
    method: str,
    train: Dict[str, torch.Tensor],
    val: Dict[str, torch.Tensor],
    test: Dict[str, torch.Tensor],
    task_names: Sequence[str],
    task_dims: Mapping[str, int],
    output_dir: Path,
    epochs: int,
    batch_size: int,
    lr: float,
    patience: int,
    seed: int,
    device: str,
    attention_dim: int,
) -> Dict[str, Any]:
    torch.manual_seed(seed)
    if method == "GAMIL":
        model = GatedMILHead(train["feats"].shape[1], task_dims, attention_dim=attention_dim).to(device)
    else:
        raise ValueError(method)
    weights = build_class_weights(train["labels"], task_dims, task_names)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    loaders = {
        "train": DataLoader(BagDataset(train), batch_size=batch_size, shuffle=True, collate_fn=bag_collate),
        "val": DataLoader(BagDataset(val), batch_size=batch_size, shuffle=False, collate_fn=bag_collate),
        "test": DataLoader(BagDataset(test), batch_size=batch_size, shuffle=False, collate_fn=bag_collate),
    }

    best_state = None
    best_f1 = -1.0
    bad = 0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        seen = 0
        for xb, mask, yb, _ in loaders["train"]:
            xb, mask, yb = xb.to(device), mask.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(xb, mask)
            loss = multitask_loss(logits, yb, task_names, weights)
            loss.backward()
            opt.step()
            total += float(loss.detach().cpu()) * xb.size(0)
            seen += xb.size(0)
        val_pred = predict_bag_model(model, loaders["val"], task_names, device)
        val_f1 = macro_f1_from_logits(val_pred["logits"], val_pred["labels"], task_names, task_dims)
        history.append({"epoch": epoch, "train_loss": total / max(seen, 1), "val_f1_macro": val_f1})
        if val_f1 > best_f1:
            best_f1 = val_f1
            bad = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if bad >= patience:
            break
    if best_state:
        model.load_state_dict(best_state)

    test_pred = predict_bag_model(model, loaders["test"], task_names, device, return_attention=(method == "GAMIL"))
    metrics = evaluate_logits(test_pred["logits"], test_pred["labels"], task_names, task_dims)
    torch.save(best_state or model.state_dict(), output_dir / f"{method.lower()}_best.pt")
    np.savez_compressed(
        output_dir / f"{method.lower()}_test_predictions.npz",
        labels=test_pred["labels"],
        groups=test_pred["groups"],
        **{f"logits_{task}": arr for task, arr in test_pred["logits"].items()},
    )
    if method == "GAMIL" and "attention" in test_pred:
        torch.save(test_pred["attention"], output_dir / "gamil_test_attention.pt")
    return {"metrics": metrics, "history": history, "best_val_f1_macro": best_f1}


def predict_bag_model(
    model: nn.Module,
    loader: DataLoader,
    task_names: Sequence[str],
    device: str,
    return_attention: bool = False,
) -> Dict[str, Any]:
    model.eval()
    logits_bucket = {task: [] for task in task_names}
    labels_bucket = []
    groups_bucket = []
    attention = {}
    with torch.inference_mode():
        for xb, mask, yb, gb in loader:
            xb, mask = xb.to(device), mask.to(device)
            if return_attention:
                logits, weights = model(xb, mask, return_attention=True)
                for i, group in enumerate(gb.numpy().tolist()):
                    n = int(mask[i].sum().item())
                    attention[int(group)] = weights[i, :n].detach().cpu()
            else:
                logits = model(xb, mask)
            for task in task_names:
                logits_bucket[task].append(logits[task].detach().cpu().numpy())
            labels_bucket.append(yb.numpy())
            groups_bucket.append(gb.numpy())
    out = {
        "logits": {task: np.concatenate(parts, axis=0) for task, parts in logits_bucket.items()},
        "labels": np.concatenate(labels_bucket, axis=0),
        "groups": np.concatenate(groups_bucket, axis=0),
    }
    if return_attention:
        out["attention"] = attention
    return out


def run(args: argparse.Namespace) -> Dict[str, Any]:
    start = time.perf_counter()
    virobench_root = Path(args.virobench_root).resolve()
    experiment_root = Path(args.output_dir).resolve() / args.dataset_name / args.model_name / f"w{args.window_len}_train{args.train_num_windows}_eval{args.eval_num_windows}"
    output_dir = experiment_root / f"seed{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    if summary_path.exists() and not args.no_skip_existing:
        with summary_path.open("r", encoding="utf-8") as f:
            summary = json.load(f)
        print(f"[skip-existing] {summary_path}", flush=True)
        return summary
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    task_names, task_dims, base, win = load_virobench_splits(
        virobench_root,
        args.dataset_name,
        args.window_len,
        args.train_num_windows,
        args.eval_num_windows,
        args.use_small_dataset,
        seed=args.seed,
    )
    limit_windowed_splits(win, args.max_train_sequences, args.max_val_sequences, args.max_test_sequences)
    embedder, model_cfg = load_embedder(virobench_root, args.model_name, args.model_dir, device=device)
    emb_pool = args.emb_pool or model_cfg.get("emb_pool", "mean")
    emb_layer_name = args.emb_layer_name or model_cfg.get("emb_layer_name")
    emb_batch_size = args.emb_batch_size or int(model_cfg.get("emb_batch_size", 16))

    cap_tag = f"maxtrain{args.max_train_sequences}_maxval{args.max_val_sequences}_maxtest{args.max_test_sequences}"
    cache_dir = experiment_root / "embeddings" / cap_tag
    splits = {
        "train": extract_embeddings(win.train, embedder, "train", cache_dir, task_names, emb_pool, emb_batch_size, emb_layer_name, args.force_recompute_embeddings),
        "val": extract_embeddings(win.val, embedder, "val", cache_dir, task_names, emb_pool, emb_batch_size, emb_layer_name, args.force_recompute_embeddings),
        "test": extract_embeddings(win.test, embedder, "test", cache_dir, task_names, emb_pool, emb_batch_size, emb_layer_name, args.force_recompute_embeddings),
    }

    summary: Dict[str, Any] = {
        "dataset_name": args.dataset_name,
        "model_name": args.model_name,
        "task_names": list(task_names),
        "task_dims": {k: int(v) for k, v in task_dims.items()},
        "window_len": args.window_len,
        "train_num_windows": args.train_num_windows,
        "eval_num_windows": args.eval_num_windows,
        "max_train_sequences": args.max_train_sequences,
        "max_val_sequences": args.max_val_sequences,
        "max_test_sequences": args.max_test_sequences,
        "seed": args.seed,
        "device": device,
        "embedding_cache_dir": str(cache_dir),
        "methods": {},
    }

    window_logits, window_train_meta = train_window_mlp(
        splits["train"], splits["val"], splits["test"],
        task_names, task_dims, output_dir,
        args.epochs, args.window_head_batch_size, args.lr, args.patience, args.seed, device,
    )
    np.savez_compressed(
        output_dir / "window_logits_test.npz",
        labels=splits["test"]["labels"].numpy(),
        groups=splits["test"]["groups"].numpy(),
        **{f"logits_{task}": arr for task, arr in window_logits["test"].items()},
    )
    posthoc = evaluate_window_posthoc(
        window_logits["train"], window_logits["val"], window_logits["test"],
        splits["train"]["labels"].numpy(), splits["val"]["labels"].numpy(), splits["test"]["labels"].numpy(),
        splits["train"]["groups"].numpy(), splits["val"]["groups"].numpy(), splits["test"]["groups"].numpy(),
        task_names, task_dims,
        quantile=args.quantile,
        c_grid=[float(x) for x in args.logreg_c_grid.split(",") if x.strip()],
    )
    summary["methods"].update(posthoc)
    summary["window_mlp"] = window_train_meta

    if args.methods in {"all", "gamil"}:
        summary["methods"]["GAMIL"] = train_bag_head(
            "GAMIL", splits["train"], splits["val"], splits["test"],
            task_names, task_dims, output_dir,
            args.epochs, args.bag_batch_size, args.lr, args.patience, args.seed, device,
            args.attention_dim,
        )["metrics"]

    summary["elapsed_sec"] = time.perf_counter() - start
    if torch.cuda.is_available():
        summary["gpu"] = {
            "name": torch.cuda.get_device_name(0),
            "max_memory_allocated_mb": torch.cuda.max_memory_allocated() / (1024 ** 2),
        }
    write_json(summary_path, summary)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="ViroBench-GAMIL classification aggregation extension")
    p.add_argument("--virobench-root", default="external/ViroBench")
    p.add_argument("--dataset-name", required=True, help="Example: ALL-host-genus or ALL-taxon-times")
    p.add_argument("--model-name", default="DNABERT2-virobench")
    p.add_argument("--model-dir", default=None)
    p.add_argument("--output-dir", default="results/virobench_gamil")
    p.add_argument("--use-small-dataset", action="store_true")
    p.add_argument("--window-len", type=int, default=2048)
    p.add_argument("--train-num-windows", type=int, default=2)
    p.add_argument("--eval-num-windows", type=int, default=-1)
    p.add_argument("--max-train-sequences", type=int, default=0, help="Smoke-test cap: keep only the first N train sequences after windowing; 0 means all.")
    p.add_argument("--max-val-sequences", type=int, default=0, help="Smoke-test cap: keep only the first N validation sequences after windowing; 0 means all.")
    p.add_argument("--max-test-sequences", type=int, default=0, help="Smoke-test cap: keep only the first N test sequences after windowing; 0 means all.")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--window-head-batch-size", type=int, default=256)
    p.add_argument("--bag-batch-size", type=int, default=16)
    p.add_argument("--emb-batch-size", type=int, default=0)
    p.add_argument("--emb-pool", default=None)
    p.add_argument("--emb-layer-name", default=None)
    p.add_argument("--force-recompute-embeddings", action="store_true")
    p.add_argument("--attention-dim", type=int, default=256)
    p.add_argument("--quantile", type=float, default=0.95)
    p.add_argument("--logreg-c-grid", default="0.1,0.3,1.0,3.0")
    p.add_argument("--methods", choices=["all", "gamil", "posthoc"], default="all")
    p.add_argument("--no-skip-existing", action="store_true", help="Recompute even when summary.json already exists.")
    p.add_argument("--device", default=None)
    return p


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_arg_parser().parse_args(argv)
    summary = run(args)
    print(json.dumps(summary["methods"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
