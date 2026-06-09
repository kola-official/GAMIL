#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def iter_fasta(path):
    name = None
    chunks = []
    with open(path) as handle:
        for line in handle:
            line = line.rstrip("\n")
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    yield name, "".join(chunks)
                name = line[1:].split()[0]
                chunks = []
            else:
                chunks.append(line.strip())
    if name is not None:
        yield name, "".join(chunks)


def write_record(handle, name, seq, width=80):
    handle.write(f">{name}\n")
    for idx in range(0, len(seq), width):
        handle.write(seq[idx:idx + width] + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--positive", required=True)
    parser.add_argument("--negative", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--n-per-class", type=int, default=2500)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    combined_path = out_dir / "efficiency_10000bp_5000.fasta"
    pos_path = out_dir / "positive_2500.fasta"
    neg_path = out_dir / "negative_2500.fasta"
    labels_path = out_dir / "labels.tsv"
    meta_path = out_dir / "dataset_meta.json"

    counts = {"positive": 0, "negative": 0}
    lengths = {"positive": [], "negative": []}

    with open(combined_path, "w") as combined, open(pos_path, "w") as pos_out, open(neg_path, "w") as neg_out, open(labels_path, "w") as labels:
        labels.write("seq_id\tlabel\toriginal_id\tlength\n")

        for label, prefix, src_path, class_out in [
            ("positive", "pos", args.positive, pos_out),
            ("negative", "neg", args.negative, neg_out),
        ]:
            for idx, (old_id, seq) in enumerate(iter_fasta(src_path), start=1):
                if idx > args.n_per_class:
                    break
                new_id = f"{prefix}_{idx:06d}"
                write_record(combined, new_id, seq)
                write_record(class_out, new_id, seq)
                labels.write(f"{new_id}\t{label}\t{old_id}\t{len(seq)}\n")
                counts[label] += 1
                lengths[label].append(len(seq))

    for label, count in counts.items():
        if count != args.n_per_class:
            raise SystemExit(f"Expected {args.n_per_class} {label} records, got {count}")

    meta = {
        "source_positive": str(Path(args.positive).resolve()),
        "source_negative": str(Path(args.negative).resolve()),
        "combined_fasta": str(combined_path.resolve()),
        "positive_fasta": str(pos_path.resolve()),
        "negative_fasta": str(neg_path.resolve()),
        "labels_tsv": str(labels_path.resolve()),
        "n_per_class": args.n_per_class,
        "total_records": sum(counts.values()),
        "counts": counts,
        "length_summary": {
            label: {
                "min": min(values),
                "max": max(values),
                "mean": sum(values) / len(values),
            }
            for label, values in lengths.items()
        },
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
