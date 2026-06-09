# Realm-Rank v4 six-model experiment

This directory contains self-contained scripts for the Realm-Rank v4 six-model
distillation and gated-attention MIL experiment. Existing `Distill` and
`ViraLM-mil` sources are not modified.

## Full run

```bash
cd GAMIL/model/code/realm_rank_v4_six_model
./run_realm_rank_v4_six_model_tmux.sh
```

Defaults:

- tmux session: `realm_rank_v4_six_model`
- CUDA devices: `0,1`
- launcher: `torchrun --nproc_per_node 2`
- data: `${GAMIL_ROOT}/processed_data/realm_rank_v4/train_csv`
- test FASTA: `${GAMIL_ROOT}/processed_data/realm_rank_v4/test.fasta.gz`
- output root: `${GAMIL_ROOT}/checkpoint/local_checkpoints/realm_rank_v4_six_model`

Useful overrides:

```bash
CUDA_VISIBLE_DEVICES=0 NPROC_PER_NODE=1 ./run_realm_rank_v4_six_model_tmux.sh
SMOKE_TEST=1 OUTPUT_ROOT="${PWD}/../../../outputs/realm_rank_v4_six_model_smoke" RUN_IN_TMUX=0 NPROC_PER_NODE=1 ./run_realm_rank_v4_six_model_tmux.sh
```

## Trained models

The launcher runs these six models in sequence:

- `viralm_o_6l_meanpool_kd`
- `viralm_r_v4_final_6l_meanpool_kd`
- `viralm_o_6l_gated_mil_kd`
- `viralm_r_v4_final_6l_gated_mil_kd`
- `viralm_o_12l_gated_mil`
- `viralm_r_v4_final_12l_gated_mil`

Mean-pool models save HF checkpoints under `models/<model_name>/`. MIL models
save `best_mil_model.pt` plus tokenizer/config/custom code under
`models/<model_name>/`.

## Benchmark

`benchmark_realm_rank_v4_test.py` benchmarks the six trained models plus the two
frozen teacher references (`viralm_o`, `viralm_r_v4_final`). It aggregates
fragment predictions by `genome` and writes:

- `benchmark/test_predictions/<model_name>.csv`
- `benchmark/test_metrics_by_model.csv`
- `benchmark/test_metrics_by_source.csv`
- `benchmark/test_metrics_by_label_group.csv`
- `benchmark/summary.json`

The v4 test FASTA currently parses to 40,930 fragments, 20,492 viral fragments,
20,438 nonviral fragments, and 4,085 unique `genome` values. The benchmark
asserts the fragment counts by default. Pass `--expected-genomes N` if a strict
genome-count assertion is needed.
