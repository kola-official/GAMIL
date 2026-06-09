# Deduplication

## Retained Files

| GAMIL path | Source | Reason | Source mtime |
| --- | --- | --- | --- |
| `raw_data/scripts/prepare_insect_data.py` | `viralm-r/scripts/prepare_insect_data_v3.py` | Realm-Rank source preparation for strict Insecta inputs | 2026-06-03 19:53:50 |
| `process_data/scripts/build_realm_rank_dataset_v4.py` | `viralm-r/scripts/build_realm_rank_dataset_v4.py` | final strict Realm-Rank v4 builder | 2026-06-04 15:09:05 |
| `process_data/scripts/realm_rank_dataset_common.py` | `viralm-r/scripts/build_realm_rank_dataset_v3.py` | compatibility helper module required by v4 builders; not retained as a v3 entrypoint | 2026-06-03 18:37:25 |
| `process_data/scripts/prepare_realm_rank_csv.py` | `viralm-r/scripts/prepare_realm_rank_csv.py` | final FASTA-to-CSV preparation | 2026-06-04 14:31:30 |
| `process_data/scripts/build_realm_rank_test_v4.py` | `viralm-r/scripts/build_realm_rank_test_v4.py` | final fixed-length euk/pro benchmark suite builder | 2026-06-04 14:39:56 |
| `train/scripts/train.py` | `viralm-r/scripts/train.py` | current teacher training script | 2026-06-02 22:20:54 |
| `train/scripts/train.sh` | `viralm-r/scripts/train.sh` | current teacher training launcher | 2026-06-04 16:16:58 |
| `model/code/realm_rank_v4_six_model/run_realm_rank_v4_six_model_tmux.sh` and `train/scripts/run_six_model_pipeline.sh` | `paper/code/realm_rank_v4_six_model/run_realm_rank_v4_six_model_tmux.sh` | final six-model pipeline launcher; train entry is a functional-name symlink | 2026-06-06 19:28:14 |
| `model/code/realm_rank_v4_six_model/train_meanpool_kd.py` | `paper/code/realm_rank_v4_six_model/train_meanpool_kd.py` | final mean-pool KD distillation | 2026-06-06 12:08:42 |
| `model/code/realm_rank_v4_six_model/train_gated_mil_kd.py` | `paper/code/realm_rank_v4_six_model/train_gated_mil_kd.py` | final gated MIL KD distillation | 2026-06-06 18:19:31 |
| `model/code/realm_rank_v4_six_model/train_gated_mil_supervised.py` | `paper/code/realm_rank_v4_six_model/train_gated_mil_supervised.py` | final supervised gated MIL training | 2026-06-06 17:34:25 |
| `model/code/realm_rank_v4_six_model/mil_train_common.py` | `paper/code/realm_rank_v4_six_model/mil_train_common.py` | shared MIL training utilities | 2026-06-06 11:58:10 |
| `model/code/realm_rank_v4_six_model/shared.py` | `paper/code/realm_rank_v4_six_model/shared.py` | shared data/model utilities | 2026-06-06 11:54:42 |
| `model/code/realm_rank_v4_six_model/mil_model.py` | `paper/code/realm_rank_v4_six_model/mil_model.py` | final MIL model implementation | 2026-06-06 16:22:43 |
| `model/code/realm_rank_v4_six_model/benchmark_realm_rank_v4_test.py` and `benchmark/scripts/run_realm_rank_v4_six_model_benchmark.py` | `paper/code/realm_rank_v4_six_model/benchmark_realm_rank_v4_test.py` | final Realm-Rank v4 six-model benchmark; benchmark entry is a functional-name symlink | 2026-06-06 12:10:38 |
| `model/code/realm_rank_v4_six_model/benchmark_euk_pro_v4_six_models.py` and `benchmark/scripts/run_euk_pro_v4_six_model_benchmark.py` | `paper/code/realm_rank_v4_six_model/benchmark_euk_pro_v4_six_models.py` | final euk/pro v4 six-model benchmark; benchmark entry is a functional-name symlink | 2026-06-07 10:56:20 |
| `benchmark/scripts/run_viralm_flash_inference.py` | `viralm-r/eval/eval_euk_pro_o_vs_r/scripts/viralm.py` | latest flash-attention adapted inference runner | 2026-06-08 14:09:28 |
| `benchmark/scripts/metrics.py` | `viralm-r/eval/eval_euk_pro_o_vs_r/scripts/metrics.py` | latest euk/pro metrics runner | 2026-06-04 09:49:38 |
| `benchmark/scripts/run_v4_bio_attention_analysis.py` | `viralm-r/scripts/run_v4_bio_attention_analysis.py` | final v4 biological attention analysis | 2026-06-07 23:29:05 |
| `benchmark/scripts/run_viralm_r_gated_mil_single.sh` | `paper/experiment/efficiency/scripts/run_viralm_r_gated_mil_single.sh` | latest v4 efficiency runner | 2026-06-08 13:57:42 |

## Excluded Files And Directories

| Source | Reason | Source mtime |
| --- | --- | --- |
| `viralm-r/scripts/build_realm_rank_dataset.py` | old Realm-Rank builder superseded by v4 | 2026-06-02 21:13:12 |
| `viralm-r/scripts/build_realm_rank_dataset_v3.py` | old v3 entrypoint; only helper functions are retained under a neutral common module | 2026-06-03 18:37:25 |
| `viralm-r/scripts/build_realm_rank_test_v2.py` | older benchmark test suite | 2026-06-03 13:13:10 |
| `viralm-r/scripts/build_realm_rank_test_v3.py` | older benchmark test suite | 2026-06-04 09:36:41 |
| `viralm-r/scripts/resume_realm_rank_dataset_v3.py` | v3 resume helper not part of final dataset | 2026-06-03 21:19:53 |
| `viralm-r/scripts/watch_realm_rank_v3_full.sh` | v3 watcher not part of final dataset | 2026-06-04 10:12:14 |
| `ViraLM/datasets` | official text datasets excluded from GAMIL body | 2026-01-18 18:04:11 |
| `paper/code/ViraLM-mil/viralm.py` | older inference runner superseded by `benchmark/scripts/run_viralm_flash_inference.py` | 2026-05-28 19:05:13 |
| `ViraLm-mil/viralm.py` | older inference runner superseded by `benchmark/scripts/run_viralm_flash_inference.py` | 2026-04-13 16:01:12 |
| `ViraLm-mil/train.py` | older training script superseded by v4 training scripts | 2026-06-01 13:32:26 |
| `ViraLm-mil/train.sh` | older training launcher superseded by v4 launchers | 2026-06-01 13:36:03 |
| `ViraLm-mil/mil_model.py` | older MIL implementation superseded by final six-model code | 2026-05-14 15:50:28 |
| `disViralm/run_distil_viralm.py` | older distillation entrypoint | 2026-05-10 14:02:51 |
| `disViralm/run_distil2_viralm.py` | older distillation entrypoint | 2026-05-10 13:54:27 |
| `disViralm/run_distil3_viralm.py` | older distillation entrypoint | 2026-05-11 10:34:11 |
| `disViralm/mil_model.py` | older MIL implementation superseded by final six-model code | 2026-05-14 15:50:28 |
| `paper/experiment/bench/run_pro_euk_img_benchmarks.sh` | IMG/IMGVR benchmark path excluded | 2026-05-29 12:28:30 |
| `paper/experiment/bench/evaluate_pro_euk_img_benchmarks.py` | IMG/IMGVR benchmark path excluded | 2026-05-28 21:36:06 |
| `data/prodigal` | Prodigal/IMGVR-derived data excluded | 2026-06-05 22:46:04 |

## Rule Summary

- Keep the Realm-Rank v4 dataset path as the only main dataset.
- Do not copy old Realm-Rank v2/v3 entrypoints as runnable scripts.
- Do not copy ViraLM official text datasets or old ViraLM-MIL/disViralm code.
- Do not copy any IMGVR or Prodigal benchmark/data path. Mentions of those names in this file are exclusion notes only.
- Large FASTA/CSV/cache/model/checkpoint files remain local-only through manifests and ignored symlinks.
- User-facing category entries use functional names. When an entry points to canonical six-model code, it is a symlink, not a duplicate copy.
