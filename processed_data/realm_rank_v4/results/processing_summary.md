# Realm-Rank v4 Processing Summary

Generated at: 2026-06-04T15:59:00
Input root: `data`
Output directory: `processed_data/realm_rank_v4`

## Parameters

- Seed: `1729`
- Threads: `54`
- Dev fraction: `0.1`
- Test fraction: `0.1`
- Fragment length: `300-2000`
- Length bin width: `100`
- Dev-vs-train and test-vs-train+dev removal threshold: `>0.5`
- Nonviral sources: `bacteria, archaea, fungi, plasmid, protozoa, insect`
- Genus caps: bacteria <= 2; insect <= 1

## Genome Split

- Duplodnaviria: train=3420 dev=427 test=427
- Riboviria: train=5045 dev=631 test=631
- SmallRealm: train=2411 dev=301 test=301
- Varidnaviria: train=210 dev=26 test=26
- archaea: train=236 dev=30 test=30
- bacteria: train=1758 dev=220 test=220
- fungi: train=73 dev=9 test=9
- insect: train=4 dev=1 test=1
- plasmid: train=63685 dev=7960 test=7960
- protozoa: train=136502 dev=17064 test=17063

## Final Fragments

- dev Duplodnaviria (virus): 9224 fragments, 10436528 bp, length 300-2000
- dev Riboviria (virus): 3614 fragments, 3676924 bp, length 300-1999
- dev SmallRealm (virus): 2021 fragments, 2171322 bp, length 300-2000
- dev Varidnaviria (virus): 1581 fragments, 1790179 bp, length 301-1999
- dev archaea (archaea): 2728 fragments, 2999601 bp, length 300-2000
- dev bacteria (bacteria): 2729 fragments, 3003547 bp, length 300-2000
- dev fungi (fungi): 2725 fragments, 3005119 bp, length 300-2000
- dev insect (insect): 2732 fragments, 3004036 bp, length 300-2000
- dev plasmid (plasmid): 2729 fragments, 3002785 bp, length 301-2000
- dev protozoa (protozoa): 2730 fragments, 3002038 bp, length 300-2000
- test Duplodnaviria (virus): 9376 fragments, 10676939 bp, length 300-2000
- test Riboviria (virus): 3471 fragments, 3517371 bp, length 300-2000
- test SmallRealm (virus): 3592 fragments, 3925079 bp, length 300-2000
- test Varidnaviria (virus): 4053 fragments, 4631152 bp, length 300-2000
- test archaea (archaea): 3410 fragments, 3783971 bp, length 300-2000
- test bacteria (bacteria): 3411 fragments, 3784017 bp, length 300-2000
- test fungi (fungi): 3411 fragments, 3783405 bp, length 300-2000
- test insect (insect): 3411 fragments, 3786005 bp, length 300-2000
- test plasmid (plasmid): 3383 fragments, 3774730 bp, length 300-2000
- test protozoa (protozoa): 3412 fragments, 3784325 bp, length 300-2000
- train Duplodnaviria (virus): 225955 fragments, 256528277 bp, length 300-2000
- train Riboviria (virus): 30016 fragments, 30276835 bp, length 300-2000
- train SmallRealm (virus): 29673 fragments, 31916115 bp, length 300-2000
- train Varidnaviria (virus): 26391 fragments, 30190581 bp, length 300-2000
- train archaea (archaea): 51975 fragments, 58144005 bp, length 300-2000
- train bacteria (bacteria): 51963 fragments, 58142470 bp, length 300-2000
- train fungi (fungi): 51950 fragments, 58144014 bp, length 300-2000
- train insect (insect): 51960 fragments, 58143282 bp, length 300-2000
- train plasmid (plasmid): 51976 fragments, 58143972 bp, length 300-2000
- train protozoa (protozoa): 51981 fragments, 58140127 bp, length 300-2000

## Verification

- No failed checks in `qc/verification.tsv`.

## Key Outputs

- `train.fasta.gz`, `dev.fasta.gz`, and `test.fasta.gz`: final compressed FASTA files.
- `metadata/fragments.tsv`: selected fragment coordinates.
- `metadata/dev_removed_by_train_blast.tsv`: dev fragments removed by train BLAST coverage.
- `metadata/test_removed_by_train_dev_blast.tsv`: test fragments removed by train+dev BLAST coverage.
- `qc/split_balance.json`: requested and actual split proportions plus binary balance details.
- `blast/*.tsv`: raw BLAST tables used for decontamination and leakage checks.
