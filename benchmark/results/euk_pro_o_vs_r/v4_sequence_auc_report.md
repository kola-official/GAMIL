# realm_rank_v4 benchmark metrics

- Data root: `${LOCAL_USER_ROOT}/hybriDNA/viralm-r/processed_data/realm_rank_test_v4`
- Results root: `${LOCAL_USER_ROOT}/hybriDNA/viralm-r/eval/eval_euk_pro_o_vs_r/results_v4`
- Models: `viralm-o`, `viralm-r-v4` (`checkpoint/virlam-r-v4`)
- Level shown below: sequence

| model | benchmark | file | precision | recall | F1 | AUROC | AUPRC |
|---|---|---|---:|---:|---:|---:|---:|
| viralm-o | bench-euk | bench-euk-500 | 0.935953 | 0.778450 | 0.849967 | 0.946341 | 0.947781 |
| viralm-r-v4 | bench-euk | bench-euk-500 | 0.830012 | 0.857143 | 0.843359 | 0.909335 | 0.912193 |
| viralm-o | bench-euk | bench-euk-1000 | 0.927961 | 0.920097 | 0.924012 | 0.971575 | 0.972779 |
| viralm-r-v4 | bench-euk | bench-euk-1000 | 0.904403 | 0.870460 | 0.887107 | 0.944642 | 0.951758 |
| viralm-o | bench-euk | bench-euk-2000 | 0.889625 | 0.975787 | 0.930716 | 0.976960 | 0.972300 |
| viralm-r-v4 | bench-euk | bench-euk-2000 | 0.888759 | 0.918886 | 0.903571 | 0.962106 | 0.965213 |
| viralm-o | bench-euk | bench-euk-10000 | 0.961176 | 0.989104 | 0.974940 | 0.998153 | 0.998316 |
| viralm-r-v4 | bench-euk | bench-euk-10000 | 0.953012 | 0.957627 | 0.955314 | 0.991902 | 0.993073 |
| viralm-o | bench-euk | bench-euk-20000 | 0.969267 | 0.992736 | 0.980861 | 0.999214 | 0.999280 |
| viralm-r-v4 | bench-euk | bench-euk-20000 | 0.974576 | 0.974576 | 0.974576 | 0.995719 | 0.996065 |
| viralm-o | bench-euk | bench-euk-mixed | 0.936222 | 0.931235 | 0.933722 | 0.975134 | 0.974426 |
| viralm-r-v4 | bench-euk | bench-euk-mixed | 0.909572 | 0.915738 | 0.912645 | 0.955804 | 0.959496 |
| viralm-o | bench-pro | bench-pro-500 | 0.935302 | 0.805085 | 0.865322 | 0.954820 | 0.958103 |
| viralm-r-v4 | bench-pro | bench-pro-500 | 0.822222 | 0.851090 | 0.836407 | 0.909651 | 0.908729 |
| viralm-o | bench-pro | bench-pro-1000 | 0.957286 | 0.922518 | 0.939581 | 0.983475 | 0.983765 |
| viralm-r-v4 | bench-pro | bench-pro-1000 | 0.895758 | 0.894673 | 0.895215 | 0.957764 | 0.961884 |
| viralm-o | bench-pro | bench-pro-2000 | 0.978102 | 0.973366 | 0.975728 | 0.994715 | 0.995150 |
| viralm-r-v4 | bench-pro | bench-pro-2000 | 0.936353 | 0.926150 | 0.931223 | 0.979922 | 0.979841 |
| viralm-o | bench-pro | bench-pro-10000 | 0.987923 | 0.990315 | 0.989117 | 0.998929 | 0.998868 |
| viralm-r-v4 | bench-pro | bench-pro-10000 | 0.981572 | 0.967312 | 0.974390 | 0.995099 | 0.995222 |
| viralm-o | bench-pro | bench-pro-20000 | 0.997567 | 0.992736 | 0.995146 | 0.999890 | 0.999890 |
| viralm-r-v4 | bench-pro | bench-pro-20000 | 0.988943 | 0.974576 | 0.981707 | 0.998081 | 0.998095 |
| viralm-o | bench-pro | bench-pro-mixed | 0.972355 | 0.936804 | 0.954248 | 0.989481 | 0.990117 |
| viralm-r-v4 | bench-pro | bench-pro-mixed | 0.923879 | 0.922760 | 0.923319 | 0.967403 | 0.967954 |

## viralm-r-v4 minus viralm-o

| benchmark | file | delta F1 | delta AUROC | delta AUPRC |
|---|---|---:|---:|---:|
| bench-euk | bench-euk-500 | -0.006608 | -0.037006 | -0.035588 |
| bench-euk | bench-euk-1000 | -0.036905 | -0.026933 | -0.021021 |
| bench-euk | bench-euk-2000 | -0.027145 | -0.014854 | -0.007087 |
| bench-euk | bench-euk-10000 | -0.019626 | -0.006251 | -0.005243 |
| bench-euk | bench-euk-20000 | -0.006285 | -0.003495 | -0.003215 |
| bench-euk | bench-euk-mixed | -0.021077 | -0.019330 | -0.014930 |
| bench-pro | bench-pro-500 | -0.028915 | -0.045169 | -0.049374 |
| bench-pro | bench-pro-1000 | -0.044366 | -0.025711 | -0.021881 |
| bench-pro | bench-pro-2000 | -0.044505 | -0.014793 | -0.015309 |
| bench-pro | bench-pro-10000 | -0.014727 | -0.003830 | -0.003646 |
| bench-pro | bench-pro-20000 | -0.013439 | -0.001809 | -0.001795 |
| bench-pro | bench-pro-mixed | -0.030929 | -0.022078 | -0.022163 |
