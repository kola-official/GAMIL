# Realm-Rank v4 six-model summary tables

Sources:
- Six-model Realm-Rank benchmark: `${LOCAL_USER_ROOT}/hybriDNA/viralm-r/checkpoint/realm_rank_v4_six_model/benchmark/test_metrics_by_model.csv`
- Six-model training logs: `${LOCAL_USER_ROOT}/hybriDNA/viralm-r/checkpoint/realm_rank_v4_six_model/logs`
- Six-model euk/pro v4 sequence metrics: `${LOCAL_USER_ROOT}/hybriDNA/viralm-r/checkpoint/realm_rank_v4_six_model/benchmark_euk_pro_v4/sequence_metrics_with_auc.csv`
- Six-model euk/pro v4 fragment metrics: `${LOCAL_USER_ROOT}/hybriDNA/viralm-r/checkpoint/realm_rank_v4_six_model/benchmark_euk_pro_v4/fragment_metrics_with_auc.csv`
- Teacher euk/pro sequence metrics: `${LOCAL_USER_ROOT}/hybriDNA/viralm-r/eval/eval_euk_pro_o_vs_r/metrics_v4/sequence_metrics_with_auc.csv`
- Teacher euk/pro fragment metrics: `${LOCAL_USER_ROOT}/hybriDNA/viralm-r/eval/eval_euk_pro_o_vs_r/metrics_v4/fragment_metrics_with_auc.csv`
- viralm-r training log: `${LOCAL_USER_ROOT}/hybriDNA/viralm-r/logs/train_20260604_165614.log`
- viralm-o training log: left blank by request

Note: `preparation_summary.md` only contains benchmark/shard preparation; formal teacher metrics were read from `metrics_v4/*_metrics_with_auc.csv`.

## Six trained models: Realm-Rank v4 test benchmark
| model | kind | precision | recall | F1 | accuracy | AUROC | AUPRC |
| --- | --- | --- | --- | --- | --- | --- | --- |
| viralm_o_6l_meanpool_kd | meanpool | 0.781746 | 0.827036 | 0.803754 | 0.882252 | 0.931284 | 0.778150 |
| viralm_r_v4_final_6l_meanpool_kd | meanpool | 0.746685 | 0.898405 | 0.815549 | 0.881518 | 0.929246 | 0.791759 |
| viralm_o_6l_gated_mil_kd | mil | 0.898886 | 0.880772 | 0.889737 | 0.936353 | 0.980214 | 0.958653 |
| viralm_r_v4_final_6l_gated_mil_kd | mil | 0.903896 | 0.876574 | 0.890026 | 0.936842 | 0.978316 | 0.956523 |
| viralm_o_12l_gated_mil | mil | 0.967577 | 0.952141 | 0.959797 | 0.976744 | 0.994870 | 0.989673 |
| viralm_r_v4_final_12l_gated_mil | mil | 0.925831 | 0.911839 | 0.918782 | 0.952999 | 0.984812 | 0.971201 |

## Realm-Rank v4 teacher references
| model | kind | precision | recall | F1 | accuracy | AUROC | AUPRC |
| --- | --- | --- | --- | --- | --- | --- | --- |
| viralm_o | meanpool | 0.891576 | 0.897565 | 0.894561 | 0.938311 | 0.971115 | 0.898765 |
| viralm_r_v4_final | meanpool | 0.776161 | 0.940386 | 0.850418 | 0.903550 | 0.949411 | 0.854505 |

## Six trained models: per-epoch dev F1
| model | epoch | val_loss | seq_f1 | precision | recall |
| --- | --- | --- | --- | --- | --- |
| viralm_o_6l_meanpool_kd | 1 | 5.955085 | 0.793887 | 0.723014 | 0.880165 |
| viralm_o_6l_meanpool_kd | 2 | 5.549751 | 0.786490 | 0.765857 | 0.808264 |
| viralm_o_6l_meanpool_kd | 3 | 5.403746 | 0.805566 | 0.756718 | 0.861157 |
| viralm_o_6l_meanpool_kd | 4 | 5.176678 | 0.805875 | 0.775401 | 0.838843 |
| viralm_o_6l_meanpool_kd | 5 | 5.188030 | 0.813926 | 0.788984 | 0.840496 |
| viralm_r_v4_final_6l_meanpool_kd | 1 | 6.184321 | 0.789700 | 0.696091 | 0.912397 |
| viralm_r_v4_final_6l_meanpool_kd | 2 | 5.448327 | 0.807573 | 0.758345 | 0.863636 |
| viralm_r_v4_final_6l_meanpool_kd | 3 | 5.461012 | 0.805364 | 0.717237 | 0.918182 |
| viralm_r_v4_final_6l_meanpool_kd | 4 | 5.331519 | 0.816342 | 0.735899 | 0.916529 |
| viralm_r_v4_final_6l_meanpool_kd | 5 | 5.301306 | 0.821509 | 0.749319 | 0.909091 |
| viralm_o_6l_gated_mil_kd | 1 | 0.267881 | 0.845196 |  |  |
| viralm_o_6l_gated_mil_kd | 2 | 0.217448 | 0.858261 |  |  |
| viralm_o_6l_gated_mil_kd | 3 | 0.196337 | 0.882571 |  |  |
| viralm_o_6l_gated_mil_kd | 4 | 0.200191 | 0.885314 |  |  |
| viralm_o_6l_gated_mil_kd | 5 | 0.187138 | 0.891139 |  |  |
| viralm_r_v4_final_6l_gated_mil_kd | 1 | 0.255118 | 0.846626 |  |  |
| viralm_r_v4_final_6l_gated_mil_kd | 2 | 0.231828 | 0.864977 |  |  |
| viralm_r_v4_final_6l_gated_mil_kd | 3 | 0.197987 | 0.885366 |  |  |
| viralm_r_v4_final_6l_gated_mil_kd | 4 | 0.195715 | 0.886278 |  |  |
| viralm_r_v4_final_6l_gated_mil_kd | 5 | 0.188878 | 0.894996 |  |  |
| viralm_o_12l_gated_mil | 1 | 0.089049 | 0.960201 |  |  |
| viralm_o_12l_gated_mil | 2 | 0.070658 | 0.963145 |  |  |
| viralm_o_12l_gated_mil | 3 | 0.078873 | 0.961941 |  |  |
| viralm_o_12l_gated_mil | 4 | 0.102326 | 0.959020 |  |  |
| viralm_o_12l_gated_mil | 5 | 0.087415 | 0.963606 |  |  |
| viralm_r_v4_final_12l_gated_mil | 1 | 0.217606 | 0.883499 |  |  |
| viralm_r_v4_final_12l_gated_mil | 2 | 0.174619 | 0.913208 |  |  |
| viralm_r_v4_final_12l_gated_mil | 3 | 0.157273 | 0.915903 |  |  |
| viralm_r_v4_final_12l_gated_mil | 4 | 0.211968 | 0.910246 |  |  |
| viralm_r_v4_final_12l_gated_mil | 5 | 0.217774 | 0.909801 |  |  |

## Teacher training per-epoch dev F1
| model | epoch | val_loss | seq_f1 | precision | recall |
| --- | --- | --- | --- | --- | --- |
| viralm-o | 1 |  |  |  |  |
| viralm-o | 2 |  |  |  |  |
| viralm-o | 3 |  |  |  |  |
| viralm-o | 4 |  |  |  |  |
| viralm-o | 5 |  |  |  |  |
| viralm-r | 1 | 0.284916 | 0.837532 | 0.750164 | 0.947934 |
| viralm-r | 2 | 0.297663 | 0.836010 | 0.762781 | 0.924793 |
| viralm-r | 3 | 0.279063 | 0.846807 | 0.765177 | 0.947934 |
| viralm-r | 4 | 0.309227 | 0.859940 | 0.789765 | 0.943802 |
| viralm-r | 5 | 0.335808 | 0.865152 | 0.798601 | 0.943802 |

## Teacher euk/pro sequence benchmark
| model | benchmark | file | precision | recall | F1 | accuracy | AUROC | AUPRC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| viralm-o | bench-euk | bench-euk-500 | 0.935953 | 0.778450 | 0.849967 | 0.862591 | 0.946341 | 0.947781 |
| viralm-r | bench-euk | bench-euk-500 | 0.830012 | 0.857143 | 0.843359 | 0.840799 | 0.909335 | 0.912193 |
| viralm-o | bench-euk | bench-euk-1000 | 0.927961 | 0.920097 | 0.924012 | 0.924334 | 0.971575 | 0.972779 |
| viralm-r | bench-euk | bench-euk-1000 | 0.904403 | 0.870460 | 0.887107 | 0.889225 | 0.944642 | 0.951758 |
| viralm-o | bench-euk | bench-euk-2000 | 0.889625 | 0.975787 | 0.930716 | 0.927361 | 0.976960 | 0.972300 |
| viralm-r | bench-euk | bench-euk-2000 | 0.888759 | 0.918886 | 0.903571 | 0.901937 | 0.962106 | 0.965213 |
| viralm-o | bench-euk | bench-euk-10000 | 0.961176 | 0.989104 | 0.974940 | 0.974576 | 0.998153 | 0.998316 |
| viralm-r | bench-euk | bench-euk-10000 | 0.953012 | 0.957627 | 0.955314 | 0.955206 | 0.991902 | 0.993073 |
| viralm-o | bench-euk | bench-euk-20000 | 0.969267 | 0.992736 | 0.980861 | 0.980630 | 0.999214 | 0.999280 |
| viralm-r | bench-euk | bench-euk-20000 | 0.974576 | 0.974576 | 0.974576 | 0.974576 | 0.995719 | 0.996065 |
| viralm-o | bench-euk | bench-euk-mixed | 0.936222 | 0.931235 | 0.933722 | 0.933898 | 0.975134 | 0.974426 |
| viralm-r | bench-euk | bench-euk-mixed | 0.909572 | 0.915738 | 0.912645 | 0.912349 | 0.955804 | 0.959496 |
| viralm-o | bench-pro | bench-pro-500 | 0.935302 | 0.805085 | 0.865322 | 0.874697 | 0.954820 | 0.958103 |
| viralm-r | bench-pro | bench-pro-500 | 0.822222 | 0.851090 | 0.836407 | 0.833535 | 0.909651 | 0.908729 |
| viralm-o | bench-pro | bench-pro-1000 | 0.957286 | 0.922518 | 0.939581 | 0.940678 | 0.983475 | 0.983765 |
| viralm-r | bench-pro | bench-pro-1000 | 0.895758 | 0.894673 | 0.895215 | 0.895278 | 0.957764 | 0.961884 |
| viralm-o | bench-pro | bench-pro-2000 | 0.978102 | 0.973366 | 0.975728 | 0.975787 | 0.994715 | 0.995150 |
| viralm-r | bench-pro | bench-pro-2000 | 0.936353 | 0.926150 | 0.931223 | 0.931598 | 0.979922 | 0.979841 |
| viralm-o | bench-pro | bench-pro-10000 | 0.987923 | 0.990315 | 0.989117 | 0.989104 | 0.998929 | 0.998868 |
| viralm-r | bench-pro | bench-pro-10000 | 0.981572 | 0.967312 | 0.974390 | 0.974576 | 0.995099 | 0.995222 |
| viralm-o | bench-pro | bench-pro-20000 | 0.997567 | 0.992736 | 0.995146 | 0.995157 | 0.999890 | 0.999890 |
| viralm-r | bench-pro | bench-pro-20000 | 0.988943 | 0.974576 | 0.981707 | 0.981840 | 0.998081 | 0.998095 |
| viralm-o | bench-pro | bench-pro-mixed | 0.972355 | 0.936804 | 0.954248 | 0.955085 | 0.989481 | 0.990117 |
| viralm-r | bench-pro | bench-pro-mixed | 0.923879 | 0.922760 | 0.923319 | 0.923366 | 0.967403 | 0.967954 |

## Six trained models: euk/pro v4 sequence benchmark

Evaluation unit: original FASTA record from `processed_data/realm_rank_test_v4`; long records are internally split into 2000 bp fragments with tail >=500 bp retained. Meanpool models average fragment probabilities; MIL models use the gated-attention sequence logit.

| model | benchmark | file | precision | recall | F1 | accuracy | AUROC | AUPRC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| viralm_o_6l_meanpool_kd | bench-euk | bench-euk-500 | 0.742785 | 0.716707 | 0.729513 | 0.734262 | 0.814822 | 0.814989 |
| viralm_o_6l_meanpool_kd | bench-euk | bench-euk-1000 | 0.734341 | 0.823245 | 0.776256 | 0.762712 | 0.850760 | 0.844166 |
| viralm_o_6l_meanpool_kd | bench-euk | bench-euk-2000 | 0.742319 | 0.906780 | 0.816349 | 0.796005 | 0.871520 | 0.848663 |
| viralm_o_6l_meanpool_kd | bench-euk | bench-euk-10000 | 0.767123 | 0.949153 | 0.848485 | 0.830508 | 0.939449 | 0.940759 |
| viralm_o_6l_meanpool_kd | bench-euk | bench-euk-20000 | 0.775828 | 0.963680 | 0.859611 | 0.842615 | 0.956547 | 0.957225 |
| viralm_o_6l_meanpool_kd | bench-euk | bench-euk-mixed | 0.753347 | 0.871913 | 0.808305 | 0.793220 | 0.876627 | 0.868464 |
| viralm_o_6l_meanpool_kd | bench-pro | bench-pro-500 | 0.825627 | 0.756659 | 0.789640 | 0.798426 | 0.888663 | 0.889713 |
| viralm_o_6l_meanpool_kd | bench-pro | bench-pro-1000 | 0.888050 | 0.854722 | 0.871067 | 0.873487 | 0.940333 | 0.946353 |
| viralm_o_6l_meanpool_kd | bench-pro | bench-pro-2000 | 0.918317 | 0.898305 | 0.908201 | 0.909201 | 0.966515 | 0.967997 |
| viralm_o_6l_meanpool_kd | bench-pro | bench-pro-10000 | 0.973945 | 0.950363 | 0.962010 | 0.962470 | 0.994655 | 0.994524 |
| viralm_o_6l_meanpool_kd | bench-pro | bench-pro-20000 | 0.990050 | 0.963680 | 0.976687 | 0.976998 | 0.997897 | 0.997785 |
| viralm_o_6l_meanpool_kd | bench-pro | bench-pro-mixed | 0.920403 | 0.884746 | 0.902222 | 0.904116 | 0.957605 | 0.959889 |
| viralm_r_v4_final_6l_meanpool_kd | bench-euk | bench-euk-500 | 0.678832 | 0.788136 | 0.729412 | 0.707627 | 0.765607 | 0.730075 |
| viralm_r_v4_final_6l_meanpool_kd | bench-euk | bench-euk-1000 | 0.722105 | 0.830508 | 0.772523 | 0.755448 | 0.817351 | 0.771599 |
| viralm_r_v4_final_6l_meanpool_kd | bench-euk | bench-euk-2000 | 0.726179 | 0.876513 | 0.794295 | 0.773002 | 0.859410 | 0.834458 |
| viralm_r_v4_final_6l_meanpool_kd | bench-euk | bench-euk-10000 | 0.754179 | 0.928571 | 0.832339 | 0.812954 | 0.922040 | 0.922932 |
| viralm_r_v4_final_6l_meanpool_kd | bench-euk | bench-euk-20000 | 0.760116 | 0.955206 | 0.846567 | 0.826877 | 0.934715 | 0.934118 |
| viralm_r_v4_final_6l_meanpool_kd | bench-euk | bench-euk-mixed | 0.729087 | 0.875787 | 0.795732 | 0.775182 | 0.852702 | 0.831020 |
| viralm_r_v4_final_6l_meanpool_kd | bench-pro | bench-pro-500 | 0.799274 | 0.800242 | 0.799758 | 0.799637 | 0.873984 | 0.871734 |
| viralm_r_v4_final_6l_meanpool_kd | bench-pro | bench-pro-1000 | 0.868486 | 0.847458 | 0.857843 | 0.859564 | 0.929764 | 0.936599 |
| viralm_r_v4_final_6l_meanpool_kd | bench-pro | bench-pro-2000 | 0.903226 | 0.881356 | 0.892157 | 0.893462 | 0.962575 | 0.963584 |
| viralm_r_v4_final_6l_meanpool_kd | bench-pro | bench-pro-10000 | 0.975949 | 0.933414 | 0.954208 | 0.955206 | 0.991742 | 0.991447 |
| viralm_r_v4_final_6l_meanpool_kd | bench-pro | bench-pro-20000 | 0.980124 | 0.955206 | 0.967505 | 0.967918 | 0.995283 | 0.995267 |
| viralm_r_v4_final_6l_meanpool_kd | bench-pro | bench-pro-mixed | 0.904561 | 0.883535 | 0.893925 | 0.895157 | 0.948886 | 0.949755 |
| viralm_o_6l_gated_mil_kd | bench-euk | bench-euk-500 | 0.699174 | 0.512107 | 0.591195 | 0.645884 | 0.708534 | 0.700250 |
| viralm_o_6l_gated_mil_kd | bench-euk | bench-euk-1000 | 0.685601 | 0.628329 | 0.655717 | 0.670097 | 0.728122 | 0.726679 |
| viralm_o_6l_gated_mil_kd | bench-euk | bench-euk-2000 | 0.674916 | 0.726392 | 0.699708 | 0.688257 | 0.742523 | 0.698021 |
| viralm_o_6l_gated_mil_kd | bench-euk | bench-euk-10000 | 0.652402 | 0.904358 | 0.757991 | 0.711259 | 0.703934 | 0.625593 |
| viralm_o_6l_gated_mil_kd | bench-euk | bench-euk-20000 | 0.647993 | 0.938257 | 0.766568 | 0.714286 | 0.694398 | 0.620143 |
| viralm_o_6l_gated_mil_kd | bench-euk | bench-euk-mixed | 0.667247 | 0.741889 | 0.702591 | 0.685956 | 0.724716 | 0.660805 |
| viralm_o_6l_gated_mil_kd | bench-pro | bench-pro-500 | 0.944812 | 0.518160 | 0.669273 | 0.743947 | 0.904103 | 0.908445 |
| viralm_o_6l_gated_mil_kd | bench-pro | bench-pro-1000 | 0.969125 | 0.684019 | 0.801987 | 0.831114 | 0.945622 | 0.952340 |
| viralm_o_6l_gated_mil_kd | bench-pro | bench-pro-2000 | 0.982201 | 0.734867 | 0.840720 | 0.860775 | 0.952954 | 0.966166 |
| viralm_o_6l_gated_mil_kd | bench-pro | bench-pro-10000 | 0.973924 | 0.904358 | 0.937853 | 0.940073 | 0.986098 | 0.987928 |
| viralm_o_6l_gated_mil_kd | bench-pro | bench-pro-20000 | 0.974843 | 0.938257 | 0.956200 | 0.957022 | 0.989223 | 0.989895 |
| viralm_o_6l_gated_mil_kd | bench-pro | bench-pro-mixed | 0.970771 | 0.755932 | 0.849986 | 0.866586 | 0.963129 | 0.967895 |
| viralm_r_v4_final_6l_gated_mil_kd | bench-euk | bench-euk-500 | 0.632042 | 0.434625 | 0.515065 | 0.590799 | 0.647634 | 0.616925 |
| viralm_r_v4_final_6l_gated_mil_kd | bench-euk | bench-euk-1000 | 0.673602 | 0.627119 | 0.649530 | 0.661622 | 0.708929 | 0.659090 |
| viralm_r_v4_final_6l_gated_mil_kd | bench-euk | bench-euk-2000 | 0.683628 | 0.748184 | 0.714451 | 0.700969 | 0.752681 | 0.699126 |
| viralm_r_v4_final_6l_gated_mil_kd | bench-euk | bench-euk-10000 | 0.682495 | 0.887409 | 0.771579 | 0.737288 | 0.787241 | 0.746245 |
| viralm_r_v4_final_6l_gated_mil_kd | bench-euk | bench-euk-20000 | 0.683453 | 0.920097 | 0.784314 | 0.746973 | 0.806348 | 0.779916 |
| viralm_r_v4_final_6l_gated_mil_kd | bench-euk | bench-euk-mixed | 0.674949 | 0.723487 | 0.698376 | 0.687530 | 0.745851 | 0.703256 |
| viralm_r_v4_final_6l_gated_mil_kd | bench-pro | bench-pro-500 | 0.936709 | 0.447942 | 0.606061 | 0.708838 | 0.885920 | 0.886087 |
| viralm_r_v4_final_6l_gated_mil_kd | bench-pro | bench-pro-1000 | 0.965035 | 0.668281 | 0.789700 | 0.822034 | 0.936415 | 0.943720 |
| viralm_r_v4_final_6l_gated_mil_kd | bench-pro | bench-pro-2000 | 0.966981 | 0.744552 | 0.841313 | 0.859564 | 0.959348 | 0.964213 |
| viralm_r_v4_final_6l_gated_mil_kd | bench-pro | bench-pro-10000 | 0.972037 | 0.883777 | 0.925808 | 0.929177 | 0.983537 | 0.984369 |
| viralm_r_v4_final_6l_gated_mil_kd | bench-pro | bench-pro-20000 | 0.973111 | 0.920097 | 0.945862 | 0.947337 | 0.989148 | 0.990545 |
| viralm_r_v4_final_6l_gated_mil_kd | bench-pro | bench-pro-mixed | 0.965550 | 0.732930 | 0.833310 | 0.853390 | 0.958969 | 0.962364 |
| viralm_o_12l_gated_mil | bench-euk | bench-euk-500 | 0.833010 | 0.519370 | 0.639821 | 0.707627 | 0.831308 | 0.838561 |
| viralm_o_12l_gated_mil | bench-euk | bench-euk-1000 | 0.878125 | 0.680387 | 0.766712 | 0.792978 | 0.899344 | 0.898538 |
| viralm_o_12l_gated_mil | bench-euk | bench-euk-2000 | 0.890392 | 0.796610 | 0.840895 | 0.849274 | 0.934569 | 0.921502 |
| viralm_o_12l_gated_mil | bench-euk | bench-euk-10000 | 0.848684 | 0.937046 | 0.890679 | 0.884988 | 0.959807 | 0.949578 |
| viralm_o_12l_gated_mil | bench-euk | bench-euk-20000 | 0.837500 | 0.973366 | 0.900336 | 0.892252 | 0.971739 | 0.967965 |
| viralm_o_12l_gated_mil | bench-euk | bench-euk-mixed | 0.856877 | 0.781356 | 0.817376 | 0.825424 | 0.921066 | 0.916094 |
| viralm_o_12l_gated_mil | bench-pro | bench-pro-500 | 0.973094 | 0.525424 | 0.682390 | 0.755448 | 0.935753 | 0.942072 |
| viralm_o_12l_gated_mil | bench-pro | bench-pro-1000 | 0.991568 | 0.711864 | 0.828753 | 0.852906 | 0.970765 | 0.974554 |
| viralm_o_12l_gated_mil | bench-pro | bench-pro-2000 | 0.996875 | 0.772397 | 0.870396 | 0.884988 | 0.985166 | 0.988148 |
| viralm_o_12l_gated_mil | bench-pro | bench-pro-10000 | 0.992405 | 0.949153 | 0.970297 | 0.970944 | 0.997855 | 0.997902 |
| viralm_o_12l_gated_mil | bench-pro | bench-pro-20000 | 0.990148 | 0.973366 | 0.981685 | 0.981840 | 0.998754 | 0.998735 |
| viralm_o_12l_gated_mil | bench-pro | bench-pro-mixed | 0.989942 | 0.786441 | 0.876535 | 0.889225 | 0.982466 | 0.984655 |
| viralm_r_v4_final_12l_gated_mil | bench-euk | bench-euk-500 | 0.923077 | 0.450363 | 0.605370 | 0.706416 | 0.818826 | 0.845076 |
| viralm_r_v4_final_12l_gated_mil | bench-euk | bench-euk-1000 | 0.963907 | 0.549637 | 0.700077 | 0.764528 | 0.874487 | 0.895681 |
| viralm_r_v4_final_12l_gated_mil | bench-euk | bench-euk-2000 | 0.941509 | 0.604116 | 0.735988 | 0.783293 | 0.892096 | 0.905541 |
| viralm_r_v4_final_12l_gated_mil | bench-euk | bench-euk-10000 | 0.910112 | 0.784504 | 0.842653 | 0.853511 | 0.915339 | 0.923796 |
| viralm_r_v4_final_12l_gated_mil | bench-euk | bench-euk-20000 | 0.891026 | 0.841404 | 0.865504 | 0.869249 | 0.938235 | 0.942484 |
| viralm_r_v4_final_12l_gated_mil | bench-euk | bench-euk-mixed | 0.921271 | 0.646005 | 0.759465 | 0.795400 | 0.887879 | 0.902156 |
| viralm_r_v4_final_12l_gated_mil | bench-pro | bench-pro-500 | 0.922330 | 0.460048 | 0.613893 | 0.710654 | 0.874337 | 0.877636 |
| viralm_r_v4_final_12l_gated_mil | bench-pro | bench-pro-1000 | 0.959916 | 0.550847 | 0.700000 | 0.763923 | 0.925236 | 0.931577 |
| viralm_r_v4_final_12l_gated_mil | bench-pro | bench-pro-2000 | 0.990403 | 0.624697 | 0.766147 | 0.809322 | 0.946066 | 0.952840 |
| viralm_r_v4_final_12l_gated_mil | bench-pro | bench-pro-10000 | 0.976574 | 0.807506 | 0.884029 | 0.894068 | 0.978569 | 0.979175 |
| viralm_r_v4_final_12l_gated_mil | bench-pro | bench-pro-20000 | 0.977496 | 0.841404 | 0.904359 | 0.911017 | 0.986418 | 0.986821 |
| viralm_r_v4_final_12l_gated_mil | bench-pro | bench-pro-mixed | 0.968583 | 0.656901 | 0.782860 | 0.817797 | 0.948656 | 0.952389 |

Fragment-level metrics are written to `${LOCAL_USER_ROOT}/hybriDNA/viralm-r/checkpoint/realm_rank_v4_six_model/tables/six_model_euk_pro_v4_fragment_benchmark.csv`.

