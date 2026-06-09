# v4 test 生物学 attention 分析方法与结果解读

本文档用尽量直观的方式说明这次分析做了什么、每一步用了什么软件、每个结果表/图是什么意思，以及目前能得到什么生物学结论。

## 1. 这次分析想回答的问题

我们比较了两个模型：

| 简称 | 模型 | 直观理解 |
|---|---|---|
| MP | `viralm_r_v4_final` | 原始模型/基线模型。它给每个 fragment 和整个 genome 一个病毒概率。 |
| GA | `viralm_r_v4_final_12l_gated_mil` | 带 gated attention 的 MIL 模型。它不仅给概率，还会给同一个 genome 内每个 fragment 一个 attention weight。 |

阈值固定为 `0.5`：概率大于等于 0.5 判为病毒，小于 0.5 判为非病毒。

核心生物学问题是：当 GA 比 MP 做得更好或更差时，GA 的 attention 是否真的落在更有病毒生物学证据的片段上？

这里的“病毒生物学证据”主要来自 geNomad 的病毒 marker、hallmark gene 和 marker enrichment；CheckV 作为辅助参考。

## 2. 数据是怎么处理的

输入数据固定为：

`processed_data/realm_rank_v4/test.fasta.gz`

关键点：

- 没有重新切割序列。
- 没有扩展窗口。
- 没有使用 `realm_rank_test_v4/bench-*`。
- 从原始 v4 test FASTA 中按 `record_id` 直接筛选目标片段。
- header 和 sequence 都保持原样。

筛选后得到：

| 数据 | 数量 |
|---|---:|
| selected records | 3300 |
| selected genomes | 848 |
| selected FASTA checksum 检查 | 3300/3300 全部与原始 v4 test 记录一致 |

所以后续 geNomad/CheckV 看到的是原始 v4 test 片段本身，不是重新加工后的序列。

## 3. case 分组是什么意思

每个 genome 根据 MP 和 GA 的 genome-level 预测是否正确来分组。

| case group | 定义 | genome 数 | record 数 | 生物学问题 |
|---|---|---:|---:|---|
| `ga_rescued_positive` | 真实是病毒；MP 判错为非病毒；GA 判对为病毒 | 33 | 253 | GA 为什么能救回这些病毒？它是否关注了有病毒 marker 的片段？ |
| `ga_corrected_negative` | 真实是非病毒；MP 判错为病毒；GA 判对为非病毒 | 280 | 359 | GA 为什么能纠正假阳性？这些片段是否缺少稳定病毒证据，或更像 plasmid/mobile/cellular？ |
| `ga_worse` | MP 判对；GA 判错 | 111 | 953 | GA 在哪些地方失败？它是否错误忽略了有病毒证据的片段？ |
| `both_correct_control` | MP 和 GA 都判对；从 3580 个 both-correct genome 中匹配抽样 | 424 | 1735 | 作为背景对照，看 attention-marker 富集是否只在关键 case 中出现。 |

匹配对照时，优先匹配 source、label、record 数、长度分布和总 bp，避免拿完全不同类型的 genome 做比较。

case 的记录组成如下：

| case group | positive records | negative records |
|---|---:|---:|
| `ga_rescued_positive` | 253 | 0 |
| `ga_corrected_negative` | 0 | 359 |
| `ga_worse` | 490 | 463 |
| `both_correct_control` | 746 | 989 |

这说明：

- `ga_rescued_positive` 全部是真病毒片段。
- `ga_corrected_negative` 全部是真非病毒片段，主要来自 plasmid、protozoa 和少量 bacteria。
- `ga_worse` 是混合错误，既包括真实病毒被 GA 漏掉，也包括真实非病毒被 GA 误报。

## 4. attention 是什么，怎么得到的

GA 模型对同一个 genome 内的多个 fragment 分配 attention weight。

直观理解：

- 一个 genome 有多个片段。
- GA 需要把这些片段综合起来判断整个 genome 是否像病毒。
- attention weight 表示模型在做 genome-level 判断时，相对更依赖哪个片段。
- 每个 genome 内所有 fragment 的 attention weight 加起来约等于 1。

这次用双 GPU 导出 attention：

- GPU0/GPU1 按 genome 分片并行。
- 最后合并 attention 结果。
- 验证结果：848 个 genome 的 attention sum 通过检查，最大误差为 `1.052e-07`。

注意：attention 只能解释“模型相对看重哪个片段”，不能直接证明“这个片段导致了预测”。所以本文档中只把 attention 当作模型行为线索，而不是因果证明。

## 5. geNomad 做了什么

geNomad 是病毒、质粒、染色体/细胞序列注释工具。它会根据序列里的基因和 marker 判断序列更像 virus、plasmid 还是 chromosome/cellular。

本分析中 geNomad 主要提供这些特征：

| 字段 | 含义 |
|---|---|
| `genomad_class` | geNomad 给出的类别：virus/plasmid/NA 等。 |
| `genomad_score` | geNomad 分类分数。 |
| `genomad_hallmark_count` | 病毒 hallmark gene 数量。hallmark gene 是比较典型的病毒特征基因。 |
| `viral_marker_density` | 每 kb 的病毒 marker 密度。越高表示病毒特征越密集。 |
| `plasmid_marker_density` | 每 kb 的质粒 marker 密度。 |
| `cellular_marker_density` | 每 kb 的细胞/染色体 marker 密度。 |
| `marker_enrichment` | 病毒 marker 密度减去 plasmid/cellular 中较大的那个值。正值越高，说明病毒信号相对非病毒信号越占优势。 |

geNomad 总体分类结果：

| geNomad class | record 数 |
|---|---:|
| virus | 2039 |
| plasmid | 518 |
| NA | 743 |

按 case 分组：

| case group | NA | plasmid | virus |
|---|---:|---:|---:|
| `ga_rescued_positive` | 17 | 28 | 208 |
| `ga_corrected_negative` | 50 | 126 | 183 |
| `ga_worse` | 239 | 120 | 594 |
| `both_correct_control` | 437 | 244 | 1054 |

解释时要注意两点：

- geNomad 的 `virus` 不是 ground truth，只是独立注释工具的判断。
- `NA` 不等于非病毒，可能是片段太短、基因信息不够，或者没有被 geNomad 明确分类。

## 6. CheckV 做了什么

CheckV 主要用于估计病毒 contig 的完整度、污染和质量等级。

本分析中 CheckV 不是主统计依据，只是辅助增强，因为 v4 records 多数是短 fragment，CheckV 对短片段通常会给低质量或无法判断。

CheckV 只跑了代表性 viral-like records：

- 真实 label 是病毒的 records；
- 或 geNomad 判为 virus 的 records；
- 或 GA genome probability 大于等于 0.5 且 attention rank 排名前 3 的 records。

实际进入 CheckV 的 records：

| 项目 | 数量 |
|---|---:|
| CheckV 输入 records | 2291 |
| CheckV `quality_summary.tsv` 行数 | 2291 |

CheckV 质量分布：

| CheckV quality | record 数 |
|---|---:|
| Low-quality | 1233 |
| Not-determined | 1051 |
| Medium-quality | 7 |
| NA | 1009 |

这里 `NA` 表示该 record 没有进入 CheckV 或没有对应结果。大量 Low-quality/Not-determined 是符合预期的，因为这些序列是短 fragment，不是完整病毒 genome。

## 7. 合并主表是什么意思

最终主表是：

`merged_bio_attention.tsv`

它把以下信息按 `record_id` 合并到一起：

- case 分组和原始标签；
- MP/GA fragment probability；
- MP/GA genome probability；
- GA attention weight 和 attention rank；
- geNomad 注释；
- CheckV 辅助质量结果。

主表规模：

| 项目 | 数量 |
|---|---:|
| records | 3300 |
| genomes | 848 |

这个表是后续所有统计表和图的基础。

## 8. 每组的总体生物学特征

下面是每个 case group 的平均 marker 特征：

| case group | mean viral marker density | mean plasmid marker density | mean cellular marker density | mean marker enrichment | mean hallmark count |
|---|---:|---:|---:|---:|---:|
| `ga_rescued_positive` | 0.7567 | 0.0187 | 0.0000 | 0.7380 | 0.2688 |
| `ga_corrected_negative` | 0.1531 | 0.2083 | 0.0582 | -0.1135 | 0.0919 |
| `ga_worse` | 0.3985 | 0.0234 | 0.1047 | 0.2709 | 0.1889 |
| `both_correct_control` | 0.4810 | 0.0578 | 0.0982 | 0.3268 | 0.1343 |

直观解释：

- `ga_rescued_positive` 的病毒 marker 密度最高，plasmid/cellular 信号很低。这说明这些被 GA 救回的真实病毒，在生物学注释上确实有比较强的病毒证据。
- `ga_corrected_negative` 的 plasmid marker density 高于 viral marker density，而且 marker enrichment 为负值。这和“MP 把非病毒误报成病毒，而 GA 纠正回来”的情形一致：这些序列可能更像 plasmid/mobile element，而不是稳定病毒信号。
- `ga_worse` 仍然有不少病毒 marker，但 attention 是否正确落在这些 marker-rich 片段上，需要看 top vs low attention 检验。
- `both_correct_control` 是背景对照，说明不是所有病毒 marker 都必然导致 GA attention 富集。

注意：不同 genome 的 record 数不同，而 attention 在每个 genome 内求和为 1，所以不能简单比较不同 case group 的平均 attention。更合理的是在同一个 genome 内比较 top-attention fragment 和其余 fragments。

## 9. top-attention vs low-attention 分析

这是主统计分析。

做法：

1. 对每个 genome 找到 GA attention rank = 1 的片段，叫 top-attention record。
2. 同一个 genome 的其他片段取平均，叫 low-attention background。
3. 比较 top 和 low 的 geNomad 特征。
4. 用 paired Wilcoxon 检验，因为 top 和 low 来自同一个 genome，是配对比较。

只有至少有 2 个 selected records 的 genome 才能参与这个分析。

### 9.1 GA rescued positives

| metric | genomes tested | top mean | low mean | top - low | p value |
|---|---:|---:|---:|---:|---:|
| hallmark count | 30 | 0.4333 | 0.1397 | 0.2936 | 0.00168 |
| viral marker density | 30 | 0.6812 | 0.4086 | 0.2726 | 0.00823 |
| marker enrichment | 30 | 0.6812 | 0.4042 | 0.2770 | 0.00823 |

含义：

GA 救回真实病毒时，它最关注的片段确实比同 genome 其他片段更富集病毒 marker 和 hallmark gene。这个结果是 strong evidence。

生物学解释：

GA 不是随机把 genome 判成病毒，而是更倾向于把权重放在有病毒特征的片段上。这支持“gated attention 帮助模型从多个片段中抓住关键病毒证据”。

### 9.2 GA corrected negatives

| metric | genomes tested | top mean | low mean | top - low | p value |
|---|---:|---:|---:|---:|---:|
| viral marker density | 54 | 0.2264 | 0.2133 | 0.0131 | 0.978 |
| marker enrichment | 54 | 0.0205 | -0.0181 | 0.0385 | 0.715 |
| plasmid marker density | 54 | 0.1795 | 0.1575 | 0.0220 | 1.000 |

含义：

在这些真实非病毒、MP 误报、GA 纠正的 genome 中，top-attention record 没有显著富集病毒 marker。

生物学解释：

这支持一个比较谨慎的说法：GA 纠正 MP 假阳性时，并没有把 attention 集中在稳定病毒 marker 上；整体上这些 records 还显示出较强 plasmid/mobile signal。因此，MP 的误报可能来自局部片段相似性，而 GA 的 genome-level 聚合降低了这种局部假阳性的影响。

### 9.3 GA worse cases

| metric | genomes tested | top mean | low mean | top - low | p value |
|---|---:|---:|---:|---:|---:|
| viral marker density | 78 | 0.4608 | 0.4218 | 0.0390 | 0.892 |
| marker enrichment | 78 | 0.3173 | 0.2846 | 0.0327 | 0.854 |

含义：

GA 出错的 case 里，top-attention record 并没有显著富集病毒 marker 或 marker enrichment。

生物学解释：

这是方法局限：GA 在某些 genome 上没有稳定地把注意力放到最有病毒证据的片段上，或者 attention 和 geNomad marker 之间没有一致对应关系。这个结果提醒我们，attention 不是完美解释器。

### 9.4 Both-correct controls

| metric | genomes tested | top mean | low mean | top - low | p value |
|---|---:|---:|---:|---:|---:|
| viral marker density | 158 | 0.3817 | 0.4841 | -0.1024 | 0.107 |
| marker enrichment | 158 | 0.2018 | 0.3266 | -0.1248 | 0.140 |

含义：

在 MP 和 GA 都已经判对的背景对照中，top-attention record 没有比其他片段更富集病毒 marker。

生物学解释：

这说明 top-attention viral-marker 富集不是所有正确预测都会自动出现的现象。它在 `ga_rescued_positive` 中更明显，因此更像是 GA 救回真实病毒时的特定模式。

## 10. evidence_assessment.tsv 怎么读

这个表把主要结论压缩成 strong/weak/negative 三类。

规则：

- effect 必须大于 0；
- p < 0.05 且 genome 数足够，才叫 strong evidence；
- p < 0.1 且 effect 大于 0，叫 weak evidence；
- 其他情况叫 negative finding。

当前结果：

| case group | claim | evidence |
|---|---|---|
| `ga_rescued_positive` | top-attention records enriched for viral marker density | strong evidence |
| `ga_rescued_positive` | top-attention records enriched for marker enrichment | strong evidence |
| `ga_corrected_negative` | 同上 | negative finding |
| `ga_worse` | 同上 | negative finding |
| `both_correct_control` | 同上 | negative finding |

最重要的结论是：只有 GA rescued positives 显示了稳定、显著、方向正确的 attention-marker 对齐。

## 11. 每张图怎么看

| 图 | 横轴/纵轴 | 主要用途 |
|---|---|---|
| `top_vs_low_attention_enrichment.svg` | 每组 top vs low 的 viral marker density | 最直观展示 GA rescued positives 中 top-attention 片段病毒 marker 更高。 |
| `attention_marker_alignment.svg` | x = GA attention weight, y = viral marker density | 看所有 records 中 attention 和病毒 marker 是否总体对齐。结果显示这种关系不是全局均匀的。 |
| `mp_probability_vs_ga_attention_quadrants.svg` | x = MP fragment probability, y = GA attention weight | 看 MP 局部高概率片段是否被 GA 高权重采用，或被 GA 降权。适合检查 MP false positive 和 GA correction。 |
| `ga_worse_sanity_check.svg` | x = GA attention weight, y = viral marker density，仅 GA worse | 专门检查 GA 出错时是否错误忽视/误用病毒 marker-rich 片段。 |

## 12. 每个输出文件的用途

| 文件 | 用途 |
|---|---|
| `case_records.tsv` | 每条 selected record 属于哪个 case group，以及原始 genome/source/label/位置/长度。 |
| `selected_v4_records.fna` | 从原始 v4 test FASTA 直接筛选出的序列，用于 geNomad/CheckV。 |
| `selected_v4_records.checksums.tsv` | 验证 selected FASTA 是否与原始 v4 test 完全一致。 |
| `attention_records.tsv` | case 信息加上 MP/GA probability 和 GA attention weight/rank。 |
| `genomad_features.tsv` | 每条 record 的 geNomad 分类、marker、hallmark 和 enrichment。 |
| `checkv_features.tsv` | 每条 record 的 CheckV quality/completeness/contamination；没有结果的保留 NA。 |
| `merged_bio_attention.tsv` | 最终主表，把 prediction、attention、geNomad、CheckV 合并。 |
| `summary_tables/case_counts.tsv` | case 分组数量。 |
| `summary_tables/case_feature_distribution.tsv` | 每个 case 的平均 marker 特征。 |
| `summary_tables/top_vs_low_attention_enrichment.tsv` | top-attention vs low-attention 的主统计检验。 |
| `summary_tables/evidence_assessment.tsv` | 把主统计结果整理为 strong/weak/negative evidence。 |
| `summary_tables/representative_sequences.tsv` | 代表性序列，用于人工查看具体案例。 |
| `figures/*.svg` | 对应的可视化图。 |
| `verification_report.tsv` | 最终完整性验证报告。 |

## 13. 最终生物学解释

可以写得比较强的结论：

> 在 GA rescued positives 中，GA 的 top-attention records 显著富集病毒 hallmark 和 viral markers。说明 gated attention 在救回 MP 漏判的真实病毒 genome 时，确实更倾向于关注具有病毒生物学证据的片段。

需要谨慎写的结论：

> 在 GA corrected negatives 中，top-attention records 没有显著富集病毒 marker；同时该组整体 plasmid marker density 较高，说明 MP 的假阳性可能与 plasmid/mobile-like 局部信号有关，GA 的 genome-level 聚合降低了这些局部信号的影响。

必须明确的局限：

> 在 GA worse cases 中，top-attention records 没有显著富集病毒 marker。这说明 attention 与生物学 marker 的对应关系不是普遍可靠的，GA 仍可能在部分 genome 上错误加权或错误聚合。

一句话总结：

> 本分析支持 gated attention 在“救回真实病毒”场景中具有可解释的生物学对齐，但不能把 attention 解释为因果证明；geNomad/CheckV 结果提供的是独立注释证据，最稳健的结论限于 GA rescued positives 的病毒 marker 富集。

## 14. 验证情况

`verification_report.tsv` 中 11 项检查全部通过：

- 固定 case counts 复现；
- selected FASTA checksum 全匹配；
- attention 在每个 genome 内求和为 1；
- merged 主表不丢 `record_id`；
- geNomad 无命中的 records 保留为 NA/0；
- 7 个 summary tables 和 4 张 figures 非空；
- pipeline 可从 `state/*.done` 一键恢复重跑。
