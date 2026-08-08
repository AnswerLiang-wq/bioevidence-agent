# 简历与面试口径

## 推荐项目名称

**BioEvidence Agent｜可审计生物医学证据 Agent 与产品 Demo**

## 中文简历 bullet（建议选 3 条）

- 独立设计并实现可审计的生物医学证据 Agent，整合 BM25、多语言
  E5、RRF、cross-encoder reranker、typed tools 与 PMID/精确片段/Hash
  溯源，并产品化为可本地检索、筛选和导出的证据包 Web Demo。
- 在 500 条公开 PubMedQA 测试题上建立冻结评测：发现标题泄漏后新增
  abstract-only 压力测试，reranked Recall@10 为 0.990、MRR 为 0.982、
  nDCG@10 为 0.984，同时保留 12 次 rank drop 与 5 次 top-10 miss。
- 设计 500 条 non-test、固定五折 OOF 的 evidence-utilization 负对照；
  正确上下文较 shuffled context 提升 5.0 个准确率百分点，但 McNemar
  `p=0.066`，因此只报告 context sensitivity，不夸大为可靠推理。
- 将一次真实形成性试点作为产品决策输入：2 条尝试任务均因记录冲突
  与证据范围问题不可评估，主动停止扩样、不报告用户成效，并将关键
  风险固化为 8 个合成 fail-closed scope controls（8/8 通过）。
- 以 CI、依赖锁、版本化报告和 release manifest 收口，并明确区分工程
  benchmark、软件合同测试和用户价值证据。

不要把所有 bullet 都塞进一段经历。应聘 AI 产品经理时优先选第 1、2、
4 条；偏 Agent 工程岗位可选第 1、2、3 条。

## English resume bullets

- Built an auditable biomedical evidence Agent combining BM25,
  multilingual-E5, RRF, a cross-encoder reranker, typed tools, and
  PMID/exact-span/hash-bound provenance; productized it as a local evidence
  review and export workflow.
- Designed a leakage-aware evaluation on 500 public PubMedQA test cases;
  abstract-only reranked retrieval reached 0.990 Recall@10 and 0.982 MRR while
  preserving 12 rank drops and five top-10 misses as failure evidence.
- Stopped a planned user-study scale-up after one formative pilot produced
  zero evaluable task records; converted the observed evidence-scope risks
  into eight deterministic fail-closed controls instead of claiming user
  value from invalid measurements.

## 60 秒中文介绍

“我做 BioEvidence Agent 时，最关心的是把三个经常被 RAG Demo 混在一起
的能力拆开：找到论文、解释证据、证明结论来自哪里。系统包含 BM25、
E5、RRF、reranker 和带 PMID、摘要精确偏移及 Hash 的 typed tools，并被
产品化为本地证据卡筛选与导出 Demo。

第一次闭集检索看起来接近完美，但我发现题目和文章标题高度重合，所以
冻结了 abstract-only 压力测试；500 题上 reranked Recall@10 是 0.990，
同时如实保留 12 次 rank drop。随后用 shuffled-context 负对照检验模型
是否使用证据：正确上下文高 5 个百分点，但 p=0.066，所以没有宣称可靠
推理。

产品侧我做了一次形成性试点，但两条任务都因计时、模式和证据范围问题
不可评估。我没有补数据制造成功率，而是停止扩样，把人群、物种、干预、
终点和时间等风险转成 8 个自动化 fail-closed controls。这个项目最能体现
的是我会定义边界、设计有效评测，并在证据不够时做出诚实的停止决策。”

## 30 秒精简版

“我做了一个可审计的生物医学证据 Agent，把 BM25、E5、RRF、reranker
与 PMID、摘要精确片段和 Hash 溯源整合起来，并做成本地 Web Demo。我
通过 abstract-only 测试发现并控制标题泄漏，通过 shuffled-context 负
对照避免夸大模型推理。一次形成性试点的数据不可评估后，我主动停止扩样，
把问题转成 8 个自动化 scope controls，而没有包装不存在的用户价值。”

## 高频面试问题

### 你解决的用户问题是什么？

产品假设是为能自行做科学判断的生命科学研究者组织少量可检查的
PubMed 证据，设计目标是减少来源、片段和笔记之间的断链。当前没有
可评估真人任务验证该价值；产品不替用户给临床或最终科学结论。

### 为什么称为 Agent？

这里的 Agent 是 bounded、deterministic、tool-using evidence workflow：
它按固定合同调用检索、抓取、解析和证据检查工具，不是会自主规划并替人
作出科学判断的通用 LLM Agent。面试中应主动说明这一边界。

### 为什么 title-assisted 的完美检索不能作为 headline？

PubMedQA 问题来自或接近论文标题，而索引文档也包含标题。这是有利的身份
匹配条件，不代表面对新问题时的开放世界检索能力。因此我冻结了模型侧只
看摘要的压力测试。

### 为什么用 shuffled-context 负对照？

question-only 模型可能利用词汇和标签先验。让同一模型分别接收正确与确定性
打乱的上下文，可以检查匹配证据是否有益地改变预测。观察到 +5.0 个百分点，
但 `p=0.066`，所以只能说与 context sensitivity 相符。

### 为什么只做一次 pilot 就停止？

不是因为结果不好，而是因为测量本身不能回答问题：2 条任务都有无法消解的
计时、模式或记录冲突，故可评估任务为 0。继续招募只会增加无效记录。对求职
作品集而言，更合理的决策是冻结失败、修复最关键的产品风险、交付可复现版本，
而不是把项目扩张为一项资源不足的人体研究。

### Pilot 的结果到底是什么？

1 次形成性试点、2 条尝试任务、0 条可评估任务、0 个 VEPS 观测。不能说
VEPS 为 0%，也不能报告节时、信任、复用或用户成功率。可报告的是它暴露了
模式、计时、日志和 evidence-scope 风险。

### 8 个 scope controls 证明了什么？

它们证明系统能对预先声明的合成元数据执行确定性的 fail-closed 策略：覆盖
direct match、人群、物种、干预、终点、时间、context-only 和 unknown。
它们不证明系统能从真实摘要自动抽取这些字段，也不证明科学判断正确。

### Hash 为什么重要？

Hash 可证明审计时使用的记录或片段字节是否改变，不能证明论文相关、结论
正确或证据等级足够。它是完整性证据，不是科学真实性评分。

### 当前主要瓶颈是什么？

不是闭集检索，而是证据解释与范围对齐。答案器在公开测试上的 macro-F1
只有 0.383，`maybe` 类尤其弱；真实任务还要处理人群、干预、终点、时间和
全文缺失。

### Web Demo 是否运行完整 E5/RRF/reranker？

没有。Web Demo 使用 PubMed 候选和本地 BM25 排序，以保证本地可运行和边界
清楚。完整 hybrid/reranker 是单独的工程评测链路。面试中必须主动说明。

### 这个项目还有哪些限制？

公开标签不是盲测；500 题来自固定闭集；负对照结果并不显著；一次 pilot
没有可评估用户指标；摘要不能替代全文；合成 scope controls 不做自然语言
理解；系统不能用于医疗建议或临床决策。

## 禁止使用的表述

- “真人试验证明效率提升”或“用户验证成功”；
- “VEPS=0%”或任何真人成功率；
- “零幻觉”或“100% 引用正确”；
- “开放世界召回率 99%”；
- “Web Demo 使用完整 hybrid reranker”；
- “自动识别真实论文的 PICO 错配”；
- “可用于诊断、治疗或临床决策”。
