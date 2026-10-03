# 实验记录归档 v1（v1 / v2 / v3）

- 编制日期：2026-10-03（第 2 版，同日修订）
- 用途：让三轮评估实验的运行条件可被第三方独立审计。**本文件不新增实验、不改写任何历史产物。**
- 事实来源：各轮冻结协议与原始报告。记忆与会话摘要不是事实来源。
- **改动归属**：本文件只声明自身被编辑，**不**以工作树状态作为本轮改动范围的证明。

## 记录分层（三类不可混用）

| 层 | 含义 | 本文件中的标记 |
|---|---|---|
| 原始运行记录 | 运行当时由程序写入的报告字段 | 直接引用字段名 |
| 历史预检记录 | 更早会话中产生的审计/预检文档，**不是**运行快照 | 标注文件名 |
| 当前环境实测 | 本文件编制时在本机测得的版本与哈希，**不能**代表实验运行时的环境 | 标注「实测」 |

**「现有归档未提供」的含义**：指本次可查阅的归档中不存在该项，**不等于**该事项在运行当时
从未发生或被忽略。只陈述归档缺口，不据此推断当时状态。

---

## 1. 三轮实验对比

| 项 | v1 | v2 | v3 |
|---|---|---|---|
| 报告文件 | `private/env_precheck/pipeline_comparison_50_final.json` | `private/env_precheck/v2_ab_50_final.json` | `private/v3/v3_ac_50_final.json` |
| 报告 sha256 | `17c6f249…` | `9f5ff7a6…` | `25c02d6b…` |
| 运行日期（报告字段） | 2026-10-01T18:35:04Z | 2026-10-02T04:45:39Z | 2026-10-02T06:32:46Z |
| 臂 | A（agent）+ 规则基线 | A + B（固定上下文）+ 规则基线 | A + C（提示词变体）+ 规则基线 |
| API 模型 | `deepseek-flash` | `deepseek-flash` | `deepseek-flash` |
| 样本文件 | `reports/eval_sample_50_v1_seed20260729.json` | `reports/eval_sample_v2_50_seed20261002.json` | `private/v3/eval_sample_v3_50_seed20261002.json` |
| 样本文件 sha256 | `bcc8753f…` | `5fe5c4ab…` | `3f0eacad…` |
| case_ids sha256 | `c42cb816…` | `f2d37d74…` | `115537cb…` |
| seed | `20260729` | `20261002` | `20261002` |
| 种子声明方式 | 报告 `sample.seed`（**无**冻结协议文件） | 协议 §2「协议日期写作 YYYYMMDD，抽样前确定，未搜索表现更好的 seed」 | 协议 §4 同一规则；协议明文「v3 协议日期写作 YYYYMMDD」 |
| 抽样池 | 500（yes 276 / no 169 / maybe 55） | 450（248 / 152 / 50） | 400（221 / 135 / 44） |
| 配额 yes/no/maybe | 28 / 17 / 5 | 27 / 17 / 6 | 28 / 17 / 5 |
| 抽样方法 | 最大余数法（Hare quota），层内 case_id 升序 + `random.Random(seed).sample` 不放回 | 同 | 同；同余数按标签升序打破 |
| 排除集合 | 无（首次抽样，池为全体 500） | 协议 §2/§8：v1 冻结样本 ∪ v1 已尝试案例（当时两者相同，等于 50 例）。§8 记录了当时的歧义与消解 | 协议 §4：v1 ∪ v2 的**全部 100 个已尝试案例**（含 v1 超时案例 `PUBMEDQA-TEST-0402`） |
| 与更早轮次重叠 | — | 0 | 0 |
| 检索条件 | BM25 title-assisted（**v1 报告本身未记录**，仅有 v2 协议 §4 的追溯声明；见 §2.1） | BM25 title-assisted；A 与 B **同一索引**；B 固定 **top-4** 静态上下文（v2 协议 §4） | 与 v2 相同（v3 协议 §1） |
| 语料 / 测试集 / 训练集 | 1,000 文档 / 500 例 / 500 例（`data/pubmedqa/v1/manifest.json`） | 同 | 同 |
| 评分映射版本 | **legacy**（报告无 `scoring_policy` 字段，由 §3 映射表核验） | **legacy**（协议 §3 明文；报告无该字段） | **v3**（报告 `metrics.scoring_policy = "v3"`、`acceptance.scoring_policy`） |
| macro-F1 覆盖 | 仅**完成案例**。**逐臂覆盖数不同**：A 49 例（support 合计 49）、规则臂 50 例（support 合计 50），两臂使用同一计划样本 | 仅完成案例，两臂各 50 例 | **全部已尝试案例**（合计 50 = `attempted`）。A 完成 49 / 失败 1；C 完成 50 / 失败 0 |
| A macro-F1 | 0.5638（accuracy 0.7551） | 0.6003（accuracy 0.78） | 0.5700（精确 0.5700247079964061） |
| B / C macro-F1 | — | B 0.5388 | C 0.6074（精确 0.607397504456328） |
| 规则基线 macro-F1 | 0.4196（accuracy 0.50） | 0.4216（accuracy 0.64） | 0.3344（精确 0.33444816053511706） |
| 完成 / 失败 / 未运行 | 49 / 1 / 0（失败 `PUBMEDQA-TEST-0402`，`APITimeoutError`） | 50 / 0 / 0 | A 49 / 1 / 0（失败 `PUBMEDQA-TEST-0043`，`budget_exhausted`，gold=no）；C 50 / 0 / 0 |
| 验收阈值 | **现有归档未提供**（报告无 `acceptance` 字段，现有归档中也无 v1 冻结协议） | **现有归档未提供**（协议无验收阈值；§9 的 \$0.002 是费用软阈值，不是验收判据） | 预声明：Δmacro-F1 ≥ 0.04 **且** C 在 gold=maybe 正确率 > A |
| criteria_met | **现有归档未提供** | **现有归档未提供** | **false**（`macro_f1_gain_met=false`；第二条件 `c_exceeds_a=true` 成立） |
| Δmacro-F1（C−A） | — | — | **0.037372796459921864** |
| 费用（估算，非账单） | A \$0.18653；`cost_complete = false`（49 confirmed / 1 partial / 0 unknown）→ **部分用量** | A \$0.196013、B \$0.044505；报告总额 \$0.240517（两臂 `cost_complete = true`）→ **完整用量** | A \$0.204216 + C \$0.235161 = \$0.439377（两臂 `cost_complete = true`）→ **完整用量** |
| 提示词指纹 | **未记录**（全系列无） | 两份**原始分段有**（A `84e8d97b…`、B `688bcf4d…`）；**合并报告无** | 两份分段**与**合并报告**都有**（A `84e8d97b…`、C `ae18019f…`） |
| 合并来源 | 3 段：trial5 + remaining45 + remaining13 | 2 段：trial5 + remaining45 | 2 段：trial5 + remaining45 |

### 1.1 评分映射差异（明文对比，不依赖代码）

| verdict | legacy（v1、v2） | v3（仅 v3） |
|---|---|---|
| `supported` | → yes | → yes |
| `contradicted` | → no | → no |
| `insufficient` | → maybe | → maybe |
| `mixed` | **不匹配任何 gold**，计为不正确，单独统计 `n_mixed_verdicts` | **→ maybe** |

**关键差异，两条：**
1. `mixed` 在 legacy 下不匹配任何 gold；在 v3 下映射为 maybe。
2. **失败案例处理**：legacy 下失败案例**不进** macro-F1；v3 下覆盖全部已尝试案例，失败**只**给其 gold 类 +1 FN，不贡献 TP 或 FP。v3 的 `run_status` 检查在 verdict 检查**之前**——失败时返回的 `insufficient` **不是**正确的 maybe 预测。

**未改写声明**：v1/v2 结果**未被** v3 映射改写。legacy 映射保留为独立实现且仍是默认值。

### 1.2 各轮 `mixed` 出现次数（报告字段 `n_mixed_verdicts`）

| 轮次 | A | B/C | 规则基线 |
|---|---|---|---|
| v1 | 6 | — | 0 |
| v2 | 8 | B 1 | 0 |
| v3 | 6 | C 7 | 0 |

### 1.3 费用语义与展示舍入

**费用一律是估算，不是账单。** 界限方向取决于用量是否完整，两者不可混为一谈：

| 情况 | 含义 | 界限方向 |
|---|---|---|
| **完整用量**（`cost_complete = true`） | 已知小计按记录的**峰值 cache-miss 费率**计算 | 在该费率假设下是**保守估算**；**不是**账单硬上限。谷时约一半、缓存命中更便宜，真实账单可能更低，但无发票可核 |
| **部分用量**（`cost_complete = false`） | 已知小计**只覆盖服务端报告了用量的轮次** | 对整轮实际账单的**界限方向未知**。未报告的轮次可能推高也可能不推高，无法据已知小计判断 |

**v2 的展示舍入差异**（是舍入，不是计费差异）：

| 项 | 值 |
|---|---|
| 逐例未舍入小计合计 | **0.2405172**（A 0.1960125 + B 0.0445047） |
| 报告展示的合并总额 `merge_notes.known_subtotal_usd` | **0.240517**（对未舍入合计四舍五入） |
| 两臂各自舍入后的展示值相加 | 0.196013 + 0.044505 = **0.240518** |

三者中只有前两者一致。引用时**不得**写成 `0.196013 + 0.044505 = 0.240517`——这是一个错误的精确等式。
v1（0.1865304 → 0.18653）与 v3（0.439377 → 0.439377）无同类差异。

**已知用量缺口不插补**。v1 `PUBMEDQA-TEST-0402` 的 `llm_cost_status = "partial"`、`llm_cost_usd = null`、
已取得小计 \$0.0021603；整轮的 `cost_complete = false` 如实标注，未改写成完整账目。

---

## 2. 复现前提清单（复现 v3 所需）

### 2.1 检索条件与来源（不同实验不可互换）

| 项 | 值 | 来源 |
|---|---|---|
| 语料 / 测试集 / 训练集 | 1,000 文档 / 500 例（yes 276、no 169、maybe 55）/ 500 例 | `data/pubmedqa/v1/manifest.json` |
| 评测性质 | public closed-corpus，**非盲**；官方测试标签公开 | 同上 `evaluation_scope`、`limitations` |
| **本轮 A 臂检索** | **BM25 title-assisted**：索引文档含 title；与剥离 title 的 `pubmedqa_stress.py` 路径不同 | v2 协议 §4；v3 协议 §1（A 与 v2 相同） |
| **v1 的检索条件** | **现有归档未提供**。v1 报告未记录任何检索字段（`bm25`/`hybrid`/`retrieval` 出现 0 次）；v2 协议 §4 以「与 v1 相同」追溯声明，v2 设计交接亦称 A 臂 same retrieval。**不以当前代码补造** | — |
| **B 的检索** | 与 A 同一索引；对原始问题**一次静态 BM25 检索，固定 top-4 摘要**作为上下文块；单次请求，无工具循环 | v2 协议 §1/§4；v2 报告 `config.baseline_arm` |
| B 的 `top_k=4` 依据 | v1 完成案例中每例不同 fetched PMID 数的中位数为 4（分布 2:7、3:11、4:29、6:1、8:1） | v2 协议 §4 |
| **规则臂训练范围** | 协议声明「现有 `TfidfLogisticAnswerer`，**仅用 train 拟合**」 | v2 协议 §1 表。**v1 报告的 `metrics.rule_based` 中无训练范围字段** |
| **标题可见条件**（旧检索基准，非本轮） | 四系统 Recall@5/10、MRR、nDCG@10 **均为 1.000**；问题是标题或标题派生且索引含该标题，属**有利源定位** | `reports/pubmedqa_title_assisted_v1.json` |
| **abstract-only 压力条件**（旧检索基准，非本轮） | 标题对所有检索器不可见；bm25 Recall@5 0.9760 / MRR 0.9532，与 title-assisted 的 MRR 差 −0.046813 | `reports/pubmedqa_abstract_only_retrieval_v1.json` |
| **旧混合检索**（非本轮） | `hybrid_rrf`、`hybrid_reranked` 的这些指标属**旧检索基准实验**，不能替代 v1/v2/v3 的实际检索记录。v2/v3 条件按各自协议记录，v1 保持未确认 | 同上两份报告；v2/v3 协议 |

> **不得混淆的旧数字**：标题可见条件下 bm25 答题基线 macro-F1 **0.383221**（500 例全集）
> 与 v1 规则基线 **0.4196**（v1 的 50 例样本、legacy 口径）**不是同一个量**。

### 2.2 环境与版本

| 项 | 值 | 状态 |
|---|---|---|
| 模型 ID | `deepseek-flash` | 已记录（报告 `config.model`） |
| base_url | `https://api.deepseek.com/v1` | 已记录 |
| thinking | disabled（`extra_body={'thinking': {'type': 'disabled'}}`） | 已记录 |
| temperature / max_steps / max_tokens_per_call | 0.0 / 8 / 1024 | 已记录 |
| request_timeout_seconds / sdk_max_retries | 30.0 / 0 | 已记录 |
| Python 版本 | — | **未记录**。运行时的解释器版本未写入任何报告。原运行已完成，**此项无法补全，不得伪造** |
| 依赖包版本 | — | **未记录**。未写入任何报告。原运行已完成，**此项无法补全，不得伪造** |
| lock 文件 | `requirements/ci.lock`、`requirements/heavy.lock` | **当前文件**，其 sha256 见下。**不能**代表 v3 运行时的依赖集合 |
| A 提示词 sha256 / 字符数 | `84e8d97b91a0940698b93a25db09d821d5ab832e17776d62a114f99d468db880` / 838 | 已记录（v3 报告 `prompt_fingerprint.agent`） |
| A 工具 schema sha256 | `98e8916fc0a01162c7d4f178da0b4ff5f1069550cb963651e62a9a11146dcb16` | 已记录 |
| C 提示词 sha256 / 字符数 | `ae18019f66f24e411297cec52fb57dafe8a7f9b988a05d3fa39fd734d7c592f1` / 1298 | 已记录（`prompt_fingerprint.arm_c`） |
| C 工具 schema sha256 | `79e1d3f8e831e9bb58a5c1939a6b2c9cef9b739610e8c92598df5f77a7ea7eb2` | 已记录 |
| 干预差分为纯插入 | `system_prompt_delta_is_insertion_only = true` | 已记录（自动断言） |
| 种子声明位置 | `private/v3/v3_protocol_frozen_20261002.md` §4 | 已记录 |
| 样本文件与哈希 | `private/v3/eval_sample_v3_50_seed20261002.json`，sha256 `3f0eacad…` | 已记录 |
| 排除集合定义 | 协议 §4：v1 ∪ v2 全部 100 个已尝试案例 | 已记录（明文，非仅代码） |
| 输入数据 | `data/pubmedqa/v1/` 下 corpus / train / test_inputs / test_gold，哈希见 v3 报告 `input_hashes` | 已记录，本轮复核一致 |
| 源码文件哈希 | v3 协议 §5：`llm_agent.py` `63e64f93…`、`prompt_variants.py` `4972966f…`、`compare_pipelines.py` `97a7ae2e…`、`pubmedqa_sample.py` `5eb2d811…` | 已记录 |
| 冻结样本 / 协议 / 历史报告 | — | **本轮未修改**（见 §4） |

**当前环境实测（不代表运行当时）**：Python 3.13.9 / arm64 / Darwin。`requirements/ci.lock` sha256 = `126d8ef0199aa505a01421928fc90f11afdb45ad557466fd35bb814c2840e9ef`；`requirements/heavy.lock` sha256 = `d00a2ad49adeb46896af3bf8b38274c6c17616d1cb03bd1f37dc10f8e0569a2d`。这些值是**编制本文件时**测得的，与 v3 运行无绑定关系。

---

## 3. 已知限制

1. **读者模型版本未独立核验**。五例诊断分类由单一 LLM 读者提交，经多轮 Codex 监督反馈修订。Codex 事先已知 v3 实验结果，**不能**计作第二名独立盲读者。**未测量独立标注者一致性。**
2. **v1 费用估算有一次使用量缺口**。`PUBMEDQA-TEST-0402`（`APITimeoutError`）的 `llm_cost_status = "partial"`、`llm_cost_usd = null`，已取得小计 \$0.0021603。该缺口**保留**，未插补；`cost_complete = false` 如实标注。**部分用量下，已知小计对整轮实际账单的界限方向未知**（见 §1.3）。
3. **费用一律是估算，不是账单**。完整用量时，按峰值 cache-miss 计价在该费率假设下是保守估算，但**不是账单硬上限**；部分用量时界限方向未知。
4. **v1 无冻结协议文件，无提示词指纹**（现有归档未提供）。因此无法对 v1 做字节级一致性断言，**不倒填**，也不推断当时是否另有未归档记录。
5. **v1/v2 的验收信息现有归档未提供**。只有 v3 有预声明的验收条件与阈值。不得把「三轮都做了预声明验收」写成事实；也不得据此推断 v1/v2 当时从未设定任何验收安排。
6. **三轮样本与评分口径不同**：池 500/450/400，v3 换用 v3 映射。三个 macro-F1 **不可**直接比较或合并成普遍结论。
7. **v1 两臂在同一计划样本上覆盖数不同**：A 的 macro-F1 覆盖 49 例、规则臂覆盖 50 例（1 例 `errored` 被排除）。比较两臂时须注意覆盖集不完全相同。
8. **原始对账文件的计数已被更正**。`private/v3/audit/reconciliation.md` 记为「其余 64 次」；正确为 **65 次成功**（66 次非 finish 含 1 次失败）。**原文件保留不改**，更正见 `private/v3/audit/reconciliation_verified_v1.md`。
9. **预算阈值是软阈值**。报告 `config.budget_enforcement` 明文：每例开始前检查，无法约束账单，在途案例可超出自身费用。
10. **v1 的延迟均值不覆盖全部案例**。`metrics_notes.avg_latency_ms_coverage` 记为「45 例」，非 50 例。
11. **v1 的检索条件现有归档未提供**（见 §2.1）。只有 v2/v3 协议对 A 臂检索的追溯声明。
12. **现有归档未提供的项汇总**：Python 版本、依赖版本、v1 提示词指纹、v1/v2 验收阈值与 `criteria_met`、v1 检索条件、v1 规则臂训练划分。均**不得**补造。

---

## 4. 本轮改动声明

**本轮未修改**：任何冻结样本、冻结协议、历史报告、评分实现、Agent 代码。
**本轮未执行**：API 调用、密钥读取、案例重跑、发布清单刷新、提交或推送。

**本轮编辑的文件**（仅声明本文件自身的编辑；`git status` 不能证明改动归属，工作树中同时存在其他未提交内容）：
- `docs/claims_registry.md`（修订）
- `docs/experiment_log.md`（本文件，修订）
- `reports/agent_experiments_summary_v1.json`（修订）
- `private/v3/diagnostic_5case/README.md`（新建）

**项目记忆**：项目内**未找到**记忆文件（无 `CLAUDE.md` / `.claude/` / `AGENTS.md`）。根目录 `EXPERIMENT_LOG.md` 与 `DECISION_LOG.md` 是 2026-08 的 v0.3.0 期日志，不含 v1/v2/v3 内容。**本轮未做记忆更新**，也未创建新的记忆文件。

---

## 5. 公开摘要与私有轨迹的审计范围差异

`reports/agent_experiments_summary_v1.json` 只含聚合指标、配置与来源哈希，**不含**逐例轨迹、摘要原文或引用跨度。

**公开摘要允许检查的范围**：口径、内部计算过程，以及其中**已声明的**来源哈希。
**超出该范围、需要取得相应原始文件才能做的**：核验某个哈希是否确实对应原文件、某个指标是否被正确提取。

因此：**摘要不能独立证明自身来源**——它引用来源，不替代来源。单例轨迹的审计须在具备 `private/` 访问权的环境内进行。
