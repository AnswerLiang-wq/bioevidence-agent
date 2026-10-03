# 主张登记表 v1

本文件是 BioEvidenceAgent 三轮评估实验（v1 / v2 / v3）所有对外表述的**单一事实来源**。
任何报告、作品集或口头叙述中出现的结论，都必须能在此表中找到对应行；表中没有的结论不得表述。

- 编制日期：2026-10-03（第 2 版，同日修订）
- 事实来源：`private/` 下的原始报告与冻结协议（见各行「支撑来源」）。记忆与会话摘要**不是**事实来源。
- 范围：只登记已有记录支撑的主张。**未记录**的事项一律标记为未记录，不推测、不补造。
- 本文件不含任何未公开数据；涉及的原始轨迹位于被忽略的 `private/` 目录。
- **改动归属**：本文件只声明自身被编辑；**不**以工作树状态作为本轮改动范围的证明。工作树中同时存在
  其他未提交内容，`git status` 既不能证明也不能否定某项改动由本轮产生。

## 强度分级

| 分级 | 含义 |
|---|---|
| 可直接引用 | 有原始 JSON 字段或冻结协议条文直接支撑，可原样引用 |
| 描述性观察 | 有记录支撑，但只能作为该轮该样本的描述，不得作因果推断或跨轮推广 |
| 待验证 | 假设或推断，必须带 [INFERRED, post-hoc] 标记，不得作为事实陈述 |
| 已撤回 | 曾有表述但无支撑或已被推翻，**不得**再出现在任何交付物中 |

## 来源与提取方式（区分两类）

| 类别 | 含义 | 本表中对应 |
|---|---|---|
| **直接提取** | 值由脚本从报告 JSON 字段读出，未做任何计算 | D-02～D-16、D-21～D-25 的评分与配置数值 |
| **由记录汇总** | 值通过遍历逐例记录计数得到，方法已写明 | D-18、D-19（诊断子集工具调用统计，逐例 `tool_calls` / `tool_returns` 配对计数） |

**「未记录」的准确含义**：本表中的「未记录」「现有归档未提供」指**本次可查阅的归档中不存在该项**，
**不等于**该事项在运行当时从未发生或从未被处理。两者不可互相推断。

**审计范围**：本表可核验的是口径、内部计算与已声明的来源哈希。
核验某个哈希是否对应原文件、某个指标是否被正确提取，**需要取得相应原始文件**。
本表**不能**独立证明自身来源；它引用来源，不替代来源。

---

## 一、可直接引用的事实

| # | 主张文本 | 强度 | 支撑来源 |
|---|---|---|---|
| D-01 | 三轮实验的 API 模型均为 `deepseek-flash`；thinking disabled、temperature=0.0、max_tokens_per_call=1024、timeout=30s、SDK max_retries=0 | 可直接引用 | 各轮报告 `config`；v2 协议 §4；v3 协议 §5 |
| D-02 | v3 的 Δmacro-F1（C − A）= **0.037372796459921864**，低于预声明的 0.04 阈值 | 可直接引用 | `v3/v3_ac_50_final.json` → `acceptance.delta_macro_f1_c_minus_a`、`acceptance.delta_threshold` |
| D-03 | v3 联合判据 `criteria_met = false`；`macro_f1_gain_met = false`；结论文本为 "exploratory engineering criterion not met" | 可直接引用 | 同上 → `acceptance.criteria_met`、`acceptance.macro_f1_gain_met`、`acceptance.conclusion_strength` |
| D-04 | v3 的第二个验收条件（C 在 gold=maybe 上正确率高于 A）**成立**：`c_exceeds_a = true`。判据未满足的原因是**第一个**条件 | 可直接引用 | 同上 → `acceptance.maybe_correct_rate.c_exceeds_a`、`acceptance.macro_f1_gain_met` |
| D-05 | v3 gold=maybe 案例正确数：A = 1/5（0.2），C = 2/5（0.4），支撑案例数 5 | 可直接引用 | 同上 → `acceptance.maybe_correct_rate` |
| D-06 | v3 逐类 F1：yes A 0.7547 / C 0.7843；no A 0.8125 / C 0.7879；maybe A 0.1429 / C 0.2500 | 可直接引用 | 同上 → `acceptance.per_class_f1` |
| D-07 | v3 的 yes/no 误转 maybe 数：A yes→maybe 6、no→maybe 2；C yes→maybe 6、no→maybe 3 | 可直接引用 | 同上 → `acceptance.maybe_misroutes` |
| D-08 | v3 精确 macro-F1：A = 0.5700247079964061，C = 0.607397504456328 | 可直接引用 | 同上 → `acceptance.macro_f1_a_exact`、`macro_f1_c_exact` |
| D-09 | v3 计划 50 例、已尝试 50、未运行 0；其中 **A 臂完成 49、失败 1**，**C 臂完成 50、失败 0**；`partial = false` | 可直接引用 | 同上 → `plan`、`metrics.llm_agent`（`n_completed`/`n_failed`）、`metrics.arm_c`（`n_completed`/`n_failed`） |
| D-10 | v3 唯一失败案例为 `PUBMEDQA-TEST-0043`（A 臂，`budget_exhausted`，gold=no），其判分按预声明规则只贡献该 gold 类的 FN | 可直接引用 | 同上 → `cases[].llm_run_status`；v3 协议 §2 |
| D-11 | v3 验收覆盖全部 50 个已尝试案例（逐类 support 合计 = 50，而 A 完成数为 49） | 可直接引用 | 同上 → `acceptance.frozen_total`、`metrics.llm_agent.macro_f1_detail.per_class` |
| D-12 | v1 Agent A macro-F1 = 0.5638，规则基线 = 0.4196。两臂使用**同一计划样本**（同 50 例），但 legacy 口径下 macro-F1 **实际覆盖的案例数不同**：A 覆盖 49 例（1 例 `errored` 被排除），规则臂覆盖 50 例 | 可直接引用 | `env_precheck/pipeline_comparison_50_final.json` → `metrics.llm_agent.macro_f1`、`metrics.rule_based.macro_f1`、各自 `macro_f1_detail.per_class` 的 support 合计 |
| D-13 | v2 Agent A macro-F1 = 0.6003，固定上下文臂 B = 0.5388，规则基线 = 0.4216（同轮、同样本、同 legacy 口径） | 可直接引用 | `env_precheck/v2_ab_50_final.json` → `metrics.llm_agent.macro_f1`、`metrics.baseline.macro_f1`、`metrics.rule_based.macro_f1` |
| D-14 | v3 Agent A macro-F1 = 0.5700，规则基线 = 0.3344（同轮、同样本、v3 口径） | 可直接引用 | `v3/v3_ac_50_final.json` → `metrics.llm_agent.macro_f1`、`metrics.rule_based.macro_f1` |
| D-15 | v1 有 1 例 `errored`（`PUBMEDQA-TEST-0402`，`APITimeoutError`），legacy macro-F1 中该臂只覆盖 49 个完成案例；同轮的规则臂覆盖其全部 50 例 | 可直接引用 | v1 报告 → `cases[].llm_run_status`、`metrics.llm_agent.macro_f1_detail.per_class`（support 合计 49）、`metrics.rule_based.macro_f1_detail.per_class`（support 合计 50） |
| D-16 | v2 全部 50 例完成，无失败 | 可直接引用 | v2 报告 → `plan.completed = 50`、`plan.failed = 0` |
| D-17 | 诊断子集为 **5 个唯一案例、10 条臂—案例记录**（TEST-0146/0150/0248/0260/0487 各 A、C 两行） | 可直接引用 | `v3/v3_ac_50_final.json` → `sample.apportionment.strata[label=maybe].selected` |
| D-18 | 诊断子集工具调用：非 finish 共 **66** 次（65 次成功、1 次失败），另有 **10** 次 `finish`，合计 **76** 次 | 可直接引用 | 同上 → 各例 `llm_tool_returns` / `arm_c_tool_returns` 与 `*_audit.tool_calls` |
| D-19 | 那次失败为 `PUBMEDQA-TEST-0248` A 臂第 11 条证据调用 `fetch_record(pmid=16147837)`，返回 `tool-call budget exhausted` | 可直接引用 | 同上 |
| D-20 | 十行诊断记录的 `run_status` 均为 `completed`，来源为原始 JSON | 可直接引用 | 同上 → 各例 `llm_run_status`、`arm_c_run_status` |
| D-21 | 三轮费用均为**按记录的峰值 cache-miss 费率计算的估算**，不是账单：v1 A \$0.18653（`cost_complete = false`，49 confirmed / 1 partial）；v2 A \$0.196013 + B \$0.044505，报告总额 \$0.240517；v3 A \$0.204216 + C \$0.235161 = \$0.439377 | 可直接引用 | 各报告 → `metrics.*.cost`、`merge_notes.known_subtotal_usd` |
| D-22 | v1 的使用量缺口保留：`PUBMEDQA-TEST-0402` 的 `llm_cost_status = "partial"`、`llm_cost_usd = null`、已取得小计 \$0.0021603。**未**插补、**未**改写成完整账目 | 可直接引用 | v1 报告 → `cases[]` 对应字段；`metrics.llm_agent.cost.usage_coverage_counts` |
| D-26 | 费用的界限方向取决于用量是否完整，两者语义不同（见第九节）：**用量完整**时，估算是在峰值 cache-miss 费率假设下的**保守估算**，但仍**不是**账单硬上限；**用量部分**时，已知小计只覆盖服务端报告了用量的轮次，对整轮实际账单的**界限方向未知** | 可直接引用 | 各报告 → `cost.semantics`、`cost.known_subtotal_semantics`、`cost_complete` |
| D-27 | v2 的费用展示存在**舍入差异**，不是计费差异：逐例未舍入小计合计 = **0.2405172**（A 0.1960125 + B 0.0445047），报告展示总额 = **0.240517**（对未舍入合计四舍五入），而两臂各自舍入后的展示值相加 = 0.196013 + 0.044505 = **0.240518**。三者中只有前两者一致；引用时**不得**写成 `0.196013 + 0.044505 = 0.240517` | 可直接引用 | v2 报告 → 逐例 `llm_cost_known_subtotal_usd` / `baseline_cost_known_subtotal_usd` 求和，与 `merge_notes.known_subtotal_usd` 对比 |
| D-23 | 三轮样本互不重叠，且各自与更早轮次的已尝试集合不重叠：v1 池 500、v2 池 450、v3 池 400 | 可直接引用 | 各报告 → `sample.apportionment.population`、`sample.verification` |
| D-24 | v3 冻结样本文件 sha256 = `3f0eacad…`，与协议 §4 记录一致；四份输入数据哈希与协议 §6 记录一致 | 可直接引用 | `v3/v3_protocol_frozen_20261002.md` §4/§6；v3 报告 → `input_hashes` |
| D-25 | v3 的 C 臂干预为**纯插入**：把插入段移除后与 A 的提示词逐字相同（`system_prompt_delta_is_insertion_only = true`） | 可直接引用 | v3 报告 → `prompt_fingerprint.arm_c.intervention` |

## 二、描述性观察（不作因果推断）

| # | 主张文本 | 强度 | 支撑来源 |
|---|---|---|---|
| O-01 | **各轮分别看**，Agent A 的 macro-F1 高于**同轮同样本同口径**的规则基线（v1 0.5638>0.4196；v2 0.6003>0.4216；v3 0.5700>0.3344）。这是三条**独立的逐轮描述**。注意 v1 两臂在同一计划样本上覆盖的**案例数不同**（A 49 / 规则 50，见 D-12） | 描述性观察 | D-12/D-13/D-14 三行 |
| O-02 | **不得**把 O-01 的三条合并表述为「三轮一致显示普遍优势」。三轮使用**不同样本**（池 500/450/400）且 v3 使用**不同评分口径**（v3 政策 vs legacy），三个数字不可直接相加、平均或互为佐证 | 描述性观察 | 各报告 `sample`、`metrics.scoring_policy` |
| O-03 | **不得**把 O-01 表述为「循环带来提升」。v1 没有固定上下文对照臂；v2 的 A 与 B 证据输入不同，协议将比较范围限定为**策略层面的差异**（含证据选择），**不能**归因于循环本身。该轮指标较高不等于已证明稳定优势；v3 的 A/C 配置差异仅为提示词材料 | 描述性观察 | v2 协议 §1；v3 协议 §1 |
| O-04 | v3 中 C 相对 A 在 gold=maybe 上多正确 1 例（1/5 → 2/5），代价是 no→maybe 多 1 例（2 → 3）；yes→maybe 两臂同为 6 | 描述性观察 | D-05、D-07 |
| O-05 | v3 中 verdict 不同的案例：TEST-0146（A=supported / C=mixed）、TEST-0260（A=supported / C=mixed）、TEST-0487（A=mixed / C=contradicted）；TEST-0150、TEST-0248 两臂相同（均 supported） | 描述性观察 | `v3/audit/reconciliation_verified_v1.md` 逐臂表 |
| O-06 | 五个诊断案例的 A/C 读者结构分类完全相同；差异只出现在模型 verdict 层 | 描述性观察 | `v3/audit/classification_reader_v1.md` 提交摘要表 |
| O-07 | v3 的 C 臂在 yes 类 F1 上升（+0.0296）、no 类下降（−0.0246）；净 macro-F1 仍为上升。同时 yes/no 误转 maybe 数量在 no 侧增加 | 描述性观察 | D-06、D-07 |
| O-08 | v2 的 A 相对 B 的差（0.6003 vs 0.5388）是**系统表现比较**，不是「模型语言理解能力」的单独贡献 | 描述性观察 | v2 协议 §1 表 |

## 三、待验证事项（必须标注 [INFERRED, post-hoc]）

| # | 事项 | 强度 | 支撑来源 / 为何未确证 |
|---|---|---|---|
| P-01 | loop（工具循环）对准确率的贡献 | 待验证 | [INFERRED, post-hoc]。v2 的 A/B 证据输入不同，A 相对 B 的优势**不能**拆分为循环与证据选择两部分；需另行设计对照臂 |
| P-02 | TEST-0174 的 verdict 归因（判定对象是否被重写） | 待验证 | [INFERRED, post-hoc]。`env_precheck/v2_case_audit_0174_0236.md`：报告未保存内部推理，无法区分「模型内部改写」与「模型按自己发出的 claim 判定」。可确证的仅为：verdict 与其自身 answer 极性不一致 |
| P-03 | TEST-0236 的 verdict 归因 | 待验证 | [INFERRED, post-hoc]。同上文件：gold=maybe 的标注理由不在任何报告中，无法判定「maybe 是否更合适」 |
| P-04 | TEST-0487 的「混合结构」分类语义有效性 | 待验证 | [INFERRED, HIGH]。`classification_review_note_v1.md`：该类别仍存在监督争议，监督者未确认其为无争议事实 |
| P-05 | gold=maybe 案例的标注原因（PubMedQA 为何标 maybe） | 待验证 | 报告不含标注说明或作者结论。仅能确证「两臂 yes 预测与 gold=maybe 不一致」这一事实 |
| P-06 | H4（verdict-target 失效）整体假设 | 待验证 | 仅在 TEST-0174 单例上观察到与假设**一致**的现象，未经确证，且未在其余案例上检验 |
| P-07 | 是否存在某个案例的共同错误机制 | 待验证 | 标签定义不一致**不等于**每个 maybe 错误的共同原因已被确认；检索命中也不证明答案得到语义支持 |

## 四、已撤回主张（不得再出现于任何交付物）

| # | 已撤回的主张 | 撤回原因 |
|---|---|---|
| W-01 | 「读者 6/10 优于模型 4/10」 | 本次**没有预声明的读者答案评测臂**，读者准确率**未被测量**。旧表述中的 6/10 依赖未经预声明的结构分类→答案映射，不构成有效的读者评分。模型侧实际为 **3/10 条臂—案例记录**（A 1/5 + C 2/5），4/10 是误计；不得复用这组比较 |
| W-02 | 「结构分类自动映射到 verdict / 答案」 | 冻结的 v3 映射**只作用于实际 verdict**（supported→yes、contradicted→no、mixed/insufficient→maybe）。协议中**不存在**结构分类→答案的自动映射。结构分类是诊断描述，不进入评分 |
| W-03 | 「END_OF_ARM 证明运行完成」 | `END_OF_ARM` 只是诊断文件的完整性标记，**不能**证明 Agent 的完成状态。完成状态取自原始 JSON 的 `run_status`（十行均为 `completed`） |
| W-04 | 「三轮实验含预声明验收标准」 | 只有 **v3** 有预声明的验收条件与阈值。v2 冻结协议**无**验收阈值（§3 仅为评分口径，§9 的 \$0.002 是费用软阈值）。**v1 的验收信息现有归档未提供**——v1 报告无 `acceptance` / `criteria_met` 字段，现有归档中也没有 v1 冻结协议。此处只陈述归档缺口，**不推断** v1 当时是否另有未归档的验收安排 |
| W-05 | 「v2 / v3 运行记录缺少 prompt_fingerprint」 | 与文件不符：v2 的两份原始分段报告**都有**指纹块（合并报告没有）；v3 的两份分段与最终合并报告**都有**。只有 **v1** 全系列无指纹，且该缺失在 v2/v3 报告中被记为 `historical_v1: "not recorded"` |

## 五、诊断方法说明（固定表述）

- 方法为「**单一 LLM 读者、经监督反馈修订**」。
- **「未测量独立标注者一致性」是可以如实写出的表述**——它陈述的是测量缺口，不是能力主张。
- **禁止**的是**虚假的肯定性声明**：不得声称已取得一致性数据、不得声称存在第二名独立标注者、
  不得使用「双盲」「独立双人标注」等描述本次并不存在的方法。禁止的是不实肯定，不是对缺口的陈述。
- 监督者（Codex）**事先已知 v3 实验结果**，因此不能计作第二名独立盲读者。
- 读者实际使用的模型版本**未独立核验**。
- 五例诊断**不覆盖、不替换** 50 例实验结论，也不改变样本、评分或预声明阈值。

## 六、事实修订记录（相对桌面端设计原文）

| # | 设计原文表述 | 修订后 | 依据 |
|---|---|---|---|
| R-01 | 「API 模型 ID（`claude-opus-5-5` / `claude-haiku-4-5` 等）」 | 三轮实验的 API 模型**均为 `deepseek-flash`**。设计者/代码助手的模型**不是**实验模型 | 各报告 `config.model`；v2 协议 §4；v3 协议 §5 |
| R-02 | 「v2/v3 运行记录中缺少 prompt_fingerprint」 | v1 全系列无；v2 两份**分段有**、合并报告无；v3 分段与合并**都有** | 见 W-05 的文件级核验 |
| R-03 | 「三轮独立实验…一致显示 LLM Agent 策略 Macro-F1 优于规则基线」（列为可直接引用） | 降为**逐轮描述性观察**（O-01/O-02）。不同样本与评分口径**不能**合并为普遍优势证明 | 各报告 `sample` 与 `metrics.scoring_policy` |
| R-04 | 「三轮含预声明验收标准」 | 仅 v3 有；**v1/v2 的验收信息现有归档未提供**（v2 协议无验收阈值，v1 报告无该字段且现有归档无 v1 协议） | 见 W-04 |
| R-05 | 「原始 reconciliation.md 的工具调用计数（64 次）」 | 应为 **65 次成功**（66 次非 finish 中 1 次失败）。原文件保留不改，由 `reconciliation_verified_v1.md` 单独更正 | `v3/audit/reconciliation_verified_v1.md` |
| R-06 | 输入路径 `private/v3/reconciliation_verified_v1.md` | 实际路径为 `private/v3/audit/reconciliation_verified_v1.md` | 文件系统核验 |
| R-07 | Δmacro-F1 引用为 `0.03737279645992186` | 全精度值为 **`0.037372796459921864`**（末位 4）。引用取全精度或用 0.0374 舍入值 | `v3_ac_50_final.json` → `acceptance.delta_macro_f1_c_minus_a` |
| R-08 | 「评分映射差异（例如 maybe→insufficient vs maybe→maybe）」 | 表述不准确，见第七节映射表原文 | v2 协议 §3；v3 协议 §2 |
| R-09 | 「记忆文件已更新」 | 项目内**未找到**任何记忆文件（无 CLAUDE.md / .claude/ / AGENTS.md）；根目录 `EXPERIMENT_LOG.md`、`DECISION_LOG.md` 是 2026-08 的 v0.3.0 期日志，不含 v1/v2/v3 内容。**本轮未做记忆更新** | 全项目搜索核验 |

## 七、评分口径原文（预声明，不得改写）

**legacy（v1、v2 使用）**
- `supported → yes`、`contradicted → no`、`insufficient → maybe`
- **`mixed` 不匹配任何 gold**，计为不正确，并单独统计 `n_mixed_verdicts`
- macro-F1 **仅覆盖完成案例**；`errored` / `budget_exhausted` / `text_exit` 不进 macro-F1
- 每类 `2TP/(2TP+FP+FN)`，分母为 0 取 0，三类固定平均
- **逐臂覆盖数可以不同**：v1 两臂使用同一计划样本，但 A 有 1 例 `errored`，故 macro-F1 覆盖 **A 49 例 / 规则臂 50 例**；v2 两臂均覆盖 50 例

**v3（仅 v3 使用）**
- `supported → yes`、`contradicted → no`、**`mixed → maybe`**、**`insufficient → maybe`**
- macro-F1 **覆盖全部已尝试案例**；失败案例**只**给其 gold 类 +1 FN，不贡献任何 TP 或 FP
- `run_status` 检查在 verdict 检查**之前**；失败时返回的 `insufficient` **不是**正确的 maybe 预测
- v1/v2 结果**未被改写**，legacy 映射保留为独立实现且仍是默认值

## 八、检索条件与来源（区分不同实验）

三轮 LLM 实验与更早的检索基准是**不同的实验**，条件不可互换。以下逐项注明来源。

| 项 | 值 | 来源 | 可确认性 |
|---|---|---|---|
| 语料规模 | **1,000 文档**（closed corpus） | `data/pubmedqa/v1/manifest.json` → `counts.all`；`reports/pubmedqa_title_assisted_v1.json` → `corpus_document_count` | 已确认 |
| 测试集 | 500 例（yes 276 / no 169 / maybe 55） | 同上 → `counts.test`、`counts.test_labels` | 已确认 |
| 训练集 | 500 例（非测试记录） | 同上 → `counts.train`；`data/pubmedqa/v1/train.jsonl` 行数 500 | 已确认 |
| 评测性质 | public closed-corpus，**非盲**；官方测试标签公开 | 同上 → `evaluation_scope`、`limitations` | 已确认 |
| **v2/v3 A 臂检索（协议声明）** | **BM25，title-assisted**：`CorpusDocument` 含 title，与剥离 title 的 `pubmedqa_stress.py` 路径不同 | v2 协议 §4；v3 协议 §1 声明 A「与 v2 相同」 | v2/v3 协议已声明；v1 见下一行 |
| v1 的检索条件 | — | v1 报告**未记录**检索条件（`bm25`/`hybrid`/`retrieval` 在报告中出现 0 次）。v2 协议 §4 以「与 v1 相同」追溯声明，v2 设计交接亦称 A 臂「same retrieval」 | **v1 本身未确认**（仅有后续协议的追溯声明） |
| B 的检索 | 与 A **同一索引**，对原始问题做**一次静态 BM25 检索，固定 top-4 摘要**作为上下文块，单次模型请求，无工具循环 | v2 协议 §1、§4；v2 报告 `config.baseline_arm`（`top_k=4`、`tool_loop=false`、`requests_per_case=1`） | 已确认 |
| B 的 `top_k=4` 依据 | v1 完成案例中每例不同 fetched PMID 数的**中位数为 4**（分布 2:7、3:11、4:29、6:1、8:1） | v2 协议 §4 | 已确认 |
| 规则臂训练范围 | 协议声明「现有 `TfidfLogisticAnswerer`，**仅用 train 拟合**」 | v2 协议 §1 表 | 协议已声明；**v1 报告的 `metrics.rule_based` 中无训练范围字段** |
| 规则臂在 `train.jsonl` 上的外部记录 | 500 条非测试记录、分层 5 折、seed `20260729` | `reports/pubmedqa_evidence_utilization_cv_v1.md` | 已确认（属**负对照实验**，非本轮三轮实验） |
| **标题可见条件** | 四个系统（bm25 / vector / hybrid_rrf / hybrid_reranked）的 Recall@5、Recall@10、MRR、nDCG@10 **均为 1.000**。问题是标题或标题派生，索引文档含该标题，属**有利的源定位条件** | `reports/pubmedqa_title_assisted_v1.json` → `retrieval_metrics`、`interpretation_limits` | 已确认 |
| **abstract-only 压力条件** | 标题对所有检索器不可见。bm25 Recall@5 0.9760 / MRR 0.9532；hybrid_reranked 0.9900 / 0.9820。与 title-assisted 的 MRR 差：bm25 −0.046813、hybrid_reranked −0.018 | `reports/pubmedqa_abstract_only_retrieval_v1.json` → `retrieval_metrics`、`comparison_with_title_assisted_v1` | 已确认 |
| **旧混合检索** | `hybrid_rrf`（RRF 融合）与 `hybrid_reranked`（加 cross-encoder）的这些指标来自**旧检索基准实验**，不能用其说明 v1/v2/v3 的 Agent 检索表现。v2/v3 检索条件按各自协议记录；v1 保持未确认 | 同上两份报告；v2/v3 协议 | 旧报告所属实验已确认；不据此补造 v1 路径 |
| 旧标题可见条件下的答题基线 | bm25 答题：accuracy 0.558（279/500）、macro-F1 **0.383221**、maybe F1 0.0625 | `reports/pubmedqa_title_assisted_v1.json` → `answer_metrics.bm25` | 已确认 |

> **不得混淆**：上表最后一行的 **0.383221** 是 500 例全部测试集上的旧答题基线，与本表 D-12 的
> v1 规则基线 **0.4196**（v1 的 50 例样本、legacy 口径）**不是同一个量**，来源、样本与口径均不同。

**无法从现有归档确认、且不得以当前代码补造的项**：
- v1 的实际检索实现与索引条件（仅有 v2/v3 协议的追溯声明）。
- v1 规则臂的实际训练数据划分（协议声明「仅用 train 拟合」，但 v1 报告无对应字段）。

## 九、费用语义（必须区分两种情况）

| 情况 | 含义 | 界限方向 |
|---|---|---|
| **完整用量**（`cost_complete = true`） | 已知小计按记录的**峰值 cache-miss 费率**计算 | 在该费率假设下是**保守估算**；**不是**账单硬上限。谷时约一半、缓存命中更便宜，故真实账单可能更低，但无发票可核 |
| **部分用量**（`cost_complete = false`） | 已知小计**只覆盖服务端报告了用量的轮次** | 对整轮实际账单的**界限方向未知**——未报告的轮次可能推高也可能不推高，无法据已知小计判断 |

- 报告 `cost.semantics` 原文：估算按峰值 cache-miss 计价，完整报告的运行给出的是**保守上限而非下界**。
- 报告 `cost.known_subtotal_semantics` 原文：已知小计**只覆盖服务端报告了用量的轮次**，失败或未报告用量的轮次不在其中，**整轮真实费用未知**。
- **未知用量不当作零，也不插补**。v1 的 `PUBMEDQA-TEST-0402` 即属部分用量（见 D-22）。
- 各轮 `cost_complete`：v1 A = `false`；v2 A/B = `true`；v3 A/C = `true`。
- 展示舍入差异见 D-27；**不得**把舍入差异写成计费差异。

## 十、未解决缺口（供桌面端决策）

| # | 缺口 | 说明 |
|---|---|---|
| G-01 | 三轮实验的解释器版本与依赖版本**现有归档未提供** | 报告里没有 Python 版本、包版本或 lock 文件哈希（已逐报告核验）。当前环境实测值不能冒充历史快照 |
| G-02 | v1 全系列的 prompt / tool-schema 指纹**现有归档未提供** | 无法对 v1 做字节级一致性断言。**不倒填**，也不推断当时是否另有未归档记录 |
| G-03 | 诊断材料的方法学强度有限 | 单一 LLM 读者 + 已知结果的监督者，无独立标注者一致性 |
| G-04 | TEST-0043（v3 A 臂 budget_exhausted）的轨迹未被诊断 | 该例不属 gold=maybe 子集，其失败原因未做逐例分析 |
| G-05 | 设计原文引用的「项目记忆中已归档的对账记录」不存在 | 该内容没有对应文件；本表的对账内容直接来自 `reconciliation_verified_v1.md` |
