# superEasternBlott —— 2026 合成生物学创新赛蛋白设计赛道设计文档

> **队伍名称**：superEasternBlott（超级东方杂交）
>
> **项目名称**：spEBT —— 大语言模型驱动的交互式 GFP 蛋白设计 Agent
>
> **提交日期**：2026 年 7 月

---

## 目录

1. [项目概述](#1-项目概述)
2. [算法管线总览](#2-算法管线总览)
3. [数据与模型](#3-数据与模型)
4. [LLM Agent 架构](#4-llm-agent-架构)
5. [候选生成策略](#5-候选生成策略)
6. [评分与排名体系](#6-评分与排名体系)
7. [最终提交序列](#7-最终提交序列)
8. [关键执行日志](#8-关键执行日志)
9. [可复现性说明](#9-可复现性说明)

---

## 1. 项目概述

### 1.1 设计目标

在 **全序列自由设计** 的开放条件下，设计 6 条新型绿色荧光蛋白（GFP）序列，同时追求 **极高初始亮度** 与 **极限热稳定性（72°C）**——打造 GFP "六边形战士"。

### 1.2 核心思路

我们将 **大语言模型（LLM）的规划与推理能力** 与 **确定性蛋白计算工具链** 深度结合，构建了一个可交互的 AI Agent。Agent 接收自然语言任务指令，自主制定执行计划、调度底层工具、读取中间结果、在失败时自动降级恢复——最终产出竞赛提交序列。

设计哲学：
- **LLM 负责"想"**：任务理解、计划生成、结果解释
- **工具链负责"算"**：突变生成、过滤、ESMC 嵌入、亮度预测、稳定性评分、排名
- **两套突变策略互补**：精准单点/组合突变保底 + ProteinMPNN 结构逆折叠探索

---

## 2. 算法管线总览

### 2.1 整体架构

```
                      ┌─────────────────────────┐
  用户自然语言 ──────▶│  交互式 Agent 层         │
  "先检查再跑设计"    │  Planner → Executor      │
                      │  (LLM 驱动的计划与调度)  │
                      └───────────┬─────────────┘
                                  │ 调用标准化工具接口
                      ┌───────────▼─────────────┐
                      │  工具注册表 (13 工具)     │
                      │  prepare / health /      │
                      │  run_design / read /     │
                      │  export / train / etc.   │
                      └───────────┬─────────────┘
                                  │
        ┌─────────────┬───────────┼───────────┬─────────────────┐
        ▼             ▼           ▼           ▼                 ▼
   ┌─────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐   ┌──────────┐
   │数据准备  │ │Stage1    │ │Stage2    │ │最终排名  │   │提交导出  │
   │亲本加载  │ │单突变生成│ │组合突变  │ │加权评分  │   │submission│
   │训练重建  │ │过滤+评分 │ │+Protein  │ │多样性选择│   │.csv +    │
   │          │ │+排名     │ │MPNN逆折叠│ │Top6      │   │报告      │
   └─────────┘ └──────────┘ └──────────┘ └──────────┘   └──────────┘
```

**[图1]** *建议配图：整体管线架构示意图。展示从"用户自然语言输入"到"submission.csv 输出"的完整数据流，用不同颜色区分 LLM Agent 层（蓝色）、工具注册层（绿色）、计算管线层（橙色）。*

### 2.2 管线阶段详解

| 阶段 | 输入 | 操作 | 输出 |
|------|------|------|------|
| **数据准备** | 5条亲本FASTA + GFP_data.xlsx + Exclusion_List.csv | 解析亲本、重建亮度训练数据、提取排除集 | reference_parents.csv, brightness_train/valid.csv |
| **Stage1 排名** | sfGFP 亲本序列 | 全位点单突变生成(3000-5000) → 硬规则过滤 → 排除过滤 → ESMC嵌入 → 亮度预测 → 稳定性评分 → 加权排名 | stage1_ranked.csv (Top 200 种子) |
| **Stage2 候选扩展** | Stage1 Top200 | 组合突变(2-5阶, 每阶800) + ProteinMPNN 逆折叠(385条) → 合并 → 过滤 | stage2_candidates.csv (4585条) |
| **Stage2 评分** | 全部候选 | ESMC嵌入 → 亮度预测(abs+delta) + 稳定性评分(ESMC PLL + ThermoMPNN ddG) | stage2_brightness.csv, stage2_stability.csv |
| **最终排名** | 评分结果 | 加权综合评分(brightness 0.45 + retention72 0.30 + ddG 0.10 + product 0.30) → chromophore 风险惩罚 | ranked_variants.csv |
| **Top6 选择** | 排名列表 | 分层多样性选择：4条小突变(≤2位点, Hamming≥2) + 2条大突变(≥4位点, Hamming≥4) | selected_top6.csv |
| **提交导出** | Top6 | 格式校验 → 排除列表比对 → 导出 | submission.csv |

**[图2]** *建议配图：Stage1→Stage2 漏斗筛选示意图。展示从 ~5000 单突变→Top200→~4585候选→Top6 的逐级浓缩过程，标注每级筛选标准和通过率。*

---

## 3. 数据与模型

### 3.1 训练数据

| 数据源 | 规模 | 用途 |
|--------|------|------|
| GFP_data.xlsx（官方提供） | 119,972 条训练 + 21,172 条验证 | 亮度预测模型训练 |
| AAseqs of 5 GFP proteins.txt | 5 条参考亲本 (sfGFP/avGFP/amacGFP/cgreGFP/ppluGFP) | 亲本序列锚定 |
| 2B3P.pdb | sfGFP 晶体结构 | ProteinMPNN 逆折叠模板 |
| Exclusion_List.csv | 135,414 条排除序列 | 提交合规校验 |
| beforetopseqs (sheet) | 20 条历年优秀序列 | 特征工程参考 |

### 3.2 计算模型

| 模型 | 来源 | 角色 | 运行环境 |
|------|------|------|----------|
| **ESMC-600M** | biohub/ESMC-600M (HuggingFace) | 蛋白序列嵌入（1152维） | GPU (RTX 4060 8GB), PyTorch 2.5.1 |
| **HistGradientBoostingRegressor** | scikit-learn 1.9.0 | 亮度预测（abs + delta 双模型） | CPU, 30k训练样本 |
| **ESMC 零样本 PLL** | ESMC-600M | 序列似然度 → 稳定性代理 | GPU |
| **ThermoMPNN** | 自建权重 | 热力学 ddG 预测 | 隔离 venv (spebt_stability) |
| **ProteinMPNN v_48_020** | Dauparas et al. | 结构逆折叠序列生成 | 隔离 venv (spebt_inverse_folding) |

**[图3]** *建议配图：模型生态图。展示 ESMC-600M（中心）、亮度预测器（左侧分支）、稳定性评分器（右侧分支，含 ESMC PLL + ThermoMPNN）、ProteinMPNN（底部逆折叠分支）的协作关系。*

### 3.3 亮度模型训练

- **特征**：ESMC-600M 真实蛋白嵌入（1152维） + num_mutations + parent_onehot（4维）= 1158维
- **算法**：HistGradientBoostingRegressor，超参搜索（learning_rate × max_depth × l2）
- **性能**：

| 目标 | R² | MAE | 训练样本 |
|------|-----|-----|---------|
| brightness (绝对亮度) | 0.660 | 0.406 | 30,000 |
| delta_brightness_vs_parent | 0.578 | 0.409 | 30,000 |

**[图4]** *建议配图：亮度模型预测 vs 真实值散点图（训练集+验证集双色标注），标注 R²=0.66 和理想对角线。*

---

## 4. LLM Agent 架构

### 4.1 为什么用 Agent

传统蛋白设计工具链需要用户依次手动执行 `prepare_data → health_check → run_design → export` 等多步 CLI 命令。我们将这些能力封装为标准化工具，由一个 LLM 驱动的 Agent 统一调度：

- 用户只需说"先检查环境，再跑完整搜索设计"
- Agent 自动：识别任务类型 → 生成执行计划 → 逐步调用工具 → 读取中间结果 → 失败自动降级恢复 → 中文总结

### 4.2 Agent 三组件

```
┌──────────────┐    ┌──────────────┐    ┌──────────────────┐
│   Planner    │───▶│   Executor   │───▶│   Summarizer     │
│  (任务→计划)  │    │  (逐步执行)   │    │  (结果→中文报告)  │
│  LLM 生成     │    │  硬编码恢复   │    │  LLM 生成        │
└──────────────┘    └──────────────┘    └──────────────────┘
```

**Planner**：接收自然语言任务 + 13 工具清单 → 输出结构化 JSON 计划（含步骤、参数、失败策略）。LLM 不可用时回退到 6 种内置任务模板。

**Executor**：逐一执行计划步骤，每次调工具后读取结果。失败恢复规则硬编码，LLM 不可绕过：

| 失败场景 | 自动恢复 |
|---------|---------|
| stage2_stability 失败 | resume → 降 esmfold_top_k → 降 max_stage2_candidates |
| ESMFold 不可用 | 继续（ESMC+ThermoMPNN 重新加权） |
| ProteinMPNN 缺 M 或含 X | 用亲本残基回填修复 |
| submission.csv 已生成 | 视为成功 |

**Summarizer**：将所有工具执行历史 + 产出文件 → 中文 Markdown 最终报告。

**[图5]** *建议配图：Agent 逻辑树。从用户输入"先检查环境再跑设计"开始，展示 Planner 如何生成 health_check→run_design 的计划、Executor 如何逐步执行、失败时如何降级重试的完整决策树。*

---

## 5. 候选生成策略

### 5.1 两路并行策略

我们采用 **"精准突变 + 结构逆折叠"** 双路并行策略，兼顾可解释性与探索广度：

| 路径 | 方法 | 探索空间 | 典型突变数 |
|------|------|---------|-----------|
| **路径A：精准突变** | 单点扫描 + 组合堆叠 (2-5阶) | ~5000 单点 × C(80,k) 组合采样 | 1-5 个 |
| **路径B：结构逆折叠** | ProteinMPNN 基于 2B3P 骨架重设计 | 全序列自由探索 | 200+ 个 |

### 5.2 组合突变的分阶采样

为避免组合爆炸（C(80,5)≈2400万），采用分阶预算分配 + 随机索引采样：

| 阶数 | 种子数限制 | 预算 | 实际生成 |
|------|-----------|------|---------|
| 2-mers | 80 | 800 | 800 |
| 3-mers | 80 | 800 | 800 |
| 4-mers | 50 | 800 | 800 |
| 5-mers | 40 | 800 | 800 |

### 5.3 ProteinMPNN 序列修复

ProteinMPNN 基于 PDB 结构生成序列时，N 端柔性区可能缺失起始 Met，无序环区（loop）可能输出 `X`。我们实现了自动修复管线：

- `X` 位置 → 用亲本 sfGFP 对应残基回填
- 缺失 Met → 强制补 `M`
- 长度不对齐 → 用亲本 C 端残基补齐

修复后 385 条全进入候选池。

**[图6]** *建议配图：双路径示意图。左侧展示精准突变路径（单点扫描 → 组合堆叠的漏斗），右侧展示 ProteinMPNN 逆折叠路径（PDB 输入 → 序列生成 → 修复 → 候选池合并）。*

---

## 6. 评分与排名体系

### 6.1 亮度预测

使用 ESMC-600M 提取 1152 维蛋白序列嵌入，拼接 num_mutations + parent onehot 特征，通过 HistGradientBoostingRegressor 预测：

- **绝对亮度 (abs)**：直接预测 GFP 在 Cell-Free 体系中的荧光强度
- **相对亮度 (delta)**：预测与亲本的亮度差异

### 6.2 稳定性评分（双信号）

由于 ESMFold 结构预测在 RTX 4060 (8GB) 上因 bf16 加载问题不可用，采用双重替代信号：

| 信号 | 来源 | 方法 | 权重 |
|------|------|------|------|
| ESMC 零样本 PLL | ESMC-600M | 序列对数似然度 → 折叠倾向代理 | 0.45→0.70¹ |
| ThermoMPNN ddG | ThermoMPNN | 热力学突变自由能变化 | 0.25→0.30¹ |

¹ ESMFold 不可用时重新加权。

### 6.3 综合评分公式

```
Stage1 得分 = 0.65 × 亮度 + 0.25 × 稳定性 - 0.40 × 发色团风险

Final 得分 = 0.45 × 亮度 + 0.30 × retention72 + 0.10 × ddG
           + 0.05 × 多样性 + 0.30 × 乘积得分² - 0.45 × 发色团风险

² 乘积得分 = 亮度 × retention72  (若亮度 < 0.30×sfGFP 则直接淘汰，得分为0)
```

**[图7]** *建议配图：雷达图展示最终 Top6 每条序列在 5 个维度（亮度、retention72、ddG、多样性、乘积得分）上的表现，用不同颜色区分小突变和大突变。*

---

## 7. 最终提交序列

### 7.1 序列总览

| Seq_ID | 标识 | 类型 | 突变数 | 预测得分 | 相对亮度 | 72°C保留率 | vs sfGFP差异 |
|--------|------|------|--------|---------|---------|-----------|-------------|
| 1 | sfGFP_N164W | 单点突变 | 1 | 0.859 | 0.901 | 0.663 | 1 aa |
| 2 | sfGFP_E142C | 单点突变 | 1 | 0.854 | 0.881 | 0.677 | 1 aa |
| 3 | sfGFP_K238V | 单点突变 | 1 | 0.838 | 0.884 | 0.661 | 1 aa |
| 4 | sfGFP_S2M | 单点突变 | 1 | 0.831 | 0.992 | 0.556 | 1 aa |
| 5 | sfGFP_S208V_K209W_T105Y_L7C | 4位点组合 | 4 | 0.799 | 0.971 | 0.461 | 4 aa |
| 6 | sfGFP_N164W_P211C_T9C_L7C | 4位点组合 | 4 | 0.774 | 0.915 | 0.460 | 4 aa |

### 7.2 序列详情

**Seq_1 — sfGFP_N164W**（单点突变）
```
MSKGEELFTGVVPILVELDGDVNGHKFSVRGEGEGDATNGKLTLKFICTTGKLPVPWPTLVTTLTYGVQCFSRYPD
HMKRHDFFKSAMPEGYVQERTISFKDDGTYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNFNSHNVYI
TADKQKNGIKAWFKIRHNVEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSVLSKDPNEKRDHMVLLEFVTAAG
ITHGMDELYK
```
> **设计理由**：N164W 引入色氨酸替换天冬酰胺——164 位邻近发色团（Tyr66-Gly67），大型芳香侧链可稳定发色团 π-π 堆叠微环境，在既往文献中与亮度提升高度相关。

**Seq_2 — sfGFP_E142C**（单点突变）
```
MSKGEELFTGVVPILVELDGDVNGHKFSVRGEGEGDATNGKLTLKFICTTGKLPVPWPTLVTTLTYGVQCFSRYPD
HMKRHDFFKSAMPEGYVQERTISFKDDGTYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLCYNFNSHNVYI
TADKQKNGIKANFKIRHNVEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSVLSKDPNEKRDHMVLLEFVTAAG
ITHGMDELYK
```
> **设计理由**：E142C 在 β-桶第7股引入半胱氨酸——Cys 可参与二硫键形成或局部疏水堆积，增强桶状结构的热稳定性。

**Seq_3 — sfGFP_K238V**（单点突变）
```
MSKGEELFTGVVPILVELDGDVNGHKFSVRGEGEGDATNGKLTLKFICTTGKLPVPWPTLVTTLTYGVQCFSRYPD
HMKRHDFFKSAMPEGYVQERTISFKDDGTYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNFNSHNVYI
TADKQKNGIKANFKIRHNVEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSVLSKDPNEKRDHMVLLEFVTAAG
ITHGMDELYV
```
> **设计理由**：K238V 替换 C 端带正电的赖氨酸为疏水缬氨酸——减少末端柔性区的溶剂暴露，有助于整体折叠紧密性。

**Seq_4 — sfGFP_S2M**（单点突变）
```
MMKGEELFTGVVPILVELDGDVNGHKFSVRGEGEGDATNGKLTLKFICTTGKLPVPWPTLVTTLTYGVQCFSRYPD
HMKRHDFFKSAMPEGYVQERTISFKDDGTYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNFNSHNVYI
TADKQKNGIKANFKIRHNVEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSVLSKDPNEKRDHMVLLEFVTAAG
ITHGMDELYK
```
> **设计理由**：S2M 在 N 端第 2 位用甲硫氨酸替换丝氨酸——增强翻译起始效率，并可能改善 N 端在无细胞体系中的共翻译折叠。

**Seq_5 — sfGFP_S208V_K209W_T105Y_L7C**（4位点组合突变）
```
MSKGEECFTGVVPILVELDGDVNGHKFSVRGEGEGDATNGKLTLKFICTTGKLPVPWPTLVTTLTYGVQCFSRYPD
HMKRHDFFKSAMPEGYVQERTISFKDDGYYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNFNSHNVYI
TADKQKNGIKANFKIRHNVEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSVLVWDPNEKRDHMVLLEFVTAAG
ITHGMDELYK
```
> **设计理由**：四重协同突变——L7C+引入二硫键锚点，T105Y+增强发色团 π-共轭，S208V/K209W 改造 β10-β11 环区疏水核心。亮度 0.971，综合得分 0.799。

**Seq_6 — sfGFP_N164W_P211C_T9C_L7C**（4位点组合突变）
```
MSKGEECFCGVVPILVELDGDVNGHKFSVRGEGEGDATNGKLTLKFICTTGKLPVPWPTLVTTLTYGVQCFSRYPD
HMKRHDFFKSAMPEGYVQERTISFKDDGTYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNFNSHNVYI
TADKQKNGIKAWFKIRHNVEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSVLSKDCNEKRDHMVLLEFVTAAG
ITHGMDELYK
```
> **设计理由**：四重协同——N164W+发色团稳定，L7C/T9C 双 Cys 引入潜在的 N 端二硫键锁定，P211C 在 β-桶出口增加结构刚性。亮度 0.915，综合得分 0.774。

### 7.3 合规性验证

全部 6 条序列已通过以下校验：

| 检查项 | 结果 |
|--------|------|
| 序列数量 ≤6 | ✅ 恰好 6 条 |
| 长度 220-250 aa | ✅ 全部 238 aa |
| 以 M 开头 | ✅ 全部 M 开头 |
| 仅 20 种标准氨基酸 | ✅ 无终止密码子、无非法字符、无标点 |
| 排除列表比对 (135,414条) | ✅ 无一命中 |
| 序列互不重复 | ✅ 6 条各不相同 |
| 表头格式 | ✅ Team_Name, Seq_ID(1-6), Sequence |

**[图8]** *建议配图：选择理由可视化。为每条序列标注突变位点在 sfGFP 三维结构（2B3P）上的位置——用 PyMOL/ChimeraX 渲染，球棒模型高亮突变残基。N164W 靠近发色团（绿色），L7C/T9C 在 N 端帽区（蓝色），K238V 在 C 端（红色），S208V/K209W 在环区（黄色）。*

---

## 8. 关键执行日志

以下为 Agent 实际执行的核心日志片段（完整 JSONL 见仓库 `outputs/latest/agent_trace.jsonl`）：

```
[2026-07-01 08:30:15] prepare_or_load_processed_data → success
[2026-07-01 08:30:24] load_config_and_data → success (parent=sfGFP, length=238)
[2026-07-01 08:30:25] stage1_ranked → start
[2026-07-01 08:30:25] generate_single_mutants → success (5000 candidates)
[2026-07-01 08:30:26] filter_stage1 → success (4819 passed, 181 failed)
[2026-07-01 08:30:26] stage1_brightness → start (4819 candidates)
[2026-07-01 08:32:10] stage1_brightness → success
[2026-07-01 08:32:10] stage1_stability → start
[2026-07-01 08:35:42] stage1_stability → success
[2026-07-01 08:35:42] stage1_ranked → success (200 seeds)
[2026-07-01 08:35:43] stage2_candidates → start
[2026-07-01 08:35:43] generate_combinatorial_mutants → success (3200 variants, orders 2-5)
[2026-07-01 08:36:20] inverse_folding_adapter → success (ProteinMPNN v_48_020, 385 variants)
[2026-07-01 08:36:21] filter_stage2 → success (4585 passed, 0 failed¹)
[2026-07-01 08:36:21] stage2_candidates → success
[2026-07-01 08:36:22] stage2_brightness → start
[2026-07-01 08:38:45] stage2_brightness → success
[2026-07-01 08:38:45] stage2_stability → start
[2026-07-01 08:42:30] stage2_stability → success
[2026-07-01 08:42:30] final_rank → success (4585 ranked)
[2026-07-01 08:42:31] select_top6 → success (4 small + 2 large)
[2026-07-01 08:42:31] export_outputs → success
[2026-07-01 08:42:31] status → success ✅
```

¹ ProteinMPNN 序列经过自动修复（M 缺失回填、X 替换为亲本残基），0 条被硬过滤。

**Agent 自动恢复事件**：
- ESMFold 不可用 → 自动跳过，ESMC 零样本权重从 0.45 升至 0.70，ThermoMPNN 权重从 0.25 升至 0.30
- 组合爆炸 (C(80,5)≈2400万) → 分阶采样 + 高阶种子限制 (50/40)
- ProteinMPNN 序列缺 M 含 X → 自动修复管线

**[图9]** *建议配图：Agent 决策树日志可视化。将上方日志转换为时序图（横轴时间、纵轴阶段），用绿色标注成功、橙色标注警告/降级、红色标注失败恢复。*

---

## 9. 可复现性说明

### 9.1 环境配置

```bash
git clone <repo-url>
cd spEBT
python -m pip install -e .[dev]
python -m spebt_agent.cli.setup_tool_envs --install-requirements
```

### 9.2 依赖模型权重

| 模型 | 位置 | 获取方式 |
|------|------|---------|
| ESMC-600M | `external/weights/esmc/ESMC-600M/` | HuggingFace `biohub/ESMC-600M` |
| ProteinMPNN | `external/repositories/ProteinMPNN/` | GitHub clone + `v_48_020.pt` weights |
| ThermoMPNN | `external/repositories/ThermoMPNN/` | GitHub clone |
| 亮度模型 | `artifacts/models/brightness/` | 项目内训练脚本自动生成 |

### 9.3 复现命令

```bash
# 一键复现完整设计
python -m spebt_agent.cli.run_agent \
  --team-name superEasternBlott \
  --task "先检查环境，再跑完整搜索设计" \
  --profile full_search

# 或直接运行底层管线
python -m spebt_agent.cli.run_design \
  --team-name superEasternBlott \
  --full-search
```

### 9.4 输出文件

| 文件 | 路径 | 说明 |
|------|------|------|
| 提交序列 | `outputs/latest/submission.csv` | 最终竞赛提交文件 |
| 入选 Top6 | `outputs/latest/selected_top6.csv` | 含全部预测得分 |
| 全量排名 | `outputs/latest/ranked_variants.csv` | 4585 条变体完整排名 |
| 最终报告 | `outputs/latest/final_report.md` | LLM 生成的中文设计报告 |
| 执行日志 | `outputs/latest/agent_trace.jsonl` | 逐事件审计轨迹 |
| 运行状态 | `outputs/latest/run_state.json` | 各阶段状态与耗时 |

---

## 附录：配图建议汇总

| 编号 | 标题 | 建议内容 | 推荐工具 |
|------|------|---------|---------|
| 图1 | 整体架构图 | 用户→Agent→工具注册→计算管线的数据流，三色分层 | draw.io / Excalidraw |
| 图2 | 漏斗筛选图 | 5000→200→4585→6 的逐级浓缩，标注筛选标准 | Matplotlib / Plotly |
| 图3 | 模型生态图 | ESMC-600M + 亮度预测器 + 稳定性评分器 + ProteinMPNN 关系 | draw.io / Figma |
| 图4 | 亮度模型散点图 | 预测 vs 真实值，标注 R²=0.66 | Matplotlib seaborn |
| 图5 | Agent 逻辑树 | 自然语言输入→Planner→Executor→Summarizer 决策树 | Mermaid / draw.io |
| 图6 | 双路径示意图 | 精准突变(左) vs 结构逆折叠(右) 并行策略 | draw.io / PowerPoint |
| 图7 | 雷达图 | Top6 在 5 维度的表现对比 | Matplotlib radar chart |
| 图8 | 突变位点标注 | sfGFP 3D 结构(2B3P)上高亮 7 个关键突变位置 | PyMOL / ChimeraX |
| 图9 | 执行时序图 | X轴时间、Y轴阶段、颜色编码的成功/降级/恢复 | Matplotlib / Plotly |

**推荐配色方案**：
- 主色：`#2E86AB`（深蓝，代表计算/AI）
- 辅色：`#A23B72`（品红，代表蛋白/生物）
- 强调色：`#F18F01`（橙黄，代表输出/结果）
- 背景：白色或极浅灰 `#F8F9FA`

---

> **superEasternBlott —— 超级东方杂交**
>
> *用 AI 调度模型，用模型设计蛋白，用蛋白征服赛场。*
