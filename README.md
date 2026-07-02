# spEBT 交互式蛋白设计 Agent

spEBT 是一个**用自然语言驱动的 GFP 蛋白设计 Agent**。你只需要用中文告诉它要做什么，它会自动理解任务、制定计划、调用工具、读取结果、并在失败时自动恢复。

底层所有候选生成、过滤、亮度/稳定性评分、排名、导出等计算由工具代码完成，LLM **只负责规划、调度和解释**，不做数值预测。

---

## 三步快速开始

```powershell
# 1. 安装项目
python -m pip install -e .[dev]

# 2. 配置 API（可选，不配置也能用）
# 复制 .env.example 为 .env，填写你的 API 密钥

# 3. 用自然语言下任务
python -m spebt_agent.cli.run_agent --team-name MyTeam --task "先检查环境，再跑一轮稳定模式 GFP 设计"
```

---

## .env 配置

复制 `.env.example` 为 `.env`，按需填写：

```ini
OPENAI_API_KEY=sk-your-key
OPENAI_BASE_URL=https://api.deepseek.com
OPENAI_MODEL=deepseek-v4-pro
```

不配置也能用——Agent 会使用内置的任务模板和确定性策略，所有数值计算不依赖 LLM。

---

## 数据文件

以下文件需要放在 `data/raw/` 目录下：

- `AAseqs of 5 GFP proteins_20260511.txt` — 亲本参考序列
- `GFP_data.xlsx` — 亮度训练数据
- `Exclusion_List.csv` — 排除序列列表
- `submission_template.csv` — 提交模板
- `structures/2B3P.pdb` — sfGFP 亲本结构

---

## 三种最常用命令

### 1. 一键设计（推荐）

```powershell
python -m spebt_agent.cli.run_agent --team-name YourTeamName --task "先检查环境，再跑一轮稳定模式 GFP 设计"
```

Agent 会自动：
1. 检查所有工具模块是否就绪
2. 运行稳定模式设计流水线
3. 生成 `submission.csv` 和 `final_report.md`
4. 失败时自动尝试恢复

### 2. 查看最近结果

```powershell
python -m spebt_agent.cli.run_agent --team-name YourTeamName --task "读取最新结果并解释为什么选这6条"
```

Agent 会读取 `outputs/latest/` 下的所有文件，用中文解释排名结果和入选理由。

### 3. 继续失败的运行

```powershell
python -m spebt_agent.cli.run_agent --team-name YourTeamName --task "继续上一次失败的运行"
```

Agent 会读取 `run_state.json`，定位失败阶段，自动 resume。

---

## 更多任务示例

```powershell
# 从文件读取任务
python -m spebt_agent.cli.run_agent --team-name YourTeamName --task-file task.txt

# 完整搜索模式
python -m spebt_agent.cli.run_agent --team-name YourTeamName --task "跑一轮完整搜索设计" --profile full_search

# 只导出提交文件
python -m spebt_agent.cli.run_agent --team-name YourTeamName --task "从现有 top6 导出 submission.csv"

# 先训练亮度模型再设计
python -m spebt_agent.cli.run_agent --team-name YourTeamName --task "先训练亮度模型，再跑稳定设计"

# 仅检查环境
python -m spebt_agent.cli.run_agent --team-name YourTeamName --task "检查所有工具环境是否就绪"

# JSON 格式输出（供脚本调用）
python -m spebt_agent.cli.run_agent --team-name YourTeamName --task "检查环境" --json
```

---

## Agent 如何工作

```
你说 "先检查环境再跑稳定设计"
        │
        ▼
┌─────────────────────────────┐
│ 1. Planner（分析任务）      │
│    → 识别为 health_and_design │
│    → 生成执行计划            │
└──────────┬──────────────────┘
           ▼
┌─────────────────────────────┐
│ 2. Executor（逐步执行）     │
│   Step 1: health_check ✓    │
│   Step 2: run_design ✓      │
│   失败时自动恢复...          │
└──────────┬──────────────────┘
           ▼
┌─────────────────────────────┐
│ 3. Summarizer（生成报告）   │
│   用中文告诉你：             │
│   - 做了什么                │
│   - 结果在哪                │
│   - 下一步建议              │
└─────────────────────────────┘
```

---

## 输出文件怎么看

每次运行输出到 `outputs/runs/<run_id>/`，最新成功运行同步到 `outputs/latest/`。

| 文件 | 说明 |
|------|------|
| `run_state.json` | 运行状态：各阶段完成情况、耗时、失败原因 |
| `agent_trace.jsonl` | 事件级日志：每一步的开始/进度/成功/失败 |
| `stage1_ranked.csv` | 阶段1单突变排名 |
| `stage2_candidates.csv` | 阶段2组合候选池 |
| `stage2_brightness.csv` | 阶段2亮度分数 |
| `stage2_stability.csv` | 阶段2稳定性分数 |
| `ranked_variants.csv` | 最终综合排名 |
| `selected_top6.csv` | 多样性筛选后的前6条 |
| `submission.csv` | 竞赛提交文件（Team_Name, Seq_ID, Sequence） |
| `final_report.md` | LLM 生成的最终报告 |

Agent 会话状态保存在 `outputs/agent_sessions/<session_id>/agent_state.json`。

---

## 失败怎么恢复

Agent 内置了硬编码的失败恢复策略，LLM 不能绕过：

| 失败场景 | 自动恢复策略 |
|---------|------------|
| `prepare_data` 失败 | 终止（检查原始数据文件） |
| `health_check` 关键模块失败 | 终止（检查工具环境安装） |
| 卡在 `stage2_stability` | resume → 降 esmfold_top_k → 降 max_stage2_candidates |
| 卡在 `stage2_brightness` | resume → 降 max_stage2_candidates |
| ESMFold 不可用 | 继续（用 ESMC+ThermoMPNN 评分） |
| ProteinMPNN 不可用 | 继续（仅 stage1+combinatorial） |
| `submission.csv` 已生成 | 视为成功 |

你也可以手动 resume：

```powershell
python -m spebt_agent.cli.run_design --team-name YourTeamName --run-id <run_id> --resume
```

---

## stable vs full_search

`stable`（默认）：快速迭代模式
- 较小的阶段2种子数
- 较小的组合搜索空间
- 较小的 ESMFold top_k

`full_search`：完整搜索模式
- 恢复完整的搜索参数
- 适合离线长时间探索

在 Agent 任务中通过 `--profile` 切换，或直接在任务描述中说"完整搜索"。

---

## 进阶：直接使用底层 CLI

Agent 是对底层工具的高级封装。高级用户仍然可以单独调用每个工具：

```powershell
# 数据准备
python -m spebt_agent.cli.prepare_data

# 环境健康检查
python -m spebt_agent.cli.health_check
python -m spebt_agent.cli.health_check --module stability --strict

# 训练亮度模型
python -m spebt_agent.cli.train_brightness --target brightness

# 直接运行设计流水线
python -m spebt_agent.cli.run_design --team-name YourTeamName
python -m spebt_agent.cli.run_design --team-name YourTeamName --full-search
python -m spebt_agent.cli.run_design --team-name YourTeamName --run-id <id> --resume
python -m spebt_agent.cli.run_design --team-name YourTeamName --max-stage2-candidates 1200 --esmfold-top-k 12

# 导出提交文件
python -m spebt_agent.cli.export_submission --selected outputs/latest/selected_top6.csv --team-name YourTeamName

# 运行测试
python -m pytest
```

---

## 目录结构

```
spEBT/
├── configs/              # YAML 配置（竞赛、生成、模型路径、评分权重）
├── data/
│   ├── raw/              # 原始竞赛文件
│   └── processed/        # 预处理后的数据
├── external/
│   ├── repositories/     # 第三方仓库（ProteinMPNN, ThermoMPNN, RaSP）
│   └── weights/          # 模型权重（ESMC-600M, ESMFold）
├── artifacts/
│   ├── embeddings/       # 嵌入缓存
│   ├── models/           # 训练好的本地模型
│   └── reports/          # 训练指标报告
├── outputs/
│   ├── runs/<run_id>/    # 每次运行的隔离输出
│   ├── latest/           # 最新成功运行的快照
│   └── agent_sessions/   # Agent 会话状态
├── src/spebt_agent/      # 源代码
│   ├── agent.py          # 主编排入口
│   ├── brain/            # LLM 集成（计划、执行、提示词）
│   ├── tools/            # 计算工具（生成、评分、排名、导出）
│   ├── cli/              # 命令行入口
│   └── ...
└── tests/                # 测试
```

---

## 环境设置

```powershell
# 安装项目
python -m pip install -e .[dev]

# 创建工具环境
python -m spebt_agent.cli.setup_tool_envs

# 创建环境并安装依赖
python -m spebt_agent.cli.setup_tool_envs --install-requirements
```

---

## 核心设计原则

- **LLM 与数值分离**：LLM 只做计划、调度、解释，不做数值预测
- **工具标准化**：所有工具统一输入输出格式，LLM 通过注册表发现和调用
- **确定性恢复**：失败恢复规则硬编码，不依赖 LLM "自己猜"
- **中文友好**：默认提示词和输出优先中文
- **向后兼容**：底层 `run_design` 继续可用，不受 Agent 层影响

## spuerEasternBlott————致敬传奇生物实验方法Southern Blot
