# spEBT 交互式 Agent 宝宝级教程

> 看完这篇，你就能用中文指挥 AI 帮你做蛋白设计了。

---

## 一、这是什么？

spEBT Agent 是一个**你用中文对话就能驱动的蛋白设计工具**。

以前你需要这样：

```powershell
python -m spebt_agent.cli.prepare_data
python -m spebt_agent.cli.health_check
python -m spebt_agent.cli.train_brightness --target brightness
python -m spebt_agent.cli.run_design --team-name MyTeam --profile stable
python -m spebt_agent.cli.export_submission --selected outputs/latest/selected_top6.csv --team-name MyTeam
```

每一步都要记住命令和参数。

现在你只需要这样：

```powershell
python -m spebt_agent.cli.run_agent --team-name MyTeam --task "先检查环境，再跑一轮稳定模式设计"
```

**一句话搞定。** Agent 会自动理解你要做什么、按顺序执行、失败了自己恢复、最后用中文告诉你结果。

---

## 二、安装（只需一次）

### 第一步：打开终端

按 `Win+R`，输入 `powershell`，回车。

### 第二步：进入项目目录

```powershell
cd "F:\competitions\合成生物学创新赛2026\蛋白设计\spEBT\spEBT"
```

> 💡 **提示**：如果你把项目放在别的位置，把路径换成你的实际路径。

### 第三步：安装

```powershell
python -m pip install -e .[dev]
```

看到 `Successfully installed` 就说明装好了。

### 第四步：验证安装

```powershell
python -m spebt_agent.cli.run_agent --help
```

如果打印出一堆帮助信息，说明安装成功。

---

## 三、配置 API（可选）

Agent 内置了任务模板，**不配置 API 也能用**。但如果你想要更智能的规划和中文解释，可以配置 LLM。

### 第一步：复制配置文件

在项目目录下，复制 `.env.example` 为 `.env`：

```powershell
copy .env.example .env
```

### 第二步：编辑 `.env`

用记事本打开 `.env`，填入你的 API 信息：

```ini
OPENAI_API_KEY=sk-你的密钥
OPENAI_BASE_URL=https://api.deepseek.com
OPENAI_MODEL=deepseek-v4-pro
```

> 💡 **不用 DeepSeek？** 改成你用的任何 OpenAI 兼容接口即可。

保存，搞定。

---

## 四、你的第一条命令

试试最简单的任务：

```powershell
cd "F:\competitions\合成生物学创新赛2026\蛋白设计\spEBT\spEBT"
$env:PYTHONPATH = "src"
python -m spebt_agent.cli.run_agent --team-name MyTeam --task "检查环境健康状态"
```

你会看到类似这样的输出：

```
============================================================
  spEBT 交互式 Agent
  任务: 检查环境健康状态
  队伍: MyTeam
  配置: stable
============================================================

正在分析任务并生成执行计划...

[OK] 成功
============================================================

# spEBT 执行报告

**会话 ID**: abc123-def456-...
**状态**: success
**任务**: 检查环境健康状态

## 执行步骤
- [success] health_check: 所有模块通过
...
```

---

## 五、常用任务速查表

把 `<YourTeamName>` 换成你的队伍名，直接复制粘贴就能用。

### 🟢 任务 1：一键设计（最常用）

```powershell
python -m spebt_agent.cli.run_agent --team-name <YourTeamName> --task "先检查环境，再跑一轮稳定模式 GFP 设计"
```

**Agent 会自动做**：
1. 检查工具环境（ESMC、ESMFold、ProteinMPNN 等是否就绪）
2. 运行完整设计流水线（生成突变→过滤→评分→排名→选前6→导出）
3. 失败时自动尝试恢复
4. 输出最终报告

**预计耗时**：几分钟到几十分钟（取决于候选数量）

**输出文件在哪里**：`outputs/runs/<run_id>/` 和 `outputs/latest/`

---

### 🟢 任务 2：查看最近结果

```powershell
python -m spebt_agent.cli.run_agent --team-name <YourTeamName> --task "读取最新结果并解释为什么选这6条"
```

**Agent 会**：
- 列出 `outputs/latest/` 下所有文件
- 展示入选的 6 条序列及其得分
- 展示排名前 10 的变体
- 用中文解释入选理由

---

### 🟢 任务 3：继续之前失败的运行

```powershell
python -m spebt_agent.cli.run_agent --team-name <YourTeamName> --task "继续上一次失败的运行"
```

**Agent 会**：
- 自动读取 `run_state.json` 找到失败点
- 从检查点恢复（已完成的阶段不会重跑）
- 如果恢复失败，自动降级参数重试

---

### 🟢 任务 4：完整搜索模式

```powershell
python -m spebt_agent.cli.run_agent --team-name <YourTeamName> --task "跑一轮完整搜索设计" --profile full_search
```

比 `stable` 搜索更广，耗时长但覆盖更全。适合最终提交前使用。

---

### 🟢 任务 5：只导出提交文件

```powershell
python -m spebt_agent.cli.run_agent --team-name <YourTeamName> --task "从现有 top6 导出 submission.csv"
```

当你已经有了 `selected_top6.csv`，只想重新导出时使用。

---

### 🟢 任务 6：先训练模型再设计

```powershell
python -m spebt_agent.cli.run_agent --team-name <YourTeamName> --task "先训练亮度模型，再跑稳定设计"
```

当你更新了训练数据后，需要重新训练亮度预测模型。

---

## 六、输出文件详解

每次运行后，去 `outputs/latest/` 目录看结果：

```
outputs/latest/
├── run_state.json          ← 运行状态总览
├── agent_trace.jsonl       ← 事件日志（调试用）
├── stage1_ranked.csv       ← 第一轮排名（单突变）
├── stage2_candidates.csv   ← 第二轮候选池（组合突变）
├── stage2_brightness.csv   ← 亮度预测分数
├── stage2_stability.csv    ← 稳定性预测分数
├── ranked_variants.csv     ← 最终综合排名
├── selected_top6.csv       ← 🌟 入选的 6 条序列（这是你要的！）
├── submission.csv          ← 🌟 竞赛提交文件（这也是你要的！）
└── final_report.md         ← 完整设计报告
```

### 最重要的两个文件

**`selected_top6.csv`** — 入选的 6 条序列：

| variant_id | sequence | final_score | predicted_brightness | predicted_retention72 |
|------------|----------|-------------|---------------------|----------------------|
| variant_001 | MSKGEEL... | 0.854 | 0.92 | 0.78 |
| ... | ... | ... | ... | ... |

**`submission.csv`** — 可以直接提交的竞赛文件：

| Team_Name | Seq_ID | Sequence |
|-----------|--------|----------|
| MyTeam | 1 | MSKGEEL... |
| MyTeam | 2 | MSKGEEL... |
| ... | ... | ... |

---

## 七、失败了怎么办？

Agent 失败时会告诉你原因和下一步建议。常见情况：

### 情况 1：工具环境没装好

```
health_check 失败: 关键工具模块不可用
```

**解决**：

```powershell
python -m spebt_agent.cli.setup_tool_envs --install-requirements
```

### 情况 2：数据没准备

```
prepare_data 失败
```

**解决**：检查 `data/raw/` 目录下是否有这些文件：
- `AAseqs of 5 GFP proteins_20260511.txt`
- `GFP_data.xlsx`
- `Exclusion_List.csv`

### 情况 3：设计跑到一半失败了

```
run_design 卡在 stage2_stability
```

**解决**：Agent 会自动尝试恢复。如果自动恢复也失败，手动尝试降级：

```powershell
python -m spebt_agent.cli.run_design --team-name <YourTeamName> --esmfold-top-k 4 --max-stage2-candidates 800
```

### 情况 4：ESMFold / ProteinMPNN 不可用

Agent 会自动跳过这些可选工具，用剩余的信号继续。只要 `submission.csv` 生成了就算成功。

---

## 八、从文件读取任务

如果你的任务描述很长，可以写到一个文本文件里：

```powershell
# 创建一个任务文件
echo "先检查环境健康，确认所有工具模块就绪后，用完整搜索模式跑一轮设计，如果失败就降级参数重试" > mytask.txt

# 让 Agent 读取
python -m spebt_agent.cli.run_agent --team-name MyTeam --task-file mytask.txt
```

---

## 九、输出 JSON 格式（给脚本用）

```powershell
python -m spebt_agent.cli.run_agent --team-name MyTeam --task "检查环境" --json
```

返回标准 JSON，方便你在脚本中解析：

```json
{
  "ok": true,
  "session_id": "abc-123",
  "goal": "检查所有工具模块环境",
  "steps_executed": ["health_check"],
  "final_summary": "...",
  "artifacts": {...}
}
```

---

## 十、完整工作流程示例

假设你第一次使用，从头到尾：

```powershell
# 1. 进入项目目录
cd "F:\competitions\合成生物学创新赛2026\蛋白设计\spEBT\spEBT"

# 2. 设置 Python 路径（每次新终端都要做）
$env:PYTHONPATH = "src"

# 3. 安装（只需一次）
python -m pip install -e .[dev]

# 4. 准备好数据文件放在 data/raw/ 下

# 5. 第一次运行：先检查环境
python -m spebt_agent.cli.run_agent --team-name MyTeam --task "检查环境健康状态"

# 6. 如果环境 OK，跑设计
python -m spebt_agent.cli.run_agent --team-name MyTeam --task "先检查环境，再跑稳定模式设计"

# 7. 查看结果
python -m spebt_agent.cli.run_agent --team-name MyTeam --task "读取最新结果并解释"

# 8. 如果满意，导出提交文件
python -m spebt_agent.cli.run_agent --team-name MyTeam --task "导出 submission.csv"

# 9. 提交文件在 outputs/latest/submission.csv
```

---

## 十一、遇到问题？

### Q: 提示 "No module named 'spebt_agent'"

设置一下 `PYTHONPATH`：

```powershell
$env:PYTHONPATH = "src"
```

### Q: 输出乱码

Windows 终端编码问题。用 `--json` 模式避免：

```powershell
python -m spebt_agent.cli.run_agent --team-name MyTeam --task "检查环境" --json
```

### Q: 执行很慢

正常。ESMC 嵌入计算、ESMFold 结构预测都需要时间。可以用 `stable` 模式（默认）而不是 `full_search`。

### Q: 如何看详细日志

```powershell
# 查看运行状态
cat outputs/latest/run_state.json

# 查看事件日志
cat outputs/latest/agent_trace.jsonl
```

---

## 总结

| 我想做的事情 | 命令 |
|------------|------|
| 跑一轮设计 | `python -m spebt_agent.cli.run_agent --team-name <队名> --task "先检查环境再跑设计"` |
| 查看结果 | `python -m spebt_agent.cli.run_agent --team-name <队名> --task "读取最新结果"` |
| 继续失败的 | `python -m spebt_agent.cli.run_agent --team-name <队名> --task "继续上一次失败的运行"` |
| 完整搜索 | 加 `--profile full_search` |
| 只要导出 | `--task "导出提交文件"` |
| 训练+设计 | `--task "先训练亮度模型再设计"` |

**记住一句话就够了**：你需要做的事情，用中文说出来，Agent 帮你做。
