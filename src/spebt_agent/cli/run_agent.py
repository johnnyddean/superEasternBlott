"""CLI entry point for the interactive agent.

Usage:
    python -m spebt_agent.cli.run_agent --team-name MyTeam --task "先检查环境，再跑一轮稳定模式 GFP 设计"
    python -m spebt_agent.cli.run_agent --team-name MyTeam --task "读取最新结果并解释为什么选这6条"
    python -m spebt_agent.cli.run_agent --team-name MyTeam --task "继续上一次失败的运行"
    python -m spebt_agent.cli.run_agent --team-name MyTeam --task-file task.txt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m spebt_agent.cli.run_agent",
        description="spEBT 交互式蛋白设计 Agent — 用自然语言驱动 GFP 蛋白设计任务",
    )
    parser.add_argument("--team-name", default=None, help="队伍名称")
    parser.add_argument("--task", default=None, help="自然语言任务描述")
    parser.add_argument("--task-file", default=None, help="从文件读取任务（与 --task 二选一）")
    parser.add_argument("--profile", default="stable", choices=["stable", "full_search"], help="设计配置 (默认 stable)")
    parser.add_argument("--session-id", default=None, help="恢复之前的会话")
    parser.add_argument("--max-stage2-candidates", type=int, default=None, help="阶段2最大候选数")
    parser.add_argument("--esmfold-top-k", type=int, default=None, help="ESMFold 评分 top-K")
    parser.add_argument("--json", action="store_true", dest="output_json", help="以 JSON 格式输出")
    args = parser.parse_args()

    # ── resolve task ────────────────────────────────────────────────────
    task = args.task
    if not task and args.task_file:
        task_path = Path(args.task_file)
        if not task_path.exists():
            print(f"错误：任务文件不存在: {args.task_file}", file=sys.stderr)
            raise SystemExit(1)
        task = task_path.read_text(encoding="utf-8").strip()
    if not task:
        print("错误：请通过 --task 或 --task-file 提供任务描述", file=sys.stderr)
        print("示例：python -m spebt_agent.cli.run_agent --team-name MyTeam --task \"先检查环境，再跑一轮稳定模式 GFP 设计\"", file=sys.stderr)
        raise SystemExit(1)

    # ── run agent ───────────────────────────────────────────────────────
    from spebt_agent.agent import run_interactive_agent

    print(f"\n{'='*60}")
    print(f"  spEBT 交互式 Agent")
    print(f"  任务: {task[:100]}{'...' if len(task) > 100 else ''}")
    print(f"  队伍: {args.team_name or '(从配置读取)'}")
    print(f"  配置: {args.profile}")
    print(f"{'='*60}\n")
    print("正在分析任务并生成执行计划...\n")

    result = run_interactive_agent(
        task=task,
        team_name=args.team_name,
        profile=args.profile,
        session_id=args.session_id,
        max_stage2_candidates=args.max_stage2_candidates,
        esmfold_top_k=args.esmfold_top_k,
    )

    # ── output ──────────────────────────────────────────────────────────
    if args.output_json:
        # Use ascii-safe output to avoid Windows GBK encoding issues
        print(json.dumps(result, ensure_ascii=True, indent=2))
    else:
        # Human-readable output (avoid emoji on Windows GBK terminals)
        status_label = "[OK] 成功" if result.get("ok") else "[FAIL] 失败"
        print(f"\n{'='*60}")
        print(f"  {status_label}")
        print(f"{'='*60}\n")

        if result.get("final_summary"):
            print(result["final_summary"])
        else:
            print(f"任务目标: {result.get('goal', 'N/A')}")
            print(f"任务类型: {result.get('task_type', 'N/A')}")
            print(f"执行步骤: {', '.join(result.get('steps_executed', []))}")
            print(f"会话 ID: {result.get('session_id', 'N/A')}")
            if result.get("run_ids"):
                print(f"关联运行: {', '.join(result['run_ids'])}")
            if result.get("warnings"):
                print(f"\n[WARNING] 警告:")
                for w in result["warnings"]:
                    print(f"  - {w}")
            if result.get("failure_reason"):
                print(f"\n失败原因: {result['failure_reason']}")

        print(f"\n{'='*60}")

    if not result.get("ok"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
