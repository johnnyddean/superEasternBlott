from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Protocol

from spebt_agent.env import load_dotenv_file


class LLMClient(Protocol):
    def summarize_strategy(self, context: dict) -> str:
        ...

    def write_report(self, context: dict) -> str:
        ...


@dataclass
class NullLLMClient:
    reason: str = "OPENAI_API_KEY is not configured."

    def summarize_strategy(self, context: dict) -> str:
        parent = context.get("parent_name", "sfGFP")
        return (
            f"Deterministic baseline strategy: generate constrained mutants from {parent}, "
            "filter hard competition constraints and exclusion sequences, score brightness with "
            "available model or baseline proxy, score 72C retention with a rule-based proxy, "
            "then select a diverse top 6."
        )

    def write_report(self, context: dict) -> str:
        selected = context.get("selected_top6", [])
        lines = [
            "# spEBT Agent Report",
            "",
            f"LLM mode: NullLLMClient ({self.reason})",
            "",
            "Thermal retention is a proxy score in this scaffold; no direct F_final/F_initial model is claimed.",
            "",
            f"Selected sequences: {len(selected)}",
        ]
        for item in selected:
            lines.append(
                f"- {item.get('variant_id')}: score={item.get('final_score', 0):.4f}, "
                f"brightness={item.get('predicted_relative_brightness', 0):.4f}, "
                f"retention72={item.get('predicted_retention72', 0):.4f}"
            )
        return "\n".join(lines) + "\n"


@dataclass
class OpenAICompatibleLLMClient:
    api_key: str
    base_url: str
    model: str

    def _complete(self, prompt: str) -> str:
        try:
            from openai import OpenAI
        except ImportError:
            return NullLLMClient("openai package is not installed.").write_report({"selected_top6": []})
        client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are the strategy/report layer for a GFP design agent. Do not make numerical predictions."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        return response.choices[0].message.content or ""

    def summarize_strategy(self, context: dict) -> str:
        return self._complete(f"Summarize the design strategy for this run:\n{context}")

    def write_report(self, context: dict) -> str:
        return self._complete(f"Write a concise reproducible report for this GFP design run:\n{context}")


def build_llm_client(llm_cfg: dict) -> LLMClient:
    load_dotenv_file()
    api_key = os.getenv(llm_cfg.get("api_key_env", "OPENAI_API_KEY"), "")
    if not api_key:
        return NullLLMClient()
    base_url = os.getenv(llm_cfg.get("base_url_env", "OPENAI_BASE_URL"), llm_cfg.get("default_base_url", "https://api.openai.com/v1"))
    model = os.getenv(llm_cfg.get("model_env", "OPENAI_MODEL"), llm_cfg.get("default_model", "gpt-4.1-mini"))
    return OpenAICompatibleLLMClient(api_key=api_key, base_url=base_url, model=model)
