from __future__ import annotations

import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from spebt_agent.assets.registry import ModelAsset, list_model_assets
from spebt_agent.paths import project_root


def is_asset_present(asset: ModelAsset) -> bool:
    target = project_root() / asset.target
    return target.exists()


def prompt_for_download(asset: ModelAsset) -> bool:
    print("")
    print(f"Asset: {asset.asset_id}")
    print(f"Tool module: {asset.tool_module}")
    print(f"Description: {asset.description}")
    print(f"Target: {project_root() / asset.target}")
    print(f"Download size: {asset.size_human}")
    print(f"Size source: {asset.size_source}")
    answer = input("Download this asset now? Type 'yes' to download, anything else to skip: ").strip().lower()
    return answer == "yes"


def download_huggingface_snapshot(asset: ModelAsset) -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("Install huggingface_hub in the active environment before downloading HuggingFace assets.") from exc
    target = project_root() / asset.target
    target.parent.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=asset.spec["repo_id"],
        revision=asset.spec.get("revision", "main"),
        local_dir=target,
        local_dir_use_symlinks=False,
        allow_patterns=asset.spec.get("allow_patterns"),
    )


def download_git_repository(asset: ModelAsset) -> None:
    target = project_root() / asset.target
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return
    subprocess.run(
        [
            "git",
            "clone",
            "--progress",
            "--depth",
            "1",
            "--branch",
            asset.spec.get("revision", "main"),
            asset.spec["repo_url"],
            str(target),
        ],
        check=True,
    )


def download_url_file(asset: ModelAsset) -> None:
    target = project_root() / asset.target
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".part")
    started = time.monotonic()
    with urllib.request.urlopen(asset.spec["url"]) as response, tmp.open("wb") as f:
        total = int(response.headers.get("Content-Length") or asset.size_bytes or 0)
        copied = 0
        last_print = 0.0
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
            copied += len(chunk)
            now = time.monotonic()
            if now - last_print >= 1.0:
                elapsed = max(now - started, 0.001)
                rate = copied / elapsed / (1024 * 1024)
                if total:
                    pct = copied / total * 100
                    print(f"{asset.asset_id}: {pct:5.1f}% {copied / (1024**2):.1f}/{total / (1024**2):.1f} MB {rate:.2f} MB/s")
                else:
                    print(f"{asset.asset_id}: {copied / (1024**2):.1f} MB {rate:.2f} MB/s")
                sys.stdout.flush()
                last_print = now
    tmp.replace(target)


def materialize_repository_file(asset: ModelAsset) -> None:
    source = project_root() / asset.spec["source_path"]
    target = project_root() / asset.target
    if not source.exists():
        raise FileNotFoundError(f"Repository-provided asset is missing: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def download_asset(asset: ModelAsset) -> None:
    if asset.kind == "huggingface_snapshot":
        download_huggingface_snapshot(asset)
    elif asset.kind == "git_repository":
        download_git_repository(asset)
    elif asset.kind == "url_file":
        download_url_file(asset)
    elif asset.kind == "repository_file":
        materialize_repository_file(asset)
    else:
        raise ValueError(f"Unsupported asset kind: {asset.kind}")


def download_assets_interactively(tool_module: str | None = None, plan_only: bool = False) -> list[tuple[str, str]]:
    results = []
    for asset in list_model_assets(tool_module=tool_module):
        if is_asset_present(asset):
            print(f"[present] {asset.asset_id}: {project_root() / asset.target}")
            results.append((asset.asset_id, "present"))
            continue
        if plan_only:
            print(f"[planned] {asset.asset_id} {asset.size_human} -> {project_root() / asset.target}")
            results.append((asset.asset_id, "planned"))
            continue
        if not prompt_for_download(asset):
            results.append((asset.asset_id, "skipped"))
            continue
        download_asset(asset)
        results.append((asset.asset_id, "downloaded"))
    return results
