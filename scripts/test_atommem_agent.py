#!/usr/bin/env python3
"""测试 atommem agent 是否能正常调用工具。"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import openai
import yaml

from codemem.add.atommem_agent_v2 import run_dir_parallel
from codemem.log import Logger
from codemem.search.toolcfg import load_catalog


async def main():
    log = Logger(level="debug", path=None)

    # 加载配置
    config_path = Path("configs/add.yaml")
    tool_config_path = Path("configs/add_tool.json")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    catalog = load_catalog(tool_config_path)

    # 创建 OpenAI client
    client = openai.AsyncOpenAI(
        api_key=config.get("api_key", "dummy"),
        base_url=config.get("base_url"),
    )

    # 只处理前 2 条 message
    result = await run_dir_parallel(
        directory=Path("data/Caroline_Melanie"),
        config=config,  # 传入完整配置
        client=client,
        catalog=catalog,
        log=log,
        limit_messages=2,
    )

    print(f"\n=== 处理结果 ===")
    print(f"状态: {result.get('status')}")
    print(f"产出 atom 数: {result.get('atom_count', 0)}")

    print("\n=== 检查产出 ===")
    atommem_file = Path("data/Caroline_Melanie/atommem.jsonl")
    if atommem_file.exists():
        lines = atommem_file.read_text().strip().split("\n")
        print(f"✓ atommem.jsonl 有 {len([l for l in lines if l.strip()])} 行")
        if lines and lines[0].strip():
            print(f"首行: {lines[0][:200]}...")
    else:
        print("✗ atommem.jsonl 不存在")


if __name__ == "__main__":
    asyncio.run(main())
