#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""阶段 1：预处理本地 jsonl 数据集。

- 扫描 data_dir 下所有 jsonl；
- 兼容 messages 格式，取最后一条 user 为问题、最后一条 assistant 为参考答案；
- 去除 assistant 内容中的思维链（<think>...</think> 等）；
- 按问题去重（可配置按文件或全局）；
- 输出精简数据集到 prepared_dir，并写 _manifest.json 统计。

用法：python qa_prepare.py --config config.yaml [--fresh]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import qa_common as C


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="阶段 1：预处理 jsonl 数据集（去重 + 去思维链）")
    parser.add_argument("--config", default="config.yaml", help="YAML 配置文件路径")
    parser.add_argument("--fresh", action="store_true", help="清空 prepared_dir 后重新生成")
    return parser.parse_args()


def reset_dir(path: Path) -> None:
    if path.exists():
        if path.parent == path or str(path) == path.anchor:
            raise SystemExit(f"错误：拒绝清空危险路径：{path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def discover_files(cfg: dict) -> list[Path]:
    data_dir = Path(cfg["paths"]["data_dir"]).resolve()
    if not data_dir.is_dir():
        raise SystemExit(f"错误：数据集目录不存在：{data_dir}")

    dataset = cfg["dataset"]
    pattern = str(dataset.get("pattern") or "*.jsonl")
    recursive = bool(dataset.get("recursive", True))

    excluded: list[Path] = [
        Path(cfg["paths"]["prepared_dir"]).resolve(),
        Path(cfg["paths"]["output_dir"]).resolve(),
    ]
    for name in dataset.get("exclude_dirs") or []:
        candidate = Path(name)
        if not candidate.is_absolute():
            candidate = data_dir / candidate
        excluded.append(candidate.resolve())

    candidates = data_dir.rglob(pattern) if recursive else data_dir.glob(pattern)
    files: list[Path] = []
    for path in sorted(candidates):
        if not path.is_file():
            continue
        resolved = path.resolve()
        if any(resolved == e or resolved.is_relative_to(e) for e in excluded):
            continue
        if resolved.name.endswith(C.OUTPUT_SUFFIXES):
            continue
        if any(part.startswith(".") for part in resolved.relative_to(data_dir).parts):
            continue
        files.append(resolved)
    return files


def parse_record(record: dict, multi_turn: bool, do_strip: bool) -> tuple[dict, bool]:
    """解析一条记录，返回 (标准化字段, 是否清洗过思维链)。"""
    messages = record.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("字段 messages 缺失或为空")

    normalized: list[dict] = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ValueError(f"messages[{index}] 不是对象")
        role = message.get("role")
        if role not in ("system", "user", "assistant"):
            raise ValueError(f"messages[{index}] 的 role 非法：{role!r}")
        content = C.content_to_text(message.get("content")).strip()
        normalized.append({"role": role, "content": content})

    last_assistant = next((i for i in range(len(normalized) - 1, -1, -1) if normalized[i]["role"] == "assistant"), None)
    if normalized[-1]["role"] == "user":
        # 数据以用户提问结尾：视为“无参考答案”的待答题，问题就是最后一条 user
        reference_raw = ""
        context = normalized
        last_user = normalized[-1]
    else:
        if last_assistant is None:
            reference_raw = ""
            context = normalized
        else:
            reference_raw = normalized[last_assistant]["content"]
            context = normalized[:last_assistant]
        last_user = next((m for m in reversed(context) if m["role"] == "user"), None)

    if last_user is None or not last_user["content"].strip():
        raise ValueError("未找到有效的用户提问（user 消息）")

    question = last_user["content"].strip()
    cot_removed = False
    reference = reference_raw
    if do_strip and C.has_cot(reference_raw):
        reference = C.strip_cot(reference_raw)
        cot_removed = True

    if multi_turn:
        prompt_messages: list[dict] = []
        for message in context:
            content = message["content"]
            if do_strip and message["role"] == "assistant":
                content = C.strip_cot(content)
            if content.strip():
                prompt_messages.append({"role": message["role"], "content": content})
        if not prompt_messages:
            prompt_messages = [{"role": "user", "content": question}]
    else:
        last_system = next((m for m in reversed(context) if m["role"] == "system"), None)
        prompt_messages = []
        if last_system and last_system["content"].strip():
            prompt_messages.append({"role": "system", "content": last_system["content"]})
        prompt_messages.append({"role": "user", "content": question})

    return {
        "question": question,
        "reference_answer": reference.strip(),
        "prompt_messages": prompt_messages,
        "cot_removed": cot_removed,
    }, cot_removed


def safe_domain(path: Path, data_dir: Path, used: set[str]) -> str:
    relative = path.relative_to(data_dir).with_suffix("")
    domain = "__".join(relative.parts)
    domain = C.dataset_stem(domain)
    if domain in used:
        domain = f"{domain}__{C.sha1_text(str(path))[:6]}"
    used.add(domain)
    return domain


def main() -> int:
    C.setup_stdout()
    args = parse_args()
    cfg = C.load_config(args.config)

    prepared_dir = Path(cfg["paths"]["prepared_dir"])
    data_dir = Path(cfg["paths"]["data_dir"])
    if args.fresh:
        reset_dir(prepared_dir)
    else:
        prepared_dir.mkdir(parents=True, exist_ok=True)

    files = discover_files(cfg)
    if not files:
        raise SystemExit(f"错误：{data_dir} 下未找到任何 {cfg['dataset'].get('pattern')} 数据集文件")

    dataset_cfg = cfg["dataset"]
    do_strip = bool(dataset_cfg.get("strip_cot", True))
    multi_turn = bool(dataset_cfg.get("multi_turn", False))
    dedupe = bool(dataset_cfg.get("dedupe", True))
    dedupe_scope = str(dataset_cfg.get("dedupe_scope") or "file").lower()
    dedupe_key = str(dataset_cfg.get("dedupe_key") or "question").lower()
    min_question = max(1, int(dataset_cfg.get("min_question_chars") or 1))
    limit = max(0, int(dataset_cfg.get("max_records_per_file") or 0))

    C.log(f"发现 {len(files)} 个数据集文件，开始预处理")
    C.log(f"  去重：{'开' if dedupe else '关'}（{dedupe_scope} / {dedupe_key}）")
    C.log(f"  思维链清洗：{'开' if do_strip else '关'}；多轮模式：{'开' if multi_turn else '关'}")

    global_seen: set[str] = set()
    used_domains: set[str] = set()
    file_stats: list[dict] = []
    total_kept = 0
    total_duplicates = 0
    total_bad = 0
    total_invalid = 0
    total_lines = 0
    started = time.time()

    for path in files:
        domain = safe_domain(path, data_dir, used_domains)
        relative = path.relative_to(data_dir).as_posix()
        seen_file: set[str] = set()
        kept: list[dict] = []
        duplicates = invalid = bad = lines = empty_ref = 0
        truncated = 0

        try:
            handle = open(path, "r", encoding="utf-8")
        except OSError as exc:
            C.log(f"  跳过无法读取的文件：{relative}（{exc}）")
            continue

        with handle:
            for line_no, raw_line in enumerate(handle, 1):
                raw_line = raw_line.strip()
                if not raw_line:
                    continue
                lines += 1
                try:
                    record = json.loads(raw_line)
                except json.JSONDecodeError:
                    bad += 1
                    continue
                if not isinstance(record, dict):
                    bad += 1
                    continue

                try:
                    parsed, _ = parse_record(record, multi_turn=multi_turn, do_strip=do_strip)
                except ValueError:
                    invalid += 1
                    continue

                question = parsed["question"]
                if len(question) < min_question:
                    invalid += 1
                    continue

                if dedupe:
                    if dedupe_key == "id" and str(record.get("id") or "").strip():
                        dedupe_value = C.normalize_for_key(record.get("id"))
                    else:
                        dedupe_value = C.normalize_for_key(question)
                    seen = global_seen if dedupe_scope == "global" else seen_file
                    if dedupe_value in seen:
                        duplicates += 1
                        continue
                    seen.add(dedupe_value)
                    if dedupe_scope == "global":
                        seen_file.add(dedupe_value)

                if limit and len(kept) >= limit:
                    truncated += 1
                    continue

                record_id = str(record.get("id") or "").strip()
                key_base = record_id or f"q_{C.sha1_text(question)[:12]}"
                key = f"{domain}#{key_base}"
                if any(item["key"] == key for item in kept):
                    key = f"{key}#{C.sha1_text(question)[:6]}"

                meta = {k: v for k, v in record.items() if k != "messages"}
                kept.append(
                    {
                        "key": key,
                        "id": record_id or key_base,
                        "domain": domain,
                        "question": question,
                        "reference_answer": parsed["reference_answer"],
                        "prompt_messages": parsed["prompt_messages"],
                        "multi_turn": multi_turn,
                        "cot_removed": parsed["cot_removed"],
                        "meta": meta,
                        "source": {"file": relative, "line": line_no},
                        "prepared_at": C.now_iso(),
                    }
                )
                if not parsed["reference_answer"]:
                    empty_ref += 1

        output_path = prepared_dir / f"{domain}.jsonl"
        tmp_path = output_path.with_name(output_path.name + ".tmp")
        with open(tmp_path, "w", encoding="utf-8", newline="\n") as out:
            for item in kept:
                out.write(json.dumps(item, ensure_ascii=False) + "\n")
            out.flush()
            os.fsync(out.fileno())
        os.replace(tmp_path, output_path)

        stats = {
            "file": relative,
            "domain": domain,
            "output": output_path.name,
            "lines": lines,
            "kept": len(kept),
            "duplicates": duplicates,
            "invalid": invalid,
            "bad_lines": bad,
            "empty_reference": empty_ref,
            "truncated_by_limit": truncated,
        }
        file_stats.append(stats)
        total_kept += len(kept)
        total_duplicates += duplicates
        total_bad += bad
        total_invalid += invalid
        total_lines += lines
        C.log(
            f"  {relative} -> {output_path.name} | 原始 {lines} 保留 {len(kept)}"
            f" | 去重 {duplicates} 无效 {invalid} 坏行 {bad}"
        )

    # 清理上一轮遗留、但本次已不存在的 prepared 文件，避免续跑时读到旧领域
    expected_outputs = {item["output"] for item in file_stats}
    stale_removed = 0
    for leftover in prepared_dir.glob("*.jsonl"):
        if leftover.name in expected_outputs:
            continue
        try:
            leftover.unlink()
            stale_removed += 1
        except OSError as exc:
            C.log(f"  警告：无法删除过期预处理文件 {leftover.name}：{exc}")
    if stale_removed:
        C.log(f"已清理 {stale_removed} 个过期预处理文件")

    manifest = {
        "generated_at": C.now_iso(),
        "config": cfg.get("_config_path"),
        "data_dir": str(data_dir),
        "prepared_dir": str(prepared_dir),
        "settings": {
            "dedupe": dedupe,
            "dedupe_scope": dedupe_scope,
            "dedupe_key": dedupe_key,
            "strip_cot": do_strip,
            "multi_turn": multi_turn,
        },
        "totals": {
            "files": len(file_stats),
            "lines": total_lines,
            "kept": total_kept,
            "duplicates": total_duplicates,
            "invalid": total_invalid,
            "bad_lines": total_bad,
            "stale_removed": stale_removed,
        },
        "files": file_stats,
        "elapsed_sec": round(time.time() - started, 2),
    }
    C.write_json_atomic(prepared_dir / "_manifest.json", manifest)

    print()
    C.print_table(
        ["数据集文件", "领域", "原始行", "保留", "去重", "无效", "坏行"],
        [
            [item["file"], item["domain"], item["lines"], item["kept"], item["duplicates"], item["invalid"], item["bad_lines"]]
            for item in file_stats
        ],
    )
    C.log(f"预处理完成：保留 {total_kept} 条，去重 {total_duplicates} 条，输出目录 {prepared_dir}")
    if total_kept == 0:
        C.log("警告：没有保留任何有效样本，请检查数据集格式")
        return 2
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        C.log("已中断")
        raise SystemExit(130)
