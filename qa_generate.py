#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""阶段 2：按 jsonl 知识领域并发调用被测模型。

每个数据集独立输出：
    outputs/<数据集名>_model_response.jsonl

每条记录实时追加保存，中断后重新运行会跳过已完成样本。

用法：python qa_generate.py --config config.yaml [--fresh]
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import qa_common as C


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="阶段 2：被测模型按知识领域并发问答")
    parser.add_argument("--config", default="config.yaml", help="YAML 配置文件路径")
    parser.add_argument("--fresh", action="store_true", help="清空各数据集的回答文件后重新生成")
    return parser.parse_args()


def make_record(cfg: dict, rec: dict, input_hash: str, result: dict | None, error: str | None) -> dict:
    model_cfg = cfg["model_under_test"]
    answer = None
    if result is not None:
        answer = C.strip_cot(result.get("text")).strip()
        if not answer:
            error = "模型回答去除思维链后为空"
            result = None

    return {
        "key": rec["key"],
        "id": rec.get("id"),
        "domain": rec.get("domain"),
        "question": rec.get("question"),
        "reference_answer": rec.get("reference_answer"),
        "answer": answer,
        "ok": bool(answer),
        "error": error,
        "input_hash": input_hash,
        "model": model_cfg.get("model"),
        "model_id": C.model_id_of(model_cfg),
        "mode": model_cfg.get("mode"),
        "latency_ms": (result or {}).get("latency_ms"),
        "usage": (result or {}).get("usage") or {},
        "source": rec.get("source"),
        "created_at": C.now_iso(),
    }


def count_complete(cfg: dict, records: list[dict], saved: dict[str, dict]) -> int:
    complete = 0
    for rec in records:
        item = saved.get(rec["key"])
        if item and item.get("ok") and item.get("input_hash") == C.content_hash_for_answer(cfg, rec):
            complete += 1
    return complete


def process_dataset(cfg: dict, domain: str, records: list[dict], fresh: bool) -> dict:
    runtime = cfg["runtime"]
    paths = C.dataset_output_paths(cfg, domain)
    Path(paths["responses"]).parent.mkdir(parents=True, exist_ok=True)
    store = C.JsonlStore(paths["responses"], fsync_every=int(runtime.get("fsync_every") or 1))

    resume = bool(runtime.get("resume", True)) and not fresh
    if not resume:
        store.truncate()

    existing, bad_lines, valid_lines = store.load_last()
    if bad_lines:
        C.log(f"  {domain}：回答文件有 {bad_lines} 行无法解析，已忽略")
    duplicates = max(0, valid_lines - len(existing))
    if duplicates:
        C.log(f"  {domain}：压缩 {duplicates} 条重复/重试记录")
        store.compact(existing.values())

    pending: list[tuple[dict, str]] = []
    reused = skipped_failed = replaced = 0
    for rec in records:
        input_hash = C.content_hash_for_answer(cfg, rec)
        old = existing.get(rec["key"]) if resume else None
        if old and old.get("input_hash") == input_hash:
            if old.get("ok"):
                reused += 1
                continue
            if not bool(runtime.get("retry_failed", True)):
                skipped_failed += 1
                continue
        if old:
            replaced += 1
        pending.append((rec, input_hash))

    C.log(f"数据集 {domain}：{len(records)} 条 | 待生成 {len(pending)} | 已有有效 {reused}")
    if not pending:
        complete = count_complete(cfg, records, existing)
        C.log(f"数据集 {domain}：无需生成，完成 {complete}/{len(records)}")
        return {"domain": domain, "total": len(records), "complete": complete, "path": paths["responses"]}

    model_cfg = cfg["model_under_test"]
    workers = max(1, min(int(runtime.get("workers") or 1), len(pending)))

    def run_one(rec: dict, input_hash: str) -> dict:
        try:
            result = C.chat_completion(
                model_cfg,
                rec["prompt_messages"],
                purpose="generate",
                runtime=runtime,
                label="被测模型",
            )
            return make_record(cfg, rec, input_hash, result, None)
        except Exception as exc:  # noqa: BLE001 - 单条失败不应中断整个数据集
            return make_record(cfg, rec, input_hash, None, str(exc))

    progress = C.Progress(
        domain,
        len(records),
        initial_done=len(records) - len(pending),
        initial_ok=reused,
        initial_fail=skipped_failed,
        interval=float(runtime.get("progress_interval_sec") or 0.2),
    )
    executor = ThreadPoolExecutor(max_workers=workers)
    futures = {executor.submit(run_one, rec, h): rec for rec, h in pending}
    interrupted = False

    try:
        for future in as_completed(futures):
            record = future.result()
            store.append(record)
            if not record["ok"]:
                C.log(f"  失败：{record['key']} -> {record['error']}")
            progress.update(ok=bool(record["ok"]))
    except KeyboardInterrupt:
        interrupted = True
        C.log("收到中断信号：已完成结果已保存，重新运行即可续跑")
        for item in futures:
            item.cancel()
    finally:
        progress.close()
        executor.shutdown(wait=not interrupted, cancel_futures=interrupted)

    final, _, _ = store.load_last()
    if replaced and not interrupted:
        store.compact(final.values())
        C.log(f"数据集 {domain}：压缩 {replaced} 条旧版本记录")

    complete = count_complete(cfg, records, final)
    C.log(f"数据集 {domain}：完成 {complete}/{len(records)}")
    return {
        "domain": domain,
        "total": len(records),
        "complete": complete,
        "path": paths["responses"],
        "interrupted": interrupted,
    }


def main() -> int:
    C.setup_stdout()
    args = parse_args()
    cfg = C.load_config(args.config)
    runtime = cfg["runtime"]
    model_cfg = cfg["model_under_test"]

    datasets = C.load_prepared_by_domain(cfg)
    if not datasets:
        C.log("错误：没有可用的预处理数据，请先运行 qa_prepare.py")
        return 2

    total_records = sum(len(records) for records in datasets.values())
    C.log(f"被测模型：{model_cfg.get('model')}（模式：{model_cfg.get('mode')}）")
    C.log(f"知识领域 {len(datasets)} 个 | 样本总数 {total_records} | 并发 {runtime.get('workers')}")
    C.log(f"输出目录：{Path(cfg['paths']['output_dir']).resolve()}")

    results = []
    for domain, records in datasets.items():
        result = process_dataset(cfg, domain, records, fresh=args.fresh)
        results.append(result)
        if result.get("interrupted"):
            C.log("阶段 2 已中断")
            return 130

    incomplete = [item for item in results if item["complete"] < item["total"]]
    C.log("阶段 2 汇总：")
    for item in results:
        C.log(f"  {item['domain']}: {item['complete']}/{item['total']} -> {Path(item['path']).name}")
    if incomplete:
        C.log(f"仍有 {len(incomplete)} 个数据集未完成，请重新运行本阶段续跑")
        return 3
    C.log("阶段 2 已完成全部知识领域")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        C.log("已中断，进度已保留")
        raise SystemExit(130)
