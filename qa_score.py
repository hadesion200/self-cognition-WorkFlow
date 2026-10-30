#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""阶段 3：按 jsonl 知识领域调用评分模型。

读取    outputs/<数据集名>_model_response.jsonl
写出    outputs/<数据集名>_model_judge.jsonl

用法：python qa_score.py --config config.yaml [--fresh] [--allow-partial]
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import qa_common as C


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="阶段 3：评分模型按知识领域多维评分")
    parser.add_argument("--config", default="config.yaml", help="YAML 配置文件路径")
    parser.add_argument("--fresh", action="store_true", help="清空各数据集的评分文件后重新评分")
    parser.add_argument("--allow-partial", action="store_true", help="回答不完整的数据集也继续评分")
    return parser.parse_args()


def count_complete(cfg: dict, records: list[dict], answers: dict[str, dict], saved: dict[str, dict]) -> int:
    complete = 0
    for rec in records:
        answer_rec = answers.get(rec["key"]) or {}
        answer = answer_rec.get("answer")
        if not answer_rec.get("ok") or not answer:
            continue
        item = saved.get(rec["key"])
        if item and item.get("ok") and item.get("input_hash") == C.content_hash_for_score(cfg, rec, answer):
            complete += 1
    return complete


def process_dataset(cfg: dict, domain: str, records: list[dict], fresh: bool, allow_partial: bool) -> dict:
    runtime = cfg["runtime"]
    weights: dict[str, float] = {str(k): float(v) for k, v in cfg["scoring"]["weights"].items()}
    paths = C.dataset_output_paths(cfg, domain)

    response_path = Path(paths["responses"])
    if not response_path.is_file():
        C.log(f"数据集 {domain}：找不到回答文件，跳过评分 -> {response_path.name}")
        return {"domain": domain, "total": len(records), "answerable": 0, "complete": 0, "missing": len(records)}

    answer_store = C.JsonlStore(response_path)
    answers, bad_lines, _ = answer_store.load_last()
    if bad_lines:
        C.log(f"  {domain}：回答文件有 {bad_lines} 行无法解析，已忽略")

    answer_by_key = {rec["key"]: answers.get(rec["key"]) for rec in records}
    missing = []
    for rec in records:
        answer_rec = answer_by_key.get(rec["key"]) or {}
        if (
            not answer_rec.get("ok")
            or not answer_rec.get("answer")
            or answer_rec.get("input_hash") != C.content_hash_for_answer(cfg, rec)
        ):
            missing.append(rec)

    answered = len(records) - len(missing)
    if missing and not allow_partial:
        C.log(f"数据集 {domain}：回答 {answered}/{len(records)}，还有 {len(missing)} 条未完成")
        C.log(f"数据集 {domain}：跳过评分（确认可放弃后可加 --allow-partial）")
        return {
            "domain": domain,
            "total": len(records),
            "answerable": answered,
            "complete": 0,
            "missing": len(missing),
            "skipped": True,
        }

    score_store = C.JsonlStore(paths["judge"], fsync_every=int(runtime.get("fsync_every") or 1))
    resume = bool(runtime.get("resume", True)) and not fresh
    if not resume:
        score_store.truncate()

    existing, score_bad, score_lines = score_store.load_last()
    if score_bad:
        C.log(f"  {domain}：评分文件有 {score_bad} 行无法解析，已忽略")
    duplicates = max(0, score_lines - len(existing))
    if duplicates:
        C.log(f"  {domain}：压缩 {duplicates} 条重复/重试评分记录")
        score_store.compact(existing.values())

    pending: list[tuple[dict, str, str]] = []
    reused = skipped_failed = replaced = 0
    for rec in records:
        answer_rec = answer_by_key.get(rec["key"]) or {}
        answer = answer_rec.get("answer")
        if not answer_rec.get("ok") or not answer:
            continue
        input_hash = C.content_hash_for_score(cfg, rec, answer)
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
        pending.append((rec, answer, input_hash))

    C.log(f"数据集 {domain}：回答 {answered}/{len(records)} | 待评分 {len(pending)} | 已有有效 {reused}")
    if not pending:
        complete = count_complete(cfg, records, answers, existing)
        C.log(f"数据集 {domain}：无需评分，完成 {complete}/{answered}")
        return {
            "domain": domain,
            "total": len(records),
            "answerable": answered,
            "complete": complete,
            "missing": len(missing),
        }

    judge_cfg = cfg["judge"]
    workers = max(1, min(int(runtime.get("workers") or 1), len(pending)))
    progress = C.Progress(
        domain,
        answered,
        initial_done=answered - len(pending),
        initial_ok=reused,
        initial_fail=skipped_failed,
        interval=float(runtime.get("progress_interval_sec") or 0.2),
    )

    def run_one(rec: dict, answer: str, input_hash: str) -> dict:
        started = time.time()
        record = {
            "key": rec["key"],
            "id": rec.get("id"),
            "domain": rec.get("domain"),
            "question": rec.get("question"),
            "reference_answer": rec.get("reference_answer"),
            "answer": answer,
            "scores": {dim: None for dim in weights},
            "overall": None,
            "reason": "",
            "ok": False,
            "error": None,
            "input_hash": input_hash,
            "judge_model": judge_cfg.get("model"),
            "mode": judge_cfg.get("mode"),
            "latency_ms": None,
            "usage": {},
            "created_at": C.now_iso(),
        }
        try:
            messages = C.build_judge_messages(cfg, rec.get("question") or "", rec.get("reference_answer") or "", answer)
            result = C.chat_completion(judge_cfg, messages, purpose="judge", runtime=runtime, label="评分模型")
            record["latency_ms"] = result.get("latency_ms")
            record["usage"] = result.get("usage") or {}
            dims, overall, reason, error = C.parse_judge_output(result["text"], weights)
            if error or dims is None or overall is None:
                record["error"] = error or "评分解析失败"
                if judge_cfg.get("keep_raw"):
                    record["raw_response"] = result["text"]
                return record
            record.update(
                {
                    "scores": dims,
                    "overall": overall,
                    "reason": reason,
                    "ok": True,
                    "elapsed_sec": round(time.time() - started, 3),
                }
            )
            if judge_cfg.get("keep_raw"):
                record["raw_response"] = result["text"]
        except Exception as exc:  # noqa: BLE001 - 单条失败继续
            record["error"] = str(exc)
            record["elapsed_sec"] = round(time.time() - started, 3)
        return record

    executor = ThreadPoolExecutor(max_workers=workers)
    futures = {executor.submit(run_one, rec, answer, h): rec for rec, answer, h in pending}
    interrupted = False
    try:
        for future in as_completed(futures):
            record = future.result()
            score_store.append(record)
            if not record["ok"]:
                C.log(f"  评分失败：{record['key']} -> {record['error']}")
            progress.update(ok=bool(record["ok"]))
    except KeyboardInterrupt:
        interrupted = True
        C.log("收到中断信号：已完成评分已保存，重新运行即可续跑")
        for item in futures:
            item.cancel()
    finally:
        progress.close()
        executor.shutdown(wait=not interrupted, cancel_futures=interrupted)

    final, _, _ = score_store.load_last()
    if replaced and not interrupted:
        score_store.compact(final.values())
        C.log(f"数据集 {domain}：压缩 {replaced} 条旧版本评分记录")

    complete = count_complete(cfg, records, answers, final)
    C.log(f"数据集 {domain}：完成 {complete}/{answered}")
    return {
        "domain": domain,
        "total": len(records),
        "answerable": answered,
        "complete": complete,
        "missing": len(missing),
        "interrupted": interrupted,
    }


def main() -> int:
    C.setup_stdout()
    args = parse_args()
    cfg = C.load_config(args.config)
    runtime = cfg["runtime"]
    judge_cfg = cfg["judge"]

    datasets = C.load_prepared_by_domain(cfg)
    if not datasets:
        C.log("错误：没有可用的预处理数据，请先运行 qa_prepare.py")
        return 2

    C.log(f"评分模型：{judge_cfg.get('model')}（模式：{judge_cfg.get('mode')}）")
    C.log(f"知识领域 {len(datasets)} 个 | 并发 {runtime.get('workers')}")

    results = []
    for domain, records in datasets.items():
        result = process_dataset(cfg, domain, records, fresh=args.fresh, allow_partial=args.allow_partial)
        results.append(result)
        if result.get("interrupted"):
            C.log("阶段 3 已中断")
            return 130

    incomplete = [
        item
        for item in results
        if item.get("skipped") or item["complete"] < item["answerable"] or item["missing"] > 0
    ]
    C.log("阶段 3 汇总：")
    for item in results:
        C.log(
            f"  {item['domain']}: 已评分 {item['complete']}/{item['answerable']}"
            f"（总样本 {item['total']}，缺回答 {item['missing']}）"
        )
    if incomplete:
        C.log(f"仍有 {len(incomplete)} 个数据集未完成，请重新运行本阶段续跑")
        return 3
    C.log("阶段 3 已完成全部知识领域")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        C.log("已中断，进度已保留")
        raise SystemExit(130)
