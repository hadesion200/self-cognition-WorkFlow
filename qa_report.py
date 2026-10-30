#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""阶段 4：为每个 jsonl 知识领域生成独立 Markdown 报告。

读取    outputs/<数据集名>_model_response.jsonl
        outputs/<数据集名>_model_judge.jsonl
写出    outputs/<数据集名>_report.md

用法：python qa_report.py --config config.yaml
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import qa_common as C


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="阶段 4：按知识领域汇总评分报告")
    parser.add_argument("--config", default="config.yaml", help="YAML 配置文件路径")
    return parser.parse_args()


def write_text_atomic(path: str | Path, text: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
        fh.flush()
        try:
            os.fsync(fh.fileno())
        except OSError:
            pass
    os.replace(tmp, target)


def md_escape(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def summarize_group(rows: list[dict], dims: list[str], pass_score: float) -> dict:
    overalls = [row["overall"] for row in rows if isinstance(row.get("overall"), (int, float))]
    stats = C.summarize(overalls)
    dim_means: dict[str, dict[str, Any]] = {}
    for dim in dims:
        values = [
            row["scores"][dim]
            for row in rows
            if isinstance(row.get("scores"), dict) and isinstance(row["scores"].get(dim), (int, float))
        ]
        dim_means[dim] = {
            "mean": round(sum(values) / len(values), 4) if values else None,
            "n": len(values),
        }
    passed = sum(1 for value in overalls if value >= pass_score)
    return {
        **stats,
        "pass_rate": round(passed / len(overalls), 4) if overalls else None,
        "passed": passed,
        "dimension_means": dim_means,
    }


def build_rows(cfg: dict, records: list[dict], answers: dict[str, dict], scores: dict[str, dict]) -> list[dict]:
    rows: list[dict] = []
    for rec in records:
        key = rec["key"]
        answer_rec = answers.get(key) or {}
        score_rec = scores.get(key) or {}
        answer_fresh = answer_rec.get("input_hash") == C.content_hash_for_answer(cfg, rec)
        answered = bool(answer_rec.get("ok") and answer_rec.get("answer") and answer_fresh)
        score_fresh = score_rec.get("input_hash") == C.content_hash_for_score(
            cfg, rec, answer_rec.get("answer") or ""
        )
        scored_ok = bool(
            answered
            and score_fresh
            and score_rec.get("ok")
            and isinstance(score_rec.get("overall"), (int, float))
            and isinstance(score_rec.get("scores"), dict)
        )
        meta = rec.get("meta") or {}
        rows.append(
            {
                "key": key,
                "id": rec.get("id"),
                "domain": rec.get("domain") or "unknown",
                "category": meta.get("category") or rec.get("category") or "",
                "question": rec.get("question") or "",
                "answered": answered,
                "score_ok": scored_ok,
                "overall": score_rec.get("overall") if scored_ok else None,
                "scores": score_rec.get("scores") if scored_ok else {},
                "reason": (score_rec.get("reason") or "") if scored_ok else "",
                "answer_error": answer_rec.get("error"),
                "score_error": score_rec.get("error"),
            }
        )
    return rows


def build_category_table(rows: list[dict], dims: list[str], pass_score: float, bands: list) -> list[dict]:
    groups: dict[str, list[dict]] = {}
    for row in rows:
        category = str(row.get("category") or "").strip()
        if category:
            groups.setdefault(category, []).append(row)
    table = []
    for category, items in groups.items():
        scored = [item for item in items if item.get("score_ok")]
        stats = summarize_group(scored, dims, pass_score)
        table.append(
            {
                "category": category,
                "prepared": len(items),
                "scored": len(scored),
                "mean": stats["mean"],
                "pass_rate": stats["pass_rate"],
                "grade": C.grade_of(stats["mean"], bands),
            }
        )
    table.sort(key=lambda item: (item["mean"] is None, -(item["mean"] or 0), item["category"].lower()))
    return table


def fmt_score(value: Any) -> str:
    return f"{value:.4f}" if isinstance(value, (int, float)) else "N/A"


def fmt_pct(value: Any) -> str:
    return f"{value:.1%}" if isinstance(value, (int, float)) else "N/A"


def render_markdown(domain: str, cfg: dict, metrics: dict) -> str:
    model_cfg = cfg["model_under_test"]
    judge_cfg = cfg["judge"]
    overall = metrics["overall"]
    totals = metrics["totals"]
    dims = metrics["dims"]
    status = "完整" if metrics["complete"] else "部分完成（存在未答或未评分样本）"

    lines = [
        f"# QA 评测报告：{domain}",
        "",
        f"- 数据集：`{domain}`",
        f"- 报告状态：{status}",
        f"- 被测模型：`{model_cfg.get('model')}`（模式：{model_cfg.get('mode')}）",
        f"- 评分模型：`{judge_cfg.get('model')}`（模式：{judge_cfg.get('mode')}）",
        f"- 样本规模：{totals['prepared']} 条；已答 {totals['answered']} 条；已评分 {totals['scored']} 条",
        f"- 生成时间：{C.now_iso()}",
        "",
        "## 综合得分（满分 1 分）",
        "",
        f"**{fmt_score(overall['mean'])} / 1.0000**（{fmt_pct(overall['mean'])}，{overall['grade']}）",
        "",
        "| 指标 | 数值 |",
        "| --- | --- |",
        f"| 平均分 | {fmt_score(overall['mean'])} |",
        f"| 中位数 | {fmt_score(overall['median'])} |",
        f"| 最低 / 最高 | {fmt_score(overall['min'])} / {fmt_score(overall['max'])} |",
        f"| 标准差 | {fmt_score(overall['std'])} |",
        f"| 通过率（≥{metrics['pass_score']:.2f}） | {fmt_pct(overall['pass_rate'])} |",
        f"| 覆盖率 | {fmt_pct(totals['coverage'])} |",
        "",
        "## 各维度均分",
        "",
        "| 维度 | 均分 | 有效样本 | 权重 |",
        "| --- | --- | --- | --- |",
    ]
    for dim in dims:
        info = overall["dimension_means"].get(dim, {})
        weight = metrics["weights"].get(dim, 0)
        lines.append(f"| {dim} | {fmt_score(info.get('mean'))} | {info.get('n', 0)} | {float(weight):.0%} |")

    lines += ["", "## 分数分布", "", "| 区间 | 数量 |", "| --- | --- |"]
    for label, count in metrics["distribution"].items():
        lines.append(f"| {label} | {count} |")

    if metrics["by_category"]:
        lines += [
            "",
            "## 按 category 字段汇总",
            "",
            "| category | 样本 | 已评分 | 平均分 | 通过率 | 等级 |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for item in metrics["by_category"]:
            lines.append(
                f"| {md_escape(item['category'])} | {item['prepared']} | {item['scored']} | "
                f"{fmt_score(item['mean'])} | {fmt_pct(item['pass_rate'])} | {item['grade']} |"
            )

    if metrics["worst_cases"]:
        lines += ["", "## 低分案例", ""]
        for index, item in enumerate(metrics["worst_cases"], 1):
            lines.append(f"### {index}. {item['key']}（{fmt_score(item['overall'])}）")
            lines.append("")
            lines.append(f"- 问题：{md_escape(item['question'])}")
            dim_text = "，".join(f"{dim}={fmt_score(item['scores'].get(dim))}" for dim in dims)
            lines.append(f"- 维度：{dim_text}")
            lines.append(f"- 理由：{md_escape(item['reason'])}")
            lines.append("")

    lines += [
        "",
        "## 说明",
        "",
        "- 所有分数均为 0~1，满分 1 分；综合分为各维度按权重加权。",
        "- 评分由 LLM 完成，分数与理由仅供质量参考，建议结合低分案例人工抽检。",
        "- 本报告只统计当前数据集中的样本；历史遗留记录不会计入。",
        "",
    ]
    return "\n".join(lines)


def process_dataset(cfg: dict, domain: str, records: list[dict]) -> dict:
    dims = list(cfg["scoring"]["weights"].keys())
    weights = {dim: float(cfg["scoring"]["weights"][dim]) for dim in dims}
    pass_score = float(cfg["scoring"].get("pass_score", 0.6))
    bands = cfg["report"].get("grade_bands") or [[0.85, "优秀"], [0.70, "良好"], [0.60, "及格"], [0.00, "待改进"]]
    top_n = max(0, int(cfg["report"].get("top_n_bad_cases") or 0))
    paths = C.dataset_output_paths(cfg, domain)

    if not Path(paths["responses"]).is_file() or not Path(paths["judge"]).is_file():
        missing = [name for name in ("responses", "judge") if not Path(paths[name]).is_file()]
        C.log(f"数据集 {domain}：缺少 {'/'.join(missing)} 文件，跳过报告")
        return {
            "domain": domain,
            "prepared": len(records),
            "answered": 0,
            "scored": 0,
            "mean": None,
            "pass_rate": None,
            "grade": "无数据",
            "complete": False,
            "report": None,
        }

    answers, answer_bad = C.read_jsonl(paths["responses"])
    scores, score_bad = C.read_jsonl(paths["judge"])
    answer_map = {str(item.get("key")): item for item in answers if item.get("key")}
    score_map = {str(item.get("key")): item for item in scores if item.get("key")}

    rows = build_rows(cfg, records, answer_map, score_map)
    answered_rows = [row for row in rows if row["answered"]]
    scored_rows = [row for row in rows if row["score_ok"]]
    overall = summarize_group(scored_rows, dims, pass_score)

    dimension_weighted = None
    if overall["dimension_means"]:
        total = 0.0
        total_weight = 0.0
        for dim, info in overall["dimension_means"].items():
            if info["mean"] is None:
                continue
            total += info["mean"] * float(weights.get(dim, 0))
            total_weight += float(weights.get(dim, 0))
        dimension_weighted = round(total / total_weight, 4) if total_weight else None

    bins = cfg["report"].get("bins") or [0.0, 0.6, 0.7, 0.8, 0.9, 1.0]
    distribution = C.score_distribution([row["overall"] for row in scored_rows], bins)
    worst_cases = sorted(scored_rows, key=lambda row: row["overall"])[:top_n] if top_n else []
    category_table = build_category_table(rows, dims, pass_score, bands)

    complete = len(answered_rows) == len(rows) and len(scored_rows) == len(rows)
    metrics = {
        "complete": complete,
        "dims": dims,
        "weights": weights,
        "pass_score": pass_score,
        "totals": {
            "prepared": len(rows),
            "answered": len(answered_rows),
            "scored": len(scored_rows),
            "coverage": round(len(answered_rows) / len(rows), 4) if rows else 0.0,
            "scored_rate": round(len(scored_rows) / len(rows), 4) if rows else 0.0,
            "bad_answer_lines": answer_bad,
            "bad_score_lines": score_bad,
        },
        "overall": {
            **overall,
            "grade": C.grade_of(overall["mean"], bands),
            "dimension_weighted_score": dimension_weighted,
            "full_score": 1.0,
        },
        "distribution": distribution,
        "by_category": category_table,
        "worst_cases": [
            {
                "key": row["key"],
                "domain": row["domain"],
                "question": C.clip(row["question"], 120),
                "overall": row["overall"],
                "scores": row["scores"],
                "reason": row["reason"],
            }
            for row in worst_cases
        ],
    }

    write_text_atomic(paths["report"], render_markdown(domain, cfg, metrics))
    C.log(f"数据集 {domain}：报告已生成 -> {Path(paths['report']).name}")
    return {
        "domain": domain,
        "prepared": len(rows),
        "answered": len(answered_rows),
        "scored": len(scored_rows),
        "mean": overall["mean"],
        "pass_rate": overall["pass_rate"],
        "grade": metrics["overall"]["grade"],
        "complete": complete,
        "report": paths["report"],
    }


def main() -> int:
    C.setup_stdout()
    args = parse_args()
    cfg = C.load_config(args.config)

    datasets = C.load_prepared_by_domain(cfg)
    if not datasets:
        C.log("错误：没有可用的预处理数据，请先运行 qa_prepare.py")
        return 2

    C.log(f"知识领域 {len(datasets)} 个 | 输出目录：{Path(cfg['paths']['output_dir']).resolve()}")
    results = [process_dataset(cfg, domain, records) for domain, records in datasets.items()]

    print()
    C.print_table(
        ["知识领域", "样本", "已答", "已评", "平均分", "通过率", "等级", "报告"],
        [
            [
                item["domain"],
                item["prepared"],
                item["answered"],
                item["scored"],
                fmt_score(item["mean"]),
                fmt_pct(item["pass_rate"]),
                item["grade"],
                Path(item["report"]).name if item["report"] else "缺失",
            ]
            for item in results
        ],
    )

    incomplete = [item for item in results if not item["complete"]]
    if incomplete:
        C.log(f"仍有 {len(incomplete)} 个数据集未完整报告，请重新运行续跑")
        return 3
    C.log("全部知识领域报告已生成")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        C.log("已中断")
        raise SystemExit(130)
