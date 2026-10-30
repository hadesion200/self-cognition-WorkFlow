#!/usr/bin/env bash
# =============================================================================
# QA 评测流水线一键运行脚本
#
# 用法：
#   sh run.sh                    # 使用 config.yaml（真实 API）
#   sh run.sh --mock             # 使用 config.mock.yaml（本地模拟，无需 API）
#   sh run.sh --config x.yaml    # 指定其他配置
#   sh run.sh --fresh            # 清空旧结果后重跑（谨慎）
# =============================================================================
set -eu
# dash 不支持 pipefail；bash 下启用，sh 下自动跳过
if (set -o pipefail) 2>/dev/null; then
  set -o pipefail
fi

cd "$(dirname "$0")" || exit 1

usage() {
  cat <<'EOF'
用法：sh run.sh [--mock] [--config <file>] [--fresh]

  --mock            使用 config.mock.yaml 做本地模拟测试
  --config <file>   指定 YAML 配置（默认 config.yaml）
  --fresh           清空预处理/回答/评分结果后重新开始
  -h, --help        显示帮助
EOF
}

CONFIG="config.yaml"
FRESH=0

while [ "$#" -gt 0 ]; do
  case "$1" in
    --mock)
      CONFIG="config.mock.yaml"
      shift
      ;;
    --config)
      if [ "$#" -lt 2 ]; then
        echo "错误：--config 需要一个文件路径" >&2
        exit 2
      fi
      CONFIG="$2"
      shift 2
      ;;
    --fresh)
      FRESH=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "错误：未知参数 $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [ ! -f "$CONFIG" ]; then
  echo "错误：配置文件不存在：$CONFIG" >&2
  exit 2
fi

if [ -n "${PYTHON:-}" ]; then
  PY="$PYTHON"
elif command -v python >/dev/null 2>&1; then
  PY="python"
elif command -v python3 >/dev/null 2>&1; then
  PY="python3"
else
  echo "错误：找不到 python，请先安装 Python 3.9+ 或设置 PYTHON 环境变量" >&2
  exit 1
fi

export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8

"$PY" - <<'PY'
import importlib.util
import sys

missing = [name for name in ("yaml", "requests") if importlib.util.find_spec(name) is None]
if missing:
    print("缺少依赖：" + ", ".join(missing), file=sys.stderr)
    print("请先执行：pip install -r requirements.txt", file=sys.stderr)
    sys.exit(1)
PY

FRESH_ARG=""
if [ "$FRESH" -eq 1 ]; then
  FRESH_ARG="--fresh"
fi

echo "============================================================"
echo "配置：$CONFIG"
echo "Python：$PY"
echo "============================================================"

echo ""
echo "===== 阶段 1/4：预处理（去重 + 去思维链）====="
"$PY" qa_prepare.py --config "$CONFIG" $FRESH_ARG

STATUS=0
run_continue() {
  if "$@"; then
    code=0
  else
    code=$?
  fi
  if [ "$code" -eq 130 ]; then
    echo "已中断，已完成结果均已保存。"
    exit 130
  fi
  if [ "$code" -eq 2 ]; then
    exit 2
  fi
  if [ "$code" -ne 0 ] && [ "$code" -gt "$STATUS" ]; then
    STATUS="$code"
  fi
  return 0
}

echo ""
echo "===== 阶段 2/4：被测模型并发问答 ====="
run_continue "$PY" qa_generate.py --config "$CONFIG" $FRESH_ARG

echo ""
echo "===== 阶段 3/4：评分模型多维评分 ====="
run_continue "$PY" qa_score.py --config "$CONFIG" $FRESH_ARG

echo ""
echo "===== 阶段 4/4：汇总报告 ====="
run_continue "$PY" qa_report.py --config "$CONFIG"

echo ""
if [ "$STATUS" -eq 0 ]; then
  echo "全部阶段完成。"
else
  echo "部分知识领域尚未完成（退出码 $STATUS），重新执行 sh run.sh 可续跑。"
fi
exit "$STATUS"
