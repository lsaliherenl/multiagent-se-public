"""Kol 1 — tek-ajanlı baseline (kontrol grubu).

Görev başına TEK LLM çağrısı: prompt → kod bloğu ayıklama → harness.
Sonuçlar logs/results_baseline_<zaman>.jsonl dosyasına satır satır yazılır.

Kullanım:
    uv run python -m pipeline.baseline --tasks 3        # ilk 3 görev (mini pilot)
    uv run python -m pipeline.baseline                  # tüm görev seti
    uv run python -m pipeline.baseline --model <litellm-model-adi>
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone

from agents.llm import call_model
from agents.parsing import extract_code
from config import (
    DEFAULT_TEMPERATURE,
    LOGS_DIR,
    MODEL_PILOT,
    PILOT_TASK_SET,
    model_alias_help,
    resolve_model,
)
from eval.harness import evaluate_base_plus, load_all_tasks

SYSTEM_PROMPT = (
    "You are an expert Python programmer. Solve the given task. "
    "Respond with a single complete Python code block containing the full "
    "function definition. Do not include tests or explanations."
)


def run_task(task: dict, model: str, temperature: float, *,
             experiment: str | None = None, run_id: str | None = None,
             repeat: int | None = None) -> dict:
    response = call_model(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": task["prompt"]},
        ],
        model=model,
        temperature=temperature,
        task_id=task["task_id"],
        agent_role="baseline",
        experiment=experiment,
        run_id=run_id,
        arm="baseline",
        repeat=repeat,
    )
    code = extract_code(response.text)
    return {
        "ts": datetime.now(timezone.utc).isoformat(),
        "arm": "baseline",
        "model": model,
        "temperature": temperature,
        "code": code,
        **evaluate_base_plus(task, code),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Kol 1 tek-ajanlı baseline koşusu")
    parser.add_argument("--tasks", type=int, default=None,
                        help="ilk N görevi koş (varsayılan: tümü)")
    # Bu CLI geliştirme/debug içindir — ana deney kayıtları eval/runner.py
    # üzerinden üretilir ve orada --model ZORUNLUDUR. Buradaki pilot varsayılan
    # bilinçli: tek görevlik hızlı denemeler için. Takma ad da kabul edilir.
    parser.add_argument("--model", default=MODEL_PILOT, help=model_alias_help())
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    # YALNIZ pilot: bu CLI manifest/provenance/resume üretmez, bu yüzden held-out
    # üzerinde ad hoc bir debug koşusu izlenebilir olmayan sonuç kaydı bırakırdı.
    parser.add_argument("--task-set", default=PILOT_TASK_SET, choices=[PILOT_TASK_SET],
                        help="YALNIZ pilot; held-out koşuları eval/runner.py üzerinden yapılır.")
    args = parser.parse_args()
    model = resolve_model(args.model)

    if not (os.environ.get("OPENROUTER_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")):
        sys.exit("API anahtarı yok — .env.example'ı .env olarak kopyalayıp doldur.")

    tasks = load_all_tasks(args.task_set)[: args.tasks]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    LOGS_DIR.mkdir(exist_ok=True)
    out_path = LOGS_DIR / f"results_baseline_{stamp}.jsonl"

    passed = 0
    with out_path.open("a", encoding="utf-8") as out:
        for i, task in enumerate(tasks, 1):
            record = run_task(task, model, args.temperature)
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
            passed += record["status"] == "passed"
            print(f"[{i}/{len(tasks)}] {task['task_id']}: {record['status']}"
                  + (f" ({record['error_class']})" if record["error_class"] else ""))

    print(f"\nSonuç: {passed}/{len(tasks)} geçti -> {out_path}")


if __name__ == "__main__":
    main()
