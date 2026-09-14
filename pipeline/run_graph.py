"""Kol 2/Kol 3 (LangGraph) çalıştırıcısı.

pipeline/baseline.py'nin CLI desenini izler; --mode ile Kol 2 (naive) ya da
Kol 3 (contract) seçilir — aynı script, mode parametreli tek graf tasarımı
(bkz. pipeline/graph.py).

Kullanım:
    uv run python -m pipeline.run_graph --mode naive --tasks 3
    uv run python -m pipeline.run_graph --mode naive
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone

from config import (
    GRAPH_MODES,
    LOGS_DIR,
    MODEL_PILOT,
    PILOT_TASK_SET,
    model_alias_help,
    resolve_model,
)
from eval.harness import load_all_tasks
from pipeline.graph import build_graph


def run_task(graph, task: dict, mode: str,
             thread_id: str | None = None, model: str | None = None, *,
             experiment: str | None = None, run_id: str | None = None,
             repeat: int | None = None) -> dict:
    """Tek görevi graftan geçirir. thread_id/model, deney çalıştırıcının
    (eval/runner.py) tekrar indeksini ve model seçimini geçirebilmesi için.
    experiment/run_id/repeat, agents/llm.py'nin çağrı logu bağlamı içindir."""
    config = {"configurable": {"thread_id": thread_id or f"{mode}-{task['task_id']}"}}
    initial = {"task": task, "mode": mode, "attempt_count": 0}
    if model:
        initial["model"] = model
    if experiment:
        initial["experiment"] = experiment
    if run_id:
        initial["run_id"] = run_id
    if repeat is not None:  # "if repeat:" KULLANILMAZ -- repeat=0 falsy ama geçerli/en yaygın değer
        initial["repeat"] = repeat
    final = graph.invoke(initial, config)
    return {
        "ts": datetime.now(timezone.utc).isoformat(),
        "arm": mode,
        "plan": final["plan"],
        "code": final["code"],
        "attempt_count": final.get("attempt_count", 0),
        # Parse başarısı (structured+contract) ile sözleşme doğrulaması
        # (yalnız contract) AYRI sütunlar -- karıştırılırsa RQ3'ün ölçtüğü
        # fark analizde görünmez olur.
        "handoff_parse_ok": final.get("handoff_parse_ok"),
        "handoff_validation": final.get("handoff_validation"),
        "raw_messages": final["raw_messages"],
        **final["test_report"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Kol 2/Kol 3 (LangGraph) koşusu")
    parser.add_argument("--mode", choices=list(GRAPH_MODES), required=True)
    parser.add_argument("--tasks", type=int, default=None,
                        help="ilk N görevi koş (varsayılan: tümü)")
    # Debug/tek-kol CLI'ı; ana deney kayıtları eval/runner.py'den üretilir.
    parser.add_argument("--model", default=MODEL_PILOT, help=model_alias_help())
    # YALNIZ pilot: bu CLI manifest/provenance/resume üretmez, bu yüzden held-out
    # üzerinde ad hoc bir debug koşusu izlenebilir olmayan sonuç kaydı bırakırdı.
    parser.add_argument("--task-set", default=PILOT_TASK_SET, choices=[PILOT_TASK_SET],
                        help="YALNIZ pilot; held-out koşuları eval/runner.py üzerinden yapılır.")
    args = parser.parse_args()
    model = resolve_model(args.model)

    if not (os.environ.get("OPENROUTER_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")):
        sys.exit("API anahtarı yok — .env.example'ı .env olarak kopyalayıp doldur.")

    tasks = load_all_tasks(args.task_set)[: args.tasks]
    graph = build_graph(args.mode)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    LOGS_DIR.mkdir(exist_ok=True)
    out_path = LOGS_DIR / f"results_{args.mode}_{stamp}.jsonl"

    passed = 0
    with out_path.open("a", encoding="utf-8") as out:
        for i, task in enumerate(tasks, 1):
            record = run_task(graph, task, args.mode, model=model)
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
            passed += record["status"] == "passed"
            print(f"[{i}/{len(tasks)}] {task['task_id']}: {record['status']}"
                  + (f" ({record['error_class']})" if record.get("error_class") else ""))

    print(f"\nSonuç: {passed}/{len(tasks)} geçti -> {out_path}")


if __name__ == "__main__":
    main()
