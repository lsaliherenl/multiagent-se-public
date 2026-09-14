"""Görev yükleme ve aday kod değerlendirme.

Aday kod + görevin test kodu tek script'te birleştirilir, sandbox'ta koşturulur
ve sonuç yapılandırılmış EvalResult'a çevrilir. Hata sınıfı ayrımı ve
kısaltılmış traceback, MAST etiketlemesinin (adım 7) ve tester→coder geri
bildirim döngüsünün (2. hafta) girdisidir.
"""

import json
from dataclasses import dataclass
from pathlib import Path

from config import (
    PLUS_TIMEOUT_S,
    SANDBOX_TIMEOUT_S,
    TASK_SETS,
    TASKS_DIR,
    TRACEBACK_MAX_LINES,
)
from eval.sandbox import run_code

TRACEBACK_MAX_BYTES = 4096


@dataclass
class EvalResult:
    task_id: str
    status: str  # "passed" | "failed" | "timeout"
    error_class: str | None  # "syntax" | "assertion" | "runtime" | "timeout" | None
    traceback: str | None  # kısaltılmış (son N satır), JSONL'e aynen yazılır
    duration_s: float


def _tasks_dir(task_set: str | None) -> Path:
    if task_set is None:
        return TASKS_DIR
    try:
        return TASK_SETS[task_set]
    except KeyError:
        raise ValueError(
            f"bilinmeyen görev seti: {task_set!r} (seçenekler: {sorted(TASK_SETS)})"
        ) from None


def load_task(task_id: str, task_set: str | None = None) -> dict:
    return json.loads((_tasks_dir(task_set) / f"{task_id}.json").read_text(encoding="utf-8"))


def load_all_tasks(task_set: str | None = None) -> list[dict]:
    """Görevleri task_id sırasıyla döndürür (deterministik koşu sırası).

    task_set=None -> pilot/development seti (geriye dönük varsayılan).
    Ana deney "heldout" setini AÇIKÇA ister; iki set ayrı dizinlerde durur ki
    yanlışlıkla karışmasınlar (EXPERIMENT_PROTOCOL.md §5).

    "_" ile başlayan dosyalar ATLANIR: görev dizini yardımcı belgeler de taşır
    (ör. _selection_manifest.json) ve bunların görev sanılması, görev sayısını
    sessizce şişirip koşuyu bozardı.
    """
    return [
        json.loads(p.read_text(encoding="utf-8"))
        for p in sorted(_tasks_dir(task_set).glob("*.json"))
        if not p.name.startswith("_")
    ]


def _classify_error(stderr: str) -> str:
    if "SyntaxError" in stderr or "IndentationError" in stderr:
        return "syntax"
    if "AssertionError" in stderr:
        return "assertion"
    return "runtime"


def _truncate_traceback(stderr: str) -> str:
    lines = stderr.strip().splitlines()[-TRACEBACK_MAX_LINES:]
    return "\n".join(lines)[-TRACEBACK_MAX_BYTES:]


def evaluate(task: dict, candidate_code: str,
             timeout_s: float = SANDBOX_TIMEOUT_S,
             test_code: str | None = None) -> EvalResult:
    """Aday kodu görevin testlerine karşı sandbox'ta çalıştırır.

    test_code verilmezse görevin varsayılan `test_code` alanı kullanılır
    (pilot seti). Held-out görevlerde base/plus test kodları ayrı alanlarda
    durur ve evaluate_base_plus() bunları açıkça geçirir.
    """
    script = (
        f"{candidate_code}\n\n"
        f"{test_code if test_code is not None else task['test_code']}\n\n"
        f"check({task['entry_point']})\n"
    )
    result = run_code(script, timeout_s=timeout_s)
    if result.status == "timeout":
        return EvalResult(task["task_id"], "timeout", "timeout", None, result.duration_s)
    if result.status == "ok":
        return EvalResult(task["task_id"], "passed", None, None, result.duration_s)
    return EvalResult(
        task_id=task["task_id"],
        status="failed",
        error_class=_classify_error(result.stderr),
        traceback=_truncate_traceback(result.stderr),
        duration_s=result.duration_s,
    )


def has_base_plus(task: dict) -> bool:
    """Görev EvalPlus tabanlı base/plus test çiftini taşıyor mu (held-out set)."""
    return "base_test_code" in task and "plus_test_code" in task


def evaluate_base_plus(task: dict, candidate_code: str) -> dict:
    """Base ve Plus testlerini AYRI çalıştırıp §5.5'in alanlarını üretir.

    Birincil başarı metriği plus_pass, ikincil base_pass (§5.5). İkisi ayrı
    koşulur çünkü "base'i geçip Plus'ta elenen çözüm oranı" RQ5'in keşifsel
    sorularından biri — tek bir birleşik koşu bu bilgiyi yok ederdi.

    Plus testleri base girdilerini DE içerir (EvalPlus semantiği: plus skoru
    her iki girdi kümesini birden geçmeyi gerektirir).

    Base/plus taşımayan (pilot) görevlerde tek koşu yapılır ve base alanları
    plus alanlarını yansıtır — analiz katmanı tek bir şema görür.
    """
    if not has_base_plus(task):
        single = evaluate(task, candidate_code)
        return {
            **{f"base_{k}": v for k, v in _result_fields(single).items()},
            **{f"plus_{k}": v for k, v in _result_fields(single).items()},
            "base_plus_available": False,
            **_legacy_fields(single),
        }

    base = evaluate(task, candidate_code, SANDBOX_TIMEOUT_S, task["base_test_code"])
    # Base düşerse Plus'ı koşturmaya gerek yok: Plus base'i kapsadığı için
    # zorunlu olarak o da düşer. Hem süre hem gereksiz sandbox yükü tasarrufu;
    # plus alanları base'in sonucundan türetilir (uydurulmaz, kopyalanır).
    plus = (evaluate(task, candidate_code, PLUS_TIMEOUT_S, task["plus_test_code"])
            if base.status == "passed" else base)
    return {
        **{f"base_{k}": v for k, v in _result_fields(base).items()},
        **{f"plus_{k}": v for k, v in _result_fields(plus).items()},
        "base_plus_available": True,
        "plus_skipped_base_failed": base.status != "passed",
        **_legacy_fields(plus),
    }


def _result_fields(r: EvalResult) -> dict:
    return {
        "status": r.status,
        "pass": r.status == "passed",
        "error_class": r.error_class,
        "traceback": r.traceback,
        "duration_s": r.duration_s,
    }


def _legacy_fields(r: EvalResult) -> dict:
    """Mevcut kayıt şemasının düz alanları (status/error_class/...).

    Birincil metrik plus_pass olduğu için bunlar PLUS sonucunu yansıtır;
    baseline/graph kayıtlarının ve mevcut özet/MAST kodunun çalışmaya devam
    etmesi için korunur.
    """
    return {
        "task_id": r.task_id,
        "status": r.status,
        "error_class": r.error_class,
        "traceback": r.traceback,
        "duration_s": r.duration_s,
    }
