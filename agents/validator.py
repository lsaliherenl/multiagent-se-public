"""Sözleşme kapısı düğümü (Kol 3): planner çıktısını PlannerOutput şemasına
karşı doğrular (yapısal) VE task_id/entry_point eşleşmesini kontrol eder
(semantik). Doğrulama başarısızsa graf planner'a geri döner (bkz.
pipeline/graph.py'deki _route_after_validation, config.MAX_PLANNER_ATTEMPTS).

Bu düğümün retry sayısı ve hata türü, MAST hata kodlamasında (adım 7) Kol
3'e özgü bir hata kategorisi olarak kullanılacak.
"""

import re

from pydantic import ValidationError

from agents.contracts import PlannerOutput
from pipeline.state import PipelineState


def _entry_point_matches(entry_point: str, signature: str) -> bool:
    """Ad, imzanın BAŞINDA (opsiyonel 'async def '/'def ' önekiyle) geçmeli --
    sadece bir yerde geçmesi YETMEZ (ör. "helper(foo(...))" foo'yu yanlışlıkla
    kabul etmesin diye). Tip anotasyonu/boşluk farkları için metinsel birebir
    eşitlik ARANMAZ, sadece tanımlanan adın doğru olması kontrol edilir."""
    return re.search(rf"^\s*(?:(?:async\s+)?def\s+)?{re.escape(entry_point)}\s*\(", signature) is not None


def validate_handoff_node(state: PipelineState) -> dict:
    plan = state["plan"]
    try:
        parsed = PlannerOutput.model_validate(plan)
    except ValidationError as e:
        return {
            "handoff_validation": {"valid": False, "errors": str(e), "raw": plan},
            "raw_messages": [{"from": "validator", "to": "planner", "content": str(e)}],
        }

    task = state["task"]  # yapısal doğrulama BAŞARILI olduktan SONRA okunur
    errors = []
    if parsed.task_id != task["task_id"]:
        errors.append(f"task_id uyuşmuyor: beklenen {task['task_id']!r}, alınan {parsed.task_id!r}")
    if not _entry_point_matches(task["entry_point"], parsed.function_signature):
        errors.append(
            f"function_signature, entry_point {task['entry_point']!r} ile başlamıyor: "
            f"{parsed.function_signature!r}"
        )
    if errors:
        message = "; ".join(errors)
        return {
            "handoff_validation": {"valid": False, "errors": message, "raw": plan},
            "raw_messages": [{"from": "validator", "to": "planner", "content": message}],
        }

    return {
        "handoff_validation": {"valid": True, "errors": None},
        "raw_messages": [{"from": "validator", "to": "coder", "content": "sözleşme geçerli"}],
    }
