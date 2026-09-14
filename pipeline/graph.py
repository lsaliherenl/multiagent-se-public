"""Çok-ajanlı kolların grafı — mode parametreli TEK dosya.

Üç kol da AYNI planner/coder/tester düğümlerini kullanır. Düğüm KÜMESİ değil,
yalnız düğümler ARASI kenarlar moda göre değişir:

    naive:                    START → planner → coder → tester → END
    structured_no_validation: START → planner → coder → tester → END
    contract:                 START → planner → validator → coder → tester → END
                                                   ↳ (geçersiz) → planner

naive ↔ structured farkı: planner'ın ürettiği/gönderdiği temsil (serbest metin
vs. canonical JSON) — RQ2'nin ölçtüğü şey.
structured ↔ contract farkı: SADECE validator düğümü + hata geri bildirimli
sınırlı retry kenarı — RQ3'ün ölçtüğü şey. structured için AYRI planner/coder
düğümü YAZILMAZ; ayrı implementasyon "tek fark validator+retry" şartını
kırardı ve RQ3'ü ölçülemez hale getirirdi.
"""

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from agents.coder import coder_node
from agents.planner import planner_node
from agents.tester import tester_node
from agents.validator import validate_handoff_node
from config import ARM_CONTRACT, ARM_NAIVE, GRAPH_MODES, MAX_PLANNER_ATTEMPTS
from pipeline.state import PipelineState


def _route_after_validation(state: PipelineState) -> str:
    validation = state.get("handoff_validation") or {}
    if validation.get("valid"):
        return "proceed"
    if state.get("attempt_count", 0) < MAX_PLANNER_ATTEMPTS:
        return "retry"
    return "proceed"  # denemeler tükendi, coder elindeki en iyi planla devam eder


def build_graph(mode: str = ARM_NAIVE, checkpointer=None):
    """Derlenmiş graf döndürür. mode: config.GRAPH_MODES üyelerinden biri."""
    if mode not in GRAPH_MODES:
        raise ValueError(f"bilinmeyen mode: {mode!r}")

    graph = StateGraph(PipelineState)
    graph.add_node("planner", planner_node)
    graph.add_node("coder", coder_node)
    graph.add_node("tester", tester_node)
    graph.add_edge(START, "planner")

    if mode == ARM_CONTRACT:
        graph.add_node("validate_handoff", validate_handoff_node)
        graph.add_edge("planner", "validate_handoff")
        graph.add_conditional_edges(
            "validate_handoff", _route_after_validation,
            {"retry": "planner", "proceed": "coder"},
        )
    else:
        # naive VE structured_no_validation: doğrulama kapısı yok, retry yok.
        graph.add_edge("planner", "coder")

    graph.add_edge("coder", "tester")
    graph.add_edge("tester", END)

    # Checkpointing: time-travel debugging + LangSmith izlenebilirlik.
    return graph.compile(checkpointer=checkpointer or InMemorySaver())
