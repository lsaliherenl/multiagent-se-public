"""validate_handoff_node birim testleri: LLM'siz, sözleşme doğrulama mantığı."""

from agents.validator import validate_handoff_node

VALID_PLAN = {
    "task_id": "t1",
    "function_signature": "def f(x: int) -> int",
    "steps": [{"description": "do it", "preconditions": [],
               "postconditions": ["returns x incremented by one"]}],
    "edge_cases": [],
}
TASK = {"task_id": "t1", "entry_point": "f"}


def test_gecerli_plan_kabul_edilir():
    result = validate_handoff_node({"plan": VALID_PLAN, "task": TASK})
    assert result["handoff_validation"]["valid"] is True
    assert result["raw_messages"][0]["from"] == "validator"


def test_eksik_alan_reddedilir():
    # steps eksik -> YAPISAL hata, state["task"]'a hiç ulaşılmaz (fixture gerekmiyor)
    result = validate_handoff_node({"plan": {"task_id": "t1"}})
    assert result["handoff_validation"]["valid"] is False
    assert "steps" in result["handoff_validation"]["errors"]


def test_bos_adim_listesi_reddedilir():
    bad = {**VALID_PLAN, "steps": []}
    result = validate_handoff_node({"plan": bad})
    assert result["handoff_validation"]["valid"] is False


def test_parse_hatali_plan_reddedilir():
    result = validate_handoff_node({"plan": {"_raw_text": "not json", "_parse_error": "..."}})
    assert result["handoff_validation"]["valid"] is False


def test_gorev_kimligi_uyusmazsa_reddedilir():
    plan = {**VALID_PLAN, "task_id": "wrong"}
    result = validate_handoff_node({"plan": plan, "task": TASK})
    assert result["handoff_validation"]["valid"] is False
    assert "task_id" in result["handoff_validation"]["errors"]


def test_entry_point_imzada_yoksa_reddedilir():
    plan = {**VALID_PLAN, "function_signature": "def other_name(x: int) -> int"}
    result = validate_handoff_node({"plan": plan, "task": TASK})
    assert result["handoff_validation"]["valid"] is False


def test_entry_point_metnin_ortasinda_reddedilir():
    # "helper(foo(...))" foo'yu YANLIŞLIKLA kabul etmemeli -- ad imzanın
    # BAŞINDA olmalı, sadece bir yerde geçmesi yetmez.
    plan = {**VALID_PLAN, "function_signature": "helper(foo(x))"}
    result = validate_handoff_node({"plan": plan, "task": {"task_id": "t1", "entry_point": "foo"}})
    assert result["handoff_validation"]["valid"] is False


def test_bos_son_kosul_reddedilir():
    # steps VAR ama postconditions boş -- steps=[] testinden FARKLI senaryo
    bad = {**VALID_PLAN, "steps": [{"description": "do it", "preconditions": [], "postconditions": []}]}
    result = validate_handoff_node({"plan": bad, "task": TASK})
    assert result["handoff_validation"]["valid"] is False


def test_on_kosul_bos_liste_olabilir():
    # regresyon: preconditions hâlâ opsiyonel (postconditions'ın aksine)
    result = validate_handoff_node({"plan": VALID_PLAN, "task": TASK})
    assert result["handoff_validation"]["valid"] is True


def test_beklenmeyen_alan_reddedilir():
    bad = {**VALID_PLAN, "surprise_field": 1}
    result = validate_handoff_node({"plan": bad, "task": TASK})
    assert result["handoff_validation"]["valid"] is False
