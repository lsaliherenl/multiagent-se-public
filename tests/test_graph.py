"""Graf akışı testi: LLM çağrıları mock'lanarak akış + checkpointing doğrulanır.

Her iki mod da artık gerçek call_model() kullanıyor (agents/planner.py,
agents/coder.py) — pytest'in ağsız/ücretsiz/deterministik kalması için
agents.planner.call_model ve agents.coder.call_model monkeypatch'lenir
(canlı doğrulama pipeline/run_graph.py ile ayrıca yapılır).

Kol 3 (contract) testleri özellikle sözleşme kapısının retry mantığını
(pipeline/graph.py'deki _route_after_validation) sınar: ilk denemede geçerli,
bir retry sonrası geçerli, ve denemeler tükenince pes etme.
"""

import json

import pytest

from agents.llm import ModelResponse
from config import MAX_PLANNER_ATTEMPTS
from eval.harness import load_all_tasks
from pipeline.graph import build_graph


def _fake_response(text: str) -> ModelResponse:
    return ModelResponse(
        text=text, model="fake", input_tokens=1, output_tokens=1,
        cost_usd=0.0, latency_s=0.0, logprobs=None, finish_reason="stop",
    )


VALID_CONTRACT_JSON = json.dumps({
    "task_id": "humaneval_000",
    "function_signature": "def has_close_elements(numbers, threshold)",
    "steps": [{"description": "sort and compare adjacent elements",
               "preconditions": [], "postconditions": ["returns True iff two elements are within threshold"]}],
    "edge_cases": ["empty list"],
})
PASSING_CODE = (
    "```python\n"
    "def has_close_elements(numbers, threshold):\n"
    "    s = sorted(numbers)\n"
    "    return any(abs(s[i+1] - s[i]) < threshold for i in range(len(s) - 1))\n"
    "```"
)


def test_naive_akisi_mock_ile_gecer(monkeypatch):
    plan_text = "Loop over all pairs and compare their absolute difference to the threshold."
    code_text = (
        "```python\n"
        "def has_close_elements(numbers, threshold):\n"
        "    for i in range(len(numbers)):\n"
        "        for j in range(len(numbers)):\n"
        "            if i != j and abs(numbers[i] - numbers[j]) < threshold:\n"
        "                return True\n"
        "    return False\n"
        "```"
    )
    monkeypatch.setattr("agents.planner.call_model", lambda *a, **k: _fake_response(plan_text))
    monkeypatch.setattr("agents.coder.call_model", lambda *a, **k: _fake_response(code_text))

    graph = build_graph("naive")
    task = load_all_tasks()[0]  # humaneval_000: has_close_elements
    config = {"configurable": {"thread_id": "test-naive"}}
    final = graph.invoke({"task": task, "mode": "naive", "attempt_count": 0}, config)

    assert final["plan"] == plan_text
    assert "def has_close_elements" in final["code"]
    assert final["test_report"]["status"] == "passed"
    senders = [m["from"] for m in final["raw_messages"]]
    assert senders == ["planner", "coder", "tester"]


def test_contract_akisi_ilk_denemede_gecerli(monkeypatch):
    monkeypatch.setattr("agents.planner.call_model", lambda *a, **k: _fake_response(VALID_CONTRACT_JSON))
    monkeypatch.setattr("agents.coder.call_model", lambda *a, **k: _fake_response(PASSING_CODE))

    graph = build_graph("contract")
    task = load_all_tasks()[0]
    config = {"configurable": {"thread_id": "test-contract-valid"}}
    final = graph.invoke({"task": task, "mode": "contract", "attempt_count": 0}, config)

    assert final["handoff_validation"]["valid"] is True
    assert final["attempt_count"] == 1
    assert final["test_report"]["status"] == "passed"
    senders = [m["from"] for m in final["raw_messages"]]
    assert senders == ["planner", "validator", "coder", "tester"]


def test_contract_task_id_prompta_gonderiliyor(monkeypatch):
    # Kritik düzeltmenin regresyon testi: task_id modele PROMPT'ta verilmezse
    # eşleşme kontrolü (agents/validator.py) her seferinde başarısız olur --
    # bu test, planner'ın gerçekten task_id'yi mesaja koyduğunu doğrular.
    captured = {}

    def fake_planner_call(messages, **kwargs):
        captured["messages"] = messages
        return _fake_response(VALID_CONTRACT_JSON)

    monkeypatch.setattr("agents.planner.call_model", fake_planner_call)
    monkeypatch.setattr("agents.coder.call_model", lambda *a, **k: _fake_response(PASSING_CODE))

    graph = build_graph("contract")
    task = load_all_tasks()[0]
    config = {"configurable": {"thread_id": "test-contract-taskid-in-prompt"}}
    graph.invoke({"task": task, "mode": "contract", "attempt_count": 0}, config)

    user_content = captured["messages"][1]["content"]
    assert task["task_id"] in user_content


def test_contract_akisi_retry_sonrasi_gecerli(monkeypatch):
    invalid_json = '{"task_id": "x"}'  # steps eksik -> sözleşme ihlali
    responses = iter([invalid_json, VALID_CONTRACT_JSON])
    monkeypatch.setattr("agents.planner.call_model", lambda *a, **k: _fake_response(next(responses)))
    monkeypatch.setattr("agents.coder.call_model", lambda *a, **k: _fake_response(PASSING_CODE))

    graph = build_graph("contract")
    task = load_all_tasks()[0]
    config = {"configurable": {"thread_id": "test-contract-retry"}}
    final = graph.invoke({"task": task, "mode": "contract", "attempt_count": 0}, config)

    assert final["handoff_validation"]["valid"] is True
    assert final["attempt_count"] == 2
    senders = [m["from"] for m in final["raw_messages"]]
    assert senders == ["planner", "validator", "planner", "validator", "coder", "tester"]


def test_contract_akisi_denemeler_tukenince_pes_eder(monkeypatch):
    always_invalid = '{"task_id": "x"}'
    monkeypatch.setattr("agents.planner.call_model", lambda *a, **k: _fake_response(always_invalid))
    monkeypatch.setattr("agents.coder.call_model", lambda *a, **k: _fake_response("```python\ndef f(): pass\n```"))

    graph = build_graph("contract")
    task = load_all_tasks()[0]
    config = {"configurable": {"thread_id": "test-contract-exhaust"}}
    final = graph.invoke({"task": task, "mode": "contract", "attempt_count": 0}, config)

    assert final["handoff_validation"]["valid"] is False
    assert final["attempt_count"] == MAX_PLANNER_ATTEMPTS
    assert final["test_report"]["status"] == "failed"  # coder tanımsız fonksiyona bakan tester'ı geçemez
    senders = [m["from"] for m in final["raw_messages"]]
    assert senders.count("planner") == MAX_PLANNER_ATTEMPTS
    assert senders[-2:] == ["coder", "tester"]  # pes edip devam ettiği doğrulanır


def test_checkpoint_gecmisi_olusur(monkeypatch):
    monkeypatch.setattr("agents.planner.call_model", lambda *a, **k: _fake_response(VALID_CONTRACT_JSON))
    monkeypatch.setattr("agents.coder.call_model", lambda *a, **k: _fake_response(PASSING_CODE))

    graph = build_graph("contract")
    config = {"configurable": {"thread_id": "test-checkpoint"}}
    task = load_all_tasks()[0]
    graph.invoke({"task": task, "mode": "contract", "attempt_count": 0}, config)
    history = list(graph.get_state_history(config))
    assert len(history) >= 5


def test_bilinmeyen_mode_reddedilir():
    with pytest.raises(ValueError):
        build_graph("yanlis")
