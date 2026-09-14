"""pipeline/run_graph.py::run_task() birim testi -- experiment/run_id/repeat
threading'inin state -> planner/coder -> call_model zincirinden GERÇEK
graph.invoke() ile geçtiğini doğrular (LLM'siz, call_model mock'lanır).
repeat=0 özellikle test edilir -- Python'da falsy, "if repeat:" ile
yanlışlıkla düşürülebilecek en yaygın/ilk değer."""

from agents.llm import ModelResponse
from eval.harness import load_all_tasks
from pipeline.graph import build_graph
from pipeline.run_graph import run_task


def _fake_response(text: str) -> ModelResponse:
    return ModelResponse(
        text=text, model="fake", input_tokens=1, output_tokens=1,
        cost_usd=0.0, latency_s=0.0, logprobs=None, finish_reason="stop",
    )


def test_experiment_run_id_repeat_sifir_dahil_threadleniyor(monkeypatch):
    captured_planner = {}
    captured_coder = {}

    def fake_planner_call(messages, **kwargs):
        captured_planner.update(kwargs)
        return _fake_response("basit bir plan")

    def fake_coder_call(messages, **kwargs):
        captured_coder.update(kwargs)
        return _fake_response(
            "```python\ndef has_close_elements(numbers, threshold):\n    return True\n```"
        )

    monkeypatch.setattr("agents.planner.call_model", fake_planner_call)
    monkeypatch.setattr("agents.coder.call_model", fake_coder_call)

    graph = build_graph("naive")
    task = load_all_tasks()[0]
    run_task(graph, task, "naive", thread_id="test-threading",
             experiment="exp1", run_id="run-xyz", repeat=0)

    assert captured_planner["experiment"] == "exp1"
    assert captured_planner["run_id"] == "run-xyz"
    assert captured_planner["repeat"] == 0
    assert captured_planner["arm"] == "naive"
    assert captured_coder["experiment"] == "exp1"
    assert captured_coder["run_id"] == "run-xyz"
    assert captured_coder["repeat"] == 0
    assert captured_coder["arm"] == "naive"


def test_repeat_ve_experiment_verilmezse_state_te_yok(monkeypatch):
    def fake_call(messages, **kwargs):
        assert kwargs.get("experiment") is None
        assert kwargs.get("repeat") is None
        return _fake_response("plan/kod")

    monkeypatch.setattr("agents.planner.call_model", fake_call)
    monkeypatch.setattr(
        "agents.coder.call_model",
        lambda messages, **k: _fake_response(
            "```python\ndef has_close_elements(numbers, threshold):\n    return True\n```"
        ),
    )
    graph = build_graph("naive")
    task = load_all_tasks()[0]
    run_task(graph, task, "naive", thread_id="test-no-threading")
