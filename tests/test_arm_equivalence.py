"""Kol denkliği testleri — deneyin İÇ GEÇERLİLİK kanıtı.

EXPERIMENT_PROTOCOL.md §3'ün iddiası: `structured_no_validation` ile `contract`
arasındaki TEK fark validator kapısı ve hata geri bildirimli sınırlı retry'dır.
Bu dosya o iddiayı prosedürel bir söz olmaktan çıkarıp TESTLE KANITLANAN bir
özelliğe dönüştürür. Buradaki bir testin kırılması, RQ3'ün ölçtüğü etkinin
artık "validator+retry" olmadığı anlamına gelir — sonuç yayımlanamaz.

Kanıtlananlar:
1. Aynı geçerli planda iki kolun coder prompt'u byte-for-byte aynı.
2. Parse edilebilen ama şema-geçersiz planda structured pydantic'i HİÇ
   çağırmaz.
3. Parse hatasında structured tek planner çağrısıyla zarfı coder'a iletir.
4. AYNI parse hatası contract'ta retry tetikler.
5. Contract denemeleri tükenirse coder yine canonical hata zarfı alır.
6. Bütün raw_messages alıcıları gerçek graf kenarlarıyla eşleşir.
"""

import json

import pytest

import agents.validator as validator_module
from agents.handoff import canonical_json, is_parse_error_envelope
from agents.llm import ModelResponse
from config import ARM_CONTRACT, ARM_NAIVE, ARM_STRUCTURED, MAX_PLANNER_ATTEMPTS
from eval.harness import load_all_tasks
from pipeline.graph import build_graph

VALID_PLAN = {
    "task_id": "humaneval_000",
    "function_signature": "def has_close_elements(numbers, threshold)",
    "steps": [{"description": "sort and compare adjacent elements",
               "preconditions": [],
               "postconditions": ["returns True iff two elements are within threshold"]}],
    "edge_cases": ["empty list"],
}
# Parse EDİLEBİLİR ama şema-geçersiz (steps eksik, extra alan var).
SCHEMA_INVALID_PLAN = {"task_id": "humaneval_000", "beklenmeyen_alan": 1}
# Hiç JSON olmayan yanıt -> parse hatası.
UNPARSEABLE_TEXT = "Bu bir plan ama JSON degil, sadece duz metin."

PASSING_CODE = (
    "```python\n"
    "def has_close_elements(numbers, threshold):\n"
    "    s = sorted(numbers)\n"
    "    return any(abs(s[i+1] - s[i]) < threshold for i in range(len(s) - 1))\n"
    "```"
)


def _fake_response(text: str) -> ModelResponse:
    return ModelResponse(text=text, model="fake", input_tokens=1, output_tokens=1,
                         cost_usd=0.0, latency_s=0.0, logprobs=None, finish_reason="stop")


def _run(monkeypatch, mode: str, planner_texts: list[str], thread_id: str):
    """Kolu mock LLM'lerle koşturur; (final_state, coder'a giden prompt'lar,
    planner çağrı sayısı) döndürür."""
    remaining = list(planner_texts)
    calls = {"planner": 0}
    coder_prompts = []

    def fake_planner(messages, **kwargs):
        calls["planner"] += 1
        # Liste tükenirse son yanıtı tekrarla (tükenme senaryosu için).
        text = remaining.pop(0) if len(remaining) > 1 else remaining[0]
        return _fake_response(text)

    def fake_coder(messages, **kwargs):
        coder_prompts.append(messages[1]["content"])
        return _fake_response(PASSING_CODE)

    monkeypatch.setattr("agents.planner.call_model", fake_planner)
    monkeypatch.setattr("agents.coder.call_model", fake_coder)

    graph = build_graph(mode)
    task = load_all_tasks()[0]  # humaneval_000
    final = graph.invoke({"task": task, "mode": mode, "attempt_count": 0},
                         {"configurable": {"thread_id": thread_id}})
    return final, coder_prompts, calls["planner"]


# --- 1. Geçerli planda iki kolun coder prompt'u AYNI ---------------------------

def test_gecerli_planda_coder_promptu_iki_kolda_ayni(monkeypatch):
    payload = json.dumps(VALID_PLAN)
    _, structured_prompts, _ = _run(monkeypatch, ARM_STRUCTURED, [payload], "eq-structured")
    _, contract_prompts, _ = _run(monkeypatch, ARM_CONTRACT, [payload], "eq-contract")

    assert structured_prompts[0] == contract_prompts[0], (
        "structured ve contract coder prompt'ları farklı -> aradaki fark artık "
        "yalnız validator+retry değil"
    )
    # Ve gerçekten canonical JSON gönderiliyor (düz metin renderer'a dönülmemiş).
    assert canonical_json(VALID_PLAN) in structured_prompts[0]


def test_coder_promptu_anahtar_sirasindan_etkilenmez(monkeypatch):
    # Aynı plan, farklı anahtar sırasıyla -> aynı coder prompt'u.
    reordered = dict(reversed(list(VALID_PLAN.items())))
    _, a, _ = _run(monkeypatch, ARM_STRUCTURED, [json.dumps(VALID_PLAN)], "ord-a")
    _, b, _ = _run(monkeypatch, ARM_STRUCTURED, [json.dumps(reordered)], "ord-b")
    assert a[0] == b[0]


# --- 2. Structured pydantic'i HİÇ çağırmaz ------------------------------------

def test_structured_sema_gecersiz_plani_pydanticsiz_iletir(monkeypatch):
    # RQ3'ün müdahalesi tam olarak budur: structured kolda sözleşme doğrulaması
    # YOKTUR. Gizlice çalıştırılırsa iki kol arasındaki fark kaybolur.
    calls = []
    original = validator_module.PlannerOutput.model_validate

    def spy(*args, **kwargs):
        calls.append(args)
        return original(*args, **kwargs)

    monkeypatch.setattr(validator_module.PlannerOutput, "model_validate", spy)

    final, coder_prompts, planner_calls = _run(
        monkeypatch, ARM_STRUCTURED, [json.dumps(SCHEMA_INVALID_PLAN)], "struct-schema-invalid")

    assert calls == [], "structured kolda pydantic doğrulaması çalıştırıldı"
    assert planner_calls == 1                      # retry yok
    assert final["handoff_parse_ok"] is True       # JSON'a dönüştü
    assert final.get("handoff_validation") is None  # doğrulama hiç yapılmadı
    assert canonical_json(SCHEMA_INVALID_PLAN) in coder_prompts[0]


def test_contract_ayni_plani_reddedip_retry_eder(monkeypatch):
    # Aynı girdi, tek fark kol -> contract'ta doğrulama çalışır ve retry olur.
    final, _, planner_calls = _run(
        monkeypatch, ARM_CONTRACT, [json.dumps(SCHEMA_INVALID_PLAN)], "contract-schema-invalid")

    assert planner_calls == MAX_PLANNER_ATTEMPTS
    assert final["handoff_validation"]["valid"] is False
    assert final["handoff_parse_ok"] is True  # parse başarılı, sözleşme başarısız


# --- 3/4. Parse hatası: structured iletir, contract retry eder ----------------

def test_structured_parse_hatasinda_tek_cagriyla_zarfi_iletir(monkeypatch):
    final, coder_prompts, planner_calls = _run(
        monkeypatch, ARM_STRUCTURED, [UNPARSEABLE_TEXT], "struct-parse-error")

    assert planner_calls == 1                   # retry YOK
    assert final["handoff_parse_ok"] is False
    assert is_parse_error_envelope(final["plan"])
    # Çıplak ham metin değil, canonical JSON zarfı gitti.
    assert canonical_json(final["plan"]) in coder_prompts[0]
    assert '"_handoff_status": "parse_error"' in coder_prompts[0]


def test_contract_ayni_parse_hatasinda_retry_eder(monkeypatch):
    final, _, planner_calls = _run(
        monkeypatch, ARM_CONTRACT, [UNPARSEABLE_TEXT], "contract-parse-error")

    assert planner_calls == MAX_PLANNER_ATTEMPTS
    assert final["handoff_parse_ok"] is False
    assert final["handoff_validation"]["valid"] is False


def test_contract_parse_hatasindan_retry_ile_kurtulabilir(monkeypatch):
    # Retry'ın FİİLEN kurtardığını gösterir: 1. deneme bozuk, 2. deneme geçerli.
    final, coder_prompts, planner_calls = _run(
        monkeypatch, ARM_CONTRACT, [UNPARSEABLE_TEXT, json.dumps(VALID_PLAN)],
        "contract-parse-recover")

    assert planner_calls == 2
    assert final["handoff_parse_ok"] is True
    assert final["handoff_validation"]["valid"] is True
    assert canonical_json(VALID_PLAN) in coder_prompts[0]


# --- 5. Denemeler tükenince coder canonical zarf alır -------------------------

def test_contract_denemeler_tukenince_coder_canonical_zarf_alir(monkeypatch):
    final, coder_prompts, planner_calls = _run(
        monkeypatch, ARM_CONTRACT, [UNPARSEABLE_TEXT], "contract-exhaust-envelope")

    assert planner_calls == MAX_PLANNER_ATTEMPTS
    # Pes etme davranışı korunuyor: coder yine de çalışıyor ve aldığı şey
    # serbest metin değil, canonical JSON zarfı.
    assert len(coder_prompts) == 1
    assert '"_handoff_status": "parse_error"' in coder_prompts[0]
    assert canonical_json(final["plan"]) in coder_prompts[0]


# --- 6. raw_messages gerçek graf kenarlarıyla eşleşiyor -----------------------

# Grafta FİİLEN var olan kenarlar (pipeline/graph.py). Bu küme dışında bir
# (from, to) çifti loglanırsa iletişim analizi olmayan bir kenarı raporlar.
GERCEK_KENARLAR = {
    ("planner", "coder"),       # naive + structured
    ("planner", "validator"),   # contract
    ("validator", "planner"),   # contract, doğrulama başarısız -> retry
    ("validator", "coder"),     # contract, doğrulama başarılı
    ("coder", "tester"),
    ("tester", "end"),
}


@pytest.mark.parametrize("mode,planner_texts", [
    (ARM_NAIVE, ["Serbest metin plan."]),
    (ARM_STRUCTURED, [json.dumps(VALID_PLAN)]),
    (ARM_CONTRACT, [json.dumps(VALID_PLAN)]),
    (ARM_CONTRACT, [UNPARSEABLE_TEXT]),  # retry kenarının da geçtiği yol
])
def test_mesaj_alicilari_gercek_graf_kenarlariyla_eslesir(monkeypatch, mode, planner_texts):
    final, _, _ = _run(monkeypatch, mode, planner_texts, f"edges-{mode}-{len(planner_texts)}")
    kenarlar = {(m["from"], m["to"]) for m in final["raw_messages"]}
    assert kenarlar <= GERCEK_KENARLAR, f"gerçekleşmeyen kenar loglandı: {kenarlar - GERCEK_KENARLAR}"


def test_tester_coder_a_geri_mesaj_gondermez(monkeypatch):
    # Regresyon: eskiden "tester -> coder" loglanıyordu ama graf tester'dan
    # END'e gidiyor; repair loop kapsam dışı (EXPERIMENT_PROTOCOL.md §12).
    final, _, _ = _run(monkeypatch, ARM_CONTRACT, [json.dumps(VALID_PLAN)], "no-tester-feedback")
    assert ("tester", "coder") not in {(m["from"], m["to"]) for m in final["raw_messages"]}


def test_structured_grafinda_validator_dugumu_yok(monkeypatch):
    final, _, _ = _run(monkeypatch, ARM_STRUCTURED, [json.dumps(VALID_PLAN)], "struct-no-validator")
    assert "validator" not in {m["from"] for m in final["raw_messages"]}
    assert [m["from"] for m in final["raw_messages"]] == ["planner", "coder", "tester"]
