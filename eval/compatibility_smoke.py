"""P1 Parça 6A — Gemini 3.5 Flash Lite ve Grok 4.3 uyumluluk smoke altyapısı.

**Bu modül tamamen OFFLINE geliştirilmiştir.** Hiçbir gerçek/ücretli API çağrısı
bu committe yapılmamıştır; bütün testler `litellm.completion`'ı mock'lar.
Gerçek çağrı yalnız kullanıcı `OPENROUTER_API_KEY` sağladığında ve açıkça
`scripts/compatibility_smoke.py run` çalıştırdığında mümkündür (P1 Parça 6B).
Bu parçanın tamamlanması, resmi Gemini/Grok uyumluluk smoke'unun
TAMAMLANDIĞI anlamına GELMEZ — yalnız altyapı hazırdır
(EXPERIMENT_PROTOCOL.md §4 uyumluluk kapısı
öncesi Grok MAST-şema/taşıma smoke'u).

Amaç
----
1. Gemini 3.5 Flash Lite ve Grok 4.3 için gerçek çağrılardan önce dondurulmuş,
   deterministik bir smoke çağrı matrisi kurmak.
2. Sonuç/provenance sözleşmesini tanımlamak (bkz. `agents/llm.py` LLM_CALL
   şeması 2.1 -- başarılı kayıt artık `response_id`/`native_finish_reason`/
   `requested_provider`/`actual_provider` taşıyor).
3. Resume ve stale-data davranışını güvenceye almak (manifest kritik alanları +
   probe log bütünlüğü).
4. GERÇEK üretim prompt ve parse yollarını kullanmak: `agents/planner.py`
   (CONTRACT_SYSTEM_PROMPT), `agents/coder.py` (SYSTEM_PROMPT),
   `agents/parsing.py` (extract_json/extract_code), `agents/contracts.py`
   (PlannerOutput), `eval/mast_labels.py` (build_evidence,
   build_adjudicator_messages, adjudicate_record), `eval/mast_schema.py`
   (MastLabel, MastAdjudication).
5. Tüm altyapıyı LiteLLM seviyesinde mock yanıtlarla uçtan uca test etmek.

Gemini matrisi (12 çağrı = 6 planner + 6 coder)
------------------------------------------------
6 pilot görev SHA-256 tabanlı deterministik seçilir (PYTHONHASHSEED'den
bağımsız). Her görev için:

* **planner probu**: `agents.planner`'ın gerçek contract/structured yolu
  (aynı CONTRACT_SYSTEM_PROMPT, aynı `response_format=json_object`, aynı
  `extract_json`); validator/retry ÇALIŞTIRILMAZ (bu bir tam kol koşusu
  değildir). `PlannerOutput` şema uyumu ayrıca kaydedilir.
* **coder probu**: `agents.coder`'ın gerçek yolu; coder'a önceden hazırlanmış
  geçerli/canonical bir `PlannerOutput` verilir. `extract_code` + `ast.parse`
  ile YALNIZ sözdizimi kontrol edilir -- kod ÇALIŞTIRILMAZ, base/plus testi
  koşulmaz, pass/fail üretilmez.

Grok matrisi (12 çağrı = 12 sentetik anlaşmazlık senaryosu)
-------------------------------------------------------------
Held-out KULLANILMAZ. Her senaryo sentetik bir başarısız result kaydı + pilot
görev metninden kurulan gerçek kanıt paketi + tam olarak İKİ dış (external)
`MastLabel` içerir (leave-self-out: self etiketi hiç oluşturulmaz, sonradan
filtrelenmez). Dağılım: 6 Gemini + 6 DeepSeek kaynaklı, hem `single_agent` hem
`multi_agent`, en az bir `insufficient_context`, en az iki farklı MAST kodu.
`eval.mast_labels.adjudicate_record()` -- GERÇEK external-only yol -- çağrılır.
"""

import ast
import hashlib
import json
import os
import platform
import re
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import litellm

from agents.coder import SYSTEM_PROMPT as CODER_SYSTEM_PROMPT
from agents.contracts import PlannerOutput, PlanStep
from agents.handoff import canonical_json
from agents.llm import ProviderResponseError, call_model
from agents.parsing import extract_code, extract_json
from agents.planner import CONTRACT_SYSTEM_PROMPT
from config import (
    ALL_ARMS,
    ARM_BASELINE,
    COMPATIBILITY_SMOKE_CALLS_PER_TARGET,
    COMPATIBILITY_SMOKE_GEMINI_TASK_COUNT,
    COMPATIBILITY_SMOKE_GROK_SCENARIO_COUNT,
    COMPATIBILITY_SMOKE_SCHEMA_VERSION,
    COMPATIBILITY_SMOKE_SELECTION_SEED,
    COMPATIBILITY_SMOKE_TARGETS,
    COMPATIBILITY_SMOKE_TASK_SET,
    DEFAULT_TEMPERATURE,
    LLM_CALL_SCHEMA_VERSION,
    LLM_MIN_INTERVAL_S,
    LLM_NUM_RETRIES,
    LLM_PROVIDER_ERROR_BACKOFF_S,
    LLM_PROVIDER_ERROR_RETRIES,
    LLM_TIMEOUT_S,
    MAST_DECISION_RULE_VERSION,
    MAST_JUDGE_TEMPERATURE,
    MAST_PANEL_HASH_VERSION,
    MAST_SCHEMA_VERSION,
    MAX_OUTPUT_TOKENS,
    MODEL_JUDGES,
    MODEL_MAIN,
    MODEL_SECONDARY,
    REASONING_CONFIG,
    ROOT,
    provider_routing_for,
)
from eval.harness import load_all_tasks, load_task
from eval.mast_labels import (
    adjudicate_record,
    build_evidence,
    evidence_digest,
    interaction_type as mast_interaction_type,
    make_judge_record,
    prompt_contract_hash,
)
from eval.mast_schema import (
    ADJUDICATOR_STATUS_OK,
    JUDGE_STATUS_OK,
    MastLabel,
    interaction_problems,
    judge_role_partition,
)

__all__ = [
    "CompatibilitySmokeError",
    "GEMINI_TARGET", "GROK_TARGET", "LOG_NAMESPACE",
    "select_pilot_task_ids", "build_gemini_plan", "build_grok_plan", "build_plan",
    "run_probe", "execute_pending", "pending_probes",
    "build_manifest_snapshot", "check_or_write_manifest",
    "verify_probe_log", "verify_call_log", "require_verified_git_state",
    "prepare", "run", "report",
    "call_log_path", "build_report",
    "GATE_PASS", "GATE_FAIL", "GATE_INCOMPLETE",
]


class CompatibilitySmokeError(RuntimeError):
    """Uyumluluk smoke güvenle çalıştırılamaz -- bozuk plan/rapor yerine durulur."""


GEMINI_TARGET = "gemini"
GROK_TARGET = "grok"
# Sıra normatiftir (COMPATIBILITY_SMOKE_TARGETS ile aynı): plan/rapor bu
# sırayla üretilir; dict insertion-order Python 3.7+'ta garantili ve
# PYTHONHASHSEED'den bağımsızdır.
TARGET_KEYS = tuple(COMPATIBILITY_SMOKE_TARGETS)

PROBE_KIND_PLANNER = "gemini_planner"
PROBE_KIND_CODER = "gemini_coder"
PROBE_KIND_ADJUDICATE = "grok_adjudicate"

STATUS_COMPLETED = "completed"
STATUS_TERMINAL_FAILURE = "terminal_failure"

# Hedef başına ön-kayıtlı uyumluluk kararı (§9 "ön-kayıtlı karar ayrımı").
# Bu bir MODEL YETENEK ölçüsü DEĞİLDİR: yalnız taşıma/provenance/şema
# uyumluluğunu ölçer.
GATE_PASS = "PASS"
GATE_FAIL = "FAIL"
GATE_INCOMPLETE = "INCOMPLETE"

# Judge/adjudicator çağrılarıyla AYNI disiplin (§9.1): formal smoke çağrıları
# ana performans/MAST çağrı loguna KARIŞMAZ.
LOG_NAMESPACE = "compat_smoke"

_CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)


# --- Deterministik seçim -----------------------------------------------------

def _sha256_hex(*parts: str) -> str:
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()


def select_pilot_task_ids(count: int, *, salt: str,
                          seed: int = COMPATIBILITY_SMOKE_SELECTION_SEED) -> list[str]:
    """SHA-256 tabanlı deterministik seçim -- PYTHONHASHSEED'DEN BAĞIMSIZ.

    EXPERIMENT_PROTOCOL.md §9'daki AYNI disiplin: Python'un `hash()`'ine,
    set/dict sırasına ya da `PYTHONHASHSEED`'e bağlanan bir sıralama süreçler
    arasında değişir ve "aynı seed aynı seçim" iddiasını çürütür. Önce görev
    kimlikleri ALFABETİK sıralanır (girdi sırasından bağımsızlık), sonra her
    biri `sha256(seed|salt|task_id)` anahtarıyla yeniden sıralanır.

    `salt`, Gemini ve Grok matrislerinin AYNI havuzdan BAĞIMSIZ (örtüşebilir)
    alt kümeler seçmesini sağlar; aynı seed'i paylaşsalar da salt farklı
    olduğu için seçimleri birbirine karışmaz.

    Yalnız PİLOT setten seçer -- `COMPATIBILITY_SMOKE_TASK_SET` sabittir ve bu
    fonksiyonun bir `task_set` parametresi YOKTUR (held-out'u sunmak dahi
    yapısal olarak imkânsız).
    """
    all_ids = sorted(t["task_id"] for t in load_all_tasks(COMPATIBILITY_SMOKE_TASK_SET))
    if len(all_ids) < count:
        raise CompatibilitySmokeError(
            f"{COMPATIBILITY_SMOKE_TASK_SET!r} görev seti {len(all_ids)} görev "
            f"içeriyor, {count} isteniyor")
    ranked = sorted(all_ids, key=lambda tid: _sha256_hex(str(seed), salt, tid))
    return sorted(ranked[:count])


# --- Gemini probe plan --------------------------------------------------------

def build_gemini_plan(task_ids: list[str]) -> list[dict]:
    """6 planner + 6 coder probu. Sıra: önce bütün planner'lar, sonra coder'lar.

    `task_ids` zaten deterministik seçilmiş VE alfabetik sıralanmıştır; burada
    ek bir rastgelelik YOKTUR.
    """
    planners = [{"probe_id": f"gemini:planner:{tid}", "target_key": GEMINI_TARGET,
                "kind": PROBE_KIND_PLANNER, "task_id": tid} for tid in task_ids]
    coders = [{"probe_id": f"gemini:coder:{tid}", "target_key": GEMINI_TARGET,
              "kind": PROBE_KIND_CODER, "task_id": tid} for tid in task_ids]
    return planners + coders


def _synthetic_planner_output(task: dict) -> PlannerOutput:
    """Coder probu için önceden hazırlanmış, GEÇERLİ, canonical bir plan.

    Yalnız görevin KENDİ alanlarından (entry_point) türer -- model çıktısına
    hiç bakmaz, deterministiktir.
    """
    return PlannerOutput(
        task_id=task["task_id"],
        function_signature=f"def {task['entry_point']}(...):",
        steps=[PlanStep(
            description="Implement the function so it satisfies the task description.",
            preconditions=[],
            postconditions=["The return value satisfies the task's documented behaviour."],
        )],
        edge_cases=[],
    )


# --- Grok probe plan -----------------------------------------------------------

# Kategori 2 (ajanlar-arası) modları tek-ajanlı (ARM_BASELINE) senaryoda
# YAPISAL OLARAK reddedilir (eval.mast_schema.interaction_problems); bu yüzden
# tek-ajanlı ve çok-ajanlı senaryolar AYRI kod havuzu kullanır.
_LABEL_CODES_SINGLE = ("1.1", "3.2")   # kategori 1/3 -- tek-ajanlıda geçerli
_LABEL_CODES_MULTI = ("2.3", "1.1")    # kategori 2 -- yalnız çok-ajanlıda geçerli


def _scenario_labels(index: int, interaction_type: str) -> tuple[dict, dict]:
    """İki FARKLI external `MastLabel` taslağı (split senaryosu, judge_model hariç).

    `index == 0`: bir taraf `insufficient_context=True`
    ("En az bir örnekte dış judge'lardan biri insufficient_context olmalı").
    Diğerleri: interaction_type'a uygun iki farklı MAST kodu.
    """
    if index == 0:
        return (
            {"insufficient_context": True, "confidence": "low",
             "rationale": "kanit yetersiz (compat-smoke sentetik senaryo)"},
            {"primary_mode": (_LABEL_CODES_SINGLE[0] if interaction_type == "single_agent"
                              else _LABEL_CODES_MULTI[0]),
             "secondary_modes": [], "confidence": "medium",
             "rationale": "b tarafi normal etiket (compat-smoke sentetik senaryo)"},
        )
    codes = (_LABEL_CODES_SINGLE if interaction_type == "single_agent" else _LABEL_CODES_MULTI)
    return (
        {"primary_mode": codes[0], "secondary_modes": [], "confidence": "high",
         "rationale": "a tarafi compat-smoke sentetik senaryo"},
        {"primary_mode": codes[1], "secondary_modes": [], "confidence": "low",
         "rationale": "b tarafi compat-smoke sentetik senaryo"},
    )


def build_grok_plan(task_ids: list[str]) -> list[dict]:
    """12 sentetik anlaşmazlık senaryosu: 6 Gemini + 6 DeepSeek kaynaklı.

    Held-out KULLANILMAZ: `task_id` yalnız pilot metninden kanıt kurmaya
    yarar; senaryonun kendisi sentetik bir başarısız result kaydıdır, gerçek
    bir koşu DEĞİLDİR. Dış judge çifti `judge_role_partition()` (gerçek
    leave-self-out yardımcısı) İLE plan zamanında belirlenir -- çalışma
    zamanında yeniden hesaplanınca AYNI sonucu verir (saf/deterministik
    fonksiyon), yani plan ile çalıştırma arasında sessiz bir ayrışma riski
    yoktur.
    """
    if not task_ids:
        raise CompatibilitySmokeError("grok senaryoları için en az 1 pilot görev gerekir")
    plan = []
    for i in range(COMPATIBILITY_SMOKE_GROK_SCENARIO_COUNT):
        source_model = MODEL_MAIN if i % 2 == 0 else MODEL_SECONDARY
        arm = ALL_ARMS[i % len(ALL_ARMS)]
        itype = "single_agent" if arm == ARM_BASELINE else "multi_agent"
        task_id = task_ids[i % len(task_ids)]
        label_a, label_b = _scenario_labels(i, itype)
        self_judge, external = judge_role_partition(source_model, MODEL_JUDGES)
        plan.append({
            "probe_id": f"grok:adjudicate:{i:02d}",
            "target_key": GROK_TARGET,
            "kind": PROBE_KIND_ADJUDICATE,
            "index": i,
            "source_model": source_model,
            "self_judge_model": self_judge,
            "external_judges": list(external),
            "arm": arm,
            "interaction_type": itype,
            "task_id": task_id,
            "external_labels": [
                {"judge_model": external[0], **label_a},
                {"judge_model": external[1], **label_b},
            ],
        })
    return plan


def build_plan(*, gemini_task_ids: list[str], grok_task_ids: list[str]) -> list[dict]:
    """Tam sıralı plan: ÖNCE gemini, SONRA grok (COMPATIBILITY_SMOKE_TARGETS sırası)."""
    return build_gemini_plan(gemini_task_ids) + build_grok_plan(grok_task_ids)


# --- Gemini probları (gerçek planner/coder yolu) ------------------------------

def run_gemini_planner_probe(spec: dict, *, model: str, experiment: str,
                             log_namespace: str) -> dict:
    """`agents/planner.py`'nin GERÇEK contract/structured yolu -- retry YOK.

    Gerçek `CONTRACT_SYSTEM_PROMPT`, gerçek `response_format=json_object`,
    gerçek `extract_json` + `PlannerOutput` doğrulaması kullanılır.
    Validator/retry çalıştırılmaz: bu tam bir kol koşusu DEĞİLDİR.
    """
    task = load_task(spec["task_id"], COMPATIBILITY_SMOKE_TASK_SET)
    user_content = f"Task ID: {task['task_id']}\n\n{task['prompt']}"
    response = call_model(
        [{"role": "system", "content": CONTRACT_SYSTEM_PROMPT},
         {"role": "user", "content": user_content}],
        model=model, temperature=DEFAULT_TEMPERATURE,
        response_format={"type": "json_object"},
        task_id=task["task_id"], agent_role="compat_smoke_planner",
        experiment=experiment, run_id=spec["probe_id"], arm="compat_smoke",
        repeat=0, agent_attempt=1, log_namespace=log_namespace,
    )
    parse_ok = False
    schema_ok = False
    parsed = None
    try:
        parsed = extract_json(response.text)
        parse_ok = isinstance(parsed, dict)
    except ValueError:
        parse_ok = False
    if parse_ok:
        try:
            PlannerOutput.model_validate(parsed)
            schema_ok = True
        except Exception:
            schema_ok = False
    return {"transport_ok": True, "planner_json_parse_ok": parse_ok,
            "planner_schema_ok": schema_ok}


def run_gemini_coder_probe(spec: dict, *, model: str, experiment: str,
                           log_namespace: str) -> dict:
    """`agents/coder.py`'nin GERÇEK yolu; `PlannerOutput` önceden hazırlanmış.

    `extract_code` + `ast.parse` ile YALNIZ Python sözdizimi kontrol edilir.
    Kod ÇALIŞTIRILMAZ, base/plus testleri koşulmaz, pass/fail üretilmez.
    """
    task = load_task(spec["task_id"], COMPATIBILITY_SMOKE_TASK_SET)
    plan_text = canonical_json(_synthetic_planner_output(task).model_dump())
    user_content = (
        f"Task:\n{task['prompt']}\n\nPlan from your teammate:\n{plan_text}\n\n"
        "Implement this as Python code."
    )
    response = call_model(
        [{"role": "system", "content": CODER_SYSTEM_PROMPT},
         {"role": "user", "content": user_content}],
        model=model, temperature=DEFAULT_TEMPERATURE,
        task_id=task["task_id"], agent_role="compat_smoke_coder",
        experiment=experiment, run_id=spec["probe_id"], arm="compat_smoke",
        repeat=0, agent_attempt=1, log_namespace=log_namespace,
    )
    code = extract_code(response.text)
    nonempty = bool(code.strip())
    extract_ok = bool(_CODE_BLOCK_RE.search(response.text))
    ast_ok = False
    if nonempty:
        try:
            ast.parse(code)
            ast_ok = True
        except SyntaxError:
            ast_ok = False
    return {"transport_ok": True, "coder_nonempty": nonempty,
            "coder_extract_ok": extract_ok, "coder_ast_parse_ok": ast_ok}


# --- Grok probu (gerçek external-only MAST adjudication yolu) -----------------

def _synthetic_result_record(spec: dict) -> dict:
    """Kanıt kurmaya yeten sentetik, BAŞARISIZ bir result kaydı.

    Gerçek bir koşu DEĞİLDİR -- sandbox'ta çalıştırılmaz, pass/fail üretmez.
    `run_id` doğrudan `probe_id`'dir: adjudication kaydının `source_run_id`'i
    ve resume kimliği bu yüzden probe kimliğiyle birebir örtüşür.
    """
    return {
        "run_id": spec["probe_id"],
        "model": spec["source_model"],
        "task_set": COMPATIBILITY_SMOKE_TASK_SET,
        "task_id": spec["task_id"],
        "arm": spec["arm"],
        "repeat": 0,
        "plan": {"task_id": spec["task_id"], "note": "compat-smoke sentetik plan"},
        "raw_messages": [{"from": "planner", "to": "coder",
                          "content": "compat-smoke sentetik mesaj"}],
        "code": "def f():\n    return None\n",
        "base_status": "failed",
        "plus_status": "failed",
        "error_class": "assertion",
        "traceback": "AssertionError (compat-smoke sentetik traceback)",
    }


def build_grok_external_records(spec: dict, evidence: dict, *, experiment: str) -> list[dict]:
    """Gerçek `make_judge_record()` ile İKİ dış etiket -- self ASLA kurulmaz.

    Üçüncü (self) etiketi oluşturup sonra filtrelemek yerine, çağrı girdisi
    baştan yalnız iki external kayıtla kurulur.
    """
    record = _synthetic_result_record(spec)
    digest = evidence_digest(evidence)
    prompt_hash = prompt_contract_hash()
    out = []
    for label_spec in spec["external_labels"]:
        judge_model = label_spec["judge_model"]
        label = MastLabel.model_validate(
            {k: v for k, v in label_spec.items() if k != "judge_model"})
        out.append(make_judge_record(
            record, experiment=experiment, evidence_sha256=digest, judge_model=judge_model,
            judge_attempt=1, judge_status=JUDGE_STATUS_OK, prompt_hash=prompt_hash,
            interaction_type=spec["interaction_type"], label=label))
    return out


def run_grok_probe(spec: dict, *, model: str, experiment: str, log_namespace: str) -> dict:
    """`eval/mast_labels.py`'nin GERÇEK external-only adjudication yolu.

    `build_evidence()` + leave-self-out dış ikilisi (plandan, `judge_role_
    partition` ile) + `build_adjudicator_messages()` (adjudicate_record
    İÇİNDE) + `adjudicate_record()` + `MastLabel`/`MastAdjudication`
    sözleşmeleri.

    Grok'un kararının SEMANTİK doğruluğu puanlanmaz -- yalnız taşıma, JSON
    parse, MastLabel, interaction invariantı ve tam MastAdjudication
    sözleşmesi ölçülür.
    """
    record = _synthetic_result_record(spec)
    task = load_task(spec["task_id"], COMPATIBILITY_SMOKE_TASK_SET)
    evidence = build_evidence(record, task)
    external_records = build_grok_external_records(spec, evidence, experiment=experiment)

    adjudication = adjudicate_record(
        record, evidence, external_records, experiment=experiment, model=model,
        expected_judges=MODEL_JUDGES, log_namespace=log_namespace)

    # `adjudicator_raw` yalnız call_model BAŞARILI olduysa mevcuttur (bkz.
    # eval/mast_labels.py::adjudicate_record). Yoksa taşıma katmanı hiç yanıt
    # üretmemiştir -- bu bir MODEL bulgusu değil, terminal taşıma hatasıdır.
    transport_ok = "adjudicator_raw" in adjudication
    if not transport_ok:
        return {
            "transport_ok": False, "json_parse_ok": False, "mast_label_schema_ok": False,
            "interaction_invariant_ok": False, "mast_adjudication_schema_ok": False,
            "adjudicator_status": adjudication.get("adjudicator_status"),
            "adjudicator_error": adjudication.get("adjudicator_error"),
        }

    raw_text = adjudication["adjudicator_raw"]
    json_parse_ok = False
    mast_label_schema_ok = False
    interaction_invariant_ok = False
    payload = None
    try:
        payload = extract_json(raw_text)
        json_parse_ok = True
    except ValueError:
        payload = None
    label = None
    if json_parse_ok:
        try:
            label = MastLabel.model_validate(payload)
            mast_label_schema_ok = True
        except Exception:
            label = None
    if label is not None:
        problems = interaction_problems(label, spec["interaction_type"])
        interaction_invariant_ok = not problems
    return {
        "transport_ok": True,
        "json_parse_ok": json_parse_ok,
        "mast_label_schema_ok": mast_label_schema_ok,
        "interaction_invariant_ok": interaction_invariant_ok,
        "mast_adjudication_schema_ok": adjudication.get("adjudicator_status") == ADJUDICATOR_STATUS_OK,
        "adjudicator_status": adjudication.get("adjudicator_status"),
        "adjudicator_error": adjudication.get("adjudicator_error"),
    }


def run_probe(spec: dict, *, experiment: str, log_namespace: str = LOG_NAMESPACE) -> dict:
    """Probu türüne göre GERÇEK üretim yoluna yönlendirir."""
    model = COMPATIBILITY_SMOKE_TARGETS[spec["target_key"]]
    kind = spec["kind"]
    if kind == PROBE_KIND_PLANNER:
        return run_gemini_planner_probe(spec, model=model, experiment=experiment,
                                        log_namespace=log_namespace)
    if kind == PROBE_KIND_CODER:
        return run_gemini_coder_probe(spec, model=model, experiment=experiment,
                                      log_namespace=log_namespace)
    if kind == PROBE_KIND_ADJUDICATE:
        return run_grok_probe(spec, model=model, experiment=experiment,
                              log_namespace=log_namespace)
    raise CompatibilitySmokeError(f"bilinmeyen probe türü: {kind!r}")


# --- Manifest / fingerprint ----------------------------------------------------

def _gemini_recipe_fingerprint(model: str) -> str:
    """Gemini tarafının 'tarifi' -- prompt/şema/model/routing/parametre.

    Manifestin kritik alanıdır: herhangi biri değişirse aynı `--name` altında
    devam etmek, farklı bir tarifle üretilmiş probları aynı turda karıştırır.
    """
    parts = [
        CONTRACT_SYSTEM_PROMPT, CODER_SYSTEM_PROMPT,
        json.dumps(PlannerOutput.model_json_schema(), sort_keys=True),
        model, json.dumps(provider_routing_for(model), sort_keys=True),
        json.dumps(REASONING_CONFIG, sort_keys=True),
        str(MAX_OUTPUT_TOKENS), str(DEFAULT_TEMPERATURE),
    ]
    return hashlib.sha256("\n---\n".join(parts).encode("utf-8")).hexdigest()


def _grok_recipe_fingerprint(model: str) -> str:
    parts = [
        prompt_contract_hash(), MAST_SCHEMA_VERSION, MAST_DECISION_RULE_VERSION,
        MAST_PANEL_HASH_VERSION, model, json.dumps(provider_routing_for(model), sort_keys=True),
        json.dumps(REASONING_CONFIG, sort_keys=True),
        str(MAX_OUTPUT_TOKENS), str(MAST_JUDGE_TEMPERATURE),
    ]
    return hashlib.sha256("\n---\n".join(parts).encode("utf-8")).hexdigest()


def _plan_fingerprint(plan: list[dict]) -> str:
    payload = json.dumps(plan, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _git_commit() -> str | None:
    """HEAD commit'i; git'e erişilemiyorsa None (= DOĞRULANAMIYOR).

    None bir "sorun yok" sinyali DEĞİLDİR: `require_verified_git_state()` bunu
    ücretli `run` aşamasında fail-closed olarak yorumlar.
    """
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                           text=True, timeout=10, cwd=ROOT)
        return (r.stdout.strip() or None) if r.returncode == 0 else None
    except Exception:
        return None


def _git_dirty() -> bool | None:
    """True=kirli, False=temiz, None=DOĞRULANAMIYOR (git yok/hata).

    None ile False AYNI ŞEY DEĞİLDİR: doğrulanamayan bir ağaçta ücretli koşu
    başlatmak, sonradan "bu çıktı hangi kaynak koddan üretildi" sorusunu
    cevapsız bırakır -- bu yüzden `require_verified_git_state()` None'ı da
    durdurur.
    """
    try:
        r = subprocess.run(["git", "status", "--porcelain"], capture_output=True,
                           text=True, timeout=10, cwd=ROOT)
        return bool(r.stdout.strip()) if r.returncode == 0 else None
    except Exception:
        return None


# Manifest değişirse aynı `--name` altına devam EDİLEMEZ -- yeni isim gerekir
# (eval/runner.py ve eval/mast_labels.py'deki AYNI desen). `created_ts`/
# `python_version`/`platform`/`litellm_version` bilinçli olarak DIŞARIDA:
# gözlemsel provenance alanlarıdır, kritik değildir.
#
# `git_commit` ise KRİTİKTİR: prompt/parse/şema yolları bu depodaki koddan
# gelir, dolayısıyla aynı `--name` altında farklı bir committe devam etmek tek
# bir smoke turunda İKİ FARKLI kaynak koddan üretilmiş probları karıştırır.
# Farklı committe koşmak için yeni bir `--name` gerekir.
MANIFEST_CRITICAL_FIELDS = (
    "compatibility_smoke_schema_version", "task_set", "selection_seed",
    "gemini_task_ids", "grok_task_ids", "targets", "calls_per_target",
    "planned_probe_ids", "plan_fingerprint",
    "gemini_temperature", "grok_temperature", "max_tokens", "reasoning_config",
    "gemini_provider_routing", "grok_provider_routing",
    "llm_num_retries", "llm_min_interval_s", "llm_timeout_s",
    "llm_provider_error_retries", "llm_provider_error_backoff_s",
    "llm_call_schema_version",
    "gemini_recipe_fingerprint", "grok_recipe_fingerprint",
    "mast_prompt_hash", "mast_schema_version", "mast_decision_rule_version",
    "mast_panel_hash_version", "git_commit",
)


def build_manifest_snapshot(gemini_task_ids: list[str], grok_task_ids: list[str]) -> dict:
    plan = build_plan(gemini_task_ids=gemini_task_ids, grok_task_ids=grok_task_ids)
    gemini_model = COMPATIBILITY_SMOKE_TARGETS[GEMINI_TARGET]
    grok_model = COMPATIBILITY_SMOKE_TARGETS[GROK_TARGET]
    return {
        "compatibility_smoke_schema_version": COMPATIBILITY_SMOKE_SCHEMA_VERSION,
        "created_ts": datetime.now(timezone.utc).isoformat(),
        "task_set": COMPATIBILITY_SMOKE_TASK_SET,
        "selection_seed": COMPATIBILITY_SMOKE_SELECTION_SEED,
        "gemini_task_ids": list(gemini_task_ids),
        "grok_task_ids": list(grok_task_ids),
        "targets": dict(COMPATIBILITY_SMOKE_TARGETS),
        "calls_per_target": COMPATIBILITY_SMOKE_CALLS_PER_TARGET,
        "planned_probe_ids": [p["probe_id"] for p in plan],
        "plan_fingerprint": _plan_fingerprint(plan),
        "gemini_temperature": DEFAULT_TEMPERATURE,
        "grok_temperature": MAST_JUDGE_TEMPERATURE,
        "max_tokens": MAX_OUTPUT_TOKENS,
        "reasoning_config": REASONING_CONFIG,
        "gemini_provider_routing": provider_routing_for(gemini_model),
        "grok_provider_routing": provider_routing_for(grok_model),
        "llm_num_retries": LLM_NUM_RETRIES,
        "llm_min_interval_s": LLM_MIN_INTERVAL_S,
        "llm_timeout_s": LLM_TIMEOUT_S,
        "llm_provider_error_retries": LLM_PROVIDER_ERROR_RETRIES,
        "llm_provider_error_backoff_s": LLM_PROVIDER_ERROR_BACKOFF_S,
        "llm_call_schema_version": LLM_CALL_SCHEMA_VERSION,
        "gemini_recipe_fingerprint": _gemini_recipe_fingerprint(gemini_model),
        "grok_recipe_fingerprint": _grok_recipe_fingerprint(grok_model),
        "mast_prompt_hash": prompt_contract_hash(),
        "mast_schema_version": MAST_SCHEMA_VERSION,
        "mast_decision_rule_version": MAST_DECISION_RULE_VERSION,
        "mast_panel_hash_version": MAST_PANEL_HASH_VERSION,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "litellm_version": getattr(litellm, "__version__", None),
        "git_commit": _git_commit(),
    }


def check_or_write_manifest(path: Path, snapshot: dict) -> dict:
    """Manifest yoksa yazar; varsa KRİTİK alanların birebir aynı olduğunu doğrular.

    Farklı bir kritik alanla aynı isme devam edilemez -- yeni bir isim
    (`--name`) gerekir (eval/runner.py ve eval/mast_labels.py'deki AYNI desen).
    """
    missing = [k for k in MANIFEST_CRITICAL_FIELDS if k not in snapshot]
    if missing:
        raise KeyError(f"manifest anlık görüntüsünde kritik alan(lar) eksik: {missing}")
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        mismatches = {k: (existing.get(k), snapshot[k]) for k in MANIFEST_CRITICAL_FIELDS
                     if existing.get(k) != snapshot[k]}
        if mismatches:
            raise CompatibilitySmokeError(
                "manifest uyuşmazlığı (aynı isimle farklı yapılandırmaya devam "
                f"edilemez, yeni isim gerekir): {mismatches}")
        return existing
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(snapshot, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return snapshot


# --- Probe log: resume / bütünlük ---------------------------------------------

def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in
            path.read_text(encoding="utf-8").splitlines() if line.strip()]


def append_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def verify_probe_log(existing: list[dict], plan: list[dict]) -> None:
    """İLK API çağrısından ÖNCE: probe logunun TAM PLANA uygunluğunu doğrular.

    Yalnız `probe_id`'nin planda bulunması YETMEZ -- bir satır plandakiyle aynı
    kimliği taşıdığını iddia edip farklı bir hedefe/prob türüne ait olabilir
    (ör. elle düzenlenmiş ya da eski bir tarifle üretilmiş dosya). Böyle bir
    satır sessizce "tamamlandı" sayılırsa, hiç yapılmamış bir çağrı yapılmış
    gibi görünür. Bu yüzden her satırda:

    * `probe_id` planda OLMALI,
    * `target_key` ve `kind` plandakiyle BİREBİR eşleşmeli,
    * `status` bilinen iki değerden biri olmalı,
    * `attempt` pozitif bir tamsayı olmalı (bool kabul edilmez),
    * aynı probe için EN FAZLA BİR `completed` kayıt bulunmalı.

    Terminal hatalar (`STATUS_TERMINAL_FAILURE`) tamamlanmış SAYILMAZ --
    yeniden denenebilirler (eval/result_schema.py'deki `run_error` disipliniyle
    aynı: bir koşunun hata alması dosyayı bozuk yapmaz, yalnız İKİ başarılı
    kayıt veya beklenmeyen bir kimlik bozuktur).
    """
    by_id = {p["probe_id"]: p for p in plan}
    completed_counts: dict[str, int] = {}
    for satir_no, row in enumerate(existing, start=1):
        pid = row.get("probe_id")
        spec = by_id.get(pid)
        if spec is None:
            raise CompatibilitySmokeError(
                f"probes.jsonl beklenmeyen probe_id içeriyor (satır {satir_no}): "
                f"{pid!r} -- güncel plana ait değil (stale çıktı dizini / farklı "
                "manifest?)")
        for alan in ("target_key", "kind"):
            if row.get(alan) != spec[alan]:
                raise CompatibilitySmokeError(
                    f"probes.jsonl satır {satir_no} ({pid!r}) planla eşleşmiyor: "
                    f"{alan}={row.get(alan)!r}, planda {spec[alan]!r} -- kayıt "
                    "başka bir tarifle/plana ait, aynı isimle devam edilemez")
        status = row.get("status")
        if status not in (STATUS_COMPLETED, STATUS_TERMINAL_FAILURE):
            raise CompatibilitySmokeError(
                f"probes.jsonl satır {satir_no} ({pid!r}) bilinmeyen status "
                f"içeriyor: {status!r} -- beklenen "
                f"{STATUS_COMPLETED!r} veya {STATUS_TERMINAL_FAILURE!r}")
        attempt = row.get("attempt")
        if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 1:
            raise CompatibilitySmokeError(
                f"probes.jsonl satır {satir_no} ({pid!r}) geçersiz attempt "
                f"içeriyor: {attempt!r} -- pozitif tamsayı olmalı")
        if status == STATUS_COMPLETED:
            completed_counts[pid] = completed_counts.get(pid, 0) + 1
    duplicated = {pid: n for pid, n in completed_counts.items() if n > 1}
    if duplicated:
        raise CompatibilitySmokeError(
            f"aynı probe_id için birden fazla TAMAMLANMIŞ kayıt: {duplicated} -- "
            "hangisinin geçerli olduğu belirsiz kalırdı")


def verify_call_log(call_rows: list[dict], probe_rows: list[dict],
                    planned_probe_ids: list[str]) -> None:
    """Rapor üretmeden ÖNCE probe--çağrı eşleşmesini doğrular.

    Rapor, probe sınıflandırmasını (probes.jsonl) çağrı provenance'ıyla
    (llm_calls.jsonl) BİRLEŞTİRİR; iki dosya birbirini tutmuyorsa üretilen
    rapor sessizce yanlış olur. Üç şart:

    * çağrı logundaki her `run_id` PLANLI bir probe olmalı (yabancı run_id =
      başka bir turun logu bu dizine karışmış),
    * bir probe için EN FAZLA BİR `status="ok"` çağrı olmalı (iki başarılı
      çağrı hangisinin rapora girdiğini belirsiz bırakır),
    * `completed` işaretli her probe'un EN AZ BİR `status="ok"` çağrısı olmalı
      (aksi halde "tamamlandı" iddiası hiçbir çağrıya dayanmıyor).
    """
    planned = set(planned_probe_ids)
    ok_counts: Counter = Counter()
    for satir_no, row in enumerate(call_rows, start=1):
        run_id = row.get("run_id")
        if run_id not in planned:
            raise CompatibilitySmokeError(
                f"çağrı logu satır {satir_no}: bilinmeyen run_id {run_id!r} -- "
                "güncel plana ait değil (başka bir turun logu mu karıştı?)")
        if row.get("status") == "ok":
            ok_counts[run_id] += 1
    duplicated = {pid: n for pid, n in ok_counts.items() if n > 1}
    if duplicated:
        raise CompatibilitySmokeError(
            f"aynı probe için birden fazla BAŞARILI (status=ok) çağrı: {duplicated} "
            "-- hangisinin rapora girdiği belirsiz kalırdı")
    eksik = sorted({r["probe_id"] for r in probe_rows
                    if r.get("status") == STATUS_COMPLETED and not ok_counts.get(r["probe_id"])})
    if eksik:
        raise CompatibilitySmokeError(
            f"tamamlanmış sayılan ama başarılı çağrı kaydı OLMAYAN probe(lar): "
            f"{eksik} -- probes.jsonl ile çağrı logu tutarsız")


def completed_probe_ids(existing: list[dict]) -> set[str]:
    return {r["probe_id"] for r in existing if r.get("status") == STATUS_COMPLETED}


def next_attempt(existing: list[dict], probe_id: str) -> int:
    attempts = [r.get("attempt", 0) for r in existing if r.get("probe_id") == probe_id]
    return (max(attempts) if attempts else 0) + 1


def pending_probes(plan: list[dict], existing: list[dict]) -> list[dict]:
    """Güncel başarılı probe'lar ÇIKARILIR -- yeniden çağrılmazlar."""
    done = completed_probe_ids(existing)
    return [p for p in plan if p["probe_id"] not in done]


def execute_pending(plan: list[dict], existing: list[dict], *, experiment: str,
                    log_namespace: str = LOG_NAMESPACE, on_result=None) -> list[dict]:
    """Bekleyen probları çalıştırır; her sonuç `on_result` ile anında yazılabilir.

    Bir probun (taşıma) istisnası diğerlerini durdurmaz -- terminal hata olarak
    kaydedilir ve aynı manifest altında yeniden denenebilir kalır.
    """
    pending = pending_probes(plan, existing)
    out: list[dict] = []
    for spec in pending:
        attempt = next_attempt(existing + out, spec["probe_id"])
        row = {"probe_id": spec["probe_id"], "target_key": spec["target_key"],
               "kind": spec["kind"], "attempt": attempt,
               "ts": datetime.now(timezone.utc).isoformat()}
        try:
            classification = run_probe(spec, experiment=experiment, log_namespace=log_namespace)
        except ProviderResponseError as e:
            row["status"] = STATUS_TERMINAL_FAILURE
            row["error"] = f"ProviderResponseError: {e}"
        except Exception as e:  # noqa: BLE001 -- kasıtlı: bir probun hatası turu durdurmasın
            row["status"] = STATUS_TERMINAL_FAILURE
            row["error"] = f"{type(e).__name__}: {e}"
        else:
            row["status"] = (STATUS_TERMINAL_FAILURE if classification.get("transport_ok") is False
                             else STATUS_COMPLETED)
            row.update(classification)
        out.append(row)
        if on_result:
            on_result(row)
    return out


# --- Rapor ----------------------------------------------------------------

def call_log_path(out_dir: Path, log_namespace: str = LOG_NAMESPACE) -> Path:
    return out_dir / log_namespace / "llm_calls.jsonl"


def _target_call_metrics(call_rows: list[dict], run_ids: set[str]) -> dict:
    """`scripts/provider_health.py::summarize()` ile AYNI üç metrik ayrımı,
    run_id (=probe_id) bazında.

    - `completed_logical_calls`: en az bir `status="ok"` denemesi olan probe.
    - `embedded_error_attempt_count`: gözlenen HTTP-200-içi hata yanıtı SAYISI
      (mantıksal çağrıdan AYRI kavram -- bir çağrı birden çok bozuk deneme
      görebilir).
    - `terminal_failure_count`: hiç `ok` denemesi OLMAYAN (yalnız `error`/
      `provider_error` denemeleri bulunan) probe sayısı.
    """
    rows = [r for r in call_rows if r.get("run_id") in run_ids]
    by_run: dict[str, list[dict]] = {}
    for r in rows:
        by_run.setdefault(r["run_id"], []).append(r)

    completed = incidents = embedded_errors = terminal = recovered = 0
    missing_usage = empty_output = truncated = 0
    total_input = total_output = total_reasoning = total_cached = 0
    total_cost = 0.0
    latencies: list[float] = []
    finish_reasons: Counter = Counter()
    native_finish_reasons: Counter = Counter()
    requested_models: Counter = Counter()
    actual_models: Counter = Counter()
    requested_providers: Counter = Counter()
    actual_providers: Counter = Counter()

    for attempts in by_run.values():
        ok_rows = [a for a in attempts if a.get("status") == "ok"]
        bad_rows = [a for a in attempts if a.get("status") in ("provider_error", "error")]
        embedded_errors += sum(1 for a in attempts if a.get("status") == "provider_error")
        if ok_rows:
            completed += 1
            final = ok_rows[-1]
            if final.get("provider_attempt", 1) > 1:
                incidents += 1
                # Retry ile KURTARILMIŞ provider hatası: raporlanır ama tek
                # başına uyumluluk kararını FAIL yapmaz (§9) -- oran ölçümü
                # 3 × 100 sağlık kapısının işidir.
                recovered += 1
            finish_reasons[final.get("finish_reason")] += 1
            native_finish_reasons[final.get("native_finish_reason")] += 1
            requested_models[final.get("model")] += 1
            actual_models[final.get("actual_model")] += 1
            requested_providers[json.dumps(final.get("requested_provider"))] += 1
            actual_providers[final.get("actual_provider")] += 1
            if not final.get("input_tokens") and not final.get("output_tokens"):
                missing_usage += 1
            if not (final.get("response_text") or "").strip():
                empty_output += 1
            if final.get("truncated_by_max_tokens"):
                truncated += 1
            total_input += final.get("input_tokens") or 0
            total_output += final.get("output_tokens") or 0
            total_reasoning += final.get("reasoning_tokens") or 0
            total_cached += final.get("cached_tokens") or 0
            total_cost += final.get("cost_usd") or 0
            if final.get("latency_s") is not None:
                latencies.append(final["latency_s"])
        elif bad_rows:
            terminal += 1
            incidents += 1

    return {
        "planned_logical_calls": len(run_ids),
        "completed_logical_calls": completed,
        "logical_call_incident_count": incidents,
        "embedded_error_attempt_count": embedded_errors,
        "provider_error_recovered_call_count": recovered,
        "terminal_failure_count": terminal,
        "requested_model_distribution": dict(requested_models),
        "actual_model_distribution": dict(actual_models),
        "requested_provider_distribution": dict(requested_providers),
        "actual_provider_distribution": dict(actual_providers),
        "finish_reason_distribution": dict(finish_reasons),
        "native_finish_reason_distribution": dict(native_finish_reasons),
        "missing_usage_count": missing_usage,
        "empty_output_count": empty_output,
        "truncated_by_max_tokens_count": truncated,
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "total_reasoning_tokens": total_reasoning,
        "total_cached_tokens": total_cached,
        "total_cost_usd": round(total_cost, 6),
        "mean_latency_s": round(sum(latencies) / len(latencies), 3) if latencies else None,
    }


def _final_rows(probe_rows: list[dict]) -> dict[str, dict]:
    """probe_id -> en SON/en anlamlı durumu yansıtan satır.

    Tamamlanmış (`STATUS_COMPLETED`) bir satır varsa -- `verify_probe_log` en
    fazla BİR tane olmasını zaten şart koşar -- o kullanılır; yoksa en yüksek
    `attempt` numaralı terminal-hata satırı.
    """
    best: dict[str, dict] = {}
    for row in probe_rows:
        pid = row["probe_id"]
        current = best.get(pid)
        if current is None or row.get("status") == STATUS_COMPLETED:
            best[pid] = row
        elif current.get("status") != STATUS_COMPLETED and row.get("attempt", 0) >= current.get("attempt", 0):
            best[pid] = row
    return best


_GROK_CONTRACT_FLAGS = ("json_parse_ok", "mast_label_schema_ok",
                        "interaction_invariant_ok", "mast_adjudication_schema_ok")


def _blocker_reasons(target_key: str, metrics: dict, completed_rows: list[dict]) -> list[str]:
    """§9'daki TEKNİK uyumluluk engelleri -- yetenek/başarı ölçüsü DEĞİL.

    Bilinçli olarak DIŞARIDA bırakılanlar:

    * retry ile kurtarılmış `provider_error` (yalnız raporlanır; oran ölçümü
      3 × 100 sağlık kapısının işidir),
    * Gemini `planner_schema_ok` ve `coder_ast_parse_ok` (TANISAL kalır --
      bunlara bakıp prompt/model ayarlamak, ön-kayıtlı tasarımı model
      çıktısına göre eğmek olurdu),
    * pozitif reasoning tokenı. [2026-07-30'a kadar bu bir ENGELDİ: ayar
      kapalıyken pozitif token, ayarın uygulanmadığı anlamına gelirdi.
      `REASONING_CONFIG` artık AÇIK olduğu için (Gemini endpoint'i zorunlu
      kılıyor) pozitif token BEKLENEN durumdur; sayaç raporda
      `total_reasoning_tokens` olarak kalır çünkü maliyet/gecikme etkisi hâlâ
      izlenmelidir.]

    Paydalar TAMAMLANMIŞ problara bağlıdır: hiç koşulmamış bir probe eksiklik
    (INCOMPLETE) sebebidir, teknik uyumsuzluk (FAIL) değil.
    """
    reasons: list[str] = []
    if metrics["terminal_failure_count"]:
        reasons.append(
            f"terminal taşıma hatası: {metrics['terminal_failure_count']} mantıksal "
            "çağrı hiç başarılı yanıt alamadı")
    if metrics["missing_usage_count"]:
        reasons.append(f"eksik/sıfır usage: {metrics['missing_usage_count']} çağrı")
    if metrics["empty_output_count"]:
        reasons.append(f"boş çıktı: {metrics['empty_output_count']} çağrı")
    native_error = metrics["native_finish_reason_distribution"].get("error", 0)
    if native_error:
        reasons.append(f"native_finish_reason='error': {native_error} çağrı")
    # Ortak ayar KAPALIYSA pozitif token, ayarın fiilen uygulanmadığını gösterir
    # (teknik uyumsuzluk). AÇIKSA -- 2026-07-30'dan beri durum bu -- pozitif
    # token beklenendir ve engel değildir; yalnız raporlanır.
    if not REASONING_CONFIG.get("enabled", False) and metrics["total_reasoning_tokens"] > 0:
        reasons.append(
            f"reasoning KAPALI olmasına rağmen pozitif reasoning token: "
            f"{metrics['total_reasoning_tokens']}")

    if target_key == GEMINI_TARGET:
        planner_rows = [r for r in completed_rows if r["kind"] == PROBE_KIND_PLANNER]
        bozuk = [r for r in planner_rows if not r.get("planner_json_parse_ok")]
        if bozuk:
            reasons.append(
                f"response_format=json_object çağrısının çıktısı ayrıştırılamadı: "
                f"{len(bozuk)}/{len(planner_rows)} planner probu")
    else:
        for bayrak in _GROK_CONTRACT_FLAGS:
            bozuk = [r for r in completed_rows if not r.get(bayrak)]
            if bozuk:
                reasons.append(f"grok {bayrak} sağlanmadı: {len(bozuk)}/{len(completed_rows)} probe")
    return reasons


def _gate(target_key: str, metrics: dict, completed_rows: list[dict],
          planned_count: int) -> tuple[str, list[str], list[str]]:
    """(gate_decision, blocker_reasons, non_blocking_notes).

    Öncelik: engel varsa FAIL (engelli bir tur zaten eksik de olabilir, ama
    bilinen bir teknik uyumsuzluğu "eksik veri" diye yumuşatmak yanıltıcı
    olurdu) -> yoksa eksikse INCOMPLETE -> ikisi de yoksa PASS.
    """
    blockers = _blocker_reasons(target_key, metrics, completed_rows)
    notes: list[str] = []
    kurtarilan = metrics["provider_error_recovered_call_count"]
    if kurtarilan:
        notes.append(
            f"retry ile kurtarılmış provider_error: {kurtarilan} çağrı "
            "(tek başına FAIL sebebi DEĞİL; oran ölçümü 3 × 100 sağlık kapısında)")
    eksik = (len(completed_rows) < planned_count
             or metrics["completed_logical_calls"] < planned_count)
    if blockers:
        return GATE_FAIL, blockers, notes
    if eksik:
        notes.append(
            f"eksik: {len(completed_rows)}/{planned_count} probe tamamlandı, "
            f"{metrics['completed_logical_calls']}/{planned_count} başarılı çağrı kaydı var")
        return GATE_INCOMPLETE, blockers, notes
    return GATE_PASS, blockers, notes


def build_report(*, manifest: dict, probe_rows: list[dict], call_rows: list[dict]) -> dict:
    """`compatibility_report.json` içeriği.

    Benchmark pass/fail, base_pass/plus_pass veya model başarı sıralaması
    BİLİNÇLİ olarak YOKTUR -- bu bir yetenek ölçümü değil, uyumluluk/taşıma
    ölçümüdür.

    Hedef başına `gate_decision` (PASS/FAIL/INCOMPLETE) + `blocker_reasons`
    üretilir; bu karar YALNIZ teknik uyumluluğa (taşıma, usage/provenance,
    şema/parse sözleşmeleri) bakar -- modelin görevi ne kadar "iyi" yaptığına
    DEĞİL.
    """
    plan_by_target = {
        GEMINI_TARGET: [pid for pid in manifest["planned_probe_ids"] if pid.startswith("gemini:")],
        GROK_TARGET: [pid for pid in manifest["planned_probe_ids"] if pid.startswith("grok:")],
    }
    final = _final_rows(probe_rows)
    report_out = {
        "compatibility_smoke_schema_version": COMPATIBILITY_SMOKE_SCHEMA_VERSION,
        "created_ts": datetime.now(timezone.utc).isoformat(),
        "targets": {},
    }
    for target_key in TARGET_KEYS:
        probe_ids = plan_by_target[target_key]
        target_report = {"target_model": manifest["targets"][target_key],
                         **_target_call_metrics(call_rows, set(probe_ids))}
        rows = [final[pid] for pid in probe_ids if pid in final]
        completed_rows = [r for r in rows if r.get("status") == STATUS_COMPLETED]
        if target_key == GEMINI_TARGET:
            planner_rows = [r for r in rows if r["kind"] == PROBE_KIND_PLANNER]
            coder_rows = [r for r in rows if r["kind"] == PROBE_KIND_CODER]
            target_report["gemini_diagnostics"] = {
                "planner_probe_completed": sum(
                    r.get("status") == STATUS_COMPLETED for r in planner_rows),
                "coder_probe_completed": sum(
                    r.get("status") == STATUS_COMPLETED for r in coder_rows),
                "planner_json_parse_ok": sum(bool(r.get("planner_json_parse_ok")) for r in planner_rows),
                "planner_schema_ok": sum(bool(r.get("planner_schema_ok")) for r in planner_rows),
                "coder_nonempty": sum(bool(r.get("coder_nonempty")) for r in coder_rows),
                "coder_extract_ok": sum(bool(r.get("coder_extract_ok")) for r in coder_rows),
                "coder_ast_parse_ok": sum(bool(r.get("coder_ast_parse_ok")) for r in coder_rows),
            }
        else:
            target_report["grok_diagnostics"] = {
                "probe_completed": len(completed_rows),
                "json_parse_ok": sum(bool(r.get("json_parse_ok")) for r in rows),
                "mast_label_schema_ok": sum(bool(r.get("mast_label_schema_ok")) for r in rows),
                "interaction_invariant_ok": sum(bool(r.get("interaction_invariant_ok")) for r in rows),
                "mast_adjudication_schema_ok": sum(bool(r.get("mast_adjudication_schema_ok")) for r in rows),
            }
        karar, blockers, notlar = _gate(target_key, target_report, completed_rows, len(probe_ids))
        target_report["gate_decision"] = karar
        target_report["blocker_reasons"] = blockers
        target_report["non_blocking_notes"] = notlar
        report_out["targets"][target_key] = target_report
    return report_out


# --- Orkestrasyon (prepare / run / report) -------------------------------------

def require_verified_git_state(manifest: dict | None = None) -> None:
    """Ücretli `run` aşaması SADECE DOĞRULANMIŞ bir git durumundan başlar.

    Üç ayrı kapı, hepsi fail-closed:

    1. Çalışma ağacı **doğrulanamıyorsa** (git yok / komut hata verdi) durulur.
       "Doğrulanamadı" ile "temiz" aynı şey değildir; bilinmeyen bir kaynak
       koddan ücretli çağrı yapmak, çıktının hangi koda ait olduğunu sonradan
       cevaplanamaz kılar.
    2. Çalışma ağacı **kirliyse** durulur (eval/runner.py ile AYNI disiplin).
    3. HEAD **çözülemiyorsa** ya da manifestteki `git_commit` ile
       **eşleşmiyorsa** durulur: aynı `--name` altında farklı bir committe
       run/resume yapılamaz (manifest kritik alanı; yeni isim gerekir).

    `prepare`/`report`/dry-run bu kapıdan GEÇMEZ -- offline aşamalar git'siz
    ortamda da çalışır.
    """
    dirty = _git_dirty()
    if dirty is None:
        raise CompatibilitySmokeError(
            "git çalışma ağacı durumu DOĞRULANAMIYOR (git yok ya da komut hata "
            "verdi) -- ücretli run fail-closed durur.")
    if dirty:
        raise CompatibilitySmokeError(
            "git çalışma ağacı kirli -- ücretli run aşaması SADECE temiz "
            "ağaçtan başlar. Önce çalışma ağacını commit ederek sabitle.")
    commit = _git_commit()
    if not commit:
        raise CompatibilitySmokeError(
            "git HEAD DOĞRULANAMIYOR -- ücretli run fail-closed durur "
            "(hangi kaynak koddan üretildiği kayda geçemeyecek çağrı yapılmaz).")
    if manifest is not None and manifest.get("git_commit") != commit:
        raise CompatibilitySmokeError(
            f"manifest git_commit ({manifest.get('git_commit')!r}) güncel HEAD "
            f"({commit!r}) ile eşleşmiyor -- aynı isimle FARKLI committe "
            "run/resume yapılamaz, yeni bir --name gerekir.")


def prepare(name: str, out_dir: Path) -> dict:
    """OFFLINE: görev seçimi + manifest + resume bütünlüğü. API anahtarı GEREKMEZ.

    Sıra bilinçli: bu fonksiyon hiçbir API çağrısı yapmadan TAMAMEN tekrar
    çalıştırılabilir (idempotent) -- her çağrıda plan/manifest yeniden
    hesaplanır ve mevcutla karşılaştırılır, asla körlemesine üstüne yazılmaz.
    """
    gemini_ids = select_pilot_task_ids(COMPATIBILITY_SMOKE_GEMINI_TASK_COUNT, salt="gemini")
    grok_ids = select_pilot_task_ids(COMPATIBILITY_SMOKE_GROK_SCENARIO_COUNT, salt="grok")
    snapshot = build_manifest_snapshot(gemini_ids, grok_ids)
    manifest = check_or_write_manifest(out_dir / "manifest.json", snapshot)
    plan = build_plan(gemini_task_ids=manifest["gemini_task_ids"],
                      grok_task_ids=manifest["grok_task_ids"])
    if [p["probe_id"] for p in plan] != manifest["planned_probe_ids"]:
        # check_or_write_manifest zaten plan_fingerprint'i doğruladı; bu satır
        # yalnız bir programlama hatası olasılığına karşı savunma derinliği.
        raise CompatibilitySmokeError(
            "plan_fingerprint eşleşti ama probe_id sırası farklı -- iç tutarsızlık")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "plan.json").write_text(
        json.dumps(plan, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    existing = load_jsonl(out_dir / "probes.jsonl")
    verify_probe_log(existing, plan)
    return {"manifest": manifest, "plan": plan, "existing_probes": existing}


def run(name: str, out_dir: Path, *, dry_run: bool = False) -> list[dict]:
    """Bekleyen probları çalıştırır.

    Sıra (§ "Resume ve stale-data"): offline hazırlık (manifest/resume/
    bütünlük) -> git durum kapısı (doğrulanabilir + temiz + manifestle aynı
    commit) -> (yalnız gerçek run) API anahtarı kontrolü -> API çağrıları.
    `dry_run=True`: yalnız bekleyen probe listesini döner, hiçbir gate/çağrı
    YAPILMAZ (API anahtarı GEREKMEZ).
    """
    prepared = prepare(name, out_dir)
    manifest, plan, existing = prepared["manifest"], prepared["plan"], prepared["existing_probes"]
    if dry_run:
        return pending_probes(plan, existing)

    require_verified_git_state(manifest)
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise CompatibilitySmokeError(
            "OPENROUTER_API_KEY yok -- .env.example'ı .env olarak kopyalayıp doldur.")

    probes_path = out_dir / "probes.jsonl"
    return execute_pending(
        plan, existing, experiment=name, log_namespace=LOG_NAMESPACE,
        on_result=lambda row: append_jsonl(probes_path, [row]))


def report(name: str, out_dir: Path) -> dict:
    """Manifest + probes.jsonl + çağrı logundan `compatibility_report.json` üretir.

    API anahtarı GEREKMEZ -- yalnız var olan artefaktları okur.
    """
    manifest_path = out_dir / "manifest.json"
    if not manifest_path.exists():
        raise CompatibilitySmokeError(f"manifest yok: {manifest_path} -- önce 'prepare' çalıştır")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    probe_rows = load_jsonl(out_dir / "probes.jsonl")
    call_rows = load_jsonl(call_log_path(out_dir))
    # Rapor iki dosyayı birleştirir; tutarsız girdiden rapor üretmek yerine
    # durulur (bozuk bir PASS/FAIL kararı, hiç karar vermemekten kötüdür).
    plan = build_plan(gemini_task_ids=manifest["gemini_task_ids"],
                      grok_task_ids=manifest["grok_task_ids"])
    verify_probe_log(probe_rows, plan)
    verify_call_log(call_rows, probe_rows, manifest["planned_probe_ids"])
    rapor = build_report(manifest=manifest, probe_rows=probe_rows, call_rows=call_rows)
    (out_dir / "compatibility_report.json").write_text(
        json.dumps(rapor, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return rapor
