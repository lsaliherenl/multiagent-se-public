"""Self-consistency testleri — LLM'siz, held-out yüklemeden, deterministik.

Kritik güvence: imza karşılaştırması süreçler arası deterministik olmalı
(set/dict hash-seed sorunu) ve eşdeğerlik metne değil DAVRANIŞA bakmalı.
"""

import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

import config
import uncertainty.self_consistency as sc
from agents import llm as llm_module
from eval.harness import load_all_tasks
from uncertainty.self_consistency import (
    SelfConsistencyError,
    build_manifest_snapshot,
    build_result_record,
    candidate_key,
    check_or_write_manifest,
    cluster_and_score,
    expected_candidate_keys,
    generate_candidate,
    output_signature,
    record_test_inputs,
    run_experiment,
    verify_candidate_records,
    verify_result_records,
)

# Kontrollü mini görev: iki argümanlı toplama
DUMMY_TASK = {
    "task_id": "dummy",
    "prompt": "İki sayıyı toplayan topla(a, b) fonksiyonunu yaz.",
    "entry_point": "topla",
    "reference_solution": "def topla(a, b):\n    return a + b\n",
    "test_code": (
        "def check(candidate):\n"
        "    assert candidate(2, 3) == 5\n"
        "    assert candidate(-1, 1) == 0\n"
        "    assert candidate(0, 0) == 0\n"
    ),
}


def _manifest(tmp_path: Path, monkeypatch, *, n=3, temperature=0.8,
              model=config.MODEL_MAIN, task_set="pilot", name="sc_test"):
    task_dir = tmp_path / "tasks"
    task_dir.mkdir(exist_ok=True)
    (task_dir / "dummy.json").write_text(
        json.dumps(DUMMY_TASK), encoding="utf-8")
    monkeypatch.setitem(sc.TASK_SETS, task_set, task_dir)
    return build_manifest_snapshot(
        name=name, model=model, task_set=task_set, tasks=[deepcopy(DUMMY_TASK)],
        n=n, temperature=temperature, git_commit="a" * 40,
    )


def _install_fast_algorithm(monkeypatch, manifest, out_dir, *, failures=None):
    calls = []
    failures = set(failures or [])

    def fake_generate(task, model, temperature, **context):
        index = context["candidate_index"]
        calls.append((task["task_id"], index, context))
        status = "error" if index in failures else "ok"
        sc.append_jsonl(out_dir / sc.CALL_FILE, {
            "schema_version": manifest["llm_call_schema_version"],
            "experiment": manifest["name"],
            "run_id": context["run_id"],
            "arm": "selfcons",
            "repeat": index,
            "model": model,
            "task_id": task["task_id"],
            "agent_role": "selfcons",
            "status": status,
        })
        if index in failures:
            failures.remove(index)
            raise RuntimeError("terminal provider failure")
        return f"def topla(a, b):\n    return a + b + {index}\n"

    monkeypatch.setattr(sc, "generate_candidate", fake_generate)
    monkeypatch.setattr(sc, "record_test_inputs", lambda task: ["((2, 3), {})"])
    monkeypatch.setattr(
        sc, "output_signature",
        lambda task, inputs, code: "same" if "+ 0" in code or "+ 1" in code else "other",
    )
    monkeypatch.setattr(
        sc, "evaluate_base_plus",
        lambda task, code: {"plus_status": "passed" if "+ 0" in code else "failed"},
    )
    return calls


def test_girdiler_kaydedilir():
    inputs = record_test_inputs(DUMMY_TASK)
    assert len(inputs) == 3
    assert inputs[0] == "((2, 3), {})"


def test_gercek_gorevden_girdi_kaydi():
    task = load_all_tasks()[0]  # humaneval_000
    inputs = record_test_inputs(task)
    assert len(inputs) > 0
    assert all(isinstance(s, str) for s in inputs)


def test_metin_farkli_davranis_ayni_ise_ayni_imza():
    inputs = record_test_inputs(DUMMY_TASK)
    sig1 = output_signature(DUMMY_TASK, inputs, "def topla(a, b):\n    return a + b\n")
    sig2 = output_signature(DUMMY_TASK, inputs, "def topla(x, y):\n    toplam = y + x\n    return toplam\n")
    sig3 = output_signature(DUMMY_TASK, inputs, "def topla(a, b):\n    return a - b\n")
    assert sig1 == sig2  # işlevsel eşdeğerlik, metinsel değil
    assert sig1 != sig3


def test_exception_imzaya_yansir():
    inputs = record_test_inputs(DUMMY_TASK)
    sig = output_signature(DUMMY_TASK, inputs, "def topla(a, b):\n    return a // (a - a)\n")
    assert "__EXC__:ZeroDivisionError" in sig


def test_bozuk_kod_crash_imzasi():
    inputs = record_test_inputs(DUMMY_TASK)
    sig = output_signature(DUMMY_TASK, inputs, "def topla(a, b)\n    return a + b\n")
    assert sig == "__CRASH__"


def test_sonsuz_dongu_timeout_imzasi():
    inputs = record_test_inputs(DUMMY_TASK)
    sig = output_signature(DUMMY_TASK, inputs, "def topla(a, b):\n    while True: pass\n",
                           timeout_s=2)
    assert sig == "__TIMEOUT__"


def test_set_donduren_kod_surecler_arasi_deterministik():
    # Hash randomizasyonu repr sırasını değiştirebilir; kanonikleştirme
    # sayesinde iki AYRI süreçteki imzalar eşit olmalı.
    code = 'def topla(a, b):\n    return {"elma", "armut", str(a + b), "kiraz", "muz"}\n'
    inputs = record_test_inputs(DUMMY_TASK)
    assert output_signature(DUMMY_TASK, inputs, code) == output_signature(DUMMY_TASK, inputs, code)


def test_kumeleme_ve_agreement():
    score = cluster_and_score(["A", "A", "A", "B", "__CRASH__"])
    assert score["n"] == 5
    assert score["n_clusters"] == 3
    assert score["cluster_sizes"] == [3, 1, 1]
    assert score["agreement"] == pytest.approx(0.6)


def test_tam_tutarlilik():
    score = cluster_and_score(["X"] * 4)
    assert score["agreement"] == 1.0
    assert score["n_clusters"] == 1


def test_algoritma_sabitleri_ve_base_only_kesfi_degismedi():
    assert sc.SYSTEM_PROMPT is __import__(
        "pipeline.baseline", fromlist=["SYSTEM_PROMPT"]).SYSTEM_PROMPT
    assert config.SELF_CONSISTENCY_TEMPERATURE == 0.8
    assert config.SELF_CONSISTENCY_N == 5
    plus_task = {**DUMMY_TASK, "base_test_code": "BASE", "plus_test_code": "PLUS"}
    assert sc._input_discovery_tests(plus_task) == "BASE"


def test_manifest_zorunlu_provenance_alanlarini_tasir(tmp_path, monkeypatch):
    manifest = _manifest(tmp_path, monkeypatch)
    for field in sc.MANIFEST_CRITICAL_FIELDS:
        assert field in manifest
    assert manifest["self_consistency_schema_version"] == config.SELF_CONSISTENCY_SCHEMA_VERSION
    assert manifest["candidate_identity_fields"] == [
        "model", "task_set", "task_id", "candidate_index"]
    assert manifest["prompt_hash"] == sc._prompt_hash()
    assert manifest["reasoning_config"] == config.REASONING_CONFIG
    assert manifest["provider_routing"] == config.provider_routing_for(config.MODEL_MAIN)
    assert manifest["llm_call_schema_version"] == config.LLM_CALL_SCHEMA_VERSION


@pytest.mark.parametrize("field,new_value", [
    ("git_commit", "b" * 40),
    ("model", config.MODEL_SECONDARY),
    ("task_set", "heldout"),
    ("n", 4),
    ("temperature", 0.7),
    ("reasoning_config", {"enabled": False}),
])
def test_manifest_farkli_commit_config_taskset_model_n_temperature_reddi(
        tmp_path, monkeypatch, field, new_value):
    manifest = _manifest(tmp_path, monkeypatch)
    path = tmp_path / "manifest.json"
    check_or_write_manifest(path, manifest)
    changed = deepcopy(manifest)
    changed[field] = new_value
    with pytest.raises(SelfConsistencyError, match="manifest uyuşmazlığı"):
        check_or_write_manifest(path, changed)


def test_tam_resume_tamamlanan_adaylari_tekrar_cagirmaz(tmp_path, monkeypatch):
    manifest = _manifest(tmp_path, monkeypatch)
    out_dir = tmp_path / "exp_sc"
    calls = _install_fast_algorithm(monkeypatch, manifest, out_dir)
    first = run_experiment([deepcopy(DUMMY_TASK)], manifest, out_dir)
    assert first["complete"] is True
    assert len(calls) == 3
    first_candidates = (out_dir / sc.CANDIDATE_FILE).read_bytes()
    first_results = (out_dir / sc.RESULT_FILE).read_bytes()

    second = run_experiment([deepcopy(DUMMY_TASK)], manifest, out_dir)
    assert second["generated_this_run"] == 0
    assert len(calls) == 3
    assert (out_dir / sc.CANDIDATE_FILE).read_bytes() == first_candidates
    assert (out_dir / sc.RESULT_FILE).read_bytes() == first_results


def test_tamamlanmis_adayin_llm_call_provenance_i_yoksa_resume_durur(
        tmp_path, monkeypatch):
    manifest = _manifest(tmp_path, monkeypatch, n=1)
    out_dir = tmp_path / "exp_sc"
    calls = _install_fast_algorithm(monkeypatch, manifest, out_dir)
    run_experiment([deepcopy(DUMMY_TASK)], manifest, out_dir)
    (out_dir / sc.CALL_FILE).unlink()
    with pytest.raises(SelfConsistencyError, match="başarılı çağrı provenance"):
        run_experiment([deepcopy(DUMMY_TASK)], manifest, out_dir)
    assert len(calls) == 1  # blocker ilk yeni çağrıdan önce çalıştı


def test_retry_ile_kurtarilan_provider_error_cagri_butunlugunu_bozmaz(
        tmp_path, monkeypatch):
    manifest = _manifest(tmp_path, monkeypatch, n=1)
    out_dir = tmp_path / "exp_sc"
    _install_fast_algorithm(monkeypatch, manifest, out_dir)
    run_experiment([deepcopy(DUMMY_TASK)], manifest, out_dir)
    rows = sc.load_jsonl(out_dir / sc.CALL_FILE)
    retry = {**rows[0], "status": "provider_error"}
    sc.verify_call_records([retry, rows[0]],
                           sc.verify_candidate_records(
                               sc.load_jsonl(out_dir / sc.CANDIDATE_FILE), manifest),
                           manifest)


@pytest.mark.parametrize("field,value", [
    ("run_id", "rogue"), ("model", config.MODEL_SECONDARY),
    ("task_id", "rogue"), ("repeat", 99), ("schema_version", "old"),
])
def test_yabanci_ve_stale_call_log_fail_fast(tmp_path, monkeypatch, field, value):
    manifest = _manifest(tmp_path, monkeypatch, n=1)
    out_dir = tmp_path / "exp_sc"
    _install_fast_algorithm(monkeypatch, manifest, out_dir)
    run_experiment([deepcopy(DUMMY_TASK)], manifest, out_dir)
    row = sc.load_jsonl(out_dir / sc.CALL_FILE)[0]
    row[field] = value
    candidates = sc.verify_candidate_records(
        sc.load_jsonl(out_dir / sc.CANDIDATE_FILE), manifest)
    with pytest.raises(SelfConsistencyError):
        sc.verify_call_records([row], candidates, manifest)


def test_aday_duzeyinde_kesinti_yalniz_eksik_adayi_yeniden_cagirir(
        tmp_path, monkeypatch):
    manifest = _manifest(tmp_path, monkeypatch)
    out_dir = tmp_path / "exp_sc"
    calls = _install_fast_algorithm(monkeypatch, manifest, out_dir, failures={1})
    first = run_experiment([deepcopy(DUMMY_TASK)], manifest, out_dir)
    assert first["completed_candidates"] == 2
    assert first["completed_tasks"] == 0
    assert first["terminal_failures_this_run"] == 1
    assert not (out_dir / sc.RESULT_FILE).exists()

    second = run_experiment([deepcopy(DUMMY_TASK)], manifest, out_dir)
    assert second["complete"] is True
    assert second["generated_this_run"] == 1
    assert [index for _, index, _ in calls] == [0, 1, 2, 1]


def test_call_model_baglami_deney_dizinine_ve_aday_indeksine_bagli(monkeypatch):
    seen = {}

    def fake_call(messages, **kwargs):
        seen.update(kwargs)
        return SimpleNamespace(text="```python\ndef topla(a, b): return a + b\n```")

    monkeypatch.setattr(sc, "call_model", fake_call)
    generate_candidate(
        DUMMY_TASK, config.MODEL_MAIN, 0.8, experiment="formal_sc",
        run_id="selfcons-abc", candidate_index=2,
    )
    assert seen["experiment"] == "formal_sc"
    assert seen["run_id"] == "selfcons-abc"
    assert seen["arm"] == "selfcons"
    assert seen["task_id"] == "dummy"
    assert seen["repeat"] == 2
    assert llm_module._log_path("formal_sc") == (
        config.LOGS_DIR / "exp_formal_sc" / "llm_calls.jsonl")
    assert llm_module._log_path("formal_sc") != llm_module.CALL_LOG


def test_gercek_call_model_mockla_deney_loguna_yazar_globale_yazmaz(
        tmp_path, monkeypatch):
    response = SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(
                content="```python\ndef topla(a, b): return a + b\n```"),
            finish_reason="stop", logprobs=None,
        )],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
    )
    monkeypatch.setattr(llm_module, "LOGS_DIR", tmp_path)
    monkeypatch.setattr(llm_module, "CALL_LOG", tmp_path / "llm_calls.jsonl")
    monkeypatch.setattr(llm_module, "_throttle", lambda: None)
    monkeypatch.setattr(llm_module.litellm, "completion", lambda **kwargs: response)
    monkeypatch.setattr(llm_module.litellm, "completion_cost", lambda **kwargs: 0.001)
    generate_candidate(
        DUMMY_TASK, config.MODEL_MAIN, 0.8, experiment="formal_sc",
        run_id="selfcons-abc", candidate_index=2,
    )
    experiment_log = tmp_path / "exp_formal_sc" / "llm_calls.jsonl"
    assert experiment_log.exists()
    assert not (tmp_path / "llm_calls.jsonl").exists()
    row = json.loads(experiment_log.read_text(encoding="utf-8").strip())
    assert (row["experiment"], row["run_id"], row["arm"], row["task_id"],
            row["repeat"]) == ("formal_sc", "selfcons-abc", "selfcons", "dummy", 2)


@pytest.mark.parametrize("mutation", [
    "duplicate", "unexpected_task", "stale_schema", "wrong_model", "wrong_task_set",
])
def test_duplicate_yabanci_stale_yanlis_model_taskset_aday_fail_fast(
        tmp_path, monkeypatch, mutation):
    manifest = _manifest(tmp_path, monkeypatch, n=1)
    out_dir = tmp_path / "source"
    calls = _install_fast_algorithm(monkeypatch, manifest, out_dir)
    run_experiment([deepcopy(DUMMY_TASK)], manifest, out_dir)
    rows = sc.load_jsonl(out_dir / sc.CANDIDATE_FILE)
    row = deepcopy(rows[0])
    if mutation == "duplicate":
        rows.append(row)
    elif mutation == "unexpected_task":
        row["task_id"] = "rogue"
        rows = [row]
    elif mutation == "stale_schema":
        row["self_consistency_schema_version"] = "0.9"
        rows = [row]
    elif mutation == "wrong_model":
        row["model"] = config.MODEL_SECONDARY
        rows = [row]
    else:
        row["task_set"] = "heldout"
        rows = [row]
    with pytest.raises(SelfConsistencyError):
        verify_candidate_records(rows, manifest)
    assert len(calls) == 1  # doğrulama yeni çağrı yapmadı


def test_eksik_adaydan_sonuc_uretilmez(tmp_path, monkeypatch):
    manifest = _manifest(tmp_path, monkeypatch, n=2)
    out_dir = tmp_path / "exp_sc"
    _install_fast_algorithm(monkeypatch, manifest, out_dir, failures={1})
    report = run_experiment([deepcopy(DUMMY_TASK)], manifest, out_dir)
    assert report["complete"] is False
    assert report["completed_tasks"] == 0
    assert not (out_dir / sc.RESULT_FILE).exists()


def test_ayni_aday_kayitlarindan_sira_bagimsiz_deterministik_sonuc(
        tmp_path, monkeypatch):
    manifest = _manifest(tmp_path, monkeypatch)
    out_dir = tmp_path / "exp_sc"
    _install_fast_algorithm(monkeypatch, manifest, out_dir)
    run_experiment([deepcopy(DUMMY_TASK)], manifest, out_dir)
    rows = sc.load_jsonl(out_dir / sc.CANDIDATE_FILE)
    a = build_result_record("dummy", rows, manifest)
    b = build_result_record("dummy", list(reversed(rows)), manifest)
    assert a == b
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def test_stale_ve_eksik_adaya_bagli_sonuc_reddedilir(tmp_path, monkeypatch):
    manifest = _manifest(tmp_path, monkeypatch, n=2)
    out_dir = tmp_path / "exp_sc"
    _install_fast_algorithm(monkeypatch, manifest, out_dir)
    run_experiment([deepcopy(DUMMY_TASK)], manifest, out_dir)
    candidates = sc.load_jsonl(out_dir / sc.CANDIDATE_FILE)
    result = sc.load_jsonl(out_dir / sc.RESULT_FILE)[0]
    by_key = verify_candidate_records(candidates[:-1], manifest)
    with pytest.raises(SelfConsistencyError, match="eksik aday"):
        verify_result_records([result], by_key, manifest)


@pytest.mark.parametrize("dirty,commit,message", [
    (None, "a" * 40, "doğrulanamadı"),
    (True, "a" * 40, "kirli"),
    (False, None, "HEAD çözülemedi"),
])
def test_git_kapisi_fail_closed(monkeypatch, dirty, commit, message):
    monkeypatch.setattr(sc, "_git_dirty", lambda: dirty)
    monkeypatch.setattr(sc, "_git_commit", lambda: commit)
    with pytest.raises(SelfConsistencyError, match=message):
        sc.require_verified_git_state()


def test_manifest_commitinden_farkli_head_resume_reddi(monkeypatch):
    monkeypatch.setattr(sc, "_git_dirty", lambda: False)
    monkeypatch.setattr(sc, "_git_commit", lambda: "b" * 40)
    with pytest.raises(SelfConsistencyError, match="eşleşmiyor"):
        sc.require_verified_git_state({"git_commit": "a" * 40})


def test_eski_global_dosya_yeni_deneye_migrate_edilmez(tmp_path, monkeypatch):
    manifest = _manifest(tmp_path, monkeypatch, n=1)
    old = tmp_path / "results_selfcons_20260731T074529Z.jsonl"
    old.write_text('{"agreement": 0.972}\n', encoding="utf-8")
    out_dir = tmp_path / "exp_formal"
    _install_fast_algorithm(monkeypatch, manifest, out_dir)
    run_experiment([deepcopy(DUMMY_TASK)], manifest, out_dir)
    assert old.read_text(encoding="utf-8") == '{"agreement": 0.972}\n'
    assert (out_dir / sc.CANDIDATE_FILE).exists()
    assert (out_dir / sc.RESULT_FILE).exists()
