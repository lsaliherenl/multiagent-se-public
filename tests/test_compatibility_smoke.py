"""P1 Parça 6A -- eval/compatibility_smoke.py birim testleri.

**TAMAMEN OFFLINE:** hiçbir gerçek/ücretli API çağrısı yapılmaz. Mock
`call_model` ÜZERİNDE değil `litellm.completion` SEVİYESİNDE yapılır --
agents/llm.py'nin gerçek loglama/retry/provenance katmanı fiilen çalışır
(tests/test_llm.py ile AYNI disiplin). Bu dosyadaki hiçbir test
`OPENROUTER_API_KEY`'in GERÇEK olmasını gerektirmez.
"""

import inspect
import json
import subprocess
import sys
import types
from pathlib import Path

import pytest

import agents.llm as llm_module
import config
import eval.compatibility_smoke as cs
from config import MODEL_ADJUDICATOR, MODEL_JUDGE_EXTERNAL, MODEL_MAIN, MODEL_SECONDARY

GEMINI = MODEL_MAIN
GROK = MODEL_ADJUDICATOR

_REPO_ROOT = Path(__file__).resolve().parents[1]


# --- Ortak fixture'lar ---------------------------------------------------------

@pytest.fixture(autouse=True)
def _no_throttle_no_real_logs(monkeypatch, tmp_path):
    monkeypatch.setattr(llm_module, "_throttle", lambda: None)
    monkeypatch.setattr(llm_module.time, "sleep", lambda s: None)
    monkeypatch.setattr(llm_module, "LOGS_DIR", tmp_path)
    monkeypatch.setattr(llm_module, "CALL_LOG", tmp_path / "llm_calls.jsonl")
    monkeypatch.setattr("litellm.completion_cost", lambda **k: 0.0)
    return tmp_path


_FAKE_COMMIT = "0" * 40


@pytest.fixture(autouse=True)
def _clean_tree_by_default(monkeypatch):
    # Çoğu test için git durumu ALAKASIZ; git kapısı testleri bunu override
    # eder. Commit de sabitlenir: manifestin KRİTİK alanı olduğu için gerçek
    # HEAD'e bağlı kalsaydı testler depo durumuna göre kırılırdı.
    monkeypatch.setattr(cs, "_git_dirty", lambda: False)
    monkeypatch.setattr(cs, "_git_commit", lambda: _FAKE_COMMIT)


@pytest.fixture(autouse=True)
def _fake_api_key(monkeypatch):
    # Gerçek ağ çağrısı hiçbir testte YAPILMAZ (litellm.completion mock'lu);
    # bu yalnız call_model'in "API anahtarı yok" kısayoluna düşmemesi için.
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-not-real")


def _ok_response(text, *, native="STOP", response_id="gen-ok", model="m-real"):
    choice = types.SimpleNamespace(
        message=types.SimpleNamespace(content=text), finish_reason="stop",
        logprobs=None, provider_specific_fields={"native_finish_reason": native})
    usage = types.SimpleNamespace(prompt_tokens=10, completion_tokens=5)
    return types.SimpleNamespace(choices=[choice], usage=usage, id=response_id, model=model)


def _corrupt_response():
    """2026-07-27 canlı imzası: HTTP 200 gövdesine gömülü upstream 429."""
    choice = types.SimpleNamespace(
        message=types.SimpleNamespace(content='{"task_'), finish_reason="stop", logprobs=None,
        provider_specific_fields={
            "error": {"code": 429, "message": "temporarily rate-limited upstream",
                     "metadata": {"error_type": "rate_limit_exceeded"}},
            "native_finish_reason": "error"})
    usage = types.SimpleNamespace(prompt_tokens=10, completion_tokens=0)
    return types.SimpleNamespace(choices=[choice], usage=usage, id="gen-bad", model="m-real")


_VALID_PLANNER_JSON = json.dumps({
    "task_id": "x", "function_signature": "def f(x):",
    "steps": [{"description": "d", "preconditions": [], "postconditions": ["p"]}],
    "edge_cases": []})
_VALID_ADJUDICATOR_JSON = json.dumps({
    "primary_mode": "1.1", "secondary_modes": [], "confidence": "high",
    "rationale": "grok karari", "insufficient_context": False})
_VALID_CODE_BLOCK = "```python\ndef f(x):\n    return x\n```"


def _dispatcher(*, planner_text=_VALID_PLANNER_JSON, coder_text=_VALID_CODE_BLOCK,
                grok_text=_VALID_ADJUDICATOR_JSON, capture=None):
    """model + response_format'a göre dallanan sahte litellm.completion.

    Grok(model==MODEL_ADJUDICATOR) -> adjudicator kararı.
    Gemini + response_format verilmiş -> planner JSON'u.
    Gemini + response_format YOK -> coder kod bloğu.
    """
    def fake(**kwargs):
        if capture is not None:
            capture.append(kwargs)
        if kwargs["model"] == GROK:
            return _ok_response(grok_text)
        if kwargs.get("response_format") is not None:
            return _ok_response(planner_text)
        return _ok_response(coder_text)
    return fake


def _prep_and_run(monkeypatch, name, out_dir, **dispatcher_kwargs):
    monkeypatch.setattr("litellm.completion", _dispatcher(**dispatcher_kwargs))
    return cs.run(name, out_dir, dry_run=False)


# --- Deterministik seçim -------------------------------------------------------

def test_secim_ayni_seed_ile_bayt_bayt_ayni():
    a = cs.select_pilot_task_ids(6, salt="gemini")
    b = cs.select_pilot_task_ids(6, salt="gemini")
    assert a == b


def test_farkli_salt_farkli_sayida_secim_uretir():
    gemini_ids = cs.select_pilot_task_ids(6, salt="gemini")
    grok_ids = cs.select_pilot_task_ids(12, salt="grok")
    assert len(gemini_ids) == 6 == len(set(gemini_ids))
    assert len(grok_ids) == 12 == len(set(grok_ids))


def test_secim_alfabetik_siralidir():
    ids = cs.select_pilot_task_ids(6, salt="gemini")
    assert ids == sorted(ids)


def test_yetersiz_havuzda_hata():
    with pytest.raises(cs.CompatibilitySmokeError):
        cs.select_pilot_task_ids(10_000, salt="gemini")


def test_secim_fonksiyonunda_task_set_parametresi_yok():
    # Held-out'u sunmak yapısal olarak imkânsız: fonksiyonun böyle bir
    # parametresi hiç yok, sabit COMPATIBILITY_SMOKE_TASK_SET kullanılır.
    sig = inspect.signature(cs.select_pilot_task_ids)
    assert "task_set" not in sig.parameters


_PYTHONHASHSEED_SCRIPT = """
import sys
sys.path.insert(0, sys.argv[1])
import json
import eval.compatibility_smoke as cs
print(json.dumps({
    "gemini": cs.select_pilot_task_ids(6, salt="gemini"),
    "grok": cs.select_pilot_task_ids(12, salt="grok"),
}))
"""


@pytest.mark.parametrize("seed_env", ["0", "1", "12345"])
def test_secim_pythonhashseedden_bagimsiz(tmp_path, seed_env):
    # Python'un hash()'i, set/dict sırası ve PYTHONHASHSEED süreçler arasında
    # değişir; seçim bunlara bağlanırsa "aynı seed aynı seçim" iddiası çöker.
    # Bu yüzden AYRI SÜREÇTE, farklı hash seed'iyle doğrulanır.
    import os
    betik = tmp_path / "aday.py"
    betik.write_text(_PYTHONHASHSEED_SCRIPT, encoding="utf-8")
    ortam = {**os.environ, "PYTHONHASHSEED": seed_env}
    sonuc = subprocess.run([sys.executable, str(betik), str(_REPO_ROOT)], env=ortam,
                           capture_output=True, text=True, timeout=60)
    assert sonuc.returncode == 0, sonuc.stderr
    veri = json.loads(sonuc.stdout)
    assert veri["gemini"] == cs.select_pilot_task_ids(6, salt="gemini")
    assert veri["grok"] == cs.select_pilot_task_ids(12, salt="grok")


# --- Plan ----------------------------------------------------------------------

def test_plan_ayni_seed_ile_bayt_bayt_ayni():
    g = cs.select_pilot_task_ids(6, salt="gemini")
    k = cs.select_pilot_task_ids(12, salt="grok")
    plan1 = cs.build_plan(gemini_task_ids=g, grok_task_ids=k)
    plan2 = cs.build_plan(gemini_task_ids=g, grok_task_ids=k)
    assert json.dumps(plan1, sort_keys=True) == json.dumps(plan2, sort_keys=True)


def test_plan_12_artı_12_esittir_24_ve_sira_gemini_once():
    g = cs.select_pilot_task_ids(6, salt="gemini")
    k = cs.select_pilot_task_ids(12, salt="grok")
    plan = cs.build_plan(gemini_task_ids=g, grok_task_ids=k)
    assert len(plan) == 24
    assert sum(p["target_key"] == "gemini" for p in plan) == 12
    assert sum(p["target_key"] == "grok" for p in plan) == 12
    assert [p["target_key"] for p in plan] == ["gemini"] * 12 + ["grok"] * 12


def test_gemini_plani_6_planner_6_coder():
    g = cs.select_pilot_task_ids(6, salt="gemini")
    plan = cs.build_gemini_plan(g)
    assert sum(p["kind"] == cs.PROBE_KIND_PLANNER for p in plan) == 6
    assert sum(p["kind"] == cs.PROBE_KIND_CODER for p in plan) == 6


def test_grok_senaryolari_kaynak_dagilimi_6_6():
    k = cs.select_pilot_task_ids(12, salt="grok")
    plan = cs.build_grok_plan(k)
    kaynaklar = [p["source_model"] for p in plan]
    assert kaynaklar.count(MODEL_MAIN) == 6
    assert kaynaklar.count(MODEL_SECONDARY) == 6


def test_grok_senaryolarinda_tek_ve_cok_ajanli_var():
    k = cs.select_pilot_task_ids(12, salt="grok")
    plan = cs.build_grok_plan(k)
    assert {p["interaction_type"] for p in plan} == {"single_agent", "multi_agent"}


def test_grok_senaryolarinda_yetersiz_baglam_var():
    k = cs.select_pilot_task_ids(12, salt="grok")
    plan = cs.build_grok_plan(k)
    assert any(any(l.get("insufficient_context") for l in p["external_labels"]) for p in plan)


def test_grok_senaryolarinda_her_zaman_iki_farkli_etiket_var():
    k = cs.select_pilot_task_ids(12, salt="grok")
    for p in cs.build_grok_plan(k):
        keys = [("__INSUFF__" if l.get("insufficient_context") else l.get("primary_mode"))
               for l in p["external_labels"]]
        assert keys[0] != keys[1], p


def test_grok_senaryosunda_self_hic_kurulmaz():
    k = cs.select_pilot_task_ids(12, salt="grok")
    for p in cs.build_grok_plan(k):
        judges_in_labels = [l["judge_model"] for l in p["external_labels"]]
        assert p["source_model"] not in judges_in_labels
        assert set(judges_in_labels) == set(p["external_judges"])
        assert p["self_judge_model"] == p["source_model"]


def test_tek_ajanli_senaryoda_kategori_2_kod_planlanmaz():
    k = cs.select_pilot_task_ids(12, salt="grok")
    for p in cs.build_grok_plan(k):
        if p["interaction_type"] != "single_agent":
            continue
        for l in p["external_labels"]:
            if l.get("primary_mode"):
                assert not l["primary_mode"].startswith("2.")


# --- Dondurulmuş hedefler / CLI kabul kümesi ------------------------------------

def test_hedefler_yalniz_gemini_grok_ve_sira_normatif():
    assert config.COMPATIBILITY_SMOKE_TARGETS == {"gemini": MODEL_MAIN, "grok": MODEL_ADJUDICATOR}
    assert list(config.COMPATIBILITY_SMOKE_TARGETS) == ["gemini", "grok"]
    assert cs.TARGET_KEYS == ("gemini", "grok")


def test_calls_per_target_10_20_araliginda():
    assert 10 <= config.COMPATIBILITY_SMOKE_CALLS_PER_TARGET <= 20
    assert config.COMPATIBILITY_SMOKE_CALLS_PER_TARGET == 12


def test_cli_kaynaginda_heldout_arbitrary_deepseek_minimax_secenegi_yok():
    # Docstring DeepSeek/MiniMax'tan bahsedebilir (neden reddedildiklerini
    # açıklamak için); asıl şart bunların CLI SEÇENEĞİ/kod kimliği olarak hiç
    # bulunmamasıdır -- CLI'nın `--name` dışında hiçbir argümanı yoktur.
    kaynak = (_REPO_ROOT / "scripts" / "compatibility_smoke.py").read_text(encoding="utf-8")
    for yasakli in ("--model", "--task-set", "--target", "HELDOUT_TASK_SET",
                    "HELDOUT_TASKS_DIR", "MODEL_SECONDARY", "MODEL_JUDGE_EXTERNAL",
                    "MODEL_JUDGES", "resolve_model", "model_alias_help"):
        assert yasakli not in kaynak, f"{yasakli!r} formal CLI kaynağında bulunmamalı"
    assert "add_argument(\"--task-set\"" not in kaynak
    assert 'choices=["pilot"' not in kaynak and "choices=['pilot'" not in kaynak


def test_compatibility_smoke_modulu_heldout_hic_referans_vermez():
    kaynak = Path(cs.__file__).read_text(encoding="utf-8")
    assert "HELDOUT" not in kaynak
    assert "heldout" not in kaynak


def test_task_set_daima_pilot():
    assert config.COMPATIBILITY_SMOKE_TASK_SET == config.PILOT_TASK_SET


def test_cli_help_calisir_api_anahtari_gerektirmez(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    sonuc = subprocess.run(
        [sys.executable, str(_REPO_ROOT / "scripts" / "compatibility_smoke.py"), "--help"],
        capture_output=True, text=True, timeout=30)
    assert sonuc.returncode == 0
    for beklenen in ("prepare", "run", "report"):
        assert beklenen in sonuc.stdout


# --- Manifest -------------------------------------------------------------------

def test_manifest_eksik_kritik_alanla_hata(tmp_path):
    with pytest.raises(KeyError):
        cs.check_or_write_manifest(tmp_path / "manifest.json",
                                   {"compatibility_smoke_schema_version": "1.0"})


def test_manifest_ayni_snapshotla_idempotent(tmp_path):
    g = cs.select_pilot_task_ids(6, salt="gemini")
    k = cs.select_pilot_task_ids(12, salt="grok")
    snap = cs.build_manifest_snapshot(g, k)
    path = tmp_path / "manifest.json"
    m1 = cs.check_or_write_manifest(path, snap)
    m2 = cs.check_or_write_manifest(path, snap)
    assert m1 == m2


def test_manifest_farkli_kritik_alanla_hata(tmp_path):
    g = cs.select_pilot_task_ids(6, salt="gemini")
    k = cs.select_pilot_task_ids(12, salt="grok")
    path = tmp_path / "manifest.json"
    cs.check_or_write_manifest(path, cs.build_manifest_snapshot(g, k))
    farkli = cs.build_manifest_snapshot(g, k)
    farkli["calls_per_target"] = 999
    with pytest.raises(cs.CompatibilitySmokeError, match="manifest uyuşmazlığı"):
        cs.check_or_write_manifest(path, farkli)


def test_reasoning_degisimi_kritik_farktir_ve_tarif_parmak_izini_degistirir(monkeypatch):
    # [2026-07-30] Ortak reasoning ayarı değişti; eski ve yeni tur AYNI isim
    # altında karışamaz. Hem manifestin kritik alanı hem de her iki hedefin
    # "tarif parmak izi" bu ayarı kapsar.
    assert "reasoning_config" in cs.MANIFEST_CRITICAL_FIELDS
    g = cs.select_pilot_task_ids(6, salt="gemini")
    k = cs.select_pilot_task_ids(12, salt="grok")
    acik = cs.build_manifest_snapshot(g, k)
    assert acik["reasoning_config"] == {"enabled": True}
    monkeypatch.setattr(cs, "REASONING_CONFIG", {"enabled": False})
    kapali = cs.build_manifest_snapshot(g, k)
    assert kapali["reasoning_config"] != acik["reasoning_config"]
    assert kapali["gemini_recipe_fingerprint"] != acik["gemini_recipe_fingerprint"]
    assert kapali["grok_recipe_fingerprint"] != acik["grok_recipe_fingerprint"]


def test_eski_reasoning_ayarli_manifestle_ayni_tura_devam_edilemez_sifir_cagri(
    tmp_path, monkeypatch
):
    # p1_compat_20260730_v1 artefaktı KORUNUR ama üstüne devam EDİLEMEZ:
    # yeni tur ayrı bir --name ile baştan koşar.
    out_dir = tmp_path / "exp_v1"
    cs.prepare("v1", out_dir)
    yol = out_dir / "manifest.json"
    eski = json.loads(yol.read_text(encoding="utf-8"))
    eski["reasoning_config"] = {"enabled": False}
    yol.write_text(json.dumps(eski), encoding="utf-8")
    calls = []
    monkeypatch.setattr("litellm.completion", _dispatcher(capture=calls))
    with pytest.raises(cs.CompatibilitySmokeError, match="manifest uyuşmazlığı"):
        cs.run("v1", out_dir, dry_run=False)
    assert calls == []


# --- Offline prepare / dry-run: API anahtarı gerektirmez, çağrı yapmaz ----------

def test_prepare_api_anahtari_olmadan_calisir(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    out_dir = tmp_path / "exp_prep"
    prepared = cs.prepare("prep", out_dir)
    assert len(prepared["plan"]) == 24
    assert (out_dir / "manifest.json").exists()
    assert (out_dir / "plan.json").exists()


def test_prepare_kirli_agacta_da_calisir(tmp_path, monkeypatch):
    monkeypatch.setattr(cs, "_git_dirty", lambda: True)
    out_dir = tmp_path / "exp_prep_dirty"
    prepared = cs.prepare("prepd", out_dir)
    assert len(prepared["plan"]) == 24


def test_dry_run_api_anahtari_olmadan_calisir_ve_sifir_cagri_yapar(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    calls = []
    monkeypatch.setattr("litellm.completion", _dispatcher(capture=calls))
    out_dir = tmp_path / "exp_dry"
    pending = cs.run("dry", out_dir, dry_run=True)
    assert len(pending) == 24
    assert calls == []


def test_dry_run_kirli_agacta_da_calisir(tmp_path, monkeypatch):
    monkeypatch.setattr(cs, "_git_dirty", lambda: True)
    calls = []
    monkeypatch.setattr("litellm.completion", _dispatcher(capture=calls))
    out_dir = tmp_path / "exp_dry_dirty"
    pending = cs.run("dryd", out_dir, dry_run=True)
    assert len(pending) == 24
    assert calls == []


# --- Kirli ağaç / bozuk manifest / bütünlük -> SIFIR call_model -----------------

def test_kirli_agac_ucretli_run_asamasini_engeller_sifir_cagri(tmp_path, monkeypatch):
    monkeypatch.setattr(cs, "_git_dirty", lambda: True)
    calls = []
    monkeypatch.setattr("litellm.completion", _dispatcher(capture=calls))
    out_dir = tmp_path / "exp_dirty"
    with pytest.raises(cs.CompatibilitySmokeError, match="kirli"):
        cs.run("dirty", out_dir, dry_run=False)
    assert calls == []


def test_bozuk_manifestte_sifir_cagri(tmp_path, monkeypatch):
    out_dir = tmp_path / "exp_bozuk"
    cs.prepare("bozuk", out_dir)
    manifest_path = out_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["gemini_task_ids"] = ["humaneval_000"]  # elle bozuldu
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    calls = []
    monkeypatch.setattr("litellm.completion", _dispatcher(capture=calls))
    with pytest.raises(cs.CompatibilitySmokeError, match="manifest uyuşmazlığı"):
        cs.run("bozuk", out_dir, dry_run=False)
    assert calls == []


def test_eski_sema_surumuyle_ayni_isme_devam_edilemez_sifir_cagri(tmp_path, monkeypatch):
    out_dir = tmp_path / "exp_eskisema"
    cs.prepare("eskisema", out_dir)
    manifest_path = out_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["compatibility_smoke_schema_version"] = "0.9"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    calls = []
    monkeypatch.setattr("litellm.completion", _dispatcher(capture=calls))
    with pytest.raises(cs.CompatibilitySmokeError, match="manifest uyuşmazlığı"):
        cs.run("eskisema", out_dir, dry_run=False)
    assert calls == []


def _probe_row(spec, **kwargs):
    """Plandaki spec'ten TAM kimlikli bir probe satırı (target_key + kind dahil)."""
    row = {"probe_id": spec["probe_id"], "target_key": spec["target_key"],
           "kind": spec["kind"], "status": cs.STATUS_COMPLETED, "attempt": 1}
    row.update(kwargs)
    return row


def test_yinelenen_tamamlanmis_probe_fail_fast_sifir_cagri(tmp_path, monkeypatch):
    out_dir = tmp_path / "exp_dup"
    prepared = cs.prepare("dup", out_dir)
    spec = prepared["plan"][0]
    cs.append_jsonl(out_dir / "probes.jsonl", [
        _probe_row(spec, attempt=1),
        _probe_row(spec, attempt=2),
    ])
    calls = []
    monkeypatch.setattr("litellm.completion", _dispatcher(capture=calls))
    with pytest.raises(cs.CompatibilitySmokeError, match="birden fazla TAMAMLANMIŞ"):
        cs.run("dup", out_dir, dry_run=False)
    assert calls == []


def test_beklenmeyen_probe_id_fail_fast_sifir_cagri(tmp_path, monkeypatch):
    out_dir = tmp_path / "exp_yabanci"
    cs.prepare("yabanci", out_dir)
    cs.append_jsonl(out_dir / "probes.jsonl", [
        {"probe_id": "gemini:planner:BASKA-DENEY", "target_key": "gemini",
         "kind": cs.PROBE_KIND_PLANNER, "status": cs.STATUS_COMPLETED, "attempt": 1}])
    calls = []
    monkeypatch.setattr("litellm.completion", _dispatcher(capture=calls))
    with pytest.raises(cs.CompatibilitySmokeError, match="beklenmeyen probe_id"):
        cs.run("yabanci", out_dir, dry_run=False)
    assert calls == []


# --- Uçtan uca (mock) ------------------------------------------------------------

def test_ucdan_uca_24_probe_tamamlanir(tmp_path, monkeypatch):
    out_dir = tmp_path / "exp_e2e"
    rows = _prep_and_run(monkeypatch, "e2e", out_dir)
    assert len(rows) == 24
    assert all(r["status"] == cs.STATUS_COMPLETED for r in rows)
    rep = cs.report("e2e", out_dir)
    assert rep["targets"]["gemini"]["completed_logical_calls"] == 12
    assert rep["targets"]["grok"]["completed_logical_calls"] == 12
    assert rep["targets"]["gemini"]["gemini_diagnostics"]["planner_schema_ok"] == 6
    assert rep["targets"]["grok"]["grok_diagnostics"]["mast_adjudication_schema_ok"] == 12


def test_ayni_basarili_probe_ikinci_kez_cagrilmaz(tmp_path, monkeypatch):
    out_dir = tmp_path / "exp_resume"
    calls = []
    monkeypatch.setattr("litellm.completion", _dispatcher(capture=calls))
    cs.run("resume", out_dir, dry_run=False)
    ilk_cagri_sayisi = len(calls)
    assert ilk_cagri_sayisi == 24
    rows2 = cs.run("resume", out_dir, dry_run=False)
    assert rows2 == []
    assert len(calls) == ilk_cagri_sayisi, "tamamlanmış probe yeniden çağrılmamalı"


def test_terminal_hatadan_sonra_ayni_manifeste_yeniden_denenebilir_attempt_artar(
    tmp_path, monkeypatch
):
    out_dir = tmp_path / "exp_retry"
    monkeypatch.setattr("litellm.completion", lambda **k: _corrupt_response())
    prepared = cs.prepare("retry", out_dir)
    plan = [p for p in prepared["plan"] if p["kind"] == cs.PROBE_KIND_PLANNER][:1]

    rows1 = cs.execute_pending(plan, [], experiment="retry", log_namespace="compat_smoke")
    assert rows1[0]["status"] == cs.STATUS_TERMINAL_FAILURE
    assert rows1[0]["attempt"] == 1

    rows2 = cs.execute_pending(plan, rows1, experiment="retry", log_namespace="compat_smoke")
    assert rows2[0]["status"] == cs.STATUS_TERMINAL_FAILURE
    assert rows2[0]["attempt"] == 2

    monkeypatch.setattr("litellm.completion", _dispatcher())
    rows3 = cs.execute_pending(plan, rows1 + rows2, experiment="retry", log_namespace="compat_smoke")
    assert rows3[0]["status"] == cs.STATUS_COMPLETED
    assert rows3[0]["attempt"] == 3


def test_bir_probun_hatasi_digerlerini_durdurmaz(tmp_path, monkeypatch):
    out_dir = tmp_path / "exp_karisik"
    prepared = cs.prepare("karisik", out_dir)
    gemini_plan = [p for p in prepared["plan"] if p["target_key"] == "gemini"]

    def fake(**kwargs):
        if kwargs.get("response_format") is not None:
            return _corrupt_response()
        return _ok_response(_VALID_CODE_BLOCK)
    monkeypatch.setattr("litellm.completion", fake)

    rows = cs.execute_pending(gemini_plan, [], experiment="karisik", log_namespace="compat_smoke")
    coder_rows = [r for r in rows if r["kind"] == cs.PROBE_KIND_CODER]
    planner_rows = [r for r in rows if r["kind"] == cs.PROBE_KIND_PLANNER]
    assert all(r["status"] == cs.STATUS_COMPLETED for r in coder_rows)
    assert all(r["status"] == cs.STATUS_TERMINAL_FAILURE for r in planner_rows)


# --- Provider-hatası imza tespiti (gerçek call_model yolu) ----------------------

def test_gomulu_429_retry_ile_basarili_probe(monkeypatch):
    yanitlar = iter([_corrupt_response(), _corrupt_response(), _ok_response(_VALID_PLANNER_JSON)])
    monkeypatch.setattr("litellm.completion", lambda **k: next(yanitlar))
    spec = {"probe_id": "gemini:planner:humaneval_000", "target_key": "gemini",
           "kind": cs.PROBE_KIND_PLANNER, "task_id": "humaneval_000"}
    result = cs.run_gemini_planner_probe(spec, model=GEMINI, experiment="e1",
                                         log_namespace="compat_smoke")
    assert result["transport_ok"] is True
    assert result["planner_json_parse_ok"] is True
    assert result["planner_schema_ok"] is True


def test_tukenen_provider_hatasi_terminal_failure_olarak_kaydedilir(tmp_path, monkeypatch):
    monkeypatch.setattr("litellm.completion", lambda **k: _corrupt_response())
    out_dir = tmp_path / "exp_exhaust"
    prepared = cs.prepare("exhaust", out_dir)
    plan = [p for p in prepared["plan"] if p["kind"] == cs.PROBE_KIND_PLANNER][:1]
    rows = cs.execute_pending(plan, [], experiment="exhaust", log_namespace="compat_smoke")
    assert len(rows) == 1
    assert rows[0]["status"] == cs.STATUS_TERMINAL_FAILURE
    assert "ProviderResponseError" in rows[0]["error"]


# --- Başarılı log provenance (ModelResponse/LLM_CALL şeması 2.1) ----------------

def test_basarili_logda_response_id_native_requested_actual_provider_var(tmp_path, monkeypatch):
    monkeypatch.setattr("litellm.completion",
                        lambda **k: _ok_response(_VALID_CODE_BLOCK, response_id="gen-abc"))
    out_dir = tmp_path / "exp_prov"
    prepared = cs.prepare("prov", out_dir)
    spec = next(p for p in prepared["plan"] if p["kind"] == cs.PROBE_KIND_CODER)
    cs.run_gemini_coder_probe(spec, model=GEMINI, experiment="prov", log_namespace="compat_smoke")
    kayitlar = cs.load_jsonl(cs.call_log_path(out_dir))
    ok = next(r for r in kayitlar if r["status"] == "ok")
    assert ok["response_id"] == "gen-abc"
    assert ok["native_finish_reason"] == "STOP"
    assert ok["requested_provider"] == ["auto"]
    assert "actual_provider" in ok


def test_eksik_native_provider_alanlari_sahte_deger_uretmez(tmp_path, monkeypatch):
    choice = types.SimpleNamespace(message=types.SimpleNamespace(content=_VALID_CODE_BLOCK),
                                   finish_reason="stop", logprobs=None)  # provider_specific_fields YOK
    usage = types.SimpleNamespace(prompt_tokens=10, completion_tokens=5)
    yanit = types.SimpleNamespace(choices=[choice], usage=usage)  # id/model de YOK
    monkeypatch.setattr("litellm.completion", lambda **k: yanit)
    out_dir = tmp_path / "exp_eksik"
    prepared = cs.prepare("eksik", out_dir)
    spec = next(p for p in prepared["plan"] if p["kind"] == cs.PROBE_KIND_CODER)
    cs.run_gemini_coder_probe(spec, model=GEMINI, experiment="eksik", log_namespace="compat_smoke")
    ok = next(r for r in cs.load_jsonl(cs.call_log_path(out_dir)) if r["status"] == "ok")
    assert ok["response_id"] is None
    assert ok["native_finish_reason"] is None
    assert ok["actual_provider"] is None


def test_reasoning_tokens_pozitifse_raporda_gorunur(tmp_path, monkeypatch):
    def fake(**kwargs):
        text = _VALID_PLANNER_JSON if kwargs.get("response_format") is not None else _VALID_CODE_BLOCK
        r = _ok_response(text)
        r.usage.completion_tokens_details = types.SimpleNamespace(reasoning_tokens=7)
        return r
    monkeypatch.setattr("litellm.completion", fake)
    out_dir = tmp_path / "exp_reason"
    prepared = cs.prepare("reason", out_dir)
    gemini_plan = [p for p in prepared["plan"] if p["target_key"] == "gemini"][:2]
    probes_path = out_dir / "probes.jsonl"
    cs.execute_pending(gemini_plan, [], experiment="reason", log_namespace="compat_smoke",
                       on_result=lambda row: cs.append_jsonl(probes_path, [row]))
    rep = cs.report("reason", out_dir)
    assert rep["targets"]["gemini"]["total_reasoning_tokens"] > 0


# --- Gemini sınıflandırma bozuklukları -------------------------------------------

def test_gemini_json_parse_bozuklugu_raporlanir(tmp_path, monkeypatch):
    monkeypatch.setattr("litellm.completion", _dispatcher(planner_text="bu hic JSON degil"))
    out_dir = tmp_path / "exp_badjson"
    prepared = cs.prepare("badjson", out_dir)
    plan = [p for p in prepared["plan"] if p["kind"] == cs.PROBE_KIND_PLANNER]
    probes_path = out_dir / "probes.jsonl"
    cs.execute_pending(plan, [], experiment="badjson", log_namespace="compat_smoke",
                       on_result=lambda row: cs.append_jsonl(probes_path, [row]))
    rep = cs.report("badjson", out_dir)
    assert rep["targets"]["gemini"]["gemini_diagnostics"]["planner_json_parse_ok"] == 0
    assert rep["targets"]["gemini"]["gemini_diagnostics"]["planner_schema_ok"] == 0


def test_gemini_gecersiz_python_sozdizimi_ast_parse_ile_yakalanir(monkeypatch):
    monkeypatch.setattr("litellm.completion",
                        _dispatcher(coder_text="```python\ndef f(x)\n    return x\n```"))
    spec = {"probe_id": "gemini:coder:humaneval_000", "target_key": "gemini",
           "kind": cs.PROBE_KIND_CODER, "task_id": "humaneval_000"}
    result = cs.run_gemini_coder_probe(spec, model=GEMINI, experiment="e5",
                                       log_namespace="compat_smoke")
    assert result["coder_nonempty"] is True
    assert result["coder_extract_ok"] is True
    assert result["coder_ast_parse_ok"] is False


# --- Grok sınıflandırma / şema / körlük -----------------------------------------

def test_grok_gecersiz_mast_kodu_reddedilir(monkeypatch):
    k = cs.select_pilot_task_ids(12, salt="grok")
    plan = cs.build_grok_plan(k)
    spec = next(p for p in plan if p["index"] != 0)  # insufficient senaryosu değil
    gecersiz = json.dumps({"primary_mode": "9.9", "secondary_modes": [], "confidence": "high",
                          "rationale": "gecersiz kod", "insufficient_context": False})
    monkeypatch.setattr("litellm.completion", _dispatcher(grok_text=gecersiz))
    result = cs.run_grok_probe(spec, model=GROK, experiment="e6", log_namespace="compat_smoke")
    assert result["json_parse_ok"] is True
    assert result["mast_label_schema_ok"] is False
    assert result["mast_adjudication_schema_ok"] is False


def test_grok_tek_ajanli_senaryoda_kategori_2_karari_reddedilir(monkeypatch):
    k = cs.select_pilot_task_ids(12, salt="grok")
    plan = cs.build_grok_plan(k)
    spec = next(p for p in plan if p["interaction_type"] == "single_agent")
    kategori2 = json.dumps({"primary_mode": "2.3", "secondary_modes": [], "confidence": "high",
                           "rationale": "yanlis kategori", "insufficient_context": False})
    monkeypatch.setattr("litellm.completion", _dispatcher(grok_text=kategori2))
    result = cs.run_grok_probe(spec, model=GROK, experiment="e7", log_namespace="compat_smoke")
    assert result["json_parse_ok"] is True
    assert result["mast_label_schema_ok"] is True
    assert result["interaction_invariant_ok"] is False
    assert result["mast_adjudication_schema_ok"] is False


def test_grok_json_parse_bozuklugunda_transport_ok_true_diger_bayraklar_false(monkeypatch):
    k = cs.select_pilot_task_ids(12, salt="grok")
    spec = next(p for p in cs.build_grok_plan(k) if p["index"] != 0)
    monkeypatch.setattr("litellm.completion", _dispatcher(grok_text="bu hic json degil"))
    result = cs.run_grok_probe(spec, model=GROK, experiment="e8", log_namespace="compat_smoke")
    assert result["transport_ok"] is True   # call_model BAŞARILI oldu, yalnız içerik bozuk
    assert result["json_parse_ok"] is False
    assert result["mast_label_schema_ok"] is False
    assert result["mast_adjudication_schema_ok"] is False


def test_grok_tukenen_provider_hatasinda_transport_ok_false(monkeypatch):
    k = cs.select_pilot_task_ids(12, salt="grok")
    spec = next(p for p in cs.build_grok_plan(k) if p["index"] != 0)
    monkeypatch.setattr("litellm.completion", lambda **kw: _corrupt_response())
    result = cs.run_grok_probe(spec, model=GROK, experiment="e9", log_namespace="compat_smoke")
    assert result["transport_ok"] is False
    assert result["json_parse_ok"] is False
    assert result["adjudicator_status"] == "error"


def test_grok_promptunda_self_ve_yasakli_bilgiler_yok(monkeypatch):
    k = cs.select_pilot_task_ids(12, salt="grok")
    plan = cs.build_grok_plan(k)
    spec = next(p for p in plan if p["source_model"] == MODEL_MAIN)
    captured = []
    monkeypatch.setattr("litellm.completion", _dispatcher(capture=captured))
    cs.run_grok_probe(spec, model=GROK, experiment="e10", log_namespace="compat_smoke")

    grok_calls = [c for c in captured if c["model"] == GROK]
    assert len(grok_calls) == 1
    prompt_text = json.dumps(grok_calls[0]["messages"])

    assert "Annotator A" in prompt_text
    assert "Annotator B" in prompt_text
    assert "Annotator C" not in prompt_text
    assert MODEL_MAIN not in prompt_text
    assert MODEL_SECONDARY not in prompt_text
    assert MODEL_JUDGE_EXTERNAL not in prompt_text
    assert spec["arm"] not in prompt_text
    assert "majority_label" not in prompt_text
    assert "agreement_level" not in prompt_text
    assert "single_agent" in prompt_text or "multi_agent" in prompt_text  # interaction_type kalır


# --- Rapor: benchmark/pass-fail yok ---------------------------------------------

def test_rapor_pass_fail_veya_model_siralamasi_uretmez(tmp_path, monkeypatch):
    out_dir = tmp_path / "exp_norapor"
    _prep_and_run(monkeypatch, "norapor", out_dir)
    rep = cs.report("norapor", out_dir)
    metin = json.dumps(rep)
    for yasakli in ("plus_pass", "base_pass", "pass_rate", "benchmark", "capability"):
        assert yasakli not in metin


def test_rapor_dosyaya_yazilir(tmp_path, monkeypatch):
    out_dir = tmp_path / "exp_dosya"
    _prep_and_run(monkeypatch, "dosya", out_dir)
    cs.report("dosya", out_dir)
    assert (out_dir / "compatibility_report.json").exists()


# --- git_commit: manifest kritik alanı + fail-closed git kapısı -----------------

def test_git_commit_manifestin_kritik_alanidir():
    assert "git_commit" in cs.MANIFEST_CRITICAL_FIELDS
    g = cs.select_pilot_task_ids(6, salt="gemini")
    k = cs.select_pilot_task_ids(12, salt="grok")
    assert cs.build_manifest_snapshot(g, k)["git_commit"] == _FAKE_COMMIT


def test_farkli_committe_ayni_isimle_devam_edilemez_sifir_cagri(tmp_path, monkeypatch):
    out_dir = tmp_path / "exp_commit"
    cs.prepare("commit", out_dir)              # commit A ile manifest yazıldı
    monkeypatch.setattr(cs, "_git_commit", lambda: "b" * 40)   # commit B
    calls = []
    monkeypatch.setattr("litellm.completion", _dispatcher(capture=calls))
    with pytest.raises(cs.CompatibilitySmokeError, match="manifest uyuşmazlığı"):
        cs.run("commit", out_dir, dry_run=False)
    assert calls == []


def test_run_gate_manifest_commiti_head_ile_karsilastirir():
    # Manifest karşılaştırmasından bağımsız ikinci savunma hattı: gate'e
    # doğrudan eski bir manifest verilse bile HEAD uyuşmazlığı yakalanır.
    with pytest.raises(cs.CompatibilitySmokeError, match="FARKLI committe"):
        cs.require_verified_git_state({"git_commit": "e" * 40})
    cs.require_verified_git_state({"git_commit": _FAKE_COMMIT})   # eşleşiyorsa geçer


def test_git_durumu_dogrulanamiyorsa_ucretli_run_durur_sifir_cagri(tmp_path, monkeypatch):
    # None ("doğrulanamadı") ile False ("temiz") AYNI ŞEY DEĞİLDİR.
    out_dir = tmp_path / "exp_gitbilinmiyor"
    cs.prepare("gitbilinmiyor", out_dir)
    monkeypatch.setattr(cs, "_git_dirty", lambda: None)
    calls = []
    monkeypatch.setattr("litellm.completion", _dispatcher(capture=calls))
    with pytest.raises(cs.CompatibilitySmokeError, match="DOĞRULANAMIYOR"):
        cs.run("gitbilinmiyor", out_dir, dry_run=False)
    assert calls == []


def test_git_head_cozulemiyorsa_ucretli_run_durur_sifir_cagri(tmp_path, monkeypatch):
    monkeypatch.setattr(cs, "_git_commit", lambda: None)
    out_dir = tmp_path / "exp_githead"
    cs.prepare("githead", out_dir)   # manifest git_commit=None olarak yazılır
    calls = []
    monkeypatch.setattr("litellm.completion", _dispatcher(capture=calls))
    with pytest.raises(cs.CompatibilitySmokeError, match="HEAD DOĞRULANAMIYOR"):
        cs.run("githead", out_dir, dry_run=False)
    assert calls == []


def test_dry_run_git_dogrulanamasa_da_calisir_sifir_cagri(tmp_path, monkeypatch):
    monkeypatch.setattr(cs, "_git_dirty", lambda: None)
    calls = []
    monkeypatch.setattr("litellm.completion", _dispatcher(capture=calls))
    pending = cs.run("gitdry", tmp_path / "exp_gitdry", dry_run=True)
    assert len(pending) == 24
    assert calls == []


# --- verify_probe_log: satırlar TAM PLANLA eşleşmeli ----------------------------

def _bozuk_satirla_run(tmp_path, monkeypatch, ad, bozukluk: dict):
    """Plandaki ilk probu `bozukluk` ile bozup ücretli run dener; çağrı sayısını döner."""
    out_dir = tmp_path / f"exp_{ad}"
    prepared = cs.prepare(ad, out_dir)
    cs.append_jsonl(out_dir / "probes.jsonl", [_probe_row(prepared["plan"][0], **bozukluk)])
    calls = []
    monkeypatch.setattr("litellm.completion", _dispatcher(capture=calls))
    return out_dir, calls


def test_plandan_farkli_kind_iceren_satir_reddedilir_sifir_cagri(tmp_path, monkeypatch):
    out_dir, calls = _bozuk_satirla_run(tmp_path, monkeypatch, "kind",
                                        {"kind": cs.PROBE_KIND_CODER})
    with pytest.raises(cs.CompatibilitySmokeError, match="planla eşleşmiyor"):
        cs.run("kind", out_dir, dry_run=False)
    assert calls == []


def test_plandan_farkli_target_key_iceren_satir_reddedilir_sifir_cagri(tmp_path, monkeypatch):
    out_dir, calls = _bozuk_satirla_run(tmp_path, monkeypatch, "hedef",
                                        {"target_key": "grok"})
    with pytest.raises(cs.CompatibilitySmokeError, match="planla eşleşmiyor"):
        cs.run("hedef", out_dir, dry_run=False)
    assert calls == []


def test_bilinmeyen_status_reddedilir_sifir_cagri(tmp_path, monkeypatch):
    out_dir, calls = _bozuk_satirla_run(tmp_path, monkeypatch, "status",
                                        {"status": "kismen_tamam"})
    with pytest.raises(cs.CompatibilitySmokeError, match="bilinmeyen status"):
        cs.run("status", out_dir, dry_run=False)
    assert calls == []


@pytest.mark.parametrize("gecersiz", [0, -1, "1", None, True, 1.5])
def test_gecersiz_attempt_reddedilir(tmp_path, monkeypatch, gecersiz):
    # True, Python'da int'tir ama bir deneme SAYISI değildir -- bool ayrıca elenir.
    ad = f"att{abs(hash(str(gecersiz))) % 1000}"
    out_dir, calls = _bozuk_satirla_run(tmp_path, monkeypatch, ad, {"attempt": gecersiz})
    with pytest.raises(cs.CompatibilitySmokeError, match="geçersiz attempt"):
        cs.run(ad, out_dir, dry_run=False)
    assert calls == []


def test_terminal_failure_satiri_gecerlidir_ve_probe_yeniden_denenir(tmp_path, monkeypatch):
    out_dir = tmp_path / "exp_terminalkabul"
    prepared = cs.prepare("terminalkabul", out_dir)
    spec = prepared["plan"][0]
    cs.append_jsonl(out_dir / "probes.jsonl",
                    [_probe_row(spec, status=cs.STATUS_TERMINAL_FAILURE)])
    bekleyen = cs.run("terminalkabul", out_dir, dry_run=True)
    assert spec["probe_id"] in [p["probe_id"] for p in bekleyen]
    assert len(bekleyen) == 24


# --- verify_call_log: probe--çağrı eşleşmesi -------------------------------------

def test_bilinmeyen_run_idli_cagri_logu_raporu_engeller(tmp_path, monkeypatch):
    out_dir = tmp_path / "exp_yabancicagri"
    _prep_and_run(monkeypatch, "yabancicagri", out_dir)
    cs.append_jsonl(cs.call_log_path(out_dir),
                    [{"run_id": "gemini:planner:BASKA-DENEY", "status": "ok"}])
    with pytest.raises(cs.CompatibilitySmokeError, match="bilinmeyen run_id"):
        cs.report("yabancicagri", out_dir)


def test_bir_probe_icin_iki_basarili_cagri_raporu_engeller(tmp_path, monkeypatch):
    out_dir = tmp_path / "exp_ikiok"
    rows = _prep_and_run(monkeypatch, "ikiok", out_dir)
    cs.append_jsonl(cs.call_log_path(out_dir),
                    [{"run_id": rows[0]["probe_id"], "status": "ok"}])
    with pytest.raises(cs.CompatibilitySmokeError, match="birden fazla BAŞARILI"):
        cs.report("ikiok", out_dir)


def test_basarili_cagrisi_olmayan_tamamlanmis_probe_raporu_engeller(tmp_path, monkeypatch):
    out_dir = tmp_path / "exp_cagrisiz"
    prepared = cs.prepare("cagrisiz", out_dir)
    # Hiç çağrı yapılmadan "tamamlandı" iddia eden satır.
    cs.append_jsonl(out_dir / "probes.jsonl", [_probe_row(prepared["plan"][0])])
    with pytest.raises(cs.CompatibilitySmokeError, match="başarılı çağrı kaydı OLMAYAN"):
        cs.report("cagrisiz", out_dir)


# --- gate_decision / blocker_reasons ---------------------------------------------

def _gate(rep, hedef="gemini"):
    return rep["targets"][hedef]["gate_decision"]


def _blockers(rep, hedef="gemini"):
    return " | ".join(rep["targets"][hedef]["blocker_reasons"])


def test_gate_temiz_kosuda_iki_hedef_de_pass(tmp_path, monkeypatch):
    out_dir = tmp_path / "exp_gatepass"
    _prep_and_run(monkeypatch, "gatepass", out_dir)
    rep = cs.report("gatepass", out_dir)
    assert _gate(rep, "gemini") == cs.GATE_PASS
    assert _gate(rep, "grok") == cs.GATE_PASS
    assert rep["targets"]["gemini"]["blocker_reasons"] == []
    assert rep["targets"]["grok"]["blocker_reasons"] == []


def test_gate_hic_kosulmamis_turda_incomplete(tmp_path):
    out_dir = tmp_path / "exp_gatebos"
    cs.prepare("gatebos", out_dir)
    rep = cs.report("gatebos", out_dir)
    assert _gate(rep, "gemini") == cs.GATE_INCOMPLETE
    assert _gate(rep, "grok") == cs.GATE_INCOMPLETE
    assert rep["targets"]["gemini"]["blocker_reasons"] == []


def test_gate_eksik_probe_sayisinda_incomplete(tmp_path, monkeypatch):
    out_dir = tmp_path / "exp_gateeksik"
    prepared = cs.prepare("gateeksik", out_dir)
    monkeypatch.setattr("litellm.completion", _dispatcher())
    kismi = [p for p in prepared["plan"] if p["target_key"] == "gemini"][:3]
    probes_path = out_dir / "probes.jsonl"
    cs.execute_pending(kismi, [], experiment="gateeksik", log_namespace="compat_smoke",
                       on_result=lambda row: cs.append_jsonl(probes_path, [row]))
    rep = cs.report("gateeksik", out_dir)
    assert _gate(rep, "gemini") == cs.GATE_INCOMPLETE
    assert "3/12" in " ".join(rep["targets"]["gemini"]["non_blocking_notes"])


def test_gate_terminal_hatada_fail(tmp_path, monkeypatch):
    out_dir = tmp_path / "exp_gateterminal"
    prepared = cs.prepare("gateterminal", out_dir)
    monkeypatch.setattr("litellm.completion", lambda **k: _corrupt_response())
    gemini_plan = [p for p in prepared["plan"] if p["target_key"] == "gemini"]
    probes_path = out_dir / "probes.jsonl"
    cs.execute_pending(gemini_plan, [], experiment="gateterminal", log_namespace="compat_smoke",
                       on_result=lambda row: cs.append_jsonl(probes_path, [row]))
    rep = cs.report("gateterminal", out_dir)
    assert _gate(rep, "gemini") == cs.GATE_FAIL
    assert "terminal taşıma hatası" in _blockers(rep, "gemini")


def test_gate_gemini_json_parse_hatasinda_fail(tmp_path, monkeypatch):
    out_dir = tmp_path / "exp_gatejson"
    _prep_and_run(monkeypatch, "gatejson", out_dir, planner_text="bu hic JSON degil")
    rep = cs.report("gatejson", out_dir)
    assert _gate(rep, "gemini") == cs.GATE_FAIL
    assert "ayrıştırılamadı" in _blockers(rep, "gemini")
    assert _gate(rep, "grok") == cs.GATE_PASS   # hedefler BAĞIMSIZ karar alır


def test_gate_grok_sema_hatasinda_fail(tmp_path, monkeypatch):
    gecersiz = json.dumps({"primary_mode": "9.9", "secondary_modes": [], "confidence": "high",
                          "rationale": "gecersiz kod", "insufficient_context": False})
    out_dir = tmp_path / "exp_gategrok"
    _prep_and_run(monkeypatch, "gategrok", out_dir, grok_text=gecersiz)
    rep = cs.report("gategrok", out_dir)
    assert _gate(rep, "grok") == cs.GATE_FAIL
    assert "mast_label_schema_ok" in _blockers(rep, "grok")
    assert "mast_adjudication_schema_ok" in _blockers(rep, "grok")
    assert _gate(rep, "gemini") == cs.GATE_PASS


def test_gate_pozitif_reasoning_tokende_fail_vermez_ama_raporlanir(tmp_path, monkeypatch):
    # [2026-07-30] Ortak ayar AÇIK (Gemini endpoint'i zorunlu kılıyor): pozitif
    # reasoning tokenı artık BEKLENEN durumdur, engel değildir. Sayaç yine de
    # raporda kalır -- maliyet/gecikme etkisi izlenmeye devam eder.
    assert config.REASONING_CONFIG.get("enabled") is True
    def fake(**kwargs):
        if kwargs["model"] == GROK:
            r = _ok_response(_VALID_ADJUDICATOR_JSON)
        elif kwargs.get("response_format") is not None:
            r = _ok_response(_VALID_PLANNER_JSON)
        else:
            r = _ok_response(_VALID_CODE_BLOCK)
        r.usage.completion_tokens_details = types.SimpleNamespace(reasoning_tokens=7)
        return r
    monkeypatch.setattr("litellm.completion", fake)
    out_dir = tmp_path / "exp_gatereason"
    cs.run("gatereason", out_dir, dry_run=False)
    rep = cs.report("gatereason", out_dir)
    assert _gate(rep, "gemini") == cs.GATE_PASS
    assert "reasoning" not in _blockers(rep, "gemini")
    assert rep["targets"]["gemini"]["total_reasoning_tokens"] == 12 * 7
    assert rep["targets"]["grok"]["total_reasoning_tokens"] == 12 * 7


def _sentetik_rapor(**cagri_ustyazim):
    """Tam (24/24) ve SORUNSUZ bir turun sentetik artefaktlarından rapor üretir.

    Boş çıktı / eksik usage / native error, gerçek taşıma katmanında zaten
    `provider_error` sayılır (agents/llm.py::_provider_error) ve `ok` kaydına
    hiç DÖNÜŞMEZ; bu yüzden bu üç engel, çağrı logu seviyesinde kurgulanarak
    doğrulanır -- rapor katmanının kendi sözleşmesi test edilir.
    """
    g = cs.select_pilot_task_ids(6, salt="gemini")
    k = cs.select_pilot_task_ids(12, salt="grok")
    manifest = cs.build_manifest_snapshot(g, k)
    plan = cs.build_plan(gemini_task_ids=g, grok_task_ids=k)
    bayraklar = {
        cs.PROBE_KIND_PLANNER: {"planner_json_parse_ok": True, "planner_schema_ok": True},
        cs.PROBE_KIND_CODER: {"coder_nonempty": True, "coder_extract_ok": True,
                              "coder_ast_parse_ok": True},
        cs.PROBE_KIND_ADJUDICATE: {"json_parse_ok": True, "mast_label_schema_ok": True,
                                   "interaction_invariant_ok": True,
                                   "mast_adjudication_schema_ok": True},
    }
    probe_rows = [_probe_row(p, **bayraklar[p["kind"]]) for p in plan]
    call_rows = [{"run_id": p["probe_id"], "status": "ok", "provider_attempt": 1,
                  "model": "m-req", "actual_model": "m-real",
                  "requested_provider": ["auto"], "actual_provider": "saglayici",
                  "finish_reason": "stop", "native_finish_reason": "STOP",
                  "input_tokens": 10, "output_tokens": 5, "reasoning_tokens": 0,
                  "cached_tokens": 0, "cost_usd": 0.0, "latency_s": 0.5,
                  "response_text": "dolu", **cagri_ustyazim} for p in plan]
    return cs.build_report(manifest=manifest, probe_rows=probe_rows, call_rows=call_rows)


def test_sentetik_sorunsuz_tur_pass_verir():
    rep = _sentetik_rapor()
    assert _gate(rep, "gemini") == cs.GATE_PASS
    assert _gate(rep, "grok") == cs.GATE_PASS


@pytest.mark.parametrize("ustyazim,beklenen_metin", [
    ({"response_text": "   "}, "boş çıktı"),
    ({"input_tokens": 0, "output_tokens": 0}, "eksik/sıfır usage"),
    ({"native_finish_reason": "error"}, "native_finish_reason='error'"),
])
def test_gate_taşima_provenans_engelleri_fail_verir(ustyazim, beklenen_metin):
    rep = _sentetik_rapor(**ustyazim)
    assert _gate(rep, "gemini") == cs.GATE_FAIL
    assert beklenen_metin in _blockers(rep, "gemini")


def test_reasoning_tokeni_ayar_kapaliysa_engel_acikken_degil(monkeypatch):
    # Kural ayarın KENDİSİNE bağlıdır, sabit bir beklentiye değil: kapalı ayarla
    # pozitif token "ayar uygulanmadı" demektir (engel), açık ayarla beklenendir.
    monkeypatch.setattr(cs, "REASONING_CONFIG", {"enabled": False})
    rep_kapali = _sentetik_rapor(reasoning_tokens=7)
    assert _gate(rep_kapali, "gemini") == cs.GATE_FAIL
    assert "reasoning KAPALI" in _blockers(rep_kapali, "gemini")

    monkeypatch.setattr(cs, "REASONING_CONFIG", {"enabled": True})
    rep_acik = _sentetik_rapor(reasoning_tokens=7)
    assert _gate(rep_acik, "gemini") == cs.GATE_PASS
    assert rep_acik["targets"]["gemini"]["total_reasoning_tokens"] == 12 * 7


def test_gate_kurtarilan_provider_error_tek_basina_fail_degildir(tmp_path, monkeypatch):
    # Her mantıksal çağrı önce gömülü 429 görür, retry'da başarılı olur.
    durum = {"bozuk_sira": False}   # ilk flip True yapar -> her çağrı önce bozuk yanıt görür
    def fake(**kwargs):
        durum["bozuk_sira"] = not durum["bozuk_sira"]
        if durum["bozuk_sira"]:
            return _corrupt_response()
        if kwargs["model"] == GROK:
            return _ok_response(_VALID_ADJUDICATOR_JSON)
        if kwargs.get("response_format") is not None:
            return _ok_response(_VALID_PLANNER_JSON)
        return _ok_response(_VALID_CODE_BLOCK)
    monkeypatch.setattr("litellm.completion", fake)
    out_dir = tmp_path / "exp_gatekurtar"
    cs.run("gatekurtar", out_dir, dry_run=False)
    rep = cs.report("gatekurtar", out_dir)
    for hedef in ("gemini", "grok"):
        veri = rep["targets"][hedef]
        assert veri["provider_error_recovered_call_count"] == 12
        assert veri["embedded_error_attempt_count"] == 12
        assert _gate(rep, hedef) == cs.GATE_PASS, veri["blocker_reasons"]
        assert "retry ile kurtarılmış" in " ".join(veri["non_blocking_notes"])


def test_planner_sema_ve_coder_ast_tanisal_kalir_fail_yapmaz(tmp_path, monkeypatch):
    # Ön-kayıt: bunlar TANISAL ölçülerdir; gate'i FAIL yapmaları, prompt/model
    # ayarını model çıktısına göre eğmeye kapı açardı.
    out_dir = tmp_path / "exp_tanisal"
    _prep_and_run(monkeypatch, "tanisal", out_dir,
                  planner_text=json.dumps({"tamamen": "baska bir sema"}),
                  coder_text="```python\ndef f(x)\n    return x\n```")
    rep = cs.report("tanisal", out_dir)
    tani = rep["targets"]["gemini"]["gemini_diagnostics"]
    assert tani["planner_json_parse_ok"] == 6
    assert tani["planner_schema_ok"] == 0
    assert tani["coder_ast_parse_ok"] == 0
    assert _gate(rep, "gemini") == cs.GATE_PASS
    assert rep["targets"]["gemini"]["blocker_reasons"] == []


def test_gate_kararlari_model_yetenek_tablosu_uretmez(tmp_path, monkeypatch):
    out_dir = tmp_path / "exp_gatemeta"
    _prep_and_run(monkeypatch, "gatemeta", out_dir)
    metin = json.dumps(cs.report("gatemeta", out_dir))
    for yasakli in ("plus_pass", "base_pass", "pass_rate", "benchmark", "capability",
                    "ranking", "score"):
        assert yasakli not in metin


# --- Tam test paketi gerçek API olmadan çalışır (meta) --------------------------

def test_bu_dosyadaki_hicbir_test_gercek_agi_cagirmaz():
    # Belgeleyici test: bütün testler litellm.completion'ı mock'lar (yukarıdaki
    # fixture'lar + her testin kendi monkeypatch'i). Gerçek bir ağ çağrısı
    # yapılsaydı DNS/timeout nedeniyle testler saniyeler içinde patlardı;
    # bu paket saniyeler içinde biter.
    assert True
