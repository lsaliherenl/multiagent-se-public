"""Takip çalışmaları (Study 1B / Study 2) — LLM'siz, Docker'sız, deterministik.

Dört şeyi kilitler (EXPERIMENT_PROTOCOL.md §13):

1. **Makale koşularıyla kimlik eşitliği.** `reproduction/paper_run_identity.json`
   makaledeki held-out koşuların manifestlerinden kopyalanmıştır. Kodun
   belirlediği her alan (prompt sözleşmesi, istek politikası, profil, rota,
   görev dosyaları, seçim parmak izi) bu depodan AYNI değerle üretilmelidir.
2. **Çalışma kapıları.** Study 2 setleri açık `--study` olmadan, Luna Study 1A
   verisi olarak, judge/adjudicator hiçbir çalışmada koşamaz.
3. **İstek gövdesi ve retry katmanları.** Luna isteğinde `temperature` anahtarı
   hiç bulunmaz; takip koşularında transport retry görünür döngüdedir.
4. **Takip analizi kapıları.** Rejimler havuzlanmaz, çözülmemiş run_error
   geçilemez, moderasyon bağımsız ve köprü eşlenik örneklenir, p-değeri yoktur.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import agents.llm as llm_module
import config
from analysis import analyze as az
from analysis import followup
from eval import runner
from eval.result_schema import make_run_error_record, make_synthetic_record

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = json.loads(
    (ROOT / "reproduction" / "paper_run_identity.json").read_text(encoding="utf-8"))
CELLS = REFERENCE["cells"]
FOLLOWUP_CELLS = {k: v for k, v in CELLS.items() if v["study_id"] != "study1a"}


# --------------------------------------------------------------------------
# 1. Makale koşularıyla kimlik eşitliği
# --------------------------------------------------------------------------

def test_referans_dosyasi_alti_hucreyi_tasir_ve_sonuc_icermez():
    assert set(CELLS) == {"study1a_gemini", "study1a_deepseek", "study1b_gemini",
                          "study1b_luna", "study2_gemini", "study2_luna"}
    metin = json.dumps(REFERENCE)
    for yasak in ("plus_pass", "base_pass", "point_estimate", "pass_count"):
        assert yasak not in metin


@pytest.mark.parametrize("cell", sorted(CELLS))
def test_prompt_ve_sema_sozlesmesi_makaledekiyle_ayni(cell):
    assert runner._prompt_contract_hash() == CELLS[cell]["prompt_contract_hash"]


@pytest.mark.parametrize("cell", sorted(CELLS))
def test_gorev_dosyalari_makaledekiyle_bayt_olarak_ayni(cell):
    ref = CELLS[cell]
    assert runner._hash_task_files(ref["task_ids"], ref["task_set"]) == ref["task_file_hashes"]
    assert len(ref["task_ids"]) == 50 and ref["repeats"] == 3
    assert ref["arm_order"] == config.ALL_ARMS


@pytest.mark.parametrize("cell", sorted(CELLS))
def test_uretim_parametreleri_makaledekiyle_ayni(cell):
    ref = CELLS[cell]
    assert ref["temperature"] == config.DEFAULT_TEMPERATURE
    assert ref["max_tokens"] == config.MAX_OUTPUT_TOKENS
    assert ref["reasoning_config"] == config.REASONING_CONFIG
    assert ref["provider_routing"] == config.provider_routing_for(ref["model"])
    assert ref["arm_rotation_scheme"] == config.ARM_ROTATION_SCHEME_VERSION
    assert ref["result_schema_version"] == config.RESULT_SCHEMA_VERSION
    assert ref["llm_call_schema_version"] == config.LLM_CALL_SCHEMA_VERSION
    beklenen = (runner._heldout_selection_fingerprint()
                if ref["task_set"] in config.STUDY1A_TASK_SETS else None)
    assert ref["heldout_selection_fingerprint"] == beklenen


@pytest.mark.parametrize("cell", sorted(FOLLOWUP_CELLS))
def test_takip_kimligi_makaledekiyle_ayni(cell):
    ref = FOLLOWUP_CELLS[cell]
    identity = config.validate_run_identity(
        task_set=ref["task_set"], model=ref["model_requested"], study=ref["study_id"])
    assert identity.model == ref["model"]
    assert identity.protocol_version == ref["protocol_version"]
    assert identity.task_regime == ref["task_regime"]
    assert identity.profile_fingerprint == ref["profile_fingerprint"]
    assert config.request_policy_fingerprint(ref["model"]) == ref["request_policy_fingerprint"]
    assert config.effective_request_policy(ref["model"])["temperature_policy"] \
        == ref["temperature_policy"]
    assert ref["followup_litellm_num_retries"] == config.FOLLOWUP_LITELLM_NUM_RETRIES
    assert ref["transport_attempts_per_provider_attempt"] \
        == config.TRANSPORT_ATTEMPTS_PER_PROVIDER_ATTEMPT
    assert ref["provider_attempts_per_logical_call"] \
        == config.PROVIDER_ATTEMPTS_PER_LOGICAL_CALL
    assert runner._task_selection_fingerprint(ref["task_set"]) \
        == ref["task_selection_fingerprint"]


def test_study2_imaj_kimligi_referansta_kayitli():
    for cell in ("study2_gemini", "study2_luna"):
        assert CELLS[cell]["container_image_digest"] == config.BIGCODEBENCH_PAPER_IMAGE_ID
    for cell in ("study1b_gemini", "study1b_luna"):
        assert CELLS[cell]["container_image_digest"] is None


@pytest.mark.parametrize("cell", sorted(FOLLOWUP_CELLS))
def test_manifest_anlik_goruntusu_referans_alanlarini_uretir(cell):
    ref = FOLLOWUP_CELLS[cell]
    identity = config.validate_run_identity(
        task_set=ref["task_set"], model=ref["model_requested"], study=ref["study_id"])
    evaluator = {"evaluation_backend": "x",
                 "container_image_digest": ref["container_image_digest"]}
    snap = runner.build_manifest_snapshot(
        name="t", identity=identity, arms=list(config.ALL_ARMS), repeats=3,
        task_ids=ref["task_ids"], evaluator=evaluator)
    for key, value in ref.items():
        assert snap[key] == value, key


# --------------------------------------------------------------------------
# 2. Çalışma kapıları ve manifest
# --------------------------------------------------------------------------

def test_study1a_manifesti_tarihsel_bicimini_korur():
    identity = config.validate_run_identity(task_set="heldout", model="main")
    snap = runner.build_manifest_snapshot(
        name="t", identity=identity, arms=list(config.ALL_ARMS), repeats=3,
        task_ids=CELLS["study1a_gemini"]["task_ids"],
        evaluator={"evaluation_backend": "evalplus_base_plus_v1",
                   "container_image_digest": None})
    assert "study_id" not in snap
    assert set(runner.LEGACY_CRITICAL_FIELDS) <= set(snap)


def test_legacy_dizin_takip_kimligiyle_devralinamaz(tmp_path):
    legacy = runner.build_manifest_snapshot(
        name="t", identity=config.validate_run_identity(task_set="heldout", model="main"),
        arms=list(config.ALL_ARMS), repeats=3, task_ids=["a"],
        evaluator={"evaluation_backend": "e", "container_image_digest": None})
    runner.check_or_write_manifest(tmp_path, legacy)
    followup_snap = runner.build_manifest_snapshot(
        name="t", identity=config.validate_run_identity(
            task_set="heldout", model="main", study="study1b"),
        arms=list(config.ALL_ARMS), repeats=3, task_ids=["a"],
        evaluator={"evaluation_backend": "e", "container_image_digest": None})
    followup_snap["git_commit"] = legacy["git_commit"]
    with pytest.raises(SystemExit):
        runner.check_or_write_manifest(tmp_path, followup_snap)


def test_takip_dizini_farkli_imajla_surdurulemez(tmp_path):
    identity = config.validate_run_identity(task_set="study2_complex", model="main",
                                            study="study2")
    def snap(digest):
        return runner.build_manifest_snapshot(
            name="t", identity=identity, arms=list(config.ALL_ARMS), repeats=3,
            task_ids=["a"], evaluator={"evaluation_backend": "e",
                                       "container_image_digest": digest})
    first = snap("sha256:" + "a" * 64)
    runner.check_or_write_manifest(tmp_path, first)
    second = snap("sha256:" + "b" * 64)
    second["git_commit"] = first["git_commit"]
    with pytest.raises(SystemExit) as exc:
        runner.check_or_write_manifest(tmp_path, second)
    assert "container_image_digest" in str(exc.value)


@pytest.mark.parametrize("argv", [
    ["--task-set", "study2_complex", "--model", "main"],             # --study yok
    ["--task-set", "followup_dev", "--model", "main"],
    ["--task-set", "heldout", "--model", "followup_secondary"],       # Luna Study 1A değil
    ["--task-set", "heldout", "--model", "secondary", "--study", "study1b"],  # DeepSeek takipte yok
    ["--task-set", "study2_complex", "--model", "openrouter/minimax/minimax-m3",
     "--study", "study2"],
    ["--task-set", "heldout", "--model", "main", "--study", "study2"],  # yanlış set
])
def test_runner_kimlik_kapisi_api_anahtarindan_once_durur(monkeypatch, argv):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(SystemExit) as exc:
        runner.main(["--name", "x", *argv])
    assert "API anahtarı" not in str(exc.value)


def test_runner_study_secenekleri_yalniz_takip_calismalari(capsys):
    with pytest.raises(SystemExit):
        runner.main(["--name", "x", "--task-set", "heldout", "--model", "main",
                     "--study", "study1a"])
    assert "invalid choice" in capsys.readouterr().err


def test_evalplus_degerlendirici_on_kosulu_docker_istemez():
    tasks = [{"task_id": "humanevalplus_002"}]
    assert runner.evaluator_preflight(tasks) == {
        "evaluation_backend": "evalplus_base_plus_v1", "container_image_digest": None}


def test_bigcodebench_on_kosulu_imaji_ve_kaynak_cildini_dogrular(monkeypatch):
    from eval import bigcodebench_backend as bcb
    calls = []
    monkeypatch.setattr(bcb, "_image_identity",
                        lambda docker: calls.append("image") or "sha256:" + "c" * 64)
    monkeypatch.setattr(bcb, "verify_resource_volume",
                        lambda docker: calls.append("resource") or {})
    tasks = [{"task_id": "bigcodebench_0013",
              "evaluation_backend": "bigcodebench_untrusted_check_v1"}]
    out = runner.evaluator_preflight(tasks)
    assert out == {"evaluation_backend": "bigcodebench_untrusted_check_v1",
                   "container_image_digest": "sha256:" + "c" * 64}
    assert calls == ["image", "resource"]


def test_karisik_degerlendiricili_gorev_seti_reddedilir():
    with pytest.raises(ValueError):
        runner.evaluator_preflight([
            {"task_id": "humanevalplus_002"},
            {"task_id": "bigcodebench_0013",
             "evaluation_backend": "bigcodebench_untrusted_check_v1"}])


@pytest.mark.parametrize("task_set,count", [("followup_dev", 16), ("study2_complex", 50)])
def test_study2_gorev_setleri_tam_ve_etiketli(task_set, count):
    from eval.harness import load_all_tasks
    tasks = load_all_tasks(task_set)
    assert len(tasks) == count
    assert {t["evaluation_backend"] for t in tasks} == {"bigcodebench_untrusted_check_v1"}
    manifest = json.loads((config.TASK_SETS[task_set] / "_selection_manifest.json")
                          .read_text(encoding="utf-8"))
    assert manifest["task_count"] == count
    assert manifest["canonical_solution_in_task_files"] is False
    hashes = runner._hash_task_files([t["task_id"] for t in tasks], task_set)
    assert hashes == manifest["task_file_sha256"]


# --------------------------------------------------------------------------
# 3. İstek gövdesi ve retry katmanları
# --------------------------------------------------------------------------

def _yanit():
    choice = SimpleNamespace(message=SimpleNamespace(content="ok"),
                             finish_reason="stop", logprobs=None)
    return SimpleNamespace(choices=[choice],
                           usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1))


@pytest.fixture
def _sessiz_llm(monkeypatch, tmp_path):
    monkeypatch.setattr(llm_module, "_throttle", lambda: None)
    monkeypatch.setattr(llm_module.time, "sleep", lambda s: None)
    monkeypatch.setattr(llm_module, "LOGS_DIR", tmp_path)
    monkeypatch.setattr(llm_module, "CALL_LOG", tmp_path / "llm_calls.jsonl")
    monkeypatch.setattr("litellm.completion_cost", lambda **k: 0.0)


def test_luna_isteginde_temperature_anahtari_yok(monkeypatch, _sessiz_llm, tmp_path):
    istekler = []
    monkeypatch.setattr("litellm.completion", lambda **k: istekler.append(k) or _yanit())
    llm_module.call_model([{"role": "user", "content": "x"}],
                          model=config.MODEL_FOLLOWUP_SECONDARY)
    llm_module.call_model([{"role": "user", "content": "x"}], model=config.MODEL_MAIN)
    assert "temperature" not in istekler[0]
    assert istekler[1]["temperature"] == config.DEFAULT_TEMPERATURE
    assert istekler[0]["extra_body"]["provider"] == config.FOLLOWUP_LUNA_PROVIDER_ROUTING
    kayit = [json.loads(l) for l in
             (tmp_path / "llm_calls.jsonl").read_text(encoding="utf-8").splitlines()]
    assert kayit[0]["temperature"] is None and kayit[1]["temperature"] == 0.2


def test_takip_yolunda_transport_retry_gorunur_dongude(monkeypatch, _sessiz_llm):
    istekler, hatalar = [], [RuntimeError("502"), RuntimeError("502")]
    def fake(**k):
        istekler.append(k)
        if hatalar:
            raise hatalar.pop(0)
        return _yanit()
    monkeypatch.setattr("litellm.completion", fake)
    with llm_module.followup_transport():
        llm_module.call_model([{"role": "user", "content": "x"}], model=config.MODEL_MAIN)
    assert len(istekler) == 3
    assert {k["num_retries"] for k in istekler} == {config.FOLLOWUP_LITELLM_NUM_RETRIES}


def test_takip_yolunda_transport_tukenince_istisna_yukselir(monkeypatch, _sessiz_llm):
    sayac = []
    def fake(**k):
        sayac.append(1)
        raise RuntimeError("down")
    monkeypatch.setattr("litellm.completion", fake)
    with llm_module.followup_transport(), pytest.raises(RuntimeError):
        llm_module.call_model([{"role": "user", "content": "x"}], model=config.MODEL_MAIN)
    assert len(sayac) == config.TRANSPORT_ATTEMPTS_PER_PROVIDER_ATTEMPT


def test_study1a_yolu_litellm_retryini_kullanir(monkeypatch, _sessiz_llm):
    istekler = []
    monkeypatch.setattr("litellm.completion", lambda **k: istekler.append(k) or _yanit())
    llm_module.call_model([{"role": "user", "content": "x"}], model=config.MODEL_MAIN)
    assert istekler[0]["num_retries"] == config.LLM_NUM_RETRIES
    assert llm_module._FOLLOWUP_TRANSPORT.get() is False


@pytest.mark.parametrize("model", [None, "", "followup_secondary"])
def test_cozulmemis_model_saglayiciya_ulasamaz(monkeypatch, _sessiz_llm, model):
    monkeypatch.setattr("litellm.completion",
                        lambda **k: pytest.fail("çağrı yapılmamalıydı"))
    with pytest.raises(ValueError):
        llm_module.call_model([{"role": "user", "content": "x"}], model=model)


def test_runner_takip_kosusunu_takip_baglaminda_yurutur(monkeypatch):
    gozlenen = []
    monkeypatch.setattr(runner, "_execute",
                        lambda *a, **k: gozlenen.append(llm_module._FOLLOWUP_TRANSPORT.get()) or {})
    runner.execute_run("baseline", {"task_id": "t"}, 0, "m", {}, experiment="e",
                       run_id="r", arm_position=0, task_set="heldout", followup=True)
    runner.execute_run("baseline", {"task_id": "t"}, 0, "m", {}, experiment="e",
                       run_id="r", arm_position=0, task_set="heldout")
    assert gozlenen == [True, False]


# --------------------------------------------------------------------------
# 4. Takip analizi
# --------------------------------------------------------------------------

ARMS = list(config.ALL_ARMS)


def _kayitlar(task_set, model, gecis, task_ids, repeats=3):
    """gecis(task_id, arm, repeat) -> bool."""
    return [make_synthetic_record(model=model, task_set=task_set, arm=arm,
                                  task_id=t, repeat=r, plus_pass=gecis(t, arm, r))
            for r in range(repeats) for t in task_ids for arm in ARMS]


def _manifest(task_set, task_ids, study_id, repeats=3):
    return {"task_set": task_set, "task_ids": list(task_ids), "arm_order": ARMS,
            "repeats": repeats, "study_id": study_id}


def _hucre(task_set, model, gecis, task_ids):
    records = _kayitlar(task_set, model, gecis, task_ids)
    rates = {m: az.task_level_rates(records, task_ids=task_ids, arms=ARMS, metric=m)
             for m in config.ANALYSIS_METRICS}
    return {"model": model, "_task_rates": rates}


def test_rejimler_havuzlanamaz():
    karisik = (_kayitlar("heldout", "m", lambda *a: True, ["a"])
               + _kayitlar("study2_complex", "m", lambda *a: True, ["b"]))
    with pytest.raises(followup.FollowupAnalysisError):
        followup.assert_single_regime(karisik)


def test_manifest_ve_kayit_rejimi_celisirse_durur():
    kayit = _kayitlar("heldout", "m", lambda *a: True, ["a"])
    with pytest.raises(followup.FollowupAnalysisError):
        followup.assert_single_regime(kayit, _manifest("heldout", ["a"], "study2"))


def test_cozulmemis_run_error_gecilemez():
    kayit = _kayitlar("heldout", "m", lambda *a: True, ["a"])
    kayit = [r for r in kayit if not (r["arm"] == "naive" and r["repeat"] == 0)]
    kayit.append(make_run_error_record(experiment="e", model="m", task_set="heldout",
                                       arm="naive", task_id="a", repeat=0, run_id="r",
                                       arm_position=0, error="x"))
    with pytest.raises(followup.FollowupAnalysisError):
        followup.assert_no_unresolved_run_error(kayit, _manifest("heldout", ["a"], "study1b"))


def test_moderasyon_bagimsiz_kopru_eslenik_orneklenir():
    a = [0.0, 1.0, 0.0, 1.0]
    b = [1.0, 0.0, 1.0, 0.0]
    # Eşlenik çekimde aynı indeksler iki vektöre uygulanır: b = 1 - a ise fark
    # her çekimde tam olarak 1 - 2*mean(a[idx]) olur.
    kopru = followup.coupled_bridge_draws(a, b, iterations=50, seed=1)
    bagimsiz = followup.independent_moderation_draws(b, a, iterations=50, seed=1)
    assert len(kopru) == len(bagimsiz) == 50
    assert kopru != bagimsiz
    with pytest.raises(followup.FollowupAnalysisError):
        followup.coupled_bridge_draws([0.0], [0.0, 1.0], iterations=5, seed=1)


def test_capraz_hucre_tahminleri_hiyerarsisi_ve_p_degeri_yok():
    ids1 = [f"h{i}" for i in range(6)]
    ids2 = [f"b{i}" for i in range(6)]
    contract_iyi = lambda t, arm, r: arm == "contract" or (int(t[1:]) % 2 == 0)
    cells = {
        "study1a_gemini": _hucre("heldout", "g", contract_iyi, ids1),
        "study1b_gemini": _hucre("heldout", "g", contract_iyi, ids1),
        "study1b_luna": _hucre("heldout", "l", contract_iyi, ids1),
        "study2_gemini": _hucre("study2_complex", "g", contract_iyi, ids2),
        "study2_luna": _hucre("study2_complex", "l", contract_iyi, ids2),
    }
    out = followup.cross_cell_estimands(cells, iterations=200, seed=7)
    assert out["primary_confirmatory"]["estimand_id"] == "S2_DIRECT_GEMINI"
    assert out["primary_confirmatory"]["point_estimate"] == pytest.approx(0.5)
    assert out["replication"]["pooled_with_gemini"] is False
    assert set(out["moderation"]) == {"gemini", "luna"}
    assert out["bridge"]["point_estimate"] == pytest.approx(0.0)
    assert out["skipped"] == []
    followup.assert_no_p_value(out)


def test_eksik_hucreye_bagli_tahmin_uretilmez():
    ids = ["b0", "b1"]
    cells = {"study2_gemini": _hucre("study2_complex", "g", lambda *a: True, ids)}
    out = followup.cross_cell_estimands(cells, iterations=20, seed=1)
    assert out["replication"] is None and out["bridge"] is None
    assert "S2_DIRECT_LUNA" in out["skipped"]
    assert "S1A_VS_S1B_GEMINI_PAIRED" in out["skipped"]


def test_p_degeri_alani_reddedilir():
    with pytest.raises(followup.FollowupAnalysisError):
        followup.assert_no_p_value({"x": {"p_value": 0.03}})


# --------------------------------------------------------------------------
# 5. BigCodeBench çalışma ortamı tarifi
# --------------------------------------------------------------------------

def _runtime_script():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "bigcodebench_runtime", ROOT / "scripts" / "bigcodebench_runtime.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_requirements_baytlari_sabitlenmis_hashle_ayni():
    _runtime_script().check_requirements_bytes()


def test_dockerfile_config_sabitleriyle_tutarli():
    metin = (ROOT / "docker" / "bigcodebench" / "Dockerfile").read_text(encoding="utf-8")
    assert f"FROM {config.BIGCODEBENCH_BASE_IMAGE}" in metin
    assert config.BIGCODEBENCH_FROZEN_COMMIT in metin
    assert config.BIGCODEBENCH_REQUIREMENTS_SHA256 in metin
    assert f"USER {config.BIGCODEBENCH_RUN_USER}" in metin
    assert config.BIGCODEBENCH_OUTPUT_DIR in metin


def test_runtime_betigi_model_cagirmaz():
    metin = (ROOT / "scripts" / "bigcodebench_runtime.py").read_text(encoding="utf-8")
    for yasak in ("litellm", "call_model", "OPENROUTER"):
        assert yasak not in metin
