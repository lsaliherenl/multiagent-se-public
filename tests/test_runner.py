"""Runner birim testleri — LLM'siz: resume, plan sırası/rotasyon, manifest koruması."""

import json
import subprocess

import pytest

from config import (
    ALL_ARMS,
    ARM_ROTATION_SCHEME_VERSION,
    LLM_CALL_SCHEMA_VERSION,
    REASONING_CONFIG,
    RESULT_SCHEMA_VERSION,
)
from eval.result_schema import make_run_error_record, make_synthetic_record
from eval.runner import (
    _heldout_selection_fingerprint,
    _prompt_contract_hash,
    _require_clean_tree,
    build_run_plan,
    check_or_write_manifest,
    load_completed,
    position_balance,
)


def _write_results(path, records):
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")


MODEL = "test/model"


def test_load_completed_run_error_haric(tmp_path):
    results = tmp_path / "results.jsonl"
    _write_results(results, [
        make_synthetic_record(model=MODEL, arm="baseline", task_id="t1", repeat=0),
        make_synthetic_record(model=MODEL, arm="naive", task_id="t1", repeat=0,
                              base_pass=False, plus_pass=False),
        make_run_error_record(experiment="x", model=MODEL, task_set="heldout",
                              arm="contract", task_id="t1", repeat=0,
                              run_id="r", arm_position=0, error="boom"),
    ])
    completed = load_completed(results)
    assert (MODEL, "baseline", "t1", 0) in completed
    assert (MODEL, "naive", "t1", 0) in completed          # failed = tamamlanmış ölçüm
    assert (MODEL, "contract", "t1", 0) not in completed   # run_error yeniden denenir


def test_load_completed_dosya_yoksa_bos(tmp_path):
    assert load_completed(tmp_path / "yok.jsonl") == set()


def test_resume_anahtari_modele_bagli(tmp_path):
    # Aynı deney dizininde farklı modelle koşulmuş kayıt, yeni modelin
    # koşusunu ATLATMAMALI -- yoksa ikinci modelin verisi eksik toplanır.
    results = tmp_path / "results.jsonl"
    _write_results(results, [
        make_synthetic_record(model="model/A", arm="baseline", task_id="t1", repeat=0),
    ])
    completed = load_completed(results)
    plan = build_run_plan(["t1"], ["baseline"], 1, completed, "model/B")
    assert plan == [(0, "t1", "baseline", 0)], "farklı model için koşu planlanmalıydı"
    assert build_run_plan(["t1"], ["baseline"], 1, completed, "model/A") == []


def test_plan_sirasi_tekrar_gorev_kol():
    plan = build_run_plan(["t1", "t2"], ["baseline", "naive"], 2, set(), MODEL)
    assert len(plan) == 8
    # position = rotasyonun bu (task,repeat) için ürettiği 0-index konum
    assert plan[:4] == [(0, "t1", "baseline", 0), (0, "t1", "naive", 1),
                        (0, "t2", "naive", 0), (0, "t2", "baseline", 1)]
    assert plan[4][0] == 1


def test_plan_rotasyonu_gorev_indeksine_gore_degisir():
    # Aynı tekrar geçişi İÇİNDE ardışık görevler farklı kol sırası görmeli
    # (sadece tekrarlar arasında değil) — karşı-dengelemenin asıl amacı.
    plan = build_run_plan(["t1", "t2", "t3"], ["baseline", "naive", "contract"], 1, set(), MODEL)
    orders = {}
    for rep, task_id, arm, position in plan:
        orders.setdefault(task_id, [None, None, None])[position] = arm
    assert orders["t1"] == ["baseline", "naive", "contract"]
    assert orders["t2"] == ["naive", "contract", "baseline"]
    assert orders["t3"] == ["contract", "baseline", "naive"]


def test_dort_kollu_rotasyon_dengeli():
    # Ana tasarım: 4 kol. Her kol her konumu yaklaşık eşit sayıda görmeli;
    # aksi halde sıra etkisi kollardan birine sistematik yüklenir.
    balance = position_balance([f"t{i}" for i in range(50)], ALL_ARMS, 3)
    assert set(balance) == set(ALL_ARMS)
    for arm, counts in balance.items():
        assert sum(counts) == 150, arm
        # 150 / 4 = 37.5 -> tam bölünmüyor, kalıntı ±1'i aşmamalı
        assert max(counts) - min(counts) <= 1, f"{arm}: {counts}"


def test_plan_tamamlananlari_atlar():
    completed = {(MODEL, "baseline", "t1", 0), (MODEL, "naive", "t2", 1)}
    plan = build_run_plan(["t1", "t2"], ["baseline", "naive"], 2, completed, MODEL)
    assert len(plan) == 6
    remaining = {(rep, task_id, arm) for rep, task_id, arm, _ in plan}
    assert (0, "t1", "baseline") not in remaining
    assert (1, "t2", "naive") not in remaining


SNAPSHOT = {
    "name": "x", "created_ts": "t", "model": "m", "temperature": 0.2,
    # Ortak üretim parametreleri config'ten okunur: sabit bir kopya, config
    # değiştiğinde (ör. 2026-07-30 reasoning kararı) sessizce eskir.
    "max_tokens": 8192, "reasoning_config": REASONING_CONFIG,
    "provider_routing": {"require_parameters": True},
    "arm_order": ["baseline", "naive"], "repeats": 1,
    "task_set": "heldout", "task_ids": ["t1"],
    "arm_rotation_scheme": ARM_ROTATION_SCHEME_VERSION,
    "result_schema_version": RESULT_SCHEMA_VERSION,
    "llm_call_schema_version": LLM_CALL_SCHEMA_VERSION,
    "heldout_selection_fingerprint": "abc123",
    "prompt_contract_hash": "def456",
    "python_version": "3.12.0", "platform": "TestOS",
    "git_commit": None, "task_file_hashes": {"t1.json": "abc"}, "uv_lock_hash": "def",
}


def test_manifest_ilk_yazim(tmp_path):
    check_or_write_manifest(tmp_path, SNAPSHOT)
    saved = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert saved["model"] == "m"


def test_manifest_ayni_config_devam_eder_ve_repeats_buyur(tmp_path):
    check_or_write_manifest(tmp_path, SNAPSHOT)
    check_or_write_manifest(tmp_path, {**SNAPSHOT, "repeats": 3})
    saved = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert saved["repeats"] == 3                              # büyüdü
    assert saved["arm_order"] == SNAPSHOT["arm_order"]         # kritik alan, değişmedi


def test_manifest_farkli_model_durdurur(tmp_path):
    check_or_write_manifest(tmp_path, SNAPSHOT)
    with pytest.raises(SystemExit):
        check_or_write_manifest(tmp_path, {**SNAPSHOT, "model": "BASKA"})


def test_manifest_farkli_arm_sirasi_durdurur(tmp_path):
    # Rotasyon formülü sıraya bağımlı -- aynı kol SETİ farklı SIRAYLA bile
    # kritik uyuşmazlık olmalı (kol seti/sırası deney ortasında genişletilemez).
    check_or_write_manifest(tmp_path, SNAPSHOT)
    with pytest.raises(SystemExit):
        check_or_write_manifest(tmp_path, {**SNAPSHOT, "arm_order": ["naive", "baseline"]})


def test_manifest_farkli_rotasyon_semasi_durdurur(tmp_path):
    check_or_write_manifest(tmp_path, SNAPSHOT)
    with pytest.raises(SystemExit):
        check_or_write_manifest(tmp_path, {**SNAPSHOT, "arm_rotation_scheme": "farkli_sema_v2"})


def test_manifest_farkli_gorev_hash_durdurur(tmp_path):
    check_or_write_manifest(tmp_path, SNAPSHOT)
    with pytest.raises(SystemExit):
        check_or_write_manifest(tmp_path, {**SNAPSHOT, "task_file_hashes": {"t1.json": "degisti"}})


def test_manifest_gorev_seti_degisirse_durdurur(tmp_path):
    # Pilot ve held-out setlerin aynı deneyde karışması, sonucu sessizce
    # geçersiz kılardı (§5.1) -> task_set kritik alan.
    check_or_write_manifest(tmp_path, SNAPSHOT)
    with pytest.raises(SystemExit):
        check_or_write_manifest(tmp_path, {**SNAPSHOT, "task_set": "pilot"})


@pytest.mark.parametrize("alan", ["max_tokens", "reasoning_config"])
def test_manifest_ortak_model_parametresi_degisirse_durdurur(tmp_path, alan):
    # Ortak üretim parametreleri deney ortasında değişemez: kollar farklı
    # koşullarda koşmuş olurdu (iç geçerlilik). max_tokens/reasoning_config
    # bu yüzden kritik alan.
    check_or_write_manifest(tmp_path, SNAPSHOT)
    with pytest.raises(SystemExit):
        check_or_write_manifest(tmp_path, {**SNAPSHOT, alan: "DEGISTI"})


@pytest.mark.parametrize("alan", [
    "provider_routing", "llm_call_schema_version",
    "heldout_selection_fingerprint", "prompt_contract_hash",
    "python_version", "platform",
])
def test_manifest_provenans_alani_degisirse_durdurur(tmp_path, alan):
    # Bunların hepsi deney ortasında değişirse kayıtlar farklı koşullarda
    # üretilmiş olur; prompt hash'i özellikle önemli çünkü bir kelimenin
    # değişmesi kolların farklı talimatla koşması demektir.
    check_or_write_manifest(tmp_path, SNAPSHOT)
    with pytest.raises(SystemExit):
        check_or_write_manifest(tmp_path, {**SNAPSHOT, alan: "DEGISTI"})


def test_prompt_contract_hashi_prompt_degisince_degisir(monkeypatch):
    # Hash gerçekten prompt gövdelerine bağlı olmalı, sabit bir değer değil.
    onceki = _prompt_contract_hash()
    monkeypatch.setattr("eval.runner.CODER_SYSTEM_PROMPT", "DEGISMIS PROMPT")
    assert _prompt_contract_hash() != onceki


def test_heldout_fingerprint_degisken_gozlemleri_dislar(tmp_path, monkeypatch):
    # created_ts ve reference_timings her üretimde değişir; dahil edilirlerse
    # fingerprint "görev seti değişti" diye yanlış alarm verirdi.
    manifest = {
        "created_ts": "2026-07-27T00:00:00Z",
        "selection_seed": 20260727,
        "sources": {"humanevalplus": {"selected_task_ids": ["a"],
                                      "reference_timings": {"a": {"reference_plus_s": 1.0}}}},
    }
    monkeypatch.setattr("eval.runner.HELDOUT_TASKS_DIR", tmp_path)
    (tmp_path / "_selection_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8")
    ilk = _heldout_selection_fingerprint()

    manifest["created_ts"] = "2026-12-31T23:59:59Z"
    manifest["sources"]["humanevalplus"]["reference_timings"] = {"a": {"reference_plus_s": 9.9}}
    (tmp_path / "_selection_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8")
    assert _heldout_selection_fingerprint() == ilk, "değişken gözlem hash'i etkilememeli"

    manifest["selection_seed"] = 1
    (tmp_path / "_selection_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8")
    assert _heldout_selection_fingerprint() != ilk, "gerçek seçim değişimi yakalanmalı"


def test_manifest_eksik_kritik_alan_acik_hata_verir(tmp_path):
    # Sessiz KeyError yerine hangi alanın eksik olduğunu söyleyen açık hata.
    eksik = {k: v for k, v in SNAPSHOT.items() if k != "reasoning_config"}
    with pytest.raises(KeyError, match="reasoning_config"):
        check_or_write_manifest(tmp_path, eksik)


def test_kirli_agac_calismayi_durdurur(monkeypatch):
    def fake_run(*a, **k):
        return subprocess.CompletedProcess(args=a, returncode=0, stdout=" M agents/planner.py\n", stderr="")
    monkeypatch.setattr("eval.runner.subprocess.run", fake_run)
    with pytest.raises(SystemExit):
        _require_clean_tree()


def test_temiz_agac_engellenmiyor(monkeypatch):
    def fake_run(*a, **k):
        return subprocess.CompletedProcess(args=a, returncode=0, stdout="", stderr="")
    monkeypatch.setattr("eval.runner.subprocess.run", fake_run)
    _require_clean_tree()  # exception atmamalı
