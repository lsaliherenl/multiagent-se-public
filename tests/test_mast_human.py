"""Kör insan turu + AI destekli adjudication birim testleri — LLM'siz.

Bu dosyanın kapattığı sessiz hatalar: kör payload'a model/kol/AI kararı
sızdırmak, bir etiketleyicinin dosyasını diğerinin turuna kabul etmek, eksik
paneli insana vermek, kilitli kör etiketleri sonradan değiştirmek, model
üretimi metnin HTML'e enjekte olması, ortam hash seed'ine bağlı örneklem.
"""

import base64
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from config import (
    ARM_BASELINE,
    ARM_CONTRACT,
    ARM_NAIVE,
    ARM_STRUCTURED,
    LLM_CALL_SCHEMA_VERSION,
    MAST_DECISION_RULE_VERSION,
    MAST_HUMAN_APP_VERSION,
    MAST_HUMAN_SCHEMA_VERSION,
    MAST_SCHEMA_VERSION,
    MODEL_ADJUDICATOR,
    MODEL_JUDGES,
    MODEL_MAIN,
    RESULT_SCHEMA_VERSION,
)
from eval import mast_human as mh
from eval.mast_labels import (
    build_evidence,
    build_mast_manifest,
    build_panel,
    external_annotator_order,
    prompt_contract_hash,
)
from eval.mast_schema import evidence_digest
from eval.result_schema import make_synthetic_record

# Gerçek rol sabitleri: leave-self-out bölümlemesi kaynak modelin ÜRETİCİ
# olmasını ve panelde tam bir kez bulunmasını şart koşar (§9.1), bu yüzden
# sentetik judge adları artık panel kuramaz.
JUDGES = tuple(MODEL_JUDGES)
ADJ_MODEL = MODEL_ADJUDICATOR
# Ana koşunun manifestiyle aynı DÖRT kol: üç kollu fixture, insan örnekleminin
# ve kör paketin `structured_no_validation` kayıtlarını hiç taşımamasına yol açardı.
ARMS = [ARM_BASELINE, ARM_NAIVE, ARM_STRUCTURED, ARM_CONTRACT]
TASKS = ["t000", "t001", "t002"]
TASK = {"task_id": "t", "prompt": "def f(x):\n    ...", "entry_point": "f"}
MODEL = MODEL_MAIN            # kaynak/üretici model -> self-judge

# --- Sentetik deney dizini ---------------------------------------------------

def _label(primary="1.1", **kw):
    return {"primary_mode": primary, "secondary_modes": [], "confidence": "high",
            "rationale": "gerekce", "insufficient_context": False, **kw}


def _kur(tmp_path, monkeypatch, *, karar=None, kod=None, preliminary=False, atla=None):
    """Tam bir deney + MAST turu üretir (results, judges, panel, adjudication).

    `atla(record, judge_model) -> bool`: o judge'ın etiketi HİÇ YAZILMAZ. Canlı
    turda judge şema uyumsuzluğu yüzünden doldurulamayan kimliğin karşılığıdır.
    """
    monkeypatch.setattr("eval.mast_labels._task_for", lambda r: TASK)
    tmp_path.mkdir(parents=True, exist_ok=True)

    kayitlar = []
    for rep in range(2):
        for t in TASKS:
            for a in ARMS:
                basarili = (t == "t000")
                r = make_synthetic_record(experiment="live", model=MODEL,
                                          task_set="heldout", arm=a, task_id=t,
                                          repeat=rep, base_pass=basarili,
                                          plus_pass=basarili)
                r["code"] = kod or f"def f(x): return '{t}'"
                r["plan"] = {"task_id": t}
                r["raw_messages"] = [{"from": "planner", "to": "coder", "content": "plan"}]
                r["traceback"] = "AssertionError: x"
                kayitlar.append(r)

    (tmp_path / "results.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in kayitlar), encoding="utf-8")
    manifest = {"name": "live", "model": MODEL, "task_set": "heldout",
                "task_ids": TASKS, "arm_order": ARMS, "repeats": 2,
                "result_schema_version": RESULT_SCHEMA_VERSION,
                "llm_call_schema_version": LLM_CALL_SCHEMA_VERSION}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    mast = tmp_path / "mast"
    mast.mkdir(parents=True, exist_ok=True)
    snapshot = build_mast_manifest(manifest, judges=JUDGES, adjudicator=ADJ_MODEL)
    if preliminary:
        snapshot.update(preliminary=True, source_results_complete=False, missing_runs=2)
    (mast / "manifest.json").write_text(json.dumps(snapshot, indent=2), encoding="utf-8")

    basarisiz = [r for r in kayitlar if r["plus_pass"] is False]
    karar = karar or (lambda record, model: _label())
    judges = []
    for record in basarisiz:
        digest = evidence_digest(build_evidence(record, TASK))
        for m in JUDGES:
            if atla and atla(record, m):
                continue
            judges.append({
                "ts": "2026-07-28T00:00:00+00:00", "judge_model": m,
                "judge_status": "ok", "judge_attempt": 1,
                "mast_schema_version": MAST_SCHEMA_VERSION,
                "source_run_id": record["run_id"], "experiment": "live",
                "source_model": record["model"], "task_set": record["task_set"],
                "task_id": record["task_id"], "arm": record["arm"],
                "repeat": record["repeat"], "error_class": record.get("error_class"),
                "interaction_type": mh.interaction_type(record["arm"]),
                "evidence_sha256": digest, "mast_prompt_hash": prompt_contract_hash(),
                **karar(record, m)})
    (mast / "ai_judges.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in judges), encoding="utf-8")

    panel = build_panel(basarisiz, judges, experiment="live", expected_judges=JUDGES)
    (mast / "ai_panel.jsonl").write_text(
        "".join(json.dumps(p) + "\n" for p in panel), encoding="utf-8")

    by_run = {r["run_id"]: r for r in basarisiz}
    adjs = []
    for p in panel:
        if not p["adjudicator_required"]:
            continue
        rec = by_run[p["source_run_id"]]
        current = [j for j in judges if j["source_run_id"] == p["source_run_id"]]
        external = [j for j in current if j["judge_model"] != rec["model"]]
        ordered = external_annotator_order(external, rec["run_id"])
        external_models = [m for m in JUDGES if m != rec["model"]]
        adjs.append({
            # Gerçek adjudicate_record() label_provenance()'ın TAMAMINI yazar;
            # fixture da öyle olmalı, aksi halde tam-kimlik filtresi test edilemez.
            "source_run_id": p["source_run_id"], "experiment": "live",
            "source_model": rec["model"], "task_set": rec["task_set"],
            "task_id": rec["task_id"], "arm": rec["arm"], "repeat": rec["repeat"],
            "mast_schema_version": MAST_SCHEMA_VERSION,
            "mast_prompt_hash": p["mast_prompt_hash"],
            "evidence_sha256": p["evidence_sha256"],
            "decision_input_sha256": p["decision_input_sha256"],
            "decision_rule_version": MAST_DECISION_RULE_VERSION,
            "self_judge_model": rec["model"],
            "external_judges": external_models,
            "external_agreement_level": "split",
            "reviewed_judges": external_models,
            "annotator_assignment": {letter: row["judge_model"]
                                     for letter, row in zip("AB", ordered)},
            "interaction_type": mh.interaction_type(rec["arm"]),
            "adjudicator_status": "ok", "adjudicator_model": ADJ_MODEL,
            "adjudicator_attempt": 1,
            "adjudicated_primary_mode": "1.1", "adjudicated_secondary_modes": [],
            "adjudicated_confidence": "high",
            "adjudicated_rationale": "adjudicator gerekcesi",
            "adjudicated_insufficient_context": False})
    (mast / "ai_adjudication.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in adjs), encoding="utf-8")
    return tmp_path


def _hazirla(tmp_path, monkeypatch, **kw):
    _kur(tmp_path, monkeypatch, **kw)
    return mh.prepare_blind(tmp_path, expected_judges=JUDGES)


def _anlasmazlik(record, model):
    """t001 kayıtlarında üçüncü judge ayrışır; t002'de biri yetersiz bağlam der."""
    if record["task_id"] == "t001" and model == JUDGES[2]:
        return _label("3.1")
    if record["task_id"] == "t002" and model == JUDGES[1] and record["repeat"] == 0:
        return _label(None, insufficient_context=True, secondary_modes=[])
    return _label()


def _export(manifest, annotator, *, primary="1.1", eksik=0, **ust):
    idler = [r["record_id"] for r in manifest["records"]]
    labels = []
    for i, rid in enumerate(idler):
        if i < eksik:
            continue
        labels.append({"record_id": rid, "primary_mode": primary,
                       "secondary_modes": [], "confidence": "medium",
                       "rationale": f"insan {annotator}",
                       "insufficient_context": False})
    return {"human_schema_version": MAST_HUMAN_SCHEMA_VERSION,
            "app_version": MAST_HUMAN_APP_VERSION,
            "dataset_fingerprint": manifest["dataset_fingerprint"],
            "package_id": mh.package_id(manifest["dataset_fingerprint"],
                                        mh.PHASE_BLIND, annotator),
            "annotator_id": annotator, "phase": mh.PHASE_BLIND,
            "labels": labels, **ust}


def _final_export(manifest, payload, primary="1.1"):
    return {"human_schema_version": MAST_HUMAN_SCHEMA_VERSION,
            "app_version": MAST_HUMAN_APP_VERSION,
            "dataset_fingerprint": manifest["dataset_fingerprint"],
            "package_id": payload["package_id"], "phase": mh.PHASE_ADJUDICATION,
            "blind_lock_sha256": payload["blind_lock_sha256"],
            "labels": [{"record_id": r["record_id"],
                        "human_adjudicated_label": _label(primary)}
                       for r in manifest["records"]]}


def _payload(html: str) -> dict:
    """HTML'e gömülü base64 payload'u çözer (kör sızıntı testleri için)."""
    govde = html.split('var PAYLOAD_B64 = "')[1].split('"')[0]
    return json.loads(base64.b64decode(govde).decode("utf-8"))


# --- Fixture kapsamı ---------------------------------------------------------

def test_aday_havuzu_DORT_kolu_da_tasir(tmp_path, monkeypatch):
    # Fixture üç kolluyken bütün insan hattı testleri `structured_no_validation`
    # kaydını hiç görmeden yeşil kalıyordu. Bu kapı borcun sessizce geri
    # dönmesini engeller: kol kümesi manifestle aynı olmalı.
    # Karşılaştırma ARMS'a değil dondurulmuş tasarıma yapılır; fixture kendisiyle
    # karşılaştırılsaydı kol düşürmek testi kırmazdı.
    from config import ALL_ARMS
    assert ARMS == list(ALL_ARMS), "fixture kol kümesi ana koşununkinden farklı"
    _kur(tmp_path, monkeypatch, karar=_anlasmazlik)
    adaylar = mh.collect_candidates(mh.load_mast_round(tmp_path),
                                    expected_judges=JUDGES)
    assert {c["arm"] for c in adaylar} == set(ALL_ARMS)


# --- Determinizm -------------------------------------------------------------

def test_ayni_girdi_ayni_ornekleme_ve_ayni_HTML_baytlarina_gotorur(tmp_path, monkeypatch):
    kaynak = tmp_path / "a"
    _kur(kaynak, monkeypatch, karar=_anlasmazlik)
    kopya = tmp_path / "b"
    shutil.copytree(kaynak, kopya)

    a = mh.prepare_blind(kaynak, expected_judges=JUDGES)
    b = mh.prepare_blind(kopya, expected_judges=JUDGES)
    assert (a["manifest"]["dataset_fingerprint"]
            == b["manifest"]["dataset_fingerprint"])
    assert ([r["record_id"] for r in a["manifest"]["records"]]
            == [r["record_id"] for r in b["manifest"]["records"]])
    for annot in a["packages"]:
        assert (a["packages"][annot].read_bytes() == b["packages"][annot].read_bytes()), \
            "aynı girdi farklı HTML üretti — paket deterministik değil"


ADAY_SCRIPT = '''
import json, sys
sys.path.insert(0, sys.argv[1])
from eval.mast_human import select_human_sample, dataset_fingerprint
adaylar = json.loads(open(sys.argv[2], encoding="utf-8").read())
secim = select_human_sample(adaylar, target=9)
satirlar = [{"record_id": c["record_id"], "selection_type": c["selection_type"]}
            for c in secim]
print(json.dumps({"ids": [c["record_id"] for c in secim],
                  "fp": dataset_fingerprint(satirlar, seed=1, target=9,
                                            strata_fields=("model", "arm", "error_class"))}))
'''


def _hashseed_adaylari():
    return [{"record_id": f"r{i:03d}", "source_run_id": f"run{i}",
             "model": f"m{i % 3}", "arm": ARM_NAIVE if i % 2 else ARM_CONTRACT,
             "error_class": "assertion" if i % 3 else "timeout",
             "external_agreement_level": "consensus",
             "external_insufficient_judges": [], "self_matches_external": True}
            for i in range(40)]


def _referans_secim() -> dict:
    """Alt sürecin ürettiğiyle karşılaştırılacak SÜREÇ İÇİ referans."""
    secim = mh.select_human_sample(_hashseed_adaylari(), target=9)
    satirlar = [{"record_id": c["record_id"], "selection_type": c["selection_type"]}
                for c in secim]
    return {"ids": [c["record_id"] for c in secim],
            "fp": mh.dataset_fingerprint(satirlar, seed=1, target=9,
                                         strata_fields=("model", "arm", "error_class"))}


@pytest.mark.parametrize("seed_env", ["0", "1", "12345"])
def test_ornekleme_PYTHONHASHSEEDden_bagimsiz(tmp_path, seed_env):
    # Python'un hash()'i, set/dict sırası ve PYTHONHASHSEED süreçler arasında
    # değişir; örneklem bunlara bağlanırsa "aynı seed aynı örneklem" iddiası
    # çöker. Bu yüzden AYRI SÜREÇTE, farklı hash seed'iyle doğrulanır.
    betik = tmp_path / "aday.py"
    betik.write_text(ADAY_SCRIPT, encoding="utf-8")
    veri = tmp_path / "adaylar.json"
    veri.write_text(json.dumps(_hashseed_adaylari()), encoding="utf-8")
    kok = str(Path(__file__).resolve().parents[1])
    ortam = {**os.environ, "PYTHONHASHSEED": seed_env}
    sonuc = subprocess.run([sys.executable, str(betik), kok, str(veri)], env=ortam,
                           capture_output=True, text=True, timeout=120)
    assert sonuc.returncode == 0, sonuc.stderr
    assert json.loads(sonuc.stdout) == _referans_secim(), "hash seed örneklemi değiştirdi"


def test_ornekleme_girdi_sirasindan_bagimsiz():
    adaylar = [_aday(i) for i in range(20)]
    ilk = [c["record_id"] for c in mh.select_human_sample(adaylar, target=5)]
    ikinci = [c["record_id"] for c in mh.select_human_sample(list(reversed(adaylar)),
                                                             target=5)]
    assert ilk == ikinci


# --- Zorunlu küme ------------------------------------------------------------

def _aday(i, *, agreement="consensus", insufficient=(), self_matches=True,
          arm=ARM_NAIVE, err="assertion"):
    return {"record_id": f"r{i:03d}", "source_run_id": f"run{i}", "model": "m",
            "arm": arm, "error_class": err, "external_agreement_level": agreement,
            "external_insufficient_judges": list(insufficient),
            "self_matches_external": self_matches}


def test_butun_external_split_kayitlari_secilir():
    adaylar = ([_aday(i, agreement="split") for i in range(5)]
               + [_aday(50 + i, agreement="incomplete") for i in range(4)]
               + [_aday(100 + i) for i in range(20)])
    secim = mh.select_human_sample(adaylar, target=3)
    zorunlu = {c["record_id"] for c in secim
               if c["selection_type"] == mh.SELECTION_MANDATORY_DISAGREEMENT}
    assert len(zorunlu) == 9


def test_yetersiz_baglam_veren_judge_varsa_kayit_secilir():
    adaylar = [_aday(0, insufficient=[JUDGES[0]])] + [_aday(i) for i in range(1, 10)]
    secim = {c["record_id"]: c for c in mh.select_human_sample(adaylar, target=1)}
    assert secim["r000"]["selection_type"] == mh.SELECTION_MANDATORY_INSUFFICIENT
    assert any("insufficient" in n for n in secim["r000"]["selection_reasons"])


def test_zorunlu_kume_hedefi_asarsa_kayit_ATILMAZ():
    # "Bütün anlaşmazlıklar incelenir" ile "en fazla 30" birlikte savunulamaz.
    adaylar = [_aday(i, agreement="split") for i in range(40)]
    secim = mh.select_human_sample(adaylar, target=30)
    assert len(secim) == 40


def test_tabakali_secim_kararli_ve_yinelemesiz():
    adaylar = [_aday(i, arm=ARM_NAIVE if i % 2 else ARM_CONTRACT,
                     err="assertion" if i % 3 else "timeout") for i in range(30)]
    a = mh.select_human_sample(adaylar, target=8)
    b = mh.select_human_sample(adaylar, target=8)
    idler = [c["record_id"] for c in a]
    assert idler == [c["record_id"] for c in b]
    assert len(set(idler)) == len(idler) == 8
    assert all(c["selection_type"] == mh.SELECTION_STRATIFIED for c in a)
    # Round-robin: tek bir tabaka örneklemi domine etmemeli.
    tabakalar = {(c["arm"], c["error_class"]) for c in a}
    assert len(tabakalar) > 1


def test_hedeften_az_hata_varsa_hepsi_alinir():
    adaylar = [_aday(i) for i in range(4)]
    assert len(mh.select_human_sample(adaylar, target=30)) == 4


def test_self_external_consensus_ayrismasi_zorunlu_secimdir():
    adaylar = [_aday(0, self_matches=False)] + [_aday(i) for i in range(1, 10)]
    secim = {c["record_id"]: c for c in mh.select_human_sample(adaylar, target=1)}
    assert secim["r000"]["selection_type"] == mh.SELECTION_MANDATORY_SELF_MISMATCH
    assert "self_disagrees_with_external_consensus" in secim["r000"]["selection_reasons"]


# --- Girdi bütünlüğü kapısı --------------------------------------------------

def test_on_etiketleme_turundan_paket_URETILEMEZ(tmp_path, monkeypatch):
    _kur(tmp_path, monkeypatch, preliminary=True)
    with pytest.raises(mh.HumanRoundError, match="ÖN ETİKETLEME"):
        mh.prepare_blind(tmp_path, expected_judges=JUDGES)


def test_eksik_panelden_paket_URETILEMEZ(tmp_path, monkeypatch):
    _kur(tmp_path, monkeypatch)
    yol = tmp_path / "mast" / "ai_panel.jsonl"
    satirlar = [json.loads(s) for s in yol.read_text(encoding="utf-8").splitlines()]
    satirlar[0].update(panel_complete=False, agreement_level="incomplete",
                       missing_judges=[JUDGES[2]])
    yol.write_text("".join(json.dumps(s) + "\n" for s in satirlar), encoding="utf-8")
    with pytest.raises(mh.HumanRoundError, match="panel (eksik|sözleşmesi geçersiz)"):
        mh.prepare_blind(tmp_path, expected_judges=JUDGES)


def test_stale_panelden_paket_URETILEMEZ(tmp_path, monkeypatch):
    _kur(tmp_path, monkeypatch)
    yol = tmp_path / "mast" / "ai_panel.jsonl"
    satirlar = [json.loads(s) for s in yol.read_text(encoding="utf-8").splitlines()]
    satirlar[0]["full_panel_input_sha256"] = "0" * 64
    yol.write_text("".join(json.dumps(s) + "\n" for s in satirlar), encoding="utf-8")
    with pytest.raises(mh.HumanRoundError, match="tanısal üçlü değişti"):
        mh.prepare_blind(tmp_path, expected_judges=JUDGES)


def test_yabanci_panel_kaydi_reddedilir(tmp_path, monkeypatch):
    _kur(tmp_path, monkeypatch)
    yol = tmp_path / "mast" / "ai_panel.jsonl"
    satirlar = [json.loads(s) for s in yol.read_text(encoding="utf-8").splitlines()]
    satirlar.append({**satirlar[0], "source_run_id": "yabanci-run"})
    yol.write_text("".join(json.dumps(s) + "\n" for s in satirlar), encoding="utf-8")
    with pytest.raises(mh.HumanRoundError, match="yabancı kayıt"):
        mh.prepare_blind(tmp_path, expected_judges=JUDGES)


def test_MAST_turu_yoksa_eyleme_donuk_hata(tmp_path, monkeypatch):
    _kur(tmp_path, monkeypatch)
    (tmp_path / "mast" / "manifest.json").unlink()
    with pytest.raises(mh.HumanRoundError, match="mast_labels"):
        mh.prepare_blind(tmp_path, expected_judges=JUDGES)


@pytest.mark.parametrize("field,value", [
    ("judges", [JUDGES[1], JUDGES[0], JUDGES[2]]),
    ("adjudicator_model", "rogue/adjudicator"),
    ("mast_schema_version", "3.1"),
    ("mast_decision_rule_version", "triple_majority_v0"),
    ("mast_panel_hash_version", "single_input_v0"),
    ("source_model", "rogue/producer"),
])
def test_MAST_manifesti_tam_kimlik_kapisindan_gecer(tmp_path, monkeypatch,
                                                    field, value):
    _kur(tmp_path, monkeypatch, karar=_anlasmazlik)
    path = tmp_path / "mast" / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest[field] = value
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(mh.HumanRoundError, match="MAST manifesti"):
        mh.prepare_blind(tmp_path, expected_judges=JUDGES)


# --- Örneklem manifesti ------------------------------------------------------

def test_elle_duzenlenmis_ornekleme_manifesti_reddedilir(tmp_path, monkeypatch):
    # Elle değiştirilen bir satır (ör. bir kaydın arm'ı) hem paket kimliğini hem
    # interaction_type doğrulamasını sessizce değiştirirdi.
    _hazirla(tmp_path, monkeypatch)
    yol = tmp_path / "mast" / "human_sample_manifest.json"
    m = json.loads(yol.read_text(encoding="utf-8"))
    # Örneklem record_id'ye göre sıralı (opaque hash) — ilk kaydın kolu sabit
    # değildir; gerçekten DEĞİŞEN bir kayıt seçilmeli.
    hedef = next(r for r in m["records"] if r["arm"] != ARM_BASELINE)
    hedef["arm"] = ARM_BASELINE
    yol.write_text(json.dumps(m), encoding="utf-8")
    with pytest.raises(mh.HumanRoundError, match="kilitlendikten sonra düzenlenmiş"):
        mh.load_sample_manifest(tmp_path)


def test_farkli_surumle_uretilmis_manifest_reddedilir(tmp_path, monkeypatch):
    _hazirla(tmp_path, monkeypatch)
    yol = tmp_path / "mast" / "human_sample_manifest.json"
    m = json.loads(yol.read_text(encoding="utf-8"))
    m["app_version"] = "0.9"
    yol.write_text(json.dumps(m), encoding="utf-8")
    with pytest.raises(mh.HumanRoundError, match="farklı bir sürümle"):
        mh.load_sample_manifest(tmp_path)


def test_ayni_ornekleme_prepare_blind_idempotent(tmp_path, monkeypatch):
    ilk = _hazirla(tmp_path, monkeypatch, karar=_anlasmazlik)
    ikinci = mh.prepare_blind(tmp_path, expected_judges=JUDGES)
    assert ilk["manifest"] == ikinci["manifest"], "created_ts korunmalı"
    assert (ilk["packages"]["annotator_a"].read_bytes()
            == ikinci["packages"]["annotator_a"].read_bytes())


def test_farkli_annotatorla_mevcut_ornekleme_UZERINE_YAZILAMAZ(tmp_path, monkeypatch):
    _hazirla(tmp_path, monkeypatch, karar=_anlasmazlik)
    with pytest.raises(mh.HumanRoundError, match="FARKLI bir insan örneklemi"):
        mh.prepare_blind(tmp_path, annotators=("insan_1", "insan_2"),
                         expected_judges=JUDGES)


def test_farkli_hedefle_mevcut_ornekleme_UZERINE_YAZILAMAZ(tmp_path, monkeypatch):
    # Sessizce ezilseydi iki etiketleyici FARKLI kayıt kümeleriyle çalışabilir
    # ve kilitli kör turlar hiçbir manifeste ait olmazdı.
    _hazirla(tmp_path, monkeypatch, karar=_anlasmazlik)
    with pytest.raises(mh.HumanRoundError, match="FARKLI bir insan örneklemi"):
        mh.prepare_blind(tmp_path, target=5, expected_judges=JUDGES)


# --- Kör payload sızıntısı ---------------------------------------------------

def test_kor_payloadda_model_kol_ve_AI_karari_BULUNMAZ(tmp_path, monkeypatch):
    sonuc = _hazirla(tmp_path, monkeypatch, karar=_anlasmazlik)
    ham = sonuc["packages"]["annotator_a"].read_text(encoding="utf-8")
    payload = _payload(ham)
    duz = mh.canonical(payload)
    for yasak in (MODEL, ARM_CONTRACT, ARM_NAIVE, *JUDGES, ADJ_MODEL):
        assert yasak not in duz, f"kör payload'da {yasak} var"
    for kayit in payload["records"]:
        assert set(kayit) == {"record_id", "evidence_sha256", "interaction_type",
                              "evidence"}
        assert set(kayit["evidence"]) == set(mh.BLIND_EVIDENCE_FIELDS)
    # Ham HTML'de de görünmemeli (base64 dışında hiçbir yerde).
    for yasak in (MODEL, ADJ_MODEL):
        assert yasak not in ham


def test_kor_payloadda_source_run_id_yok(tmp_path, monkeypatch):
    sonuc = _hazirla(tmp_path, monkeypatch)
    payload = _payload(sonuc["packages"]["annotator_a"].read_text(encoding="utf-8"))
    duz = mh.canonical(payload)
    for satir in sonuc["manifest"]["records"]:
        assert satir["source_run_id"] not in duz
    for yasak in mh.FORBIDDEN_BLIND_KEYS:
        assert f'"{yasak}"' not in duz


def test_iki_annotator_ayni_kaniti_farkli_kimligi_tasir(tmp_path, monkeypatch):
    sonuc = _hazirla(tmp_path, monkeypatch, karar=_anlasmazlik)
    a = _payload(sonuc["packages"]["annotator_a"].read_text(encoding="utf-8"))
    b = _payload(sonuc["packages"]["annotator_b"].read_text(encoding="utf-8"))
    assert a["records"] == b["records"], "iki etiketleyici aynı kanıtı görmeli"
    assert a["dataset_fingerprint"] == b["dataset_fingerprint"]
    assert a["annotator_id"] != b["annotator_id"]
    assert a["package_id"] != b["package_id"], "paket kimlikleri karışabilirdi"


def test_annotator_kimligi_dogrulanir(tmp_path, monkeypatch):
    _kur(tmp_path, monkeypatch)
    with pytest.raises(mh.HumanRoundError, match="yalnız"):
        mh.prepare_blind(tmp_path, annotators=("a/../b", "c"), expected_judges=JUDGES)
    with pytest.raises(mh.HumanRoundError, match="farklı olmalı"):
        mh.prepare_blind(tmp_path, annotators=("ayni", "ayni"), expected_judges=JUDGES)


# --- HTML güvenliği ----------------------------------------------------------

ZARARLI = ('</script><script>window.SIZDI=1;</script>'
           '<img src=x onerror="alert(1)">')


def test_script_kapatan_kanit_ham_HTMLde_gorunmez_ama_payloadda_TAM_kalir(
        tmp_path, monkeypatch):
    sonuc = _hazirla(tmp_path, monkeypatch, kod=ZARARLI)
    ham = sonuc["packages"]["annotator_a"].read_text(encoding="utf-8")
    assert "window.SIZDI" not in ham, "model metni ham HTML'e sızdı"
    assert "onerror=" not in ham
    payload = _payload(ham)
    assert any(ZARARLI in k["evidence"]["code"] for k in payload["records"]), \
        "base64 çözümünde kanıt eksik kaldı"


@pytest.mark.parametrize("sablon", ["mast_blind.html", "mast_human_adjudication.html"])
def test_arayuz_yetersiz_baglamda_da_confidence_ister(sablon):
    # Arayüzün complete() kuralı MastLabel'dan GEVŞEK olursa, etiketleyici turu
    # "tamam" görüp final dosyayı indirir ve Python kilidi reddeder — kişi tur
    # bittikten sonra geri çağrılır. İki kural aynı sıkılıkta olmalı.
    metin = (mh.TEMPLATE_DIR / sablon).read_text(encoding="utf-8")
    assert "return s.rationale.trim().length > 0 && !!s.confidence;" in metin, (
        f"{sablon}: yetersiz bağlamda confidence kontrolü yok")
    # Şemanın gerçekten istediğini de sabitle (kural değişirse test kırılsın).
    with pytest.raises(mh.HumanRoundError):
        mh._label_from({"primary_mode": None, "secondary_modes": [], "confidence": "",
                        "rationale": "yetersiz", "insufficient_context": True},
                       "r0", "multi_agent")


@pytest.mark.parametrize("sablon", ["mast_blind.html", "mast_human_adjudication.html",
                                    "mast_post_lock_diagnostic.html"])
def test_sablonlarda_tehlikeli_API_ve_dis_URL_yok(sablon):
    metin = (mh.TEMPLATE_DIR / sablon).read_text(encoding="utf-8")
    for yasak in ("innerHTML", "outerHTML", "eval(", "document.write",
                  "http://", "https://", "fetch(", "XMLHttpRequest",
                  "insertAdjacentHTML"):
        assert yasak not in metin, f"{sablon} içinde {yasak}"
    assert "Content-Security-Policy" in metin
    assert mh.PAYLOAD_TOKEN in metin


# --- Kör export doğrulaması --------------------------------------------------

def test_dogru_export_kabul_edilir_ve_manifest_sirasina_normalize_olur(
        tmp_path, monkeypatch):
    manifest = _hazirla(tmp_path, monkeypatch)["manifest"]
    export = _export(manifest, "annotator_a")
    export["labels"] = list(reversed(export["labels"]))
    icerik = mh.validate_blind_export(export, manifest, "annotator_a")
    assert ([e["record_id"] for e in icerik["labels"]]
            == [r["record_id"] for r in manifest["records"]])


def test_annotator_A_exportu_B_icin_kabul_EDILMEZ(tmp_path, monkeypatch):
    manifest = _hazirla(tmp_path, monkeypatch)["manifest"]
    export = _export(manifest, "annotator_a")
    with pytest.raises(mh.HumanRoundError, match="bu pakete ait değil"):
        mh.validate_blind_export(export, manifest, "annotator_b")


@pytest.mark.parametrize("bozan", [
    lambda e: e["labels"].pop(),
    lambda e: e["labels"].append(dict(e["labels"][0], record_id="yabanci")),
    lambda e: e["labels"].append(dict(e["labels"][0])),
])
def test_eksik_fazla_ve_yinelenen_kayit_reddedilir(tmp_path, monkeypatch, bozan):
    manifest = _hazirla(tmp_path, monkeypatch)["manifest"]
    export = _export(manifest, "annotator_a")
    bozan(export)
    with pytest.raises(mh.HumanRoundError, match="record_id|kayıt kümesi"):
        mh.validate_blind_export(export, manifest, "annotator_a")


def test_bilinmeyen_ust_duzey_alan_reddedilir(tmp_path, monkeypatch):
    manifest = _hazirla(tmp_path, monkeypatch)["manifest"]
    export = _export(manifest, "annotator_a", ekstra="?")
    with pytest.raises(mh.HumanRoundError, match="bilinmeyen üst düzey"):
        mh.validate_blind_export(export, manifest, "annotator_a")


@pytest.mark.parametrize("bozuk", [
    {"primary_mode": "9.9"},
    {"primary_mode": "1.1", "secondary_modes": ["1.1"]},
    {"primary_mode": "none", "secondary_modes": ["1.2"]},
    {"primary_mode": "1.1", "insufficient_context": True},
    {"primary_mode": "1.1", "rationale": "   "},
    {"primary_mode": "1.1", "confidence": "cok"},
])
def test_gecersiz_etiket_kombinasyonlari_reddedilir(tmp_path, monkeypatch, bozuk):
    manifest = _hazirla(tmp_path, monkeypatch)["manifest"]
    export = _export(manifest, "annotator_a")
    export["labels"][0].update(bozuk)
    with pytest.raises(mh.HumanRoundError):
        mh.validate_blind_export(export, manifest, "annotator_a")


def test_tek_ajanli_kayitta_kategori_2_etiketi_reddedilir(tmp_path, monkeypatch):
    manifest = _hazirla(tmp_path, monkeypatch)["manifest"]
    hedef = next(r for r in manifest["records"] if r["arm"] == ARM_BASELINE)
    export = _export(manifest, "annotator_a")
    for e in export["labels"]:
        if e["record_id"] == hedef["record_id"]:
            e["primary_mode"] = "2.3"
    with pytest.raises(mh.HumanRoundError, match="ajanlar-arası"):
        mh.validate_blind_export(export, manifest, "annotator_a")


def test_export_import_round_trip_etiket_kaybetmez(tmp_path, monkeypatch):
    manifest = _hazirla(tmp_path, monkeypatch)["manifest"]
    export = _export(manifest, "annotator_a")
    icerik = mh.validate_blind_export(export, manifest, "annotator_a")
    tekrar = mh.validate_blind_export(
        {**export, "labels": icerik["labels"]}, manifest, "annotator_a")
    assert tekrar == icerik


# --- Kilit -------------------------------------------------------------------

def test_ayni_icerik_idempotent_farkli_icerik_REDDEDILIR(tmp_path, monkeypatch):
    manifest = _hazirla(tmp_path, monkeypatch)["manifest"]
    (tmp_path / "in_a.json").write_text(json.dumps(_export(manifest, "annotator_a")),
                                        encoding="utf-8")
    ilk = mh.lock_blind(tmp_path, "annotator_a", tmp_path / "in_a.json")
    tekrar = mh.lock_blind(tmp_path, "annotator_a", tmp_path / "in_a.json")
    assert ilk["content_sha256"] == tekrar["content_sha256"]

    (tmp_path / "in_a2.json").write_text(
        json.dumps(_export(manifest, "annotator_a", primary="1.2")), encoding="utf-8")
    with pytest.raises(mh.HumanRoundError, match="İÇERİĞİ FARKLI"):
        mh.lock_blind(tmp_path, "annotator_a", tmp_path / "in_a2.json")


def test_kilit_icerigi_elle_degistirilince_hash_dogrulamasi_kirilir(
        tmp_path, monkeypatch):
    manifest = _hazirla(tmp_path, monkeypatch)["manifest"]
    (tmp_path / "in_a.json").write_text(json.dumps(_export(manifest, "annotator_a")),
                                        encoding="utf-8")
    mh.lock_blind(tmp_path, "annotator_a", tmp_path / "in_a.json")
    yol = mh.blind_lock_path(tmp_path, "annotator_a")
    zarf = json.loads(yol.read_text(encoding="utf-8"))
    zarf["content"]["labels"][0]["primary_mode"] = "3.3"
    yol.write_text(json.dumps(zarf), encoding="utf-8")
    with pytest.raises(mh.HumanRoundError, match="hash doğrulaması BAŞARISIZ"):
        mh.read_lock(yol)


# --- Adjudication ------------------------------------------------------------

def _iki_kilit(tmp_path, manifest, *, b_primary="1.2"):
    for annot, primary in (("annotator_a", "1.1"), ("annotator_b", b_primary)):
        yol = tmp_path / f"in_{annot}.json"
        yol.write_text(json.dumps(_export(manifest, annot, primary=primary)),
                       encoding="utf-8")
        mh.lock_blind(tmp_path, annot, yol)


def test_tek_kor_kilitle_adjudication_URETILEMEZ(tmp_path, monkeypatch):
    manifest = _hazirla(tmp_path, monkeypatch, karar=_anlasmazlik)["manifest"]
    (tmp_path / "in_a.json").write_text(json.dumps(_export(manifest, "annotator_a")),
                                        encoding="utf-8")
    mh.lock_blind(tmp_path, "annotator_a", tmp_path / "in_a.json")
    with pytest.raises(mh.HumanRoundError, match="kilitli kör tur eksik"):
        mh.prepare_adjudication(tmp_path, expected_judges=JUDGES)


def test_ayni_kimlige_ait_iki_kilit_reddedilir(tmp_path, monkeypatch):
    manifest = _hazirla(tmp_path, monkeypatch, karar=_anlasmazlik)["manifest"]
    _iki_kilit(tmp_path, manifest)
    yol = mh.blind_lock_path(tmp_path, "annotator_b")
    zarf = json.loads(yol.read_text(encoding="utf-8"))
    zarf["content"]["annotator_id"] = "annotator_a"
    zarf["content_sha256"] = mh.sha256_of(zarf["content"])
    yol.write_text(json.dumps(zarf), encoding="utf-8")
    with pytest.raises(mh.HumanRoundError, match="başka bir etiketleyiciye ait"):
        mh.prepare_adjudication(tmp_path, expected_judges=JUDGES)


def test_farkli_ornekleme_ait_kilit_birlestirilemez(tmp_path, monkeypatch):
    manifest = _hazirla(tmp_path, monkeypatch, karar=_anlasmazlik)["manifest"]
    _iki_kilit(tmp_path, manifest)
    yol = mh.blind_lock_path(tmp_path, "annotator_b")
    zarf = json.loads(yol.read_text(encoding="utf-8"))
    zarf["content"]["dataset_fingerprint"] = "baska-ornekleme"
    zarf["content_sha256"] = mh.sha256_of(zarf["content"])
    yol.write_text(json.dumps(zarf), encoding="utf-8")
    with pytest.raises(mh.HumanRoundError, match="başka bir örnekleme ait"):
        mh.prepare_adjudication(tmp_path, expected_judges=JUDGES)


def test_adjudication_paketi_kor_kararlari_ve_external_AI_kararini_acar(
        tmp_path, monkeypatch):
    manifest = _hazirla(tmp_path, monkeypatch, karar=_anlasmazlik)["manifest"]
    _iki_kilit(tmp_path, manifest)
    payload = mh.prepare_adjudication(tmp_path, expected_judges=JUDGES)["payload"]
    kayit = payload["records"][0]
    assert [b["annotator_id"] for b in kayit["blind_labels"]] == ["annotator_a",
                                                                 "annotator_b"]
    assert [j["label"] for j in kayit["external_ai_judges"]] == ["A", "B"]
    duz = mh.canonical(payload)
    for yasak in (MODEL, ARM_CONTRACT, ARM_NAIVE, *JUDGES, ADJ_MODEL):
        assert yasak not in duz, f"adjudication paketinde {yasak} var"
    anlasmazlik = [k for k in payload["records"] if k["grok_adjudication"]]
    assert anlasmazlik, "anlaşmazlık kaydında AI adjudicator sonucu görünmeli"


def test_AI_paneli_degisirse_eski_adjudication_exportu_kabul_EDILMEZ(
        tmp_path, monkeypatch):
    manifest = _hazirla(tmp_path, monkeypatch, karar=_anlasmazlik)["manifest"]
    _iki_kilit(tmp_path, manifest)
    payload = mh.prepare_adjudication(tmp_path, expected_judges=JUDGES)["payload"]
    (tmp_path / "final.json").write_text(
        json.dumps(_final_export(manifest, payload, "1.3")), encoding="utf-8")
    mh.lock_adjudication(tmp_path, tmp_path / "final.json", expected_judges=JUDGES)

    # AI paneli tazelenirse aynı export artık BAŞKA bir AI görünümüne dayanır.
    kilit = mh.adjudication_lock_path(tmp_path)
    kilit.unlink()
    yol = tmp_path / "mast" / "ai_judges.jsonl"
    satirlar = [json.loads(s) for s in yol.read_text(encoding="utf-8").splitlines()]
    satirlar[0]["primary_mode"] = "3.2"
    yol.write_text("".join(json.dumps(s) + "\n" for s in satirlar), encoding="utf-8")
    assert _yeniden_panel(tmp_path, monkeypatch, satirlar)
    # AI görünümü değiştiği için hem paket yeniden üretilemez hem eski export
    # kilitlenemez; mesaj yeni bir tur açılması gerektiğini söylemeli.
    with pytest.raises(mh.HumanRoundError, match="bu pakete ait olmayan bir AI"):
        mh.lock_adjudication(tmp_path, tmp_path / "final.json", expected_judges=JUDGES)
    with pytest.raises(mh.HumanRoundError, match="prepare-blind"):
        mh.prepare_adjudication(tmp_path, expected_judges=JUDGES)


def _yeniden_panel(tmp_path, monkeypatch, judges):
    """Judge dosyası değiştikten sonra paneli tazeler (AI tarafı yeniden koştu)."""
    from eval.mast_labels import labelable_records, load_experiment
    _manifest, records, _rap = load_experiment(tmp_path)
    hedef = labelable_records(records)
    panel = build_panel(hedef, judges, experiment="live", expected_judges=JUDGES)
    (tmp_path / "mast" / "ai_panel.jsonl").write_text(
        "".join(json.dumps(p) + "\n" for p in panel), encoding="utf-8")
    yol = tmp_path / "mast" / "ai_adjudication.jsonl"
    adjs = [json.loads(s) for s in yol.read_text(encoding="utf-8").splitlines()]
    by_run = {p["source_run_id"]: p for p in panel}
    records_by_run = {r["run_id"]: r for r in hedef}
    for a in adjs:
        p = by_run[a["source_run_id"]]
        a["decision_input_sha256"] = p["decision_input_sha256"]
    for p in panel:
        if p["adjudicator_required"] and not any(
                a["source_run_id"] == p["source_run_id"] for a in adjs):
            rec = records_by_run[p["source_run_id"]]
            current = [j for j in judges if j["source_run_id"] == p["source_run_id"]]
            external = [j for j in current if j["judge_model"] != rec["model"]]
            ordered = external_annotator_order(external, rec["run_id"])
            external_models = [m for m in JUDGES if m != rec["model"]]
            adjs.append({
                "source_run_id": p["source_run_id"], "experiment": "live",
                "source_model": p["source_model"], "task_set": p["task_set"],
                "task_id": p["task_id"], "arm": p["arm"], "repeat": p["repeat"],
                "mast_schema_version": MAST_SCHEMA_VERSION,
                "mast_prompt_hash": p["mast_prompt_hash"],
                "evidence_sha256": p["evidence_sha256"],
                "decision_input_sha256": p["decision_input_sha256"],
                "decision_rule_version": MAST_DECISION_RULE_VERSION,
                "self_judge_model": rec["model"],
                "external_judges": external_models,
                "external_agreement_level": "split",
                "reviewed_judges": external_models,
                "annotator_assignment": {letter: row["judge_model"]
                                         for letter, row in zip("AB", ordered)},
                "interaction_type": mh.interaction_type(rec["arm"]),
                "adjudicator_status": "ok", "adjudicator_model": ADJ_MODEL,
                "adjudicator_attempt": 1,
                "adjudicated_primary_mode": "1.1", "adjudicated_secondary_modes": [],
                "adjudicated_confidence": "high",
                "adjudicated_rationale": "adj", "adjudicated_insufficient_context": False})
    yol.write_text("".join(json.dumps(a) + "\n" for a in adjs), encoding="utf-8")
    return True


def _adj_satirlari(tmp_path):
    yol = tmp_path / "mast" / "ai_adjudication.jsonl"
    return yol, [json.loads(s) for s in yol.read_text(encoding="utf-8").splitlines()]


def _adj_yaz(yol, satirlar):
    yol.write_text("".join(json.dumps(s) + "\n" for s in satirlar), encoding="utf-8")


def test_adjudicator_KARARI_degisirse_snapshot_ve_paket_kimligi_degisir(
        tmp_path, monkeypatch):
    # KRİTİK: snapshot yalnız panel_input_sha256'yı hash'liyordu; adjudicator
    # AYNI panel üzerinde 1.1 -> 3.3 dediğinde paket kimliği hiç değişmiyor ve
    # eski nihai insan kararı yeni AI kararına karşı kabul ediliyordu.
    manifest = _hazirla(tmp_path, monkeypatch, karar=_anlasmazlik)["manifest"]
    _iki_kilit(tmp_path, manifest)
    onceki = mh.prepare_adjudication(tmp_path, expected_judges=JUDGES)["payload"]

    yol, satirlar = _adj_satirlari(tmp_path)
    satirlar[0].update(adjudicated_primary_mode="3.3",
                       adjudicated_rationale="tamamen farklı karar")
    _adj_yaz(yol, satirlar)
    sonraki = mh.prepare_adjudication(tmp_path, expected_judges=JUDGES)["payload"]

    assert onceki["ai_snapshot_sha256"] != sonraki["ai_snapshot_sha256"]
    assert onceki["package_id"] != sonraki["package_id"]


def test_eski_paket_kimligiyle_uretilmis_export_reddedilir(tmp_path, monkeypatch):
    manifest = _hazirla(tmp_path, monkeypatch, karar=_anlasmazlik)["manifest"]
    _iki_kilit(tmp_path, manifest)
    eski = mh.prepare_adjudication(tmp_path, expected_judges=JUDGES)["payload"]
    export = _final_export(manifest, eski)
    (tmp_path / "final.json").write_text(json.dumps(export), encoding="utf-8")

    yol, satirlar = _adj_satirlari(tmp_path)
    satirlar[0].update(adjudicated_primary_mode="3.3",
                       adjudicated_rationale="tamamen farklı karar")
    _adj_yaz(yol, satirlar)
    with pytest.raises(mh.HumanRoundError, match="bu pakete ait değil"):
        mh.lock_adjudication(tmp_path, tmp_path / "final.json", expected_judges=JUDGES)


@pytest.mark.parametrize("alan,deger", [
    ("experiment", "BASKA-DENEY"),
    ("source_model", "baska/model"),
    ("task_set", "pilot"),
    ("task_id", "baska_gorev"),
    ("arm", ARM_BASELINE),
    ("repeat", 9),
    ("mast_schema_version", "0.9"),
    ("mast_prompt_hash", "eski-prompt"),
    ("adjudicator_model", "baska/adjudicator"),
])
def test_yanlis_provenance_tasiyan_AI_adjudication_GUNCEL_sayilmaz(
        tmp_path, monkeypatch, alan, deger):
    # 6A'da judge katmanında kapatılan açığın insan katmanındaki tekrarı.
    manifest = _hazirla(tmp_path, monkeypatch, karar=_anlasmazlik)["manifest"]
    _iki_kilit(tmp_path, manifest)
    yol, satirlar = _adj_satirlari(tmp_path)
    for s in satirlar:
        s[alan] = deger
    _adj_yaz(yol, satirlar)
    with pytest.raises(mh.HumanRoundError, match="güncel AI adjudication yok"):
        mh.prepare_adjudication(tmp_path, expected_judges=JUDGES)


def test_yanlis_provenance_yanindaki_dogru_AI_adjudication_secilir(
        tmp_path, monkeypatch):
    manifest = _hazirla(tmp_path, monkeypatch, karar=_anlasmazlik)["manifest"]
    _iki_kilit(tmp_path, manifest)
    yol, satirlar = _adj_satirlari(tmp_path)
    bozuk = [dict(s, experiment="BASKA-DENEY", adjudicated_primary_mode="3.3",
                  adjudicated_rationale="bozuk kayıt") for s in satirlar]
    _adj_yaz(yol, bozuk + satirlar)
    payload = mh.prepare_adjudication(tmp_path, expected_judges=JUDGES)["payload"]
    kararlar = {k["grok_adjudication"]["primary_mode"]
                for k in payload["records"] if k["grok_adjudication"]}
    assert kararlar == {"1.1"}, "bozuk provenance'lı karar insana gösterildi"


def test_ayni_tam_kimlikte_iki_AI_adjudication_fail_fast(tmp_path, monkeypatch):
    manifest = _hazirla(tmp_path, monkeypatch, karar=_anlasmazlik)["manifest"]
    _iki_kilit(tmp_path, manifest)
    yol, satirlar = _adj_satirlari(tmp_path)
    _adj_yaz(yol, satirlar + [dict(satirlar[0], adjudicated_primary_mode="3.3")])
    with pytest.raises(mh.HumanRoundError, match="birden fazla başarılı AI"):
        mh.prepare_adjudication(tmp_path, expected_judges=JUDGES)


def test_adjudication_exportu_kor_kararlari_DEGISTIREMEZ(tmp_path, monkeypatch):
    manifest = _hazirla(tmp_path, monkeypatch, karar=_anlasmazlik)["manifest"]
    _iki_kilit(tmp_path, manifest)
    payload = mh.prepare_adjudication(tmp_path, expected_judges=JUDGES)["payload"]
    kilitler = mh._load_blind_locks(tmp_path, manifest)
    export = {
        "human_schema_version": MAST_HUMAN_SCHEMA_VERSION,
        "app_version": MAST_HUMAN_APP_VERSION,
        "dataset_fingerprint": manifest["dataset_fingerprint"],
        "package_id": payload["package_id"], "phase": mh.PHASE_ADJUDICATION,
        "blind_lock_sha256": payload["blind_lock_sha256"],
        "labels": [{"record_id": r["record_id"],
                    "human_adjudicated_label": _label(),
                    "blind_labels": [{"primary_mode": "3.3"}]}
                   for r in manifest["records"]]}
    with pytest.raises(mh.HumanRoundError, match="yalnız"):
        mh.validate_adjudication_export(export, manifest, kilitler,
                                        payload["package_id"])


# --- Uyum özeti --------------------------------------------------------------

def test_cohens_kappa_bilinen_fixturda_dogru():
    # 10 kayıt, 8 uyum. po=0.8; a: 6×"1.1" 4×"1.2", b: 6×"1.1" 4×"1.2"
    a = ["1.1"] * 6 + ["1.2"] * 4
    b = ["1.1"] * 5 + ["1.2"] + ["1.2"] * 3 + ["1.1"]
    kappa, not_ = mh.cohens_kappa(a, b)
    po = sum(1 for x, y in zip(a, b) if x == y) / 10
    pe = (6 / 10) * (6 / 10) + (4 / 10) * (4 / 10)
    assert not_ is None
    assert kappa == pytest.approx(round((po - pe) / (1 - pe), 4))


def test_cohens_kappa_tanimsiz_durumu_null_doner():
    kappa, not_ = mh.cohens_kappa(["1.1"] * 5, ["1.1"] * 5)
    assert kappa is None and "tanımsız" in not_


def test_external_consensus_ve_grok_split_paydalari_ayridir():
    manifest = {"experiment": "x", "dataset_fingerprint": "fp",
                "records": [{"record_id": f"r{i}", "selection_type": mh.SELECTION_STRATIFIED}
                            for i in range(4)]}
    kor = {}
    for annot in ("annotator_a", "annotator_b"):
        kor[annot] = {"content_sha256": "h", "content": {"labels": [
            {"record_id": f"r{i}", "primary_mode": "1.1", "secondary_modes": [],
             "confidence": "high", "rationale": "r", "insufficient_context": False}
            for i in range(4)]}}
    adjudication = {"content": {"labels": [
        {"record_id": f"r{i}", "human_adjudicated_label": _label()} for i in range(4)]}}
    judge_label = _label()
    sample = [{"record_id": f"r{i}", "source_run_id": f"run{i}",
               "external_agreement_level": "consensus" if i < 2 else "split",
               "external_consensus_label": "1.1" if i < 2 else None,
               "external_judge_records": [judge_label, judge_label],
               "self_judge_record": judge_label,
               "self_judge_available": True,
               "adjudicator_required": i >= 2}
              for i in range(4)]
    ai_adj = {f"run{i}": {"adjudicated_primary_mode": "1.1",
                          "adjudicated_insufficient_context": False}
              for i in (2, 3)}
    ozet = mh.build_agreement_summary(manifest, kor, adjudication, sample, ai_adj)
    assert ozet["human_final_vs_external_consensus_n_comparable"] == 2
    assert ozet["human_final_vs_external_consensus_undefined_count"] == 2
    assert ozet["human_final_vs_external_consensus_agreement"] == 1.0
    assert ozet["human_final_vs_grok_split_n_comparable"] == 2
    assert ozet["human_final_vs_grok_split_agreement"] == 1.0
    assert ozet["self_vs_external_consensus_n_comparable"] == 2
    assert ozet["blind_annotator_a_vs_external_consensus_agreement"] == 1.0
    assert ozet["blind_annotator_b_vs_grok_split_agreement"] == 1.0


def test_uyum_ozeti_genelleme_uyarisini_tasir(tmp_path, monkeypatch):
    ozet = _tam_zincir(tmp_path, monkeypatch)
    assert "genellenemez" in ozet["generalization_note"]
    assert "Cohen kappa" in ozet["generalization_note"]
    assert "κ" not in ozet["generalization_note"]
    assert ozet["n_records"] > 0
    assert ozet["human_human_exact_primary_agreement"] == 0.0
    assert set(ozet["annotators"]) == {"annotator_a", "annotator_b"}
    assert ozet["changed_from_blind_to_final"]["annotator_a"] >= 0


def _tam_zincir(tmp_path, monkeypatch):
    """örneklem → iki kör paket → iki kilit → adjudication → nihai kilit → özet."""
    manifest = _hazirla(tmp_path, monkeypatch, karar=_anlasmazlik)["manifest"]
    _iki_kilit(tmp_path, manifest)
    payload = mh.prepare_adjudication(tmp_path, expected_judges=JUDGES)["payload"]
    (tmp_path / "final.json").write_text(
        json.dumps(_final_export(manifest, payload)), encoding="utf-8")
    mh.lock_adjudication(tmp_path, tmp_path / "final.json", expected_judges=JUDGES)
    mh.prepare_diagnostic(tmp_path, expected_judges=JUDGES)
    return mh.summarize(tmp_path, expected_judges=JUDGES)


def test_tam_offline_zincir_butun_artefaktlari_uretir(tmp_path, monkeypatch):
    _tam_zincir(tmp_path, monkeypatch)
    mast = tmp_path / "mast"
    for ad in ("human_sample_manifest.json", "blind_annotator_a.locked.json",
               "blind_annotator_b.locked.json", "human_adjudication.locked.json",
               "agreement_summary.json", "packages/blind_annotator_a.html",
               "packages/blind_annotator_b.html", "packages/human_adjudication.html",
               "packages/post_lock_diagnostic.html"):
        assert (mast / ad).exists(), ad


# --- Leave-self-out insan hattı ters örnekleri ------------------------------

def test_kilit_oncesi_adjudication_payload_html_ve_sablonda_self_YOK(
        tmp_path, monkeypatch):
    marker = "SELF_JUDGE_GIZLI_GEREKCE_9f31"

    def karar(record, model):
        base = _anlasmazlik(record, model)
        if model == record["model"]:
            return {**base, "rationale": marker}
        return base

    manifest = _hazirla(tmp_path, monkeypatch, karar=karar)["manifest"]
    _iki_kilit(tmp_path, manifest)
    result = mh.prepare_adjudication(tmp_path, expected_judges=JUDGES)
    payload = result["payload"]
    html = result["package"].read_text(encoding="utf-8")
    decoded = mh.canonical(payload)
    assert marker not in decoded
    assert marker not in html
    assert "self_judge_label" not in decoded
    assert "Annotator C" not in decoded
    assert all(len(r["external_ai_judges"]) == 2 for r in payload["records"])


def test_self_only_degisim_grok_hashini_degil_insan_snapshotini_degisir(
        tmp_path, monkeypatch):
    _kur(tmp_path, monkeypatch, karar=_anlasmazlik)
    round1 = mh.load_mast_round(tmp_path)
    sample1 = mh.collect_candidates(round1, expected_judges=JUDGES)
    adj1 = mh._current_adjudications(round1, sample1)
    snap1 = mh.ai_snapshot_digest(sample1, adj1)
    by_run1 = {c["source_run_id"]: c for c in sample1}

    path = tmp_path / "mast" / "ai_judges.jsonl"
    judges = [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines()]
    changed_run = next(c["source_run_id"] for c in sample1
                       if c["external_agreement_level"] == "split")
    for row in judges:
        if row["source_run_id"] == changed_run and row["judge_model"] == MODEL:
            row["primary_mode"] = "3.3"
            row["rationale"] = "self-only değişim"
    path.write_text("".join(json.dumps(r) + "\n" for r in judges), encoding="utf-8")
    _yeniden_panel(tmp_path, monkeypatch, judges)

    round2 = mh.load_mast_round(tmp_path)
    sample2 = mh.collect_candidates(round2, expected_judges=JUDGES)
    adj2 = mh._current_adjudications(round2, sample2)
    snap2 = mh.ai_snapshot_digest(sample2, adj2)
    by_run2 = {c["source_run_id"]: c for c in sample2}
    assert by_run1[changed_run]["decision_input_sha256"] == \
           by_run2[changed_run]["decision_input_sha256"]
    assert by_run1[changed_run]["full_panel_input_sha256"] != \
           by_run2[changed_run]["full_panel_input_sha256"]
    assert snap1 != snap2


def test_external_degisim_karar_hashini_ve_insan_snapshotini_degisir(
        tmp_path, monkeypatch):
    _kur(tmp_path, monkeypatch, karar=_anlasmazlik)
    round1 = mh.load_mast_round(tmp_path)
    sample1 = mh.collect_candidates(round1, expected_judges=JUDGES)
    by_run1 = {c["source_run_id"]: c for c in sample1}
    changed_run = next(c["source_run_id"] for c in sample1
                       if c["external_agreement_level"] == "split")

    path = tmp_path / "mast" / "ai_judges.jsonl"
    judges = [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines()]
    external_model = next(m for m in JUDGES if m != MODEL)
    for row in judges:
        if row["source_run_id"] == changed_run and row["judge_model"] == external_model:
            row["primary_mode"] = "3.2"
            row["rationale"] = "external değişim"
    path.write_text("".join(json.dumps(r) + "\n" for r in judges), encoding="utf-8")
    _yeniden_panel(tmp_path, monkeypatch, judges)
    round2 = mh.load_mast_round(tmp_path)
    sample2 = mh.collect_candidates(round2, expected_judges=JUDGES)
    by_run2 = {c["source_run_id"]: c for c in sample2}
    assert by_run1[changed_run]["decision_input_sha256"] != \
           by_run2[changed_run]["decision_input_sha256"]


def test_external_split_guncel_grok_olmadan_paket_uretilemez(tmp_path, monkeypatch):
    manifest = _hazirla(tmp_path, monkeypatch, karar=_anlasmazlik)["manifest"]
    _iki_kilit(tmp_path, manifest)
    (tmp_path / "mast" / "ai_adjudication.jsonl").write_text("", encoding="utf-8")
    with pytest.raises(mh.HumanRoundError, match="güncel AI adjudication yok"):
        mh.prepare_adjudication(tmp_path, expected_judges=JUDGES)


def test_external_consensuste_grok_aranmaz_ve_gosterilmez(tmp_path, monkeypatch):
    manifest = _hazirla(tmp_path, monkeypatch)["manifest"]
    _iki_kilit(tmp_path, manifest)
    payload = mh.prepare_adjudication(tmp_path, expected_judges=JUDGES)["payload"]
    assert all(r["external_agreement_level"] == "consensus" for r in payload["records"])
    assert all(r["grok_adjudication"] is None for r in payload["records"])


def test_eski_insan_1_0_manifesti_otomatik_migrate_edilmez(tmp_path, monkeypatch):
    manifest = _hazirla(tmp_path, monkeypatch, karar=_anlasmazlik)["manifest"]
    path = tmp_path / "mast" / "human_sample_manifest.json"
    old = dict(manifest, human_schema_version="1.0", app_version="1.0")
    path.write_text(json.dumps(old), encoding="utf-8")
    before = path.read_bytes()
    with pytest.raises(mh.HumanRoundError, match="farklı bir sürüm"):
        mh.load_sample_manifest(tmp_path)
    assert path.read_bytes() == before


def test_self_diagnostic_yalniz_guncel_nihai_kilit_sonrasinda_acilir(
        tmp_path, monkeypatch):
    marker = "POST_LOCK_SELF_DIAGNOSTIC_41ab"

    def karar(record, model):
        base = _anlasmazlik(record, model)
        return {**base, "rationale": marker} if model == record["model"] else base

    manifest = _hazirla(tmp_path, monkeypatch, karar=karar)["manifest"]
    _iki_kilit(tmp_path, manifest)
    with pytest.raises(mh.HumanRoundError, match="yalnız nihai insan kararı"):
        mh.prepare_diagnostic(tmp_path, expected_judges=JUDGES)

    payload = mh.prepare_adjudication(tmp_path, expected_judges=JUDGES)["payload"]
    final_path = tmp_path / "final.json"
    final_path.write_text(json.dumps(_final_export(manifest, payload)), encoding="utf-8")
    mh.lock_adjudication(tmp_path, final_path, expected_judges=JUDGES)
    diagnostic = mh.prepare_diagnostic(tmp_path, expected_judges=JUDGES)
    assert marker in mh.canonical(diagnostic["payload"])
    assert marker in _payload(diagnostic["package"].read_text(encoding="utf-8"))["records"][0]["self_judge_label"]["rationale"] or marker in mh.canonical(diagnostic["payload"])
    assert diagnostic["payload"]["phase"] == mh.PHASE_DIAGNOSTIC


# --- CLI ve dosya anlık görüntüsü yardımcıları -------------------------------

DIZIN_SENTINEL = "__dir__"


def _dosya_kumesi(kok: Path) -> dict:
    """yol -> içerik SHA-256 (dizinler ayrı sentinel).

    Yalnız YOL kümesini karşılaştırmak "hiçbir şey yazılmadı" iddiasını
    kanıtlamaz: mevcut bir dosyanın (ör. mast/human_sample_manifest.json ya da
    kilitli bir kör tur) İÇERİĞİ üzerine yazılsa yol kümesi değişmezdi.
    """
    anlik = {}
    for p in kok.rglob("*"):
        anahtar = str(p.relative_to(kok))
        anlik[anahtar] = (DIZIN_SENTINEL if p.is_dir()
                          else hashlib.sha256(p.read_bytes()).hexdigest())
    return anlik


def test_anlik_goruntu_icerik_degisimini_yakalar(tmp_path):
    # Embargo testlerinin dayandığı yardımcı KENDİSİ sınanır: yalnız yol kümesi
    # karşılaştırsaydı, aşağıdaki üzerine yazma fark edilmezdi.
    (tmp_path / "alt").mkdir()
    hedef = tmp_path / "alt" / "kilit.json"
    hedef.write_text("eski", encoding="utf-8")
    once = _dosya_kumesi(tmp_path)
    hedef.write_text("yeni", encoding="utf-8")
    assert _dosya_kumesi(tmp_path) != once
    assert set(_dosya_kumesi(tmp_path)) == set(once), "yol kümesi aynı kaldı"


def test_insan_CLI_yardimi_calismaya_devam_eder(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["mast_human", "--help"])
    with pytest.raises(SystemExit) as e:
        mh.main()
    assert e.value.code == 0
    cikti = capsys.readouterr().out
    assert "prepare-blind" in cikti and "summarize" in cikti
    assert "prepare-diagnostic" in cikti


# --- Self-judge NULLABLE (2026-08-03 hotfix, insan şeması 2.1) ---------------
#
# Tasarım invariantı: self-judge YALNIZ TANISALDIR. Kör insan etiketi, dış karar
# ve nihai insan kararı için gerekli değildir. Eksik bir self yüzünden kaydı
# örneklem EVRENİNDEN düşürmek, evreni tanısal bir AI çıktısının başarısına
# koşullandırırdı — canlı turda Gemini'nin bir self etiketi üç denemede
# doldurulamadı ve eski kapı 148 kaydın TAMAMI için paket üretmeyi reddetti.

def _self_eksik(record, model):
    """t002/naive/r1 kaydında SELF (üretici) etiketi hiç üretilmemiş sayılır.

    Bu kayıt her iki `karar` fixture'ında da DIŞ KONSENSÜSTEDİR: hotfix'in
    çekirdek durumu tam olarak budur — karar tam, yalnız tanısal self eksik.
    """
    return (record["task_id"] == "t002" and record["arm"] == ARM_NAIVE
            and record["repeat"] == 1 and model == MODEL)


def _dis_eksik(record, model):
    """Aynı kayıtta bir DIŞ judge eksik — karar girdisi kurulamaz."""
    return (record["task_id"] == "t001" and record["arm"] == ARM_NAIVE
            and record["repeat"] == 0 and model != MODEL)


def test_self_eksik_kayit_ADAYDIR_ve_zorunlu_degildir(tmp_path, monkeypatch):
    _kur(tmp_path, monkeypatch, atla=_self_eksik)
    round_ = mh.load_mast_round(tmp_path)
    adaylar = mh.collect_candidates(round_, expected_judges=JUDGES)
    hedef = [c for c in adaylar if not c["self_judge_available"]]
    assert len(hedef) == 1, "self eksik kayıt aday havuzunda kalmalı"
    c = hedef[0]
    assert c["self_judge_record"] is None
    assert c["self_matches_external"] is None     # tanımsız, "farklı" DEĞİL
    assert c["external_agreement_level"] == "consensus"
    assert c["full_panel_input_sha256"] is None   # tanısal üçlü hash tanımsız
    assert c["decision_input_sha256"] is not None # karar girdisi TAM
    # Zorunlu küme kriterlerinin hiçbirini tetiklemez.
    secim = mh.select_human_sample(adaylar, target=0)
    assert c["record_id"] not in {s["record_id"] for s in secim}


def test_aday_evreni_self_eksik_kayitla_AYNI_kalir(tmp_path, monkeypatch):
    """Kapsam düzeltmesinin çekirdeği: evren 1 eksilmemeli."""
    _kur(tmp_path, monkeypatch)
    tam = len(mh.collect_candidates(mh.load_mast_round(tmp_path),
                                    expected_judges=JUDGES))
    shutil.rmtree(tmp_path)
    _kur(tmp_path, monkeypatch, atla=_self_eksik)
    eksikli = mh.collect_candidates(mh.load_mast_round(tmp_path),
                                    expected_judges=JUDGES)
    assert len(eksikli) == tam
    assert sum(not c["self_judge_available"] for c in eksikli) == 1


def test_DIS_judge_eksik_kayit_ADAY_DEGILDIR_ve_fail_fast(tmp_path, monkeypatch):
    """Dış eksikliği hâlâ fail-closed: karar girdisi kurulamaz."""
    _kur(tmp_path, monkeypatch, atla=_dis_eksik)
    round_ = mh.load_mast_round(tmp_path)
    with pytest.raises(mh.HumanRoundError, match="DIŞ judge eksik"):
        mh.collect_candidates(round_, expected_judges=JUDGES)


def test_tam_panel_davranisi_DEGISMEDI(tmp_path, monkeypatch):
    """Hiç eksik yokken bütün adaylar self taşır (DeepSeek biçimi)."""
    _kur(tmp_path, monkeypatch)
    adaylar = mh.collect_candidates(mh.load_mast_round(tmp_path),
                                    expected_judges=JUDGES)
    assert adaylar and all(c["self_judge_available"] for c in adaylar)
    assert all(c["self_judge_record"] is not None for c in adaylar)
    assert all(c["full_panel_input_sha256"] is not None for c in adaylar)


def _self_eksik_zincir(tmp_path, monkeypatch):
    """self-eksik kaydı ÖRNEKLEME ZORLA sokup tam zinciri koşturur."""
    def karar(record, model):
        # t001'de üçüncü judge ayrışır -> zorunlu küme; t002'de yetersiz bağlam.
        return _anlasmazlik(record, model)
    _kur(tmp_path, monkeypatch, karar=karar, atla=_self_eksik)
    # target'ı büyük tutarak tabakalı kontenjanın self-eksik kaydı da almasını sağla.
    sonuc = mh.prepare_blind(tmp_path, expected_judges=JUDGES,
                             target=999)
    return sonuc["manifest"]


def test_self_eksik_kayit_secilirse_kor_tur_ve_adjudication_TAMAMLANIR(
        tmp_path, monkeypatch):
    manifest = _self_eksik_zincir(tmp_path, monkeypatch)
    eksik_satir = [r for r in manifest["records"] if not r["self_judge_available"]]
    assert len(eksik_satir) == 1, "self-eksik kayıt örnekleme girmeli"
    _iki_kilit(tmp_path, manifest)
    payload = mh.prepare_adjudication(tmp_path, expected_judges=JUDGES)["payload"]
    # Adjudication paketi self olmadan kurulur ve self'i GÖSTERMEZ.
    ham = json.dumps(payload, ensure_ascii=False)
    assert "self_judge" not in ham and "gerekce" in ham
    (tmp_path / "final.json").write_text(
        json.dumps(_final_export(manifest, payload)), encoding="utf-8")
    kilit = mh.lock_adjudication(tmp_path, tmp_path / "final.json",
                                 expected_judges=JUDGES)
    assert len(kilit["content"]["labels"]) == len(manifest["records"])


def test_diagnostic_paket_self_olmadan_NULL_SAFE_calisir(tmp_path, monkeypatch):
    manifest = _self_eksik_zincir(tmp_path, monkeypatch)
    _iki_kilit(tmp_path, manifest)
    payload = mh.prepare_adjudication(tmp_path, expected_judges=JUDGES)["payload"]
    (tmp_path / "final.json").write_text(
        json.dumps(_final_export(manifest, payload)), encoding="utf-8")
    mh.lock_adjudication(tmp_path, tmp_path / "final.json", expected_judges=JUDGES)
    tani = mh.prepare_diagnostic(tmp_path, expected_judges=JUDGES)["payload"]
    eksik = [r for r in tani["records"] if not r["self_judge_available"]]
    assert len(eksik) == 1
    assert eksik[0]["self_judge_label"] is None, "boş etiket nesnesi uydurulmamalı"
    assert all(r["self_judge_label"] is not None
               for r in tani["records"] if r["self_judge_available"])
    html = (tmp_path / "mast/packages/post_lock_diagnostic.html").read_text(
        encoding="utf-8")
    assert "self etiketi mevcut değil" in html


def test_agreement_summary_self_paydasini_azaltir_ve_eksigi_sayar(
        tmp_path, monkeypatch):
    manifest = _self_eksik_zincir(tmp_path, monkeypatch)
    _iki_kilit(tmp_path, manifest)
    payload = mh.prepare_adjudication(tmp_path, expected_judges=JUDGES)["payload"]
    (tmp_path / "final.json").write_text(
        json.dumps(_final_export(manifest, payload)), encoding="utf-8")
    mh.lock_adjudication(tmp_path, tmp_path / "final.json", expected_judges=JUDGES)
    ozet = mh.summarize(tmp_path, expected_judges=JUDGES)

    assert ozet["self_judge_missing_count"] == 1
    # Self-eksik kayıt dış KONSENSÜSTEDİR: consensus paydasında sayılır ama
    # self karşılaştırma paydasında sayılmaz -> ikisi tam olarak 1 fark eder.
    consensus_n = ozet["human_final_vs_external_consensus_n_comparable"]
    assert ozet["self_vs_external_consensus_n_comparable"] == consensus_n - 1


def test_iki_annotator_ayni_kayit_kumesini_tasir_self_eksikken(
        tmp_path, monkeypatch):
    manifest = _self_eksik_zincir(tmp_path, monkeypatch)
    idler = {}
    for annot in manifest["annotators"]:
        html = (tmp_path / f"mast/packages/blind_{annot}.html").read_text(
            encoding="utf-8")
        p = _payload(html)
        idler[annot] = [r["record_id"] for r in p["records"]]
        # Kör pakette self ile ilgili HİÇBİR alan bulunmaz (açık izin listesi).
        assert "self_judge" not in json.dumps(p, ensure_ascii=False)
    a, b = manifest["annotators"]
    assert idler[a] == idler[b] == [r["record_id"] for r in manifest["records"]]
