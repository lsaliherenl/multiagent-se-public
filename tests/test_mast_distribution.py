"""analysis/mast_distribution.py birim testleri — LLM'siz, tamamen sentetik.

Panel ve judge etiketleri GERÇEK üretim yollarından kurulur
(`eval.mast_labels.build_panel`, `judge_record` zarfı, `panel_verdict`): elle
kurulmuş bir panel sözlüğü, kayıt üreten kodun invariantlarını atlayıp testi
kendi kendini onaylayan hale getirirdi.

Amaç: iki paydanın karışması, self/majority etiketinin karara sızması, panel
kurulduktan SONRA değişen etiketlerin fark edilmemesi, stale/şema dışı
adjudication'ın karar sayılması, ÖN ETİKETLEME turundan formal rapor üretilmesi
ve `insufficient_context`'in bir MAST modu gibi sayılması gibi SESSİZ hataları
kilitlemek.
"""

import pytest

from analysis.mast_distribution import (
    DECISION_ADJUDICATED,
    DECISION_EXTERNAL_CONSENSUS,
    DECISION_UNDECIDED_INCOMPLETE,
    DECISION_UNDECIDED_SPLIT,
    MastDistributionError,
    build_distribution,
    check_mast_provenance,
    decision_label,
    load_mast,
    verify_mast_artifacts,
    write_distribution_outputs,
)
from config import (
    ALL_ARMS,
    ARM_BASELINE,
    ARM_CONTRACT,
    ARM_NAIVE,
    ARM_STRUCTURED,
    LLM_CALL_SCHEMA_VERSION,
    MAST_DECISION_RULE_VERSION,
    MAST_PANEL_HASH_VERSION,
    MAST_SCHEMA_VERSION,
    MODEL_ADJUDICATOR,
    MODEL_JUDGES,
    MODEL_MAIN,
    MODEL_PRODUCERS,
    RESULT_SCHEMA_VERSION,
)
from eval.mast_labels import (
    build_evidence,
    build_panel,
    external_annotator_order,
    labelable_records,
    prompt_contract_hash,
    source_manifest_fingerprint,
)
from eval.mast_schema import (
    INSUFFICIENT_SENTINEL,
    MastLabel,
    evidence_digest,
    judge_role_partition,
    label_provenance,
    make_judge_record,
)
from eval.result_schema import make_run_error_record, make_synthetic_record

MODEL = MODEL_MAIN
TASKS = ["t00", "t01", "t02"]
REPEATS = 2
ARMS = list(ALL_ARMS)
EXPERIMENT = "ana"
TASK = {"task_id": "t00", "prompt": "def f(x):\n    ...", "entry_point": "f"}

# Oylar ROL üzerinden kurulur, panel sırasına göre DEĞİL: `MODEL_JUDGES` sırası
# değişirse konumsal bir yardımcı testleri sessizce başka bir senaryoya çevirirdi
# (self oyu dış oy sanılır ve tam da leave-self-out'un ölçtüğü şey kaybolur).
EXTERNALS = tuple(m for m in MODEL_JUDGES if m != MODEL)


def _oy(self_vote, external_a, external_b):
    """{judge_model: oy} — self kaydı üreten model, diğer ikisi dış judge."""
    return {MODEL: self_vote, EXTERNALS[0]: external_a, EXTERNALS[1]: external_b}


def _main_manifest(**over):
    manifest = {"name": EXPERIMENT, "model": MODEL, "task_set": "heldout",
                "task_ids": list(TASKS), "arm_order": list(ARMS), "repeats": REPEATS,
                "result_schema_version": RESULT_SCHEMA_VERSION,
                "llm_call_schema_version": LLM_CALL_SCHEMA_VERSION,
                "git_commit": "deadbeef", "prompt_contract_hash": "abc",
                "arm_rotation_scheme": "v1",
                "task_file_hashes": {f"{t}.json": f"h-{t}" for t in TASKS}}
    manifest.update(over)
    return manifest


def _mast_manifest(main_manifest=None, **over):
    main_manifest = main_manifest or _main_manifest()
    manifest = {
        "experiment": main_manifest["name"],
        "mast_schema_version": MAST_SCHEMA_VERSION,
        "mast_decision_rule_version": MAST_DECISION_RULE_VERSION,
        "mast_panel_hash_version": MAST_PANEL_HASH_VERSION,
        "judges": list(MODEL_JUDGES), "adjudicator_model": MODEL_ADJUDICATOR,
        "mast_prompt_hash": prompt_contract_hash(),
        "source_manifest_fingerprint": source_manifest_fingerprint(main_manifest),
        "source_model": main_manifest["model"],
        "source_task_set": main_manifest["task_set"],
        "git_commit": "deadbeef", "judge_temperature": 0.0,
        "preliminary": False, "source_results_complete": True, "missing_runs": 0,
    }
    manifest.update(over)
    return manifest


def _records(failures, *, model=MODEL):
    """failures: {(task_id, arm, repeat)} -> plus_pass=False olan koşular."""
    out = []
    for rep in range(REPEATS):
        for task_id in TASKS:
            for arm in ARMS:
                basarisiz = (task_id, arm, rep) in failures
                record = make_synthetic_record(
                    experiment=EXPERIMENT, model=model, arm=arm, task_id=task_id,
                    repeat=rep, run_id=f"{task_id}-{arm}-{rep}",
                    base_pass=True, plus_pass=not basarisiz)
                record.setdefault("code", "def f(x):\n    return 0\n")
                record.setdefault("traceback", "AssertionError: Error")
                record.setdefault("raw_messages",
                                  [{"from": "planner", "to": "coder", "content": "p"}])
                out.append(record)
    return out


def _judge(record, judge_model, mode):
    """GERÇEK kanıt hash'i ve prompt hash'iyle judge kaydı (mod None = yetersiz)."""
    evidence = build_evidence(record, TASK)
    label = (MastLabel(confidence="low", rationale="yetersiz", insufficient_context=True)
             if mode is None else
             MastLabel(primary_mode=mode, confidence="high", rationale="x"))
    return make_judge_record(
        record, experiment=EXPERIMENT, evidence_sha256=evidence_digest(evidence),
        judge_model=judge_model, judge_attempt=1, judge_status="ok",
        prompt_hash=prompt_contract_hash(),
        interaction_type=evidence["interaction_type"], label=label)


def _kurulum(monkeypatch, failures, votes_by_run, *, missing_by_run=None,
             model=MODEL):
    """(ana manifest, records, mast manifest, judges, panel) — hepsi gerçek yoldan."""
    monkeypatch.setattr("eval.mast_labels._task_for", lambda r: TASK)
    ana = _main_manifest(model=model)
    records = _records(failures, model=model)
    hedef = labelable_records(records)
    missing_by_run = missing_by_run or {}
    judges = []
    for record in hedef:
        votes = votes_by_run[record["run_id"]]
        atlanan = missing_by_run.get(record["run_id"], ())
        judges += [_judge(record, m, votes[m])
                   for m in MODEL_JUDGES if m not in atlanan]
    panel = build_panel(hedef, judges, experiment=EXPERIMENT)
    return ana, records, _mast_manifest(ana), judges, panel


def _external_records(judges, record):
    _self, external = judge_role_partition(record["model"], MODEL_JUDGES)
    by_model = {j["judge_model"]: j for j in judges
                if j["source_run_id"] == record["run_id"]}
    return [by_model[m] for m in external if m in by_model]


def _adjudication(record, panel_row, judges, mode, *, status="ok", **over):
    """MastAdjudication sözleşmesine uyan GERÇEK biçimli karar kaydı."""
    dis = _external_records(judges, record)
    self_judge, external = judge_role_partition(record["model"], MODEL_JUDGES)
    sirali = external_annotator_order(dis, record["run_id"])
    karar = {
        "decision_rule_version": MAST_DECISION_RULE_VERSION,
        "source_model": record["model"],
        "self_judge_model": self_judge,
        "external_judges": list(external),
        "external_agreement_level": "split",
        "reviewed_judges": list(external),
        "annotator_assignment": {h: r["judge_model"] for h, r in zip("AB", sirali)},
        "decision_input_sha256": panel_row["decision_input_sha256"],
        "interaction_type": panel_row["interaction_type"],
        "adjudicator_model": MODEL_ADJUDICATOR,
        "adjudicator_attempt": 1,
        "adjudicator_status": status,
    }
    if status == "ok":
        karar.update(adjudicated_primary_mode=mode, adjudicated_secondary_modes=[],
                     adjudicated_confidence="high", adjudicated_rationale="x",
                     adjudicated_insufficient_context=mode is None)
    row = {**label_provenance(record, experiment=EXPERIMENT,
                              evidence_sha256=panel_row["evidence_sha256"],
                              prompt_hash=panel_row["mast_prompt_hash"],
                              interaction_type=panel_row["interaction_type"]),
           **karar}
    row.update(over)
    return row


def _panel_of(panel, run_id):
    return next(p for p in panel if p["source_run_id"] == run_id)


# --- Provenance kapıları ------------------------------------------------------

def test_ayni_deney_temiz_gecer():
    ana = _main_manifest()
    assert check_mast_provenance(_mast_manifest(ana), ana) == []


def test_farkli_karar_kurali_reddedilir():
    ana = _main_manifest()
    sorunlar = check_mast_provenance(
        _mast_manifest(ana, mast_decision_rule_version="majority_v0"), ana)
    assert any("karar kuralı" in s for s in sorunlar)


def test_farkli_mast_semasi_reddedilir():
    ana = _main_manifest()
    sorunlar = check_mast_provenance(_mast_manifest(ana, mast_schema_version="3.0"), ana)
    assert any("MAST şeması" in s for s in sorunlar)


def test_farkli_kaynak_model_havuzlanmayi_reddeder():
    ana = _main_manifest()
    baska = [m for m in MODEL_PRODUCERS if m != MODEL][0]
    sorunlar = check_mast_provenance(_mast_manifest(ana, source_model=baska), ana)
    assert any("HAVUZLANMAZ" in s for s in sorunlar)


def test_fingerprint_uyusmazligi_reddedilir():
    ana = _main_manifest()
    sorunlar = check_mast_provenance(
        _mast_manifest(ana, source_manifest_fingerprint="0" * 64), ana)
    assert any("fingerprint" in s for s in sorunlar)


def test_manifestsiz_mast_dizini_reddedilir(tmp_path):
    (tmp_path / "ai_panel.jsonl").write_text("{}\n", encoding="utf-8")
    with pytest.raises(MastDistributionError, match="manifesti yok"):
        load_mast(tmp_path)


def test_panelsiz_mast_dizini_reddedilir(tmp_path):
    (tmp_path / "manifest.json").write_text("{}", encoding="utf-8")
    with pytest.raises(MastDistributionError, match="panel kaydı yok"):
        load_mast(tmp_path)


def test_judge_dosyasi_olmayan_tur_reddedilir(tmp_path):
    """Panel tazeliği ham etiketler olmadan doğrulanamaz."""
    (tmp_path / "manifest.json").write_text("{}", encoding="utf-8")
    (tmp_path / "ai_panel.jsonl").write_text('{"source_run_id": "x"}\n',
                                             encoding="utf-8")
    with pytest.raises(MastDistributionError, match="judge etiketi yok"):
        load_mast(tmp_path)


# --- Panel tazeliği (saklanmış bayraklara GÜVENİLMEZ) -------------------------

def _tek_kayit(monkeypatch, votes, **kw):
    failures = {("t00", ARM_NAIVE, 0)}
    return _kurulum(monkeypatch, failures, {"t00-naive-0": votes}, **kw)


def test_saglam_panel_yeniden_turetilebilir(monkeypatch):
    ana, records, mast, judges, panel = _tek_kayit(
        monkeypatch, _oy("1.1", "1.1", "1.1"))
    verified, adj, sayaclar = verify_mast_artifacts(
        labelable_records(records), judges, panel, [], experiment=EXPERIMENT)
    assert set(verified) == {"t00-naive-0"}
    assert verified["t00-naive-0"].verdict["external_consensus_label"] == "1.1"
    assert adj == {} and sayaclar["current_ok"] == 0


def test_panel_sonrasi_DIS_etiket_degisimi_yakalanir(monkeypatch):
    """Saklanmış evidence_consistent=True bu durumu gizlerdi."""
    ana, records, mast, judges, panel = _tek_kayit(
        monkeypatch, _oy("1.1", "1.1", "1.1"))
    dis = next(j for j in judges if j["judge_model"] == EXTERNALS[0])
    dis["primary_mode"] = "2.3"    # panel kurulduktan SONRA değişti
    with pytest.raises(MastDistributionError, match="DIŞ judge"):
        verify_mast_artifacts(labelable_records(records), judges, panel, [],
                              experiment=EXPERIMENT)


def test_panel_sonrasi_SELF_etiket_degisimi_tanisal_olarak_yakalanir(monkeypatch):
    """Self değişimi kararı bozmaz ama panel yine de bayatlamıştır."""
    ana, records, mast, judges, panel = _tek_kayit(
        monkeypatch, _oy("1.1", "1.1", "1.1"))
    self_j = next(j for j in judges if j["judge_model"] == MODEL)
    self_j["primary_mode"] = "3.1"
    with pytest.raises(MastDistributionError, match="TANISAL"):
        verify_mast_artifacts(labelable_records(records), judges, panel, [],
                              experiment=EXPERIMENT)


def test_bayatlamis_judge_kaydi_eksik_panel_sayilir(monkeypatch):
    ana, records, mast, judges, panel = _tek_kayit(
        monkeypatch, _oy("1.1", "1.1", "1.1"))
    # Kanıt hash'i değişmiş bir etiket artık GÜNCEL değildir.
    judges[0]["evidence_sha256"] = "0" * 64
    with pytest.raises(MastDistributionError, match="güncel panel eksik"):
        verify_mast_artifacts(labelable_records(records), judges, panel, [],
                              experiment=EXPERIMENT)


def test_elle_true_yapilmis_evidence_consistent_yakalanir(monkeypatch):
    ana, records, mast, judges, panel = _tek_kayit(
        monkeypatch, _oy("1.1", "1.1", "1.1"))
    panel[0]["evidence_consistent"] = False   # dosyadaki iddia yeniden hesaplanır
    with pytest.raises(MastDistributionError, match="evidence_consistent"):
        verify_mast_artifacts(labelable_records(records), judges, panel, [],
                              experiment=EXPERIMENT)


def test_eksik_panel_satiri_fail_fast(monkeypatch):
    failures = {("t00", ARM_BASELINE, 0), ("t01", ARM_NAIVE, 0)}
    votes = {"t00-baseline-0": _oy("1.1", "1.1", "1.1"),
             "t01-naive-0": _oy("1.1", "1.1", "1.1")}
    ana, records, mast, judges, panel = _kurulum(monkeypatch, failures, votes)
    with pytest.raises(MastDistributionError, match="eksik panel"):
        verify_mast_artifacts(labelable_records(records), judges, panel[:1], [],
                              experiment=EXPERIMENT)


def test_yabanci_panel_satiri_fail_fast(monkeypatch):
    ana, records, mast, judges, panel = _tek_kayit(
        monkeypatch, _oy("1.1", "1.1", "1.1"))
    sahte = {**panel[0], "source_run_id": "yok-boyle-bir-kosu"}
    with pytest.raises(MastDistributionError, match="olmayan satır"):
        verify_mast_artifacts(labelable_records(records), judges, [*panel, sahte], [],
                              experiment=EXPERIMENT)


def test_sema_disi_panel_satiri_fail_fast(monkeypatch):
    ana, records, mast, judges, panel = _tek_kayit(
        monkeypatch, _oy("1.1", "1.1", "1.1"))
    panel[0]["agreement_level"] = "majority"   # üç oy aynıyken graf olarak imkânsız
    with pytest.raises(MastDistributionError, match="geçersiz"):
        verify_mast_artifacts(labelable_records(records), judges, panel, [],
                              experiment=EXPERIMENT)


# --- Karar etiketi ------------------------------------------------------------

def test_dis_konsensus_karari_verir(monkeypatch):
    ana, records, mast, judges, panel = _tek_kayit(
        monkeypatch, _oy("2.3", "1.1", "1.1"))
    # Self 2.3 dedi ama iki dış judge 1.1'de uzlaştı: karar 1.1'dir.
    etiket, kaynak = decision_label(panel[0], None)
    assert (etiket, kaynak) == ("1.1", DECISION_EXTERNAL_CONSENSUS)
    assert panel[0]["majority_label"] == "1.1"
    assert panel[0]["self_matches_external"] is False


def test_self_etiketi_karari_belirlemez(monkeypatch):
    ana, r1, mast, j1, p1 = _tek_kayit(monkeypatch, _oy("2.3", "1.1", "1.1"))
    ana2, r2, mast2, j2, p2 = _tek_kayit(monkeypatch, _oy("3.1", "1.1", "1.1"))
    assert decision_label(p1[0], None) == decision_label(p2[0], None)


def test_dis_split_adjudication_olmadan_kararsiz(monkeypatch):
    ana, records, mast, judges, panel = _tek_kayit(
        monkeypatch, _oy("1.1", "1.1", "2.3"))
    etiket, kaynak = decision_label(panel[0], None)
    assert etiket is None and kaynak == DECISION_UNDECIDED_SPLIT


def test_dis_split_grok_karariyla_cozulur(monkeypatch):
    ana, records, mast, judges, panel = _tek_kayit(
        monkeypatch, _oy("1.1", "1.1", "2.3"))
    kayit = _adjudication(labelable_records(records)[0], panel[0], judges, "2.3")
    etiket, kaynak = decision_label(panel[0], kayit)
    assert etiket == "2.3" and kaynak == DECISION_ADJUDICATED


def test_eksik_panel_kararsiz_sayilir(monkeypatch):
    ana, records, mast, judges, panel = _tek_kayit(
        monkeypatch, _oy("1.1", "1.1", "2.3"),
        missing_by_run={"t00-naive-0": (EXTERNALS[1],)})
    etiket, kaynak = decision_label(panel[0], None)
    assert etiket is None and kaynak == DECISION_UNDECIDED_INCOMPLETE


# --- Adjudication doğrulaması -------------------------------------------------

def _split_kurulum(monkeypatch):
    ana, records, mast, judges, panel = _tek_kayit(
        monkeypatch, _oy("1.1", "1.1", "2.3"))
    hedef = labelable_records(records)
    return ana, records, mast, judges, panel, hedef[0]


def test_gecerli_adjudication_guncel_sayilir(monkeypatch):
    ana, records, mast, judges, panel, rec = _split_kurulum(monkeypatch)
    kayit = _adjudication(rec, panel[0], judges, "2.3")
    _v, guncel, sayaclar = verify_mast_artifacts(
        labelable_records(records), judges, panel, [kayit], experiment=EXPERIMENT)
    assert set(guncel) == {rec["run_id"]} and sayaclar["current_ok"] == 1


def test_stale_adjudication_guncel_sayilmaz(monkeypatch):
    ana, records, mast, judges, panel, rec = _split_kurulum(monkeypatch)
    bayat = _adjudication(rec, panel[0], judges, "2.3",
                          decision_input_sha256="f" * 64)
    _v, guncel, sayaclar = verify_mast_artifacts(
        labelable_records(records), judges, panel, [bayat], experiment=EXPERIMENT)
    assert guncel == {} and sayaclar["stale"] == 1 and sayaclar["current_ok"] == 0


def test_hatali_adjudication_denemesi_karar_sayilmaz(monkeypatch):
    ana, records, mast, judges, panel, rec = _split_kurulum(monkeypatch)
    hata = _adjudication(rec, panel[0], judges, None, status="error")
    _v, guncel, sayaclar = verify_mast_artifacts(
        labelable_records(records), judges, panel, [hata], experiment=EXPERIMENT)
    assert guncel == {} and sayaclar["error_attempts"] == 1


def test_ayni_kimlikte_iki_guncel_adjudication_fail_fast(monkeypatch):
    ana, records, mast, judges, panel, rec = _split_kurulum(monkeypatch)
    kayit = _adjudication(rec, panel[0], judges, "2.3")
    with pytest.raises(MastDistributionError, match="birden fazla başarılı"):
        verify_mast_artifacts(labelable_records(records), judges, panel,
                              [kayit, dict(kayit)], experiment=EXPERIMENT)


@pytest.mark.parametrize("alan,deger", [
    ("experiment", "BASKA-DENEY"),
    ("source_model", "openrouter/deepseek/deepseek-v4-flash"),
    ("task_set", "pilot"),
    ("task_id", "baska-gorev"),
    ("arm", ARM_BASELINE),
    ("repeat", 99),
])
def test_yanlis_provenance_adjudication_fail_fast(monkeypatch, alan, deger):
    """Dört hash eşleşse bile deney/model/görev/kol zarfı yanlış olabilir."""
    ana, records, mast, judges, panel, rec = _split_kurulum(monkeypatch)
    kayit = _adjudication(rec, panel[0], judges, "2.3", **{alan: deger})
    with pytest.raises(MastDistributionError, match="gecersiz|geçersiz"):
        verify_mast_artifacts(labelable_records(records), judges, panel, [kayit],
                              experiment=EXPERIMENT)


def test_gecersiz_mast_kodu_adjudication_fail_fast(monkeypatch):
    ana, records, mast, judges, panel, rec = _split_kurulum(monkeypatch)
    kayit = _adjudication(rec, panel[0], judges, "9.9")
    with pytest.raises(MastDistributionError, match="gecersiz|geçersiz"):
        verify_mast_artifacts(labelable_records(records), judges, panel, [kayit],
                              experiment=EXPERIMENT)


def test_eksik_sozlesme_alani_adjudication_fail_fast(monkeypatch):
    ana, records, mast, judges, panel, rec = _split_kurulum(monkeypatch)
    kayit = _adjudication(rec, panel[0], judges, "2.3")
    kayit.pop("reviewed_judges")
    with pytest.raises(MastDistributionError, match="eksik sozlesme|eksik sözleşme"):
        verify_mast_artifacts(labelable_records(records), judges, panel, [kayit],
                              experiment=EXPERIMENT)


def test_yanlis_adjudicator_modeli_fail_fast(monkeypatch):
    ana, records, mast, judges, panel, rec = _split_kurulum(monkeypatch)
    kayit = _adjudication(rec, panel[0], judges, "2.3",
                          adjudicator_model="openrouter/rogue/adjudicator")
    # Model kimliğin parçası olduğu için kayıt kendiliğinden "stale" sayılıp
    # sessizce atlanırdı; dondurulmuş adjudicator hiç değişmediğine göre bu
    # tarih değil tahrifat işaretidir ve DURDURUR.
    with pytest.raises(MastDistributionError, match="dondurulmuş adjudicator"):
        verify_mast_artifacts(labelable_records(records), judges, panel, [kayit],
                              experiment=EXPERIMENT)


def test_dis_konsensuslu_kayitta_adjudication_fail_fast(monkeypatch):
    """Adjudicator yalnız external split'te çalışabilir."""
    ana, records, mast, judges, panel = _tek_kayit(
        monkeypatch, _oy("2.3", "1.1", "1.1"))
    rec = labelable_records(records)[0]
    kayit = _adjudication(rec, panel[0], judges, "2.3")
    with pytest.raises(MastDistributionError, match="external split"):
        verify_mast_artifacts(labelable_records(records), judges, panel, [kayit],
                              experiment=EXPERIMENT)


def test_yabanci_adjudication_fail_fast(monkeypatch):
    ana, records, mast, judges, panel, rec = _split_kurulum(monkeypatch)
    kayit = _adjudication(rec, panel[0], judges, "2.3", source_run_id="BASKA-RUN")
    with pytest.raises(MastDistributionError, match="yabancı adjudication"):
        verify_mast_artifacts(labelable_records(records), judges, panel, [kayit],
                              experiment=EXPERIMENT)


# --- ÖN ETİKETLEME kapısı -----------------------------------------------------

def test_preliminary_tur_varsayilan_olarak_durdurur(monkeypatch):
    ana, records, mast, judges, panel = _tek_kayit(
        monkeypatch, _oy("1.1", "1.1", "1.1"))
    on_etiketleme = _mast_manifest(ana, preliminary=True,
                                   source_results_complete=False, missing_runs=7)
    with pytest.raises(MastDistributionError, match="ÖN ETİKETLEME"):
        build_distribution(ana, records, on_etiketleme, judges, panel, [])


def test_source_results_complete_false_tek_basina_durdurur(monkeypatch):
    ana, records, mast, judges, panel = _tek_kayit(
        monkeypatch, _oy("1.1", "1.1", "1.1"))
    yarim = _mast_manifest(ana, preliminary=False, source_results_complete=False)
    with pytest.raises(MastDistributionError, match="ÖN ETİKETLEME"):
        build_distribution(ana, records, yarim, judges, panel, [])


def test_preliminary_tur_allow_missing_ile_damgalanir(monkeypatch):
    ana, records, mast, judges, panel = _tek_kayit(
        monkeypatch, _oy("1.1", "1.1", "1.1"))
    on_etiketleme = _mast_manifest(ana, preliminary=True, missing_runs=7)
    sonuc = build_distribution(ana, records, on_etiketleme, judges, panel, [],
                               allow_missing=True)
    assert sonuc["preliminary"] is True and sonuc["mast_round_preliminary"] is True


# --- Paydalar -----------------------------------------------------------------

def _tam_kurulum(monkeypatch):
    """naive: 2 hata (ikisi 1.1) | contract: 1 hata (2.3). Diğer kollar temiz."""
    failures = {("t00", ARM_NAIVE, 0), ("t01", ARM_NAIVE, 0),
                ("t02", ARM_CONTRACT, 1)}
    votes = {
        "t00-naive-0": _oy("1.1", "1.1", "1.1"),
        "t01-naive-0": _oy("2.3", "1.1", "1.1"),
        "t02-contract-1": _oy("1.1", "2.3", "2.3"),
    }
    return _kurulum(monkeypatch, failures, votes)


def test_iki_payda_ayri_hesaplanir(monkeypatch):
    ana, records, mast, judges, panel = _tam_kurulum(monkeypatch)
    sonuc = build_distribution(ana, records, mast, judges, panel, [])
    d = sonuc["denominators"]["by_arm"]
    assert d[ARM_NAIVE]["completed_arm_runs"] == 6          # 3 görev × 2 tekrar
    assert d[ARM_NAIVE]["labeled_failures"] == 2 == d[ARM_NAIVE]["decided"]
    assert d[ARM_CONTRACT]["labeled_failures"] == 1

    assert sonuc["error_composition"][ARM_NAIVE]["1.1"] == 1.0
    assert sonuc["arm_incidence"][ARM_NAIVE]["1.1"] == round(2 / 6, 6)
    # contract'ın tek hatası da "%100" kompozisyon verir — insidans ayrımı bunu
    # yanıltıcı olmaktan çıkarır (1/6 vs 2/6).
    assert sonuc["error_composition"][ARM_CONTRACT]["2.3"] == 1.0
    assert sonuc["arm_incidence"][ARM_CONTRACT]["2.3"] == round(1 / 6, 6)


def test_hatasiz_kolun_oranlari_sifir_ve_none_ayrilir(monkeypatch):
    ana, records, mast, judges, panel = _tam_kurulum(monkeypatch)
    sonuc = build_distribution(ana, records, mast, judges, panel, [])
    # baseline'da hiç hata yok: kompozisyon paydası 0 -> None (ölçülemedi),
    # insidans paydası 6 -> 0.0 (ölçüldü ve sıfır çıktı).
    assert sonuc["error_composition"][ARM_BASELINE]["1.1"] is None
    assert sonuc["arm_incidence"][ARM_BASELINE]["1.1"] == 0.0


def test_run_error_paydaya_girmez(monkeypatch):
    ana, records, mast, judges, panel = _tam_kurulum(monkeypatch)
    temiz = build_distribution(ana, records, mast, judges, panel, [])
    hatali = make_run_error_record(
        experiment=EXPERIMENT, model=MODEL, task_set="heldout", arm=ARM_NAIVE,
        task_id="t00", repeat=0, run_id="err-1", arm_position=0, error="boom")
    kirli = build_distribution(ana, [*records, hatali], mast, judges, panel, [])
    assert temiz["denominators"] == kirli["denominators"]


def test_insufficient_context_mod_tablosuna_girmez(monkeypatch):
    # İki dış judge da yetersiz bağlam dedi -> konsensüs = insufficient sentinel.
    ana, records, mast, judges, panel = _tek_kayit(
        monkeypatch, _oy("1.1", None, None))
    sonuc = build_distribution(ana, records, mast, judges, panel, [])
    assert sonuc["arm_by_primary_mode"][ARM_NAIVE] == {}
    assert INSUFFICIENT_SENTINEL not in sonuc["primary_modes"]
    ic = sonuc["insufficient_context"]
    assert ic["decision_insufficient_count"] == 1
    assert ic["decision_insufficient_denominator"] == 1
    assert ic["any_external_judge_insufficient_count"] == 1
    assert ic["self_judge_insufficient_count"] == 0


def test_self_vs_dis_konsensus_tanisal_raporlanir(monkeypatch):
    ana, records, mast, judges, panel = _tam_kurulum(monkeypatch)
    sonuc = build_distribution(ana, records, mast, judges, panel, [])
    sv = sonuc["self_vs_external_consensus"]
    # t00: self 1.1 == dış 1.1 | t01: self 2.3 != dış 1.1 | t02: self 1.1 != dış 2.3
    assert sv["agree"] == 1 and sv["disagree"] == 2 and sv["n_comparable"] == 3


def test_karara_baglanmamis_split_varsayilan_olarak_durdurur(monkeypatch):
    ana, records, mast, judges, panel = _tek_kayit(
        monkeypatch, _oy("1.1", "1.1", "2.3"))
    with pytest.raises(MastDistributionError, match="GÜNCEL"):
        build_distribution(ana, records, mast, judges, panel, [])


def test_allow_missing_ile_preliminary_damgalanir(monkeypatch):
    ana, records, mast, judges, panel = _tek_kayit(
        monkeypatch, _oy("1.1", "1.1", "2.3"))
    sonuc = build_distribution(ana, records, mast, judges, panel, [],
                               allow_missing=True)
    assert sonuc["preliminary"] is True and sonuc["blockers"]
    assert sonuc["decision_source_counts"][DECISION_UNDECIDED_SPLIT] == 1
    assert sonuc["denominators"]["by_arm"][ARM_NAIVE]["undecided"] == 1
    assert sonuc["arm_by_primary_mode"][ARM_NAIVE] == {}


def test_modeller_havuzlanamaz(monkeypatch):
    ana, records, mast, judges, panel = _tam_kurulum(monkeypatch)
    baska = [m for m in MODEL_PRODUCERS if m != MODEL][0]
    records[0] = {**records[0], "model": baska}
    from analysis.analyze import AnalysisError
    with pytest.raises(AnalysisError, match="HAVUZLANMAZ"):
        build_distribution(ana, records, mast, judges, panel, [])


def test_eksik_arm_run_varsayilan_olarak_durdurur(monkeypatch):
    ana, records, mast, judges, panel = _tam_kurulum(monkeypatch)
    kirpik = [r for r in records
              if not (r["task_id"] == "t02" and r["arm"] == ARM_STRUCTURED)]
    with pytest.raises(MastDistributionError, match="eksik arm-run"):
        build_distribution(ana, kirpik, mast, judges, panel, [])


def test_ozet_kaynak_provenansini_tasir(monkeypatch):
    ana, records, mast, judges, panel = _tam_kurulum(monkeypatch)
    sonuc = build_distribution(ana, records, mast, judges, panel, [])
    p = sonuc["source_provenance"]
    assert p["main_git_commit"] == "deadbeef" and p["mast_git_commit"] == "deadbeef"
    assert p["mast_prompt_hash_recomputed"] == prompt_contract_hash()
    assert p["mast_prompt_hash_manifest"] == p["mast_prompt_hash_recomputed"]
    assert p["mast_source_manifest_fingerprint"] == source_manifest_fingerprint(ana)
    assert p["main_task_file_hashes_sha256"]


def test_ayni_veri_bayt_bayt_ayni_ciktiyi_uretir(monkeypatch, tmp_path):
    ana, records, mast, judges, panel = _tam_kurulum(monkeypatch)
    ilk = write_distribution_outputs(
        build_distribution(ana, records, mast, judges, panel, []), tmp_path / "a")
    ikinci = write_distribution_outputs(
        build_distribution(ana, records, mast, judges, panel, []), tmp_path / "b")
    for ad in ilk:
        assert ilk[ad].read_bytes() == ikinci[ad].read_bytes()


def test_csv_iki_orani_da_tasir(monkeypatch, tmp_path):
    ana, records, mast, judges, panel = _tam_kurulum(monkeypatch)
    sonuc = build_distribution(ana, records, mast, judges, panel, [])
    paths = write_distribution_outputs(sonuc, tmp_path)
    basliklar = paths["mast_arm_mode"].read_text(encoding="utf-8").splitlines()[0]
    assert "error_composition" in basliklar and "arm_incidence" in basliklar
    assert "decided_failures_in_arm" in basliklar
    assert "completed_arm_runs_in_arm" in basliklar


def test_dort_kol_da_tabloda_bulunur(monkeypatch):
    """P1 kapanış denetiminin fixture borcu: structured_no_validation atlanmasın."""
    ana, records, mast, judges, panel = _tam_kurulum(monkeypatch)
    sonuc = build_distribution(ana, records, mast, judges, panel, [])
    assert set(sonuc["arm_incidence"]) == set(ALL_ARMS)
    assert set(sonuc["denominators"]["by_arm"]) == set(ALL_ARMS)


def test_insan_ornekleminin_genellenemezligi_notlarda_yazili(monkeypatch):
    ana, records, mast, judges, panel = _tam_kurulum(monkeypatch)
    sonuc = build_distribution(ana, records, mast, judges, panel, [])
    assert any("GENELLENEMEZ" in n for n in sonuc["notes"])
    assert any("insidans" in n for n in sonuc["notes"])
