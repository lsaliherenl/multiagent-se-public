"""External-only Grok adjudication (§9.2, MAST 3.2) — LLM'siz, çağrılar sayılır.

Kapatılan sessiz hata: **adjudicator üç etiketi görüyordu.** Leave-self-out
self'in OYUNU karardan çıkarır, ama self'in GEREKÇESİ prompt'ta kaldığı sürece
üretici model kendi başarısızlığının hata etiketini dolaylı olarak belirlemeye
devam ederdi — panel katmanında kapatılan yol, adjudication katmanında açık
kalmış olurdu.

Bu dosyanın çekirdek iddiaları:

* Grok'a giden metinde self etiketi, self gerekçesi, "Annotator C", model
  slug'ları, kol adı ve üçlü çoğunluk kararı BULUNMAZ.
* Karar kaydı, Grok'un GERÇEKTEN gördüğü kümeyi (`reviewed_judges`) ve o kümenin
  hash'ini taşır; kayıt sözleşmesi bunları çapraz doğrular.
* Bozuk bir girdide TOPLAM çağrı sayısı sıfırdır (ön geçiş atomiktir).
"""

import json

import pytest

from config import (
    ARM_BASELINE,
    ARM_CONTRACT,
    ARM_STRUCTURED,
    MAST_DECISION_RULE_VERSION,
    MAST_SCHEMA_VERSION,
    MODEL_ADJUDICATOR,
    MODEL_JUDGE_EXTERNAL,
    MODEL_JUDGES,
    MODEL_MAIN,
    MODEL_SECONDARY,
)
from eval.mast_labels import (
    adjudicate_record,
    build_adjudicator_messages,
    build_evidence,
    build_panel,
    current_external_judges_for_record,
    external_annotator_order,
    prompt_contract_hash,
    run_adjudication,
)
from eval.mast_schema import (
    MastAdjudication,
    MastPipelineError,
    evidence_digest,
)
from eval.result_schema import make_synthetic_record

JUDGES = tuple(MODEL_JUDGES)
TASK = {"task_id": "t000", "prompt": "def f(x):\n    ...", "entry_point": "f"}


def _record(model=MODEL_MAIN, arm=ARM_CONTRACT, **extra):
    record = make_synthetic_record(experiment="sentetik", model=model, task_set="heldout",
                                   arm=arm, task_id="t000", repeat=0,
                                   base_pass=False, plus_pass=False)
    record.setdefault("code", "def f(x):\n    return 0\n")
    record.setdefault("plan", {"task_id": "t000"})
    record.setdefault("raw_messages", [{"from": "planner", "to": "coder", "content": "p"}])
    record.update(extra)
    return record


def _jr(record, model, primary="1.1", *, rationale="gerekce", insufficient=False, **kw):
    return {"judge_model": model, "judge_status": "ok", "judge_attempt": 1,
            "ts": "2026-07-29T00:00:00+00:00",
            "primary_mode": None if insufficient else primary, "secondary_modes": [],
            "confidence": "high", "rationale": rationale,
            "insufficient_context": insufficient,
            "source_run_id": record["run_id"], "experiment": "sentetik",
            "source_model": record["model"], "task_set": record["task_set"],
            "task_id": record["task_id"], "arm": record["arm"],
            "repeat": record["repeat"], "mast_schema_version": MAST_SCHEMA_VERSION,
            "evidence_sha256": evidence_digest(build_evidence(record, TASK)),
            "mast_prompt_hash": prompt_contract_hash(), **kw}


def _dis(kaynak):
    return [m for m in JUDGES if m != kaynak]


def _kur(monkeypatch, *, kaynak=MODEL_MAIN, etiketler=None, arm=ARM_CONTRACT):
    """(record, judges, panel) — varsayılan: dış judge'lar AYRIŞIR (Grok gerekir).

    `etiketler`: {judge_model: dict(_jr kwargs)} biçiminde kısmi override.
    """
    monkeypatch.setattr("eval.mast_labels._task_for", lambda r: TASK)
    record = _record(model=kaynak, arm=arm)
    d1, d2 = _dis(kaynak)
    varsayilan = {kaynak: {}, d1: {}, d2: {"primary": "2.3"}}
    over = {**varsayilan, **(etiketler or {})}
    judges = [_jr(record, m, **over.get(m, {})) for m in JUDGES]
    panel = build_panel([record], judges, experiment="sentetik", expected_judges=JUDGES)
    return record, judges, panel


class _Yanit:
    def __init__(self, text):
        self.text = text


def _sayaci(monkeypatch, text=None):
    """call_model'i sayan mock; `cagrilar` listesi prompt metinlerini tutar."""
    cagrilar = []
    govde = text if text is not None else json.dumps(
        {"primary_mode": "1.1", "secondary_modes": [], "confidence": "high",
         "rationale": "adjudicator karari", "insufficient_context": False})

    def _sahte(messages, **kw):
        cagrilar.append(json.dumps(messages, ensure_ascii=False))
        return _Yanit(govde)

    monkeypatch.setattr("eval.mast_labels.call_model", _sahte)
    return cagrilar


def _adj(record, judges, panel, existing=None, **kw):
    return run_adjudication([record], judges, panel, experiment="sentetik",
                            expected_judges=JUDGES, existing=existing, **kw)


# --- 1-2: rol bölümlemesi iki üretici için de doğru --------------------------

@pytest.mark.parametrize("kaynak", [MODEL_MAIN, MODEL_SECONDARY],
                         ids=["gemini", "deepseek"])
def test_kaynak_modele_gore_dis_ikili_secilir(monkeypatch, kaynak):
    record, judges, _panel = _kur(monkeypatch, kaynak=kaynak)
    digest = evidence_digest(build_evidence(record, TASK))
    self_judge, dis, _sup = current_external_judges_for_record(
        judges, record, experiment="sentetik", evidence_sha256=digest,
        prompt_hash=prompt_contract_hash(), expected_judges=JUDGES)
    assert self_judge == kaynak
    assert [r["judge_model"] for r in dis] == _dis(kaynak)
    assert MODEL_JUDGE_EXTERNAL in [r["judge_model"] for r in dis], \
        "üretici-dışı judge her iki kaynakta da karara girmeli"


# --- 3-5: prompt sızıntısı ----------------------------------------------------

SELF_KODU = "3.3"
SELF_GEREKCESI = "SELF_JUDGE_SIZINTI_KANITI"


@pytest.mark.parametrize("kaynak", [MODEL_MAIN, MODEL_SECONDARY],
                         ids=["gemini", "deepseek"])
def test_self_etiketi_ve_gerekcesi_prompta_SIZMAZ(monkeypatch, kaynak):
    record, judges, panel = _kur(
        monkeypatch, kaynak=kaynak,
        etiketler={kaynak: {"primary": SELF_KODU, "rationale": SELF_GEREKCESI}})
    cagrilar = _sayaci(monkeypatch)
    _adj(record, judges, panel)
    assert len(cagrilar) == 1
    prompt = cagrilar[0]
    assert SELF_GEREKCESI not in prompt, "self gerekçesi Grok'un kararına girebilirdi"
    assert f"primary={SELF_KODU}" not in prompt, "self etiketi prompt'a sızdı"


def test_promptta_yalniz_A_ve_B_var(monkeypatch):
    record, judges, panel = _kur(monkeypatch)
    cagrilar = _sayaci(monkeypatch)
    _adj(record, judges, panel)
    prompt = cagrilar[0]
    assert "Annotator A" in prompt and "Annotator B" in prompt
    assert "Annotator C" not in prompt
    assert "Two independent annotators" in prompt
    assert "Three independent annotators" not in prompt


@pytest.mark.parametrize("arm", [ARM_CONTRACT, ARM_STRUCTURED])
def test_promptta_model_kol_ve_ucluk_karari_yok(monkeypatch, arm):
    record, judges, panel = _kur(monkeypatch, arm=arm)
    cagrilar = _sayaci(monkeypatch)
    _adj(record, judges, panel)
    prompt = cagrilar[0]
    for model in (*JUDGES, MODEL_ADJUDICATOR):
        assert model not in prompt, "model kimliği otorite yanlılığı üretir"
    assert arm not in prompt, "kol adı beklenti yanlılığı üretir"
    for alan in ("majority_label", "agreement_level", "unanimous", "majority"):
        assert alan not in prompt, f"üçlü panel kararı prompt'a sızdı: {alan}"


def test_baseline_kaydinda_tek_ajan_kurali_adjudicatora_da_gider(monkeypatch):
    # Tek-ajanlı koşuda kategori 2 modu yapısal olarak imkânsızdır; ayrışma
    # kategori 1/3 kodlarıyla kurulur.
    record, judges, panel = _kur(monkeypatch, arm=ARM_BASELINE,
                                 etiketler={MODEL_JUDGE_EXTERNAL: {"primary": "3.1"}})
    mesajlar = build_adjudicator_messages(
        build_evidence(record, TASK), [_jr(record, m) for m in _dis(MODEL_MAIN)])
    assert "DATA to be analyzed, not" in mesajlar[0]["content"]


# --- 6-9: ne zaman çağrılır ---------------------------------------------------

def test_dis_konsensusta_SIFIR_cagri(monkeypatch):
    # İki dış judge aynı etiketi verdi: karar nettir, adjudicator gereksizdir.
    record, judges, panel = _kur(monkeypatch, etiketler={MODEL_JUDGE_EXTERNAL: {}})
    assert panel[0]["external_agreement_level"] == "consensus"
    assert panel[0]["adjudicator_required"] is False
    cagrilar = _sayaci(monkeypatch)
    assert _adj(record, judges, panel) == []
    assert cagrilar == []


def test_dis_splitte_TAM_BIR_cagri(monkeypatch):
    record, judges, panel = _kur(monkeypatch)
    assert panel[0]["adjudicator_required"] is True
    cagrilar = _sayaci(monkeypatch)
    sonuc = _adj(record, judges, panel)
    assert len(cagrilar) == 1 and len(sonuc) == 1
    assert sonuc[0]["adjudicator_status"] == "ok"


def test_bir_dis_judge_yetersiz_baglam_derse_split_ve_grok(monkeypatch):
    # "Yetersiz bağlam" bir MAST kodu DEĞİLDİR; normal bir kodla uzlaşamaz.
    record, judges, panel = _kur(
        monkeypatch,
        etiketler={MODEL_JUDGE_EXTERNAL: {"insufficient": True, "primary": None}})
    assert panel[0]["external_agreement_level"] == "split"
    cagrilar = _sayaci(monkeypatch)
    assert len(_adj(record, judges, panel)) == 1
    assert len(cagrilar) == 1
    assert "insufficient_context" in cagrilar[0]


def test_iki_dis_judge_de_yetersizse_konsensus_ve_SIFIR_cagri(monkeypatch):
    d1, d2 = _dis(MODEL_MAIN)
    record, judges, panel = _kur(
        monkeypatch,
        etiketler={d1: {"insufficient": True, "primary": None},
                   d2: {"insufficient": True, "primary": None}})
    assert panel[0]["external_agreement_level"] == "consensus"
    cagrilar = _sayaci(monkeypatch)
    assert _adj(record, judges, panel) == []
    assert cagrilar == []


# --- 10-12: resume tazeliği ---------------------------------------------------

@pytest.mark.parametrize("kaynak", [MODEL_MAIN, MODEL_SECONDARY],
                         ids=["gemini", "deepseek"])
def test_SELF_degisimi_yeni_cagri_URETMEZ(monkeypatch, kaynak):
    # Bu dosyanın en pahalı invariantı: karara hiç katılmayan bir etiket
    # yüzünden Grok yeniden çağrılırsa hem para gider hem karar yanlış bir
    # girdiye bağımlı gösterilir.
    record, judges, panel = _kur(monkeypatch, kaynak=kaynak)
    cagrilar = _sayaci(monkeypatch)
    ilk = _adj(record, judges, panel)
    assert len(ilk) == 1 and len(cagrilar) == 1

    for j in judges:
        if j["judge_model"] == kaynak:
            j["primary_mode"] = "3.1"
            j["rationale"] = "self tamamen fikir degistirdi"
    panel2 = build_panel([record], judges, experiment="sentetik", expected_judges=JUDGES)
    assert panel2[0]["full_panel_input_sha256"] != panel[0]["full_panel_input_sha256"]
    assert panel2[0]["decision_input_sha256"] == panel[0]["decision_input_sha256"]

    assert _adj(record, judges, panel2, existing=ilk) == []
    assert len(cagrilar) == 1, "self değişimi Grok'u yeniden çağırdı"


def test_EXTERNAL_degisimi_yeni_cagri_URETIR(monkeypatch):
    record, judges, panel = _kur(monkeypatch)
    cagrilar = _sayaci(monkeypatch)
    ilk = _adj(record, judges, panel)

    for j in judges:
        if j["judge_model"] == MODEL_JUDGE_EXTERNAL:
            j["primary_mode"] = "3.1"
    panel2 = build_panel([record], judges, experiment="sentetik", expected_judges=JUDGES)
    yeni = _adj(record, judges, panel2, existing=ilk)
    assert len(yeni) == 1 and len(cagrilar) == 2
    assert yeni[0]["decision_input_sha256"] != ilk[0]["decision_input_sha256"]
    assert yeni[0]["adjudicator_attempt"] == 2, "deneme sayacı sıfırlanmamalı"


@pytest.mark.parametrize("alan,deger", [
    ("judge_attempt", 9),
    ("ts", "2027-01-01T00:00:00+00:00"),
    ("judge_raw", '{"tamamen": "farkli"}'),
])
def test_prompt_disi_alan_degisimi_yeni_cagri_URETMEZ(monkeypatch, alan, deger):
    record, judges, panel = _kur(monkeypatch)
    cagrilar = _sayaci(monkeypatch)
    ilk = _adj(record, judges, panel)
    for j in judges:
        j[alan] = deger
    panel2 = build_panel([record], judges, experiment="sentetik", expected_judges=JUDGES)
    assert _adj(record, judges, panel2, existing=ilk) == []
    assert len(cagrilar) == 1


def test_ayni_karar_kimliginde_iki_basarili_adjudication_fail_fast(monkeypatch):
    record, judges, panel = _kur(monkeypatch)
    cagrilar = _sayaci(monkeypatch)
    ilk = _adj(record, judges, panel)
    with pytest.raises(MastPipelineError, match="birden fazla başarılı"):
        _adj(record, judges, panel, existing=[*ilk, dict(ilk[0])])
    assert len(cagrilar) == 1, "belirsiz durumda yeni çağrı yapıldı"


# --- 15-17: ön geçiş atomik ---------------------------------------------------

def test_eksik_dis_judge_SIFIR_cagriyla_durdurur(monkeypatch):
    record, judges, panel = _kur(monkeypatch)
    cagrilar = _sayaci(monkeypatch)
    eksik = [j for j in judges if j["judge_model"] != MODEL_JUDGE_EXTERNAL]
    with pytest.raises(MastPipelineError, match="güncel panel eksik"):
        _adj(record, eksik, panel)
    assert cagrilar == []


def test_yinelenen_dis_judge_SIFIR_cagriyla_durdurur(monkeypatch):
    record, judges, panel = _kur(monkeypatch)
    cagrilar = _sayaci(monkeypatch)
    with pytest.raises(MastPipelineError, match="birden fazla GÜNCEL"):
        _adj(record, [*judges, dict(judges[0])], panel)
    assert cagrilar == []


def test_beklenmeyen_judge_SIFIR_cagriyla_durdurur(monkeypatch):
    record, judges, panel = _kur(monkeypatch)
    cagrilar = _sayaci(monkeypatch)
    yabanci = dict(judges[0], judge_model="openrouter/rogue/judge")
    with pytest.raises(MastPipelineError, match="beklenmeyen judge"):
        _adj(record, [*judges, yabanci], panel)
    assert cagrilar == []


def test_panel_hash_uyusmazligi_SIFIR_cagriyla_durdurur(monkeypatch):
    record, judges, panel = _kur(monkeypatch)
    cagrilar = _sayaci(monkeypatch)
    panel[0]["decision_input_sha256"] = "0" * 64
    with pytest.raises(MastPipelineError, match="kurulduktan sonra DIŞ judge"):
        _adj(record, judges, panel)
    assert cagrilar == []


def test_yinelenen_panel_satiri_SIFIR_cagriyla_durdurur(monkeypatch):
    record, judges, panel = _kur(monkeypatch)
    cagrilar = _sayaci(monkeypatch)
    with pytest.raises(MastPipelineError, match="yinelenen kayıt"):
        _adj(record, judges, [*panel, dict(panel[0])])
    assert cagrilar == []


def test_yabanci_panel_satiri_SIFIR_cagriyla_durdurur(monkeypatch):
    record, judges, panel = _kur(monkeypatch)
    cagrilar = _sayaci(monkeypatch)
    yabanci = dict(panel[0], source_run_id="BASKA-RUN")
    with pytest.raises(MastPipelineError, match="olmayan satır"):
        _adj(record, judges, [*panel, yabanci])
    assert cagrilar == []


@pytest.mark.parametrize("alan,deger", [
    ("experiment", "BASKA-DENEY"),
    ("source_model", MODEL_SECONDARY),
    ("task_set", "pilot"),
    ("task_id", "baska-gorev"),
    ("arm", ARM_BASELINE),
    ("repeat", 99),
    ("mast_schema_version", "3.1"),
    ("evidence_sha256", "0" * 64),
    ("mast_prompt_hash", "0" * 64),
    ("interaction_type", "single_agent"),
])
def test_panel_provenance_uyusmazligi_SIFIR_cagriyla_durdurur(
        monkeypatch, alan, deger):
    record, judges, panel = _kur(monkeypatch)
    panel[0][alan] = deger
    cagrilar = _sayaci(monkeypatch)
    with pytest.raises(MastPipelineError, match="panel.*provenance|panel.*kimlik"):
        _adj(record, judges, panel)
    assert cagrilar == []


def test_eksik_panel_satiri_sessizce_atlanamaz(monkeypatch):
    record, judges, _panel = _kur(monkeypatch)
    cagrilar = _sayaci(monkeypatch)
    with pytest.raises(MastPipelineError, match="eksik panel"):
        _adj(record, judges, [])
    assert cagrilar == []


def test_iki_kayittan_birinin_paneli_eksikse_TOPLAM_cagri_sifir(monkeypatch):
    monkeypatch.setattr("eval.mast_labels._task_for", lambda r: TASK)
    r1 = _record()
    r2 = _record()
    r2["run_id"] = "ikinci-run"
    r2["task_id"] = "t001"
    d1, d2 = _dis(MODEL_MAIN)
    j1 = [_jr(r1, MODEL_MAIN), _jr(r1, d1), _jr(r1, d2, "2.3")]
    j2 = [_jr(r2, MODEL_MAIN), _jr(r2, d1), _jr(r2, d2, "2.3")]
    panel = build_panel([r1], j1, experiment="sentetik", expected_judges=JUDGES)
    cagrilar = _sayaci(monkeypatch)
    with pytest.raises(MastPipelineError, match="eksik panel"):
        run_adjudication([r1, r2], [*j1, *j2], panel, experiment="sentetik",
                         expected_judges=JUDGES)
    assert cagrilar == []


def test_yinelenen_kaynak_record_SIFIR_cagriyla_durdurur(monkeypatch):
    record, judges, panel = _kur(monkeypatch)
    cagrilar = _sayaci(monkeypatch)
    with pytest.raises(MastPipelineError, match="yinelenen.*record|yinelenen.*sonuc"):
        run_adjudication([record, dict(record)], judges, panel,
                         experiment="sentetik", expected_judges=JUDGES)
    assert cagrilar == []


def test_elle_duzenlenmis_adjudicator_required_SIFIR_cagriyla_durdurur(monkeypatch):
    # Dış konsensüsü olan bir kayıtta bayrak True yapılırsa ücretli çağrı doğardı.
    record, judges, panel = _kur(monkeypatch, etiketler={MODEL_JUDGE_EXTERNAL: {}})
    cagrilar = _sayaci(monkeypatch)
    panel[0]["adjudicator_required"] = True
    # Panel sözleşmesi bunu ön geçişte zaten yakalar: `adjudicator_required`
    # yalnız external split'ten türetilebilir.
    with pytest.raises(MastPipelineError, match="external split"):
        _adj(record, judges, panel)
    assert cagrilar == []


def test_eski_semali_panel_satiri_SIFIR_cagriyla_durdurur(monkeypatch):
    # MAST 3.1 paneli: karar alanları var ama sözleşme alanları eksik olabilir.
    record, judges, panel = _kur(monkeypatch)
    cagrilar = _sayaci(monkeypatch)
    eski = {k: v for k, v in panel[0].items() if k != "panel_hash_version"}
    with pytest.raises(MastPipelineError, match="güncel sözleşmeye uymuyor"):
        _adj(record, judges, [eski])
    assert cagrilar == []


def test_coklu_kayitta_besinci_bozuksa_TOPLAM_cagri_sifir(monkeypatch):
    # Doğrulama döngü içinde yapılsaydı, beşinci kayıttaki sorun ancak dört
    # çağrı harcandıktan sonra fark edilirdi.
    monkeypatch.setattr("eval.mast_labels._task_for", lambda r: TASK)
    kayitlar, judges, paneller = [], [], []
    for i in range(5):
        r = _record()
        r["run_id"] = f"run-{i}"
        r["task_id"] = f"t{i:03d}"
        d1, d2 = _dis(MODEL_MAIN)
        js = [_jr(r, MODEL_MAIN), _jr(r, d1), _jr(r, d2, "2.3")]
        kayitlar.append(r)
        judges.extend(js)
        paneller.extend(build_panel([r], js, experiment="sentetik",
                                    expected_judges=JUDGES))
    paneller[4]["decision_input_sha256"] = "0" * 64
    cagrilar = _sayaci(monkeypatch)
    with pytest.raises(MastPipelineError):
        run_adjudication(kayitlar, judges, paneller, experiment="sentetik",
                         expected_judges=JUDGES)
    assert cagrilar == [], "bozuk girdide çağrı yapıldı"


# --- 18-19: A/B rotasyonu -----------------------------------------------------

def test_AB_sirasi_deterministik_ve_giris_sirasindan_bagimsiz(monkeypatch):
    record, judges, _panel = _kur(monkeypatch)
    dis = [j for j in judges if j["judge_model"] != MODEL_MAIN]
    a = [r["judge_model"] for r in external_annotator_order(dis, record["run_id"])]
    b = [r["judge_model"] for r in external_annotator_order(list(reversed(dis)),
                                                            record["run_id"])]
    assert a == b
    siralar = {tuple(r["judge_model"] for r in external_annotator_order(dis, f"run-{i}"))
               for i in range(40)}
    assert len(siralar) == 2, "harf ataması kayıtlar arasında dönmüyor"


def test_ucluk_verilirse_sirali_yardimci_reddeder(monkeypatch):
    record, judges, _panel = _kur(monkeypatch)
    with pytest.raises(MastPipelineError, match="tam iki etiket"):
        external_annotator_order(judges, record["run_id"])


def test_annotator_assignment_yalniz_iki_external_tasir(monkeypatch):
    record, judges, panel = _kur(monkeypatch)
    _sayaci(monkeypatch)
    atama = _adj(record, judges, panel)[0]["annotator_assignment"]
    assert sorted(atama) == ["A", "B"]
    assert sorted(atama.values()) == sorted(_dis(MODEL_MAIN))
    assert MODEL_MAIN not in atama.values(), "self harfe atandı"


# --- Kayıt sözleşmesi ---------------------------------------------------------

def test_karar_kaydi_zorunlu_alanlari_tasir(monkeypatch):
    record, judges, panel = _kur(monkeypatch)
    _sayaci(monkeypatch)
    kayit = _adj(record, judges, panel)[0]
    for alan in ("decision_rule_version", "self_judge_model", "external_judges",
                 "external_agreement_level", "reviewed_judges", "annotator_assignment",
                 "decision_input_sha256", "adjudicator_model", "adjudicator_attempt",
                 "adjudicator_status", "adjudicated_primary_mode",
                 "adjudicated_secondary_modes", "adjudicated_confidence",
                 "adjudicated_rationale", "adjudicated_insufficient_context",
                 "source_run_id", "experiment", "source_model", "task_set", "task_id",
                 "arm", "repeat", "evidence_sha256", "mast_prompt_hash",
                 "mast_schema_version"):
        assert alan in kayit, alan
    assert kayit["decision_rule_version"] == MAST_DECISION_RULE_VERSION
    assert kayit["self_judge_model"] == record["model"] == kayit["source_model"]
    assert kayit["external_judges"] == _dis(MODEL_MAIN)
    assert kayit["reviewed_judges"] == kayit["external_judges"]
    assert kayit["external_agreement_level"] == "split"
    assert kayit["adjudicator_model"] == MODEL_ADJUDICATOR
    assert "full_panel_input_sha256" not in kayit, \
        "tanısal hash karar kaydına girerse resume yanlış bağımlılık kurar"


def test_karar_kaydindaki_hash_PROMPTA_giden_kumeden_hesaplanir(monkeypatch):
    record, judges, panel = _kur(monkeypatch)
    _sayaci(monkeypatch)
    kayit = _adj(record, judges, panel)[0]
    assert kayit["decision_input_sha256"] == panel[0]["decision_input_sha256"]


def test_basarili_kayit_ham_yaniti_ve_hashini_saklar(monkeypatch):
    import hashlib
    govde = json.dumps({"primary_mode": "2.3", "secondary_modes": [],
                        "confidence": "medium", "rationale": "karar",
                        "insufficient_context": False})
    record, judges, panel = _kur(monkeypatch)
    _sayaci(monkeypatch, text=govde)
    kayit = _adj(record, judges, panel)[0]
    assert kayit["adjudicator_raw"] == govde
    assert kayit["adjudicator_raw_sha256"] == hashlib.sha256(
        govde.encode("utf-8")).hexdigest()
    assert kayit["adjudicated_primary_mode"] == "2.3"


@pytest.mark.parametrize("govde,desen", [
    ("bu JSON degil", "adjudicator_error"),
    (json.dumps({"primary_mode": "9.9", "secondary_modes": [], "confidence": "high",
                 "rationale": "r", "insufficient_context": False}), "adjudicator_error"),
])
def test_parse_ve_dogrulama_hatasi_provenansi_ve_hashi_KORUR(monkeypatch, govde, desen):
    record, judges, panel = _kur(monkeypatch)
    _sayaci(monkeypatch, text=govde)
    kayit = _adj(record, judges, panel)[0]
    assert kayit["adjudicator_status"] == "error"
    assert desen in kayit
    assert kayit["decision_input_sha256"] == panel[0]["decision_input_sha256"]
    assert kayit["source_run_id"] == record["run_id"]
    assert kayit["reviewed_judges"] == _dis(MODEL_MAIN)
    # Hata kaydı YARIM karar taşımaz.
    for alan in ("adjudicated_primary_mode", "adjudicated_rationale"):
        assert kayit[alan] is None


def test_cagri_hatasi_provenansi_KORUR(monkeypatch):
    def _patla(*a, **k):
        raise RuntimeError("taşıma hatası")

    record, judges, panel = _kur(monkeypatch)
    monkeypatch.setattr("eval.mast_labels.call_model", _patla)
    kayit = _adj(record, judges, panel)[0]
    assert kayit["adjudicator_status"] == "error"
    assert "RuntimeError" in kayit["adjudicator_error"]
    assert kayit["decision_input_sha256"] == panel[0]["decision_input_sha256"]
    assert "adjudicator_raw" not in kayit, "çağrı hiç dönmediyse ham yanıt olamaz"


def _gecerli_karar(**over):
    alanlar = {
        "decision_rule_version": MAST_DECISION_RULE_VERSION,
        "source_model": MODEL_MAIN, "self_judge_model": MODEL_MAIN,
        "external_judges": _dis(MODEL_MAIN),
        "external_agreement_level": "split",
        "reviewed_judges": _dis(MODEL_MAIN),
        "annotator_assignment": dict(zip("AB", _dis(MODEL_MAIN))),
        "decision_input_sha256": "a" * 64,
        "interaction_type": "multi_agent",
        "adjudicator_model": MODEL_ADJUDICATOR, "adjudicator_attempt": 1,
        "adjudicator_status": "ok",
        "adjudicated_primary_mode": "1.1", "adjudicated_secondary_modes": [],
        "adjudicated_confidence": "high", "adjudicated_rationale": "r",
        "adjudicated_insufficient_context": False,
    }
    alanlar.update(over)
    return alanlar


def test_karar_sozlesmesi_gecerli_kaydi_kabul_eder():
    assert MastAdjudication(**_gecerli_karar()).adjudicator_status == "ok"


@pytest.mark.parametrize("bozuk", [
    # Sessiz fallback: başka bir model "Grok kararı" olarak kaydedilemez.
    {"adjudicator_model": MODEL_JUDGE_EXTERNAL},
    # Self, kaynak modelden farklı olamaz.
    {"self_judge_model": MODEL_SECONDARY},
    # Üretici olmayan kaynak.
    {"source_model": MODEL_JUDGE_EXTERNAL, "self_judge_model": MODEL_JUDGE_EXTERNAL},
    # Dış ikili kanonik değil / ters sıralı.
    {"external_judges": list(reversed(_dis(MODEL_MAIN)))},
    {"external_judges": [MODEL_MAIN, MODEL_JUDGE_EXTERNAL]},
    # Grok'un gördüğü küme karar kümesinden farklı.
    {"reviewed_judges": list(JUDGES)},
    {"reviewed_judges": [MODEL_MAIN, MODEL_JUDGE_EXTERNAL]},
    # Dış konsensüs varken adjudication kaydı üretilemez.
    {"external_agreement_level": "consensus"},
    {"external_agreement_level": "incomplete"},
    # Harf ataması üçlü ya da self içeriyor.
    {"annotator_assignment": dict(zip("ABC", JUDGES))},
    {"annotator_assignment": {"A": MODEL_MAIN, "B": MODEL_JUDGE_EXTERNAL}},
    # Hash biçimi.
    {"decision_input_sha256": "kisa"},
    # Hata kaydı yarım karar taşıyamaz.
    {"adjudicator_status": "error"},
    # Başarılı kayıt geçersiz etiket taşıyamaz.
    {"adjudicated_primary_mode": "9.9"},
    {"adjudicated_primary_mode": None},
    # Tek-ajanlı koşuda ajanlar-arası mod.
    {"interaction_type": "single_agent", "adjudicated_primary_mode": "2.5"},
    # Karar kuralı sürümü sessizce değişemez.
    {"decision_rule_version": "eski_kural"},
])
def test_karar_sozlesmesi_capraz_ihlalleri_reddeder(bozuk):
    with pytest.raises(Exception):
        MastAdjudication(**_gecerli_karar(**bozuk))


def test_karar_sozlesmesi_tanisal_hashi_kabul_ETMEZ():
    # `extra="forbid"`: full hash karar kaydına sessizce eklenemez.
    with pytest.raises(Exception):
        MastAdjudication(**_gecerli_karar(), full_panel_input_sha256="a" * 64)


@pytest.mark.parametrize("interaction", [None, "bilinmeyen"])
def test_karar_sozlesmesi_interaction_type_zorunlu_ve_sinirlidir(interaction):
    with pytest.raises(Exception):
        MastAdjudication(**_gecerli_karar(interaction_type=interaction))


def test_karar_sozlesmesi_bool_attempti_reddeder():
    with pytest.raises(Exception):
        MastAdjudication(**_gecerli_karar(adjudicator_attempt=True))


def test_hata_kaydi_karar_alanlari_olmadan_gecerli():
    kayit = MastAdjudication(**_gecerli_karar(
        adjudicator_status="error", adjudicated_primary_mode=None,
        adjudicated_secondary_modes=None, adjudicated_confidence=None,
        adjudicated_rationale=None, adjudicated_insufficient_context=None))
    assert kayit.adjudicator_status == "error"


# --- 20: MAST 3.1 artefaktları -----------------------------------------------

def test_31_semasiyla_yazilmis_judge_etiketi_32_turunda_oy_kullanamaz(monkeypatch):
    monkeypatch.setattr("eval.mast_labels._task_for", lambda r: TASK)
    record = _record()
    eski = [dict(_jr(record, m), mast_schema_version="3.1") for m in JUDGES]
    panel = build_panel([record], eski, experiment="sentetik", expected_judges=JUDGES)
    assert panel[0]["panel_complete"] is False
    assert panel[0]["superseded_judges"] == sorted(JUDGES)


def test_31_adjudication_kaydi_yeni_kimlikle_eslesmez(monkeypatch):
    record, judges, panel = _kur(monkeypatch)
    cagrilar = _sayaci(monkeypatch)
    ilk = _adj(record, judges, panel)
    eski = dict(ilk[0], mast_schema_version="3.1")
    # 3.1 kaydı "tamamlanmış" sayılmaz -> yeni bir karar üretilir.
    assert len(_adj(record, judges, panel, existing=[eski])) == 1
    assert len(cagrilar) == 2


def test_adjudicate_record_expected_judges_kadrosunu_zorlar(monkeypatch):
    record, judges, _panel = _kur(monkeypatch)
    monkeypatch.setattr("eval.mast_labels.call_model",
                        lambda *a, **k: pytest.fail("model çağrıldı"))
    dis = [j for j in judges if j["judge_model"] != MODEL_MAIN]
    with pytest.raises(MastPipelineError, match="dondurulmuş judge kadrosu"):
        adjudicate_record(record, build_evidence(record, TASK), dis,
                          experiment="sentetik",
                          expected_judges=(MODEL_MAIN, MODEL_SECONDARY,
                                           "openrouter/rogue/judge"))


def test_adjudicate_record_eksik_etiket_zarfiyla_modele_ulasamaz(monkeypatch):
    record = _record()
    cagrilar = _sayaci(monkeypatch)
    yalniz_model = [{"judge_model": m} for m in _dis(record["model"])]
    with pytest.raises(MastPipelineError, match="external.*geçersiz|etiket.*geçersiz"):
        adjudicate_record(record, build_evidence(record, TASK), yalniz_model,
                          experiment="sentetik")
    assert cagrilar == []


@pytest.mark.parametrize("alan,deger", [
    ("experiment", "BASKA-DENEY"),
    ("evidence_sha256", "0" * 64),
    ("mast_prompt_hash", "0" * 64),
])
def test_adjudicate_record_stale_external_zarfiyla_modele_ulasamaz(
        monkeypatch, alan, deger):
    record, judges, _panel = _kur(monkeypatch)
    dis = [dict(j, **{alan: deger}) for j in judges
           if j["judge_model"] != record["model"]]
    cagrilar = _sayaci(monkeypatch)
    with pytest.raises(MastPipelineError, match="external.*gecersiz|provenance|kimlik"):
        adjudicate_record(record, build_evidence(record, TASK), dis,
                          experiment="sentetik")
    assert cagrilar == []


def test_guncel_kimlikli_bozuk_basarili_adjudication_resume_olamaz(monkeypatch):
    record, judges, panel = _kur(monkeypatch)
    cagrilar = _sayaci(monkeypatch)
    ilk = _adj(record, judges, panel)
    bozuk = dict(ilk[0])
    bozuk.pop("adjudicated_primary_mode")
    with pytest.raises(MastPipelineError, match="basarili adjudication.*gecersiz"):
        _adj(record, judges, panel, existing=[bozuk])
    assert len(cagrilar) == 1


def test_guncel_kimlikli_self_review_tasiyan_basarili_kayit_reddedilir(monkeypatch):
    record, judges, panel = _kur(monkeypatch)
    cagrilar = _sayaci(monkeypatch)
    ilk = _adj(record, judges, panel)
    bozuk = dict(ilk[0], reviewed_judges=[record["model"], MODEL_JUDGE_EXTERNAL])
    with pytest.raises(MastPipelineError, match="basarili adjudication.*gecersiz"):
        _adj(record, judges, panel, existing=[bozuk])
    assert len(cagrilar) == 1


def test_gecerli_ve_bozuk_ayni_guncel_kimlikte_birlikte_reddedilir(monkeypatch):
    record, judges, panel = _kur(monkeypatch)
    cagrilar = _sayaci(monkeypatch)
    ilk = _adj(record, judges, panel)
    bozuk = dict(ilk[0], reviewed_judges=[record["model"], MODEL_JUDGE_EXTERNAL])
    with pytest.raises(MastPipelineError, match="basarili adjudication.*gecersiz"):
        _adj(record, judges, panel, existing=[ilk[0], bozuk])
    assert len(cagrilar) == 1


def test_eski_31_ve_guncel_32_birlikteyken_yalniz_guncel_resume_edilir(monkeypatch):
    record, judges, panel = _kur(monkeypatch)
    cagrilar = _sayaci(monkeypatch)
    ilk = _adj(record, judges, panel)
    eski = dict(ilk[0], mast_schema_version="3.1",
                reviewed_judges=[record["model"], MODEL_JUDGE_EXTERNAL])
    assert _adj(record, judges, panel, existing=[eski, ilk[0]]) == []
    assert len(cagrilar) == 1


# --- Ön geçiş KAPSAMI (2026-08-03 hotfix) ------------------------------------
#
# Global kapı, Grok çağrısı hiç gerektirmeyen tek bir kaydın bütün turu
# kilitlemesine yol açıyordu (Gemini turunda 1 self eksiği 69 split kaydını
# engelledi). Kapı artık yalnız ÇAĞRI PLANINA girecek kayıtları kapsar; karar
# kuralı, prompt, hash ve şema DEĞİŞMEDİ.

def _cok_kayit(monkeypatch, senaryolar, kaynak=MODEL_MAIN):
    """senaryolar: [(etiket_override, atlanan_judge_kumesi)] -> (records, judges, panel)."""
    monkeypatch.setattr("eval.mast_labels._task_for", lambda r: TASK)
    records, judges = [], []
    for i, (over, atlanan) in enumerate(senaryolar):
        rec = _record(model=kaynak)
        rec["run_id"] = f"run-{i:03d}"
        rec["task_id"] = f"t{i:03d}"
        records.append(rec)
        judges += [_jr(rec, m, **over.get(m, {})) for m in JUDGES if m not in atlanan]
    panel = build_panel(records, judges, experiment="sentetik", expected_judges=JUDGES)
    return records, judges, panel


def _split_etiketleri(kaynak=MODEL_MAIN):
    d1, d2 = _dis(kaynak)
    return {kaynak: {}, d1: {}, d2: {"primary": "2.3"}}


def _consensus_etiketleri(kaynak=MODEL_MAIN):
    d1, d2 = _dis(kaynak)
    return {kaynak: {}, d1: {}, d2: {}}


def test_self_eksik_consensus_kaydi_diger_split_kaydi_ENGELLEMEZ(monkeypatch):
    """Hotfix'in çekirdek iddiası: ilgisiz bir kayıt turu kilitlemesin."""
    records, judges, panel = _cok_kayit(monkeypatch, [
        (_consensus_etiketleri(), {MODEL_MAIN}),   # self eksik, dış konsensüs
        (_split_etiketleri(), set()),              # tam panel, dış split
    ])
    cagrilar = _sayaci(monkeypatch)
    sonuc = run_adjudication(records, judges, panel, experiment="sentetik",
                             expected_judges=JUDGES)
    assert len(cagrilar) == 1, "yalnız gerçekten split olan kayıt çağrılmalı"
    assert len(sonuc) == 1 and sonuc[0]["source_run_id"] == "run-001"
    # Eksik self DOLDURULMUŞ SAYILMAZ: tanısal payda kaybı korunur.
    eksik_satir = next(p for p in panel if p["source_run_id"] == "run-000")
    assert eksik_satir["panel_complete"] is False
    assert eksik_satir["external_agreement_level"] == "consensus"
    assert eksik_satir["adjudicator_required"] is False


def test_split_kayitta_self_eksikse_SIFIR_cagriyla_durur(monkeypatch):
    """Grok'a giden kayıtta tam self + iki external şartı KORUNUR."""
    records, judges, panel = _cok_kayit(monkeypatch, [
        (_split_etiketleri(), {MODEL_MAIN}),   # dış split AMA self eksik
    ])
    cagrilar = _sayaci(monkeypatch)
    with pytest.raises(MastPipelineError, match="adjudication gerektiren kayıtta panel eksik"):
        run_adjudication(records, judges, panel, experiment="sentetik",
                         expected_judges=JUDGES)
    assert cagrilar == []


def test_dis_judge_eksik_kayit_icin_cagri_URETILMEZ(monkeypatch):
    """Eksik external ile karar girdisi kurulamaz; kayıt kararsız kalır."""
    d1, _d2 = _dis(MODEL_MAIN)
    records, judges, panel = _cok_kayit(monkeypatch, [
        (_split_etiketleri(), {d1}),
    ])
    cagrilar = _sayaci(monkeypatch)
    sonuc = run_adjudication(records, judges, panel, experiment="sentetik",
                             expected_judges=JUDGES)
    assert cagrilar == [] and sonuc == []
    satir = panel[0]
    assert satir["external_agreement_level"] == "incomplete"
    assert satir["adjudicator_required"] is False
    assert satir["decision_input_sha256"] is None


def test_kanit_uyusmazligi_TURUN_TAMAMINI_ilk_cagridan_once_durdurur(monkeypatch):
    """Kapsam daralması kanıt/prompt kapısını GEVŞETMEZ: o hâlâ globaldir."""
    records, judges, panel = _cok_kayit(monkeypatch, [
        (_consensus_etiketleri(), set()),   # Grok gerektirmeyen sağlam kayıt
        (_split_etiketleri(), set()),       # çağrı planındaki kayıt
    ])
    cagrilar = _sayaci(monkeypatch)
    # Plan DIŞINDAKİ kaydın kanıt hash'i bozulsa bile bütün tur durmalı.
    next(p for p in panel if p["source_run_id"] == "run-000")["evidence_sha256"] = "0" * 64
    with pytest.raises(MastPipelineError, match="provenance|kimlik"):
        run_adjudication(records, judges, panel, experiment="sentetik",
                         expected_judges=JUDGES)
    assert cagrilar == []


def test_gemini_bicimi_147_tam_1_self_eksik_ile_69_cagri_plani(monkeypatch):
    """Canlı Gemini turunun sentetik karşılığı (§34): 148 panel, 69 split."""
    senaryolar = [(_split_etiketleri(), set()) for _ in range(69)]
    senaryolar += [(_consensus_etiketleri(), set()) for _ in range(78)]
    senaryolar += [(_consensus_etiketleri(), {MODEL_MAIN})]   # self eksik consensus
    records, judges, panel = _cok_kayit(monkeypatch, senaryolar)
    assert len(panel) == 148
    assert sum(p["panel_complete"] for p in panel) == 147
    assert sum(p["adjudicator_required"] for p in panel) == 69

    cagrilar = _sayaci(monkeypatch)
    sonuc = run_adjudication(records, judges, panel, experiment="sentetik",
                             expected_judges=JUDGES)
    assert len(cagrilar) == 69, "self eksiği 69 split kaydını engellememeli"
    assert len(sonuc) == 69


def test_deepseek_bicimi_tam_panel_davranisi_degismedi(monkeypatch):
    """Hiç eksik panel yokken davranış birebir aynı kalmalı."""
    senaryolar = [(_split_etiketleri(MODEL_SECONDARY), set()) for _ in range(5)]
    senaryolar += [(_consensus_etiketleri(MODEL_SECONDARY), set()) for _ in range(4)]
    records, judges, panel = _cok_kayit(monkeypatch, senaryolar, kaynak=MODEL_SECONDARY)
    assert all(p["panel_complete"] for p in panel)
    cagrilar = _sayaci(monkeypatch)
    sonuc = run_adjudication(records, judges, panel, experiment="sentetik",
                             expected_judges=JUDGES)
    assert len(cagrilar) == 5 and len(sonuc) == 5
