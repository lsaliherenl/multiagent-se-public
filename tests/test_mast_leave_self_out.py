"""Leave-self-out karar semantiği (§9.1-§9.2) — LLM'siz, deterministik.

Kapatılan sessiz hata: **kaydı üreten model kendi çıktısının hata etiketine oy
veriyordu.** Panelde iki üretici bulunduğu için bu, üreticinin kendi
başarısızlığının AI kararını doğrudan (oy) ve dolaylı (gerekçe → adjudicator)
biçimde etkilemesi demekti.

Yöntemsel karşı-örnek (bu dosyanın çekirdeği): tam üçlü panelde AYNI 2/1
majority deseni İKİ FARKLI karar durumu saklar —

* çoğunluğu iki DIŞ judge oluşturuyorsa   -> external consensus (adjudicator YOK)
* çoğunluğu self + bir dış judge yapıyorsa -> external SPLIT (adjudicator VAR)

Yani üçlü `majority_label` self-judge etkisini karar katmanından çıkaramaz;
bu yüzden tanısal alan olarak kalır ve karar iki dış judge'a bağlanır.

Fixture'lar GERÇEK rol sabitlerini kullanır: rol, etiket içeriğinden değil
kaynak-model provenance'ından türediği için `j1/j2/j3` gibi kaynağın panelde
bulunmadığı yapay kurulumlar bu semantiği sınayamaz.
"""

import hashlib
import json
import sys

import pytest

from config import (
    ARM_CONTRACT,
    MAST_DECISION_RULE_VERSION,
    MAST_PANEL_HASH_VERSION,
    MAST_SCHEMA_VERSION,
    MODEL_ADJUDICATOR,
    MODEL_JUDGE_EXTERNAL,
    MODEL_JUDGES,
    MODEL_MAIN,
    MODEL_SECONDARY,
)
from eval import mast_labels as mast_cli
from eval.mast_labels import (
    MAST_MANIFEST_CRITICAL,
    build_evidence,
    build_mast_manifest,
    build_panel,
    check_or_write_mast_manifest,
    panel_blockers,
    preflight_roles,
    prompt_contract_hash,
    run_judges,
)
from eval.mast_schema import (
    INSUFFICIENT_SENTINEL,
    MastPanelVerdict,
    MastPipelineError,
    evidence_digest,
    judge_role_partition,
    panel_verdict,
    validate_expected_judges,
    validate_frozen_panel,
)
from eval.result_schema import make_synthetic_record

JUDGES = tuple(MODEL_JUDGES)
TASK = {"task_id": "t000", "prompt": "def f(x):\n    ...", "entry_point": "f"}


def _record(model=MODEL_MAIN, **extra):
    record = make_synthetic_record(experiment="sentetik", model=model, task_set="heldout",
                                   arm=ARM_CONTRACT, task_id="t000", repeat=0,
                                   base_pass=False, plus_pass=False)
    record.setdefault("code", "def f(x):\n    return 0\n")
    record.setdefault("plan", {"task_id": "t000"})
    record.setdefault("raw_messages", [{"from": "planner", "to": "coder", "content": "p"}])
    record.update(extra)
    return record


def _jr(model, primary="1.1", *, insufficient=False, status="ok"):
    return {"judge_model": model, "judge_status": status,
            "primary_mode": None if insufficient else primary,
            "secondary_modes": [], "confidence": "high", "rationale": "r",
            "insufficient_context": insufficient}


def _verdict(etiketler: dict, *, source_model=MODEL_MAIN, expected=JUDGES):
    """etiketler: {judge_model: primary | INSUFFICIENT_SENTINEL}"""
    kayitlar = [_jr(m, insufficient=k is INSUFFICIENT_SENTINEL,
                    primary=None if k is INSUFFICIENT_SENTINEL else k)
                for m, k in etiketler.items()]
    return panel_verdict(kayitlar, expected, source_model=source_model)


# --- A: tam üçlü majority, external CONSENSUS --------------------------------

def test_A_ucluk_majority_dis_konsensus_olabilir():
    # Eski kural ("üçlü tamamen aynı değilse adjudicator") burada ölür: self
    # ayrışsa bile iki dış judge uzlaşıyorsa karar nettir.
    v = _verdict({MODEL_MAIN: "2.3", MODEL_SECONDARY: "1.1", MODEL_JUDGE_EXTERNAL: "1.1"})
    assert v["agreement_level"] == "majority"      # tanısal
    assert v["majority_label"] == "1.1"            # tanısal
    assert v["external_agreement_level"] == "consensus"
    assert v["external_consensus_label"] == "1.1"
    assert v["self_matches_external"] is False
    assert v["adjudicator_required"] is False


# --- B: AYNI majority deseni, external SPLIT ---------------------------------

def test_B_ucluk_majority_varken_dis_split_olabilir():
    # A ile B'nin üçlü tanısal alanları AYNIDIR (majority / 1.1); karar
    # durumları zıttır. `majority_label`'a bakan bir karar katmanı bu iki
    # durumu ayırt edemezdi — self-judge çoğunluğu tek başına kurabiliyor.
    v = _verdict({MODEL_MAIN: "1.1", MODEL_SECONDARY: "1.1", MODEL_JUDGE_EXTERNAL: "2.3"})
    assert v["agreement_level"] == "majority"
    assert v["majority_label"] == "1.1"
    assert v["external_agreement_level"] == "split"
    assert v["external_consensus_label"] is None
    assert v["self_matches_external"] is None
    assert v["adjudicator_required"] is True


def test_A_ve_B_ayni_tanisal_deseni_farkli_karari_verir():
    a = _verdict({MODEL_MAIN: "2.3", MODEL_SECONDARY: "1.1", MODEL_JUDGE_EXTERNAL: "1.1"})
    b = _verdict({MODEL_MAIN: "1.1", MODEL_SECONDARY: "1.1", MODEL_JUDGE_EXTERNAL: "2.3"})
    assert (a["agreement_level"], a["majority_label"]) == (b["agreement_level"],
                                                           b["majority_label"])
    assert a["adjudicator_required"] != b["adjudicator_required"]


# --- C: self değişimi kararı DEĞİŞTİREMEZ ------------------------------------

@pytest.mark.parametrize("self_etiketi", ["1.1", "2.3", INSUFFICIENT_SENTINEL])
def test_C_self_degisimi_karari_degistirmez(self_etiketi):
    v = _verdict({MODEL_MAIN: self_etiketi,
                  MODEL_SECONDARY: "1.1", MODEL_JUDGE_EXTERNAL: "1.1"})
    assert v["external_agreement_level"] == "consensus"
    assert v["external_consensus_label"] == "1.1"
    assert v["adjudicator_required"] is False
    # Değişen TEK şey öz-değerlendirme uyumu (tanısal metrik).
    assert v["self_matches_external"] is (self_etiketi == "1.1")


# --- D: external değişimi kararı DEĞİŞTİRİR ----------------------------------

def test_D_dis_judge_degisimi_karari_degistirir():
    sabit_self = {MODEL_MAIN: "1.1"}
    once = _verdict({**sabit_self, MODEL_SECONDARY: "1.1", MODEL_JUDGE_EXTERNAL: "1.1"})
    sonra = _verdict({**sabit_self, MODEL_SECONDARY: "1.1", MODEL_JUDGE_EXTERNAL: "3.2"})
    assert (once["external_agreement_level"], once["adjudicator_required"]) == (
        "consensus", False)
    assert once["external_consensus_label"] == "1.1"
    assert (sonra["external_agreement_level"], sonra["adjudicator_required"]) == (
        "split", True)
    assert sonra["external_consensus_label"] is None


# --- E: kaynak model rolü belirler -------------------------------------------

def test_E_kaynak_model_degisince_rol_bolumlemesi_degisir():
    etiketler = {MODEL_MAIN: "1.1", MODEL_SECONDARY: "1.1", MODEL_JUDGE_EXTERNAL: "2.3"}
    gemini = _verdict(etiketler, source_model=MODEL_MAIN)
    deepseek = _verdict(etiketler, source_model=MODEL_SECONDARY)

    assert gemini["self_judge_model"] == MODEL_MAIN
    assert gemini["external_judges"] == [MODEL_SECONDARY, MODEL_JUDGE_EXTERNAL]
    assert deepseek["self_judge_model"] == MODEL_SECONDARY
    assert deepseek["external_judges"] == [MODEL_MAIN, MODEL_JUDGE_EXTERNAL]
    # AYNI üç etiket, farklı kaynak: karar da değişebilir. Rol etiketin
    # içeriğinden değil provenance'tan türüyor.
    assert gemini["adjudicator_required"] is True
    assert deepseek["adjudicator_required"] is True
    assert gemini["primary_modes"] == deepseek["primary_modes"]


def test_E_kaynak_degisimi_karari_tersine_cevirebilir():
    etiketler = {MODEL_MAIN: "1.1", MODEL_SECONDARY: "2.3", MODEL_JUDGE_EXTERNAL: "2.3"}
    gemini = _verdict(etiketler, source_model=MODEL_MAIN)
    deepseek = _verdict(etiketler, source_model=MODEL_SECONDARY)
    assert gemini["adjudicator_required"] is False    # dış ikili 2.3'te uzlaşıyor
    assert deepseek["adjudicator_required"] is True   # dış ikili 1.1 vs 2.3


# --- F: self eksik ------------------------------------------------------------

def test_F_self_eksikken_dis_durum_yine_turetilir(monkeypatch):
    v = _verdict({MODEL_SECONDARY: "1.1", MODEL_JUDGE_EXTERNAL: "1.1"})
    assert v["panel_complete"] is False
    assert v["agreement_level"] == "incomplete"
    assert v["external_agreement_level"] == "consensus"
    assert v["external_consensus_label"] == "1.1"
    assert v["self_matches_external"] is None
    assert v["missing_judges"] == [MODEL_MAIN]


def test_F_eksik_self_adjudicationi_engeller(monkeypatch):
    # Dış taraf split olsa bile: eksik panel üzerinde adjudication YAPILMAZ ve
    # tek bir API çağrısı bile yapılmaz.
    record = _record()
    monkeypatch.setattr("eval.mast_labels._task_for", lambda r: TASK)

    def _patla(*a, **k):
        raise AssertionError("model çağrıldı — kapı çalışmıyor")

    monkeypatch.setattr("eval.mast_labels.call_model", _patla)
    judges = [_tam_jr(record, MODEL_SECONDARY, "1.1"),
              _tam_jr(record, MODEL_JUDGE_EXTERNAL, "2.3")]
    panel = build_panel([record], judges, experiment="sentetik", expected_judges=JUDGES)
    assert panel[0]["adjudicator_required"] is True
    assert panel[0]["panel_complete"] is False
    assert panel_blockers(panel)


# --- G: dış judge eksik -------------------------------------------------------

def test_G_dis_judge_eksikse_karar_verilemez():
    v = _verdict({MODEL_MAIN: "1.1", MODEL_SECONDARY: "1.1"})
    assert v["external_agreement_level"] == "incomplete"
    assert v["external_consensus_label"] is None
    assert v["self_matches_external"] is None
    assert v["adjudicator_required"] is False
    assert v["panel_complete"] is False


# --- H: geçersiz rol kümeleri -------------------------------------------------

@pytest.mark.parametrize("source_model", [MODEL_JUDGE_EXTERNAL, MODEL_ADJUDICATOR,
                                          "openrouter/baska/model"])
def test_H_uretici_olmayan_kaynak_model_reddedilir(source_model):
    with pytest.raises(MastPipelineError, match="kaynak model"):
        judge_role_partition(source_model, JUDGES)


def test_H_kaynak_model_panelde_yoksa_reddedilir():
    baska_panel = (MODEL_SECONDARY, MODEL_JUDGE_EXTERNAL, MODEL_ADJUDICATOR)
    with pytest.raises(MastPipelineError):
        judge_role_partition(MODEL_MAIN, baska_panel)


@pytest.mark.parametrize("expected", [
    (MODEL_MAIN, MODEL_MAIN, MODEL_JUDGE_EXTERNAL),        # iki judge aynı
    (MODEL_MAIN, MODEL_SECONDARY),                          # iki beklenen judge
    (MODEL_MAIN, MODEL_SECONDARY, MODEL_JUDGE_EXTERNAL, MODEL_ADJUDICATOR),  # dört
])
def test_H_gecersiz_beklenen_kume_reddedilir(expected):
    with pytest.raises(MastPipelineError):
        validate_expected_judges(expected)


def test_H_adjudicator_panelde_bulunamaz():
    with pytest.raises(MastPipelineError, match="adjudicator"):
        validate_expected_judges((MODEL_MAIN, MODEL_SECONDARY, MODEL_ADJUDICATOR))


def test_H_beklenmeyen_judge_etiketi_fail_fast():
    kayitlar = [_jr(m) for m in JUDGES] + [_jr("yabanci/judge")]
    with pytest.raises(MastPipelineError, match="beklenmeyen judge"):
        panel_verdict(kayitlar, JUDGES, source_model=MODEL_MAIN)


def test_H_ayni_judge_icin_iki_guncel_etiket_fail_fast():
    kayitlar = [_jr(m) for m in JUDGES] + [_jr(MODEL_MAIN, "2.3")]
    with pytest.raises(MastPipelineError, match="birden fazla"):
        panel_verdict(kayitlar, JUDGES, source_model=MODEL_MAIN)


# --- Preflight: API çağrısından ÖNCE -----------------------------------------

def test_preflight_gecersiz_kaynak_modelde_hic_cagri_yapilmaz(monkeypatch):
    def _patla(*a, **k):
        raise AssertionError("judge çağrıldı — preflight çalışmıyor")

    monkeypatch.setattr("eval.mast_labels.call_model", _patla)
    monkeypatch.setattr("eval.mast_labels._task_for", lambda r: TASK)
    with pytest.raises(MastPipelineError, match="kaynak model"):
        run_judges([_record(model=MODEL_JUDGE_EXTERNAL)], experiment="sentetik")


def test_preflight_gecersiz_beklenen_kumede_hic_cagri_yapilmaz(monkeypatch):
    def _patla(*a, **k):
        raise AssertionError("judge çağrıldı — preflight çalışmıyor")

    monkeypatch.setattr("eval.mast_labels.call_model", _patla)
    monkeypatch.setattr("eval.mast_labels._task_for", lambda r: TASK)
    with pytest.raises(MastPipelineError):
        run_judges([_record()], experiment="sentetik",
                   judges=(MODEL_MAIN, MODEL_SECONDARY))


def test_preflight_bos_kayit_listesinde_de_kumeyi_dogrular():
    with pytest.raises(MastPipelineError):
        preflight_roles([], (MODEL_MAIN, MODEL_MAIN, MODEL_JUDGE_EXTERNAL))


def test_build_panel_rol_dogrulamasini_bagimsiz_yapar(monkeypatch):
    # build_panel, judge aşaması hiç koşmadan da çağrılabilir (resume, 6B).
    monkeypatch.setattr("eval.mast_labels._task_for", lambda r: TASK)
    with pytest.raises(MastPipelineError, match="kaynak model"):
        build_panel([_record(model=MODEL_ADJUDICATOR)], [], experiment="sentetik",
                    expected_judges=JUDGES)


# --- Dondurulmuş kadro kapısı (ücretli çağrı sınırı) --------------------------

ROGUE = "openrouter/rogue/judge"


def _kaynak_manifest():
    """MAST manifestinin türetildiği KAYNAK deney manifesti."""
    return {"name": "live", "model": MODEL_MAIN, "task_set": "heldout",
            "task_ids": ["t000"], "arm_order": [ARM_CONTRACT], "repeats": 1,
            "result_schema_version": "2.0", "llm_call_schema_version": "2.0"}


def _cagri_yasak(monkeypatch):
    def _patla(*a, **k):
        raise AssertionError("model çağrıldı — kadro kapısı çalışmıyor")

    monkeypatch.setattr("eval.mast_labels.call_model", _patla)
    monkeypatch.setattr("eval.mast_labels._task_for", lambda r: TASK)


@pytest.mark.parametrize("kadro", [
    # Yapısal olarak KUSURSUZ (üç farklı model, adjudicator yok) ama ön-kayıtta
    # olmayan bir modele ücretli çağrı yaptırırdı.
    (MODEL_MAIN, MODEL_SECONDARY, ROGUE),
    # Doğru modeller, FARKLI SIRA: dış judge sırası ve Parça 4'te adjudicator'a
    # giden sunum sırası buradan türer, yani sıra kimliğin parçasıdır.
    (MODEL_JUDGE_EXTERNAL, MODEL_SECONDARY, MODEL_MAIN),
])
def test_kadro_kapisi_run_judgesi_sifir_cagriyla_durdurur(monkeypatch, kadro):
    _cagri_yasak(monkeypatch)
    with pytest.raises(MastPipelineError, match="dondurulmuş judge kadrosu"):
        run_judges([_record()], experiment="sentetik", judges=kadro)


def test_kadro_kapisi_yapisal_dogrulamadan_AYRI_kalir():
    # Yapısal doğrulama generic kalmalı (ters örnek ve şema testleri onu
    # kullanır); üretim kapısı ayrı ve sıkıdır.
    assert validate_expected_judges((MODEL_MAIN, MODEL_SECONDARY, ROGUE))
    with pytest.raises(MastPipelineError, match="ön-kayıtta olmayan judge"):
        validate_frozen_panel((MODEL_MAIN, MODEL_SECONDARY, ROGUE))


def test_kadro_kapisi_ayni_modellerin_farkli_sirasini_reddeder():
    with pytest.raises(MastPipelineError, match="FARKLI SIRADA"):
        validate_frozen_panel((MODEL_JUDGE_EXTERNAL, MODEL_SECONDARY, MODEL_MAIN))


def test_kadro_kapisi_build_paneli_de_durdurur(monkeypatch):
    _cagri_yasak(monkeypatch)
    with pytest.raises(MastPipelineError, match="dondurulmuş judge kadrosu"):
        build_panel([_record()], [], experiment="sentetik",
                    expected_judges=(MODEL_MAIN, MODEL_SECONDARY, ROGUE))


def test_kadro_kapisi_preflightta_kayit_yuklenmeden_calisir():
    # Boş kayıt listesinde bile kadro doğrulanır: kapı, görev yükleme ve kanıt
    # kurmadan ÖNCE koşmalı.
    with pytest.raises(MastPipelineError, match="dondurulmuş judge kadrosu"):
        preflight_roles([], (MODEL_MAIN, MODEL_SECONDARY, ROGUE))


def test_manifest_rogue_judgei_dosya_YAZILMADAN_reddeder(tmp_path):
    yol = tmp_path / "manifest.json"
    with pytest.raises(MastPipelineError, match="dondurulmuş judge kadrosu"):
        check_or_write_mast_manifest(
            yol, build_mast_manifest(_kaynak_manifest(),
                                     judges=(MODEL_MAIN, MODEL_SECONDARY, ROGUE)))
    assert not yol.exists(), "reddedilen kadro için manifest dosyası yazıldı"


def test_manifest_farkli_adjudicatoru_reddeder(tmp_path):
    yol = tmp_path / "manifest.json"
    with pytest.raises(MastPipelineError, match="dondurulmuş adjudicator"):
        build_mast_manifest(_kaynak_manifest(), adjudicator="openrouter/rogue/adj")
    assert not yol.exists()


def test_manifest_tam_kadroyu_kabul_eder(tmp_path):
    snapshot = build_mast_manifest(_kaynak_manifest(), judges=JUDGES,
                                   adjudicator=MODEL_ADJUDICATOR)
    assert snapshot["judges"] == list(JUDGES)
    assert snapshot["adjudicator_model"] == MODEL_ADJUDICATOR


# --- I: yetersiz bağlam -------------------------------------------------------

def test_I_iki_dis_judge_de_yetersiz_baglamda_uzlasir():
    v = _verdict({MODEL_MAIN: "1.1",
                  MODEL_SECONDARY: INSUFFICIENT_SENTINEL,
                  MODEL_JUDGE_EXTERNAL: INSUFFICIENT_SENTINEL})
    assert v["external_agreement_level"] == "consensus"
    assert v["external_consensus_label"] == INSUFFICIENT_SENTINEL
    assert v["self_matches_external"] is False
    assert v["adjudicator_required"] is False


def test_I_bir_dis_judge_yetersizse_split():
    # "Yetersiz bağlam" bir MAST kodu DEĞİLDİR; normal bir kodla eşleşemez.
    v = _verdict({MODEL_MAIN: "1.1",
                  MODEL_SECONDARY: INSUFFICIENT_SENTINEL,
                  MODEL_JUDGE_EXTERNAL: "1.1"})
    assert v["external_agreement_level"] == "split"
    assert v["adjudicator_required"] is True
    assert v["external_consensus_label"] is None


def test_I_self_yetersizken_dis_konsensus_korunur():
    v = _verdict({MODEL_MAIN: INSUFFICIENT_SENTINEL,
                  MODEL_SECONDARY: "2.3", MODEL_JUDGE_EXTERNAL: "2.3"})
    assert v["external_agreement_level"] == "consensus"
    assert v["external_consensus_label"] == "2.3"
    assert v["self_matches_external"] is False
    assert v["adjudicator_required"] is False


# --- Panel sözleşmesi (çapraz invariantlar) ----------------------------------

def _hex(seed: str) -> str:
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()


def _gecerli_verdict_alanlari(**over):
    alanlar = {
        "panel_hash_version": MAST_PANEL_HASH_VERSION,
        "full_panel_input_sha256": _hex("full"),
        "decision_input_sha256": _hex("decision"),
        "expected_judges": list(JUDGES), "valid_judges": 3, "missing_judges": [],
        "judge_models": sorted(JUDGES), "primary_modes": ["1.1", "1.1", "1.1"],
        "agreement_level": "unanimous", "majority_label": "1.1",
        "judge_disagreement": False, "panel_complete": True,
        "decision_rule_version": MAST_DECISION_RULE_VERSION,
        "self_judge_model": MODEL_MAIN,
        "external_judges": [MODEL_SECONDARY, MODEL_JUDGE_EXTERNAL],
        "external_agreement_level": "consensus", "external_consensus_label": "1.1",
        "self_matches_external": True, "adjudicator_required": False,
    }
    alanlar.update(over)
    return alanlar


@pytest.mark.parametrize("bozuk", [
    # Üçlü majority tek başına adjudicator gerektiremez.
    {"agreement_level": "majority", "adjudicator_required": True},
    # Konsensüs varken etiket None olamaz (ve tersi).
    {"external_consensus_label": None},
    {"external_agreement_level": "split", "external_consensus_label": "1.1"},
    # Self bölümlemesi tutarsız.
    {"external_judges": [MODEL_MAIN, MODEL_SECONDARY]},
    {"external_judges": [MODEL_SECONDARY, MODEL_SECONDARY]},
    {"external_judges": [MODEL_SECONDARY]},
    # Eksik judge varken panel tamamlanmış görünemez.
    {"missing_judges": [MODEL_JUDGE_EXTERNAL]},
    # Eksik self ile self karşılaştırması yapılamaz.
    {"missing_judges": [MODEL_MAIN], "panel_complete": False,
     "agreement_level": "incomplete", "self_matches_external": True},
    # Karar kuralı sürümü sessizce değişemez.
    {"decision_rule_version": "eski_kural"},
])
def test_panel_sozlesmesi_capraz_ihlalleri_reddeder(bozuk):
    with pytest.raises(Exception):
        MastPanelVerdict(**_gecerli_verdict_alanlari(**bozuk))


def test_panel_sozlesmesi_gecerli_kaydi_kabul_eder():
    v = MastPanelVerdict(**_gecerli_verdict_alanlari())
    assert v.adjudicator_required is False


def test_panel_sozlesmesi_graf_olarak_IMKANSIZ_kaydi_reddeder():
    # Her alan tek başına "geçerli tip"tir; graf olarak imkânsızdır:
    # üç oy da aynıyken majority, oy kullanmayan bir judge sayılmış
    # (valid_judges=3 ama iki model) ve kimsenin vermediği bir etiket
    # çoğunluk ilan edilmiş. Alan bazlı doğrulama bunu göremezdi.
    with pytest.raises(Exception):
        MastPanelVerdict(**_gecerli_verdict_alanlari(
            judge_models=[MODEL_SECONDARY, MODEL_MAIN],
            agreement_level="majority", majority_label="9.9",
            judge_disagreement=False))


@pytest.mark.parametrize("bozuk", [
    # Yinelenen/fazla judge içeren beklenen küme.
    {"expected_judges": [MODEL_MAIN, MODEL_MAIN, MODEL_SECONDARY,
                         MODEL_JUDGE_EXTERNAL]},
    {"expected_judges": [MODEL_MAIN, MODEL_MAIN, MODEL_JUDGE_EXTERNAL]},
    # Dondurulmuş kadro dışında bir panel kaydı.
    {"expected_judges": [MODEL_MAIN, MODEL_SECONDARY, ROGUE]},
    # missing_judges yinelenen ya da beklenen kümenin dışında.
    {"missing_judges": [MODEL_JUDGE_EXTERNAL, MODEL_JUDGE_EXTERNAL]},
    {"missing_judges": [ROGUE]},
    # judge_models yinelenen / beklenen-eksik kümesine eşit değil.
    {"judge_models": [MODEL_MAIN, MODEL_MAIN, MODEL_SECONDARY]},
    # Sayaçlar oy sayısıyla uyuşmuyor.
    {"valid_judges": 2},
    {"primary_modes": ["1.1", "1.1"]},
    # Oy taksonomiye ait değil.
    {"primary_modes": ["1.1", "1.1", "9.9"], "agreement_level": "majority",
     "external_consensus_label": "1.1"},
    {"majority_label": "9.9"},
    # Türetilebilir alanlar oy dağılımıyla çelişiyor.
    {"judge_disagreement": True},
    {"external_agreement_level": "incomplete", "external_consensus_label": None,
     "self_matches_external": None},
    {"self_matches_external": False},
])
def test_panel_sozlesmesi_turetilebilir_alanlari_yeniden_hesaplar(bozuk):
    with pytest.raises(Exception):
        MastPanelVerdict(**_gecerli_verdict_alanlari(**bozuk))


@pytest.mark.parametrize("bozuk", [
    # Doğru üç model, FARKLI SIRA: küme eşitliği yeterli değil.
    {"expected_judges": [MODEL_MAIN, MODEL_SECONDARY, MODEL_JUDGE_EXTERNAL],
     "judge_models": [MODEL_MAIN, MODEL_SECONDARY, MODEL_JUDGE_EXTERNAL]},
    {"expected_judges": [MODEL_JUDGE_EXTERNAL, MODEL_SECONDARY, MODEL_MAIN],
     "judge_models": [MODEL_JUDGE_EXTERNAL, MODEL_SECONDARY, MODEL_MAIN]},
    # Üretici olmayan model self-judge olamaz (MiniMax held-out veri üretmez;
    # onu self ilan etmek İKİ üreticiyi dış judge yapıp kaydı üreten modelin
    # oyunu karara geri sokardı).
    {"self_judge_model": MODEL_JUDGE_EXTERNAL,
     "external_judges": [MODEL_MAIN, MODEL_SECONDARY]},
    {"self_judge_model": MODEL_ADJUDICATOR,
     "external_judges": [MODEL_SECONDARY, MODEL_JUDGE_EXTERNAL]},
    # Doğru dış küme, TERS SIRA: aynı mantıksal panel farklı kanonik girdi olurdu.
    {"external_judges": [MODEL_JUDGE_EXTERNAL, MODEL_SECONDARY]},
])
def test_panel_sozlesmesi_kanonik_SIRAYI_zorlar(bozuk):
    with pytest.raises(Exception):
        MastPanelVerdict(**_gecerli_verdict_alanlari(**bozuk))


def test_panel_sozlesmesi_missing_judges_ters_sirasini_reddeder():
    # İki eksik judge: küme aynı, sıra dondurulmuş panel sırası değil.
    tersi = [JUDGES[2], JUDGES[1]]
    with pytest.raises(Exception):
        MastPanelVerdict(**_gecerli_verdict_alanlari(
            missing_judges=tersi, valid_judges=1, judge_models=[MODEL_MAIN],
            primary_modes=["1.1"], agreement_level="incomplete", majority_label=None,
            judge_disagreement=True, panel_complete=False,
            external_agreement_level="incomplete", external_consensus_label=None,
            self_matches_external=None, full_panel_input_sha256=None,
            decision_input_sha256=None))


def test_panel_sozlesmesi_missing_judges_kanonik_sirayi_kabul_eder():
    v = MastPanelVerdict(**_gecerli_verdict_alanlari(
        missing_judges=[JUDGES[1], JUDGES[2]], valid_judges=1,
        judge_models=[JUDGES[0]], primary_modes=["1.1"],
        agreement_level="incomplete", majority_label=None,
        judge_disagreement=True, panel_complete=False,
        self_judge_model=MODEL_SECONDARY,
        external_judges=[MODEL_MAIN, MODEL_JUDGE_EXTERNAL],
        external_agreement_level="incomplete", external_consensus_label=None,
        self_matches_external=None, full_panel_input_sha256=None,
        decision_input_sha256=None))
    assert v.missing_judges == [JUDGES[1], JUDGES[2]]


@pytest.mark.parametrize("kaynak", [MODEL_MAIN, MODEL_SECONDARY])
def test_iki_uretici_icin_de_gecerli_panel_kurulur(kaynak):
    # Gemini ve DeepSeek kaynaklı paneller kabul edilir; bölümleme kaynağa göre
    # değişir ama her ikisinde de kanonik sıra korunur.
    v = _verdict({MODEL_MAIN: "1.1", MODEL_SECONDARY: "1.1",
                  MODEL_JUDGE_EXTERNAL: "1.1"}, source_model=kaynak)
    assert v["self_judge_model"] == kaynak
    assert v["external_judges"] == [m for m in JUDGES if m != kaynak]
    assert v["expected_judges"] == list(JUDGES)
    assert v["adjudicator_required"] is False
    MastPanelVerdict(**{k: v[k] for k in MastPanelVerdict.model_fields})


def test_panel_sozlesmesi_fazladan_alani_sessizce_yutmaz():
    # Yanlış yazılmış bir karar alanı sessizce yutulup varsayılanla
    # doldurulamaz: `adjudicator_requiered=True` taşıyan bir kayıt, kararı
    # False görünen bir panel üretirdi.
    with pytest.raises(Exception):
        MastPanelVerdict(**_gecerli_verdict_alanlari(adjudicator_requiered=True))


def test_panel_sozlesmesi_eksik_self_ile_dis_konsensusu_korur():
    # Self eksik: dış konsensüs KORUNUR (karar verilebilir), ama panel
    # tamamlanmamıştır ve self karşılaştırması tanımsızdır.
    v = MastPanelVerdict(**_gecerli_verdict_alanlari(
        missing_judges=[MODEL_MAIN], valid_judges=2,
        judge_models=[MODEL_SECONDARY, MODEL_JUDGE_EXTERNAL],
        primary_modes=["1.1", "1.1"], agreement_level="incomplete",
        majority_label=None, judge_disagreement=True, panel_complete=False,
        self_matches_external=None, full_panel_input_sha256=None))
    assert v.external_agreement_level == "consensus"
    assert v.external_consensus_label == "1.1"
    assert v.adjudicator_required is False


def test_eski_panel_kaydi_yeni_alanlar_varmis_gibi_yorumlanmaz():
    # MAST 2.0 panel satırında yeni karar alanları YOKTUR; varsayılan üretmek
    # (ör. adjudicator_required=False) 2.0 verisini 3.0 kararı gibi gösterirdi.
    eski = {k: v for k, v in _gecerli_verdict_alanlari().items()
            if k in ("expected_judges", "valid_judges", "missing_judges", "judge_models",
                     "primary_modes", "agreement_level", "majority_label",
                     "judge_disagreement", "panel_complete")}
    with pytest.raises(Exception):
        MastPanelVerdict(**eski)


# --- Panel kaydı ve provenance ------------------------------------------------

def _tam_jr(record, model, primary="1.1", **kw):
    digest = evidence_digest(build_evidence(record, TASK))
    return {**_jr(model, primary),
            "source_run_id": record["run_id"], "experiment": "sentetik",
            "source_model": record["model"], "task_set": record["task_set"],
            "task_id": record["task_id"], "arm": record["arm"],
            "repeat": record["repeat"], "mast_schema_version": MAST_SCHEMA_VERSION,
            "evidence_sha256": digest, "mast_prompt_hash": prompt_contract_hash(), **kw}


def test_panel_kaydi_karar_alanlarini_tasir(monkeypatch):
    monkeypatch.setattr("eval.mast_labels._task_for", lambda r: TASK)
    record = _record()
    judges = [_tam_jr(record, m) for m in JUDGES]
    panel = build_panel([record], judges, experiment="sentetik", expected_judges=JUDGES)
    for alan in ("decision_rule_version", "self_judge_model", "external_judges",
                 "external_agreement_level", "external_consensus_label",
                 "self_matches_external", "adjudicator_required"):
        assert alan in panel[0], alan
    assert panel[0]["decision_rule_version"] == MAST_DECISION_RULE_VERSION
    assert panel[0]["self_judge_model"] == record["model"]


def test_panel_kaydi_kaynak_modeli_provenanstan_alir(monkeypatch):
    # Kaynak model panel ETİKETLERİNDEN tahmin edilmez: DeepSeek kaydında
    # self-judge DeepSeek'tir, etiketlerin içeriği ne olursa olsun.
    monkeypatch.setattr("eval.mast_labels._task_for", lambda r: TASK)
    record = _record(model=MODEL_SECONDARY)
    judges = [_tam_jr(record, m) for m in JUDGES]
    panel = build_panel([record], judges, experiment="sentetik", expected_judges=JUDGES)
    assert panel[0]["self_judge_model"] == MODEL_SECONDARY
    assert panel[0]["external_judges"] == [MODEL_MAIN, MODEL_JUDGE_EXTERNAL]


def test_panel_kaydi_json_serilestirilebilir(monkeypatch):
    monkeypatch.setattr("eval.mast_labels._task_for", lambda r: TASK)
    record = _record()
    panel = build_panel([record], [_tam_jr(record, m) for m in JUDGES],
                        experiment="sentetik", expected_judges=JUDGES)
    assert json.loads(json.dumps(panel[0]))["adjudicator_required"] is False


# --- MAST 2.0 artefaktları: otomatik migration YOK ---------------------------

def _mast_manifest(**over):
    snapshot = build_mast_manifest(_kaynak_manifest())
    snapshot.update(over)
    return snapshot


def test_20_manifesti_30_turuna_devam_ederken_fail_fast(tmp_path):
    # 2.0 turu üçlü çoğunlukla üretilmişti; aynı dosyaya leave-self-out
    # kararlarını eklemek iki farklı kuralın kararlarını tek artefaktta karıştırırdı.
    path = tmp_path / "manifest.json"
    eski = _mast_manifest(mast_schema_version="2.0")
    eski.pop("mast_decision_rule_version", None)
    path.write_text(json.dumps(eski), encoding="utf-8")
    with pytest.raises(MastPipelineError, match="manifest uyuşmazlığı"):
        check_or_write_mast_manifest(path, _mast_manifest())
    # Eski artefakt SİLİNMEZ/EZİLMEZ: tarih olarak durur.
    assert json.loads(path.read_text(encoding="utf-8"))["mast_schema_version"] == "2.0"


def test_karar_kurali_surumu_manifestte_kritik_alandir(tmp_path):
    assert "mast_decision_rule_version" in MAST_MANIFEST_CRITICAL
    path = tmp_path / "manifest.json"
    check_or_write_mast_manifest(path, _mast_manifest())
    yazilan = json.loads(path.read_text(encoding="utf-8"))
    assert yazilan["mast_decision_rule_version"] == MAST_DECISION_RULE_VERSION
    with pytest.raises(MastPipelineError, match="manifest uyuşmazlığı"):
        check_or_write_mast_manifest(
            path, _mast_manifest(mast_decision_rule_version="baska_kural"))


def test_20_semasiyla_yazilmis_etiket_oy_kullanamaz(monkeypatch):
    monkeypatch.setattr("eval.mast_labels._task_for", lambda r: TASK)
    record = _record()
    eski = [_tam_jr(record, m, mast_schema_version="2.0") for m in JUDGES]
    panel = build_panel([record], eski, experiment="sentetik", expected_judges=JUDGES)
    assert panel[0]["panel_complete"] is False
    assert panel[0]["superseded_judges"] == sorted(JUDGES)
    assert panel[0]["external_agreement_level"] == "incomplete"


def test_append_only_dosyada_yalnizca_30_etiketler_oy_kullanir(monkeypatch):
    # Dosya append-only: 2.0 etiketleri kalır ama karara giremez. Eskiler
    # "2.3" derken yeniler "1.1" diyor; panel yenileri görmeli.
    monkeypatch.setattr("eval.mast_labels._task_for", lambda r: TASK)
    record = _record()
    eski = [_tam_jr(record, m, "2.3", mast_schema_version="2.0") for m in JUDGES]
    yeni = [_tam_jr(record, m, "1.1") for m in JUDGES]
    kopya = json.dumps(eski, sort_keys=True)
    panel = build_panel([record], eski + yeni, experiment="sentetik",
                        expected_judges=JUDGES)
    assert panel[0]["panel_complete"] is True
    assert set(panel[0]["primary_modes"]) == {"1.1"}
    assert panel[0]["external_consensus_label"] == "1.1"
    # Eski kayıtlar DEĞİŞTİRİLMEZ.
    assert json.dumps(eski, sort_keys=True) == kopya


def test_mast_sema_surumu_uc_iki():
    assert MAST_SCHEMA_VERSION == "3.2"
    assert MAST_DECISION_RULE_VERSION == "leave_self_out_v1"


# --- CLI: bütün stage'ler açık, geçiş embargosu kalmadı ----------------------

@pytest.mark.parametrize("stage", ["judge", "adjudicate", "all"])
def test_cli_stageleri_embargoya_degil_eksik_deneye_takilir(monkeypatch, tmp_path, stage):
    # P1 Parça 4'te external-only Grok açıldı; hiçbir stage geçiş embargosuyla
    # durmamalı. Kapanan tek yol yanlış/eksik girdidir.
    monkeypatch.setattr("eval.mast_labels.call_model",
                        lambda *a, **k: pytest.fail("model çağrıldı"))
    monkeypatch.setenv("OPENROUTER_API_KEY", "x")
    monkeypatch.setattr(sys, "argv", ["prog", "--exp", str(tmp_path), "--stage", stage])
    with pytest.raises(SystemExit) as exc:
        mast_cli.main()
    assert "manifest" in str(exc.value)
    assert not hasattr(mast_cli, "ADJUDICATION_EMBARGO")
