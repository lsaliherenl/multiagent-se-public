"""Çift panel hash'i ve adjudication resume tazeliği (§9.2) — LLM'siz.

Kapatılan sessiz hata: **tek bir `panel_input_sha256`, iki farklı tazelik
sorusunun cevabıymış gibi kullanılıyordu.**

* "Tanısal panel hâlâ güncel mi?" — üç etiketin tamamına bakar.
* "Grok'un kararı hâlâ geçerli mi?" — YALNIZ karara giren iki dış etikete bakar.

İkisi tek hash'te toplandığında, kaydı ÜRETEN modelin kendi etiketi (karara hiç
katılmayan etiket) değiştiğinde geçerli bir Grok kararı stale sayılırdı: gereksiz
ücretli çağrı ve kararın yanlış bir girdiye bağımlı gösterilmesi.

Hash TEK BAŞINA kimlik değildir. Kanıt paketi körlenmiş olduğu için iki farklı
deneyin/modelin aynı görevdeki girdisi aynı hash'i verebilir; bu yüzden resume
kimliğinde source run, adjudicator modeli, şema sürümü, kanıt ve prompt hash'i
AYRI alanlar olarak durur.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from config import (
    ARM_CONTRACT,
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
    ADJUDICATION_IDENTITY_FIELDS,
    MAST_MANIFEST_CRITICAL,
    _stored_adjudication_identity,
    adjudicate_record,
    adjudication_identity,
    build_evidence,
    build_mast_manifest,
    build_panel,
    check_or_write_mast_manifest,
    panel_blockers,
    prompt_contract_hash,
    run_adjudication,
)
from eval.mast_schema import (
    MastPanelVerdict,
    MastPipelineError,
    decision_input_digest,
    evidence_digest,
    full_panel_input_digest,
)
from eval.result_schema import make_synthetic_record

JUDGES = tuple(MODEL_JUDGES)
TASK = {"task_id": "t000", "prompt": "def f(x):\n    ...", "entry_point": "f"}
HEX64 = 64


def _record(model=MODEL_MAIN, **extra):
    record = make_synthetic_record(experiment="sentetik", model=model, task_set="heldout",
                                   arm=ARM_CONTRACT, task_id="t000", repeat=0,
                                   base_pass=False, plus_pass=False)
    record.setdefault("code", "def f(x):\n    return 0\n")
    record.setdefault("plan", {"task_id": "t000"})
    record.setdefault("raw_messages", [{"from": "planner", "to": "coder", "content": "p"}])
    record.update(extra)
    return record


def _jr(model, primary="1.1", **kw):
    """Prompt'ta görülen alanlar + görülmeyen provenance/deneme alanları."""
    return {"judge_model": model, "judge_status": "ok",
            "primary_mode": primary, "secondary_modes": [], "confidence": "high",
            "rationale": f"{model} gerekçesi", "insufficient_context": False,
            "judge_attempt": 1, "ts": "2026-07-29T00:00:00+00:00",
            "judge_raw": "{}", "judge_raw_sha256": "0" * 64, **kw}


def _uclu(**over):
    return [_jr(m, **over.get(m, {})) for m in JUDGES]


def _tam(kayitlar):
    return full_panel_input_digest(kayitlar, JUDGES)


def _karar(kayitlar, kaynak=MODEL_MAIN):
    return decision_input_digest(kayitlar, JUDGES, source_model=kaynak)


# --- Kanonikleştirme ---------------------------------------------------------

def test_full_hash_liste_sirasindan_bagimsiz():
    a = _uclu()
    b = [a[2], a[0], a[1]]
    assert _tam(a) == _tam(b)


def test_decision_hash_liste_sirasindan_bagimsiz():
    a = _uclu()
    b = list(reversed(a))
    assert _karar(a) == _karar(b)


def test_iki_hash_ayni_kayitlardan_bile_FARKLI():
    # Payload hash TÜRÜNÜ içerir; içermeseydi self eksikken hesaplanan full ve
    # decision hash'leri çakışıp "tam panel" ile "karar girdisi" ayrımı kaybolurdu.
    kayitlar = _uclu()
    assert _tam(kayitlar) != _karar(kayitlar)


def test_hash_surumu_payloadun_icinde():
    # Sürüm payload'da olmasaydı, kanonikleştirme kuralı değiştiğinde eski ve
    # yeni hash'ler sessizce karşılaştırılır ve resume yanlış çalışırdı.
    kayitlar = _uclu()
    assert MAST_PANEL_HASH_VERSION == "dual_input_v1"
    assert len(_tam(kayitlar)) == HEX64 and _tam(kayitlar).islower()


HASHSEED_SCRIPT = '''
import json, sys
sys.path.insert(0, sys.argv[1])
from config import MODEL_JUDGES
from eval.mast_schema import decision_input_digest, full_panel_input_digest
kayitlar = json.loads(open(sys.argv[2], encoding="utf-8").read())
print(json.dumps({
    "full": full_panel_input_digest(kayitlar, MODEL_JUDGES),
    "decision": decision_input_digest(kayitlar, MODEL_JUDGES, source_model=sys.argv[3]),
}))
'''


@pytest.mark.parametrize("seed_env", ["0", "1", "12345"])
def test_hashler_PYTHONHASHSEEDden_bagimsiz(tmp_path, seed_env):
    # Sözlük/küme sırası süreçler arasında değişir; hash bunlara bağlanırsa
    # "aynı içerik aynı hash" iddiası (ve dolayısıyla resume) çöker.
    betik = tmp_path / "h.py"
    betik.write_text(HASHSEED_SCRIPT, encoding="utf-8")
    veri = tmp_path / "kayitlar.json"
    kayitlar = _uclu()
    veri.write_text(json.dumps(kayitlar), encoding="utf-8")
    kok = str(Path(__file__).resolve().parents[1])
    sonuc = subprocess.run(
        [sys.executable, str(betik), kok, str(veri), MODEL_MAIN],
        env={**os.environ, "PYTHONHASHSEED": seed_env},
        capture_output=True, text=True, timeout=120)
    assert sonuc.returncode == 0, sonuc.stderr
    assert json.loads(sonuc.stdout) == {"full": _tam(kayitlar),
                                        "decision": _karar(kayitlar)}


@pytest.mark.parametrize("hesap", [_tam, _karar], ids=["full", "decision"])
def test_yinelenen_judge_fail_fast(hesap):
    kayitlar = [*_uclu(), _jr(MODEL_MAIN, "3.1")]
    with pytest.raises(MastPipelineError, match="birden fazla etiket"):
        hesap(kayitlar)


@pytest.mark.parametrize("hesap", [_tam, _karar], ids=["full", "decision"])
def test_beklenmeyen_judge_fail_fast(hesap):
    kayitlar = [*_uclu(), _jr("openrouter/rogue/judge")]
    with pytest.raises(MastPipelineError, match="beklenmeyen judge"):
        hesap(kayitlar)


def test_full_hash_eksik_judgeda_fail_fast():
    with pytest.raises(MastPipelineError, match="eksik judge"):
        _tam(_uclu()[:2])


def test_decision_hash_eksik_dis_judgeda_fail_fast():
    # Kaynak Gemini; MiniMax (dış judge) eksik.
    kayitlar = [r for r in _uclu() if r["judge_model"] != MODEL_JUDGE_EXTERNAL]
    with pytest.raises(MastPipelineError, match="eksik judge"):
        _karar(kayitlar)


def test_decision_hash_self_etiketi_OLMADAN_hesaplanabilir():
    # Parça 4'te prompt'a yalnız iki dış etiket gidecek; hash o kümeden
    # hesaplanabilmeli. Self'in kümede bulunması hash'i DEĞİŞTİRMEZ.
    kayitlar = _uclu()
    yalniz_dis = [r for r in kayitlar if r["judge_model"] != MODEL_MAIN]
    assert _karar(yalniz_dis) == _karar(kayitlar)


# --- Self değişimi: karar girdisi değişmez ------------------------------------

SELF_DEGISIMLERI = [
    ("primary_mode", "3.1"),
    ("rationale", "tamamen farklı bir gerekçe"),
    ("confidence", "low"),
]


@pytest.mark.parametrize("kaynak", [MODEL_MAIN, MODEL_SECONDARY],
                         ids=["gemini", "deepseek"])
@pytest.mark.parametrize("alan,deger", SELF_DEGISIMLERI)
def test_self_degisimi_yalniz_full_hashi_degistirir(kaynak, alan, deger):
    once = _uclu()
    sonra = [dict(r, **({alan: deger} if r["judge_model"] == kaynak else {}))
             for r in once]
    assert _tam(once) != _tam(sonra), "tanısal panel tazeliği yakalanmadı"
    assert _karar(once, kaynak) == _karar(sonra, kaynak), \
        "self etiketi karar girdisine sızdı"


@pytest.mark.parametrize("kaynak", [MODEL_MAIN, MODEL_SECONDARY],
                         ids=["gemini", "deepseek"])
def test_self_degisimi_adjudication_kimligini_bozmaz(kaynak):
    record = _record(model=kaynak)
    once, sonra = _uclu(), _uclu(**{kaynak: {"primary_mode": "3.1"}})

    def _kimlik(kayitlar):
        return adjudication_identity(
            record, model=MODEL_ADJUDICATOR, evidence_sha256="e" * 64,
            prompt_hash="p" * 64,
            decision_input_sha256=_karar(kayitlar, kaynak))

    assert _kimlik(once) == _kimlik(sonra), \
        "karara katılmayan bir etiket yüzünden Grok kararı stale sayıldı"


# --- External değişimi: karar girdisi DEĞİŞİR ---------------------------------

@pytest.mark.parametrize("alan,deger", [
    ("primary_mode", "3.1"),
    ("secondary_modes", ["3.2"]),
    ("rationale", "başka bir gerekçe"),
    ("confidence", "low"),
    ("insufficient_context", True),
])
def test_dis_judge_degisimi_iki_hashi_de_degistirir(alan, deger):
    once = _uclu()
    sonra = [dict(r, **({alan: deger} if r["judge_model"] == MODEL_JUDGE_EXTERNAL else {}))
             for r in once]
    assert _tam(once) != _tam(sonra)
    assert _karar(once) != _karar(sonra)


def test_dis_judge_degisimi_adjudication_kimligini_bozar():
    record = _record()
    once = _uclu()
    sonra = _uclu(**{MODEL_JUDGE_EXTERNAL: {"primary_mode": "3.1"}})

    def _kimlik(kayitlar):
        return adjudication_identity(
            record, model=MODEL_ADJUDICATOR, evidence_sha256="e" * 64,
            prompt_hash="p" * 64, decision_input_sha256=_karar(kayitlar))

    assert _kimlik(once) != _kimlik(sonra), \
        "değişen karar girdisi üzerinde eski Grok kararı güncel sayıldı"


@pytest.mark.parametrize("alan,deger", [
    ("judge_attempt", 7),
    ("ts", "2027-01-01T00:00:00+00:00"),
    ("judge_raw", '{"tamamen": "farklı ham yanıt"}'),
    ("judge_raw_sha256", "f" * 64),
])
def test_prompt_disi_alanlar_hicbir_hashi_degistirmez(alan, deger):
    # Yeniden denenip AYNI kararı veren bir judge adjudication'ı geçersiz
    # kılmamalı: adjudicator etiketin İÇERİĞİNE bakar, kaçıncı denemede hangi
    # ham yanıttan üretildiğine değil.
    once = _uclu()
    sonra = [dict(r, **{alan: deger}) for r in once]
    assert _tam(once) == _tam(sonra)
    assert _karar(once) == _karar(sonra)


# --- Panel kaydındaki hash'ler ------------------------------------------------

def _tam_jr(record, model, **kw):
    digest = evidence_digest(build_evidence(record, TASK))
    return {**_jr(model, **kw),
            "source_run_id": record["run_id"], "experiment": "sentetik",
            "source_model": record["model"], "task_set": record["task_set"],
            "task_id": record["task_id"], "arm": record["arm"],
            "repeat": record["repeat"], "mast_schema_version": MAST_SCHEMA_VERSION,
            "evidence_sha256": digest, "mast_prompt_hash": prompt_contract_hash()}


def _panel(monkeypatch, record, modeller):
    monkeypatch.setattr("eval.mast_labels._task_for", lambda r: TASK)
    judges = [_tam_jr(record, m) for m in modeller]
    return build_panel([record], judges, experiment="sentetik", expected_judges=JUDGES)[0]


def test_tam_panelde_iki_hash_de_64_hex(monkeypatch):
    p = _panel(monkeypatch, _record(), JUDGES)
    for alan in ("full_panel_input_sha256", "decision_input_sha256"):
        assert len(p[alan]) == HEX64 and p[alan].islower(), alan
    assert p["panel_hash_version"] == MAST_PANEL_HASH_VERSION
    assert "panel_input_sha256" not in p


def test_self_eksikken_karar_hashi_KORUNUR(monkeypatch):
    # Karar girdisi tamamdır; eksik olan yalnız tanısal alandır. Yine de panel
    # tamamlanmamıştır ve panel_blockers adjudication'ı durdurur.
    record = _record()
    p = _panel(monkeypatch, record, [MODEL_SECONDARY, MODEL_JUDGE_EXTERNAL])
    assert p["full_panel_input_sha256"] is None
    assert len(p["decision_input_sha256"]) == HEX64
    assert p["panel_complete"] is False
    assert p["missing_judges"] == [MODEL_MAIN]
    assert panel_blockers([p]), "eksik panelde adjudication engellenmedi"


def test_bir_dis_judge_eksikken_iki_hash_de_yok(monkeypatch):
    p = _panel(monkeypatch, _record(), [MODEL_MAIN, MODEL_SECONDARY])
    assert p["full_panel_input_sha256"] is None
    assert p["decision_input_sha256"] is None
    assert p["external_agreement_level"] == "incomplete"


def test_panel_hashleri_oy_kullanan_kayitlardan_uretilir(monkeypatch):
    # Hash, panelin oy saydığı kümenin AYNISINDAN gelmeli; ayrı hesaplanırsa
    # iki küme sessizce ayrışabilirdi.
    record = _record()
    monkeypatch.setattr("eval.mast_labels._task_for", lambda r: TASK)
    judges = [_tam_jr(record, m) for m in JUDGES]
    p = build_panel([record], judges, experiment="sentetik", expected_judges=JUDGES)[0]
    assert p["full_panel_input_sha256"] == full_panel_input_digest(judges, JUDGES)
    assert p["decision_input_sha256"] == decision_input_digest(
        judges, JUDGES, source_model=record["model"])


# --- Panel sözleşmesi: hash alanları -----------------------------------------

def _panel_alanlari(monkeypatch, record=None):
    p = _panel(monkeypatch, record or _record(), JUDGES)
    return {k: p[k] for k in MastPanelVerdict.model_fields}


def test_eski_tek_hashli_panel_kaydi_reddedilir(monkeypatch):
    # MAST 3.0 panel satırı: yeni alanlar yok, geçici alan var.
    alanlar = _panel_alanlari(monkeypatch)
    eski = {k: v for k, v in alanlar.items()
            if k not in ("panel_hash_version", "full_panel_input_sha256",
                         "decision_input_sha256")}
    eski["panel_input_sha256"] = "a" * 64
    with pytest.raises(Exception):
        MastPanelVerdict(**eski)


def test_panel_kaydi_eski_alani_fazladan_tasiyamaz(monkeypatch):
    alanlar = _panel_alanlari(monkeypatch)
    with pytest.raises(Exception):
        MastPanelVerdict(**alanlar, panel_input_sha256="a" * 64)


@pytest.mark.parametrize("bozuk", [
    {"panel_hash_version": "tek_hash_v0"},
    {"full_panel_input_sha256": "KISA"},
    {"full_panel_input_sha256": "A" * 64},          # büyük harf
    {"full_panel_input_sha256": "z" * 64},          # hex değil
    {"full_panel_input_sha256": None},              # tam panelde zorunlu
    {"decision_input_sha256": None},                # iki dış etiket varken zorunlu
])
def test_hash_alanlarinin_capraz_invariantlari(monkeypatch, bozuk):
    alanlar = _panel_alanlari(monkeypatch)
    with pytest.raises(Exception):
        MastPanelVerdict(**{**alanlar, **bozuk})


# --- Manifest -----------------------------------------------------------------

def _kaynak_manifest():
    return {"name": "live", "model": MODEL_MAIN, "task_set": "heldout",
            "task_ids": ["t000"], "arm_order": [ARM_CONTRACT], "repeats": 1,
            "result_schema_version": "2.0", "llm_call_schema_version": "2.0"}


def test_hash_surumu_manifestin_kritik_alani(tmp_path):
    assert "mast_panel_hash_version" in MAST_MANIFEST_CRITICAL
    yol = tmp_path / "manifest.json"
    snapshot = build_mast_manifest(_kaynak_manifest())
    assert snapshot["mast_panel_hash_version"] == MAST_PANEL_HASH_VERSION
    check_or_write_mast_manifest(yol, snapshot)
    with pytest.raises(MastPipelineError, match="manifest uyuşmazlığı"):
        check_or_write_mast_manifest(
            yol, dict(snapshot, mast_panel_hash_version="tek_hash_v0"))


def test_MAST_30_manifesti_31_turuna_devam_ederken_fail_fast(tmp_path):
    yol = tmp_path / "manifest.json"
    eski = build_mast_manifest(_kaynak_manifest())
    eski["mast_schema_version"] = "3.0"
    eski.pop("mast_panel_hash_version")
    yol.write_text(json.dumps(eski), encoding="utf-8")
    with pytest.raises(MastPipelineError, match="manifest uyuşmazlığı"):
        check_or_write_mast_manifest(yol, build_mast_manifest(_kaynak_manifest()))
    # Eski artefakt EZİLMEZ.
    assert json.loads(yol.read_text(encoding="utf-8"))["mast_schema_version"] == "3.0"


def test_30_semasiyla_yazilmis_etiket_31_turunda_oy_kullanamaz(monkeypatch):
    monkeypatch.setattr("eval.mast_labels._task_for", lambda r: TASK)
    record = _record()
    eski = [dict(_tam_jr(record, m), mast_schema_version="3.0") for m in JUDGES]
    p = build_panel([record], eski, experiment="sentetik", expected_judges=JUDGES)[0]
    assert p["panel_complete"] is False
    assert p["superseded_judges"] == sorted(JUDGES)
    assert p["full_panel_input_sha256"] is None


# --- Adjudication resume kimliği ---------------------------------------------

def test_resume_kimliginde_full_hash_YOK():
    assert "decision_input_sha256" in ADJUDICATION_IDENTITY_FIELDS
    assert "full_panel_input_sha256" not in ADJUDICATION_IDENTITY_FIELDS
    assert "panel_input_sha256" not in ADJUDICATION_IDENTITY_FIELDS


def test_eski_panel_input_sha256_tasiyan_adjudication_eslesmez():
    record = _record()
    kayitlar = _uclu()
    kimlik = adjudication_identity(
        record, model=MODEL_ADJUDICATOR, evidence_sha256="e" * 64,
        prompt_hash="p" * 64, decision_input_sha256=_karar(kayitlar))
    # MAST 3.0 döneminde yazılmış kayıt: alan adı bile farklı.
    eski_kayit = {"source_run_id": record["run_id"],
                  "adjudicator_model": MODEL_ADJUDICATOR,
                  "mast_schema_version": "3.0", "evidence_sha256": "e" * 64,
                  "mast_prompt_hash": "p" * 64,
                  "panel_input_sha256": _tam(kayitlar)}
    assert _stored_adjudication_identity(eski_kayit) != kimlik
    # Otomatik alan kopyalama YOK: yeni alan hiç yazılmamıştır.
    assert eski_kayit.get("decision_input_sha256") is None


# --- External-only kapı: Grok'a self verilebilen yol yok ----------------------

def test_adjudicate_record_TAM_UCLUYU_reddeder(monkeypatch):
    # Programatik doğrudan çağrı, run_adjudication'ın ön geçişini atlar; kapı
    # burada da olmasaydı self etiketi Grok prompt'una girerdi.
    monkeypatch.setattr("eval.mast_labels.call_model",
                        lambda *a, **k: pytest.fail("model çağrıldı — external-only kapısı yok"))
    record = _record()
    with pytest.raises(MastPipelineError, match="kanonik dış ikili"):
        adjudicate_record(record, build_evidence(record, TASK), _uclu(),
                          experiment="sentetik")


def test_adjudicate_record_yanlis_adjudicatoru_cagridan_once_reddeder(monkeypatch):
    monkeypatch.setattr("eval.mast_labels.call_model",
                        lambda *a, **k: pytest.fail("model çağrıldı"))
    record = _record()
    dis = [r for r in _uclu() if r["judge_model"] != MODEL_MAIN]
    with pytest.raises(MastPipelineError, match="dondurulmuş adjudicator"):
        adjudicate_record(record, build_evidence(record, TASK), dis,
                          experiment="sentetik", model=MODEL_JUDGE_EXTERNAL)


def test_run_adjudication_yanlis_adjudicatoru_cagridan_once_reddeder(monkeypatch):
    monkeypatch.setattr("eval.mast_labels.call_model",
                        lambda *a, **k: pytest.fail("model çağrıldı"))
    with pytest.raises(MastPipelineError, match="dondurulmuş adjudicator"):
        run_adjudication([], [], [], experiment="sentetik", model="openrouter/rogue/adj")


def test_resume_kimligi_hala_yalniz_karar_hashine_bagli():
    assert "decision_input_sha256" in ADJUDICATION_IDENTITY_FIELDS
    assert "full_panel_input_sha256" not in ADJUDICATION_IDENTITY_FIELDS


@pytest.mark.parametrize("stage", ["judge", "adjudicate", "all"])
def test_cli_stageleri_acik(monkeypatch, tmp_path, stage):
    monkeypatch.setattr("eval.mast_labels.call_model",
                        lambda *a, **k: pytest.fail("model çağrıldı"))
    monkeypatch.setenv("OPENROUTER_API_KEY", "x")
    monkeypatch.setattr(sys, "argv", ["prog", "--exp", str(tmp_path), "--stage", stage])
    with pytest.raises(SystemExit) as exc:
        mast_cli.main()
    # Geçiş embargosu değil, eksik deney dizini.
    assert "manifest" in str(exc.value)


def test_cli_yardimi_calismaya_devam_eder(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["prog", "--help"])
    with pytest.raises(SystemExit) as exc:
        mast_cli.main()
    assert exc.value.code == 0
    assert "--stage" in capsys.readouterr().out
