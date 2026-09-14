"""eval/result_schema.py birim testleri — runner ↔ analiz sözleşmesi.

Bu sözleşme iki katmanı birbirine bağlar: runner kayıtları BU şemaya göre
yazar, analiz BU şemaya göre okur, testler BU şemadan fixture üretir. Üçünün
tek kaynaktan beslenmesi, "testler yeşil ama gerçek kayıt farklı" durumunu
yapısal olarak imkânsız kılar (EXPERIMENT_PROTOCOL.md §7-§8).
"""

import pytest

from config import ALL_ARMS, ARM_CONTRACT, ARM_NAIVE, ARM_STRUCTURED, RESULT_SCHEMA_VERSION
from eval.result_schema import (
    IDENTITY_FIELDS,
    RESUME_KEY_FIELDS,
    expected_resume_keys,
    find_duplicates,
    find_missing,
    find_unexpected,
    integrity_report,
    is_run_error,
    make_run_error_record,
    make_synthetic_record,
    resume_key,
    stamp_record,
    validate_record,
    verify_resumable,
)

ERROR_KW = dict(experiment="x", model="m", task_set="heldout", arm="baseline",
                task_id="t1", repeat=0, run_id="r1", arm_position=0, error="boom")


# --- Anahtarlar --------------------------------------------------------------

def test_resume_anahtari_modeli_icerir():
    # Model dahil DEĞİLSE, ikinci modelin koşusu birincininkiyle karışır.
    assert "model" in RESUME_KEY_FIELDS
    assert "experiment" in IDENTITY_FIELDS


def test_resume_anahtari_alan_sirasina_gore_uretilir():
    r = make_synthetic_record(model="m1", arm="naive", task_id="t7", repeat=2)
    assert resume_key(r) == ("m1", "naive", "t7", 2)


# --- Sentetik kayıt fabrikası ------------------------------------------------

@pytest.mark.parametrize("arm", ALL_ARMS)
def test_sentetik_kayit_her_kolda_gecerli(arm):
    # Fixture üreticisi geçersiz kayıt üretirse Parça 5'in bütün testleri
    # yanlış bir şema üzerinde çalışırdı.
    assert validate_record(make_synthetic_record(arm=arm)) == []


def test_sentetik_kayit_base_dususe_plusu_da_dusurur():
    # Plus base'i kapsar; base_pass=False + plus_pass=True geçersiz bir kayıttır.
    r = make_synthetic_record(base_pass=False, plus_pass=True)
    assert r["base_pass"] is False and r["plus_pass"] is False
    assert validate_record(r) == []


def test_sentetik_kayit_ek_alan_kabul_eder():
    r = make_synthetic_record(arm=ARM_CONTRACT, attempt_count=3)
    assert r["attempt_count"] == 3


# --- Doğrulama ---------------------------------------------------------------

def test_eksik_zorunlu_alan_yakalanir():
    r = make_synthetic_record()
    del r["model"]
    assert any("model" in p for p in validate_record(r))


def test_sema_surumu_uyusmazligi_yakalanir():
    r = make_synthetic_record()
    r["schema_version"] = "0.9"
    assert any("şema sürümü" in p for p in validate_record(r))


def test_bilinmeyen_kol_yakalanir():
    r = make_synthetic_record()
    r["arm"] = "uydurma_kol"
    assert any("bilinmeyen kol" in p for p in validate_record(r))


def test_tutarsiz_base_plus_yakalanir():
    # Elle kurulan tutarsız kayıt (fabrika bunu üretmez) doğrulamadan geçmemeli.
    r = make_synthetic_record(base_pass=False)
    r["plus_pass"] = True
    assert any("tutarsız sonuç" in p for p in validate_record(r))


def test_bool_olmayan_pass_alani_yakalanir():
    r = make_synthetic_record()
    r["plus_pass"] = "evet"
    assert any("plus_pass" in p for p in validate_record(r))


@pytest.mark.parametrize("arm,alan", [
    (ARM_STRUCTURED, "handoff_parse_ok"),
    (ARM_CONTRACT, "handoff_validation"),
])
def test_kola_ozgu_zorunlu_alan_yakalanir(arm, alan):
    # Bu alanlar eksilirse RQ3 ölçülemez hale gelir -> şema düzeyinde zorunlu.
    r = make_synthetic_record(arm=arm)
    del r[alan]
    assert any(alan in p for p in validate_record(r))


def test_naive_kolda_handoff_parse_ok_zorunlu_degil():
    # naive'de JSON parse'ı diye bir şey yok; zorunlu tutmak anlamsız olurdu.
    r = make_synthetic_record(arm=ARM_NAIVE)
    r.pop("handoff_parse_ok", None)
    assert validate_record(r) == []


# --- run_error ---------------------------------------------------------------

def test_run_error_kaydi_gecerli_ve_isaretli():
    r = make_run_error_record(**ERROR_KW)
    assert is_run_error(r)
    assert validate_record(r) == []
    assert r["schema_version"] == RESULT_SCHEMA_VERSION


def test_run_error_hata_metni_olmadan_gecersiz():
    r = make_run_error_record(**{**ERROR_KW, "error": ""})
    assert any("error" in p for p in validate_record(r))


def test_run_error_sonuc_alani_istemez():
    # Ölçüm yapılamadı; base/plus alanları beklenmez.
    assert "plus_pass" not in make_run_error_record(**ERROR_KW)


# --- Damgalama ---------------------------------------------------------------

def test_stamp_ham_kayit_otoritatif_alanlari_EZEMEZ():
    # Bir kol implementasyonu kendi "model"/"experiment"/"schema_version"
    # alanını yazarsa runner'ın kimliği sessizce geçersiz kalırdı ve kayıt
    # analizde YANLIŞ MODELE atfedilirdi.
    ham = {"ts": "t", "status": "passed", "schema_version": "SAHTE",
           "experiment": "SAHTE", "model": "SAHTE", "task_set": "SAHTE",
           "run_id": "SAHTE", "arm_position": 99,
           "base_status": "passed", "base_pass": True, "base_error_class": None,
           "base_duration_s": 0.1, "plus_status": "passed", "plus_pass": True,
           "plus_error_class": None, "plus_duration_s": 0.1,
           "base_plus_available": True, "error_class": None}
    r = stamp_record(ham, experiment="gercek", model="gercek/model",
                     task_set="heldout", arm="baseline", task_id="t1", repeat=1,
                     run_id="gercek-run", arm_position=2)
    assert r["schema_version"] == RESULT_SCHEMA_VERSION
    assert r["experiment"] == "gercek"
    assert r["model"] == "gercek/model"
    assert r["task_set"] == "heldout"
    assert r["run_id"] == "gercek-run"
    assert r["arm_position"] == 2
    assert validate_record(r) == []


def test_stamp_kimlik_alanlarini_ezer():
    # Kol implementasyonu yanlış bir arm/task_id yazsa bile runner'ın verdiği
    # kimlik kazanır: kimliğin tek kaynağı runner'dır.
    ham = {"ts": "t", "arm": "YANLIS", "task_id": "YANLIS", "status": "passed",
           "base_status": "passed", "base_pass": True, "base_error_class": None,
           "base_duration_s": 0.1, "plus_status": "passed", "plus_pass": True,
           "plus_error_class": None, "plus_duration_s": 0.1,
           "base_plus_available": True, "error_class": None}
    r = stamp_record(ham, experiment="e", model="m", task_set="heldout",
                     arm="baseline", task_id="t1", repeat=1, run_id="r", arm_position=2)
    assert (r["arm"], r["task_id"], r["repeat"], r["arm_position"]) == ("baseline", "t1", 1, 2)
    assert r["schema_version"] == RESULT_SCHEMA_VERSION
    assert validate_record(r) == []


# --- Bütünlük kontrolleri ----------------------------------------------------

def test_beklenen_anahtar_sayisi():
    keys = expected_resume_keys(["t1", "t2"], ALL_ARMS, 3, "m")
    assert len(keys) == 2 * 4 * 3


def test_duplicate_tespiti():
    a = make_synthetic_record(model="m", arm="naive", task_id="t1", repeat=0)
    b = make_synthetic_record(model="m", arm="naive", task_id="t1", repeat=0)
    dups = find_duplicates([a, b])
    assert dups == {("m", "naive", "t1", 0): 2}


def test_run_error_sonrasi_yeniden_deneme_duplicate_sayilmaz():
    # Normal resume akışı: bir run_error + bir gerçek kayıt aynı anahtarda.
    hata = make_run_error_record(**{**ERROR_KW, "arm": "naive", "task_id": "t1"})
    ok = make_synthetic_record(model="m", arm="naive", task_id="t1", repeat=0)
    assert find_duplicates([hata, ok]) == {}


def test_eksik_kosu_tespiti():
    expected = expected_resume_keys(["t1"], ["baseline", "naive"], 1, "m")
    kayitlar = [make_synthetic_record(model="m", arm="baseline", task_id="t1", repeat=0)]
    assert find_missing(kayitlar, expected) == {("m", "naive", "t1", 0)}


def test_run_error_eksik_sayilir():
    expected = expected_resume_keys(["t1"], ["baseline"], 1, "m")
    hata = make_run_error_record(**ERROR_KW)
    assert find_missing([hata], expected) == {("m", "baseline", "t1", 0)}


def test_butunluk_raporu_eksiksiz_deneyi_onaylar():
    kayitlar = [make_synthetic_record(model="m", arm=arm, task_id=t, repeat=rep)
                for rep in range(2) for t in ("t1", "t2") for arm in ALL_ARMS]
    rapor = integrity_report(kayitlar, ["t1", "t2"], ALL_ARMS, 2, "m")
    assert rapor["complete"] is True
    assert rapor["expected_count"] == rapor["completed_count"] == 16
    assert rapor["duplicates"] == {} and rapor["missing"] == []


def test_beklenmeyen_anahtar_tespiti():
    # Yanlış modelin/görevin kaydı dosyaya girmişken beklenenlerin hepsi de
    # tamamlanmışsa, yalnız `missing`e bakan bir kontrol deneyi yanlışlıkla
    # `complete=True` sayardı.
    expected = expected_resume_keys(["t1"], ["baseline"], 1, "m")
    kayitlar = [
        make_synthetic_record(model="m", arm="baseline", task_id="t1", repeat=0),
        make_synthetic_record(model="BASKA", arm="baseline", task_id="t1", repeat=0),
    ]
    assert find_unexpected(kayitlar, expected) == {("BASKA", "baseline", "t1", 0)}
    rapor = integrity_report(kayitlar, ["t1"], ["baseline"], 1, "m")
    assert rapor["missing"] == []          # beklenenlerin hepsi tamam
    assert rapor["unexpected"]             # ama fazladan kayıt var
    assert rapor["complete"] is False


def test_devam_denetimi_temiz_dosyayi_gecirir():
    expected = expected_resume_keys(["t1", "t2"], ["baseline"], 1, "m")
    kayitlar = [make_synthetic_record(model="m", arm="baseline", task_id="t1", repeat=0)]
    assert verify_resumable(kayitlar, expected) == []   # eksik olması sorun DEĞİL


def test_devam_denetimi_gecmis_run_errora_izin_verir():
    expected = expected_resume_keys(["t1"], ["baseline"], 1, "m")
    hata = make_run_error_record(**ERROR_KW)
    assert verify_resumable([hata], expected) == []


@pytest.mark.parametrize("bozucu", ["gecersiz", "duplicate", "beklenmeyen"])
def test_devam_denetimi_bozuk_dosyada_durdurur(bozucu):
    # Yeni (pahalı) API çağrıları yapılmadan ÖNCE durulmalı.
    expected = expected_resume_keys(["t1"], ["baseline"], 1, "m")
    temel = make_synthetic_record(model="m", arm="baseline", task_id="t1", repeat=0)
    if bozucu == "gecersiz":
        temel["schema_version"] = "0.1"
        kayitlar = [temel]
    elif bozucu == "duplicate":
        kayitlar = [temel, make_synthetic_record(model="m", arm="baseline",
                                                 task_id="t1", repeat=0)]
    else:
        kayitlar = [temel, make_synthetic_record(model="BASKA", arm="baseline",
                                                 task_id="t1", repeat=0)]
    assert verify_resumable(kayitlar, expected)


def test_butunluk_raporu_gecersiz_kaydi_yakalar():
    bozuk = make_synthetic_record(model="m", arm="baseline", task_id="t1", repeat=0)
    bozuk["schema_version"] = "0.1"
    rapor = integrity_report([bozuk], ["t1"], ["baseline"], 1, "m")
    assert rapor["complete"] is False
    assert rapor["invalid_records"]
