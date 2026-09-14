"""analysis/analyze.py birim testleri — LLM'siz, tamamen sentetik.

Her fixture `eval/result_schema` fabrikalarından üretilir; alan adları burada
elle yazılmaz. Böylece sonuç sözleşmesi değişirse testler de yeni sözleşmeye
göre üretir ve "testler yeşil ama gerçek kayıt farklı" durumu oluşmaz.

Bu dosyanın amacı, analiz katmanının SESSİZCE yanlış sonuç üretebileceği
noktaları önceden kilitlemek: kümelenmenin yok sayılması, bootstrap biriminin
kayması, çift sayım, base/plus karışması, model havuzlanması, kullanım
verisinin yanlış birleşmesi.
"""

import json

import pytest

from analysis.analyze import (
    AnalysisError,
    analyze,
    bootstrap_draws,
    paired_diffs,
    task_level_rates,
    usage_by_run,
    write_outputs,
)
from config import (
    ARM_BASELINE,
    ARM_CONTRACT,
    ARM_NAIVE,
    ARM_STRUCTURED,
    LLM_CALL_SCHEMA_VERSION,
    MAX_PLANNER_ATTEMPTS,
    RESULT_SCHEMA_VERSION,
)
from eval.result_schema import make_run_error_record, make_synthetic_llm_call, make_synthetic_record

ARMS = [ARM_BASELINE, ARM_NAIVE, ARM_STRUCTURED, ARM_CONTRACT]
MODEL = "test/model"
# Bootstrap testlerde küçük tutulur (10.000 yineleme × onlarca test = gereksiz
# yavaşlık); belirlenimcilik yineleme sayısından bağımsızdır.
ITER = 200


def _manifest(task_ids, arms=ARMS, repeats=3, model=MODEL, name="sentetik"):
    return {"name": name, "model": model, "task_set": "heldout",
            "task_ids": list(task_ids), "arm_order": list(arms), "repeats": repeats,
            "result_schema_version": RESULT_SCHEMA_VERSION,
            "llm_call_schema_version": LLM_CALL_SCHEMA_VERSION, "git_commit": "deadbeef"}


def _records(task_ids, passes, *, arms=ARMS, repeats=3, model=MODEL,
             base_passes=None):
    """passes: (task_id, arm) -> kaç tekrarda geçtiği (0..repeats).

    base_passes verilmezse base ile plus aynı kabul edilir.
    """
    out = []
    for rep in range(repeats):
        for task_id in task_ids:
            for arm in arms:
                plus = rep < passes(task_id, arm)
                base = plus if base_passes is None else rep < base_passes(task_id, arm)
                out.append(make_synthetic_record(
                    model=model, arm=arm, task_id=task_id, repeat=rep,
                    base_pass=base, plus_pass=plus))
    return out


def _calls(records, **kwargs):
    """Her tamamlanmış koşu için en az bir başarılı çağrı kaydı.

    Analiz artık bunu ZORUNLU kılıyor (eksik kullanım logu sıfır maliyet gibi
    görünürdü), bu yüzden kullanımla ilgilenmeyen testler de üretmek zorunda.
    """
    return [make_synthetic_llm_call(run_id=r["run_id"], model=r["model"], arm=r["arm"],
                                    task_id=r["task_id"], repeat=r["repeat"], **kwargs)
            for r in records if r["status"] != "run_error"]


def _split(favor_a, arm_a=ARM_CONTRACT, arm_b=ARM_NAIVE, repeats=3):
    """arm_a, favor_a görevinde tam geçer; arm_b hiç geçmez; diğerleri berabere."""
    def passes(task_id, arm):
        if arm == arm_a:
            return repeats
        if arm == arm_b:
            return repeats if task_id not in favor_a else 0
        return repeats
    return passes


# --- 1. Kümelenme: 150 tekrar 150 bağımsız gözlem DEĞİL ----------------------

def test_uc_tekrar_bagimsiz_gozlem_sayilmaz():
    task_ids = [f"t{i:02d}" for i in range(10)]
    records = _records(task_ids, _split({"t00", "t01", "t02", "t03"}))
    result = analyze(_manifest(task_ids), records, _calls(records), iterations=ITER)

    etki = result["paired_effects"]["plus_pass"][f"{ARM_CONTRACT}_vs_{ARM_NAIVE}"]
    # Analiz birimi görev: n=10, 10×3=30 değil.
    assert etki["n_tasks"] == 10
    assert result["analysis_unit"] == "task"
    # Gözlem sayısı AYRI alanda raporlanır; estimand'ın örneklem büyüklüğü değil.
    assert result["arm_summary"][ARM_NAIVE]["plus_pass"]["observation_count"] == 30

    # Kümelenmeyi yok saymak CI'yı daraltırdı: gözlem düzeyinde bootstrap
    # (30 birim) aynı veride belirgin biçimde dar bir aralık verir.
    gorev_genisligi = etki["ci_high"] - etki["ci_low"]
    gozlem_farklari = [d for t in task_ids for d in [
        1.0 if t in {"t00", "t01", "t02", "t03"} else 0.0] * 3]
    gozlem = sorted(bootstrap_draws(gozlem_farklari, iterations=ITER, seed=1))
    gozlem_genisligi = gozlem[int(0.975 * len(gozlem)) - 1] - gozlem[int(0.025 * len(gozlem))]
    assert gorev_genisligi > gozlem_genisligi, (
        "görev-düzeyi CI, gözlem-düzeyi CI'dan dar çıktı — kümelenme yok sayılmış")


# --- 2. Bootstrap örnekleme birimi görev -------------------------------------

def test_bootstrap_ornekleme_birimi_gorev():
    # İki görev, farkları 0.0 ve 1.0. Görev düzeyinde örneklenirse mümkün
    # ortalamalar YALNIZ {0, 0.5, 1}. Görev İÇİNDEKİ tekrarlar ayrı ayrı
    # örneklenseydi (6 gözlem) 1/6, 1/3, 2/3 gibi değerler de görülürdü.
    cekimler = set(bootstrap_draws([0.0, 1.0], iterations=500, seed=7))
    assert cekimler <= {0.0, 0.5, 1.0}
    assert cekimler == {0.0, 0.5, 1.0}, "örnekleme fiilen değişkenlik üretmiyor"


def test_bootstrap_her_cekimde_gorev_sayisi_kadar_secim_yapar():
    # Tek görevlik veride ortalama sabittir: n kadar seçim yapılıyorsa
    # dağılımın tamamı tek değerden oluşur.
    assert set(bootstrap_draws([0.42], iterations=50, seed=3)) == {0.42}


# --- 3. Aynı seed -> bayt-bayt aynı çıktı ------------------------------------

def test_ayni_seed_bayt_bayt_ayni_ozet(tmp_path):
    task_ids = [f"t{i:02d}" for i in range(6)]
    records = _records(task_ids, _split({"t00", "t01"}))
    calls = [make_synthetic_llm_call(run_id=r["run_id"], arm=r["arm"],
                                     task_id=r["task_id"], repeat=r["repeat"])
             for r in records]
    manifest = _manifest(task_ids)

    yollar = []
    for ad in ("kosu1", "kosu2"):
        sonuc = analyze(manifest, records, calls, iterations=ITER)
        yollar.append(write_outputs(sonuc, tmp_path / ad))

    for anahtar in yollar[0]:
        a = yollar[0][anahtar].read_bytes()
        b = yollar[1][anahtar].read_bytes()
        assert a == b, f"{anahtar} iki koşuda farklı — belirlenimcilik bozuk"
    # Zaman damgası içeriğe KARIŞMAMALI (görev üretiminde aynı hata yapılmıştı).
    ozet = json.loads(yollar[0]["analysis_summary"].read_text(encoding="utf-8"))
    assert not any("ts" == k or k.endswith("_ts") for k in ozet)


def test_farkli_seed_farkli_ci_verir():
    task_ids = [f"t{i:02d}" for i in range(8)]
    records = _records(task_ids, _split({"t00", "t03"}))
    manifest = _manifest(task_ids)
    a = analyze(manifest, records, _calls(records), iterations=ITER, seed=1)
    b = analyze(manifest, records, _calls(records), iterations=ITER, seed=2)
    anahtar = f"{ARM_CONTRACT}_vs_{ARM_NAIVE}"
    # Nokta tahmini seed'den BAĞIMSIZ; yalnız CI örneklemeye bağlı.
    assert (a["paired_effects"]["plus_pass"][anahtar]["point_estimate"]
            == b["paired_effects"]["plus_pass"][anahtar]["point_estimate"])
    # CI örneklemeye bağlıdır: aynı veride bile farklı seed farklı dağılım verir.
    # (Uç yüzdelikler tesadüfen eşit çıkabilir; karşılaştırma dağılım üzerinde.)
    assert (bootstrap_draws([1.0, 0.0] * 4, iterations=ITER, seed=1)
            != bootstrap_draws([1.0, 0.0] * 4, iterations=ITER, seed=2))


# --- 4. Bilinen sentetik etki -> doğru nokta tahmini -------------------------

def test_bilinen_etki_dogru_hesaplanir():
    # 10 görevin 4'ünde naive hiç geçmiyor, contract hep geçiyor -> +0.4.
    task_ids = [f"t{i:02d}" for i in range(10)]
    records = _records(task_ids, _split({"t00", "t01", "t02", "t03"}))
    result = analyze(_manifest(task_ids), records, _calls(records), iterations=ITER)
    etki = result["paired_effects"]["plus_pass"][f"{ARM_CONTRACT}_vs_{ARM_NAIVE}"]
    assert etki["point_estimate"] == pytest.approx(0.4)
    assert etki["tasks_favoring_a"] == 4
    assert etki["tasks_tied"] == 6
    assert etki["ci_low"] <= 0.4 <= etki["ci_high"]


def test_kismi_gecis_orani_gorev_duzeyinde_ortalanir():
    # Tek görev, contract 2/3, naive 1/3 -> fark 1/3 (gözlem sayımı değil).
    def passes(task_id, arm):
        return {ARM_CONTRACT: 2, ARM_NAIVE: 1}.get(arm, 3)
    rates = task_level_rates(_records(["t00"], passes), task_ids=["t00"],
                             arms=ARMS, metric="plus_pass")
    assert rates["t00"][ARM_CONTRACT]["rate"] == pytest.approx(2 / 3)
    assert paired_diffs(rates, ARM_CONTRACT, ARM_NAIVE, ["t00"]) == pytest.approx([1 / 3])


# --- 5. Bozuk veri analizi durdurur ------------------------------------------

def test_yinelenen_kayit_analizi_durdurur():
    task_ids = ["t00", "t01"]
    records = _records(task_ids, _split(set()))
    records.append(dict(records[0], run_id="ayri"))  # aynı anahtar, ikinci gerçek kayıt
    with pytest.raises(AnalysisError, match="yinelenen"):
        analyze(_manifest(task_ids), records, _calls(records), iterations=ITER)


def test_eksik_kosu_analizi_durdurur():
    task_ids = ["t00", "t01"]
    records = _records(task_ids, _split(set()))[:-1]
    with pytest.raises(AnalysisError, match="eksik"):
        analyze(_manifest(task_ids), records, _calls(records), iterations=ITER)


def test_eksik_kosu_yalniz_acik_bayrakla_ve_preliminary_damgasiyla_gecer():
    task_ids = ["t00", "t01"]
    records = _records(task_ids, _split(set()))[:-1]
    result = analyze(_manifest(task_ids), records, _calls(records), iterations=ITER,
                     allow_missing=True)
    assert result["preliminary"] is True


def test_allow_missing_yinelenen_kaydi_MAZUR_GORMEZ():
    # Bayrak yalnız eksikliği mazur görür; çift sayım her hâlükârda durdurur.
    task_ids = ["t00", "t01"]
    records = _records(task_ids, _split(set()))
    records.append(dict(records[0], run_id="ayri"))
    with pytest.raises(AnalysisError, match="yinelenen"):
        analyze(_manifest(task_ids), records, _calls(records), iterations=ITER, allow_missing=True)


def test_beklenmeyen_anahtar_analizi_durdurur():
    task_ids = ["t00", "t01"]
    records = _records(task_ids, _split(set()))
    records.append(make_synthetic_record(model=MODEL, arm=ARM_NAIVE,
                                         task_id="planlanmamis", repeat=0))
    with pytest.raises(AnalysisError, match="beklenmeyen"):
        analyze(_manifest(task_ids), records, _calls(records), iterations=ITER)


def test_gecersiz_kayit_analizi_durdurur():
    task_ids = ["t00", "t01"]
    records = _records(task_ids, _split(set()))
    records[0].pop("plus_pass")
    with pytest.raises(AnalysisError, match="şema ihlali"):
        analyze(_manifest(task_ids), records, _calls(records), iterations=ITER)


# --- 6. run_error + başarılı resume çift sayılmaz ----------------------------

def test_run_error_ve_basarili_resume_cift_sayilmaz():
    task_ids = ["t00"]
    records = _records(task_ids, _split(set()))
    # Aynı (model, kol, görev, tekrar) için önce bir run_error yazılmış, sonra
    # yeniden denenip başarılı olmuş: dosyada İKİ kayıt var.
    records.insert(0, make_run_error_record(
        experiment="sentetik", model=MODEL, task_set="heldout", arm=ARM_CONTRACT,
        task_id="t00", repeat=0, run_id="hatali", arm_position=0,
        error="ProviderResponseError: tükendi"))

    rates = task_level_rates(records, task_ids=task_ids, arms=ARMS, metric="plus_pass")
    hucre = rates["t00"][ARM_CONTRACT]
    assert hucre["repeat_count"] == 3, "run_error paydaya girmiş (oran yapay düşer)"
    assert hucre["rate"] == 1.0
    # Analiz de sorunsuz tamamlanmalı: bu NORMAL bir resume akışıdır.
    result = analyze(_manifest(task_ids), records, _calls(records), iterations=ITER)
    assert result["integrity"]["run_error_count"] == 1


# --- 7. Base ve Plus karışmaz ------------------------------------------------

def test_base_ve_plus_sonuclari_karismaz():
    task_ids = [f"t{i:02d}" for i in range(4)]
    # contract: base'i geçer, plus'ı geçemez. naive: ikisini de geçemez.
    def plus(task_id, arm):
        return 0 if arm in (ARM_CONTRACT, ARM_NAIVE) else 3

    def base(task_id, arm):
        return 0 if arm == ARM_NAIVE else 3

    records = _records(task_ids, plus, base_passes=base)
    result = analyze(_manifest(task_ids), records, _calls(records), iterations=ITER)
    anahtar = f"{ARM_CONTRACT}_vs_{ARM_NAIVE}"
    assert result["paired_effects"]["base_pass"][anahtar]["point_estimate"] == pytest.approx(1.0)
    assert result["paired_effects"]["plus_pass"][anahtar]["point_estimate"] == pytest.approx(0.0)


def test_bilinmeyen_metrik_reddedilir():
    with pytest.raises(AnalysisError, match="bilinmeyen metrik"):
        task_level_rates([], task_ids=["t00"], arms=ARMS, metric="pass")


# --- 8. Modeller havuzlanmaz -------------------------------------------------

def test_farkli_modeller_havuzlanmaz():
    task_ids = ["t00", "t01"]
    records = (_records(task_ids, _split(set()), model="model/a")
               + _records(task_ids, _split(set()), model="model/b"))
    with pytest.raises(AnalysisError, match="HAVUZLANMAZ"):
        analyze(_manifest(task_ids, model="model/a"), records, _calls(records), iterations=ITER)


# --- 9. Kullanım verisi run_id üzerinden birleşir ----------------------------

def test_kullanim_run_id_uzerinden_birlesir():
    task_ids = ["t00"]
    records = _records(task_ids, _split(set()))
    calls = []
    for r in records:
        # contract kolunda planner + coder iki çağrı; diğerlerinde tek çağrı.
        roller = ("planner", "coder") if r["arm"] == ARM_CONTRACT else ("coder",)
        for rol in roller:
            calls.append(make_synthetic_llm_call(
                run_id=r["run_id"], arm=r["arm"], task_id=r["task_id"],
                repeat=r["repeat"], agent_role=rol, cost_usd=0.002,
                input_tokens=10, output_tokens=20, latency_s=1.0))

    satirlar = usage_by_run(calls, records)
    assert len(satirlar) == len(records)
    contract_satirlari = [s for s in satirlar if s["arm"] == ARM_CONTRACT]
    assert all(s["successful_calls"] == 2 for s in contract_satirlari)
    assert all(s["cost_usd"] == pytest.approx(0.004) for s in contract_satirlari)
    assert all(s["in_results"] and not s["is_run_error"] for s in satirlar)

    result = analyze(_manifest(task_ids), records, calls, iterations=ITER)
    assert result["usage"]["by_arm"][ARM_CONTRACT]["successful_calls"] == 6  # 3 tekrar × 2
    assert result["usage"]["by_arm"][ARM_BASELINE]["successful_calls"] == 3


def test_provider_error_gecikmesi_cift_sayilmaz():
    # Başarılı çağrının latency_s'i retry + backoff süresini ZATEN içerir
    # (agents/llm.py sayacı ilk denemeden önce başlar). Bozuk denemelerin
    # gecikmesi üstüne eklenirse aynı süre iki kez sayılır.
    calls = [
        make_synthetic_llm_call(run_id="r1", status="provider_error", latency_s=99.0),
        make_synthetic_llm_call(run_id="r1", status="ok", provider_attempt=2,
                                latency_s=5.0, cost_usd=0.01),
    ]
    kayit = make_synthetic_record(model=MODEL, arm=ARM_BASELINE, task_id="t00",
                                  repeat=0, run_id="r1")
    satir = usage_by_run(calls, [kayit])[0]
    assert satir["latency_s"] == 5.0
    assert satir["provider_error_attempts"] == 1
    assert satir["successful_calls"] == 1
    assert satir["cost_usd"] == pytest.approx(0.01)


def test_run_error_maliyeti_basari_estimandina_katilmaz():
    # Ölçüm üretmemiş koşunun maliyeti AYRI "altyapı overhead" tablosunda.
    task_ids = ["t00"]
    records = _records(task_ids, _split(set()))
    hatali = make_run_error_record(
        experiment="sentetik", model=MODEL, task_set="heldout", arm=ARM_CONTRACT,
        task_id="t00", repeat=0, run_id="hatali", arm_position=0, error="hata")
    records.insert(0, hatali)
    calls = [make_synthetic_llm_call(run_id=hatali["run_id"], arm=ARM_CONTRACT,
                                     task_id="t00", cost_usd=0.5)]
    calls += [make_synthetic_llm_call(run_id=r["run_id"], arm=r["arm"],
                                      task_id=r["task_id"], repeat=r["repeat"],
                                      cost_usd=0.001)
              for r in records if r is not hatali]

    result = analyze(_manifest(task_ids), records, calls, iterations=ITER)
    assert result["usage"]["infrastructure_overhead"]["cost_usd"] == pytest.approx(0.5)
    assert result["usage"]["scored_runs"]["cost_usd"] == pytest.approx(0.012)
    assert result["usage"]["infrastructure_overhead"]["runs"] == 1


def test_sonuc_kaydiyla_eslesmeyen_cagri_overhead_sayilir():
    # Yetim çağrı (ör. yarıda kesilen koşu) sessizce kaybolmaz ama ölçüm
    # maliyetine de karışmaz.
    task_ids = ["t00"]
    records = _records(task_ids, _split(set()))
    calls = _calls(records, cost_usd=0.001)
    calls.append(make_synthetic_llm_call(run_id="yetim", cost_usd=0.3))
    result = analyze(_manifest(task_ids), records, calls, iterations=ITER)
    assert result["usage"]["infrastructure_overhead"]["runs"] == 1
    assert result["usage"]["infrastructure_overhead"]["cost_usd"] == pytest.approx(0.3)
    assert result["usage"]["scored_runs"]["cost_usd"] == pytest.approx(0.012)


# --- 10. Manifest–sonuç provenance kapısı ------------------------------------
# Tek model bulunması, DOĞRU model bulunduğu anlamına gelmiyor: kendi içinde
# tutarlı ama yanlış manifestin altında duran bir kayıt kümesi, yalnız
# "tek model var mı" diye bakan bir kontrolden geçerdi.

def test_manifest_modeliyle_uyusmayan_kayit_analizi_durdurur():
    task_ids = ["t00", "t01"]
    records = _records(task_ids, _split(set()), model="model/B")
    with pytest.raises(AnalysisError, match="manifest–sonuç uyuşmazlığı"):
        analyze(_manifest(task_ids, model="model/A"), records, _calls(records),
                iterations=ITER)


def test_manifest_deney_adiyla_uyusmayan_kayit_analizi_durdurur():
    task_ids = ["t00", "t01"]
    records = _records(task_ids, _split(set()))
    for r in records:
        r["experiment"] = "baska_deney"
    with pytest.raises(AnalysisError, match="experiment"):
        analyze(_manifest(task_ids, name="sentetik"), records, _calls(records),
                iterations=ITER)


def test_manifest_gorev_setiyle_uyusmayan_kayit_analizi_durdurur():
    # Pilot kayıtların held-out manifestin altında raporlanması, ana istatistiği
    # development verisiyle kirletirdi (§5.1).
    task_ids = ["t00", "t01"]
    records = _records(task_ids, _split(set()))
    for r in records:
        r["task_set"] = "pilot"
    with pytest.raises(AnalysisError, match="task_set"):
        analyze(_manifest(task_ids), records, _calls(records), iterations=ITER)


def test_manifest_sema_surumuyle_uyusmayan_kayit_analizi_durdurur():
    task_ids = ["t00", "t01"]
    records = _records(task_ids, _split(set()))
    manifest = _manifest(task_ids)
    manifest["result_schema_version"] = "1.0"
    with pytest.raises(AnalysisError, match="schema_version"):
        analyze(manifest, records, _calls(records), iterations=ITER)


@pytest.mark.parametrize("alan", ["task_set", "llm_call_schema_version"])
def test_manifestte_zorunlu_alan_eksikse_analiz_durur(alan):
    task_ids = ["t00", "t01"]
    records = _records(task_ids, _split(set()))
    manifest = _manifest(task_ids)
    del manifest[alan]
    with pytest.raises(AnalysisError, match="zorunlu alan"):
        analyze(manifest, records, _calls(records), iterations=ITER)


def test_beklenen_anahtarlar_manifest_modelinden_uretilir():
    # Kayıtlardan çıkarılan modelle üretilseydi, yanlış modelli tutarlı bir küme
    # kendi kendini doğrulardı. Model uyuşmazlığı bütünlük katmanında da görünür.
    from analysis.analyze import check_integrity
    task_ids = ["t00"]
    records = _records(task_ids, _split(set()), model="model/B")
    rapor = check_integrity(records, _manifest(task_ids, model="model/A"))
    assert rapor["missing"], "manifest modelinin koşuları eksik görünmeli"
    assert rapor["unexpected"], "yanlış modelin kayıtları beklenmeyen sayılmalı"


# --- 11. Çağrı logu doğrulaması ve kullanım completeness ---------------------

def test_eksik_kullanim_logu_sifir_maliyet_gibi_gorunmez():
    task_ids = [f"t{i:02d}" for i in range(3)]
    records = _records(task_ids, _split(set()))
    with pytest.raises(AnalysisError, match="SIFIR MALİYET"):
        analyze(_manifest(task_ids), records, [], iterations=ITER)


def test_allow_missing_usage_ile_maliyet_null_olur_sifir_degil():
    task_ids = [f"t{i:02d}" for i in range(3)]
    records = _records(task_ids, _split(set()))
    result = analyze(_manifest(task_ids), records, [], iterations=ITER,
                     allow_missing_usage=True)
    assert result["preliminary"] is True
    assert result["usage"]["complete"] is False
    assert result["usage"]["scored_runs"]["cost_usd"] is None, "eksik veri sıfır yazılmış"
    assert result["usage"]["scored_runs"]["input_tokens"] is None
    # Performans (görev başarısı) analizi yine de üretilir.
    assert result["paired_effects"]["plus_pass"]


def test_tek_kosunun_kullanim_verisi_eksikse_bile_analiz_durur():
    task_ids = [f"t{i:02d}" for i in range(3)]
    records = _records(task_ids, _split(set()))
    calls = _calls(records)[:-1]
    with pytest.raises(AnalysisError, match="tamamlanmış koşunun başarılı LLM"):
        analyze(_manifest(task_ids), records, calls, iterations=ITER)


def test_yalniz_basarisiz_cagrisi_olan_kosu_kullanim_verisi_saymaz():
    # provider_error kaydı VAR ama status="ok" yok: koşu tamamlanmışsa bu
    # tutarsızdır — kullanım verisi ölçülmüş sayılmaz.
    task_ids = ["t00"]
    records = _records(task_ids, _split(set()))
    calls = _calls(records)
    calls[0] = make_synthetic_llm_call(run_id=calls[0]["run_id"], status="provider_error")
    with pytest.raises(AnalysisError, match="başarılı LLM"):
        analyze(_manifest(task_ids), records, calls, iterations=ITER)


def test_run_id_olmayan_cagri_sessizce_atilmaz():
    task_ids = ["t00"]
    records = _records(task_ids, _split(set()))
    calls = _calls(records)
    calls.append(make_synthetic_llm_call(run_id=None, cost_usd=0.9))
    with pytest.raises(AnalysisError, match="run_id yok"):
        analyze(_manifest(task_ids), records, calls, iterations=ITER)


def test_bilinmeyen_cagri_statusu_basarili_sayilmaz():
    task_ids = ["t00"]
    records = _records(task_ids, _split(set()))
    calls = _calls(records)
    calls.append(make_synthetic_llm_call(run_id=calls[0]["run_id"], status="tamamlandi"))
    with pytest.raises(AnalysisError, match="bilinmeyen çağrı status"):
        analyze(_manifest(task_ids), records, calls, iterations=ITER)


def test_eski_semali_cagri_logu_reddedilir():
    task_ids = ["t00"]
    records = _records(task_ids, _split(set()))
    calls = _calls(records)
    calls[0]["schema_version"] = "1.0"
    with pytest.raises(AnalysisError, match="çağrı logu şema sürümü"):
        analyze(_manifest(task_ids), records, calls, iterations=ITER)


def test_cagri_kimligi_sonuc_kaydiyla_celisirse_analiz_durur():
    # Yanlış koşuya atfedilen maliyet: kol karşılaştırması sessizce kayar.
    task_ids = ["t00"]
    records = _records(task_ids, _split(set()))
    calls = _calls(records)
    calls[0]["arm"] = ARM_NAIVE if calls[0]["arm"] != ARM_NAIVE else ARM_CONTRACT
    with pytest.raises(AnalysisError, match="çelişiyor"):
        analyze(_manifest(task_ids), records, calls, iterations=ITER)


def test_yalniz_hata_kaydi_olan_tamamlanmis_kosu_null_kullanim_tasir():
    # --allow-missing-usage altında: call_records>0 olduğu için "veri var"
    # sanılıp maliyet 0 raporlanırdı. Tamamlanmış koşunun ölçütü başarılı
    # çağrıdır (validate_calls ile aynı ölçüt), kayıt VARLIĞI değil.
    task_ids = ["t00"]
    records = _records(task_ids, _split(set()))
    calls = [make_synthetic_llm_call(run_id=r["run_id"], model=r["model"], arm=r["arm"],
                                     task_id=r["task_id"], repeat=r["repeat"],
                                     status="error", latency_s=7.0)
             for r in records]
    satirlar = usage_by_run(calls, records)
    assert all(s["call_records"] == 1 for s in satirlar)
    assert all(s["usage_missing"] is True for s in satirlar), "hata kaydı 'veri var' sayıldı"
    assert all(s["cost_usd"] is None for s in satirlar)

    result = analyze(_manifest(task_ids), records, calls, iterations=ITER,
                     allow_missing_usage=True)
    assert result["usage"]["complete"] is False
    assert result["usage"]["scored_runs"]["cost_usd"] is None
    # Boşa geçen süre yine de raporlanır — ama alt sınır olarak etiketli.
    assert result["usage"]["scored_runs"]["error_latency_s"] == pytest.approx(7.0 * 12)
    assert result["usage"]["scored_runs"]["error_latency_is_lower_bound"] is True


def test_run_error_kosusunda_hata_kaydi_veri_var_sayilir():
    # run_error'da başarılı çağrı bulunmaması NORMAL; orada ölçüt "hiç kayıt yok".
    hatali = make_run_error_record(
        experiment="sentetik", model=MODEL, task_set="heldout", arm=ARM_CONTRACT,
        task_id="t00", repeat=0, run_id="hatali", arm_position=0, error="x")
    calls = [make_synthetic_llm_call(run_id="hatali", status="error", latency_s=4.0)]
    satir = usage_by_run(calls, [hatali])[0]
    assert satir["usage_missing"] is False
    assert satir["cost_usd"] == 0
    assert satir["error_latency_s"] == pytest.approx(4.0)


def test_kullanim_verisi_eksik_kosu_tabloda_gorunur():
    # Satır çağrı logundan değil SONUÇ kaydından tohumlanır: hiç çağrısı olmayan
    # koşu tablodan düşmez, usage_missing=True olarak görünür.
    kayit = make_synthetic_record(model=MODEL, arm=ARM_BASELINE, task_id="t00", repeat=0)
    satir = usage_by_run([], [kayit])[0]
    assert satir["usage_missing"] is True
    assert satir["cost_usd"] is None and satir["input_tokens"] is None
    assert satir["call_records"] == 0


# --- 12. RQ3 uyum metrikleri -------------------------------------------------

def test_contract_uyum_metrikleri_uretilir():
    task_ids = ["t00", "t01"]
    records = _records(task_ids, _split(set()))
    contract = [r for r in records if r["arm"] == ARM_CONTRACT]
    # 6 contract koşusu: 3'ü ilk denemede geçerli, 2'si retry ile kurtarıldı,
    # 1'i denemeleri tüketip geçersiz kaldı.
    contract[3].update(attempt_count=2, handoff_validation={"valid": True})
    contract[4].update(attempt_count=3, handoff_validation={"valid": True})
    contract[5].update(attempt_count=3, handoff_validation={"valid": False, "errors": "x"})

    c = analyze(_manifest(task_ids), records, _calls(records),
                iterations=ITER)["compliance"][ARM_CONTRACT]
    assert c["unit"] == "arm_run" and c["runs"] == 6
    assert c["first_attempt_valid_count"] == 3
    assert c["first_attempt_compliance"] == pytest.approx(0.5)
    assert c["final_valid_count"] == 5
    assert c["final_compliance"] == pytest.approx(5 / 6)
    assert c["mean_attempt_count"] == pytest.approx((1 + 1 + 1 + 2 + 3 + 3) / 6)
    # Kurtarma oranının paydası TÜM koşular değil, retry'a fiilen girenler.
    assert c["retried_count"] == 3
    assert c["retry_recovered_count"] == 2
    assert c["retry_recovery_rate"] == pytest.approx(2 / 3)
    assert c["validation_exhausted_count"] == 1
    assert c["validation_exhaustion_rate"] == pytest.approx(1 / 6)
    assert c["attempt_count_distribution"] == {"1": 3, "2": 1, "3": 2}


def test_structured_kolunda_parse_orani_ayri_raporlanir():
    task_ids = ["t00", "t01"]
    records = _records(task_ids, _split(set()))
    yapisal = [r for r in records if r["arm"] == ARM_STRUCTURED]
    yapisal[0]["handoff_parse_ok"] = False
    uyum = analyze(_manifest(task_ids), records, _calls(records),
                   iterations=ITER)["compliance"]
    assert uyum[ARM_STRUCTURED]["parse_ok_rate"] == pytest.approx(5 / 6)
    # structured'da doğrulama kapısı yok: uyum alanları da olmamalı.
    assert "final_compliance" not in uyum[ARM_STRUCTURED]
    assert ARM_NAIVE not in uyum and ARM_BASELINE not in uyum


# --- 13. Uyum alanlarının TİPİ ve ARALIĞI ------------------------------------

@pytest.mark.parametrize("arm,alan,deger,desen", [
    (ARM_CONTRACT, "handoff_parse_ok", "evet", "bool değil"),
    (ARM_CONTRACT, "attempt_count", None, "int değil"),
    (ARM_CONTRACT, "attempt_count", True, "int değil"),      # bool, int'in alt sınıfı
    (ARM_CONTRACT, "attempt_count", 0, "izinli aralık"),
    (ARM_CONTRACT, "attempt_count", 4, "izinli aralık"),     # MAX_PLANNER_ATTEMPTS=3
    (ARM_CONTRACT, "handoff_validation", "gecerli", "dict değil"),
    (ARM_CONTRACT, "handoff_validation", {"valid": "yes"}, "valid bool değil"),
    (ARM_STRUCTURED, "attempt_count", 2, "retry kenarı yok"),
])
def test_uyum_alanlarinin_tipi_ve_araligi_dogrulanir(arm, alan, deger, desen):
    from eval.result_schema import validate_record
    kayit = make_synthetic_record(model=MODEL, arm=arm, task_id="t00", repeat=0)
    kayit[alan] = deger
    assert any(desen in p for p in validate_record(kayit)), validate_record(kayit)


def test_contract_erken_cikis_kaydi_reddedilir():
    # Graf, doğrulama başarısızken attempt < MAX ise planner'a DÖNER. Yani
    # valid=False + attempt_count=1 pipeline'ın üretemeyeceği bir durumdur;
    # geçerse retry kapısının fiilen çalışmadığı gizlenirdi (RQ3 geçersiz olur).
    from eval.result_schema import validate_record
    kayit = make_synthetic_record(model=MODEL, arm=ARM_CONTRACT, task_id="t00", repeat=0)
    kayit.update(attempt_count=1, handoff_validation={"valid": False, "errors": "x"})
    assert any("erken çıkış" in p for p in validate_record(kayit))
    # Denemeler tükendiyse aynı kayıt GEÇERLİDİR.
    kayit["attempt_count"] = MAX_PLANNER_ATTEMPTS
    assert validate_record(kayit) == []


def test_gecerli_dogrulama_ayristirilmamis_planla_birlikte_olamaz():
    # Validator ancak ayrıştırılmış bir nesneyi doğrulayabilir; valid=True +
    # parse_ok=False, iki alanın farklı denemelerden kaldığını gösterir.
    from eval.result_schema import validate_record
    kayit = make_synthetic_record(model=MODEL, arm=ARM_CONTRACT, task_id="t00", repeat=0)
    kayit.update(handoff_parse_ok=False, handoff_validation={"valid": True, "errors": None})
    assert any("tutarsız durum" in p for p in validate_record(kayit))


def test_bozuk_uyum_alani_analizi_durdurur():
    task_ids = ["t00", "t01"]
    records = _records(task_ids, _split(set()))
    next(r for r in records if r["arm"] == ARM_CONTRACT)["attempt_count"] = 9
    with pytest.raises(AnalysisError, match="şema ihlali"):
        analyze(_manifest(task_ids), records, _calls(records), iterations=ITER)


# --- 14. Kullanım için normalize betimleyiciler ------------------------------

def test_kosu_basina_ortalama_medyan_ve_iqr_raporlanir():
    # Maliyet/gecikme sağa çarpık olabilir; yalnız toplam dağılımı gizler.
    task_ids = ["t00"]
    records = _records(task_ids, _split(set()))
    baseline = [r for r in records if r["arm"] == ARM_BASELINE]
    maliyetler = [0.001, 0.002, 0.030]  # bir uç değer
    calls = _calls(records, cost_usd=0.001)
    for r, maliyet in zip(baseline, maliyetler):
        for c in calls:
            if c["run_id"] == r["run_id"]:
                c["cost_usd"] = maliyet

    blok = analyze(_manifest(task_ids), records, calls,
                   iterations=ITER)["usage"]["by_arm"][ARM_BASELINE]
    assert blok["cost_usd"] == pytest.approx(0.033)
    assert blok["cost_usd_per_run"]["median"] == pytest.approx(0.002)
    assert blok["cost_usd_per_run"]["mean"] == pytest.approx(0.011)
    # Medyan ortalamadan belirgin küçük: çarpıklık toplamda görünmezdi.
    assert blok["cost_usd_per_run"]["median"] < blok["cost_usd_per_run"]["mean"]
    assert blok["cost_usd_per_run"]["iqr"] > 0
    assert set(blok["latency_s_per_run"]) == {"n", "mean", "median", "q1", "q3",
                                              "iqr", "p95", "observed_max"}
    assert "total_tokens_per_run" in blok


# --- 14.1 Uzun kuyruk: n / p95 / gözlenen maksimum ---------------------------

def _latency_bloklari(latencies: list[float]):
    """Verilen gecikmelerle tek kollu bir analiz koşusu; (genel, by_arm) döner.

    Her koşunun TEK çağrısı olduğu için arm-run gecikmesi = çağrı gecikmesidir.
    `latencies` uzunluğu tekrar sayısının katı olmalı ki baseline koşularının
    TAMAMI atansın (artık koşu kalırsa n beklenenden büyük çıkar).
    """
    assert len(latencies) % 3 == 0, "fixture 3 tekrar üretir; uzunluk 3'ün katı olmalı"
    task_ids = [f"t{i:02d}" for i in range(len(latencies) // 3)]
    records = _records(task_ids, _split(set()))
    baseline = [r for r in records if r["arm"] == ARM_BASELINE]
    assert len(baseline) == len(latencies)
    calls = _calls(records)
    atama = {r["run_id"]: g for r, g in zip(baseline, latencies)}
    for c in calls:
        if c["run_id"] in atama:
            c["latency_s"] = atama[c["run_id"]]
    kullanim = analyze(_manifest(task_ids), records, calls, iterations=ITER)["usage"]
    return kullanim["scored_runs"], kullanim["by_arm"][ARM_BASELINE]


def test_uzun_kuyruk_metrikleri_bilinen_carpik_dagilimda_dogru():
    # 1..12 -> lineer interpolasyon (R-7): P50=6.5, P95=11.45, maks=12.
    # Sağlık koşusundaki imzanın minyatürü: IQR gövdeyi anlatır, kuyruğu değil.
    _, kol = _latency_bloklari([float(i) for i in range(1, 13)])
    g = kol["latency_s_per_run"]
    assert g["n"] == 12
    assert g["median"] == pytest.approx(6.5)
    assert g["p95"] == pytest.approx(11.45)
    assert g["observed_max"] == pytest.approx(12.0)
    assert g["observed_max"] > g["q3"]   # kuyruk IQR'ın dışında


def test_uzun_kuyruk_hem_genel_hem_by_arm_blogunda_var():
    genel, kol = _latency_bloklari([1.0, 1.0, 1.0, 1.0, 1.0, 60.0])
    for blok in (genel, kol):
        for alan in ("n", "p95", "observed_max"):
            assert alan in blok["latency_s_per_run"], alan
            assert alan in blok["cost_usd_per_run"], alan
            assert alan in blok["total_tokens_per_run"], alan
    assert kol["latency_s_per_run"]["observed_max"] == pytest.approx(60.0)
    # Tek uç değer ortalamayı çeker ama medyan gövdede kalır.
    assert kol["latency_s_per_run"]["median"] < kol["latency_s_per_run"]["mean"]


def test_median_p50_ile_ayni_olcudur():
    # `median` alan adı geriye uyumluluk için korunuyor; ikinci bir p50 kopyası
    # YAZILMAZ (iki alan zamanla ayrışabilirdi).
    _, kol = _latency_bloklari([2.0, 4.0, 6.0, 8.0, 10.0, 12.0])
    g = kol["latency_s_per_run"]
    assert "p50" not in g
    assert g["median"] == pytest.approx(7.0)


def test_kullanim_eksikse_uzun_kuyruk_alanlari_null():
    # Eksik kullanım verisi "sıfır gecikme" gibi okunamaz: blok tamamen None.
    task_ids = ["t00"]
    records = _records(task_ids, _split(set()))
    calls = [c for c in _calls(records)
             if c["arm"] != ARM_BASELINE]     # baseline koşularının çağrıları YOK
    kullanim = analyze(_manifest(task_ids), records, calls, iterations=ITER,
                       allow_missing_usage=True)["usage"]
    assert kullanim["by_arm"][ARM_BASELINE]["latency_s_per_run"] is None
    assert kullanim["complete"] is False


def test_bos_alt_kumede_uzun_kuyruk_alanlari_null_ve_n_sifir():
    from analysis.analyze import _describe
    o = _describe([], 2)
    assert o["n"] == 0
    assert o["p95"] is None and o["observed_max"] is None
    assert o["median"] is None and o["iqr"] is None


def test_retry_gecikmesi_uzun_kuyrukta_cift_sayilmaz():
    # provider_error denemesinin süresi, sonunda başarılı olan çağrının
    # latency_s'ine ZATEN dahildir (agents/llm.py sayacı ilk denemeden önce
    # başlar). Ayrıca eklenirse p95/maks yapay olarak şişer.
    task_ids = ["t00"]
    records = _records(task_ids, _split(set()))
    hedef = next(r for r in records if r["arm"] == ARM_BASELINE)
    calls = _calls(records)
    for c in calls:
        if c["run_id"] == hedef["run_id"]:
            c["latency_s"] = 30.0            # retry süresi buna dahil
            c["provider_attempt"] = 2
    calls.append(make_synthetic_llm_call(
        run_id=hedef["run_id"], model=hedef["model"], arm=hedef["arm"],
        task_id=hedef["task_id"], repeat=hedef["repeat"],
        status="provider_error", provider_attempt=1, latency_s=25.0))

    sonuc = analyze(_manifest(task_ids), records, calls, iterations=ITER)
    g = sonuc["usage"]["by_arm"][ARM_BASELINE]["latency_s_per_run"]
    assert g["observed_max"] == pytest.approx(30.0), "25 s'lik bozuk deneme eklenmemeli"
    assert sonuc["usage"]["by_arm"][ARM_BASELINE]["provider_error_attempts"] == 1
    satir = next(r for r in sonuc["_usage_rows"] if r["run_id"] == hedef["run_id"])
    assert satir["latency_s"] == pytest.approx(30.0)
    assert satir["error_latency_s"] == 0.0   # provider_error terminal hata DEĞİL


# --- Çıktı dosyaları ---------------------------------------------------------

def test_cikti_dosyalari_uretilir_ve_gorev_duzeyi_satirlari_tamdir(tmp_path):
    task_ids = [f"t{i:02d}" for i in range(5)]
    records = _records(task_ids, _split({"t00"}))
    result = analyze(_manifest(task_ids), records, _calls(records), iterations=ITER)
    yollar = write_outputs(result, tmp_path / "analysis")

    assert set(yollar) == {"task_level", "paired_effects", "usage_by_run",
                           "integrity_report", "analysis_summary"}
    satirlar = yollar["task_level"].read_text(encoding="utf-8").strip().splitlines()
    assert len(satirlar) == 1 + len(task_ids) * len(ARMS)     # başlık + hücreler
    assert "plus_rate" in satirlar[0] and "base_rate" in satirlar[0]

    etkiler = json.loads(yollar["paired_effects"].read_text(encoding="utf-8"))
    assert set(etkiler) == {"plus_pass", "base_pass"}
    # Ön-kayıt: p-değeri/hipotez testi YOK.
    metin = yollar["paired_effects"].read_text(encoding="utf-8")
    assert "p_value" not in metin and "p_val" not in metin


# --- Base -> Plus elenmesi (RQ5 üçüncü sorusu, §2) ---------------------------

def test_base_plus_elenmesi_paydasi_base_gecen_kosulardir():
    """Payda BÜTÜN koşular olsaydı, base'te zaten düşenler elenmeyi seyreltirdi."""
    task_ids = [f"t{i:02d}" for i in range(4)]
    # baseline: base 3/3 geçer ama plus yalnız 1/3 geçer -> 2/3 elenme.
    # naive:    base 1/3 geçer, plus 1/3 -> 0 elenme (payda 1, pay 0).
    def passes(task_id, arm):
        return 1 if arm in (ARM_BASELINE, ARM_NAIVE) else 3

    def base_passes(task_id, arm):
        return 3 if arm != ARM_NAIVE else 1

    records = _records(task_ids, passes, base_passes=base_passes)
    sonuc = analyze(_manifest(task_ids), records, _calls(records), iterations=ITER)
    a = sonuc["base_plus_attrition"]

    b = a["by_arm"][ARM_BASELINE]
    assert b["unit"] == "arm_run"
    assert (b["runs"], b["base_pass_runs"], b["plus_pass_runs"]) == (12, 12, 4)
    assert b["base_pass_plus_fail_runs"] == 8
    assert b["attrition_rate"] == pytest.approx(8 / 12)

    n = a["by_arm"][ARM_NAIVE]
    assert (n["base_pass_runs"], n["plus_pass_runs"]) == (4, 4)
    assert n["attrition_rate"] == 0.0

    # Elenmesi olmayan kollar toplamı seyreltmemeli: toplam da yalnız base
    # geçenler üzerinden hesaplanır.
    assert a["overall"]["base_pass_runs"] == 12 + 4 + 12 + 12
    assert a["overall"]["base_pass_plus_fail_runs"] == 8


def test_base_hic_gecmeyen_kolda_elenme_orani_none():
    """Sıfır payda 'elenme yok (0)' diye raporlanamaz — ölçüm yapılamamıştır."""
    task_ids = [f"t{i:02d}" for i in range(3)]
    records = _records(task_ids, lambda t, a: 0 if a == ARM_BASELINE else 3)
    sonuc = analyze(_manifest(task_ids), records, _calls(records), iterations=ITER)
    b = sonuc["base_plus_attrition"]["by_arm"][ARM_BASELINE]
    assert b["base_pass_runs"] == 0 and b["attrition_rate"] is None


def test_run_error_elenme_paydasina_girmez():
    task_ids = [f"t{i:02d}" for i in range(3)]
    records = _records(task_ids, lambda t, a: 3)
    temiz = analyze(_manifest(task_ids), records, _calls(records), iterations=ITER)
    hatali = make_run_error_record(
        experiment="sentetik", model=MODEL, task_set="heldout", arm=ARM_BASELINE,
        task_id="t00", repeat=0, run_id="err-1", arm_position=0, error="boom")
    kirli = analyze(_manifest(task_ids), [*records, hatali], _calls(records),
                    iterations=ITER)
    assert temiz["base_plus_attrition"] == kirli["base_plus_attrition"]
