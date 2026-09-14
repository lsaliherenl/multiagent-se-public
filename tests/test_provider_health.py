"""scripts/provider_health.py birim testleri — LLM'siz, deterministik.

Sağlık kapısı ana model kararını verecek; yanlış "geçti" demesi, ölçülmemiş bir
taşıma arızasıyla 3.000 çağrılık ana koşuya başlamak demektir. Bu yüzden karar
kuralı ve program üretimi ayrı ayrı sınanır.

Uçtan uca testler `litellm.completion` SEVİYESİNDE mock'lar (call_model
seviyesinde DEĞİL): dizin yerleşimi, provenance ve loglama katmanı fiilen
çalışsın diye. Hiçbir test gerçek ağ çağrısı yapmaz.
"""

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

import agents.llm as llm_module
import config
from config import (
    HEALTH_GATE_BLOCK_SIZE,
    HEALTH_GATE_PASS_RATE,
    HEALTH_GATE_WARN_RATE,
)

# scripts/ bir paket değil; dosyadan yükle.
_spec = importlib.util.spec_from_file_location(
    "provider_health", Path(__file__).parent.parent / "scripts" / "provider_health.py")
provider_health = importlib.util.module_from_spec(_spec)
sys.modules["provider_health"] = provider_health
_spec.loader.exec_module(provider_health)

CONFIGS = ["main", "secondary", "judge_external"]
TASKS = [{"task_id": f"t{i}", "prompt": "p"} for i in range(4)]


# --- Rol konfigürasyonu ------------------------------------------------------

def test_saglik_konfigurasyonu_uc_rol():
    # Script kendi listesini tutmaz; tek kaynak config.HEALTH_GATE_MODELS.
    assert list(provider_health.CONFIGS) == CONFIGS
    assert provider_health.CONFIGS == dict(config.HEALTH_GATE_MODELS)


def test_judge_external_gercekten_minimaxe_cozulur():
    assert provider_health.CONFIGS["judge_external"] == config.MODEL_JUDGE_EXTERNAL
    assert "minimax" in provider_health.CONFIGS["judge_external"]


def test_main_ve_secondary_uretici_modellere_cozulur():
    assert provider_health.CONFIGS["main"] == config.MODEL_MAIN
    assert provider_health.CONFIGS["secondary"] == config.MODEL_SECONDARY


def test_upgrade_ve_dev_saglik_konfigurasyonunda_yok():
    # `upgrade` kadrodan çıktı; `dev` ile `secondary` aynı DeepSeek slug'ına
    # çözülüyor -> ayrı bir sağlık anahtarı olarak durması eski mimariyi
    # yeniden üretirdi.
    assert "upgrade" not in provider_health.CONFIGS
    assert "dev" not in provider_health.CONFIGS
    assert "pilot" not in provider_health.CONFIGS


def test_cli_yalnizca_formal_saglik_anahtarlarini_kabul_eder():
    # Serbest slug veya takma ad bu kümeye sessizce giremez: girseydi rapor,
    # deneyde kullanılmayan bir rotayı "kapı geçti" diye kaydedebilirdi.
    kaynak = (Path(__file__).parent.parent / "scripts" / "provider_health.py").read_text(
        encoding="utf-8")
    assert "choices=list(CONFIGS)" in kaynak


def _kayit(config="main", status="ok", provider_attempt=1, prompt_kind="json_planner",
           **extra):
    return {"config": config, "model": "m", "prompt_kind": prompt_kind,
            "status": status, "provider_attempt": provider_attempt,
            "provider": "P", "latency_s": 1.0, "cost_usd": 0.001, **extra}


# --- Program üretimi ---------------------------------------------------------

def test_her_konfigurasyon_tam_sayida_cagri_alir():
    program = provider_health.build_schedule(CONFIGS, 20, TASKS, block_size=10)
    for config in CONFIGS:
        assert sum(c == config for c, _, _ in program) == 20


def test_blok_icinde_model_ardisik_kosar():
    # Blok içi ardışıklık ana koşunun temposunu taklit eder; tek tek dönüşümlü
    # çağrıda throttle modeller arasında bölünür ve rate-limit arızası
    # OLDUĞUNDAN DÜŞÜK ölçülür.
    program = provider_health.build_schedule(CONFIGS, 10, TASKS, block_size=5)
    ilk_blok = [c for c, _, _ in program[:5]]
    assert len(set(ilk_blok)) == 1, "blok içinde tek model ardışık koşmalı"


def test_blok_sirasi_bloklar_arasinda_dondurulur():
    # Zamana bağlı bir kesintinin hep aynı modele yığılmaması için.
    program = provider_health.build_schedule(CONFIGS, 9, TASKS, block_size=3)
    blok_basi = [program[i * 3][0] for i in range(3)]
    assert len(set(blok_basi)) == 3, f"blok sırası dönmedi: {blok_basi}"


def test_iki_prompt_tipi_de_kullanilir():
    program = provider_health.build_schedule(CONFIGS, 10, TASKS, block_size=5)
    assert set(k for _, k, _ in program) == set(provider_health.PROMPT_KINDS)


def test_gecersiz_blok_boyu_reddedilir():
    with pytest.raises(ValueError):
        provider_health.build_schedule(CONFIGS, 10, TASKS, block_size=0)


def test_varsayilan_blok_boyu_config_ten_gelir():
    assert HEALTH_GATE_BLOCK_SIZE == 10


# --- Karar kuralı ------------------------------------------------------------

def test_temiz_kosu_kullanilabilir():
    kayitlar = [_kayit() for _ in range(20)]
    o = provider_health.summarize(kayitlar, "main")
    assert o["logical_call_incident_rate"] == 0.0
    assert o["decision"] == "KULLANILABILIR"


def test_tasima_istisnasi_karara_GIRER():
    # KRİTİK: yalnız incident oranına bakan bir kural, 20 çağrının TAMAMI ağ
    # istisnasıyla bitse bile "KULLANILABILIR" derdi (oran 0, exhausted 0).
    kayitlar = [_kayit(status="error") for _ in range(20)]
    o = provider_health.summarize(kayitlar, "main")
    assert o["terminal_failure_count"] == 20
    assert o["decision"] == "ROTA/MODEL DEGISTIR"


def test_tek_tasima_istisnasi_bile_kullanilabilir_demeyi_engeller():
    kayitlar = [_kayit() for _ in range(99)] + [_kayit(status="error")]
    o = provider_health.summarize(kayitlar, "main")
    assert o["decision"] == "ROTA/MODEL DEGISTIR"


def test_exhaustion_karara_girer():
    kayitlar = [_kayit() for _ in range(99)] + [_kayit(status="exhausted")]
    o = provider_health.summarize(kayitlar, "main")
    assert o["exhausted"] == 1
    assert o["decision"] == "ROTA/MODEL DEGISTIR"


def test_esik_altinda_retry_uyariyla_kullanilabilir():
    # %5 < oran <= %15 ve terminal hata yok -> uyarıyla kullanılabilir.
    kayitlar = ([_kayit(provider_attempt=2) for _ in range(10)]
                + [_kayit() for _ in range(90)])
    o = provider_health.summarize(kayitlar, "main")
    assert HEALTH_GATE_PASS_RATE < o["logical_call_incident_rate"] <= HEALTH_GATE_WARN_RATE
    assert o["decision"] == "UYARIYLA KULLANILABILIR"


def test_yuksek_oran_rota_degistir():
    kayitlar = ([_kayit(provider_attempt=2) for _ in range(20)]
                + [_kayit() for _ in range(80)])
    o = provider_health.summarize(kayitlar, "main")
    assert o["decision"] == "ROTA/MODEL DEGISTIR"


# --- Metrik ayrımı -----------------------------------------------------------

def test_gomulu_hata_sayisi_mantiksal_cagridan_ayri():
    # Bir mantıksal çağrı birden çok gömülü hata yanıtı görebilir; iki metrik
    # aynı şey DEĞİL ve karıştırılırsa oran yanlış raporlanır.
    kayitlar = [_kayit(provider_attempt=3), _kayit(), _kayit()]
    o = provider_health.summarize(kayitlar, "main")
    assert o["logical_call_incident_count"] == 1     # tek mantıksal çağrı etkilendi
    assert o["embedded_error_attempt_count"] == 2    # ama iki bozuk yanıt görüldü


def test_tukenen_cagrinin_gomulu_hatalari_tam_sayilir():
    # Tükenen çağrıda denemelerin HEPSİ bozuk yanıttır (son deneme de sağlam
    # değil). provider_attempt-1 kuralı burada bir hatayı eksik sayardı; kayıt
    # hiç provider_attempt taşımazsa dördü birden sıfır sayılırdı.
    kayitlar = [_kayit(status="exhausted", provider_attempt=4), _kayit()]
    o = provider_health.summarize(kayitlar, "main")
    assert o["embedded_error_attempt_count"] == 4
    assert o["terminal_failure_count"] == 1


def test_prompt_tipine_gore_olay_ayrilir():
    # JSON mode'un hata oranı farklı olabilir ve sözleşme kolları ona bağımlı.
    kayitlar = [_kayit(prompt_kind="json_planner", provider_attempt=2),
                _kayit(prompt_kind="code")]
    o = provider_health.summarize(kayitlar, "main")
    assert o["incidents_by_prompt_kind"] == {"json_planner": 1, "code": 0}


# --- Gecikme özeti (n / mean / p50 / p95 / observed_max) ---------------------

def test_gecikme_ozeti_bilinen_carpik_dagilimda_dogru():
    # Bilinen dağılım: 1..100 -> lineer interpolasyon (R-7) ile p50=50.5,
    # p95=95.05, maks=100. Ortalama tek başına kuyruğu göstermez.
    v = [float(i) for i in range(1, 101)]
    o = provider_health.latency_summary(v)
    assert o["n"] == 100
    assert o["mean"] == pytest.approx(50.5)
    assert o["p50"] == pytest.approx(50.5)
    assert o["p95"] == pytest.approx(95.05)
    assert o["observed_max"] == pytest.approx(100.0)


def test_gecikme_ozeti_uzun_kuyrugu_ortalamadan_ayirir():
    # 19 hızlı + 1 çok yavaş çağrı: ortalama şişer, P50 gövdeyi korur,
    # gözlenen maksimum kuyruğu açıkça gösterir (2026-07-30 secondary imzası).
    v = [1.0] * 19 + [200.0]
    o = provider_health.latency_summary(v)
    assert o["p50"] == pytest.approx(1.0)
    assert o["observed_max"] == pytest.approx(200.0)
    assert o["mean"] > o["p50"]
    assert o["p95"] > o["p50"]


def test_gecikme_ozeti_sirasiz_girdide_de_ayni_sonuc():
    import random
    v = [3.0, 1.0, 2.0, 10.0, 5.0]
    karisik = v[:]
    random.Random(0).shuffle(karisik)
    assert provider_health.latency_summary(v) == provider_health.latency_summary(karisik)


def test_gecikme_ozeti_tek_gozlemde_tanimli():
    o = provider_health.latency_summary([4.2])
    assert o == {"n": 1, "mean": 4.2, "p50": 4.2, "p95": 4.2, "observed_max": 4.2}


def test_gecikme_ozeti_bos_girdide_null_dondurur():
    o = provider_health.latency_summary([])
    assert o["n"] == 0
    assert o["mean"] is o["p50"] is o["p95"] is o["observed_max"] is None


def test_ozet_gecikme_alanlarini_rapora_koyar():
    kayitlar = [_kayit(latency_s=x) for x in (1.0, 2.0, 3.0, 100.0)]
    o = provider_health.summarize(kayitlar, "main")
    assert set(o["latency_summary_s"]) == {"n", "mean", "p50", "p95", "observed_max"}
    assert o["latency_summary_s"]["n"] == 4
    assert o["latency_summary_s"]["observed_max"] == pytest.approx(100.0)
    assert o["mean_latency_s"] is not None   # geriye uyumlu alan korunuyor


def test_gecikme_ozeti_yalniz_basarili_cagrilardan_hesaplanir():
    # Terminal hata alan çağrının süresi gecikme dağılımına GİRMEZ: o çağrı bir
    # ölçüm üretmedi. Başarılı çağrının latency_s'i retry/backoff'u zaten içerir.
    kayitlar = [_kayit(latency_s=2.0), _kayit(status="exhausted", latency_s=999.0)]
    o = provider_health.summarize(kayitlar, "main")
    assert o["latency_summary_s"]["n"] == 1
    assert o["latency_summary_s"]["observed_max"] == pytest.approx(2.0)


def test_saglik_raporunda_gecikme_alanlari_var(kosu_ortami):
    provider_health.main()
    rapor = json.loads((_deney_dizini(kosu_ortami["tmp_path"]) / "health_report.json")
                       .read_text(encoding="utf-8"))
    ozet = rapor["configs"]["main"]["latency_summary_s"]
    assert set(ozet) == {"n", "mean", "p50", "p95", "observed_max"}
    assert ozet["n"] == 2
    assert "latency_summary_s" in rapor["metric_definitions"]
    assert "lineer interpolasyon" in rapor["metric_definitions"]["latency_summary_s"]


# --- Held-out yasağı ---------------------------------------------------------

def test_heldout_gorev_seti_kabul_edilmez():
    # §5.4: seçim commit edildikten sonra held-out görevlerde model pilotu
    # yapılmaz. CLI bunu seçenek olarak dahi sunmamalı.
    kaynak = (Path(__file__).parent.parent / "scripts" / "provider_health.py").read_text(
        encoding="utf-8")
    assert 'choices=["pilot"]' in kaynak
    assert 'default="pilot"' in kaynak


def test_heldout_argumani_argparse_tarafindan_reddedilir(monkeypatch):
    # Kaynak denetimi tek başına yeterli değil: CLI'nın kendisi de reddetmeli.
    monkeypatch.setattr(sys, "argv", ["provider_health.py", "--task-set", "heldout"])
    with pytest.raises(SystemExit) as e:
        provider_health.main()
    assert e.value.code == 2   # argparse "invalid choice"


def test_heldout_dizini_hicbir_yerde_referans_edilmez():
    kaynak = (Path(__file__).parent.parent / "scripts" / "provider_health.py").read_text(
        encoding="utf-8")
    for yasakli in ("HELDOUT_TASKS_DIR", "HELDOUT_TASK_SET", "tasks_heldout"):
        assert yasakli not in kaynak


# --- Kaynak provenance kapısı: kirli/doğrulanamayan git -> SIFIR çağrı --------

def _ok_response(text="ok", *, native="STOP", response_id="gen-1", model="m-real"):
    choice = types.SimpleNamespace(
        message=types.SimpleNamespace(content=text), finish_reason="stop",
        logprobs=None, provider_specific_fields={"native_finish_reason": native})
    usage = types.SimpleNamespace(prompt_tokens=11, completion_tokens=7)
    return types.SimpleNamespace(choices=[choice], usage=usage, id=response_id, model=model)


@pytest.fixture
def kosu_ortami(monkeypatch, tmp_path):
    """Gerçek ağ/log/throttle olmadan uçtan uca koşu ortamı; çağrıları sayar."""
    monkeypatch.setattr(llm_module, "_throttle", lambda: None)
    monkeypatch.setattr(llm_module.time, "sleep", lambda s: None)
    monkeypatch.setattr(llm_module, "LOGS_DIR", tmp_path)
    monkeypatch.setattr(llm_module, "CALL_LOG", tmp_path / "llm_calls.jsonl")
    monkeypatch.setattr("litellm.completion_cost", lambda **k: 0.0)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-not-real")
    monkeypatch.setattr(provider_health, "LOGS_DIR", tmp_path)
    monkeypatch.setattr(provider_health, "_git_dirty", lambda: False)
    monkeypatch.setattr(provider_health, "_git_commit", lambda: "a" * 40)
    cagrilar = []
    monkeypatch.setattr("litellm.completion",
                        lambda **k: (cagrilar.append(k), _ok_response())[1])
    monkeypatch.setattr(sys, "argv",
                        ["provider_health.py", "--calls", "2", "--block-size", "1",
                         "--configs", "main"])
    return {"tmp_path": tmp_path, "calls": cagrilar}


def _deney_dizini(tmp_path):
    dizinler = [p for p in tmp_path.iterdir() if p.is_dir() and p.name.startswith("exp_health_")]
    assert len(dizinler) == 1, f"tam olarak bir deney dizini beklenir: {dizinler}"
    return dizinler[0]


def test_kirli_agac_sifir_cagri(kosu_ortami, monkeypatch):
    monkeypatch.setattr(provider_health, "_git_dirty", lambda: True)
    with pytest.raises(SystemExit, match="kirli"):
        provider_health.main()
    assert kosu_ortami["calls"] == []


def test_dogrulanamayan_git_sifir_cagri(kosu_ortami, monkeypatch):
    # None ("doğrulanamadı") ile False ("temiz") AYNI ŞEY DEĞİLDİR.
    monkeypatch.setattr(provider_health, "_git_dirty", lambda: None)
    with pytest.raises(SystemExit, match="DOĞRULANAMIYOR"):
        provider_health.main()
    assert kosu_ortami["calls"] == []


def test_cozulemeyen_head_sifir_cagri(kosu_ortami, monkeypatch):
    monkeypatch.setattr(provider_health, "_git_commit", lambda: None)
    with pytest.raises(SystemExit, match="HEAD DOĞRULANAMIYOR"):
        provider_health.main()
    assert kosu_ortami["calls"] == []


# --- Tek deney dizini + manifest + provenance --------------------------------

def test_butun_artefaktlar_ayni_deney_dizininde(kosu_ortami):
    provider_health.main()
    out_dir = _deney_dizini(kosu_ortami["tmp_path"])
    for dosya in ("calls.jsonl", "llm_calls.jsonl", "health_manifest.json",
                  "health_report.json"):
        assert (out_dir / dosya).exists(), f"{dosya} deney dizininde olmalı"
    # Eski hata: calls.jsonl `logs/health_<ts>/`e, llm_calls.jsonl
    # `logs/exp_health_<ts>/`e yazılıyordu — iki ayrı dizin.
    assert not (kosu_ortami["tmp_path"] / out_dir.name.removeprefix("exp_")).exists()


def test_manifest_kaynak_ve_routing_provenansi_tasir(kosu_ortami):
    provider_health.main()
    m = json.loads((_deney_dizini(kosu_ortami["tmp_path"]) / "health_manifest.json")
                   .read_text(encoding="utf-8"))
    assert m["git_commit"] == "a" * 40
    assert m["reasoning_config"] == config.REASONING_CONFIG
    assert m["health_gate_models"] == dict(config.HEALTH_GATE_MODELS)
    assert m["measured_models"]["main"] == config.MODEL_MAIN
    assert m["provider_routing"]["main"] == config.provider_routing_for(config.MODEL_MAIN)
    assert m["calls_per_config"] == 2 and m["block_size"] == 1
    assert m["task_set"] == "pilot" and m["task_ids"]
    assert m["temperature"] == config.DEFAULT_TEMPERATURE
    assert m["max_tokens"] == config.MAX_OUTPUT_TOKENS
    assert m["llm_call_schema_version"] == config.LLM_CALL_SCHEMA_VERSION
    for alan in ("llm_num_retries", "llm_min_interval_s", "llm_timeout_s",
                 "llm_provider_error_retries", "llm_provider_error_backoff_s"):
        assert alan in m, alan
    assert set(m["prompt_hashes"]) == {"json_planner_system", "code_system"}
    assert len(m["prompt_hashes"]["json_planner_system"]) == 64
    assert m["python_version"] and m["platform"]


def test_basarili_saglik_satiri_tam_provenans_tasir(kosu_ortami):
    provider_health.main()
    satirlar = [json.loads(s) for s in
                (_deney_dizini(kosu_ortami["tmp_path"]) / "calls.jsonl")
                .read_text(encoding="utf-8").splitlines() if s.strip()]
    assert satirlar and all(s["status"] == "ok" for s in satirlar)
    s = satirlar[0]
    for alan in ("actual_model", "actual_provider", "response_id", "finish_reason",
                 "native_finish_reason", "input_tokens", "output_tokens",
                 "reasoning_tokens", "cached_tokens", "provider_attempt",
                 "cost_usd", "latency_s"):
        assert alan in s, alan
    assert s["actual_model"] == "m-real"
    assert s["native_finish_reason"] == "STOP"
    assert s["response_id"] == "gen-1"
    assert s["provider_attempt"] == 1


def test_model_ciktisi_saglik_kaydina_YAZILMAZ(kosu_ortami, monkeypatch):
    gizli = "BU-METIN-KAYDEDILMEMELI"
    monkeypatch.setattr("litellm.completion", lambda **k: _ok_response(gizli))
    provider_health.main()
    metin = (_deney_dizini(kosu_ortami["tmp_path"]) / "calls.jsonl").read_text(encoding="utf-8")
    assert gizli not in metin
    for yasakli in ("response_text", "content", "text"):
        assert f'"{yasakli}"' not in metin


def test_rapor_manifest_kimligini_tekrarlar(kosu_ortami):
    provider_health.main()
    rapor = json.loads((_deney_dizini(kosu_ortami["tmp_path"]) / "health_report.json")
                       .read_text(encoding="utf-8"))
    assert rapor["git_commit"] == "a" * 40
    assert rapor["reasoning_config"] == config.REASONING_CONFIG
    assert rapor["manifest_file"] == "health_manifest.json"
    assert rapor["configs"]["main"]["decision"] == "KULLANILABILIR"
