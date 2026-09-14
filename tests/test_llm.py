"""agents/llm.py::call_model() birim testleri -- litellm tamamen mock'lanır
(gerçek call_model() bugüne kadar hiçbir testte doğrudan çağrılmadı; hem
_throttle hem LOGS_DIR/CALL_LOG mock'lanmazsa gerçek 3s bekleme + repo'nun
gerçek logs/llm_calls.jsonl'ına yazma riski var)."""

import json
from types import SimpleNamespace

import pytest

import agents.llm as llm_module
import config
from agents.llm import call_model


def _fake_response(text="ok", finish_reason="stop", completion_tokens=5):
    choice = SimpleNamespace(
        message=SimpleNamespace(content=text),
        finish_reason=finish_reason,
        logprobs=None,
    )
    usage = SimpleNamespace(prompt_tokens=10, completion_tokens=completion_tokens)
    return SimpleNamespace(choices=[choice], usage=usage)


def _corrupt_response(text='{"task_'):
    """Sağlayıcı hatası imzası: HTTP 200 ama hiç çıktı tokeni yok."""
    return _fake_response(text=text, completion_tokens=0)


def _rate_limited_response():
    """2026-07-27'de canlı gözlenen gerçek biçim: 429 HTTP 200 gövdesinde."""
    r = _fake_response(text='{\n  "task_id": "x",\n  "function_sig', completion_tokens=0)
    r.choices[0].provider_specific_fields = {
        "error": {"code": 429, "message": "temporarily rate-limited upstream",
                  "metadata": {"error_type": "rate_limit_exceeded"}},
        "native_finish_reason": "error",
    }
    r.id = "gen-test-1"
    return r


def _yanit_ile(**provider_fields):
    r = _fake_response()
    r.choices[0].provider_specific_fields = provider_fields
    return r


def _usage_siz_yanit():
    r = _fake_response()
    r.usage = None
    return r


@pytest.fixture(autouse=True)
def _no_throttle_no_real_logs(monkeypatch, tmp_path):
    monkeypatch.setattr(llm_module, "_throttle", lambda: None)
    monkeypatch.setattr(llm_module.time, "sleep", lambda s: None)  # retry backoff'u atla
    monkeypatch.setattr(llm_module, "LOGS_DIR", tmp_path)
    monkeypatch.setattr(llm_module, "CALL_LOG", tmp_path / "llm_calls.jsonl")
    monkeypatch.setattr("litellm.completion_cost", lambda **k: 0.001)
    return tmp_path


def _read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_experiment_verilince_deney_bazli_yola_yazar(monkeypatch, _no_throttle_no_real_logs):
    monkeypatch.setattr("litellm.completion", lambda **k: _fake_response())
    call_model([{"role": "user", "content": "hi"}], model="fake-model",
               task_id="t1", agent_role="planner",
               experiment="exp1", run_id="run-abc", arm="contract", repeat=2, agent_attempt=3)

    log_path = _no_throttle_no_real_logs / "exp_exp1" / "llm_calls.jsonl"
    assert log_path.exists()
    records = _read_jsonl(log_path)
    assert len(records) == 1
    r = records[0]
    assert r["status"] == "ok"
    assert r["experiment"] == "exp1"
    assert r["run_id"] == "run-abc"
    assert r["arm"] == "contract"
    assert r["repeat"] == 2
    assert r["agent_attempt"] == 3
    assert r["task_id"] == "t1"
    assert r["agent_role"] == "planner"


def test_experiment_verilmeyince_global_yola_agent_attempt_1_varsayilaniyla_yazar(
    monkeypatch, _no_throttle_no_real_logs
):
    monkeypatch.setattr("litellm.completion", lambda **k: _fake_response())
    call_model([{"role": "user", "content": "hi"}], model="fake-model")

    records = _read_jsonl(_no_throttle_no_real_logs / "llm_calls.jsonl")
    assert len(records) == 1
    r = records[0]
    assert r["status"] == "ok"
    assert r["experiment"] is None
    assert r["agent_attempt"] == 1


def test_yeni_kwarglar_litellm_completiona_sizmiyor(monkeypatch, _no_throttle_no_real_logs):
    captured = {}

    def fake_completion(**kwargs):
        captured.update(kwargs)
        return _fake_response()

    monkeypatch.setattr("litellm.completion", fake_completion)
    call_model([{"role": "user", "content": "hi"}], model="fake-model",
               experiment="exp1", run_id="run-abc", arm="contract", repeat=1, agent_attempt=2)

    for leaky_kwarg in ("experiment", "run_id", "arm", "repeat", "agent_attempt", "task_id", "agent_role"):
        assert leaky_kwarg not in captured


def test_ortak_model_parametreleri_tek_noktadan_uygulanir(monkeypatch, _no_throttle_no_real_logs):
    # İç geçerlilik şartı: max_tokens/reasoning/provider routing agent kodundan
    # DEĞİL, config'ten tek noktadan gelir -- kollar arası fark yaratamasın.
    captured = {}

    def fake_completion(**kwargs):
        captured.update(kwargs)
        return _fake_response()

    monkeypatch.setattr("litellm.completion", fake_completion)
    call_model([{"role": "user", "content": "hi"}], model="openrouter/test/model")

    assert captured["max_tokens"] == llm_module.MAX_OUTPUT_TOKENS
    assert captured["extra_body"]["reasoning"] == llm_module.REASONING_CONFIG
    assert captured["extra_body"]["provider"] == config.OPENROUTER_PROVIDER_ROUTING
    assert captured["extra_body"]["usage"] == config.OPENROUTER_USAGE_ACCOUNTING


def test_minimax_kendi_saglayicisina_sabitlenir(monkeypatch, _no_throttle_no_real_logs):
    # 2026-07-27 probe: MiniMax M3 aynı parametrelerle dört farklı sağlayıcıya
    # gidiyordu (DeepInfra/Minimax/Morph/Together) -> gizli varyans kaynağı.
    # İstisna artık ÜÇÜNCÜ-JUDGE kimliğine bağlı, bir role değil.
    captured = {}

    def fake_completion(**kwargs):
        captured.update(kwargs)
        return _fake_response()

    monkeypatch.setattr("litellm.completion", fake_completion)
    call_model([{"role": "user", "content": "hi"}], model=config.MODEL_JUDGE_EXTERNAL,
               experiment="exp1")

    routing = captured["extra_body"]["provider"]
    assert routing["order"] == ["minimax"]
    assert routing["allow_fallbacks"] is False
    # Politika ayrıca çağrı kaydına da yazılır (provenance).
    r = _read_jsonl(_no_throttle_no_real_logs / "exp_exp1" / "llm_calls.jsonl")[0]
    assert r["provider_routing"]["order"] == ["minimax"]


def test_deepseek_cagrilari_minimax_rotasina_zorlanmaz(
    monkeypatch, _no_throttle_no_real_logs
):
    # REGRESYON (2026-07-29): routing istisnası eskiden `MODEL_SECONDARY`
    # anahtarındaydı ve o sabit MiniMax'ti. Replikasyon modeli DeepSeek'e
    # çevrilirken satır olduğu gibi bırakılsaydı, ikinci üreticinin BÜTÜN
    # çağrıları MiniMax sağlayıcı rotasına gider ve 600 arm-run yanlış rotada
    # toplanırdı.
    captured = {}

    def fake_completion(**kwargs):
        captured.update(kwargs)
        return _fake_response()

    monkeypatch.setattr("litellm.completion", fake_completion)
    call_model([{"role": "user", "content": "hi"}], model=config.MODEL_SECONDARY,
               experiment="exp_ds")

    routing = captured["extra_body"]["provider"]
    assert routing.get("order") != ["minimax"]
    assert routing == config.OPENROUTER_PROVIDER_ROUTING
    # Gerçekleşen politika provenance'a da doğru yazılıyor.
    r = _read_jsonl(_no_throttle_no_real_logs / "exp_exp_ds" / "llm_calls.jsonl")[0]
    assert "order" not in r["provider_routing"]
    assert r["provider_routing"]["allow_fallbacks"] is True


def test_logprobs_rutin_olarak_istenmiyor(monkeypatch, _no_throttle_no_real_logs):
    # 2026-07-27 canlı probe: require_parameters + logprobs, Gemini için UYGUN
    # ENDPOINT BIRAKMIYOR (404) ve diğer modellerde provider seçimini
    # değiştiriyor. Regresyon koruması: logprobs geri eklenirse ana model
    # tamamen çalışmaz hale gelir.
    captured = {}

    def fake_completion(**kwargs):
        captured.update(kwargs)
        return _fake_response()

    monkeypatch.setattr("litellm.completion", fake_completion)
    call_model([{"role": "user", "content": "hi"}], model="openrouter/test/model")

    assert "logprobs" not in captured


def test_openrouter_disi_modelde_openrouter_govde_alanlari_gonderilmez(
    monkeypatch, _no_throttle_no_real_logs
):
    # reasoning/provider OpenRouter'a özgü gövde alanları; başka sağlayıcıya
    # gönderilirse hata riski var. max_tokens ise her sağlayıcıda ortak kalır.
    captured = {}

    def fake_completion(**kwargs):
        captured.update(kwargs)
        return _fake_response()

    monkeypatch.setattr("litellm.completion", fake_completion)
    call_model([{"role": "user", "content": "hi"}], model="anthropic/claude-x")

    assert captured["max_tokens"] == llm_module.MAX_OUTPUT_TOKENS
    assert "extra_body" not in captured


def test_deney_kosusunda_model_verilmezse_hata(monkeypatch, _no_throttle_no_real_logs):
    # Ana deney verisinin sessizce MODEL_PILOT'a düşmesi en pahalı hata olurdu:
    # experiment verilmişse model AÇIKÇA istenir.
    monkeypatch.setattr("litellm.completion", lambda **k: _fake_response())
    with pytest.raises(ValueError, match="model açıkça belirtilmelidir"):
        call_model([{"role": "user", "content": "hi"}], experiment="exp1")


def test_deney_disi_kullanimda_pilot_modele_dusebilir(monkeypatch, _no_throttle_no_real_logs):
    # Debug/bağımsız kullanımda (experiment=None) varsayılan korunur.
    monkeypatch.setattr("litellm.completion", lambda **k: _fake_response())
    result = call_model([{"role": "user", "content": "hi"}])
    assert result.model == llm_module.MODEL_PILOT


def test_log_kaydinda_sema_surumu_ve_saglayici_provenansi_var(
    monkeypatch, _no_throttle_no_real_logs
):
    response = _fake_response()
    response.model = "test/model-2026-01"      # sağlayıcının döndürdüğü gerçek ad
    response.provider = "TestProvider"
    response.id = "gen-ok-1"
    response.choices[0].provider_specific_fields = {"native_finish_reason": "STOP"}
    response.usage.completion_tokens_details = SimpleNamespace(reasoning_tokens=42)
    monkeypatch.setattr("litellm.completion", lambda **k: response)

    result = call_model([{"role": "user", "content": "hi"}], model="openrouter/test/model",
                        experiment="exp1")

    r = _read_jsonl(_no_throttle_no_real_logs / "exp_exp1" / "llm_calls.jsonl")[0]
    assert r["schema_version"] == llm_module.LLM_CALL_SCHEMA_VERSION
    assert r["model"] == "openrouter/test/model"      # İSTENEN slug
    assert r["actual_model"] == "test/model-2026-01"  # DÖNEN ad
    assert r["provider"] == "TestProvider"
    assert r["reasoning_tokens"] == 42                # ortak ayar AÇIK; harcanan gizli bütçe her çağrıda loglanır
    assert r["max_tokens"] == llm_module.MAX_OUTPUT_TOKENS
    assert r["reasoning_config"] == llm_module.REASONING_CONFIG
    assert result.reasoning_tokens == 42
    # P1 Parça 6A: başarılı kayıt artık provider_error kayıtlarıyla AYNI dört
    # provenance alanını taşır.
    assert r["response_id"] == "gen-ok-1"
    assert r["native_finish_reason"] == "STOP"
    # Varsayılan routing'de "order" hiç yok -> istenen sağlayıcı ["auto"].
    assert "order" not in config.OPENROUTER_PROVIDER_ROUTING
    assert r["requested_provider"] == ["auto"]
    assert r["actual_provider"] == "TestProvider" == r["provider"]
    assert result.response_id == "gen-ok-1"
    assert result.native_finish_reason == "STOP"
    assert result.actual_provider == result.provider == "TestProvider"


def test_maliyet_openrouter_usage_accountingden_okunur(monkeypatch, _no_throttle_no_real_logs):
    # 2026-07-27 probe: litellm.completion_cost() deneyin ÜÇ modelini de
    # tanımıyor. Sağlayıcının kendi faturaladığı usage.cost öncelikli olmalı;
    # aksi halde bütün maliyet sütunu None kalırdı.
    response = _fake_response()
    response.usage.cost = 0.000123
    monkeypatch.setattr("litellm.completion", lambda **k: response)
    monkeypatch.setattr("litellm.completion_cost", lambda **k: 999.0)  # kullanılmamalı

    result = call_model([{"role": "user", "content": "hi"}], model="openrouter/test/model")
    assert result.cost_usd == 0.000123


def test_usage_costu_yoksa_litellm_hesabina_duser(monkeypatch, _no_throttle_no_real_logs):
    monkeypatch.setattr("litellm.completion", lambda **k: _fake_response())
    monkeypatch.setattr("litellm.completion_cost", lambda **k: 0.5)

    result = call_model([{"role": "user", "content": "hi"}], model="openrouter/test/model")
    assert result.cost_usd == 0.5


def test_cache_isabeti_loglanir(monkeypatch, _no_throttle_no_real_logs):
    # Prompt cache isabeti kollar arasında farklılaşabilir -> maliyet
    # karşılaştırmasının kontrol değişkeni.
    response = _fake_response()
    response.usage.prompt_tokens_details = SimpleNamespace(cached_tokens=114)
    monkeypatch.setattr("litellm.completion", lambda **k: response)

    call_model([{"role": "user", "content": "hi"}], model="openrouter/test/model",
               experiment="exp1")
    r = _read_jsonl(_no_throttle_no_real_logs / "exp_exp1" / "llm_calls.jsonl")[0]
    assert r["cached_tokens"] == 114


def test_saglayici_provenans_alanlari_yoksa_none_dondurur(monkeypatch, _no_throttle_no_real_logs):
    # Sağlayıcı/litellm bu alanları vermezse analizde eksik veri görünür, çökmez.
    # response.id/provider_specific_fields hiç YOK -- sahte response_id/
    # native_finish_reason UYDURULMAZ.
    monkeypatch.setattr("litellm.completion", lambda **k: _fake_response())
    result = call_model([{"role": "user", "content": "hi"}], model="openrouter/test/model",
                        experiment="exp1")
    assert result.provider is None
    assert result.actual_provider is None
    assert result.reasoning_tokens is None
    assert result.response_id is None
    assert result.native_finish_reason is None
    r = _read_jsonl(_no_throttle_no_real_logs / "exp_exp1" / "llm_calls.jsonl")[0]
    assert r["response_id"] is None
    assert r["native_finish_reason"] is None
    assert r["actual_provider"] is None


def test_istenen_saglayici_provider_routingden_okunur(monkeypatch, _no_throttle_no_real_logs):
    # requested_provider actual_provider'dan UYDURULMAZ; routing.order'dan gelir.
    monkeypatch.setattr("litellm.completion", lambda **k: _fake_response())
    call_model([{"role": "user", "content": "hi"}], model=config.MODEL_JUDGE_EXTERNAL,
               experiment="exp1")
    r = _read_jsonl(_no_throttle_no_real_logs / "exp_exp1" / "llm_calls.jsonl")[0]
    assert r["requested_provider"] == ["minimax"]
    assert r["actual_provider"] is None  # sağlayıcı bu testte hiç bildirmedi


@pytest.mark.parametrize("kurucu,beklenen_imza", [
    (lambda: _rate_limited_response(), "error_in_response_body"),
    (lambda: _yanit_ile(native_finish_reason="error"), "native_finish_reason_error"),
    (lambda: _fake_response(finish_reason="error"), "explicit_error_finish_reason"),
    (lambda: _fake_response(completion_tokens=0), "zero_completion_tokens"),
    (lambda: _fake_response(completion_tokens=-1), "zero_completion_tokens"),
    (lambda: _fake_response(text="   "), "empty_content"),
    (lambda: _usage_siz_yanit(), "missing_usage"),
])
def test_saglayici_hata_imzalari_tespit_edilir(kurucu, beklenen_imza):
    # Tespit tek bir sinyale (completion_tokens==0) bağlı kalmamalı: sağlayıcı
    # arızasını birden çok biçimde bildiriyor.
    assert llm_module._provider_error(kurucu()) == beklenen_imza


def test_saglam_yanitta_imza_yok():
    assert llm_module._provider_error(_yanit_ile(native_finish_reason="STOP")) is None


def test_max_tokens_kesilmesi_saglayici_hatasi_sayilmaz(monkeypatch, _no_throttle_no_real_logs):
    # finish_reason="length" model düzeyinde NORMAL kesilmedir; taşıma hatası
    # sayılıp yeniden denenirse gerçek bir model davranışı gizlenmiş olur.
    yanit = _fake_response(text="kod", finish_reason="length")
    assert llm_module._provider_error(yanit) is None

    monkeypatch.setattr("litellm.completion", lambda **k: yanit)
    call_model([{"role": "user", "content": "hi"}], model="openrouter/test/model",
               experiment="exp1")
    kayit = _read_jsonl(_no_throttle_no_real_logs / "exp_exp1" / "llm_calls.jsonl")[0]
    assert kayit["status"] == "ok"
    assert kayit["truncated_by_max_tokens"] is True   # ayrıca işaretlenir


def test_rate_limit_kaydi_tam_provenans_tasir(monkeypatch, _no_throttle_no_real_logs):
    yanitlar = iter([_rate_limited_response(), _fake_response("saglam")])
    monkeypatch.setattr("litellm.completion", lambda **k: next(yanitlar))
    call_model([{"role": "user", "content": "hi"}], model="openrouter/test/model",
               experiment="exp1")

    hata = _read_jsonl(_no_throttle_no_real_logs / "exp_exp1" / "llm_calls.jsonl")[0]
    assert hata["status"] == "provider_error"
    assert hata["error_signature"] == "error_in_response_body"
    assert hata["upstream_error_code"] == 429
    assert hata["upstream_error_type"] == "rate_limit_exceeded"
    assert hata["provider_attempt"] == 1
    assert hata["requested_model"] == "openrouter/test/model"
    assert hata["response_id"] == "gen-test-1"
    assert hata["completion_tokens"] == 0
    assert hata["content_length"] > 0
    assert isinstance(hata["latency_ms"], int)


def test_bozuk_saglayici_yaniti_yeniden_denenir(monkeypatch, _no_throttle_no_real_logs):
    # 2026-07-27 canlı bulgu: Gemini çağrılarının %20-40'ı HTTP 200 ile ama
    # completion_tokens=0 ve JSON'un ortasında kesik içerikle dönüyor. İstisna
    # atılmadığı için transport retry'ı devreye girmiyordu; bozuk plan sessizce
    # coder'a gidiyordu.
    yanitlar = iter([_corrupt_response(), _corrupt_response(), _fake_response("saglam")])
    monkeypatch.setattr("litellm.completion", lambda **k: next(yanitlar))

    r = call_model([{"role": "user", "content": "hi"}], model="openrouter/test/model",
                   experiment="exp1")
    assert r.text == "saglam"

    kayitlar = _read_jsonl(_no_throttle_no_real_logs / "exp_exp1" / "llm_calls.jsonl")
    # Bozuk denemeler AYRI loglanır -> oran bildiride raporlanabilir.
    assert [k["status"] for k in kayitlar] == ["provider_error", "provider_error", "ok"]
    assert kayitlar[-1]["provider_error_retries"] == 2


def test_bozuk_yanit_israr_ederse_hata_firlatilir(monkeypatch, _no_throttle_no_real_logs):
    # Tükenirse istisna: runner run_error yazar, kayıt analizde ölçüm sayılmaz.
    monkeypatch.setattr("litellm.completion", lambda **k: _corrupt_response())
    with pytest.raises(llm_module.ProviderResponseError) as exc:
        call_model([{"role": "user", "content": "hi"}], model="openrouter/test/model",
                   experiment="exp1")

    kayitlar = _read_jsonl(_no_throttle_no_real_logs / "exp_exp1" / "llm_calls.jsonl")
    assert len(kayitlar) == llm_module.LLM_PROVIDER_ERROR_RETRIES + 1
    assert all(k["status"] == "provider_error" for k in kayitlar)
    # İstisna deneme sayısını TAŞIMALI: taşımazsa çağıranın (sağlık kapısı)
    # elinde yalnız "tükendi" bilgisi kalır ve o çağrının bütün bozuk yanıtları
    # gömülü hata sayımında sıfır görünür.
    assert exc.value.provider_attempt == llm_module.LLM_PROVIDER_ERROR_RETRIES + 1
    assert exc.value.error_signature == kayitlar[-1]["error_signature"]


def test_cagri_logu_analizin_okudugu_alanlari_tasir(monkeypatch, _no_throttle_no_real_logs):
    # Anti-drift: analysis/analyze.py maliyeti/tokenı/gecikmeyi bu alanlardan
    # okur ve eksikse SIFIR sayar (çökmez). llm.py'nin log formatı değişirse
    # sessiz sıfır yerine burada kırılsın.
    from eval.result_schema import LLM_CALL_USAGE_FIELDS

    monkeypatch.setattr("litellm.completion", lambda **k: _fake_response())
    call_model([{"role": "user", "content": "hi"}], model="openrouter/test/model",
               experiment="exp1", run_id="r1", arm="contract", task_id="t00", repeat=0)
    kayit = _read_jsonl(_no_throttle_no_real_logs / "exp_exp1" / "llm_calls.jsonl")[0]
    eksik = [f for f in LLM_CALL_USAGE_FIELDS if f not in kayit]
    assert not eksik, f"çağrı logu analizin beklediği alanları taşımıyor: {eksik}"


def test_saglam_yanitta_retry_sayaci_sifir(monkeypatch, _no_throttle_no_real_logs):
    monkeypatch.setattr("litellm.completion", lambda **k: _fake_response())
    call_model([{"role": "user", "content": "hi"}], model="openrouter/test/model",
               experiment="exp1")
    kayit = _read_jsonl(_no_throttle_no_real_logs / "exp_exp1" / "llm_calls.jsonl")[0]
    assert kayit["provider_error_retries"] == 0


def test_litellm_hata_atinca_status_error_loglanir_ve_hata_yukari_firlatilir(
    monkeypatch, _no_throttle_no_real_logs
):
    def fake_completion(**kwargs):
        raise RuntimeError("sağlayıcı hatası")

    monkeypatch.setattr("litellm.completion", fake_completion)
    with pytest.raises(RuntimeError, match="sağlayıcı hatası"):
        call_model([{"role": "user", "content": "hi"}], model="fake-model",
                   experiment="exp1", run_id="run-abc", arm="naive", repeat=0, agent_attempt=1)

    records = _read_jsonl(_no_throttle_no_real_logs / "exp_exp1" / "llm_calls.jsonl")
    assert len(records) == 1
    r = records[0]
    assert r["status"] == "error"
    assert r["error_type"] == "RuntimeError"
    assert "sağlayıcı hatası" in r["error_message"]
    assert r["run_id"] == "run-abc"
