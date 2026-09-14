"""Tek LLM giriş noktası: call_model().

Agent/pipeline kodu hiçbir yerde SDK'ya doğrudan dokunmaz; loglama, retry,
throttle, maliyet takibi ve fırsatçı logprobs toplama tek noktada yapılır.

Ortak model parametreleri (max_tokens, reasoning, OpenRouter provider routing)
da buradan uygulanır — hiçbir agent kendi değerini geçirmez. Deneyin iç
geçerlilik şartı bunu gerektiriyor: kollar arasındaki TEK fark iletişim katmanı
olmalı, model parametreleri değil (EXPERIMENT_PROTOCOL.md §3-§4).

Model seçimi: experiment verilmişse model AÇIKÇA belirtilmek zorunda — ana
deney verisinin sessizce MODEL_PILOT'a düşmesi engellenir (aşağıda ValueError).

Logprobs notu: birincil belirsizlik metriği self-consistency proxy'sidir.
Logprobs ARTIK rutin olarak İSTENMEZ (config.REQUEST_LOGPROBS=False) —
require_parameters routing'iyle birlikte Gemini'yi tamamen çalıştırılamaz
hale getiriyordu ve provider seçimini değiştiriyordu (2026-07-27 canlı probe;
gerekçe config.py'de). Yanıtta yine de gelirse fırsatçı olarak okunur.

Loglama birleştirme anahtarı: experiment + run_id + arm + task_id + repeat +
agent_role + agent_attempt. experiment verilmişse birincil kayıt
logs/exp_<experiment>/llm_calls.jsonl'a gider; verilmezse (bağımsız/debug
kullanım — uncertainty/self_consistency.py, eval/mast_labels.py gibi) global
logs/llm_calls.jsonl'a düşer. "agent_attempt" bilerek "attempt" DEĞİL:
LiteLLM'nin kendi num_retries transport-katmanı retry'ıyla (aşağıda,
görünmez) karışmasın diye -- farklı katman, farklı kavram.
"""

import json
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import litellm

from config import (
    DEFAULT_TEMPERATURE,
    LLM_CALL_SCHEMA_VERSION,
    LLM_MIN_INTERVAL_S,
    LLM_NUM_RETRIES,
    LLM_PROVIDER_ERROR_BACKOFF_S,
    LLM_PROVIDER_ERROR_RETRIES,
    LLM_TIMEOUT_S,
    LOGS_DIR,
    MAX_OUTPUT_TOKENS,
    MODEL_PILOT,
    OPENROUTER_USAGE_ACCOUNTING,
    REASONING_CONFIG,
    REQUEST_LOGPROBS,
    provider_routing_for,
)

__all__ = ["ModelResponse", "ProviderResponseError", "call_model"]

# Sağlayıcının desteklemediği parametreler (örn. Anthropic'te logprobs)
# hata fırlatmak yerine sessizce düşürülür.
litellm.drop_params = True
# Ücretsiz modellerde maliyet hesaplanamayınca basılan "Provider List"
# uyarı gürültüsünü kapat.
litellm.suppress_debug_info = True

CALL_LOG = LOGS_DIR / "llm_calls.jsonl"  # experiment verilmediğinde fallback yolu

_throttle_lock = threading.Lock()
_last_call_ts = 0.0


@dataclass
class ModelResponse:
    text: str
    model: str                    # İSTENEN model slug'ı
    input_tokens: int
    output_tokens: int
    cost_usd: float | None
    latency_s: float
    logprobs: list[float] | None  # token başına log-olasılık; sağlayıcı desteklemiyorsa None
    finish_reason: str | None
    reasoning_tokens: int | None = None   # gizli düşünme tokenları (sağlayıcı bildiriyorsa)
    cached_tokens: int | None = None      # prompt cache'inden gelen girdi tokenları
    provider_attempt: int = 1             # kaçıncı TAŞIMA denemesinde başarıldı (1 = ilk)
    actual_model: str | None = None       # sağlayıcının döndürdüğü model adı
    provider: str | None = None           # OpenRouter'ın yönlendirdiği gerçek sağlayıcı (geriye uyumlu alan)
    # --- P1 Parça 6A: provider_error kayıtlarıyla AYNI provenance kavramları ---
    response_id: str | None = None        # sağlayıcının yanıt kimliği (ör. "gen-...")
    native_finish_reason: str | None = None  # upstream'in ham finish_reason'ı (STOP/error/...)
    requested_provider: list | None = None   # istenen routing.order (yoksa ["auto"])
    actual_provider: str | None = None    # provider ile AYNI değer -- kanonik ad (provider korunur)


def _throttle() -> None:
    """İstekler arası minimum bekleme (OpenRouter free ~20 istek/dk)."""
    global _last_call_ts
    with _throttle_lock:
        wait = LLM_MIN_INTERVAL_S - (time.monotonic() - _last_call_ts)
        if wait > 0:
            time.sleep(wait)
        _last_call_ts = time.monotonic()


def _extract_logprobs(response) -> list[float] | None:
    try:
        content = response.choices[0].logprobs["content"]
        return [tok["logprob"] for tok in content]
    except (AttributeError, KeyError, TypeError, IndexError):
        return None


def _common_params(model: str) -> dict:
    """Bütün rollerde/kollarda aynı olan model parametreleri (config'ten).

    İç geçerlilik şartı: agent kodu kendi max_tokens/reasoning/routing değerini
    geçirmez — tek kaynak burasıdır. reasoning ve provider routing OpenRouter'a
    özgü gövde alanları olduğu için yalnız openrouter/ slug'larında gönderilir.
    """
    params: dict = {"max_tokens": MAX_OUTPUT_TOKENS}
    if REQUEST_LOGPROBS:
        params["logprobs"] = True
    if model.startswith("openrouter/"):
        extra_body = {}
        if REASONING_CONFIG is not None:
            extra_body["reasoning"] = REASONING_CONFIG
        routing = provider_routing_for(model)
        if routing is not None:
            extra_body["provider"] = routing
        if OPENROUTER_USAGE_ACCOUNTING is not None:
            extra_body["usage"] = OPENROUTER_USAGE_ACCOUNTING
        if extra_body:
            params["extra_body"] = extra_body
    return params


def _cost(response) -> float | None:
    """Çağrının gerçek maliyeti.

    Önce OpenRouter'ın usage accounting'inden okunur (sağlayıcının fiilen
    faturalandırdığı değer). litellm.completion_cost() yalnız yedek: 2026-07-27
    probe'unda deneyin ÜÇ modelini de tanımadı ("This model isn't mapped yet"),
    yani tek başına güvenilemez.
    """
    cost = getattr(response.usage, "cost", None)
    if cost is not None:
        return cost
    try:
        return litellm.completion_cost(completion_response=response)
    except Exception:
        return None


def _cached_tokens(usage) -> int | None:
    """Prompt cache'inden gelen girdi tokenları (sağlayıcı bildiriyorsa).

    Maliyet karşılaştırmasında kontrol değişkeni: kollar aynı prompt öneklerini
    farklı sıklıkta tekrarladığı için cache isabeti kollar arasında farklılaşıp
    maliyet farkını yapay olarak büyütebilir/küçültebilir.
    """
    try:
        return usage.prompt_tokens_details.cached_tokens
    except AttributeError:
        return None


def _reasoning_tokens(usage) -> int | None:
    """Gizli reasoning/thinking tokenları (sağlayıcı usage'da bildiriyorsa).

    EXPERIMENT_PROTOCOL.md §4: reasoning tokenları usage'a dahilse ayrıca
    kaydedilir. Ayar 2026-07-30'da AÇIK'a alındı (Gemini endpoint'i zorunlu
    kılıyor, bkz. config.REASONING_CONFIG) — bu sayaç yine de gerekli: gizli
    bütçe çağrıdan çağrıya değişir, maliyet/gecikme farkının ve `config` ile
    fiilen uygulanan ayar arasındaki sapmanın tek kanıtı bu alandır.
    """
    try:
        return usage.completion_tokens_details.reasoning_tokens
    except AttributeError:
        return None


def _provider(response) -> str | None:
    """OpenRouter'ın isteği fiilen yönlendirdiği sağlayıcı.

    litellm sürümleri bu alanı farklı yerlerde taşıyabildiği için savunmacı
    okunur; bulunamazsa None (analizde eksik veri olarak görünür, çökmez).
    """
    for source in (response, getattr(response, "_hidden_params", None)):
        if source is None:
            continue
        value = (source.get("provider") if isinstance(source, dict)
                 else getattr(source, "provider", None))
        if value:
            return str(value)
    return None


class ProviderResponseError(RuntimeError):
    """Sağlayıcı HTTP 200 döndürdü ama yanıt kullanılamaz (bkz. _provider_error).

    `provider_attempt` ve `error_signature` istisnanın ÜSTÜNDE taşınır: çağıran
    (sağlık kapısı, runner) tükenen bir çağrının kaç bozuk yanıt gördüğünü
    gövdeyi ayrıştırmadan öğrenebilsin. Taşınmazsa tükenen bir çağrının bütün
    gömülü hataları sayımda sıfır görünür.
    """

    def __init__(self, message: str, *, provider_attempt: int = 1,
                 error_signature: str | None = None):
        super().__init__(message)
        self.provider_attempt = provider_attempt
        self.error_signature = error_signature


def _choice_fields(response) -> dict:
    """OpenRouter'ın choice'a iliştirdiği sağlayıcıya özgü alanlar."""
    fields = getattr(response.choices[0], "provider_specific_fields", None)
    return fields if isinstance(fields, dict) else {}


def _provider_error(response) -> str | None:
    """Taşıma katmanı hatasının imzasını döndürür; sağlam yanıtta None.

    2026-07-27 canlı teşhis: OpenRouter, upstream hatasını **HTTP 200 gövdesine
    gömerek** döndürüyor. Gözlenen somut örnek:

        provider_specific_fields = {
          "error": {"code": 429, "message": "... temporarily rate-limited
                    upstream ...", "metadata": {"error_type": "rate_limit_exceeded"}},
          "native_finish_reason": "error"}
        usage.completion_tokens = 0, usage.cost = 0
        içerik: JSON'un ortasında kesik

    Sağlam yanıtta aynı alan yalnız {"native_finish_reason": "STOP"} taşıyor.
    LiteLLM native 'error' değerini choice.finish_reason='stop'a eşlediği için
    İSTİSNA ATILMIYOR -> transport retry'ı tetiklenmiyor -> bozuk yanıt normal
    yanıt gibi ajan zincirine giriyordu.

    Bu ayrım deneyin İÇ GEÇERLİLİĞİ için zorunlu: kesik yanıttan structured ve
    contract kolları (geçerli JSON bekledikleri için) naive'den daha çok zarar
    görür; ayrıca contract'ın retry'ı sağlayıcı arızasını planlayıcı
    başarısızlığı sanabilir. Yani taşıma hatası, sözleşme müdahalesinin
    aleyhine SİSTEMATİK yanlılık üretir.

    finish_reason == "length" BURADA HATA SAYILMAZ: o, model düzeyinde normal
    bir kesilmedir (max_tokens'a dayanma) ve ayrıca loglanır.
    """
    choice = response.choices[0]
    fields = _choice_fields(response)

    if fields.get("error"):
        return "error_in_response_body"
    if str(fields.get("native_finish_reason") or "").lower() == "error":
        return "native_finish_reason_error"
    if getattr(choice, "finish_reason", None) == "error":
        return "explicit_error_finish_reason"

    usage = getattr(response, "usage", None)
    completion_tokens = getattr(usage, "completion_tokens", None) if usage else None
    if completion_tokens is None:
        return "missing_usage"
    if completion_tokens <= 0:
        return "zero_completion_tokens"

    if not (choice.message.content or "").strip():
        return "empty_content"
    return None


def _provider_error_details(response, model: str, latency_s: float,
                            provider_attempt: int, signature: str) -> dict:
    """Bozuk denemenin tam provenance kaydı (sağlık kapısı bunu okur)."""
    choice = response.choices[0]
    usage = getattr(response, "usage", None)
    routing = provider_routing_for(model) or {}
    body_error = _choice_fields(response).get("error") or {}
    return {
        "status": "provider_error",
        "error_signature": signature,
        "provider_attempt": provider_attempt,
        "requested_model": model,
        "actual_model": getattr(response, "model", None),
        "requested_provider": routing.get("order", ["auto"]),
        "actual_provider": _provider(response),
        "response_id": getattr(response, "id", None),
        "finish_reason": getattr(choice, "finish_reason", None),
        "native_finish_reason": _choice_fields(response).get("native_finish_reason"),
        "upstream_error_code": body_error.get("code"),
        "upstream_error_type": (body_error.get("metadata") or {}).get("error_type"),
        "completion_tokens": getattr(usage, "completion_tokens", None) if usage else None,
        "content_length": len(choice.message.content or ""),
        "latency_ms": round(latency_s * 1000),
    }


def _log_path(experiment: str | None, namespace: str | None = None) -> Path:
    """Çağrı logunun yolu.

    namespace, aynı deneyin FARKLI amaçlı çağrılarını ayırır (ör. MAST
    etiketleme). Ayrılmazsa judge çağrıları ana performans çağrı loguna karışır
    ve "bir arm-run kaç çağrı yaptı / ne kadar tuttu" sorusu bozulur.
    """
    if not experiment:
        return CALL_LOG
    base = LOGS_DIR / f"exp_{experiment}"
    return (base / namespace / "llm_calls.jsonl") if namespace else base / "llm_calls.jsonl"


def _log_call(record: dict, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def call_model(
    messages: list[dict],
    model: str | None = None,
    temperature: float = DEFAULT_TEMPERATURE,
    *,
    task_id: str | None = None,
    agent_role: str | None = None,
    experiment: str | None = None,
    run_id: str | None = None,
    arm: str | None = None,
    repeat: int | None = None,
    agent_attempt: int = 1,
    log_namespace: str | None = None,
    **kwargs,
) -> ModelResponse:
    """LLM çağrısı yapar; her çağrıyı (başarılı VEYA başarısız) çağrı logune kaydeder.

    model=None → config.MODEL_PILOT, FAKAT yalnız deney dışı (experiment=None)
    kullanımda. experiment verilmişse model AÇIKÇA belirtilmek zorundadır:
    ana deney verisinin sessizce pilot modele düşmesi, fark edilmesi en zor ve
    en pahalı hata olurdu (bütün koşu çöpe gider). Bkz. EXPERIMENT_PROTOCOL.md §4.

    Ortak model parametreleri (max_tokens, reasoning, provider routing) burada
    tek noktadan uygulanır — çağıran katman geçirmez.

    task_id/agent_role/experiment/run_id/arm/repeat/agent_attempt SADECE log
    bağlamı içindir (litellm.completion'a asla sızmaz) — analiz aşamasında
    çağrıları deneye/kola/göreve/tekrara/role/denemeye göre gruplamak için.
    """
    if experiment and not model:
        raise ValueError(
            "Deney koşusunda (experiment verilmiş) model açıkça belirtilmelidir; "
            "örtük MODEL_PILOT'a düşmek ana deney verisini bozar."
        )
    model = model or MODEL_PILOT
    log_path = _log_path(experiment, log_namespace)
    base_record = {
        "schema_version": LLM_CALL_SCHEMA_VERSION,
        "experiment": experiment, "run_id": run_id, "arm": arm, "repeat": repeat,
        "agent_attempt": agent_attempt, "model": model, "task_id": task_id,
        "agent_role": agent_role, "temperature": temperature,
        "max_tokens": MAX_OUTPUT_TOKENS, "reasoning_config": REASONING_CONFIG,
        "provider_routing": provider_routing_for(model),
    }

    # Sağlayıcı-hatası retry döngüsü: bozuk yanıt istisna ATMADIĞI için
    # LiteLLM'in transport retry'ı devreye girmez (bkz. _is_corrupt).
    start = time.monotonic()
    provider_attempts = 0
    last_signature = None
    for attempt in range(1, LLM_PROVIDER_ERROR_RETRIES + 2):
        provider_attempts = attempt
        attempt_start = time.monotonic()
        _throttle()
        try:
            response = litellm.completion(
                model=model,
                messages=messages,
                temperature=temperature,
                num_retries=LLM_NUM_RETRIES,
                timeout=LLM_TIMEOUT_S,
                **_common_params(model),
                **kwargs,
            )
        except Exception as exc:
            _log_call({
                "ts": datetime.now(timezone.utc).isoformat(),
                **base_record,
                "status": "error",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "provider_attempt": attempt,
                "latency_s": round(time.monotonic() - start, 3),
            }, log_path)
            raise

        last_signature = _provider_error(response)
        if last_signature is None:
            break

        # Bozuk denemeler AYRI kayıt olarak loglanır: oran ve imza dağılımı
        # sağlık kapısında ölçülebilsin, sessizce yutulmasın.
        _log_call({
            "ts": datetime.now(timezone.utc).isoformat(),
            **base_record,
            **_provider_error_details(response, model, time.monotonic() - attempt_start,
                                      attempt, last_signature),
        }, log_path)
        if attempt <= LLM_PROVIDER_ERROR_RETRIES:
            time.sleep(LLM_PROVIDER_ERROR_BACKOFF_S * attempt)
    else:
        raise ProviderResponseError(
            f"{model}: {provider_attempts} denemenin tamamında taşıma katmanı hatası "
            f"(son imza: {last_signature}). Sağlayıcı tarafı sorunu — ajan/sözleşme "
            "başarısızlığı DEĞİL.",
            provider_attempt=provider_attempts,
            error_signature=last_signature,
        )
    latency = time.monotonic() - start

    usage = response.usage
    cost = _cost(response)
    choice = response.choices[0]
    logprobs = _extract_logprobs(response)
    routing = provider_routing_for(model) or {}

    result = ModelResponse(
        text=choice.message.content or "",
        model=model,
        input_tokens=usage.prompt_tokens,
        output_tokens=usage.completion_tokens,
        cost_usd=cost,
        latency_s=latency,
        logprobs=logprobs,
        finish_reason=choice.finish_reason,
        reasoning_tokens=_reasoning_tokens(usage),
        cached_tokens=_cached_tokens(usage),
        provider_attempt=provider_attempts,
        actual_model=getattr(response, "model", None),
        provider=_provider(response),
        # Upstream alan yoksa None -- sahte ID/finish_reason UYDURULMAZ.
        response_id=getattr(response, "id", None),
        native_finish_reason=_choice_fields(response).get("native_finish_reason"),
        requested_provider=routing.get("order", ["auto"]),
        actual_provider=_provider(response),
    )
    _log_call({
        "ts": datetime.now(timezone.utc).isoformat(),
        **base_record,
        "status": "ok",
        # provider_attempt: TAŞIMA katmanı denemesi (bu döngü).
        # agent_attempt: AJAN düzeyi deneme — contract kolunda planner'ın
        # sözleşme retry'ı. İki kavram bilinçli olarak ayrı alanlarda: biri
        # altyapı arızasını, diğeri modelin sözleşmeye uyma kapasitesini ölçer;
        # karıştırılırsa RQ3 sağlayıcı gürültüsüyle kirlenir.
        "provider_attempt": provider_attempts,
        "provider_error_retries": provider_attempts - 1,
        # finish_reason=="length" taşıma hatası DEĞİL; model düzeyi kesilme.
        # Ayrı işaretlenir ki analiz max_tokens'a dayanan koşuları görebilsin.
        "truncated_by_max_tokens": result.finish_reason == "length",
        "actual_model": result.actual_model,
        "provider": result.provider,  # geriye uyumlu alan (actual_provider'la aynı değer)
        # provider_error kayıtlarıyla AYNI dört alan (LLM_CALL_SCHEMA_VERSION
        # 2.1): başarılı ve hatalı kayıtlar aynı provenance kavramları için
        # farklı ad taşımasın (P1 Parça 6A).
        "response_id": result.response_id,
        "native_finish_reason": result.native_finish_reason,
        "requested_provider": result.requested_provider,
        "actual_provider": result.actual_provider,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "reasoning_tokens": result.reasoning_tokens,
        "cached_tokens": result.cached_tokens,
        "cost_usd": result.cost_usd,
        "latency_s": round(latency, 3),
        "finish_reason": result.finish_reason,
        "mean_logprob": (sum(logprobs) / len(logprobs)) if logprobs else None,
        "logprobs": logprobs,
        "messages": messages,
        "response_text": result.text,
    }, log_path)
    return result
