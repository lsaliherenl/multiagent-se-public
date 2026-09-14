"""Model × sağlayıcı sağlık kapısı — ana koşudan ÖNCE zorunlu (P1).

EXPERIMENT_PROTOCOL.md §4. Amaç, ana modeli "daha pahalı model daha iyidir"
varsayımıyla değil, ölçülmüş **taşıma katmanı güvenilirliği** ile seçmek.

Neden gerekli
-------------
2026-07-27 teşhisi: OpenRouter upstream hatasını (429 rate limit) HTTP 200
gövdesine gömerek döndürüyor; LiteLLM native 'error'ı 'stop'a eşlediği için
istisna atılmıyordu. Bu arıza kollara EŞİT dağılmaz: structured ve contract
kolları geçerli JSON beklediği için kesik yanıttan naive'den daha çok zarar
görür ve contract'ın retry'ı sağlayıcı arızasını planlayıcı başarısızlığı
sanabilir. Yani ölçülmemiş bir taşıma arızası, sözleşme müdahalesinin aleyhine
sistematik yanlılık üretir.

Tasarım kararları
-----------------
- **HELD-OUT GÖREVLER KULLANILMAZ (sert yasak).** Plan §5.4: seçim commit
  edildikten sonra held-out görevlerde model pilotu yapılmaz. Sağlık ölçümü
  yalnız pilot/development setinde koşar ve **model ÇIKTILARINI hiçbir şekilde
  değerlendirmez/raporlamaz** — ölçülen tek şey taşıma katmanı sağlığıdır.
- **Ölçülen üç rol** (`config.HEALTH_GATE_MODELS`, §4): `main` birincil üretici
  rotası, `secondary` tam çapraz-model replikasyon + judge rotası,
  `judge_external` üretici-dışı judge rotası. Bu liste üç üretici ADAYI değil,
  deneyde fiilen yüksek hacimle kullanılacak model–sağlayıcı rotalarıdır.
- **Karşı-dengelenmiş BLOKLAR** (dönüşümlü tek tek çağrı DEĞİL). Model başına
  ardışık `block_size` çağrı yapılır, blok sırası bloklar arasında döndürülür:

      Blok 1: main×10 → secondary×10 → judge_external×10
      Blok 2: secondary×10 → judge_external×10 → main×10
      Blok 3: judge_external×10 → main×10 → secondary×10

  Gerekçe: tek tek dönüşümlü çağrıda global throttle üç modele bölünür, yani
  her model ~3×throttle aralıkla çağrılır. Ana koşuda ise tek model ARDIŞIK
  çalışacağı için gerçek aralık throttle kadardır. Rate-limit kaynaklı bir
  arızayı dönüşümlü ölçmek onu OLDUĞUNDAN DÜŞÜK gösterir. Blok içi tempo ana
  koşuyu taklit eder, blok sırası rotasyonu zamansal kesintinin tek modele
  yığılmasını engeller.
- **İki prompt tipi de bulunur** (JSON planner + kod üretimi): JSON mode'un
  hata oranı farklı olabilir ve sözleşme kolları tam olarak ona bağımlıdır.
- Ölçüm `call_model()` üzerinden yapılır, yani ana koşuyla AYNI throttle, retry
  ve loglama yolundan geçer.
- **TEK deney dizini** (`logs/exp_health_<ts>/`): sağlık özeti (`calls.jsonl`),
  ham çağrı provenance'ı (`llm_calls.jsonl`), `health_manifest.json` ve
  `health_report.json` aynı yerde durur. Ayrı dizinlere dağılsalardı bir kapı
  kararının hangi ham çağrılardan çıktığı ancak zaman damgası eşleştirerek
  tahmin edilebilirdi. NOT: `llm_calls.jsonl` proje genelindeki standart çağrı
  logudur ve `response_text` içerir; sağlık kaydı (`calls.jsonl`) ve KARAR ise
  model çıktısına hiç bakmaz — ölçülen tek şey taşıma katmanıdır.
- **Kaynak ve routing provenance'ına bağlıdır:** ölçüm, DOĞRULANMIŞ ve temiz bir
  git durumundan başlar (fail-closed) ve manifest hangi kodun/parametrelerin
  ölçüldüğünü kaydeder. Slug/routing/retry/reasoning değişirse kapı baştan
  koşar — manifest bu karşılaştırmanın kanıtıdır.

Raporlanan metrikler (üçü AYRI kavram, karıştırılmamalı)
--------------------------------------------------------
- `logical_call_incident_rate` — retry gerektiren VEYA terminal hata alan
  mantıksal çağrıların oranı.
- `embedded_error_attempt_count` — gözlenen HTTP-200-içi hata yanıtı sayısı.
- `terminal_failure_count` — bütün retry'lardan sonra sonuç alınamayan çağrılar
  (taşıma istisnaları dahil).

"Ham HTTP deneme oranı" DENMEZ: LiteLLM'in kendi iç retry denemeleri bu
wrapper'dan tek tek görülmediği için gerçek HTTP denemelerinin tamamı
gözlenmiyor; bu ifade fazla güçlü olurdu.

Karar: her rol AYRI ve BAĞIMSIZ bir kapıdır (§4). Bir rotanın düşmesi başka bir
modele sessiz fallback veya rol devri yaptırmaz; yalnız o rolün koşusu
başlatılmaz. Tablo yetenek/kod başarısı sıralaması DEĞİLDİR.

Kullanım:
    uv run python scripts/provider_health.py --calls 100
    uv run python scripts/provider_health.py --calls 6 --block-size 3   # hızlı deneme
"""

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from agents.llm import ProviderResponseError, call_model  # noqa: E402
from agents.planner import CONTRACT_SYSTEM_PROMPT  # noqa: E402
from config import (  # noqa: E402
    DEFAULT_TEMPERATURE,
    HEALTH_GATE_BLOCK_SIZE,
    HEALTH_GATE_CALLS_PER_CONFIG,
    HEALTH_GATE_MODELS,
    HEALTH_GATE_PASS_RATE,
    HEALTH_GATE_WARN_RATE,
    LLM_CALL_SCHEMA_VERSION,
    LLM_MIN_INTERVAL_S,
    LLM_NUM_RETRIES,
    LLM_PROVIDER_ERROR_BACKOFF_S,
    LLM_PROVIDER_ERROR_RETRIES,
    LLM_TIMEOUT_S,
    LOGS_DIR,
    MAX_OUTPUT_TOKENS,
    REASONING_CONFIG,
    ROOT,
    provider_routing_for,
)
from eval.harness import load_all_tasks  # noqa: E402
from pipeline.baseline import SYSTEM_PROMPT as CODER_SYSTEM_PROMPT  # noqa: E402

# Sağlık kapısına giren üç rol — TEK KAYNAK config.HEALTH_GATE_MODELS (§4).
# Burada ikinci bir liste tutulmaz: script ile config ayrışırsa ölçülen rota ile
# deneyde kullanılan rota sessizce farklılaşırdı.
CONFIGS = dict(HEALTH_GATE_MODELS)

PROMPT_KINDS = ("json_planner", "code")


class HealthGateError(RuntimeError):
    """Sağlık kapısı güvenle çalıştırılamaz — ölçüm başlatılmadan durulur."""


# --- Kaynak provenance kapısı (İLK API ÇAĞRISINDAN ÖNCE) ----------------------
# NOT: aynı üç yardımcı `eval/compatibility_smoke.py`'de de var. Ortak bir
# modüle çıkarmak doğru olur ama bu hotfix'in kapsamı dışında; iki kopya da
# AYNI semantiği uygular (None = "doğrulanamadı" ≠ "temiz").

def _git_commit() -> str | None:
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                           text=True, timeout=10, cwd=ROOT)
        return (r.stdout.strip() or None) if r.returncode == 0 else None
    except Exception:
        return None


def _git_dirty() -> bool | None:
    """True=kirli, False=temiz, None=DOĞRULANAMIYOR (git yok/komut hata verdi)."""
    try:
        r = subprocess.run(["git", "status", "--porcelain"], capture_output=True,
                           text=True, timeout=10, cwd=ROOT)
        return bool(r.stdout.strip()) if r.returncode == 0 else None
    except Exception:
        return None


def require_verified_git_state() -> str:
    """Doğrulanmış + temiz git durumu şart; değilse HİÇ çağrı yapılmadan durulur.

    Sağlık kapısı bir KARAR üretir ("bu rota ana koşuda kullanılabilir"), ve o
    karar ancak ölçülen kodun kimliği bilinirse geçerlidir. Kirli ya da
    doğrulanamayan bir ağaçta 300 ücretli çağrı yapmak, sonradan "hangi
    slug/routing/retry ayarıyla ölçüldü" sorusunu cevapsız bırakır. Bu yüzden
    fail-closed: "doğrulanamadı" (None) ile "temiz" (False) AYNI ŞEY DEĞİLDİR.

    Returns: doğrulanmış HEAD commit'i (manifeste yazılır).
    """
    dirty = _git_dirty()
    if dirty is None:
        raise HealthGateError(
            "git çalışma ağacı durumu DOĞRULANAMIYOR (git yok ya da komut hata "
            "verdi) — sağlık kapısı fail-closed durur.")
    if dirty:
        raise HealthGateError(
            "git çalışma ağacı kirli — sağlık kapısı SADECE temiz ağaçtan "
            "başlar. Önce çalışma ağacını commit ederek sabitle.")
    commit = _git_commit()
    if not commit:
        raise HealthGateError(
            "git HEAD DOĞRULANAMIYOR — sağlık kapısı fail-closed durur "
            "(hangi kaynak koddan ölçüldüğü kayda geçemeyecek çağrı yapılmaz).")
    return commit


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_health_manifest(*, experiment: str, git_commit: str, configs: list[str],
                          calls: int, block_size: int, task_set: str,
                          task_ids: list[str]) -> dict:
    """Kapının hangi koşullarda ölçüldüğünün tam kaydı.

    Sağlık kararı bu koşullara BAĞLIDIR: slug, provider routing, reasoning,
    temperature/max_tokens, retry/throttle/timeout veya prompt'lar değişirse
    ölçüm artık ana koşunun yapacağı çağrıyı temsil etmez ve kapı baştan
    koşmalıdır (§4). Manifest, bu karşılaştırmayı sonradan yapılabilir kılar.
    """
    return {
        "created_ts": datetime.now(timezone.utc).isoformat(),
        "experiment": experiment,
        "git_commit": git_commit,
        # Rol -> slug: kapının hangi ROTALARI ölçtüğü (yetenek sıralaması değil).
        "health_gate_models": dict(CONFIGS),
        "measured_configs": list(configs),
        "measured_models": {c: CONFIGS[c] for c in configs},
        "calls_per_config": calls,
        "block_size": block_size,
        "schedule": "counterbalanced_blocks",
        "task_set": task_set,
        "task_ids": list(task_ids),
        "provider_routing": {c: provider_routing_for(CONFIGS[c]) for c in configs},
        "reasoning_config": REASONING_CONFIG,
        "temperature": DEFAULT_TEMPERATURE,
        "max_tokens": MAX_OUTPUT_TOKENS,
        "llm_num_retries": LLM_NUM_RETRIES,
        "llm_min_interval_s": LLM_MIN_INTERVAL_S,
        "llm_timeout_s": LLM_TIMEOUT_S,
        "llm_provider_error_retries": LLM_PROVIDER_ERROR_RETRIES,
        "llm_provider_error_backoff_s": LLM_PROVIDER_ERROR_BACKOFF_S,
        "llm_call_schema_version": LLM_CALL_SCHEMA_VERSION,
        "prompt_hashes": {
            "json_planner_system": _sha256(CONTRACT_SYSTEM_PROMPT),
            "code_system": _sha256(CODER_SYSTEM_PROMPT),
        },
        "thresholds": {"pass_rate": HEALTH_GATE_PASS_RATE,
                       "warn_rate": HEALTH_GATE_WARN_RATE},
        "python_version": platform.python_version(),
        "platform": platform.platform(),
    }


def _messages(kind: str, task: dict) -> tuple[list[dict], dict]:
    """Ana koşuda fiilen kullanılan iki prompt tipinden biri."""
    if kind == "json_planner":
        return ([{"role": "system", "content": CONTRACT_SYSTEM_PROMPT},
                 {"role": "user", "content": f"Task ID: {task['task_id']}\n\n{task['prompt']}"}],
                {"response_format": {"type": "json_object"}})
    return ([{"role": "system", "content": CODER_SYSTEM_PROMPT},
             {"role": "user", "content": task["prompt"]}], {})


def build_schedule(configs: list[str], calls: int, tasks: list[dict],
                   block_size: int) -> list[tuple]:
    """Karşı-dengelenmiş bloklu program: (config, prompt_kind, task).

    Blok İÇİNDE tek model ardışık koşar (ana koşunun gerçek temposu); blok
    sırası bloklar arasında döndürülür (zamansal kesinti tek modele yığılmasın).
    Her konfigürasyon tam `calls` çağrı alır.
    """
    if block_size < 1:
        raise ValueError("block_size >= 1 olmalı")
    n_blocks = -(-calls // block_size)  # yukarı yuvarlama
    schedule = []
    counts = {c: 0 for c in configs}
    prompt_index = 0
    for block in range(n_blocks):
        offset = block % len(configs)
        for config in configs[offset:] + configs[:offset]:
            for _ in range(block_size):
                if counts[config] >= calls:
                    break
                kind = PROMPT_KINDS[prompt_index % len(PROMPT_KINDS)]
                task = tasks[prompt_index % len(tasks)]
                prompt_index += 1
                counts[config] += 1
                schedule.append((config, kind, task))
    return schedule


def _percentile(sorted_values: list[float], q: float) -> float:
    """Sıralı örnek üzerinde LİNEER İNTERPOLASYONLU yüzdelik (numpy'siz).

    `pos = q * (n - 1)`, komşu iki gözlem arasında doğrusal geçiş — R-7 /
    numpy varsayılanı ile aynı tanım ve `analysis/analyze.py::_percentile` ile
    AYNI yöntem (iki tablo karşılaştırılabilir kalsın diye). Deterministiktir:
    aynı girdi her zaman aynı sayıyı verir, rastgelelik veya sıralama
    belirsizliği yoktur (eşit değerler sonucu değiştirmez).
    """
    pos = q * (len(sorted_values) - 1)
    low = int(pos)
    high = min(low + 1, len(sorted_values) - 1)
    frac = pos - low
    return sorted_values[low] * (1 - frac) + sorted_values[high] * frac


def latency_summary(values: list[float], digits: int = 3) -> dict:
    """Çağrı düzeyi gecikme özeti: n, mean, p50, p95, observed_max.

    Ortalama tek başına yanıltıcıdır: 2026-07-30 koşusunda `secondary` rotası
    ort. 26.4 s iken medyan 15.8 s, p95 78.8 s ve gözlenen maksimum 166.7 s
    çıktı — yani ortalama tipik çağrıyı da en kötü hâli de anlatmıyor.

    İki ölçünün İŞİ FARKLIDIR: beklenen TOPLAM süre ampirik ORTALAMA (ve
    gözlenen toplam süre) ile tahmin edilir; p50/p95/observed_max ise dağılımın
    çarpıklığını, zaman tamponunu ve timeout riskini belirlemek için raporlanır.
    p95'i çağrı sayısıyla çarpmak toplam süre tahmini DEĞİLDİR — her çağrının
    kuyrukta olduğunu varsayar ve ciddi biçimde aşırı tahmin üretir.

    `observed_max` bir tahmin edici DEĞİL, gözlenen tek bir uç değerdir; bu
    yüzden `n` ile birlikte raporlanır. Ölçülen şey ROTA'nın operasyonel
    gecikmesidir (sağlayıcı seçimi + kuyruk + retry dahil), modelin içsel hızı
    değildir.
    """
    if not values:
        return {"n": 0, "mean": None, "p50": None, "p95": None, "observed_max": None}
    ordered = sorted(values)
    return {
        "n": len(ordered),
        "mean": round(sum(ordered) / len(ordered), digits),
        "p50": round(_percentile(ordered, 0.5), digits),
        "p95": round(_percentile(ordered, 0.95), digits),
        "observed_max": round(ordered[-1], digits),
    }


def summarize(kayitlar: list[dict], config: str) -> dict:
    """Konfigürasyonun sağlık özeti + ön-kayıtlı eşiklere göre karar.

    Karar `terminal_failures` (exhausted + taşıma istisnası) içerir: yalnız
    incident oranına bakan bir kural, 100 çağrının TAMAMI ağ istisnasıyla
    bitse bile "KULLANILABILIR" diyebilirdi.
    """
    alt = [k for k in kayitlar if k["config"] == config]
    toplam = len(alt)
    ok = [k for k in alt if k["status"] == "ok"]
    exhausted = sum(k["status"] == "exhausted" for k in alt)
    transport_exception = sum(k["status"] == "error" for k in alt)
    terminal_failures = exhausted + transport_exception

    # Retry gerektiren VEYA terminal hata alan mantıksal çağrılar.
    recovered = sum(k.get("provider_attempt", 1) > 1 for k in ok)
    incidents = recovered + terminal_failures
    incident_rate = incidents / toplam if toplam else 0.0
    # HTTP-200 içine gömülü hata yanıtı SAYISI (mantıksal çağrı sayısı değil).
    # Başarılı çağrıda son deneme sağlamdır (attempt-1 bozuk); TÜKENEN çağrıda
    # denemelerin HEPSİ bozuktur, bu yüzden attempt'in kendisi sayılır.
    embedded_errors = sum(
        k["provider_attempt"] if k["status"] == "exhausted" and "provider_attempt" in k
        else max(k.get("provider_attempt", 1) - 1, 0)
        for k in alt)

    if incident_rate <= HEALTH_GATE_PASS_RATE and terminal_failures == 0:
        karar = "KULLANILABILIR"
    elif incident_rate <= HEALTH_GATE_WARN_RATE and terminal_failures == 0:
        karar = "UYARIYLA KULLANILABILIR"
    else:
        karar = "ROTA/MODEL DEGISTIR"

    gecikme = [k["latency_s"] for k in ok if k.get("latency_s") is not None]
    return {
        "model": alt[0]["model"] if alt else None,
        "calls": toplam,
        "logical_call_incident_rate": round(incident_rate, 4),
        "logical_call_incident_count": incidents,
        "embedded_error_attempt_count": embedded_errors,
        "terminal_failure_count": terminal_failures,
        "exhausted": exhausted,
        "transport_exception": transport_exception,
        "recovered_by_retry": recovered,
        # `actual_provider` yeni kanonik ad; `provider` geriye uyumlu (eski
        # sağlık artefaktları yalnız onu taşır) — ikisi de aynı kavramdır.
        "provider_distribution": dict(Counter(
            k.get("actual_provider", k.get("provider")) for k in ok)),
        "incidents_by_prompt_kind": {
            kind: sum(k.get("provider_attempt", 1) > 1 or k["status"] != "ok"
                      for k in alt if k["prompt_kind"] == kind)
            for kind in PROMPT_KINDS
        },
        # `mean_latency_s` geriye uyumluluk için korunur (2 basamak); dağılımın
        # kendisi `latency_summary_s`tedir. Yalnız BAŞARILI çağrıların gecikmesi
        # girer ve o değer retry/backoff süresini ZATEN içerir (agents/llm.py
        # sayacı ilk denemeden önce başlar) — bozuk denemelerin süresi ayrıca
        # eklenmez, yoksa retry çift sayılırdı.
        "mean_latency_s": round(sum(gecikme) / len(gecikme), 2) if gecikme else None,
        "latency_summary_s": latency_summary(gecikme),
        "total_cost_usd": round(sum(k.get("cost_usd") or 0 for k in ok), 6),
        "decision": karar,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Model × sağlayıcı sağlık kapısı")
    parser.add_argument("--calls", type=int, default=HEALTH_GATE_CALLS_PER_CONFIG,
                        help="konfigürasyon başına çağrı sayısı")
    # Yalnız formal sağlık ROLLERİ; `dev`/`pilot` gibi takma adlar veya serbest
    # slug bu kümeye sessizce giremez (girseydi rapor, deneyde kullanılmayan bir
    # rotayı "sağlık kapısı geçti" diye kaydedebilirdi).
    parser.add_argument("--configs", nargs="+", default=list(CONFIGS),
                        choices=list(CONFIGS),
                        help=f"alt küme smoke için; varsayılan üçü birden: {list(CONFIGS)}")
    parser.add_argument("--block-size", type=int, default=HEALTH_GATE_BLOCK_SIZE,
                        help="blok başına ardışık çağrı (ana koşu temposunu taklit eder)")
    parser.add_argument("--task-set", default="pilot", choices=["pilot"],
                        help="YALNIZ pilot. Held-out görevlerde model pilotu yapılmaz (§5.4).")
    args = parser.parse_args()

    # SIRA ÖNEMLİ: kaynak provenance kapısı, görev yüklemeden ve TEK bir API
    # çağrısından önce koşar (fail-closed).
    try:
        git_commit = require_verified_git_state()
    except HealthGateError as e:
        sys.exit(str(e))

    tasks = load_all_tasks(args.task_set)[:10]
    if not tasks:
        sys.exit(f"{args.task_set!r} görev seti boş.")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    # TEK deney dizini: call_model(experiment=...) da `logs/exp_<experiment>/`
    # altına yazar, yani llm_calls.jsonl bu dizinde oluşur.
    experiment_name = f"health_{stamp}"
    out_dir = LOGS_DIR / f"exp_{experiment_name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "calls.jsonl"

    schedule = build_schedule(args.configs, args.calls, tasks, args.block_size)
    manifest = build_health_manifest(
        experiment=experiment_name, git_commit=git_commit, configs=args.configs,
        calls=args.calls, block_size=args.block_size, task_set=args.task_set,
        task_ids=[t["task_id"] for t in tasks])
    (out_dir / "health_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"sağlık kapısı: {len(args.configs)} konfigürasyon × {args.calls} çağrı "
          f"= {len(schedule)} çağrı | blok boyu {args.block_size} (karşı-dengelenmiş)")
    print(f"görev seti: {args.task_set} (held-out KULLANILMAZ) | commit: {git_commit[:12]}")
    print(f"deney dizini: {out_dir}\n")

    kayitlar = []
    with out_path.open("a", encoding="utf-8") as out:
        for i, (config, kind, task) in enumerate(schedule, 1):
            model = CONFIGS[config]
            messages, extra = _messages(kind, task)
            kayit = {"ts": datetime.now(timezone.utc).isoformat(), "config": config,
                     "model": model, "prompt_kind": kind, "task_id": task["task_id"]}
            try:
                r = call_model(messages, model=model, temperature=DEFAULT_TEMPERATURE,
                               task_id=task["task_id"], agent_role=f"health_{kind}",
                               experiment=experiment_name, **extra)
                # Model ÇIKTISI (response_text) bilinçli olarak KAYDEDİLMEZ ve
                # değerlendirilmez: bu bir yetenek ölçümü değil, taşıma sağlığı
                # ölçümüdür. Kaydedilen her alan taşıma/provenance alanıdır ve
                # provider_error kayıtlarıyla AYNI adları taşır.
                kayit.update(status="ok", provider_attempt=r.provider_attempt,
                             actual_model=r.actual_model, actual_provider=r.actual_provider,
                             # `provider` geriye uyumluluk için korunur (eski
                             # sağlık artefaktları bu adı kullanıyor).
                             provider=r.provider, response_id=r.response_id,
                             finish_reason=r.finish_reason,
                             native_finish_reason=r.native_finish_reason,
                             input_tokens=r.input_tokens, output_tokens=r.output_tokens,
                             reasoning_tokens=r.reasoning_tokens,
                             cached_tokens=r.cached_tokens,
                             cost_usd=r.cost_usd, latency_s=round(r.latency_s, 3))
            except ProviderResponseError as e:
                # Bütün denemeler tükendi: bu konfigürasyonun sağlık notu.
                # provider_attempt istisnadan okunur — tükenen bir çağrının
                # BÜTÜN denemeleri bozuk yanıttır; kaydedilmezse gömülü hata
                # sayımı bu çağrıyı sıfır sayar.
                kayit.update(status="exhausted", provider_attempt=e.provider_attempt,
                             error_signature=e.error_signature, error=str(e)[:300])
            except Exception as e:
                kayit.update(status="error", error=f"{type(e).__name__}: {str(e)[:200]}")
            out.write(json.dumps(kayit, ensure_ascii=False) + "\n")
            out.flush()
            kayitlar.append(kayit)
            if i % 20 == 0 or i == len(schedule):
                print(f"  [{i}/{len(schedule)}]")

    _report(kayitlar, out_dir, args.block_size, args.task_set, manifest)


def _report(kayitlar: list[dict], out_dir: Path, block_size: int, task_set: str,
            manifest: dict | None = None) -> None:
    """Ön-kayıtlı eşiklere göre konfigürasyon başına karar."""
    print("\n=== Sağlık raporu (taşıma katmanı; model çıktısı değerlendirilmez) ===")
    ozet = {c: summarize(kayitlar, c) for c in sorted({k["config"] for k in kayitlar})}
    for config, o in ozet.items():
        print(f"\n{config} ({o['model']})")
        print(f"  mantıksal çağrı olayı {o['logical_call_incident_count']}/{o['calls']} "
              f"(%{o['logical_call_incident_rate']*100:.1f}) | "
              f"gömülü hata yanıtı {o['embedded_error_attempt_count']}")
        print(f"  retry ile kurtarılan {o['recovered_by_retry']} | "
              f"terminal hata {o['terminal_failure_count']} "
              f"(tükenen {o['exhausted']}, taşıma istisnası {o['transport_exception']})")
        print(f"  prompt tipine göre olay: {o['incidents_by_prompt_kind']}")
        print(f"  sağlayıcı dağılımı: {o['provider_distribution']}")
        g = o["latency_summary_s"]
        print(f"  gecikme (n={g['n']}): ort {g['mean']}s | P50 {g['p50']}s | "
              f"P95 {g['p95']}s | gözlenen maks {g['observed_max']}s")
        print(f"  maliyet ${o['total_cost_usd']}")
        print(f"  KARAR: {o['decision']}")

    rapor = {
        "created_ts": datetime.now(timezone.utc).isoformat(),
        # Rapor, kendi koşulunun provenance'ına bağlı okunmalı: manifest AYNI
        # dizindedir, kimlik alanları burada da tekrarlanır ki tek bir dosyaya
        # bakan biri hangi commit/ayarla ölçüldüğünü görsün.
        "experiment": (manifest or {}).get("experiment"),
        "git_commit": (manifest or {}).get("git_commit"),
        "reasoning_config": (manifest or {}).get("reasoning_config"),
        "llm_call_schema_version": (manifest or {}).get("llm_call_schema_version"),
        "manifest_file": "health_manifest.json",
        "task_set": task_set,
        "block_size": block_size,
        "schedule": "counterbalanced_blocks",
        "note": ("Yalnız taşıma katmanı sağlığı ölçülür; model çıktıları "
                 "değerlendirilmez ve kaydedilmez. Held-out görevler kullanılmaz. "
                 "Roller bağımsızdır: bir rotanın düşmesi başka bir modele rol "
                 "devri veya sessiz fallback yaptırmaz."),
        "metric_definitions": {
            "logical_call_incident_rate": "retry gerektiren veya terminal hata alan mantıksal çağrı oranı",
            "embedded_error_attempt_count": "gözlenen HTTP-200-içi hata yanıtı sayısı",
            "terminal_failure_count": "bütün retry'lardan sonra sonuç alınamayan çağrılar",
            "latency_summary_s": ("başarılı çağrıların gecikme dağılımı (n/mean/p50/p95/"
                                  "observed_max). Yüzdelikler SIRALI örnek üzerinde lineer "
                                  "interpolasyonla hesaplanır (pos = q*(n-1); R-7 / numpy "
                                  "varsayılanı) — deterministiktir. Başarılı çağrının "
                                  "gecikmesi retry/backoff süresini zaten içerir; bozuk "
                                  "denemeler ayrıca eklenmez. observed_max bir tahmin edici "
                                  "değil, gözlenen tek bir uç değerdir. Ölçülen şey ROTANIN "
                                  "operasyonel gecikmesidir, modelin içsel hızı değil."),
        },
        "thresholds": {"pass_rate": HEALTH_GATE_PASS_RATE, "warn_rate": HEALTH_GATE_WARN_RATE},
        "routing_policy": {c: provider_routing_for(o["model"]) for c, o in ozet.items()},
        "configs": ozet,
    }
    path = out_dir / "health_report.json"
    path.write_text(json.dumps(rapor, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nRapor -> {path}")
    print("\nRol-bazlı kapılar (EXPERIMENT_PROTOCOL §4) — bağımsızdır, rol devri YOKTUR:")
    print("  main geçmezse           -> Gemini held-out koşusu BAŞLAMAZ (eski sürüme dönülmez)")
    print("  secondary geçmezse      -> DeepSeek held-out koşusu ve MAST paneli BAŞLAMAZ")
    print("  judge_external geçmezse -> MAST etiketleme BAŞLAMAZ (üretici verisi geçersiz olmaz)")
    print("Slug/routing/retry/tespit değişirse 3 × 100 kapı baştan koşar. Bu tablo "
          "yetenek sıralaması değil, taşıma katmanı sağlığıdır.")


if __name__ == "__main__":
    main()
