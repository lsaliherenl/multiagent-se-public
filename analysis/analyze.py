"""Ana analiz — görev-düzeyi oranlar + eşleştirilmiş fark + cluster bootstrap.

EXPERIMENT_PROTOCOL.md §8. Bu modül gerçek veriden BAĞIMSIZ olarak test edilir:
bütün fixture'lar `eval/result_schema.make_synthetic_record()` üzerinden üretilir,
yani alan adları testlerde elle taklit edilmez (sözleşme değişirse testler de
yeni sözleşmeye göre üretir).

Dört tasarım kararı — üçü sessizce yanlış sonuç üretecek hataları kapatıyor:

1. **Analiz birimi GÖREV.** Her (model, görev, kol) için önce tekrarlar üzerinden
   `pass_count / repeat_count` hesaplanır. 50 görev × 3 tekrar = 150 kayıt ASLA
   150 bağımsız gözlem sayılmaz: aynı görevin tekrarları görev zorluğu üzerinden
   ilişkilidir, kümeleme yok sayılırsa CI olduğundan DAR çıkar ve etki
   olduğundan güçlü görünür.

2. **Bootstrap birimi de GÖREV.** Her yinelemede görevler replacement ile
   seçilir; seçilen görevin bütün kol/tekrar kümesi BİRLİKTE taşınır. Görev
   içindeki tekrarları ayrı ayrı örneklemek aynı hatanın bootstrap'a taşınmış
   hâli olurdu.

3. **Analiz eksik/bozuk veri üzerinde ÇALIŞMAZ.** Yinelenen, beklenmeyen veya
   geçersiz kayıtta durur (waive edilemez). `--allow-missing` YALNIZ eksik koşuyu
   mazur görür ve çıktıyı `preliminary: true` diye damgalar.

4. **Özet dosyası zaman damgası İÇERMEZ.** Aynı seed + aynı veri bayt-bayt aynı
   özeti üretmeli; üretim zamanı gibi ortama bağlı bir GÖZLEM içeriğe karışırsa
   "aynı analiz aynı sonucu verir" iddiası doğrulanamaz hâle gelir (görev
   üretiminde aynı hata oluşmaması için).

Hipotez testi / p-değeri BİLİNÇLİ OLARAK YOK: ön-kayıtlı ana çıktı nokta
tahmini + cluster-bootstrap %95 CI'dır (§8).

Kullanım:
    uv run python -m analysis.analyze --exp logs/exp_gemini_main
"""

import argparse
import csv
import json
import random
import sys
from collections import Counter
from pathlib import Path

from config import (
    ANALYSIS_METRICS,
    ANALYSIS_SCHEMA_VERSION,
    ARM_CONTRACT,
    BOOTSTRAP_CI_LEVEL,
    BOOTSTRAP_ITERATIONS,
    BOOTSTRAP_SEED,
    LLM_CALL_SCHEMA_VERSION,
    MAX_PLANNER_ATTEMPTS,
    PRIMARY_COMPARISON,
    PRIMARY_METRIC,
    SECONDARY_COMPARISONS,
    STRUCTURED_MODES,
)
from eval.result_schema import (
    MANIFEST_REQUIRED_FIELDS,
    check_provenance,
    integrity_report,
    is_run_error,
)

__all__ = [
    "AnalysisError", "load_jsonl", "load_experiment", "single_model",
    "check_provenance", "check_integrity", "validate_calls", "task_level_rates",
    "paired_diffs", "bootstrap_draws", "bootstrap_ci", "paired_effect",
    "base_plus_attrition", "compliance_summary", "usage_by_run", "usage_summary",
    "analyze", "write_outputs",
]

# Manifest/provenance sözleşmesi eval/result_schema.py'de (analiz VE MAST hattı
# aynı kaynağı kullanır); buradan yalnız yeniden dışa aktarılır.

# agents/llm.py'nin ürettiği bilinen durumlar. Bilinmeyen bir değer BAŞARILI
# SAYILMAZ ve sessizce yok da sayılmaz — yeni bir durum eklenmişse analiz
# onu görmezden gelmek yerine durur.
CALL_STATUSES = ("ok", "provider_error", "error")

# Kullanım toplamları: veri eksikse SIFIR değil None yazılır (bkz. usage_summary).
USAGE_NUMERIC_FIELDS = ("input_tokens", "output_tokens", "reasoning_tokens",
                        "cached_tokens", "cost_usd", "latency_s")


class AnalysisError(RuntimeError):
    """Analiz güvenle yapılamaz — sessizce yanlış sonuç üretmektense durulur."""


# --- Yükleme ------------------------------------------------------------------

def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in
            path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_experiment(exp_dir: Path) -> tuple[dict, list[dict], list[dict]]:
    """(manifest, results, llm_calls). Manifest yoksa analiz yapılamaz.

    Manifest zorunlu: görev listesi, kol seti, tekrar sayısı ve model oradan
    gelir. Onlarsız "eksik koşu" ile "hiç planlanmamış koşu" ayırt edilemez.
    """
    manifest_path = exp_dir / "manifest.json"
    if not manifest_path.exists():
        raise AnalysisError(f"manifest yok: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return (manifest,
            load_jsonl(exp_dir / "results.jsonl"),
            load_jsonl(exp_dir / "llm_calls.jsonl"))


def single_model(records: list[dict]) -> str:
    """Kayıtlardaki TEK modeli döndürür; birden fazlaysa durur.

    Modeller havuzlanmaz (§8.4): Gemini ve MiniMax ayrı analiz edilir, çünkü
    RQ4 tam olarak "etki model ailesine göre değişiyor mu" sorusudur. Havuzlama
    o soruyu ortadan kaldırır ve iki farklı dağılımın ortalamasını tek bir
    "etki" gibi gösterir.
    """
    models = sorted({r["model"] for r in records})
    if len(models) != 1:
        raise AnalysisError(
            f"tek modelli analiz bekleniyor, {len(models)} model bulundu: {models}. "
            "Modeller HAVUZLANMAZ — her model için ayrı deney dizini/analiz.")
    return models[0]


# --- Bütünlük -----------------------------------------------------------------

def check_integrity(records: list[dict], manifest: dict, *,
                    allow_missing: bool = False) -> dict:
    """Analiz öncesi zorunlu kapı. Rapora ek olarak karar alanları döner.

    Beklenen anahtarlar MANİFESTİN modeliyle üretilir, kayıtlardan çıkarılan
    modelle DEĞİL: kayıtlardan çıkarmak, yanlış modelle koşulmuş tutarlı bir
    kümeyi kendi kendini doğrulayan hale getirirdi.

    Yinelenen / beklenmeyen / geçersiz kayıt MAZUR GÖRÜLMEZ: üçü de sonucu
    sessizce kaydırır (çift sayım, yanlış modele atıf, bozuk alan). Eksik koşu
    yalnız açık bayrakla ve `preliminary` damgasıyla geçilebilir.
    """
    report = integrity_report(records, manifest["task_ids"], manifest["arm_order"],
                              manifest["repeats"], manifest["model"])
    blockers = []
    if report["duplicates"]:
        blockers.append(f"{len(report['duplicates'])} yinelenen tamamlanmış kayıt "
                        f"(çift sayım): {list(report['duplicates'])[:3]}")
    if report["unexpected"]:
        blockers.append(f"{len(report['unexpected'])} beklenmeyen anahtar "
                        f"(yanlış model/görev/kol?): {report['unexpected'][:3]}")
    if report["invalid_records"]:
        blockers.append(f"{len(report['invalid_records'])} şema ihlali: "
                        f"{list(report['invalid_records'])[:3]}")
    if report["missing"] and not allow_missing:
        blockers.append(f"{len(report['missing'])} eksik koşu — ana analiz eksik veri "
                        f"üzerinde çalışmaz: {report['missing'][:3]}")
    report["blockers"] = blockers
    report["preliminary"] = bool(report["missing"]) and allow_missing
    return report


def validate_calls(calls: list[dict], records: list[dict], manifest: dict, *,
                   allow_missing_usage: bool = False) -> dict:
    """Çağrı logunu sonuç kayıtlarına karşı denetler.

    Kapatılan sessiz hata: `llm_calls.jsonl` hiç yoksa ya da eksikse maliyet
    toplamı 0 çıkar ve "kullanım verisi yok" ile "kullanım gerçekten sıfır"
    ayırt edilemez. Bu, hiç ölçülmemiş bir maliyet farkını "fark yok" diye
    raporlamaya yol açar. Bu yüzden: tamamlanmış her sonuç koşusunun EN AZ BİR
    `status="ok"` çağrısı olmalıdır.

    Ayrıca:
    - `run_id=None` çağrı sessizce atılmaz (kimliği bilinmeyen maliyet).
    - Bilinmeyen `status` değeri başarılı SAYILMAZ; analiz durur.
    - Çağrı logunun `schema_version`'ı doğrulanır (eski format = farklı alan
      anlamları).
    - Çağrının model/arm/task_id/repeat alanları bağlandığı sonuç kaydıyla
      çelişiyorsa durulur (yanlış koşuya atfedilen maliyet).
    - Sonuçla eşleşmeyen ama geçerli `run_id` taşıyan çağrılar SORUN DEĞİL;
      "altyapı overhead" olarak raporlanır.
    """
    expected_schema = manifest.get("llm_call_schema_version", LLM_CALL_SCHEMA_VERSION)
    by_run = {r["run_id"]: r for r in records}
    completed = {r["run_id"] for r in records if not is_run_error(r)}

    problems, conflicts = [], []
    null_run_ids = 0
    bad_schema, bad_status = Counter(), Counter()
    runs_with_ok = set()

    for call in calls:
        if call.get("schema_version") != expected_schema:
            bad_schema[repr(call.get("schema_version"))] += 1
        status = call.get("status")
        if status not in CALL_STATUSES:
            bad_status[repr(status)] += 1
        run_id = call.get("run_id")
        if run_id is None:
            null_run_ids += 1
            continue
        record = by_run.get(run_id)
        if record is not None:
            for field in ("model", "arm", "task_id", "repeat"):
                if field in call and call[field] != record[field]:
                    conflicts.append(f"{run_id}: çağrı {field}={call[field]!r}, "
                                     f"sonuç kaydı {record[field]!r}")
        if status == "ok":
            runs_with_ok.add(run_id)

    if bad_schema:
        problems.append(f"çağrı logu şema sürümü uyuşmuyor (beklenen "
                        f"{expected_schema!r}): {dict(bad_schema)}")
    if bad_status:
        problems.append(f"bilinmeyen çağrı status değeri (başarılı sayılmaz): "
                        f"{dict(bad_status)}")
    if null_run_ids:
        problems.append(f"{null_run_ids} çağrıda run_id yok — hangi koşuya ait "
                        "olduğu bilinmeyen maliyet sessizce atılamaz")
    if conflicts:
        problems.append(f"{len(conflicts)} çağrı bağlandığı sonuç kaydıyla çelişiyor: "
                        f"{conflicts[:3]}")

    without_usage = sorted(completed - runs_with_ok)
    if without_usage and not allow_missing_usage:
        problems.append(
            f"{len(without_usage)}/{len(completed)} tamamlanmış koşunun başarılı LLM "
            f"çağrı kaydı yok — eksik kullanım verisi SIFIR MALİYET gibi görünürdü: "
            f"{without_usage[:3]}. Yalnız performans analizi isteniyorsa "
            "--allow-missing-usage.")

    return {"problems": problems, "call_count": len(calls),
            "completed_runs": len(completed),
            "runs_without_usage": without_usage,
            "complete": not without_usage}


# --- Görev düzeyi oranlar -----------------------------------------------------

def task_level_rates(records: list[dict], *, task_ids: list[str], arms: list[str],
                     metric: str) -> dict[str, dict[str, dict]]:
    """{task_id: {arm: {pass_count, repeat_count, rate}}}.

    `run_error` kayıtları SAYILMAZ (ne pay ne payda): altyapı hatası ne başarı
    ne başarısızlıktır (§7). Bir koşu hata alıp yeniden denendiğinde dosyada iki
    kayıt bulunur; run_error atlandığı için repeat_count yine 3 kalır — aksi
    halde payda şişer ve o görevin oranı sistematik olarak DÜŞER.
    """
    if metric not in ANALYSIS_METRICS:
        raise AnalysisError(f"bilinmeyen metrik: {metric!r} (izinli: {ANALYSIS_METRICS})")
    rates = {t: {a: {"pass_count": 0, "repeat_count": 0, "rate": None}
                 for a in arms} for t in task_ids}
    for r in records:
        if is_run_error(r):
            continue
        cell = rates.get(r["task_id"], {}).get(r["arm"])
        if cell is None:
            continue  # beklenmeyen anahtar; check_integrity zaten durdurur
        cell["repeat_count"] += 1
        cell["pass_count"] += 1 if r.get(metric) is True else 0
    for task_id in task_ids:
        for arm in arms:
            cell = rates[task_id][arm]
            if cell["repeat_count"]:
                cell["rate"] = cell["pass_count"] / cell["repeat_count"]
    return rates


def paired_diffs(rates: dict, arm_a: str, arm_b: str,
                 task_ids: list[str]) -> list[float]:
    """Görev başına eşleştirilmiş fark (arm_a - arm_b), görev sırasına göre.

    Eşleştirme görev üzerindedir: aynı görevin iki koldaki oranı karşılaştırılır,
    böylece görev zorluğu farkı estimand'dan düşer.
    """
    diffs = []
    for task_id in task_ids:
        a = rates[task_id][arm_a]["rate"]
        b = rates[task_id][arm_b]["rate"]
        if a is None or b is None:
            raise AnalysisError(
                f"{task_id}: {arm_a} veya {arm_b} kolunda tamamlanmış kayıt yok — "
                "eşleştirilmiş fark hesaplanamaz.")
        diffs.append(a - b)
    return diffs


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _percentile(sorted_values: list[float], q: float) -> float:
    """Lineer interpolasyonlu yüzdelik (numpy'siz — ek bağımlılık yok)."""
    if not sorted_values:
        raise AnalysisError("boş dağılımda yüzdelik hesaplanamaz")
    pos = q * (len(sorted_values) - 1)
    low = int(pos)
    high = min(low + 1, len(sorted_values) - 1)
    frac = pos - low
    return sorted_values[low] * (1 - frac) + sorted_values[high] * frac


def bootstrap_draws(diffs: list[float], *, iterations: int, seed: int) -> list[float]:
    """Cluster (görev) bootstrap dağılımı.

    Örnekleme birimi GÖREVDİR: her yinelemede `len(diffs)` görev replacement ile
    seçilir. `diffs[i]` zaten i. görevin BÜTÜN kol/tekrar kümesinden türetilmiş
    tek bir sayı olduğu için, görevi seçmek onun bütün gözlemlerini birlikte
    taşımaya denktir (kümenin içinden ayrı ayrı örnekleme YAPILMAZ).

    Seed config'te dondurulmuştur; aynı seed + aynı veri aynı dağılımı verir.
    """
    if not diffs:
        raise AnalysisError("bootstrap için en az bir görev gerekir")
    rng = random.Random(seed)
    n = len(diffs)
    return [_mean([diffs[rng.randrange(n)] for _ in range(n)])
            for _ in range(iterations)]


def bootstrap_ci(diffs: list[float], *, iterations: int = BOOTSTRAP_ITERATIONS,
                 seed: int = BOOTSTRAP_SEED,
                 ci_level: float = BOOTSTRAP_CI_LEVEL) -> dict:
    draws = sorted(bootstrap_draws(diffs, iterations=iterations, seed=seed))
    alpha = (1 - ci_level) / 2
    return {"ci_low": _percentile(draws, alpha),
            "ci_high": _percentile(draws, 1 - alpha),
            "iterations": iterations, "seed": seed, "ci_level": ci_level}


def paired_effect(rates: dict, arm_a: str, arm_b: str, task_ids: list[str], *,
                  iterations: int = BOOTSTRAP_ITERATIONS,
                  seed: int = BOOTSTRAP_SEED) -> dict:
    """Nokta tahmini + cluster-bootstrap CI. p-değeri YOK (§8, ön-kayıtlı)."""
    diffs = paired_diffs(rates, arm_a, arm_b, task_ids)
    effect = {
        "arm_a": arm_a, "arm_b": arm_b,
        "n_tasks": len(diffs),
        "mean_rate_a": _mean([rates[t][arm_a]["rate"] for t in task_ids]),
        "mean_rate_b": _mean([rates[t][arm_b]["rate"] for t in task_ids]),
        "point_estimate": _mean(diffs),
        "tasks_favoring_a": sum(d > 0 for d in diffs),
        "tasks_favoring_b": sum(d < 0 for d in diffs),
        "tasks_tied": sum(d == 0 for d in diffs),
    }
    effect.update(bootstrap_ci(diffs, iterations=iterations, seed=seed))
    return effect


# --- LLM kullanımı ------------------------------------------------------------

def compliance_summary(records: list[dict], arms: list[str]) -> dict:
    """RQ2/RQ3 uyum metrikleri — mevcut alanlardan türetilir, yeni alan gerekmez.

    Birim ARM-RUN'dır (görev değil): bunlar süreç betimleyicileridir, birincil
    estimand değil. Bu yüzden ayrı `unit` alanıyla etiketlenir — görev-düzeyi
    başarı oranıyla aynı tabloda karıştırılmasınlar.

    Tanımlar (contract grafında ikinci denemeye YALNIZ doğrulama başarısızsa
    geçilir; dolayısıyla `attempt_count > 1` ilk denemenin uyumsuz olduğunu
    zaten kanıtlar — ayrı bir history alanına gerek yok):

        first_attempt_valid  = attempt_count == 1 and handoff_validation.valid
        final_valid          = handoff_validation.valid
        retry_recovered      = attempt_count > 1 and final_valid
        validation_exhausted = attempt_count == MAX_PLANNER_ATTEMPTS
                               and not final_valid

    `retry_recovery_rate`ın paydası TÜM koşular değil, fiilen retry'a giren
    koşulardır: "retry işe yaradı mı" sorusunun doğru paydası budur.
    """
    out = {}
    for arm in arms:
        if arm not in STRUCTURED_MODES:
            continue
        recs = [r for r in records if r["arm"] == arm and not is_run_error(r)]
        n = len(recs)
        parse_ok = sum(r.get("handoff_parse_ok") is True for r in recs)
        entry = {"unit": "arm_run", "runs": n,
                 "parse_ok_count": parse_ok,
                 "parse_ok_rate": parse_ok / n if n else None}
        if arm == ARM_CONTRACT:
            valid = [bool((r.get("handoff_validation") or {}).get("valid")) for r in recs]
            attempts = [r["attempt_count"] for r in recs]
            first = sum(a == 1 and v for a, v in zip(attempts, valid))
            final = sum(valid)
            retried = sum(a > 1 for a in attempts)
            recovered = sum(a > 1 and v for a, v in zip(attempts, valid))
            exhausted = sum(a == MAX_PLANNER_ATTEMPTS and not v
                            for a, v in zip(attempts, valid))
            entry.update({
                "first_attempt_valid_count": first,
                "first_attempt_compliance": first / n if n else None,
                "final_valid_count": final,
                "final_compliance": final / n if n else None,
                "mean_attempt_count": sum(attempts) / n if n else None,
                "attempt_count_distribution": {str(k): attempts.count(k)
                                               for k in sorted(set(attempts))},
                "retried_count": retried,
                "retry_recovered_count": recovered,
                "retry_recovery_rate": recovered / retried if retried else None,
                "validation_exhausted_count": exhausted,
                "validation_exhaustion_rate": exhausted / n if n else None,
                "max_planner_attempts": MAX_PLANNER_ATTEMPTS,
            })
        out[arm] = entry
    return out


def base_plus_attrition(records: list[dict], arms: list[str]) -> dict:
    """RQ5 §2 üçüncü sorusu: base geçen çözümlerin ne kadarı Plus'ta eleniyor?

    Birim ARM-RUN'dır, görev DEĞİL — bu bir betimleyicidir, ön-kayıtlı estimand
    değil; görev-düzeyi oranlarla aynı tabloya karışmasın diye `unit` alanıyla
    etiketlenir.

    Payda BİLİNÇLİ olarak `base_pass=True` koşulardır: "elenme" ancak base'i
    geçmiş bir çözüm için tanımlıdır. Bütün koşulara bölünseydi, base'te zaten
    başarısız olan koşular oranı aşağı çeker ve elenme az görünürdü.

    `run_error` ne paya ne paydaya girer (§7). `base_pass=False` iken
    `plus_pass=True` kaydı şema düzeyinde zaten reddedilir, dolayısıyla
    `plus_pass_runs <= base_pass_runs` invarianti veriyle garanti altındadır.
    """
    def block(subset: list[dict]) -> dict:
        base_ok = [r for r in subset if r.get("base_pass") is True]
        plus_ok = sum(r.get("plus_pass") is True for r in base_ok)
        dropped = len(base_ok) - plus_ok
        return {"unit": "arm_run", "runs": len(subset),
                "base_pass_runs": len(base_ok), "plus_pass_runs": plus_ok,
                "base_pass_plus_fail_runs": dropped,
                "attrition_rate": dropped / len(base_ok) if base_ok else None}

    scored = [r for r in records if not is_run_error(r)]
    return {"overall": block(scored),
            "by_arm": {arm: block([r for r in scored if r["arm"] == arm])
                       for arm in arms}}


def usage_by_run(calls: list[dict], records: list[dict]) -> list[dict]:
    """Çağrı logunu sonuçlara `run_id` ile bağlar; arm-run başına kullanım.

    Satırlar SONUÇ kayıtlarından tohumlanır (çağrı logundan değil): bir koşunun
    hiç çağrı kaydı yoksa satır yine üretilir ve `usage_missing=True` olur.
    Aksi halde o koşu tabloda hiç görünmez ve eksik veri "sıfır maliyet" gibi
    okunur.

    İki çift-sayma tuzağı bilinçli olarak kapatıldı:

    - **Gecikme:** başarılı çağrının `latency_s` değeri retry ve backoff süresini
      ZATEN içerir (agents/llm.py sayacı ilk denemeden önce başlar). Bozuk
      denemelerin gecikmesi bunun üstüne eklenmez.
    - **Maliyet/token:** yalnız `status="ok"` kayıtlarından toplanır. Bozuk
      denemeler zaten faturalanmaz (usage.cost=0) ama toplanmaları hâlinde
      "sözleşme kolu daha pahalı" gibi yapay bir fark üretirlerdi.
    """
    def _row(run_id, **identity) -> dict:
        row = {"run_id": run_id, "model": None, "arm": None, "task_id": None,
               "repeat": None, "call_records": 0, "successful_calls": 0,
               "provider_error_attempts": 0, "transport_error_calls": 0,
               "error_latency_s": 0.0,
               "in_results": False, "is_run_error": False, "usage_missing": True}
        row.update({f: 0 for f in USAGE_NUMERIC_FIELDS})
        row.update(identity)
        return row

    rows: dict[str, dict] = {}
    # Kimlik otoritesi SONUÇ kaydıdır, çağrı logu değil.
    for record in records:
        rows[record["run_id"]] = _row(
            record["run_id"], model=record["model"], arm=record["arm"],
            task_id=record["task_id"], repeat=record["repeat"],
            in_results=True, is_run_error=is_run_error(record))

    for call in calls:
        run_id = call.get("run_id")
        if run_id is None:
            continue  # validate_calls zaten durdurur; toplamlara karıştırılmaz
        row = rows.get(run_id)
        if row is None:  # yetim çağrı: sonuç kaydı yok ama kimliği var
            row = rows[run_id] = _row(run_id, model=call.get("model"),
                                      arm=call.get("arm"), task_id=call.get("task_id"),
                                      repeat=call.get("repeat"))
        row["call_records"] += 1
        status = call.get("status")
        if status == "provider_error":
            row["provider_error_attempts"] += 1
            continue
        if status != "ok":
            row["transport_error_calls"] += 1
            # Terminal taşıma hatasının süresi AYRI alanda: bu çağrı hiç ok
            # kaydı üretmediği için başka hiçbir yerde sayılmıyor, ama
            # başarılı çağrı gecikmesiyle aynı sütunda toplanırsa "ölçümün
            # gecikmesi" ile "boşa geçen süre" karışırdı.
            row["error_latency_s"] += call.get("latency_s") or 0.0
            continue
        row["successful_calls"] += 1
        for field in ("input_tokens", "output_tokens", "reasoning_tokens", "cached_tokens"):
            row[field] += call.get(field) or 0
        row["cost_usd"] += call.get("cost_usd") or 0.0
        row["latency_s"] += call.get("latency_s") or 0.0

    for row in rows.values():
        # TAMAMLANMIŞ bir koşunun kullanım ölçütü "en az bir BAŞARILI çağrı"dır
        # (validate_calls ile aynı ölçüt). Yalnız hata kayıtları bulunan bir
        # koşu, call_records>0 olduğu için "veri var" sanılıp maliyet 0
        # raporlanırdı. run_error ve yetim satırlarda ise başarılı çağrı
        # bulunmaması normaldir; orada ölçüt "hiç kayıt yok".
        if row["in_results"] and not row["is_run_error"]:
            row["usage_missing"] = row["successful_calls"] == 0
        else:
            row["usage_missing"] = row["call_records"] == 0
        if row["usage_missing"]:
            row.update({f: None for f in USAGE_NUMERIC_FIELDS})
        else:
            row["cost_usd"] = round(row["cost_usd"], 8)
            row["latency_s"] = round(row["latency_s"], 3)
        row["error_latency_s"] = round(row["error_latency_s"], 3)
    return [rows[k] for k in sorted(rows)]


def _describe(values: list[float], digits: int) -> dict:
    """Ortalama + medyan + IQR + uzun kuyruk (n, p95, gözlenen maksimum).

    Maliyet ve gecikme sağa çarpıktır; yalnız toplam/ortalama dağılımı gizler.
    IQR gövdeyi anlatır ama KUYRUĞU anlatmaz: 2026-07-30 sağlık koşusunda aynı
    rotada medyan ~16 s iken gözlenen maksimum ~167 s çıktı (§26). Rate-limit'e
    duyarlı bir koşuda "en kötü hâl ne kadar sürer" sorusunun cevabı p95 ve
    gözlenen maksimumdadır.

    Alan sözleşmesi:
    - `median` ile `p50` AYNI ölçüdür (median = P50); geriye uyumluluk için
      alan adı `median` korunur, ikinci bir kopya yazılmaz.
    - `p95`/`median` yöntemi: SIRALI örnek üzerinde LİNEER İNTERPOLASYON
      (`_percentile`, R-7 / numpy varsayılanı), `pos = q*(n-1)`. Deterministik
      ve bağımlılıksızdır; aynı girdi her zaman aynı sayıyı verir.
    - `observed_max` bir tahmin edici DEĞİL, gözlenen tek bir uç değerdir; n
      küçükken oynaktır, bu yüzden `n` ile birlikte raporlanır.

    Hepsi BETİMLEYİCİDİR: eşik, hipotez testi veya p-değeri üretmez.
    """
    if not values:
        return {"n": 0, "mean": None, "median": None, "q1": None, "q3": None,
                "iqr": None, "p95": None, "observed_max": None}
    ordered = sorted(values)
    q1, q3 = _percentile(ordered, 0.25), _percentile(ordered, 0.75)
    return {"n": len(ordered),
            "mean": round(_mean(ordered), digits),
            "median": round(_percentile(ordered, 0.5), digits),
            "q1": round(q1, digits), "q3": round(q3, digits),
            "iqr": round(q3 - q1, digits),
            "p95": round(_percentile(ordered, 0.95), digits),
            "observed_max": round(ordered[-1], digits)}


def usage_summary(rows: list[dict], arms: list[str]) -> dict:
    """Kol başına kullanım + AYRI "altyapı overhead" tablosu.

    `run_error` alan koşuların maliyeti başarı estimand'ına KATILMAZ (§7): o
    çağrılar bir ölçüm üretmedi. Şeffaflık için ayrı raporlanır — "deneyin
    gerçek parasal maliyeti" ile "bir ölçümün maliyeti" farklı sorulardır.

    Bir blokta kullanım verisi eksik satır varsa o bloğun token/maliyet/gecikme
    toplamları **None** olur, 0 DEĞİL: sıfır yazmak "ölçüldü ve sıfır çıktı"
    demek olurdu.
    """
    scored = [r for r in rows if r["in_results"] and not r["is_run_error"]]
    overhead = [r for r in rows if not r["in_results"] or r["is_run_error"]]

    def block(subset: list[dict]) -> dict:
        eksik = [r for r in subset if r["usage_missing"]]
        out = {
            "runs": len(subset),
            "runs_missing_usage": len(eksik),
            "complete": not eksik,
            "successful_calls": sum(r["successful_calls"] for r in subset),
            "provider_error_attempts": sum(r["provider_error_attempts"] for r in subset),
            "transport_error_calls": sum(r["transport_error_calls"] for r in subset),
            # Terminal taşıma hatalarında boşa geçen süre. ALT SINIR: bütün
            # denemeleri tükenip ProviderResponseError atan çağrılar terminal
            # bir kayıt yazmadığı (yalnız deneme başına provider_error kaydı
            # bırakır) için bu toplama girmez.
            "error_latency_s": round(sum(r["error_latency_s"] for r in subset), 2),
            "error_latency_is_lower_bound": True,
        }
        if eksik:
            out.update({f: None for f in USAGE_NUMERIC_FIELDS})
            out.update({"cost_usd_per_run": None, "latency_s_per_run": None,
                        "total_tokens_per_run": None})
            return out
        out.update({f: sum(r[f] for r in subset) for f in USAGE_NUMERIC_FIELDS})
        out["cost_usd"] = round(out["cost_usd"], 6)
        out["latency_s"] = round(out["latency_s"], 2)
        # Toplamların yanında arm-run başına dağılım: sağa çarpık maliyet/gecikme
        # yalnız toplamla raporlanırsa birkaç uç koşu farkı sürükleyebilir.
        out["cost_usd_per_run"] = _describe([r["cost_usd"] for r in subset], 6)
        out["latency_s_per_run"] = _describe([r["latency_s"] for r in subset], 2)
        out["total_tokens_per_run"] = _describe(
            [r["input_tokens"] + r["output_tokens"] for r in subset], 1)
        return out

    scored_block = block(scored)
    return {
        "complete": scored_block["complete"],
        "scored_runs": scored_block,
        "by_arm": {arm: block([r for r in scored if r["arm"] == arm]) for arm in arms},
        # Ölçüm üretmemiş çağrılar: run_error'lar + sonuç kaydıyla eşleşmeyenler.
        "infrastructure_overhead": block(overhead),
    }


# --- Üst düzey ----------------------------------------------------------------

def analyze(manifest: dict, records: list[dict], calls: list[dict], *,
            allow_missing: bool = False, allow_missing_usage: bool = False,
            iterations: int = BOOTSTRAP_ITERATIONS,
            seed: int = BOOTSTRAP_SEED) -> dict:
    """Bütün analizi tek sözlükte üretir (CSV/JSON yazımı ayrı adımda).

    Kapı sırası bilinçli: önce "bu kayıtlar doğru deneye mi ait" (provenance),
    sonra "eksiksiz mi" (bütünlük), sonra "maliyet verisi var mı" (çağrı logu).
    Ters sırada, yanlış deneyin verisi üzerinde eksiklik raporlanırdı.
    """
    if not records:
        raise AnalysisError("sonuç kaydı yok.")
    model = single_model(records)
    provenance = check_provenance(records, manifest)
    if provenance:
        raise AnalysisError("analiz durduruldu (provenance):\n  - " + "\n  - ".join(provenance))

    integrity = check_integrity(records, manifest, allow_missing=allow_missing)
    if integrity["blockers"]:
        raise AnalysisError("analiz durduruldu:\n  - " + "\n  - ".join(integrity["blockers"]))

    call_check = validate_calls(calls, records, manifest,
                                allow_missing_usage=allow_missing_usage)
    if call_check["problems"]:
        raise AnalysisError("analiz durduruldu (çağrı logu):\n  - "
                            + "\n  - ".join(call_check["problems"]))

    task_ids, arms = manifest["task_ids"], manifest["arm_order"]
    comparisons = [tuple(PRIMARY_COMPARISON)] + [tuple(c) for c in SECONDARY_COMPARISONS]

    rates = {m: task_level_rates(records, task_ids=task_ids, arms=arms, metric=m)
             for m in ANALYSIS_METRICS}
    effects = {
        m: {f"{a}_vs_{b}": paired_effect(rates[m], a, b, task_ids,
                                         iterations=iterations, seed=seed)
            for a, b in comparisons if a in arms and b in arms}
        for m in ANALYSIS_METRICS
    }
    arm_summary = {
        arm: {m: {"task_mean_rate": _mean([rates[m][t][arm]["rate"] for t in task_ids]),
                  "pass_count": sum(rates[m][t][arm]["pass_count"] for t in task_ids),
                  "observation_count": sum(rates[m][t][arm]["repeat_count"] for t in task_ids)}
              for m in ANALYSIS_METRICS}
        for arm in arms
    }
    usage_rows = usage_by_run(calls, records)
    return {
        "analysis_schema_version": ANALYSIS_SCHEMA_VERSION,
        "experiment": manifest.get("name"),
        "model": model,
        "task_set": manifest.get("task_set"),
        "result_schema_version": manifest.get("result_schema_version"),
        "git_commit": manifest.get("git_commit"),
        "analysis_unit": "task",
        "primary_metric": PRIMARY_METRIC,
        "primary_comparison": f"{PRIMARY_COMPARISON[0]}_vs_{PRIMARY_COMPARISON[1]}",
        "n_tasks": len(task_ids),
        "repeats": manifest["repeats"],
        "arms": arms,
        "preliminary": integrity["preliminary"] or not call_check["complete"],
        "integrity": integrity,
        "call_log_check": call_check,
        "arm_summary": arm_summary,
        "paired_effects": effects,
        "base_plus_attrition": base_plus_attrition(records, arms),
        "compliance": compliance_summary(records, arms),
        "usage": usage_summary(usage_rows, arms),
        "_task_rates": rates,
        "_usage_rows": usage_rows,
    }


def _write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(result: dict, out_dir: Path) -> dict[str, Path]:
    """Makine-okunur çıktılar. Notebook bunları OKUR, yeniden hesaplamaz.

    Hiçbirinde zaman damgası yok: aynı seed + aynı veri bayt-bayt aynı dosyaları
    üretmeli (testle güvence altında).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    rates, usage_rows = result["_task_rates"], result["_usage_rows"]

    task_rows = []
    for task_id in sorted({t for m in rates for t in rates[m]}):
        for arm in result["arms"]:
            row = {"model": result["model"], "task_id": task_id, "arm": arm}
            for metric in ANALYSIS_METRICS:
                cell = rates[metric][task_id][arm]
                prefix = metric.replace("_pass", "")
                row[f"{prefix}_pass_count"] = cell["pass_count"]
                row[f"{prefix}_repeat_count"] = cell["repeat_count"]
                row[f"{prefix}_rate"] = cell["rate"]
            task_rows.append(row)
    task_columns = ["model", "task_id", "arm"]
    for metric in ANALYSIS_METRICS:
        prefix = metric.replace("_pass", "")
        task_columns += [f"{prefix}_pass_count", f"{prefix}_repeat_count", f"{prefix}_rate"]

    usage_columns = ["run_id", "model", "arm", "task_id", "repeat", "call_records",
                     "successful_calls", "provider_error_attempts",
                     "transport_error_calls", "input_tokens", "output_tokens",
                     "reasoning_tokens", "cached_tokens", "cost_usd", "latency_s",
                     "error_latency_s", "in_results", "is_run_error", "usage_missing"]

    summary = {k: v for k, v in result.items() if not k.startswith("_")}
    paths = {
        "task_level": out_dir / "task_level.csv",
        "paired_effects": out_dir / "paired_effects.json",
        "usage_by_run": out_dir / "usage_by_run.csv",
        "integrity_report": out_dir / "integrity_report.json",
        "analysis_summary": out_dir / "analysis_summary.json",
    }
    _write_csv(paths["task_level"], task_rows, task_columns)
    _write_csv(paths["usage_by_run"], usage_rows, usage_columns)
    for key, payload in (("paired_effects", result["paired_effects"]),
                         ("integrity_report", result["integrity"]),
                         ("analysis_summary", summary)):
        paths[key].write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Görev-düzeyi analiz + cluster bootstrap")
    parser.add_argument("--exp", required=True, type=Path,
                        help="deney dizini (logs/exp_<name>)")
    parser.add_argument("--out", type=Path, default=None,
                        help="çıktı dizini (varsayılan: <exp>/analysis)")
    parser.add_argument("--allow-missing", action="store_true",
                        help="YALNIZ eksik koşuyu mazur görür; çıktı preliminary "
                             "damgalanır. Yinelenen/beklenmeyen/geçersiz kayıt asla.")
    parser.add_argument("--allow-missing-usage", action="store_true",
                        help="Kullanım verisi eksikken yalnız performans analizi. "
                             "Maliyet alanları null (sıfır DEĞİL), çıktı preliminary.")
    parser.add_argument("--iterations", type=int, default=BOOTSTRAP_ITERATIONS)
    parser.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    args = parser.parse_args()

    manifest, records, calls = load_experiment(args.exp)
    try:
        result = analyze(manifest, records, calls, allow_missing=args.allow_missing,
                         allow_missing_usage=args.allow_missing_usage,
                         iterations=args.iterations, seed=args.seed)
    except AnalysisError as e:
        sys.exit(str(e))

    paths = write_outputs(result, args.out or args.exp / "analysis")
    print(f"deney: {result['experiment']} | model: {result['model']} | "
          f"{result['n_tasks']} görev × {result['repeats']} tekrar × {len(result['arms'])} kol")
    if result["preliminary"]:
        print("UYARI: çıktı PRELIMINARY damgalı; doğrulayıcı rapora girmez.")
        if result["integrity"]["preliminary"]:
            print(f"  eksik koşu: {len(result['integrity']['missing'])}")
    print(f"\nkol ortalamaları (görev-düzeyi, {PRIMARY_METRIC}):")
    for arm in result["arms"]:
        s = result["arm_summary"][arm][PRIMARY_METRIC]
        print(f"  {arm:26s} {s['task_mean_rate']:.3f}  "
              f"({s['pass_count']}/{s['observation_count']} gözlem)")
    print(f"\neşleştirilmiş etkiler ({PRIMARY_METRIC}, %"
          f"{int(BOOTSTRAP_CI_LEVEL * 100)} cluster-bootstrap CI):")
    for name, e in result["paired_effects"][PRIMARY_METRIC].items():
        print(f"  {name:38s} {e['point_estimate']:+.3f}  "
              f"[{e['ci_low']:+.3f}, {e['ci_high']:+.3f}]  "
              f"(lehte {e['tasks_favoring_a']} / aleyhte {e['tasks_favoring_b']} / "
              f"berabere {e['tasks_tied']})")
    print("\nbase -> plus elenmesi (birim: arm-run, betimleyici):")
    for arm in result["arms"]:
        a = result["base_plus_attrition"]["by_arm"][arm]
        print(f"  {arm:26s} {a['base_pass_plus_fail_runs']}/{a['base_pass_runs']} "
              f"base-geçen koşu plus'ta elendi (oran {a['attrition_rate']})")

    if result["compliance"]:
        print("\nsözleşme uyumu (birim: arm-run):")
        for arm, c in result["compliance"].items():
            satir = f"  {arm:26s} parse {c['parse_ok_rate']}"
            if arm == ARM_CONTRACT:
                satir += (f" | ilk deneme {c['first_attempt_compliance']} "
                          f"| nihai {c['final_compliance']} "
                          f"| ort. deneme {c['mean_attempt_count']} "
                          f"| retry kurtarma {c['retry_recovery_rate']} "
                          f"| exhaustion {c['validation_exhaustion_rate']}")
            print(satir)

    u = result["usage"]["scored_runs"]
    o = result["usage"]["infrastructure_overhead"]
    if not result["usage"]["complete"]:
        print(f"\nUYARI: {u['runs_missing_usage']} koşuda kullanım verisi YOK — "
              "maliyet/token/gecikme null raporlanır (sıfır değil).")
    print(f"\nölçüm üreten koşular: {u['runs']} koşu, {u['successful_calls']} çağrı, "
          f"${u['cost_usd']}")
    if u["complete"]:
        print(f"  koşu başına maliyet: ort {u['cost_usd_per_run']['mean']} | "
              f"medyan {u['cost_usd_per_run']['median']} | "
              f"IQR {u['cost_usd_per_run']['iqr']}")
        gec = u["latency_s_per_run"]
        print(f"  koşu başına gecikme: ort {gec['mean']}s | "
              f"medyan(P50) {gec['median']}s | IQR {gec['iqr']}s")
        print(f"    uzun kuyruk (n={gec['n']}): p95 {gec['p95']}s | "
              f"gözlenen maks {gec['observed_max']}s")
    print(f"altyapı overhead (ölçüm üretmeyen): {o['runs']} koşu, ${o['cost_usd']}")
    print("\nÇıktılar:")
    for name, path in paths.items():
        print(f"  {name:18s} -> {path}")


if __name__ == "__main__":
    main()
