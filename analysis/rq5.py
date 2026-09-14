"""RQ5 keşifsel bağlantı: görev-düzeyi self-consistency agreement ↔ başarısızlık.

EXPERIMENT_PROTOCOL.md §8. Bu modül ANA analizden (analysis/analyze.py)
bilinçli olarak AYRIDIR: ana performans analizi self-consistency girdisi HİÇ YOKKEN
de çalışabilmelidir, aksi halde keşifsel bir yan soru birincil estimand'ı rehin
alırdı.

**Ön-kayıt sınırları (sonuç görülmeden dondurulmuştur):**

1. **Birim (model, task_id)**, n = held-out görev sayısı. Modeller ASLA
   havuzlanmaz — Gemini ve DeepSeek ayrı deney dizinleri, ayrı çıktı dosyalarıdır
   (§8.4). Havuzlama, RQ4'ün sorduğu "etki model ailesine göre değişiyor mu"
   sorusunu ortadan kaldırır.
2. **Birincil keşifsel eşleştirme: agreement ↔ `baseline` kolunun üç tekrardaki
   Plus başarısızlık oranı** (`config.RQ5_PRIMARY_ARM`). Gerekçe yapısaldır,
   sonuçlara bakılarak seçilmemiştir: `uncertainty/self_consistency.py` planner ya
   da contract katmanı KULLANMAZ, doğrudan `pipeline.baseline.SYSTEM_PROMPT` ile
   tek kodlayıcı adayı üretir. Yani agreement'ın kavramsal eşi baseline kolunun
   başarısızlığıdır.
3. Diğer üç kol, 12 arm-run üzerinden genel görev başarısızlık oranı ve
   self-consistency turunun KENDİ oracle başarısızlık oranı yalnız
   **İKİNCİL/betimleyici** olarak raporlanır.
4. **P-değeri ÜRETİLMEZ** (§8.5: RQ5 keşifseldir). Ana analiz modülüyle aynı
   politika; rho, eşleşen görev sayısı ve bağ (tie) davranışı raporlanır.
5. Eksik/fazla görev, model/task-set/görev-hash uyuşmazlığı, eski veya
   provenance'sız self-consistency artefaktı **fail-fast**tır.

**Bilinçli olarak KARŞILAŞTIRILMAYAN alan:** `heldout_selection_fingerprint`.
Ana runner ile self-consistency hattı aynı seçim manifestini farklı JSON
serileştirmesiyle hash'ler; dizeler eşit değildir ama içerik aynıdır. Görev
kimliği bunun yerine
`task_ids` + `task_file_hashes` üzerinden birebir doğrulanır — bu, fingerprint'ten
daha güçlü bir kontroldür.

**Sınırlama:** self-consistency adayları `temperature=0.8`
ile, baseline kolu `config.DEFAULT_TEMPERATURE` ile üretilir. Yani agreement,
baseline kolunun kendi örnekleme dağılımının doğrudan ölçümü DEĞİL, aynı
prompt/model altındaki görev-düzeyi belirsizliğin bir proxy'sidir.

Kullanım:
    uv run python -m analysis.rq5 --exp logs/exp_gemini_main \\
        --selfcons logs/exp_gemini_selfcons_formal
"""

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

from config import (
    RQ5_PRIMARY_ARM,
    RQ5_SCHEMA_VERSION,
    RQ5_TIE_METHOD,
    SELF_CONSISTENCY_SCHEMA_VERSION,
)
from analysis.analyze import AnalysisError, load_experiment, single_model, task_level_rates
from eval.result_schema import check_provenance, integrity_report
from uncertainty.self_consistency import (
    CALL_FILE,
    CANDIDATE_FILE,
    MANIFEST_FILE,
    RESULT_FILE,
    SelfConsistencyError,
    expected_candidate_keys,
    load_jsonl,
    verify_call_records,
    verify_candidate_records,
    verify_result_records,
)

__all__ = [
    "Rq5Error", "average_ranks", "spearman", "load_selfcons",
    "check_selfcons_provenance", "verify_selfcons", "failure_series",
    "build_rq5", "write_rq5_outputs",
]

# Self-consistency turunun KENDİ oracle başarısızlığı: ana koşunun bir kolu
# DEĞİL, aynı üretim ayarındaki (temp 0.8, N aday) ikinci bir betimleyici seri.
SELFCONS_ORACLE_KEY = "selfcons_oracle_failure"
# 4 kol × 3 tekrar = 12 arm-run üzerinden görev başarısızlığı.
OVERALL_KEY = "overall_arm_runs"


class Rq5Error(RuntimeError):
    """RQ5 girdisi güvenle eşleştirilemiyor — yanlış eşleştirmektense durulur."""


# --- Spearman -----------------------------------------------------------------

def average_ranks(values: list[float]) -> list[float]:
    """Bağlarda ORTALAMA rank (fractional ranking).

    Bağları keyfi bir sırayla numaralandırmak (ör. ilk gelen küçük rank alır)
    veri sırasına bağlı bir rho üretirdi: aynı ölçümün satır sırası değişince
    korelasyon değişirdi. Tavan etkisi beklenen bir veri kümesinde bağlar
    istisna değil KURALDIR, bu yüzden kural açıkça sürümlenir
    (`config.RQ5_TIE_METHOD`).
    """
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        shared = (i + j) / 2 + 1  # 1 tabanlı ortalama rank
        for k in range(i, j + 1):
            ranks[order[k]] = shared
        i = j + 1
    return ranks


def _tie_group_count(values: list[float]) -> int:
    """Bağ GRUBU sayısı (aynı değeri paylaşan 2+ gözlemden oluşan küme)."""
    counts: dict[float, int] = {}
    for v in values:
        counts[v] = counts.get(v, 0) + 1
    return sum(1 for n in counts.values() if n > 1)


def spearman(x: list[float], y: list[float]) -> dict:
    """Spearman rho — bağlarda ortalama rank, rank'lerin Pearson korelasyonu.

    P-DEĞERİ ÜRETMEZ (§8.5). Sıfır varyanslı bir seride rho TANIMSIZDIR ve `None`
    döner: bu durumda "korelasyon yok (0)" yazmak yanlış olurdu — 0, ölçülmüş bir
    ilişkisizliktir; tanımsız ise ölçümün hiç yapılamadığıdır. Tavan etkisi
    altında (ör. bütün görevlerde agreement = 1.0) bu gerçekten olabilir.
    """
    if len(x) != len(y):
        raise Rq5Error(f"eşleşmeyen seri uzunlukları: {len(x)} != {len(y)}")
    n = len(x)
    out = {
        "n": n, "rho": None, "tie_method": RQ5_TIE_METHOD,
        "tied_groups_x": _tie_group_count(x), "tied_groups_y": _tie_group_count(y),
        "p_value_reported": False, "undefined_reason": None,
    }
    if n < 3:
        out["undefined_reason"] = f"en az 3 eşleşen görev gerekir (n={n})"
        return out
    rx, ry = average_ranks(x), average_ranks(y)
    mx, my = sum(rx) / n, sum(ry) / n
    dx = [v - mx for v in rx]
    dy = [v - my for v in ry]
    sxx = sum(v * v for v in dx)
    syy = sum(v * v for v in dy)
    if sxx == 0 or syy == 0:
        out["undefined_reason"] = (
            "sıfır varyans: " + ", ".join(
                ad for ad, s in (("x", sxx), ("y", syy)) if s == 0)
            + " serisinde bütün görevler aynı değeri taşıyor (rho tanımsız)")
        return out
    out["rho"] = round(sum(a * b for a, b in zip(dx, dy)) / (sxx * syy) ** 0.5, 6)
    return out


def _canonical_hash(value) -> str | None:
    """Uzun provenance yapılarının kompakt kimliği (yok ise None)."""
    if value is None:
        return None
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False,
                         separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _describe(values: list[float]) -> dict:
    if not values:
        return {"n": 0, "mean": None, "min": None, "max": None, "distinct_values": 0}
    return {"n": len(values), "mean": round(sum(values) / len(values), 6),
            "min": min(values), "max": max(values),
            "distinct_values": len(set(values))}


# --- Self-consistency girdisi -------------------------------------------------

def load_selfcons(sc_dir: Path) -> tuple[dict, list[dict], list[dict], list[dict]]:
    """(manifest, aday kayıtları, sonuç kayıtları, çağrı kayıtları).

    DÖRT dosya da yüklenir. Yalnız `results.jsonl` okumak yetmez: sonuç kaydı
    türetilmiş bir özettir (`agreement`, `oracle_pass_rate`) ve elle
    düzenlendiğinde ya da yarım bir turdan kaldığında kendi başına tutarlı
    görünür. Kaynak zinciri ancak adaylar + adayların başarılı çağrı provenance'ı
    ile birlikte doğrulanabilir.

    Dizin TAHMİN EDİLMEZ, dosya ARANMAZ: yol açıkça verilir. Eski global
    `logs/results_selfcons_*.jsonl` dosyaları formal girdi DEĞİLDİR (manifest ve
    deney-bağlı çağrı provenance'ı taşımazlar, §P2 2026-08-01 notu) — bir dosya
    yolu verildiğinde bu açıkça söylenerek durulur, sessizce okunmaz.
    """
    if sc_dir.is_file():
        raise Rq5Error(
            f"self-consistency girdisi bir DİZİN olmalı: {sc_dir}. Eski, global "
            "results_selfcons_*.jsonl dosyaları formal RQ5 girdisi değildir "
            "(manifest ve deney-bağlı çağrı provenance'ı yok).")
    manifest_path = sc_dir / MANIFEST_FILE
    if not manifest_path.exists():
        raise Rq5Error(
            f"self-consistency manifesti yok: {manifest_path} — provenance'sız bir "
            "tur formal RQ5 girdisi olamaz.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    try:
        candidates = load_jsonl(sc_dir / CANDIDATE_FILE)
        results = load_jsonl(sc_dir / RESULT_FILE)
        calls = load_jsonl(sc_dir / CALL_FILE)
    except SelfConsistencyError as exc:
        raise Rq5Error(f"self-consistency dosyası okunamadı: {exc}") from exc
    for ad, rows in ((CANDIDATE_FILE, candidates), (RESULT_FILE, results),
                     (CALL_FILE, calls)):
        if not rows:
            raise Rq5Error(
                f"self-consistency kaynak dosyası boş/yok: {sc_dir / ad} — kaynak "
                "zinciri (aday -> cagri -> sonuc) dogrulanamaz.")
    return manifest, candidates, results, calls


def check_selfcons_provenance(sc_manifest: dict, main_manifest: dict) -> list[str]:
    """İki manifestin AYNI model + AYNI görev popülasyonu olduğunu doğrular.

    §8.5: "Self-consistency, ilişkilendirildiği ana üretici modelle AYNI model ve
    görev setinde çalıştırılır." Bu kontrol olmadan bir modelin agreement'ı başka
    bir modelin başarısızlık oranıyla eşleştirilebilir ve korelasyon TAMAMEN
    anlamsız olurdu — üstelik sayısal olarak kusursuz görünürdü.

    Görev kimliği `task_ids` + `task_file_hashes` üzerinden denetlenir;
    `heldout_selection_fingerprint` BİLİNÇLİ olarak karşılaştırılmaz (modül
    docstring'i).
    """
    problems = []
    surum = sc_manifest.get("self_consistency_schema_version")
    if surum != SELF_CONSISTENCY_SCHEMA_VERSION:
        problems.append(
            f"self-consistency şema sürümü uyuşmuyor: {surum!r} != "
            f"{SELF_CONSISTENCY_SCHEMA_VERSION!r} — eski/provisional artefakt "
            "otomatik migrate edilmez.")
    for alan in ("model", "task_set"):
        if sc_manifest.get(alan) != main_manifest.get(alan):
            problems.append(
                f"{alan} uyuşmuyor: self-consistency {sc_manifest.get(alan)!r}, "
                f"ana koşu {main_manifest.get(alan)!r} — modeller HAVUZLANMAZ.")
    sc_ids, main_ids = sc_manifest.get("task_ids"), main_manifest.get("task_ids")
    if not isinstance(sc_ids, list) or not isinstance(main_ids, list):
        problems.append("iki manifestte de task_ids listesi bulunmalı")
    elif set(sc_ids) != set(main_ids) or len(set(sc_ids)) != len(sc_ids):
        eksik = sorted(set(main_ids) - set(sc_ids))
        fazla = sorted(set(sc_ids) - set(main_ids))
        problems.append(
            f"görev kümesi uyuşmuyor (eksik {eksik[:3]}, fazla {fazla[:3]}, "
            f"self-consistency {len(sc_ids)} / ana koşu {len(main_ids)} görev)")
    sc_hash = sc_manifest.get("task_file_hashes")
    main_hash = main_manifest.get("task_file_hashes")
    if not sc_hash or not main_hash:
        problems.append("iki manifestte de task_file_hashes bulunmalı")
    elif sc_hash != main_hash:
        farkli = sorted(k for k in set(sc_hash) | set(main_hash)
                        if sc_hash.get(k) != main_hash.get(k))
        problems.append(
            f"görev dosyası hash'leri uyuşmuyor ({len(farkli)} dosya, ör. "
            f"{farkli[:3]}) — aynı görev metni ölçülmemiş.")
    return problems


def verify_selfcons(sc_manifest: dict, candidates: list[dict], results: list[dict],
                    calls: list[dict]) -> dict[str, dict]:
    """Kaynak zincirini ÜRETEN hattın kendi doğrulayıcılarıyla denetler.

    `uncertainty.self_consistency` içindeki üç doğrulayıcı yeniden kullanılır —
    burada ikinci bir implementasyon YAZILMAZ; yazılsaydı iki taraf zamanla
    ayrışır ve analiz, üretim hattının reddedeceği bir turu kabul edebilirdi:

    1. `verify_candidate_records` — yabancı/yinelenen/stale aday, kimlik ve
       manifest bağlamı (`run_id` deterministik aday kimliğinden türer).
    2. `verify_call_records` — her tamamlanmış adayın deney-bağlı BAŞARILI çağrı
       provenance'ı; yabancı `run_id` ve yanlış bağlamlı (deney/model/görev/
       indeks/rol) çağrı.
    3. `verify_result_records` — her sonuç satırı adaylardan DETERMİNİSTİK olarak
       yeniden kurulur ve birebir karşılaştırılır. Elle değiştirilmiş bir
       `agreement`/`oracle_pass_rate` ya da adayı olmayan bir sonuç burada düşer.

    Ayrıca TAMLIK zorunludur (doğrulayıcılar tek başına bunu bakmaz): tam
    `n × görev` aday ve her görev için bir sonuç. Yarım bir turdan üretilen
    korelasyon, görevlerin bir kısmında ölçülmemiş bir belirsizliği "düşük
    agreement" gibi gösterirdi.
    """
    try:
        by_key = verify_candidate_records(candidates, sc_manifest)
        verify_call_records(calls, by_key, sc_manifest)
        by_task = verify_result_records(results, by_key, sc_manifest)
    except SelfConsistencyError as exc:
        raise Rq5Error(f"self-consistency kaynak zinciri doğrulanamadı: {exc}") from exc

    beklenen_aday = expected_candidate_keys(sc_manifest)
    if len(by_key) != len(beklenen_aday):
        eksik = sorted(beklenen_aday - set(by_key))
        raise Rq5Error(
            f"self-consistency turu eksik: {len(by_key)}/{len(beklenen_aday)} aday "
            f"(ör. {eksik[:3]}) — yarım turdan korelasyon üretilmez.")
    eksik_gorev = sorted(set(sc_manifest["task_ids"]) - set(by_task))
    if eksik_gorev:
        raise Rq5Error(
            f"{len(eksik_gorev)} görevin self-consistency sonucu yok "
            f"(ör. {eksik_gorev[:3]}) — eksik seride korelasyon hesaplanmaz.")
    return by_task


# --- Eşleştirme ---------------------------------------------------------------

def failure_series(rates: dict, task_ids: list[str], arm: str) -> list[float]:
    """Kol başına görev-düzeyi Plus BAŞARISIZLIK oranı (1 - başarı oranı)."""
    seri = []
    for task_id in task_ids:
        rate = rates[task_id][arm]["rate"]
        if rate is None:
            raise Rq5Error(
                f"{task_id}: {arm} kolunda tamamlanmış kayıt yok — RQ5 eşleştirmesi "
                "eksik veri üzerinde yapılmaz.")
        seri.append(1.0 - rate)
    return seri


def overall_failure_series(rates: dict, task_ids: list[str],
                           arms: list[str]) -> list[float]:
    """Bütün kolların arm-run'ları üzerinden görev başarısızlık oranı (İKİNCİL).

    Kolları tek bir sayıda topladığı için kol etkisiyle görev zorluğunu
    karıştırır; bu yüzden yalnız betimleyicidir, birincil eşleştirme değildir.
    """
    seri = []
    for task_id in task_ids:
        passes = sum(rates[task_id][a]["pass_count"] for a in arms)
        total = sum(rates[task_id][a]["repeat_count"] for a in arms)
        if not total:
            raise Rq5Error(f"{task_id}: hiçbir kolda tamamlanmış kayıt yok")
        seri.append(1.0 - passes / total)
    return seri


def build_rq5(main_manifest: dict, records: list[dict], sc_manifest: dict,
              sc_candidates: list[dict], sc_results: list[dict],
              sc_calls: list[dict]) -> dict:
    """Bütün RQ5 çıktısını tek sözlükte üretir (dosya yazımı ayrı adımda).

    Kapı sırası: önce "bu kayıtlar doğru deneye mi ait", sonra "eksiksiz mi",
    sonra "self-consistency turu aynı model/görev mi", en sonda hesap.
    """
    if not records:
        raise Rq5Error("ana koşu sonuç kaydı yok.")
    model = single_model(records)
    provenance = check_provenance(records, main_manifest)
    if provenance:
        raise Rq5Error("RQ5 durduruldu (ana koşu provenance):\n  - "
                       + "\n  - ".join(provenance))
    report = integrity_report(records, main_manifest["task_ids"],
                              main_manifest["arm_order"], main_manifest["repeats"],
                              main_manifest["model"])
    engeller = [ad for ad, deger in (
        ("yinelenen kayıt", report["duplicates"]),
        ("beklenmeyen anahtar", report["unexpected"]),
        ("şema ihlali", report["invalid_records"]),
        ("eksik koşu", report["missing"])) if deger]
    if engeller:
        raise Rq5Error(
            "RQ5 durduruldu (ana koşu bütünlüğü): " + ", ".join(engeller)
            + " — keşifsel bir korelasyon bile eksik/çift sayılmış veriden üretilmez.")

    sorunlar = check_selfcons_provenance(sc_manifest, main_manifest)
    if sorunlar:
        raise Rq5Error("RQ5 durduruldu (self-consistency provenance):\n  - "
                       + "\n  - ".join(sorunlar))
    # Kaynak zinciri: aday → başarılı çağrı provenance'ı → deterministik sonuç.
    sc_by_task = verify_selfcons(sc_manifest, sc_candidates, sc_results, sc_calls)

    task_ids = list(main_manifest["task_ids"])
    arms = list(main_manifest["arm_order"])
    if RQ5_PRIMARY_ARM not in arms:
        raise Rq5Error(
            f"birincil eşleştirme kolu {RQ5_PRIMARY_ARM!r} deneyde yok: {arms}")
    rates = task_level_rates(records, task_ids=task_ids, arms=arms,
                             metric="plus_pass")

    agreement = [float(sc_by_task[t]["agreement"]) for t in task_ids]
    seriler = {arm: failure_series(rates, task_ids, arm) for arm in arms}
    seriler[OVERALL_KEY] = overall_failure_series(rates, task_ids, arms)
    seriler[SELFCONS_ORACLE_KEY] = [
        1.0 - float(sc_by_task[t]["oracle_pass_rate"]) for t in task_ids]

    pairings = {
        ad: {
            "role": "primary" if ad == RQ5_PRIMARY_ARM else "secondary",
            "y_definition": _y_tanimi(ad),
            "spearman": spearman(agreement, seri),
            "y_descriptives": _describe(seri),
        }
        for ad, seri in seriler.items()
    }

    task_rows = [
        {"model": model, "task_id": task_id,
         "agreement": agreement[i],
         "selfcons_n_clusters": sc_by_task[task_id]["n_clusters"],
         "selfcons_oracle_pass_rate": sc_by_task[task_id]["oracle_pass_rate"],
         **{f"{arm}_plus_failure_rate": seriler[arm][i] for arm in arms},
         f"{OVERALL_KEY}_plus_failure_rate": seriler[OVERALL_KEY][i]}
        for i, task_id in enumerate(task_ids)
    ]

    return {
        "rq5_schema_version": RQ5_SCHEMA_VERSION,
        "exploratory": True,
        "p_values_reported": False,
        "analysis_unit": "task",
        "model": model,
        "experiment": main_manifest.get("name"),
        "selfcons_experiment": sc_manifest.get("name"),
        "task_set": main_manifest.get("task_set"),
        "n_tasks": len(task_ids),
        "repeats": main_manifest["repeats"],
        "arms": arms,
        "primary_pairing": RQ5_PRIMARY_ARM,
        "self_consistency": {
            "schema_version": sc_manifest["self_consistency_schema_version"],
            "n_candidates": sc_manifest["n"],
            "temperature": sc_manifest["temperature"],
            "git_commit": sc_manifest.get("git_commit"),
            "verified_candidates": len(expected_candidate_keys(sc_manifest)),
            "verified_tasks": len(sc_by_task),
            "agreement_descriptives": _describe(agreement),
        },
        # Bir rho'nun hangi kod/veri/prompt sürümünden çıktığı özetin KENDİSİNDEN
        # okunabilmeli: dosya artefakt paketine kopyalandığında yanındaki iki
        # manifest kaybolabilir, sayı ise rapora girer.
        "source_provenance": {
            "rq5_schema_version": RQ5_SCHEMA_VERSION,
            "tie_method": RQ5_TIE_METHOD,
            "main_experiment": main_manifest.get("name"),
            "main_git_commit": main_manifest.get("git_commit"),
            "main_result_schema_version": main_manifest.get("result_schema_version"),
            "main_task_ids_sha256": _canonical_hash(main_manifest.get("task_ids")),
            "main_task_file_hashes_sha256": _canonical_hash(
                main_manifest.get("task_file_hashes")),
            "selfcons_experiment": sc_manifest.get("name"),
            "selfcons_git_commit": sc_manifest.get("git_commit"),
            "selfcons_schema_version": sc_manifest.get(
                "self_consistency_schema_version"),
            "selfcons_prompt_hash": sc_manifest.get("prompt_hash"),
            "selfcons_algorithm_contract": sc_manifest.get("algorithm_contract"),
            "selfcons_llm_call_schema_version": sc_manifest.get(
                "llm_call_schema_version"),
            "selfcons_task_file_hashes_sha256": _canonical_hash(
                sc_manifest.get("task_file_hashes")),
            # İki hat aynı seçim manifestini farklı serileştirmeyle hash'lediği
            # için bu iki dize EŞİT DEĞİLDİR ve karşılaştırılmaz (§32); kimlik
            # yukarıdaki task_file_hashes üzerinden kurulur. Yine de kaydedilir:
            # hangi seçim turundan geldikleri sonradan izlenebilsin.
            "main_heldout_selection_fingerprint": main_manifest.get(
                "heldout_selection_fingerprint"),
            "selfcons_heldout_selection_fingerprint": sc_manifest.get(
                "heldout_selection_fingerprint"),
        },
        "pairings": pairings,
        "notes": NOTES,
        "_task_rows": task_rows,
    }


def _y_tanimi(ad: str) -> str:
    if ad == OVERALL_KEY:
        return "1 - (bütün kolların arm-run'larındaki plus_pass oranı)"
    if ad == SELFCONS_ORACLE_KEY:
        return "1 - self-consistency turunun kendi plus oracle geçme oranı"
    return f"1 - {ad} kolunun görev-düzeyi plus_pass oranı"


NOTES = [
    "RQ5 KEŞİFSELDİR: p-değeri üretilmez, çoklu-test düzeltmesi uygulanmaz ve "
    "hiçbir rho doğrulayıcı bir bulgu olarak sunulmaz.",
    "Birincil eşleştirme baseline koludur çünkü self-consistency hattı planner/"
    "contract kullanmayan tek kodlayıcı üretimidir; kol seçimi sonuçlara "
    "bakılarak yapılmamıştır.",
    "Self-consistency adayları temperature=0.8 ile, baseline kolu "
    "DEFAULT_TEMPERATURE ile üretilir: agreement, baseline kolunun kendi "
    "örnekleme dağılımının doğrudan ölçümü değil bir belirsizlik proxy'sidir.",
    "Modeller havuzlanmaz; her model için ayrı deney dizini ve ayrı çıktı.",
    "Tavan etkisi altında bir seri sıfır varyanslı olabilir; o durumda rho "
    "tanımsızdır (null) ve 'ilişki yok' diye okunamaz.",
]


def write_rq5_outputs(result: dict, out_dir: Path) -> dict[str, Path]:
    """Makine-okunur çıktılar. Zaman damgası YOK: aynı veri aynı baytları verir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = result["_task_rows"]
    paths = {"rq5_summary": out_dir / "rq5_self_consistency.json",
             "rq5_task_pairs": out_dir / "rq5_task_pairs.csv"}
    summary = {k: v for k, v in result.items() if not k.startswith("_")}
    paths["rq5_summary"].write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8")
    with paths["rq5_task_pairs"].open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RQ5 kesifsel self-consistency <-> basarisizlik iliskisi")
    parser.add_argument("--exp", required=True, type=Path,
                        help="ana deney dizini (logs/exp_<name>)")
    # AÇIK argüman: dizin adı ana deneyden TÜRETİLMEZ ve dosya sistemi
    # taranmaz — yanlış modelin turuna sessizce bağlanma riski kapanır.
    parser.add_argument("--selfcons", required=True, type=Path,
                        help="formal self-consistency deney dizini "
                             "(logs/exp_<name>, manifest.json içermeli)")
    parser.add_argument("--out", type=Path, default=None,
                        help="çıktı dizini (varsayılan: <exp>/analysis)")
    args = parser.parse_args()

    try:
        main_manifest, records, _ = load_experiment(args.exp)
        sc_manifest, sc_candidates, sc_results, sc_calls = load_selfcons(args.selfcons)
        result = build_rq5(main_manifest, records, sc_manifest, sc_candidates,
                           sc_results, sc_calls)
    except (AnalysisError, Rq5Error) as e:
        sys.exit(str(e))

    paths = write_rq5_outputs(result, args.out or args.exp / "analysis")
    sc = result["self_consistency"]
    print(f"deney: {result['experiment']} | model: {result['model']} | "
          f"{result['n_tasks']} görev")
    print(f"self-consistency: {result['selfcons_experiment']} | "
          f"N={sc['n_candidates']} | temp={sc['temperature']} | "
          f"ort. agreement {sc['agreement_descriptives']['mean']}")
    print(f"kaynak zinciri doğrulandı: {sc['verified_candidates']} aday, "
          f"{sc['verified_tasks']} gorev (aday -> basarili cagri -> sonuc)")
    print(f"\nSpearman rho (KEŞİFSEL, p-değeri yok; bağ: {RQ5_TIE_METHOD}):")
    for ad, p in result["pairings"].items():
        s = p["spearman"]
        rho = "tanımsız" if s["rho"] is None else f"{s['rho']:+.4f}"
        print(f"  [{p['role']:9s}] {ad:26s} rho {rho}  (n={s['n']}, "
              f"bağ grubu x/y {s['tied_groups_x']}/{s['tied_groups_y']})")
        if s["undefined_reason"]:
            print(f"      -> {s['undefined_reason']}")
    print("\nÇıktılar:")
    for name, path in paths.items():
        print(f"  {name:18s} -> {path}")


if __name__ == "__main__":
    main()
