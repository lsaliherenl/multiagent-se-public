"""Held-out ana görev setini üretir: EvalPlus (HumanEval+ / MBPP+) → tasks_heldout/.

EXPERIMENT_PROTOCOL.md §5. Hiçbir aşamada MODEL ÇAĞRISI YAPILMAZ — seçim tamamen
deterministiktir ve model çıktısı görülmeden tamamlanır (§5.4 md. 7).

Boru hattı:
  1. Kaynakları sürüm-sabitli URL'den indir, SHA-256 doğrula.
  2. Statik uygunluk filtresi (kod çalıştırmadan): pilot ID'ler, özel-oracle
     görevleri, izinsiz import, determinizm riski, harness ad çakışması.
  3. Dinamik doğrulama (sandbox'ta, deterministik):
     a. Girdi deserializasyon varyantını EvalPlus'ın kendi `contract`'ıyla
        DOĞRULA (tahmin etme).
     b. Referans çözümü koşturup beklenen çıktıları üret.
     c. Üretilen base/plus testlerini REFERANS çözümle (contract'sız, aday koda
        benzer halde) tekrar koştur — hem ölçüm aletini doğrular hem süreçler
        arası determinizmi (set/dict sırası, hash seed) sınar.
     d. Referansın plus süresi bütçeyi aşarsa reddet (aday koda headroom kalsın).
  4. Sabit seed ile 30 HumanEval+ + 20 MBPP+ seç.
  5. tasks_heldout/*.json + selection_manifest.json yaz (her ret nedeniyle).

Kullanım:
    uv run python scripts/fetch_evalplus.py            # tam koşu (~20-40 dk)
    uv run python scripts/fetch_evalplus.py --limit 40 # hızlı deneme
"""

import argparse
import gzip
import hashlib
import io
import json
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (  # noqa: E402
    HELDOUT_COUNTS,
    HELDOUT_SELECTION_SEED,
    HELDOUT_TASKS_DIR,
    PLUS_TIMEOUT_S,
    REFERENCE_PLUS_BUDGET_S,
    SANDBOX_TIMEOUT_S,
    TASKS_DIR,
)
from eval.sandbox import run_code  # noqa: E402
from eval.task_selection import (  # noqa: E402
    EVALPLUS_SOURCES,
    INPUT_VARIANTS,
    apply_variant,
    build_groundtruth_script,
    build_test_code,
    deterministic_select,
    parse_groundtruth_output,
    reference_solution,
    static_eligibility,
    task_content_hash,
)

GT_MARKER = "__GROUNDTRUTH__"
VARIANT_PROBE_INPUTS = 25  # varyant aramasında kullanılan girdi sayısı (hız)
# Beklenen çıktı üretiminde stdout sınırı (bin vakalık repr'ı taşıyacak kadar).
# Sadece referans çözüm için; aday kod varsayılan sınırla koşar.
GROUNDTRUTH_OUTPUT_LIMIT_BYTES = 40_000_000


def _download(url: str, expected_sha256: str) -> bytes:
    print(f"  indiriliyor: {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "multiagent-se"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        data = resp.read()
    actual = hashlib.sha256(data).hexdigest()
    if actual != expected_sha256:
        raise ValueError(f"checksum uyuşmazlığı ({url}): beklenen {expected_sha256}, alınan {actual}")
    print(f"  {len(data)} bayt, sha256 doğrulandı")
    return data


def _load_jsonl_gz(raw: bytes) -> list[dict]:
    with gzip.open(io.BytesIO(raw), "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _pilot_source_ids() -> set[str]:
    """Pilot görevlerin EvalPlus'taki karşılıkları — held-out'tan dışlanır.

    Pilot dosyaları humaneval_000 / mbpp_002 gibi adlandırılmıştır; buradan
    EvalPlus ID biçimine (HumanEval/0, Mbpp/2) geri çevrilir. Dizin yoksa boş
    küme (ilk kurulumda pilot henüz üretilmemiş olabilir).
    """
    excluded = set()
    for path in sorted(TASKS_DIR.glob("*.json")):
        stem = path.stem
        prefix, _, number = stem.rpartition("_")
        if not number.isdigit():
            continue
        if prefix == "humaneval":
            excluded.add(f"HumanEval/{int(number)}")
        elif prefix == "mbpp":
            excluded.add(f"Mbpp/{int(number)}")
    return excluded


def _find_variant(record: dict, source: str) -> tuple[str | None, str | None]:
    """Girdi deserializasyon varyantını contract ile DOĞRULAYARAK bulur."""
    reference = reference_solution(record, source, with_contract=True)
    if reference is None:
        return None, "contract_enjekte_edilemedi"
    probe_inputs = (record["base_input"] + record["plus_input"])[:VARIANT_PROBE_INPUTS]
    for variant in INPUT_VARIANTS:
        converted = [apply_variant(args, variant) for args in probe_inputs]
        script = build_groundtruth_script(reference, record["entry_point"],
                                          converted, variant, GT_MARKER)
        result = run_code(script, timeout_s=SANDBOX_TIMEOUT_S)
        if result.status == "ok":
            return variant, None
    return None, "girdi_varyanti_bulunamadi"


def _groundtruth(record: dict, source: str, variant: str) -> tuple[list | None, str | None]:
    """Referans çözümü TÜM girdilerde koşturup beklenen çıktıları üretir."""
    reference = reference_solution(record, source, with_contract=True)
    all_inputs = [apply_variant(args, variant)
                  for args in record["base_input"] + record["plus_input"]]
    script = build_groundtruth_script(reference, record["entry_point"],
                                      all_inputs, variant, GT_MARKER)
    # Çıktı bilerek büyük (bine kadar vaka) -> varsayılan 10KB kırpma sınırı
    # burada devre dışı. Yalnız GÜVENİLEN referans çözüm için (LLM kodu değil).
    result = run_code(script, timeout_s=PLUS_TIMEOUT_S,
                      output_limit_bytes=GROUNDTRUTH_OUTPUT_LIMIT_BYTES)
    if result.status == "timeout":
        return None, "referans_groundtruth_timeout"
    if result.status != "ok":
        return None, "referans_groundtruth_hatasi"
    try:
        outputs = parse_groundtruth_output(result.stdout, GT_MARKER)
    except (ValueError, SyntaxError):
        return None, "cikti_literal_degil"
    if len(outputs) != len(all_inputs):
        return None, "cikti_sayisi_uyusmuyor"
    return list(zip(all_inputs, outputs)), None


def _verify_with_reference(record: dict, source: str,
                           base_test: str, plus_test: str) -> tuple[dict | None, str | None]:
    """Üretilen testleri contract'SIZ referans çözümle doğrular.

    İki şeyi birden sınar: (a) ölçüm aleti sağlam mı (referans kendi testini
    geçiyor mu), (b) çıktı süreçler arası deterministik mi — testler AYRI bir
    süreçte koşuyor, set/dict sırası hash seed'e bağlı değişseydi burada
    yakalanırdı.
    """
    reference = reference_solution(record, source, with_contract=False)
    entry_point = record["entry_point"]

    base_result = run_code(f"{reference}\n\n{base_test}\n\ncheck({entry_point})\n",
                           timeout_s=SANDBOX_TIMEOUT_S)
    if base_result.status != "ok":
        return None, f"referans_base_gecemedi:{base_result.status}"

    plus_result = run_code(f"{reference}\n\n{plus_test}\n\ncheck({entry_point})\n",
                           timeout_s=PLUS_TIMEOUT_S)
    if plus_result.status != "ok":
        return None, f"referans_plus_gecemedi:{plus_result.status}"
    if plus_result.duration_s > REFERENCE_PLUS_BUDGET_S:
        # Ret NEDENİ saf bir kategori olmalı: ölçülen süre buraya gömülürse
        # manifest her koşuda değişir (ölçüm ≠ karar). Süre ayrıca saklanır.
        return None, "referans_plus_yavas"
    return {"reference_base_s": round(base_result.duration_s, 3),
            "reference_plus_s": round(plus_result.duration_s, 3)}, None


def _task_id(source: str, source_task_id: str) -> str:
    number = int(source_task_id.split("/")[-1])
    return f"{EVALPLUS_SOURCES[source]['prefix']}_{number:03d}"


def build_task(record: dict, source: str) -> tuple[dict | None, str | None]:
    """Tek EvalPlus kaydını görev sözleşmemize çevirir; uygunsa (task, None)."""
    variant, reason = _find_variant(record, source)
    if reason:
        return None, reason

    cases, reason = _groundtruth(record, source, variant)
    if reason:
        return None, reason

    n_base = len(record["base_input"])
    atol = record.get("atol", 0)
    base_test = build_test_code(cases[:n_base], atol)
    # Plus testi base girdilerini DE içerir (EvalPlus semantiği: plus skoru her
    # iki girdi kümesini birden geçmeyi gerektirir).
    plus_test = build_test_code(cases, atol)

    timings, reason = _verify_with_reference(record, source, base_test, plus_test)
    if reason:
        return None, reason

    task = {
        "task_id": _task_id(source, record["task_id"]),
        "source": source,
        "source_task_id": record["task_id"],
        "source_version": EVALPLUS_SOURCES[source]["version"],
        "prompt": record["prompt"],
        "entry_point": record["entry_point"],
        "reference_solution": reference_solution(record, source, with_contract=False),
        "atol": atol,
        "input_variant": variant,
        "n_base_cases": n_base,
        "n_plus_cases": len(cases),
        "base_test_code": base_test,
        "plus_test_code": plus_test,
    }
    task["content_sha256"] = task_content_hash(task)
    # Süreler görev İÇERİĞİ değil, ortama bağlı GÖZLEMdir: görev dosyasına
    # yazılırsa her koşuda değişir ve dosyalar (dolayısıyla content_sha256)
    # yeniden üretilemez hale gelir. Manifestte provenance olarak saklanır.
    return {"task": task, "timings": timings}, None


def process_source(source: str, records: list[dict], excluded: set[str],
                   limit: int | None) -> tuple[list[dict], dict, dict]:
    """Statik + dinamik filtreyi uygular.

    Döndürür: (uygun görevler, ret kayıtları, görev başına referans süreleri).
    Süreler görev dosyasına DEĞİL manifeste gider (bkz. build_task).
    """
    rejections: dict[str, str] = {}
    timings: dict[str, dict] = {}
    eligible: list[dict] = []
    candidates = records[:limit] if limit else records

    for i, record in enumerate(candidates, 1):
        source_id = record["task_id"]
        reason = static_eligibility(record, source, excluded)
        if reason:
            rejections[source_id] = reason
            continue
        try:
            built, reason = build_task(record, source)
        except Exception as exc:
            # Uzun bir toplu koşuda tek görevin beklenmedik hatası bütün koşuyu
            # öldürmemeli; ret olarak kaydedilip devam edilir (neden manifestte).
            rejections[source_id] = f"beklenmeyen_hata:{type(exc).__name__}"
            continue
        if reason:
            rejections[source_id] = reason
            continue
        timings[built["task"]["task_id"]] = built["timings"]
        eligible.append(built["task"])
        if i % 25 == 0 or i == len(candidates):
            print(f"    [{i}/{len(candidates)}] uygun: {len(eligible)}, ret: {len(rejections)}")
    return eligible, rejections, timings


def main() -> None:
    parser = argparse.ArgumentParser(description="Held-out EvalPlus görev seti üretici")
    parser.add_argument("--limit", type=int, default=None,
                        help="kaynak başına ilk N kaydı incele (hızlı deneme; ana koşuda VERİLMEZ)")
    parser.add_argument("--out", type=Path, default=HELDOUT_TASKS_DIR)
    args = parser.parse_args()

    excluded = _pilot_source_ids()
    print(f"pilot ID dışlaması: {len(excluded)} görev\n")

    all_selected: list[dict] = []
    manifest_sources = {}
    for source, meta in EVALPLUS_SOURCES.items():
        print(f"=== {source} ({meta['version']}) ===")
        raw = _download(meta["url"], meta["sha256"])
        records = _load_jsonl_gz(raw)
        print(f"  {len(records)} kayıt; filtre uygulanıyor...")

        eligible, rejections, timings = process_source(source, records, excluded, args.limit)
        eligible_by_id = {t["task_id"]: t for t in eligible}
        count = HELDOUT_COUNTS[source]
        chosen_ids = deterministic_select(list(eligible_by_id), count, HELDOUT_SELECTION_SEED)
        chosen = [eligible_by_id[tid] for tid in chosen_ids]
        all_selected.extend(chosen)

        print(f"  uygun havuz: {len(eligible)} | seçilen: {len(chosen)}")
        examined = len(records[: args.limit] if args.limit else records)
        # Ret nedenleri iki biçimde saklanır: ID başına tam neden (denetlenebilirlik)
        # ve nedene göre özet sayım (denetimde doğrudan raporlanabilsin).
        reason_counts: dict[str, int] = {}
        for reason in rejections.values():
            reason_counts[reason.split(":")[0]] = reason_counts.get(reason.split(":")[0], 0) + 1
        manifest_sources[source] = {
            **{k: meta[k] for k in ("version", "url", "sha256", "size_bytes")},
            "records_total": len(records),
            "records_examined": examined,
            "excluded_count": len(rejections),
            "eligible_count": len(eligible),
            "selected_count": len(chosen),
            # Bütünlük özdeşliği: incelenen = uygun + dışlanan. Analiz katmanı
            # bunu doğrulayabilsin diye açıkça saklanır.
            "counts_consistent": examined == len(eligible) + len(rejections),
            "selected_task_ids": chosen_ids,
            "eligible_pool_task_ids": sorted(eligible_by_id),
            "excluded_task_ids": sorted(rejections),
            "rejection_reason_counts": dict(sorted(reason_counts.items())),
            "rejections": dict(sorted(rejections.items())),
            # Ortama bağlı gözlem: seçilen görevlerin referans çözüm süreleri.
            # Görev dosyalarında DEĞİL burada -- görev dosyaları bayt-bayt
            # yeniden üretilebilir kalmalı.
            "reference_timings": {tid: timings[tid] for tid in chosen_ids},
        }

    args.out.mkdir(parents=True, exist_ok=True)
    for task in all_selected:
        (args.out / f"{task['task_id']}.json").write_text(
            json.dumps(task, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    manifest = {
        "created_ts": datetime.now(timezone.utc).isoformat(),
        # Raporlarda ve artefakt paketinde kullanılacak resmi ad. "HumanEval+ /
        # MBPP+ skoru" DEĞİL: eşitlik-dışı oracle gerektiren görevler dışlandığı
        # için sayılar EvalPlus liderlik tablosuyla doğrudan kıyaslanamaz.
        "dataset_name": "EvalPlus-derived equality-compatible held-out subset",
        "primary_metric": "Plus-test pass rate (plus_pass)",
        "selection_seed": HELDOUT_SELECTION_SEED,
        "counts": HELDOUT_COUNTS,
        "reference_plus_budget_s": REFERENCE_PLUS_BUDGET_S,
        "plus_timeout_s": PLUS_TIMEOUT_S,
        "excluded_pilot_source_ids": sorted(excluded),
        "sources": manifest_sources,
        "task_content_hashes": {t["task_id"]: t["content_sha256"] for t in all_selected},
    }
    # "_" öneki: eval.harness.load_all_tasks bu dosyayı görev sanmasın.
    manifest_path = args.out / "_selection_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
                             encoding="utf-8")

    print(f"\n{len(all_selected)} görev yazıldı -> {args.out}")
    print(f"seçim manifesti      -> {manifest_path}")
    for source, info in manifest_sources.items():
        print(f"  {source:16s} uygun {info['eligible_count']:>3d} / "
              f"incelenen {info['records_examined']:>3d}, seçilen {info['selected_count']}")


if __name__ == "__main__":
    main()
