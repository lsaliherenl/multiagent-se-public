"""Deney çalıştırıcı (build sırası adım 8): kollar × görevler × tekrarlar.

Tasarım:
- Koşu sırası: tekrar → görev → kol. Aynı görevin (rotasyona göre sıralanmış)
  kolları ART ARDA koşar. Kol sırası ARTIK SABİT DEĞİL: her (görev, tekrar)
  çifti için (task_index + repeat) % len(arms) ofsetiyle döndürülür — hem
  tekrarlar arasında hem AYNI tekrar geçişi içindeki ardışık görevler arasında
  sıra değişir, böylece sağlayıcı tarafındaki zamana bağlı dalgalanma (geçiş
  içi VE geçişler arası) kollar arasında dengelenir. Gerçekleşen sıra her
  sonuç kaydına arm_position olarak yazılır (yeniden hesaplamaya gerek kalmaz).
- Kesintiden devam: anahtar (model, arm, task_id, repeat) --
  eval/result_schema.RESUME_KEY_FIELDS. MODEL DAHİLDİR: aynı dizinde farklı
  modelle koşulmuş kayıtlar yeni modelin koşusunu atlatmamalı. Çıktıdaki
  tamamlanmış anahtarlar atlanır; status="run_error" kayıtları tamamlanmış
  SAYILMAZ (yeniden denenir) ama SİLİNMEZ (şeffaflık). Aynı komutu tekrar
  çalıştırmak güvenlidir.
- Sonuç sözleşmesi: her kayıt yazılmadan ÖNCE eval/result_schema ile
  doğrulanır; ihlalde koşu durur. Bozuk bir kayıt dosyaya girerse analiz
  aşamasına kadar fark edilmeyebilir ve o noktada koşuyu tekrarlamak çok
  pahalı olur. Analiz katmanı aynı modülü kullanır -> alan adları tek kaynakta.
- Bütünlük raporu: koşu sonunda beklenen/tamamlanan/run_error sayıları,
  yinelenen ve eksik anahtarlar raporlanır (50 × 3 × 4 = 600 kayıt/model).
- manifest.json: deney yapılandırmasının anlık görüntüsü (model, sıcaklık,
  görev listesi, kol sırası, rotasyon şeması, git commit, görev/uv.lock
  hash'leri). Var olan deneye FARKLI yapılandırmayla devam etmeye çalışmak
  veri setini zehirler → uyumsuzlukta durdurulur. Kol seti/sırası deney
  ortasında GENİŞLETİLEMEZ (rotasyon hem içeriğe hem sıraya bağlı olduğu
  için) — repeats büyütülebilir (zaten çalışmış tekrarların rotasyonunu
  değiştirmiyor).
- Bütünlük: her koşu (ilk yazım VE resume) git çalışma ağacının temiz
  olmasını şart koşar — kod/görev içeriği deney ortasında sessizce
  değişmesin diye (EXPERIMENT_PROTOCOL.md §12).
- Rate-limit: agents/llm.py'deki throttle + retry katmanı devralır; runner
  sıralı çalışır, ek paralellik yok (bilinçli — free/ucuz katman limitleri).

- Model: --model ZORUNLUDUR (varsayılan yok). Ana deney verisinin sessizce
  pilot modele düşmesi, fark edilmesi en zor hata olurdu. Takma ad
  (config.MODEL_ALIASES) ya da tam LiteLLM slug'ı verilebilir; manifeste
  ve loglara her zaman çözümlenmiş tam slug yazılır. Held-out sette yalnız
  config.MODEL_PRODUCERS kabul edilir (judge/adjudicator modelleri ve tanımsız
  slug'lar reddedilir) — kontrol, API anahtarı/görev/çıktı dizini adımlarından
  ÖNCE yapılır.

Kullanım:
    uv run python -m eval.runner --name gemini_main --model main --task-set heldout --repeats 3
    uv run python -m eval.runner --name gemini_main --model main --task-set heldout --repeats 3  # devam
    uv run python -m eval.runner --name smoke --model dev --task-set pilot --tasks 2 --repeats 1
"""

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from agents.coder import SYSTEM_PROMPT as CODER_SYSTEM_PROMPT
from agents.contracts import PlannerOutput
from agents.planner import CONTRACT_SYSTEM_PROMPT, NAIVE_SYSTEM_PROMPT
from config import (
    ALL_ARMS,
    ARM_BASELINE,
    ARM_ROTATION_SCHEME_VERSION,
    DEFAULT_TEMPERATURE,
    HELDOUT_TASKS_DIR,
    LLM_CALL_SCHEMA_VERSION,
    LOGS_DIR,
    MAX_OUTPUT_TOKENS,
    REASONING_CONFIG,
    RESULT_SCHEMA_VERSION,
    ROOT,
    TASK_SETS,
    model_alias_help,
    provider_routing_for,
    validate_model_for_task_set,
)
from eval.harness import load_all_tasks
from eval.result_schema import (
    expected_resume_keys,
    integrity_report,
    is_run_error,
    make_run_error_record,
    resume_key,
    stamp_record,
    validate_record,
    verify_resumable,
)
from pipeline.baseline import SYSTEM_PROMPT as BASELINE_SYSTEM_PROMPT
from pipeline.baseline import run_task as baseline_run_task
from pipeline.graph import build_graph
from pipeline.run_graph import run_task as graph_run_task

# Kol listesi config.ALL_ARMS'tan gelir (tek kaynak). Rotasyon formülü kol
# SAYISINA bağlı olduğu için kol seti değişimi manifest üzerinden korunur.


def load_records(results_path: Path) -> list[dict]:
    if not results_path.exists():
        return []
    return [json.loads(line) for line in
            results_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_completed(results_path: Path) -> set[tuple]:
    """Tamamlanmış resume anahtarları; run_error'lar hariç (yeniden denenirler).

    Anahtar MODELİ de içerir (eval/result_schema.RESUME_KEY_FIELDS): aynı deney
    dizininde farklı modelle koşulmuş eski kayıtlar, yeni modelin koşusunu
    sessizce atlatmamalı. Manifest zaten model değişimini engelliyor, fakat
    resume mantığının kendisi de buna dayanmamalı (savunma derinliği).
    """
    return {resume_key(r) for r in load_records(results_path) if not is_run_error(r)}


def build_run_plan(task_ids: list[str], arms: list[str], repeats: int,
                   completed: set[tuple], model: str) -> list[tuple[int, str, str, int]]:
    """Bekleyen koşular: (repeat, task_id, arm, position).

    position: kolun bu (task, repeat) için gerçekleşen sıradaki 0-index
    konumu — karşı-dengelemenin fiilen uygulandığının kaydı için.

    Rotasyon formülü kol SAYISINDAN bağımsızdır ((task_index + repeat) %
    len(arms)); dört kolda da geçerlidir. Fiilen gerçekleşen konum dağılımı
    arm_position üzerinden denetlenebilir (bkz. position_balance).
    """
    plan = []
    n = len(arms)
    for rep in range(repeats):
        for task_index, task_id in enumerate(task_ids):
            offset = (task_index + rep) % n
            order = arms[offset:] + arms[:offset]
            for position, arm in enumerate(order):
                if (model, arm, task_id, rep) not in completed:
                    plan.append((rep, task_id, arm, position))
    return plan


def position_balance(task_ids: list[str], arms: list[str], repeats: int) -> dict[str, list[int]]:
    """Kol -> her konumda kaç kez koşacağı. Karşı-dengelemenin denetimi.

    Tam denge, len(task_ids)*repeats sayısının kol sayısına bölünebildiği
    durumlarda mümkündür; bölünmüyorsa kalıntı dengesizlik kaçınılmazdır ve
    burada görünür olur (denetlenebilirlik için).
    """
    counts = {arm: [0] * len(arms) for arm in arms}
    n = len(arms)
    for rep in range(repeats):
        for task_index in range(len(task_ids)):
            offset = (task_index + rep) % n
            for position, arm in enumerate(arms[offset:] + arms[:offset]):
                counts[arm][position] += 1
    return counts


def check_or_write_manifest(out_dir: Path, config_snapshot: dict) -> None:
    """Manifest yoksa yazar; varsa kritik alanların değişmediğini doğrular."""
    manifest_path = out_dir / "manifest.json"
    # max_tokens/reasoning_config da KRİTİK: deney ortasında değişirlerse kollar
    # farklı üretim koşullarında koşmuş olur (iç geçerlilik kırılır).
    critical = ("model", "temperature", "max_tokens", "reasoning_config",
                "provider_routing", "task_set", "task_ids", "arm_rotation_scheme",
                "arm_order", "git_commit", "task_file_hashes", "uv_lock_hash",
                "result_schema_version", "llm_call_schema_version",
                "heldout_selection_fingerprint", "prompt_contract_hash",
                "python_version", "platform")
    # Kritik alan listesi büyüdüğünde çağıran taraf onu eklemeyi unutursa,
    # sessiz KeyError yerine açık hata: eksik alan = korunmayan alan demektir.
    missing = [k for k in critical if k not in config_snapshot]
    if missing:
        raise KeyError(f"manifest anlık görüntüsünde kritik alan(lar) eksik: {missing}")
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        mismatches = {k: (existing.get(k), config_snapshot[k])
                      for k in critical if existing.get(k) != config_snapshot[k]}
        if mismatches:
            sys.exit(f"manifest uyuşmazlığı (aynı deneye farklı yapılandırmayla "
                     f"devam edilemez): {mismatches}\nYeni deney için --name değiştir.")
        # repeats büyüyebilir (zaten çalışmış tekrarların rotasyonunu değiştirmiyor);
        # arm_order/arm_rotation_scheme İSE kritik -- kol seti/sırası genişletilemez.
        existing["repeats"] = max(existing.get("repeats", 0), config_snapshot["repeats"])
        manifest_path.write_text(json.dumps(existing, indent=2, ensure_ascii=False),
                                 encoding="utf-8")
    else:
        manifest_path.write_text(json.dumps(config_snapshot, indent=2, ensure_ascii=False),
                                 encoding="utf-8")


def _git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, timeout=10).stdout.strip() or None
    except Exception:
        return None


def _require_clean_tree() -> None:
    """Git çalışma ağacı kirliyse durur -- deney SADECE commit edilmiş bir
    durumdan başlar/devam eder (reprodüktibilite şartı;
    EXPERIMENT_PROTOCOL.md §12). git'e erişilemezse (best-effort) engellenmez."""
    try:
        r = subprocess.run(["git", "status", "--porcelain"], capture_output=True,
                           text=True, timeout=10)
        dirty = bool(r.stdout.strip()) if r.returncode == 0 else None
    except Exception:
        dirty = None
    if dirty:
        sys.exit("git çalışma ağacı kirli — tam koşu öncesi commit at. Deney SADECE "
                 "temiz ağaçtan başlar/devam eder (reprodüktibilite şartı).")


def _hash_task_files(task_ids: list[str], task_set: str) -> dict[str, str] | None:
    """Dosya adı -> sha256 haritası (tek birleşik hash DEĞİL -- hangi dosyanın
    değiştiğini doğrudan görebilmek için denetlenebilir tutulur)."""
    tasks_dir = TASK_SETS[task_set]
    try:
        return {f"{tid}.json": hashlib.sha256((tasks_dir / f"{tid}.json").read_bytes()).hexdigest()
                for tid in sorted(task_ids)}
    except Exception:
        return None


def _hash_uv_lock() -> str | None:
    lock_path = ROOT / "uv.lock"
    if not lock_path.exists():
        return None
    try:
        return hashlib.sha256(lock_path.read_bytes()).hexdigest()
    except Exception:
        return None


def _heldout_selection_fingerprint() -> str | None:
    """Held-out seçim manifestinin DEĞİŞMEZ kısmının hash'i.

    created_ts ve reference_timings bilinçli olarak DIŞLANIR: ikisi de ortama
    bağlı gözlemdir, her üretimde değişir ve dahil edilirlerse fingerprint
    "görev seti değişti" diye yanlış alarm verirdi.
    """
    path = HELDOUT_TASKS_DIR / "_selection_manifest.json"
    if not path.exists():
        return None
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    manifest.pop("created_ts", None)
    for source in manifest.get("sources", {}).values():
        source.pop("reference_timings", None)
    payload = json.dumps(manifest, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _prompt_contract_hash() -> str:
    """Prompt gövdeleri + sözleşme şemasının hash'i.

    Deney ortasında bir prompt kelimesinin değişmesi, kolların farklı
    talimatlarla koşması demektir — git commit'i bunu yakalar ama yalnız
    dolaylı olarak. Bu hash, doğrudan ve okunabilir bir kritik alan sağlar.
    """
    parts = [
        BASELINE_SYSTEM_PROMPT,
        NAIVE_SYSTEM_PROMPT,
        CONTRACT_SYSTEM_PROMPT,
        CODER_SYSTEM_PROMPT,
        json.dumps(PlannerOutput.model_json_schema(), sort_keys=True),
    ]
    return hashlib.sha256("\n---\n".join(parts).encode("utf-8")).hexdigest()


def execute_run(arm: str, task: dict, repeat: int, model: str, graphs: dict, *,
                experiment: str, run_id: str, arm_position: int,
                task_set: str) -> dict:
    """Tek koşuyu yürütür ve sonucu sözleşmeye uygun biçimde damgalar."""
    if arm == ARM_BASELINE:
        record = baseline_run_task(task, model, DEFAULT_TEMPERATURE,
                                   experiment=experiment, run_id=run_id, repeat=repeat)
    else:
        record = graph_run_task(graphs[arm], task, arm,
                                thread_id=f"{arm}-{task['task_id']}-r{repeat}",
                                model=model,
                                experiment=experiment, run_id=run_id, repeat=repeat)
    return stamp_record(record, experiment=experiment, model=model, task_set=task_set,
                        arm=arm, task_id=task["task_id"], repeat=repeat,
                        run_id=run_id, arm_position=arm_position)


def main() -> None:
    parser = argparse.ArgumentParser(description="4 kollu deney çalıştırıcı")
    parser.add_argument("--name", required=True,
                        help="deney adı (çıktı: logs/exp_<name>/); devam için aynı adı ver")
    parser.add_argument("--arms", nargs="+", choices=ALL_ARMS, default=ALL_ARMS)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--tasks", type=int, default=None, help="ilk N görev (varsayılan: tümü)")
    parser.add_argument("--model", required=True,
                        help=f"{model_alias_help()}. ZORUNLU: ana deneyin örtük bir "
                             "varsayılana düşmesi engellenir. Held-out sette yalnız "
                             "üretici modeller kabul edilir.")
    # Görev seti de ZORUNLU: pilot ve held-out setlerin karışması, sonucu
    # sessizce geçersiz kılan türden bir hata olurdu (§5.1).
    parser.add_argument("--task-set", required=True, choices=sorted(TASK_SETS),
                        help="pilot (development, 20 görev) | heldout (ana deney, 50 görev)")
    args = parser.parse_args()
    # Üretici kapısı EN ÖNDE: anahtar kontrolü, görev yükleme ve çıktı dizini
    # oluşturmadan önce. Aksi halde yanlış modelle açılmış bir deney dizini ve
    # manifest geride kalır; sonraki doğru koşu ya bu dizine devam etmeye
    # çalışır ya da yarım kalmış bir artefakt bırakır.
    try:
        model = validate_model_for_task_set(args.model, args.task_set)
    except ValueError as e:
        sys.exit(str(e))

    if not (os.environ.get("OPENROUTER_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")):
        sys.exit("API anahtarı yok — .env.example'ı .env olarak kopyalayıp doldur.")
    _require_clean_tree()

    tasks = load_all_tasks(args.task_set)[: args.tasks]
    if not tasks:
        sys.exit(f"{args.task_set!r} görev seti boş — önce üretilmeli "
                 "(scripts/fetch_tasks.py veya scripts/fetch_evalplus.py).")
    task_ids = [t["task_id"] for t in tasks]
    tasks_by_id = {t["task_id"]: t for t in tasks}

    out_dir = LOGS_DIR / f"exp_{args.name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    check_or_write_manifest(out_dir, {
        "name": args.name,
        "created_ts": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "temperature": DEFAULT_TEMPERATURE,
        "max_tokens": MAX_OUTPUT_TOKENS,
        "reasoning_config": REASONING_CONFIG,
        "provider_routing": provider_routing_for(model),
        "arm_order": args.arms,   # SIRALANMAMIŞ -- rotasyonun fiilen kullandığı liste
        "repeats": args.repeats,
        "task_set": args.task_set,
        "task_ids": task_ids,
        "arm_rotation_scheme": ARM_ROTATION_SCHEME_VERSION,
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "llm_call_schema_version": LLM_CALL_SCHEMA_VERSION,
        "heldout_selection_fingerprint": _heldout_selection_fingerprint(),
        "prompt_contract_hash": _prompt_contract_hash(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "task_file_hashes": _hash_task_files(task_ids, args.task_set),
        "uv_lock_hash": _hash_uv_lock(),
    })

    results_path = out_dir / "results.jsonl"
    existing = load_records(results_path)
    # HİÇBİR API ÇAĞRISI YAPILMADAN ÖNCE mevcut dosya denetlenir: bozuk bir veri
    # kümesinin üstüne pahalı yeni koşu eklemek, hatayı hem büyütür hem de
    # geri almayı zorlaştırır. Eksik koşular ve geçmiş run_error'lar devam
    # etmeye ENGEL DEĞİLDİR (zaten devam etmenin sebebi onlar).
    expected = expected_resume_keys(task_ids, args.arms, args.repeats, model)
    blockers = verify_resumable(existing, expected)
    if blockers:
        sys.exit("mevcut sonuç dosyası devam etmeye uygun değil "
                 f"({results_path}):\n  - " + "\n  - ".join(blockers[:20]))

    completed = {resume_key(r) for r in existing if not is_run_error(r)}
    plan = build_run_plan(task_ids, args.arms, args.repeats, completed, model)
    total = len(task_ids) * len(args.arms) * args.repeats
    print(f"deney: {args.name} | model: {model} | görev seti: {args.task_set}")
    print(f"beklenen {total} kayıt ({len(task_ids)} görev × {args.repeats} tekrar "
          f"× {len(args.arms)} kol); {len(completed)} tamam, {len(plan)} bekliyor")
    print("kol × konum dengesi:", position_balance(task_ids, args.arms, args.repeats), "\n")

    graphs = {arm: build_graph(arm) for arm in args.arms if arm != ARM_BASELINE}
    errors = 0
    with results_path.open("a", encoding="utf-8") as out:
        for i, (rep, task_id, arm, position) in enumerate(plan, 1):
            run_id = uuid4().hex
            try:
                record = execute_run(arm, tasks_by_id[task_id], rep, model, graphs,
                                     experiment=args.name, run_id=run_id,
                                     arm_position=position, task_set=args.task_set)
            except Exception as e:
                errors += 1
                record = make_run_error_record(
                    experiment=args.name, model=model, task_set=args.task_set,
                    arm=arm, task_id=task_id, repeat=rep, run_id=run_id,
                    arm_position=position, error=f"{type(e).__name__}: {e}")
            # Sözleşme ihlali YAZILMADAN önce durdurulur: bozuk bir kayıt
            # dosyaya girerse analiz aşamasına kadar fark edilmeyebilir ve o
            # noktada koşuyu tekrarlamak çok pahalı olur.
            problems = validate_record(record)
            if problems:
                out.flush()
                sys.exit(f"sonuç sözleşmesi ihlali ({arm}/{task_id}/r{rep}): {problems}")
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
            print(f"[{i}/{len(plan)}] r{rep} {task_id} {arm}: {record['status']}"
                  + (f" ({record['error_class']})" if record.get("error_class") else ""))

    _print_summary(load_records(results_path), task_ids, args.arms, args.repeats,
                   model, errors, results_path)


def _print_summary(all_records: list[dict], task_ids: list[str], arms: list[str],
                   repeats: int, model: str, errors: int, results_path: Path) -> None:
    """Kol özeti + bütünlük raporu (dosyanın TAMAMI, önceki oturumlar dahil)."""
    print("\n--- Özet (tüm deney) ---")
    for arm in sorted({r["arm"] for r in all_records}):
        arm_recs = [r for r in all_records if r["arm"] == arm and not is_run_error(r)]
        if not arm_recs:
            continue
        base = sum(r.get("base_pass") is True for r in arm_recs)
        plus = sum(r.get("plus_pass") is True for r in arm_recs)
        print(f"{arm:26s}: plus {plus}/{len(arm_recs)} | base {base}/{len(arm_recs)}")

    report = integrity_report(all_records, task_ids, arms, repeats, model)
    print("\n--- Bütünlük ---")
    print(f"beklenen {report['expected_count']} | tamamlanan {report['completed_count']} "
          f"| run_error {report['run_error_count']}")
    if report["duplicates"]:
        print(f"UYARI: {len(report['duplicates'])} yinelenen anahtar: "
              f"{list(report['duplicates'])[:3]}")
    if report["missing"]:
        print(f"UYARI: {len(report['missing'])} eksik koşu (aynı komutla devam edilebilir)")
    if report["unexpected"]:
        print(f"UYARI: {len(report['unexpected'])} BEKLENMEYEN anahtar "
              f"(yanlış model/görev/kol?): {report['unexpected'][:3]}")
    if report["invalid_records"]:
        print(f"UYARI: {len(report['invalid_records'])} geçersiz kayıt")
    if report["complete"]:
        print("Deney EKSİKSİZ: yinelenen/eksik/geçersiz kayıt yok.")
    if errors:
        print(f"\nUYARI: bu oturumda {errors} koşu run_error aldı — aynı komutla tekrar denenebilir.")
    print(f"\nSonuçlar -> {results_path}")


if __name__ == "__main__":
    main()
