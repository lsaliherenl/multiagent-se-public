"""Formal P1 uyumluluk smoke CLI'sı -- Gemini 3.5 Flash Lite ve Grok 4.3.

Bu, `scripts/smoke_llm.py`'nin (basit geliştirici tanısı) yerini ALMAZ; formal
hattın manifest/resume/provenance sözleşmesi ayrıdır ve gerçek üretim prompt/
parse yollarını kullanır (bkz. `eval/compatibility_smoke.py` modül dokümanı).

**Bu CLI yalnız `gemini` ve `grok` hedeflerini kabul eder.** DeepSeek/MiniMax/
main/dev gibi takma adlar veya serbest bir LiteLLM slug'ı formal hedef olarak
sunulmaz -- CLI'da böyle bir seçenek yoktur (yalnız `--name` alır). Held-out
görev seti bu CLI'da hiçbir aşamada seçenek olarak SUNULMAZ.

**P1 Parça 6A bu CLI'yı yalnız ALTYAPI olarak sağlar.** `run` (dry-run hariç)
bu commit'e kadar hiçbir gerçek API çağrısı yapacak şekilde ÇALIŞTIRILMADI;
formal Gemini/Grok uyumluluk smoke'u "tamamlandı" sayılmaz.

Kullanım:
    uv run python scripts/compatibility_smoke.py prepare --name gemini_grok_smoke
    uv run python scripts/compatibility_smoke.py run --name gemini_grok_smoke --dry-run
    uv run python scripts/compatibility_smoke.py run --name gemini_grok_smoke      # ÜCRETLİ
    uv run python scripts/compatibility_smoke.py report --name gemini_grok_smoke

Gereksinim (yalnız gerçek `run`, dry-run/prepare/report DEĞİL):
    .env içinde OPENROUTER_API_KEY + DOĞRULANABİLİR, temiz bir git çalışma
    ağacı. Bir tur, manifestteki commit'e bağlıdır: aynı `--name` ile farklı
    bir committe run/resume yapılamaz (yeni bir isim gerekir).
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import LOGS_DIR  # noqa: E402  (config, .env'i yükler)
from eval.compatibility_smoke import (  # noqa: E402
    CompatibilitySmokeError,
    prepare as _prepare,
    report as _report,
    run as _run,
)


def _out_dir(name: str) -> Path:
    return LOGS_DIR / f"exp_{name}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="P1 formal Gemini/Grok uyumluluk smoke -- yalnız pilot; "
                     "ücretli run ayrıca onaylanmalıdır")
    sub = parser.add_subparsers(dest="stage", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--name", required=True,
                        help="smoke turu adı (çıktı: logs/exp_<name>/); devam için aynı adı ver")

    sub.add_parser("prepare", parents=[common],
                  help="OFFLINE: görev seçimi + manifest + plan (API anahtarı gerekmez)")

    p_run = sub.add_parser("run", parents=[common], help="bekleyen probları çalıştırır")
    p_run.add_argument("--dry-run", action="store_true",
                       help="hiçbir API çağrısı yapmaz; yalnız bekleyen probe listesini gösterir "
                            "(API anahtarı gerekmez)")

    sub.add_parser("report", parents=[common],
                  help="mevcut artefaktlardan compatibility_report.json üretir (API anahtarı gerekmez)")

    args = parser.parse_args()
    out_dir = _out_dir(args.name)

    try:
        if args.stage == "prepare":
            prepared = _prepare(args.name, out_dir)
            print(f"manifest -> {out_dir / 'manifest.json'}")
            print(f"plan     -> {out_dir / 'plan.json'} ({len(prepared['plan'])} probe)")
            print(f"gemini pilot görevleri: {prepared['manifest']['gemini_task_ids']}")
            print(f"grok pilot görevleri  : {prepared['manifest']['grok_task_ids']}")
        elif args.stage == "run":
            sonuc = _run(args.name, out_dir, dry_run=args.dry_run)
            if args.dry_run:
                print(f"[dry-run] {len(sonuc)} probe bekliyor (HİÇBİR API çağrısı YAPILMADI):")
                for spec in sonuc:
                    print(f"  {spec['probe_id']}")
            else:
                tamam = sum(r["status"] == "completed" for r in sonuc)
                print(f"{len(sonuc)} probe çalıştı ({tamam} tamamlandı) -> "
                      f"{out_dir / 'probes.jsonl'}")
        elif args.stage == "report":
            rapor = _report(args.name, out_dir)
            print(json.dumps(rapor, indent=2, ensure_ascii=False))
            print(f"\nRapor -> {out_dir / 'compatibility_report.json'}")
            for hedef, veri in rapor["targets"].items():
                print(f"  {hedef}: {veri['gate_decision']}"
                      + (f" -- {'; '.join(veri['blocker_reasons'])}"
                         if veri["blocker_reasons"] else ""))
    except CompatibilitySmokeError as e:
        sys.exit(str(e))


if __name__ == "__main__":
    main()
