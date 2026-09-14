"""call_model() smoke testi: seçilen modelle tek çağrı.

Bu, basit bir geliştirici tanısıdır (tek çağrı, manifest/resume/provenance
sözleşmesi YOK). P1'in FORMAL Gemini/Grok uyumluluk smoke'u için
`scripts/compatibility_smoke.py`'ye bakın (dondurulmuş 12+12 çağrılık
deterministik matris, gerçek planner/coder/adjudicator yolları, manifest +
resume + `compatibility_report.json`).

Doğruladıkları: (a) anahtar/ağ zinciri çalışıyor, (b) çağrı logu yazılıyor,
(c) model logprobs döndürüyor mu (belirsizlik analizi için önemli),
(d) config'teki ortak parametreler (max_tokens + reasoning + provider routing)
    sağlayıcı tarafından REDDEDİLMİYOR — özellikle require_parameters=True ile
    logprobs birlikte gönderildiğinde uygun provider kalmama riski var,
(e) ortak reasoning ayarının (2026-07-30'dan beri AÇIK — Gemini endpoint'i
    zorunlu kılıyor) fiilen ne kadar reasoning tokeni harcattığı.

Kullanım:
    uv run python scripts/smoke_llm.py --model main
    uv run python scripts/smoke_llm.py --model secondary
    uv run python scripts/smoke_llm.py --model dev --json   # JSON mode kontrolü
Gereksinim: .env içinde OPENROUTER_API_KEY
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from agents.llm import CALL_LOG, call_model  # config, .env'i yükler
from config import model_alias_help, resolve_model


def main() -> None:
    parser = argparse.ArgumentParser(description="call_model() smoke testi")
    # Görev seti yok: bu script görev çalıştırmaz, tek bir sabit prompt gönderir.
    # Held-out kapısı bu yüzden uygulanmaz; tanımsız bir slug'la uyumluluk
    # smoke'u yapılabilmesi bilinçlidir.
    parser.add_argument("--model", default="dev", help=model_alias_help())
    parser.add_argument("--json", action="store_true",
                        help="response_format=json_object ile dene (Kol 3/structured ön koşulu)")
    args = parser.parse_args()

    if not os.environ.get("OPENROUTER_API_KEY"):
        sys.exit("OPENROUTER_API_KEY bulunamadı — .env.example'ı .env olarak kopyalayıp doldur.")

    model = resolve_model(args.model)
    kwargs = {"response_format": {"type": "json_object"}} if args.json else {}
    prompt = ('Reply with a JSON object: {"ok": true}' if args.json
              else "Reply with exactly one word: OK")

    r = call_model([{"role": "user", "content": prompt}],
                   model=model, task_id="smoke", agent_role="smoke", **kwargs)
    print(f"istenen model : {r.model}")
    print(f"donen model   : {r.actual_model}")
    print(f"saglayici     : {r.provider}")
    print(f"yanit         : {r.text.strip()[:80]}")
    print(f"tokenlar      : {r.input_tokens} giris / {r.output_tokens} cikis")
    print(f"reasoning tok : {r.reasoning_tokens}  (reasoning ACIK -> pozitif olabilir)")
    print(f"cached tok    : {r.cached_tokens}")
    print(f"finish_reason : {r.finish_reason}")
    print(f"maliyet       : {r.cost_usd}")
    print(f"gecikme       : {r.latency_s:.2f}s")
    print(f"logprobs      : {'VAR (' + str(len(r.logprobs)) + ' token)' if r.logprobs else 'YOK (saglayici dondurmedi)'}")
    print(f"cagri logu    : {CALL_LOG}")


if __name__ == "__main__":
    main()
