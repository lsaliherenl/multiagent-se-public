"""LLM üretimi kodun izole çalıştırılması.

Kural (EXPERIMENT_PROTOCOL.md §12): üretilen kod ASLA ana süreçte exec() ile çalıştırılmaz.
Her çalıştırma ayrı bir Python subprocess'inde, timeout + kısıtlı ortam
değişkeni listesiyle yapılır. Windows'ta `resource` modülü olmadığı için
bellek/CPU sınırı yoktur; bu sınırlama güvenlik dokümantasyonunda açıkça
edilecek.

Bu İZOLE BİR SUBPROCESS'TİR, "güvenli sandbox" DEĞİLDİR — açık riskler:
  1. Çıktı boyutu sınırı (_clip) subprocess TAMAMEN bitip tüm stdout/stderr
     bellekte biriktikten SONRA uygulanır — sınırsız bellek tüketimine karşı
     korumasız.
  2. Dosya sistemi/ağ erişimi engellenmiyor — sadece timeout + ortam
     değişkeni kısıtlaması var.
  3. Windows'ta timeout sonrası proc.kill() SADECE doğrudan çocuk süreci
     öldürür; aday kod kendi alt süreç başlatırsa (torun süreç), timeout'ta
     yetim kalıp yaşamaya devam edebilir (process-tree kill yok).
"""

import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from config import SANDBOX_ENV_ALLOWLIST, SANDBOX_OUTPUT_LIMIT_BYTES, SANDBOX_TIMEOUT_S


@dataclass
class SandboxResult:
    status: str  # "ok" (exit 0) | "error" (exit != 0) | "timeout"
    stdout: str
    stderr: str
    duration_s: float
    exit_code: int | None  # timeout'ta None


def _clip(text: str | bytes | None, limit: int | None = None) -> str:
    if text is None:
        return ""
    if isinstance(text, bytes):
        text = text.decode("utf-8", errors="replace")
    limit = SANDBOX_OUTPUT_LIMIT_BYTES if limit is None else limit
    return text[:limit]


def _sandbox_env() -> dict[str, str]:
    """os.environ'ı SANDBOX_ENV_ALLOWLIST'e indirger. -I SADECE PYTHON*-önekli
    değişkenleri yok sayar, çalışma zamanında os.environ'ı TEMİZLEMEZ — bu
    yüzden env= burada elle kısıtlanıyor (ör. ANTHROPIC_API_KEY/OPENROUTER_API_KEY
    sızmasın diye). Windows'ta ortam değişkeni adları büyük/küçük harfe
    duyarsızdır — karşılaştırma .upper() ile yapılır."""
    allow = {name.upper() for name in SANDBOX_ENV_ALLOWLIST}
    return {k: v for k, v in os.environ.items() if k.upper() in allow}


def run_code(code: str, timeout_s: float = SANDBOX_TIMEOUT_S,
             output_limit_bytes: int | None = None) -> SandboxResult:
    """Verilen Python kaynağını izole bir subprocess'te çalıştırır (bkz. modül
    docstring'indeki açık riskler — bu "güvenli bir sandbox" değildir).

    output_limit_bytes: stdout/stderr kırpma sınırını geçici olarak büyütür.
    SADECE GÜVENİLEN kod için kullanılır (EvalPlus referans çözümünden beklenen
    çıktı üretimi, scripts/fetch_evalplus.py) — orada çıktı bilerek büyüktür
    (bine kadar test vakası) ve varsayılan sınır onu sessizce kesip görevi
    haksız yere reddettirirdi. LLM ÜRETİMİ ADAY KOD bu parametreyi ASLA
    kullanmaz; varsayılan sınır orada bilinçli bir korumadır.
    """
    with tempfile.TemporaryDirectory(prefix="mas_sandbox_") as tmpdir:
        script = Path(tmpdir) / "snippet.py"
        script.write_text(code, encoding="utf-8")
        start = time.monotonic()
        try:
            # -I: isolated mode (env değişkenleri ve user site-packages yok sayılır)
            proc = subprocess.run(
                [sys.executable, "-I", str(script)],
                capture_output=True,
                timeout=timeout_s,
                cwd=tmpdir,
                env=_sandbox_env(),
            )
        except subprocess.TimeoutExpired as exc:
            return SandboxResult(
                status="timeout",
                stdout=_clip(exc.stdout, output_limit_bytes),
                stderr=_clip(exc.stderr, output_limit_bytes),
                duration_s=time.monotonic() - start,
                exit_code=None,
            )
        return SandboxResult(
            status="ok" if proc.returncode == 0 else "error",
            stdout=_clip(proc.stdout, output_limit_bytes),
            stderr=_clip(proc.stderr, output_limit_bytes),
            duration_s=time.monotonic() - start,
            exit_code=proc.returncode,
        )
