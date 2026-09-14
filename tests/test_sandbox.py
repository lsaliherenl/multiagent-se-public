"""Sandbox birim testleri — LLM çağrısı yok, tamamen deterministik."""

from eval.sandbox import run_code


def test_basari_exit_0():
    r = run_code("print('merhaba')")
    assert r.status == "ok"
    assert r.exit_code == 0
    assert "merhaba" in r.stdout


def test_hata_exit_nonzero():
    r = run_code("raise ValueError('patladi')")
    assert r.status == "error"
    assert r.exit_code != 0
    assert "ValueError" in r.stderr


def test_sonsuz_dongu_timeout():
    r = run_code("while True: pass", timeout_s=2)
    assert r.status == "timeout"
    assert r.exit_code is None
    assert r.duration_s >= 2


def test_cikti_boyut_siniri():
    r = run_code("print('x' * 100_000)")
    assert r.status == "ok"
    assert len(r.stdout) <= 10_000


def test_izolasyon_ayri_surec():
    # Sandbox'taki kod ana süreci etkileyemez (ör. exit çağrısı)
    r = run_code("import sys; sys.exit(7)")
    assert r.status == "error"
    assert r.exit_code == 7


def test_ust_surec_ortam_degiskeni_sizmiyor(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-secret-should-not-leak")
    r = run_code("import os; print(repr(os.environ.get('ANTHROPIC_API_KEY')))")
    assert r.status == "ok"
    assert "sk-test-secret-should-not-leak" not in r.stdout
    assert "None" in r.stdout


def test_temel_ortam_degiskenleri_korunuyor():
    r = run_code("import os; print(os.environ.get('SystemRoot'))")
    assert r.status == "ok"
    assert r.stdout.strip() not in ("", "None")
