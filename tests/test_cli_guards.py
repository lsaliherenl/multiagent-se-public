"""CLI giriş kapıları — LLM'siz, deterministik.

Kapı iki katmanlıdır ve ikisi de burada sınanır:

1. **Held-out üretici kapısı** (`config.validate_model_for_task_set`): ana koşu
   CLI'ları (`eval/runner.py`, `uncertainty/self_consistency.py`) held-out sette
   yalnız `MODEL_PRODUCERS`'ı kabul eder. Kontrol API anahtarı doğrulamasından,
   görev yüklemesinden ve çıktı dizini oluşturmadan ÖNCE çalışır — yanlış
   modelle açılmış bir deney dizini/manifesti geride kalmasın diye.
2. **Debug CLI kısıtı**: `pipeline/baseline.py` ve `pipeline/run_graph.py`
   manifest/provenance/resume üretmez; held-out'u seçenek olarak dahi sunmazlar.

Testler yalnız argparse ve kapı mantığını çalıştırır; hiçbir gerçek LLM çağrısı
yapılmaz (kapılar zaten çağrı katmanından önce durur).
"""

import importlib.util
import sys
from pathlib import Path

import pytest

import config
import pipeline.baseline as baseline_cli
import pipeline.run_graph as run_graph_cli
import uncertainty.self_consistency as selfcons_cli
from eval import runner as runner_cli

# scripts/ bir paket değil; dosyadan yükle.
_spec = importlib.util.spec_from_file_location(
    "smoke_llm", Path(__file__).parent.parent / "scripts" / "smoke_llm.py")
smoke_llm = importlib.util.module_from_spec(_spec)
sys.modules["smoke_llm"] = smoke_llm
_spec.loader.exec_module(smoke_llm)

ANA_KOSU_CLILARI = [
    (runner_cli, ["--name", "x", "--task-set", "heldout", "--model"]),
    (selfcons_cli, ["--name", "x", "--task-set", "heldout", "--model"]),
]
BUTUN_CLILAR = [runner_cli, selfcons_cli, baseline_cli, run_graph_cli, smoke_llm]


def _cli_calistir(monkeypatch, modul, argv: list[str]):
    monkeypatch.setattr(sys, "argv", ["prog", *argv])
    with pytest.raises(SystemExit) as exc:
        modul.main()
    return exc.value


# --- Held-out üretici kapısı -------------------------------------------------

@pytest.mark.parametrize("modul,onek", ANA_KOSU_CLILARI)
@pytest.mark.parametrize("model", [
    "openrouter/minimax/minimax-m3",   # üretici-dışı judge
    "openrouter/x-ai/grok-4.3",        # panel-dışı adjudicator
    "openrouter/baska/model",          # tanımsız slug
])
def test_heldout_uretici_disi_model_cagri_oncesi_reddedilir(
    monkeypatch, modul, onek, model
):
    # Tam slug alias tablosunu tamamen atlar; kapı bu yüzden çözülmüş slug
    # üzerinde ve CLI'nın en başında durmalı.
    exc = _cli_calistir(monkeypatch, modul, [*onek, model])
    assert "held-out" in str(exc.code)


@pytest.mark.parametrize("modul,onek", ANA_KOSU_CLILARI)
def test_heldout_kapisi_api_anahtarindan_once_calisir(monkeypatch, modul, onek):
    # Anahtar YOKKEN bile hata mesajı model kapısından gelmeli: sıra tersine
    # dönerse, anahtarı olan bir makinede yanlış model sessizce ilerlerdi.
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    exc = _cli_calistir(monkeypatch, modul, [*onek, config.MODEL_JUDGE_EXTERNAL])
    assert "held-out" in str(exc.code)
    assert "API anahtarı" not in str(exc.code)


@pytest.mark.parametrize("model", [config.MODEL_MAIN, config.MODEL_SECONDARY, "main", "dev"])
def test_uretici_modeller_heldoutta_kapidan_gecer(model):
    # Kabul yolu CLI'yı fiilen koşturmadan doğrulanır (koşmak API çağrısı
    # gerektirirdi); kapı fonksiyonu CLI'ların kullandığı fonksiyonun aynısıdır.
    assert config.validate_model_for_task_set(model, "heldout") in config.MODEL_PRODUCERS


def test_selfcons_name_zorunlu_ve_gorev_yuklemeden_reddedilir(monkeypatch):
    monkeypatch.setattr(
        selfcons_cli, "load_all_tasks",
        lambda *args, **kwargs: pytest.fail("--name yokken görev yüklenmemeli"),
    )
    exc = _cli_calistir(
        monkeypatch, selfcons_cli,
        ["--task-set", "pilot", "--model", config.MODEL_MAIN],
    )
    assert exc.code == 2


# --- Debug CLI'ları held-out sunmaz ------------------------------------------

@pytest.mark.parametrize("modul,argv", [
    (baseline_cli, ["--task-set", "heldout"]),
    (run_graph_cli, ["--mode", "naive", "--task-set", "heldout"]),
])
def test_debug_clilari_heldout_secenegi_sunmaz(monkeypatch, modul, argv):
    exc = _cli_calistir(monkeypatch, modul, argv)
    assert exc.code == 2  # argparse: geçersiz seçenek


@pytest.mark.parametrize("modul", [baseline_cli, run_graph_cli])
def test_debug_clilarinin_gorev_seti_secenegi_yalnizca_pilot(monkeypatch, modul, capsys):
    _cli_calistir(monkeypatch, modul, ["--help"])
    yardim = capsys.readouterr().out
    assert "--task-set {pilot}" in yardim


# --- Yardım metinleri --------------------------------------------------------

@pytest.mark.parametrize("modul", BUTUN_CLILAR)
def test_yardim_metinlerinde_upgrade_kalmadi(monkeypatch, modul, capsys):
    # `upgrade` takma adı 2026-07-29'da kadrodan çıktı; yardım metninde kalması
    # kullanıcıyı çözülmeyen bir slug'a yönlendirirdi.
    _cli_calistir(monkeypatch, modul, ["--help"])
    assert "upgrade" not in capsys.readouterr().out


@pytest.mark.parametrize("modul", [runner_cli, selfcons_cli, baseline_cli,
                                   run_graph_cli, smoke_llm])
def test_yardim_metinleri_alias_listesini_config_ten_alir(monkeypatch, modul, capsys):
    _cli_calistir(monkeypatch, modul, ["--help"])
    yardim = " ".join(capsys.readouterr().out.split())
    assert "|".join(config.MODEL_ALIASES) in yardim
