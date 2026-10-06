"""Takip analizi (Study 1B / Study 2): hücreler arası tahminler.

Her model-çalışma hücresi `analysis/analyze.py` ile AYRI analiz edilir (aynı
görev-düzeyi eşleştirilmiş fark ve 10.000 yinelemeli bootstrap). Bu modül
orada BULUNMAYAN dört şeyi ekler (EXPERIMENT_PROTOCOL.md §13):

1. **Rejim havuzlama reddi.** Study 1B (`heldout`) ile Study 2
   (`study2_complex`) kayıtları aynı çağrıda karıştırılamaz; iki rejim farklı
   görev evrenleridir.
2. **Çözümlenmemiş `run_error` reddi.** Bir kimliğin tek kaydı `run_error`
   ise analiz durur; bayrakla geçilemez.
3. **Moderasyon bootstrap'ı — BAĞIMSIZ.** `M[m] = Delta[m,Study2] -
   Delta[m,Study1B]`. İki rejimin görevleri farklı olduğu için her iterasyonda
   iki görev kümesi AYRI AYRI yeniden örneklenir.
4. **Köprü bootstrap'ı — EŞLENİK.** Study 1A ile Study 1B Gemini aynı 50
   görevde koşuldu; her iterasyonda aynı `task_id` İKİ çalışmada BİRLİKTE
   seçilir.

Çıktılarda p-değeri veya anlamlılık alanı bulunamaz. Ağ, model, evaluator ve
Docker çağrısı yoktur.

Kullanım (her dizin bir `logs/exp_<name>/` deney dizinidir):

    uv run python -m analysis.followup \
        --study1a-gemini logs/exp_gemini_main \
        --study1b-gemini logs/exp_study1b_gemini --study1b-luna logs/exp_study1b_luna \
        --study2-gemini logs/exp_study2_gemini --study2-luna logs/exp_study2_luna
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from config import (
    ANALYSIS_METRICS,
    ARM_CONTRACT,
    ARM_NAIVE,
    BOOTSTRAP_CI_LEVEL,
    BOOTSTRAP_ITERATIONS,
    BOOTSTRAP_SEED,
    PRIMARY_METRIC,
    SECONDARY_COMPARISONS,
    STUDY1B,
    STUDY2,
)
from analysis import analyze as az
from analysis.analyze import (
    AnalysisError,
    _mean,
    _percentile,
    paired_diffs,
    single_model,
)
from eval.result_schema import is_run_error

SCHEMA_VERSION = "1.0"

# Görev kümesi → rejim. Rejim, görev evreninin kimliğidir; model değildir.
REGIME_BY_TASK_SET = {
    "heldout": STUDY1B,
    "study2_complex": STUDY2,
}

# Analiz edilebilir rejimler. Study 3 ertelendi; burada YOKTUR.
ANALYSABLE_REGIMES = (STUDY1B, STUDY2)

# Bir çıktı gövdesinde bulunması yasak alan adları. Ön-kayıt p-değeri
# üretmemeyi taahhüt eder; taahhüt bir yorum değil, kapı olmalıdır.
FORBIDDEN_INFERENCE_FIELDS = frozenset({
    "p_value", "pvalue", "p", "significance", "significant", "alpha_level",
    "reject_null", "hypothesis_test", "t_statistic", "z_statistic",
})


class FollowupAnalysisError(RuntimeError):
    """Takip analizi sözleşmesi ihlal edildi (fail-closed)."""


# --------------------------------------------------------------------------
# A. Havuzlama reddi
# --------------------------------------------------------------------------

def regime_of(task_set: str) -> str:
    """Görev kümesinin rejimi. Bilinmeyen küme sessizce kabul EDİLMEZ."""
    regime = REGIME_BY_TASK_SET.get(task_set)
    if regime is None:
        raise FollowupAnalysisError(
            f"bilinmeyen görev kümesi: {task_set!r} "
            f"(izinli: {sorted(REGIME_BY_TASK_SET)})")
    return regime


def assert_single_regime(records: list[dict], manifest: dict | None = None) -> str:
    """Kayıtlar TEK rejime ait olmalı; iki rejim havuzlanamaz.

    Manifest verilirse ayrıca manifest–kayıt rejim uyumu aranır: kendi içinde
    tutarlı bir kayıt kümesi, DOĞRU rejimin kümesi demek değildir.
    """
    if not records:
        raise FollowupAnalysisError("rejim belirlenemez: kayıt yok")
    task_sets = sorted({r.get("task_set") for r in records})
    if len(task_sets) != 1:
        raise FollowupAnalysisError(
            f"tek rejimli analiz bekleniyor, {len(task_sets)} görev kümesi "
            f"bulundu: {task_sets}. Study 1B ve Study 2 HAVUZLANMAZ.")
    regime = regime_of(task_sets[0])
    if manifest is not None:
        manifest_regime = regime_of(manifest["task_set"])
        if manifest_regime != regime:
            raise FollowupAnalysisError(
                f"manifest rejimi {manifest_regime!r}, kayıtların rejimi "
                f"{regime!r} — analiz durduruldu.")
        declared = manifest.get("study_id")
        if declared is not None and declared != regime:
            raise FollowupAnalysisError(
                f"manifest study_id={declared!r} ile görev kümesinden türeyen "
                f"rejim {regime!r} çelişiyor.")
    return regime


def assert_single_model(records: list[dict]) -> str:
    """Model havuzlama reddi — Study 1A'daki kapının aynısı, yeniden yazılmaz."""
    return single_model(records)


# --------------------------------------------------------------------------
# B. Çözümlenmemiş `run_error` — mazur görülemez
# --------------------------------------------------------------------------

def unresolved_run_error_identities(records: list[dict], manifest: dict) -> list:
    """Tek kaydı `run_error` olan (task_id, arm, repeat) kimlikleri.

    `check_integrity` bunu `missing` olarak görür ve `allow_missing` ile
    geçilebilir. Ayrımı burada AÇIKÇA kurmak gerekir: "hiç koşulmamış" ile
    "koşulmuş ve altyapı hatası almış, yerine geçerli kayıt konmamış" farklı
    olaylardır ve ikincisi bayrakla geçilemez.
    """
    valid, failed = set(), set()
    for record in records:
        key = (record.get("task_id"), record.get("arm"), record.get("repeat"))
        (failed if is_run_error(record) else valid).add(key)
    expected = {(t, a, k)
                for t in manifest["task_ids"]
                for a in manifest["arm_order"]
                for k in range(manifest["repeats"])}
    return sorted(k for k in failed - valid if k in expected)


def assert_no_unresolved_run_error(records: list[dict], manifest: dict) -> None:
    unresolved = unresolved_run_error_identities(records, manifest)
    if unresolved:
        raise FollowupAnalysisError(
            f"{len(unresolved)} çözümlenmemiş `run_error` kimliği var; takip "
            f"analizi bunları bayrakla bile geçemez: {unresolved[:3]}")


# --------------------------------------------------------------------------
# C. Bootstrap — iki AYRI eşleme kuralı
# --------------------------------------------------------------------------

def _ci(draws: list[float], *, ci_level: float) -> dict:
    ordered = sorted(draws)
    alpha = (1 - ci_level) / 2
    return {"ci_low": _percentile(ordered, alpha),
            "ci_high": _percentile(ordered, 1 - alpha)}


def independent_moderation_draws(diffs_a: list[float], diffs_b: list[float], *,
                                 iterations: int, seed: int) -> list[float]:
    """İki AYRI görev evreninin BAĞIMSIZ yeniden örneklenmesi.

    Her iterasyonda önce A rejiminin görevleri, sonra B rejiminin görevleri
    kendi içlerinde replacement ile seçilir. İki seçim birbirine bağlanmaz:
    görevler farklı olduğu için eşleme yapmak sahte bir korelasyon üretir ve
    farkın aralığını yanlış DARALTIR.
    """
    if not diffs_a or not diffs_b:
        raise FollowupAnalysisError("moderasyon için iki rejimde de görev gerekir")
    rng = random.Random(seed)
    na, nb = len(diffs_a), len(diffs_b)
    draws = []
    for _ in range(iterations):
        mean_a = _mean([diffs_a[rng.randrange(na)] for _ in range(na)])
        mean_b = _mean([diffs_b[rng.randrange(nb)] for _ in range(nb)])
        draws.append(mean_a - mean_b)
    return draws


def moderation_effect(diffs_study2: list[float], diffs_study1b: list[float], *,
                      model: str,
                      iterations: int = BOOTSTRAP_ITERATIONS,
                      seed: int = BOOTSTRAP_SEED,
                      ci_level: float = BOOTSTRAP_CI_LEVEL) -> dict:
    """M[m] = Delta[m, Study 2] − Delta[m, Study 1B]. Eşiksiz, p-değersiz.

    "Birinde anlamlı, diğerinde anlamsız" bir moderasyon kanıtı DEĞİLDİR; bu
    yüzden çıktı iki bacağın ayrı anlamlılığını taşımaz, yalnız farkın nokta
    tahminini ve aralığını verir.
    """
    draws = independent_moderation_draws(diffs_study2, diffs_study1b,
                                         iterations=iterations, seed=seed)
    effect = {
        "estimand_id": "S2_VS_S1B_MODERATION",
        "model": model,
        "definition": "Delta[m, Study 2] - Delta[m, Study 1B]",
        "resampling": "independent_two_regime_task_cluster",
        "n_tasks_study2": len(diffs_study2),
        "n_tasks_study1b": len(diffs_study1b),
        "delta_study2": _mean(diffs_study2),
        "delta_study1b": _mean(diffs_study1b),
        "point_estimate": _mean(diffs_study2) - _mean(diffs_study1b),
        "iterations": iterations, "seed": seed, "ci_level": ci_level,
        "threshold_used": None,
        "significance_by_eye_forbidden": True,
        "status": "SECONDARY_DESCRIPTIVE",
    }
    effect.update(_ci(draws, ci_level=ci_level))
    return effect


def coupled_bridge_draws(diffs_x: list[float], diffs_y: list[float], *,
                         iterations: int, seed: int) -> list[float]:
    """AYNI görev kimliklerinin İKİ çalışmada BİRLİKTE seçilmesi.

    `diffs_x[i]` ve `diffs_y[i]` aynı `task_id`'ye aittir. Her iterasyonda TEK
    bir indeks listesi çekilir ve İKİ vektöre de uygulanır; böylece görev
    zorluğunun ortak bileşeni farktan düşer.
    """
    if len(diffs_x) != len(diffs_y):
        raise FollowupAnalysisError(
            f"eşlenik köprü aynı görev sayısını gerektirir: "
            f"{len(diffs_x)} != {len(diffs_y)}")
    if not diffs_x:
        raise FollowupAnalysisError("eşlenik köprü için en az bir görev gerekir")
    rng = random.Random(seed)
    n = len(diffs_x)
    draws = []
    for _ in range(iterations):
        idx = [rng.randrange(n) for _ in range(n)]
        draws.append(_mean([diffs_y[i] for i in idx])
                     - _mean([diffs_x[i] for i in idx]))
    return draws


def paired_bridge_effect(diffs_study1a: list[float], diffs_study1b: list[float], *,
                         task_ids: list[str],
                         iterations: int = BOOTSTRAP_ITERATIONS,
                         seed: int = BOOTSTRAP_SEED,
                         ci_level: float = BOOTSTRAP_CI_LEVEL) -> dict:
    """Study 1A ↔ Study 1B Gemini eşlenik köprüsü (aynı 50 görev)."""
    if len(task_ids) != len(diffs_study1a):
        raise FollowupAnalysisError("köprü görev listesi fark vektörüyle uyuşmuyor")
    draws = coupled_bridge_draws(diffs_study1a, diffs_study1b,
                                 iterations=iterations, seed=seed)
    effect = {
        "estimand_id": "S1A_VS_S1B_GEMINI_PAIRED",
        "model": "gemini",
        "definition": "(d_Study1B[t] - d_Study1A[t]) gorev-duzeyi eslenik farki",
        "resampling": "coupled_same_task_ids_in_both_studies",
        "n_tasks": len(task_ids),
        "delta_study1a": _mean(diffs_study1a),
        "delta_study1b": _mean(diffs_study1b),
        "point_estimate": _mean(diffs_study1b) - _mean(diffs_study1a),
        "iterations": iterations, "seed": seed, "ci_level": ci_level,
        "luna_or_study2_mixed_in": False,
        "interpretation_limit":
            "Zaman, saglayici rotasi ve olasi model snapshot degisimini "
            "birbirinden AYIRMAZ; nedensel 'model yukseltmesi etkisi' olarak "
            "okunamaz.",
        "status": "SECONDARY_DESCRIPTIVE",
    }
    effect.update(_ci(draws, ci_level=ci_level))
    return effect


def bridge_task_ids(manifest_a: dict, manifest_b: dict) -> list[str]:
    """İki çalışmanın ORTAK ve BİREBİR aynı görev listesi; değilse durur."""
    a = sorted(manifest_a.get("task_ids") or [])
    b = sorted(manifest_b.get("task_ids") or [])
    if not a or a != b:
        raise FollowupAnalysisError(
            "eşlenik köprü aynı görev kimliklerini gerektirir; listeler farklı")
    if manifest_a.get("model") != manifest_b.get("model"):
        raise FollowupAnalysisError(
            "eşlenik köprü tek modelde kurulur; iki manifest farklı model taşıyor")
    return a


# --------------------------------------------------------------------------
# D. p-değeri kapısı
# --------------------------------------------------------------------------

def _walk(value, path=""):
    if isinstance(value, dict):
        for key, sub in value.items():
            yield f"{path}/{key}", key, sub
            yield from _walk(sub, f"{path}/{key}")
    elif isinstance(value, (list, tuple)):
        for index, sub in enumerate(value):
            yield from _walk(sub, f"{path}/{index}")


def scan_for_inference_fields(payload) -> list[str]:
    """Yasak çıkarım alanlarını arar. Boş liste = temiz."""
    return [f"{path}: yasak çıkarım alanı {key!r}"
            for path, key, _ in _walk(payload)
            if key in FORBIDDEN_INFERENCE_FIELDS]


def assert_no_p_value(payload) -> None:
    findings = scan_for_inference_fields(payload)
    if findings:
        raise FollowupAnalysisError(
            "Takip analizi çıktısı p-değeri/eşik alanı taşıyor: " + "; ".join(findings[:5]))


def branch_diffs(records: list[dict], manifest: dict, *, arm_a: str, arm_b: str,
                 metric: str) -> list[float]:
    """Tek dalın görev-düzeyi eşleştirilmiş fark vektörü.

    Kapı sırası: rejim → model → çözümlenmemiş `run_error` → oranlar. Ters
    sırada, yanlış rejimin verisi üzerinde `run_error` raporlanırdı.
    """
    if metric not in ANALYSIS_METRICS:
        raise FollowupAnalysisError(f"bilinmeyen metrik: {metric!r}")
    from analysis.analyze import task_level_rates
    assert_single_regime(records, manifest)
    assert_single_model(records)
    assert_no_unresolved_run_error(records, manifest)
    rates = task_level_rates(records, task_ids=manifest["task_ids"],
                             arms=manifest["arm_order"], metric=metric)
    return paired_diffs(rates, arm_a, arm_b, manifest["task_ids"])




# --------------------------------------------------------------------------
# E. Hücre ve hücreler-arası tahminler
# --------------------------------------------------------------------------

CELLS = ("study1a_gemini", "study1b_gemini", "study1b_luna",
         "study2_gemini", "study2_luna")


def analyse_cell(exp_dir: Path, *, iterations: int = BOOTSTRAP_ITERATIONS,
                 seed: int = BOOTSTRAP_SEED) -> dict:
    """Tek hücrenin analizi; havuzlama ve run_error kapıları ÖNCE koşar."""
    manifest, records, calls = az.load_experiment(exp_dir)
    if manifest.get("study_id") in ANALYSABLE_REGIMES:
        assert_single_regime(records, manifest)
    assert_single_model(records)
    assert_no_unresolved_run_error(records, manifest)
    return az.analyze(manifest, records, calls, iterations=iterations, seed=seed)


def _cell_diffs(result: dict, arm_a: str, arm_b: str, metric: str) -> list[float]:
    rates = result["_task_rates"][metric]
    return paired_diffs(rates, arm_a, arm_b, sorted(rates))


def cross_cell_estimands(cells: dict, *, iterations: int = BOOTSTRAP_ITERATIONS,
                         seed: int = BOOTSTRAP_SEED,
                         ci_level: float = BOOTSTRAP_CI_LEVEL) -> dict:
    """Makaledeki tahmin hiyerarşisi; hücreler HAVUZLANMAZ.

    `cells` anahtarları `CELLS` içindendir; değerleri `analyse_cell` sonucudur.
    Eksik hücreye bağlı tahmin üretilmez ve `skipped` altında adıyla listelenir.
    """
    def effect(cell, arm_a, arm_b, metric=PRIMARY_METRIC):
        rates = cells[cell]["_task_rates"][metric]
        return az.paired_effect(rates, arm_a, arm_b, sorted(rates),
                                iterations=iterations, seed=seed)

    out = {"primary_confirmatory": None, "replication": None,
           "mechanism": {}, "moderation": {}, "bridge": None, "skipped": []}
    if "study2_gemini" in cells:
        out["primary_confirmatory"] = {
            "estimand_id": "S2_DIRECT_GEMINI", "cell": "study2_gemini",
            "contrast": "contract - naive",
            **effect("study2_gemini", ARM_CONTRACT, ARM_NAIVE)}
    else:
        out["skipped"].append("S2_DIRECT_GEMINI")
    if "study2_luna" in cells:
        out["replication"] = {
            "estimand_id": "S2_DIRECT_LUNA", "cell": "study2_luna",
            "contrast": "contract - naive", "pooled_with_gemini": False,
            **effect("study2_luna", ARM_CONTRACT, ARM_NAIVE)}
    else:
        out["skipped"].append("S2_DIRECT_LUNA")
    for arm_a, arm_b in [(ARM_CONTRACT, ARM_NAIVE), *SECONDARY_COMPARISONS]:
        key = f"{arm_a}_vs_{arm_b}"
        out["mechanism"][key] = {
            cell: effect(cell, arm_a, arm_b) for cell in CELLS if cell in cells}
    for model in ("gemini", "luna"):
        s2, s1b = f"study2_{model}", f"study1b_{model}"
        if s2 in cells and s1b in cells:
            out["moderation"][model] = moderation_effect(
                _cell_diffs(cells[s2], ARM_CONTRACT, ARM_NAIVE, PRIMARY_METRIC),
                _cell_diffs(cells[s1b], ARM_CONTRACT, ARM_NAIVE, PRIMARY_METRIC),
                model=model, iterations=iterations, seed=seed, ci_level=ci_level)
        else:
            out["skipped"].append(f"S2_VS_S1B_MODERATION[{model}]")
    if "study1a_gemini" in cells and "study1b_gemini" in cells:
        a, b = cells["study1a_gemini"], cells["study1b_gemini"]
        task_ids = bridge_task_ids(
            {"task_ids": sorted(a["_task_rates"][PRIMARY_METRIC]), "model": a["model"]},
            {"task_ids": sorted(b["_task_rates"][PRIMARY_METRIC]), "model": b["model"]})
        out["bridge"] = paired_bridge_effect(
            _cell_diffs(a, ARM_CONTRACT, ARM_NAIVE, PRIMARY_METRIC),
            _cell_diffs(b, ARM_CONTRACT, ARM_NAIVE, PRIMARY_METRIC),
            task_ids=task_ids, iterations=iterations, seed=seed, ci_level=ci_level)
    else:
        out["skipped"].append("S1A_VS_S1B_GEMINI_PAIRED")
    assert_no_p_value(out)
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Takip analizi (hücreler arası)")
    for cell in CELLS:
        parser.add_argument(f"--{cell.replace('_', '-')}", type=Path, default=None,
                            help=f"{cell} deney dizini (logs/exp_<name>)")
    parser.add_argument("--out", type=Path, default=Path("logs/followup_analysis.json"))
    args = parser.parse_args(argv)
    dirs = {cell: getattr(args, cell) for cell in CELLS if getattr(args, cell)}
    if not dirs:
        parser.error("en az bir hücre dizini verilmeli")
    try:
        cells = {cell: analyse_cell(path) for cell, path in dirs.items()}
        estimands = cross_cell_estimands(cells)
    except (AnalysisError, FollowupAnalysisError) as exc:
        print(f"analiz durduruldu: {exc}", file=sys.stderr)
        return 1
    payload = {
        "schema_version": SCHEMA_VERSION,
        "cells": {cell: {k: v for k, v in r.items() if not k.startswith("_")}
                  for cell, r in cells.items()},
        "estimands": estimands,
    }
    assert_no_p_value(payload)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8")
    print(f"yazıldı: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
