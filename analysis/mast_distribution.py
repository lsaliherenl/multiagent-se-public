"""MAST hata dağılımı — İKİ AYRI PAYDA ile (§2 RQ5, §8.5).

Ana analizden (analysis/analyze.py) ayrıdır: MAST etiketleri HENÜZ YOKKEN de
birincil performans analizi çalışabilmelidir. Girdiler açıkça verilir; dizin
tahmin edilmez.

**Neden iki payda?** Tek bir koşullu yüzde ("bu koldaki hataların %X'i mod 1.1")
farklı sayıda başarısızlığa sahip kolları YANILTICI biçimde karşılaştırılabilir
gösterir: 40 hatanın 20'si 1.1 olan bir kol ile 2 hatanın 1'i 1.1 olan bir kol
aynı "%50" değerini alır. Bu yüzden her mod iki oranla raporlanır:

- `error_composition` = (kol, mod) sayısı / o kolda KARARA BAĞLANMIŞ hata sayısı
  — "bu kolun hataları nasıl dağılıyor" (kompozisyon).
- `arm_incidence`   = (kol, mod) sayısı / o koldaki BÜTÜN tamamlanmış arm-run
  sayısı — "bu kolda bu hata ne sıklıkta oluyor" (insidans). Kollar arası
  karşılaştırma için doğru olan budur.

**Karar etiketi leave-self-out'tur (§9.1-§9.2).** Bir kaydın MAST modu:

1. iki DIŞ judge uzlaştıysa `external_consensus_label`,
2. dış judge'lar ayrıştıysa GÜNCEL Grok adjudication kararı,
3. panel eksikse ya da güncel adjudication yoksa **kararsız** (hiçbir modun
   payına girmez, ayrıca sayılır).

Üçlü `majority_label` ve kaydı üreten modelin kendi (`self`) etiketi yalnız
TANISALDIR ve hiçbir kompozisyon oranının payına girmez (§8.5).

`insufficient_context` bir MAST modu DEĞİLDİR — etiket verilememesidir. Bu yüzden
mod tablolarına karışmaz, kendi sayısı ve kendi paydasıyla ayrı raporlanır.

**İnsan turu bu modülün kapsamı DIŞINDADIR.** İnsan–AI uyumu, seçilimli 30
kayıtlık örneklem üzerinden `eval/mast_human.py::build_agreement_summary()`
tarafından üretilir ve o oranlar bütün hata evrenine genellenemez (§8.5); burada
üretilen sayılar ise TÜM etiketlenmiş başarısızlıkların evrenidir.

Kullanım:
    uv run python -m analysis.mast_distribution --exp logs/exp_gemini_main \\
        --mast logs/exp_gemini_main/mast
"""

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

from config import (
    MAST_DECISION_RULE_VERSION,
    MAST_DISTRIBUTION_SCHEMA_VERSION,
    MAST_PANEL_HASH_VERSION,
    MAST_SCHEMA_VERSION,
    MODEL_ADJUDICATOR,
)
from analysis.analyze import AnalysisError, load_experiment, single_model
from eval.mast_labels import (
    labelable_records,
    prompt_contract_hash,
    source_manifest_fingerprint,
    verify_adjudications,
    verify_panel,
)
from eval.mast_schema import INSUFFICIENT_SENTINEL, MastPipelineError
from eval.result_schema import check_provenance, integrity_report, is_run_error

__all__ = [
    "MastDistributionError", "load_mast", "check_mast_provenance",
    "decision_label", "build_distribution", "write_distribution_outputs",
]

DECISION_EXTERNAL_CONSENSUS = "external_consensus"
DECISION_ADJUDICATED = "grok_adjudicated"
DECISION_UNDECIDED_SPLIT = "undecided_split"
DECISION_UNDECIDED_INCOMPLETE = "undecided_incomplete_panel"


class MastDistributionError(RuntimeError):
    """MAST dağılımı güvenle üretilemiyor — yanlış saymaktansa durulur."""


def _canonical_hash(value) -> str | None:
    """Uzun provenance yapılarının kompakt kimliği (yok ise None)."""
    if value is None:
        return None
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False,
                         separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in
            path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_mast(mast_dir: Path) -> tuple[dict, list[dict], list[dict], list[dict]]:
    """(mast manifest, judge etiketleri, panel satırları, adjudication satırları).

    `ai_judges.jsonl` DE yüklenir: panel türetilmiş bir görünümdür ve tazeliği
    ancak ham etiketlerden yeniden hesaplanarak doğrulanabilir. Yalnız paneli
    okuyup içindeki `evidence_consistent=True` bayrağına güvenmek, panelin kendi
    iddiasını kanıt olarak kabul etmek olurdu.
    """
    if mast_dir.is_file():
        raise MastDistributionError(f"MAST girdisi bir DİZİN olmalı: {mast_dir}")
    manifest_path = mast_dir / "manifest.json"
    if not manifest_path.exists():
        raise MastDistributionError(
            f"MAST manifesti yok: {manifest_path} — etiketleme turunun "
            "yapılandırması bilinmeden dağılım raporlanamaz.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    panel = _load_jsonl(mast_dir / "ai_panel.jsonl")
    if not panel:
        raise MastDistributionError(
            f"panel kaydı yok: {mast_dir / 'ai_panel.jsonl'} — önce "
            "`--stage judge` çalıştırılmalı.")
    judges = _load_jsonl(mast_dir / "ai_judges.jsonl")
    if not judges:
        raise MastDistributionError(
            f"judge etiketi yok: {mast_dir / 'ai_judges.jsonl'} — panel tazeliği "
            "ham etiketler olmadan doğrulanamaz.")
    return manifest, judges, panel, _load_jsonl(mast_dir / "ai_adjudication.jsonl")


def check_mast_provenance(mast_manifest: dict, main_manifest: dict) -> list[str]:
    """Etiketleme turunun BU deneye ve BU karar kuralına ait olduğunu doğrular.

    Kendi içinde tutarlı bir MAST turu, DOĞRU deneyin turu demek değildir: başka
    bir modelin etiketleri aynı görev kimlikleriyle sorunsuz görünür ve hata
    dağılımı yanlış modele atfedilirdi. `source_manifest_fingerprint` model/görev
    seti/kol kümesi/prompt değişimini tek alanda yakalar.
    """
    problems = []
    for alan, beklenen, ad in (
            ("mast_schema_version", MAST_SCHEMA_VERSION, "MAST şeması"),
            ("mast_decision_rule_version", MAST_DECISION_RULE_VERSION, "karar kuralı"),
            ("mast_panel_hash_version", MAST_PANEL_HASH_VERSION, "panel hash sürümü"),
            ("adjudicator_model", MODEL_ADJUDICATOR, "adjudicator")):
        if mast_manifest.get(alan) != beklenen:
            problems.append(
                f"{ad} uyuşmuyor: {mast_manifest.get(alan)!r} != {beklenen!r} — "
                "farklı kuralla üretilmiş etiketler aynı tabloda toplanamaz.")
    if mast_manifest.get("experiment") != main_manifest.get("name"):
        problems.append(
            f"deney adı uyuşmuyor: MAST {mast_manifest.get('experiment')!r}, "
            f"ana koşu {main_manifest.get('name')!r}")
    if mast_manifest.get("source_model") != main_manifest.get("model"):
        problems.append(
            f"kaynak model uyuşmuyor: MAST {mast_manifest.get('source_model')!r}, "
            f"ana koşu {main_manifest.get('model')!r} — modeller HAVUZLANMAZ.")
    if mast_manifest.get("source_task_set") != main_manifest.get("task_set"):
        problems.append(
            f"görev seti uyuşmuyor: MAST {mast_manifest.get('source_task_set')!r}, "
            f"ana koşu {main_manifest.get('task_set')!r}")
    beklenen_fp = source_manifest_fingerprint(main_manifest)
    if mast_manifest.get("source_manifest_fingerprint") != beklenen_fp:
        problems.append(
            "source_manifest_fingerprint uyuşmuyor: etiketleme turu bu deneyin "
            "güncel yapılandırmasından üretilmemiş.")
    return problems


def verify_mast_artifacts(records: list[dict], judges: list[dict], panel: list[dict],
                          adjudications: list[dict], *, experiment: str):
    """Panel + adjudication tazeliğini ORTAK doğrulayıcıyla yeniden hesaplar.

    Bu modül kendi panel/adjudication doğrulamasını YAZMAZ:
    `eval.mast_labels.verify_panel()` ve `verify_adjudications()` ücretli
    adjudication ön geçişinin de kullandığı fonksiyonlardır. İkinci bir
    implementasyon, zamanla ayrışıp "analiz paneli güncel sandı, adjudicator eski
    kanıta baktı" durumunu tam olarak geri getirirdi.

    Yeniden hesaplanan (dosyadan OKUNMAYAN) şeyler: kanıt hash'i (görev
    dosyasından), güncel judge kümesi (tam provenance kimliğiyle),
    `full_panel_input_sha256`, `decision_input_sha256` ve bütün dış karar
    alanları. Saklanmış `evidence_consistent`/`prompt_consistent` bayrakları
    kanıt değil, doğrulanan birer iddiadır.
    """
    try:
        verified = verify_panel(records, judges, panel, experiment=experiment,
                                prompt_hash=prompt_contract_hash())
        current_adj, adj_counts = verify_adjudications(
            verified, adjudications, experiment=experiment)
    except MastPipelineError as e:
        raise MastDistributionError(f"MAST artefakt doğrulaması başarısız: {e}") from e
    return verified, current_adj, adj_counts


def decision_label(panel_row: dict, adjudication: dict | None
                   ) -> tuple[str | None, str]:
    """(karar etiketi, karar kaynağı) — leave-self-out.

    Etiket `None` ise kayıt KARARSIZDIR: hiçbir modun payına girmez. Kararsızı
    "none" moduna saymak, ölçülmemiş bir kaydı "hata modu yok" diye raporlamak
    olurdu.
    """
    seviye = panel_row["external_agreement_level"]
    if seviye == "consensus":
        return panel_row["external_consensus_label"], DECISION_EXTERNAL_CONSENSUS
    if seviye == "split":
        if adjudication is None:
            return None, DECISION_UNDECIDED_SPLIT
        etiket = (INSUFFICIENT_SENTINEL
                  if adjudication.get("adjudicated_insufficient_context")
                  else adjudication.get("adjudicated_primary_mode"))
        return etiket, DECISION_ADJUDICATED
    return None, DECISION_UNDECIDED_INCOMPLETE


def _self_vote(panel_row: dict) -> str | None:
    """Kaydı ÜRETEN modelin kendi oyu — yalnız TANISAL (karara girmez)."""
    modeller = panel_row["judge_models"]
    self_model = panel_row["self_judge_model"]
    if self_model not in modeller:
        return None
    return panel_row["primary_modes"][modeller.index(self_model)]


def _external_votes(panel_row: dict) -> list[str]:
    modeller = panel_row["judge_models"]
    return [panel_row["primary_modes"][modeller.index(m)]
            for m in panel_row["external_judges"] if m in modeller]


def build_distribution(main_manifest: dict, records: list[dict], mast_manifest: dict,
                       judges: list[dict], panel: list[dict],
                       adjudications: list[dict], *,
                       allow_missing: bool = False) -> dict:
    """Bütün dağılım çıktısını tek sözlükte üretir (dosya yazımı ayrı adımda)."""
    if not records:
        raise MastDistributionError("ana koşu sonuç kaydı yok.")
    model = single_model(records)
    provenance = check_provenance(records, main_manifest)
    if provenance:
        raise MastDistributionError("MAST dağılımı durduruldu (provenance):\n  - "
                                    + "\n  - ".join(provenance))
    report = integrity_report(records, main_manifest["task_ids"],
                              main_manifest["arm_order"], main_manifest["repeats"],
                              main_manifest["model"])
    for ad, deger in (("yinelenen kayıt", report["duplicates"]),
                      ("beklenmeyen anahtar", report["unexpected"]),
                      ("şema ihlali", report["invalid_records"])):
        if deger:
            raise MastDistributionError(
                f"MAST dağılımı durduruldu (ana koşu bütünlüğü): {ad}. "
                "Yinelenen bir sonuç kaydı aynı başarısızlığı iki kez saydırır.")
    if report["missing"] and not allow_missing:
        raise MastDistributionError(
            f"{len(report['missing'])} eksik arm-run — hata dağılımı eksik veri "
            "üzerinde raporlanmaz (--allow-missing ile preliminary damgalanır).")

    sorunlar = check_mast_provenance(mast_manifest, main_manifest)
    if sorunlar:
        raise MastDistributionError("MAST dağılımı durduruldu (MAST provenance):\n  - "
                                    + "\n  - ".join(sorunlar))

    # ÖN ETİKETLEME KAPISI (§9.1): `--allow-missing` ile açılmış, kaynak koşu
    # tamamlanmadan üretilmiş bir tur eksik bir hata evrenini temsil eder. Böyle
    # bir turdan üretilen dağılım FORMAL rapora giremez; varsayılan fail-fast'tir.
    tur_on_etiketleme = bool(mast_manifest.get("preliminary")) or (
        mast_manifest.get("source_results_complete") is False)
    if tur_on_etiketleme and not allow_missing:
        raise MastDistributionError(
            f"MAST turu ÖN ETİKETLEME (preliminary={mast_manifest.get('preliminary')!r}, "
            f"source_results_complete={mast_manifest.get('source_results_complete')!r}, "
            f"missing_runs={mast_manifest.get('missing_runs')!r}) — eksik bir kaynak "
            "koşudan üretilen panel bütün hata evrenini temsil etmez. Formal rapor "
            "bu turdan üretilemez; yalnız --allow-missing ile preliminary damgalı "
            "çıktı alınabilir.")

    arms = list(main_manifest["arm_order"])
    hedef = labelable_records(records)
    verified, adj_by_run, adj_counts = verify_mast_artifacts(
        hedef, judges, panel, adjudications, experiment=main_manifest["name"])
    # Karar KAYITTAN OKUNMAZ, güncel etiketlerden yeniden hesaplanan verdict'ten
    # gelir; panel satırı yalnız provenance zarfı olarak kullanılır.
    by_run = {run_id: satir.verdict | {"arm": satir.record["arm"]}
              for run_id, satir in verified.items()}

    tamamlanan = [r for r in records if not is_run_error(r)]
    completed_by_arm = {arm: sum(r["arm"] == arm for r in tamamlanan) for arm in arms}
    labeled_by_arm = {arm: sum(r["arm"] == arm for r in hedef) for arm in arms}

    counts = {arm: {} for arm in arms}
    decided_by_arm = {arm: 0 for arm in arms}
    undecided_by_arm = {arm: 0 for arm in arms}
    decision_sources = {DECISION_EXTERNAL_CONSENSUS: 0, DECISION_ADJUDICATED: 0,
                        DECISION_UNDECIDED_SPLIT: 0, DECISION_UNDECIDED_INCOMPLETE: 0}
    external_levels = {"consensus": 0, "split": 0, "incomplete": 0}
    decision_insufficient = 0
    external_any_insufficient = 0
    self_insufficient = 0
    self_agree = self_disagree = self_undefined = 0
    undecided_runs = []

    for run_id, row in by_run.items():
        arm = row["arm"]
        external_levels[row["external_agreement_level"]] = (
            external_levels.get(row["external_agreement_level"], 0) + 1)
        etiket, kaynak = decision_label(row, adj_by_run.get(run_id))
        decision_sources[kaynak] += 1
        if etiket is None:
            undecided_by_arm[arm] += 1
            undecided_runs.append(run_id)
        else:
            decided_by_arm[arm] += 1
            if etiket == INSUFFICIENT_SENTINEL:
                # Yetersiz bağlam bir MAST modu DEĞİL; mod tablosuna girmez.
                decision_insufficient += 1
            else:
                counts[arm][etiket] = counts[arm].get(etiket, 0) + 1
        if INSUFFICIENT_SENTINEL in _external_votes(row):
            external_any_insufficient += 1
        if _self_vote(row) == INSUFFICIENT_SENTINEL:
            self_insufficient += 1
        uyum = row["self_matches_external"]
        if uyum is None:
            self_undefined += 1
        elif uyum:
            self_agree += 1
        else:
            self_disagree += 1

    labeled_total = len(hedef)
    blockers = []
    if decision_sources[DECISION_UNDECIDED_SPLIT]:
        blockers.append(
            f"{decision_sources[DECISION_UNDECIDED_SPLIT]} dış-split kaydın GÜNCEL "
            f"Grok kararı yok (stale: {adj_counts['stale']}) — önce "
            "`--stage adjudicate` çalıştır.")
    if decision_sources[DECISION_UNDECIDED_INCOMPLETE]:
        blockers.append(
            f"{decision_sources[DECISION_UNDECIDED_INCOMPLETE]} panelde eksik judge "
            "var — önce `--stage judge` ile tamamla.")
    if blockers and not allow_missing:
        raise MastDistributionError(
            "MAST dağılımı durduruldu:\n  - " + "\n  - ".join(blockers)
            + "\n  - kararsız kayıtlar paydayı küçültüp kompozisyon oranını yukarı "
              "çeker; --allow-missing ile preliminary damgalanarak geçilebilir.")

    def _oran(pay: int, payda: int) -> float | None:
        return round(pay / payda, 6) if payda else None

    modes = sorted({m for arm in arms for m in counts[arm]})
    return {
        "mast_distribution_schema_version": MAST_DISTRIBUTION_SCHEMA_VERSION,
        "experiment": main_manifest.get("name"),
        "model": model,
        "task_set": main_manifest.get("task_set"),
        "mast_schema_version": mast_manifest["mast_schema_version"],
        "decision_rule_version": mast_manifest["mast_decision_rule_version"],
        "panel_hash_version": mast_manifest["mast_panel_hash_version"],
        "judges": mast_manifest.get("judges"),
        "adjudicator_model": mast_manifest.get("adjudicator_model"),
        "preliminary": bool(blockers) or bool(report["missing"]) or tur_on_etiketleme,
        "blockers": blockers,
        "mast_round_preliminary": tur_on_etiketleme,
        # Bir sayının hangi kod/veri/prompt sürümünden çıktığı özetin KENDİSİNDEN
        # okunabilmeli: dosya artefakt paketine kopyalandığında yanındaki manifest
        # kaybolabilir, tablo ise rapora girer.
        "source_provenance": {
            "main_experiment": main_manifest.get("name"),
            "main_git_commit": main_manifest.get("git_commit"),
            "main_result_schema_version": main_manifest.get("result_schema_version"),
            "main_task_ids_sha256": _canonical_hash(main_manifest.get("task_ids")),
            "main_task_file_hashes_sha256": _canonical_hash(
                main_manifest.get("task_file_hashes")),
            "mast_git_commit": mast_manifest.get("git_commit"),
            "mast_source_manifest_fingerprint": mast_manifest.get(
                "source_manifest_fingerprint"),
            "mast_prompt_hash_manifest": mast_manifest.get("mast_prompt_hash"),
            # Manifestteki prompt hash'i bir İDDİA; koddan yeniden hesaplanan
            # değer, etiketlerin bu prompt sürümüyle üretildiğinin kanıtıdır.
            "mast_prompt_hash_recomputed": prompt_contract_hash(),
            "mast_judge_temperature": mast_manifest.get("judge_temperature"),
            "mast_missing_runs": mast_manifest.get("missing_runs"),
        },
        "arms": arms,
        "denominators": {
            "completed_arm_runs_total": len(tamamlanan),
            "labeled_failures_total": labeled_total,
            "decided_total": sum(decided_by_arm.values()),
            "undecided_total": sum(undecided_by_arm.values()),
            "by_arm": {
                arm: {"completed_arm_runs": completed_by_arm[arm],
                      "labeled_failures": labeled_by_arm[arm],
                      "decided": decided_by_arm[arm],
                      "undecided": undecided_by_arm[arm]}
                for arm in arms},
        },
        "decision_source_counts": decision_sources,
        "external_agreement_level_counts": external_levels,
        "adjudication": adj_counts,
        "insufficient_context": {
            "decision_insufficient_count": decision_insufficient,
            "decision_insufficient_denominator": sum(decided_by_arm.values()),
            "any_external_judge_insufficient_count": external_any_insufficient,
            "self_judge_insufficient_count": self_insufficient,
            "judge_level_denominator": labeled_total,
        },
        "self_vs_external_consensus": {
            "agree": self_agree, "disagree": self_disagree,
            "n_comparable": self_agree + self_disagree,
            "undefined": self_undefined,
            "agreement_rate": _oran(self_agree, self_agree + self_disagree),
            "note": "TANISALDIR: self etiketi karara oy vermez (leave-self-out).",
        },
        "primary_modes": modes,
        "arm_by_primary_mode": {arm: dict(sorted(counts[arm].items())) for arm in arms},
        "error_composition": {
            arm: {m: _oran(counts[arm].get(m, 0), decided_by_arm[arm]) for m in modes}
            for arm in arms},
        "arm_incidence": {
            arm: {m: _oran(counts[arm].get(m, 0), completed_by_arm[arm]) for m in modes}
            for arm in arms},
        "mode_totals": {m: sum(counts[arm].get(m, 0) for arm in arms) for m in modes},
        "undecided_run_ids": sorted(undecided_runs),
        "notes": NOTES,
    }


NOTES = [
    "İKİ PAYDA ayrıdır: error_composition paydası o koldaki KARARA BAĞLANMIŞ "
    "hata sayısı, arm_incidence paydası o koldaki bütün tamamlanmış arm-run "
    "sayısıdır. Kollar arası karşılaştırma için insidans kullanılmalıdır.",
    "Karar etiketi leave-self-out'tur: iki dış judge'ın konsensüsü ya da "
    "dış split'te Grok adjudication'ı. Üçlü majority_label ve self etiketi "
    "yalnız tanısaldır ve hiçbir oranın payına girmez.",
    "insufficient_context bir MAST modu değildir; mod tablolarına girmez, "
    "kendi sayısı ve paydasıyla raporlanır.",
    "Kararsız kayıtlar (eksik panel / güncel adjudication yok) hiçbir modun "
    "payına girmez ve ayrıca sayılır.",
    "run_error kayıtları ne etiketlenir ne paydaya girer (§7): altyapı arızası "
    "ajan başarısızlığı değildir.",
    "Modeller havuzlanmaz; her model için ayrı deney dizini ve ayrı tablo.",
    "İnsan turundan (eval/mast_human.py, 30 kayıtlık seçilimli örneklem) gelen "
    "uyum oranları bütün hata evrenine GENELLENEMEZ; buradaki sayılar ise tüm "
    "etiketlenmiş başarısızlıkların evrenidir. İki tablo birleştirilmemelidir.",
]


def write_distribution_outputs(result: dict, out_dir: Path) -> dict[str, Path]:
    """Makine-okunur çıktılar. Zaman damgası YOK (aynı veri = aynı baytlar)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {"mast_distribution": out_dir / "mast_distribution.json",
             "mast_arm_mode": out_dir / "mast_arm_mode.csv"}
    paths["mast_distribution"].write_text(
        json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8")
    rows = [
        {"model": result["model"], "arm": arm, "primary_mode": mode,
         "count": result["arm_by_primary_mode"][arm].get(mode, 0),
         "decided_failures_in_arm": result["denominators"]["by_arm"][arm]["decided"],
         "completed_arm_runs_in_arm":
             result["denominators"]["by_arm"][arm]["completed_arm_runs"],
         "error_composition": result["error_composition"][arm][mode],
         "arm_incidence": result["arm_incidence"][arm][mode]}
        for arm in result["arms"] for mode in result["primary_modes"]
    ]
    columns = ["model", "arm", "primary_mode", "count", "decided_failures_in_arm",
               "completed_arm_runs_in_arm", "error_composition", "arm_incidence"]
    with paths["mast_arm_mode"].open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MAST hata dağılımı (kompozisyon + insidans, iki ayrı payda)")
    parser.add_argument("--exp", required=True, type=Path,
                        help="ana deney dizini (logs/exp_<name>)")
    # AÇIK argüman: `<exp>/mast` varsayılmaz — `--limit` ile üretilen
    # mast_smoke_* dizinleri de aynı biçimdedir ve sessizce seçilmemelidir.
    parser.add_argument("--mast", required=True, type=Path,
                        help="MAST tur dizini (ör. logs/exp_<name>/mast)")
    parser.add_argument("--out", type=Path, default=None,
                        help="çıktı dizini (varsayılan: <exp>/analysis)")
    parser.add_argument("--allow-missing", action="store_true",
                        help="eksik arm-run / kararsız kayıt / ÖN ETİKETLEME turunu "
                             "mazur görür; çıktı preliminary damgalanır "
                             "(doğrulayıcı rapora giremez)")
    args = parser.parse_args()

    try:
        main_manifest, records, _ = load_experiment(args.exp)
        mast_manifest, judges, panel, adjudications = load_mast(args.mast)
        result = build_distribution(main_manifest, records, mast_manifest, judges,
                                    panel, adjudications,
                                    allow_missing=args.allow_missing)
    except (AnalysisError, MastDistributionError) as e:
        sys.exit(str(e))

    paths = write_distribution_outputs(result, args.out or args.exp / "analysis")
    d = result["denominators"]
    print(f"deney: {result['experiment']} | model: {result['model']} | "
          f"karar kuralı: {result['decision_rule_version']}")
    if result["preliminary"]:
        print("UYARI: çıktı PRELIMINARY damgalı; doğrulayıcı rapora girmez.")
        for engel in result["blockers"]:
            print(f"  - {engel}")
    print(f"\netiketlenmiş başarısızlık {d['labeled_failures_total']} "
          f"({d['decided_total']} karara bağlandı, {d['undecided_total']} kararsız) "
          f"| tamamlanmış arm-run {d['completed_arm_runs_total']}")
    print(f"karar kaynağı: {result['decision_source_counts']}")
    print(f"dış anlaşma: {result['external_agreement_level_counts']} | "
          f"adjudication: {result['adjudication']}")
    print("\nkol × primary_mode (sayı | kompozisyon | insidans):")
    for arm in result["arms"]:
        a = d["by_arm"][arm]
        print(f"  {arm} — {a['decided']}/{a['labeled_failures']} karara bağlı hata, "
              f"{a['completed_arm_runs']} arm-run")
        for mode, n in result["arm_by_primary_mode"][arm].items():
            print(f"      {mode:6s} {n:4d} | {result['error_composition'][arm][mode]} "
                  f"| {result['arm_incidence'][arm][mode]}")
    ic = result["insufficient_context"]
    print(f"\ninsufficient_context — karar: {ic['decision_insufficient_count']}/"
          f"{ic['decision_insufficient_denominator']} | herhangi bir dış judge: "
          f"{ic['any_external_judge_insufficient_count']}/{ic['judge_level_denominator']} "
          f"| self: {ic['self_judge_insufficient_count']}/{ic['judge_level_denominator']}")
    sv = result["self_vs_external_consensus"]
    print(f"self <-> dis konsensus (TANISAL): {sv['agree']}/{sv['n_comparable']} "
          f"(oran {sv['agreement_rate']}, tanımsız {sv['undefined']})")
    print("\nÇıktılar:")
    for name, path in paths.items():
        print(f"  {name:18s} -> {path}")


if __name__ == "__main__":
    main()
