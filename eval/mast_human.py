"""Kör insan MAST etiketleme turu + AI destekli insan adjudication (§9.3–§9.6).

MAST 3.2 / insan şeması 2.0 karar semantiği leave-self-out'tur: iki dış judge
karar girdisidir; üretici modelin self-judge etiketi yalnız nihai insan kararı
kilitlendikten sonra ayrı diagnostic pakette açılabilir.

Akış (her adım bir CLI komutu, her komut ÖNCE doğrular SONRA yazar):

    mast/ai_panel.jsonl + ai_judges.jsonl
      → deterministik örneklem            → human_sample_manifest.json
      → iki KÖR statik HTML paketi        → packages/blind_annotator_*.html
      → (tarayıcıda etiketleme, JSON export)
      → doğrulama + kilit                 → blind_annotator_*.locked.json
      → AI destekli adjudication paketi   → packages/human_adjudication.html
      → doğrulama + kilit                 → human_adjudication.locked.json
      → betimleyici uyum özeti            → agreement_summary.json

Dört tasarım kararı:

1. **Kör payload ALAN İZİN LİSTESİYLE kurulur.** Ham sonuç/panel sözlüğü HTML'e
   hiç girmez; yeni bir provenance alanı eklendiğinde kör pakete sızamaz. CSS
   ile gizlemek körlük sayılmaz — veri orada YOKTUR.

2. **Payload base64 gömülür.** Model üretimi kod/traceback içindeki bir
   `</script>` dizisi, JSON doğrudan script bloğuna yazılsaydı bloğu kapatıp
   kalanını HTML olarak çalıştırabilirdi. Şablona giren TEK değişken base64
   metnidir.

3. **Tarayıcı deposu kanıt DEĞİLDİR.** `file://` üzerinde `localStorage`
   davranışı tarayıcılar arasında standart değil; tek otoritatif kayıt JSON
   export + buradaki doğrulama/kilitleme hattıdır.

4. **Kilit tek yönlüdür.** Aynı içerik tekrar kilitlenirse idempotenttir; FARKLI
   içerikle üzerine yazma reddedilir ve `--force` YOKTUR. Etiket değişmesi
   gerekiyorsa bu yeni bir turdur — sessizce güncellenen bir "kilit" kilit
   değildir.

Kullanım:
    uv run python -m eval.mast_human prepare-blind --exp logs/exp_gemini_main
    uv run python -m eval.mast_human lock-blind --exp logs/exp_gemini_main \
        --annotator annotator_a --input export_a.json
    uv run python -m eval.mast_human prepare-adjudication --exp logs/exp_gemini_main
    uv run python -m eval.mast_human lock-adjudication --exp logs/exp_gemini_main \
        --input final.json
    uv run python -m eval.mast_human summarize --exp logs/exp_gemini_main
"""

import argparse
import base64
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

from config import (
    MAST_CONFIDENCE_ANCHORS,
    MAST_CONFIDENCE_LEVELS,
    MAST_HUMAN_ANNOTATORS,
    MAST_HUMAN_APP_VERSION,
    MAST_HUMAN_LOCK_SCHEMA_VERSION,
    MAST_HUMAN_SAMPLE_SEED,
    MAST_HUMAN_SAMPLE_TARGET,
    MAST_HUMAN_SCHEMA_VERSION,
    MAST_HUMAN_STRATA_FIELDS,
    MAST_LOG_NAMESPACE,
    MAST_DECISION_RULE_VERSION,
    MAST_PANEL_HASH_VERSION,
    MAST_SCHEMA_VERSION,
    MODEL_ADJUDICATOR,
    MODEL_JUDGES,
)
from eval import mast_labels
from eval.mast_labels import (
    EVIDENCE_FIELDS,
    external_annotator_order,
    build_evidence,
    current_external_judges_for_record,
    current_judges_for_record,
    interaction_type,
    labelable_records,
    load_experiment,
    load_jsonl,
    prompt_contract_hash,
    source_manifest_fingerprint,
)
from eval.mast_schema import (
    INSUFFICIENT_SENTINEL,
    MAST_MODES,
    MastAdjudication,
    MastLabel,
    MastPanelVerdict,
    MastPipelineError,
    evidence_digest,
    interaction_problems,
    full_panel_input_digest,
    decision_input_digest,
    validate_frozen_panel,
)

TEMPLATE_DIR = Path(__file__).parent / "templates"
PAYLOAD_TOKEN = "__PAYLOAD_BASE64__"
ANNOTATOR_ID_PATTERN = re.compile(r"^[a-z0-9_-]+$")

PHASE_BLIND = "blind"
PHASE_ADJUDICATION = "human_adjudication"
PHASE_DIAGNOSTIC = "post_lock_diagnostic"

SELECTION_MANDATORY_DISAGREEMENT = "mandatory_disagreement"
SELECTION_MANDATORY_INSUFFICIENT = "mandatory_insufficient"
SELECTION_MANDATORY_SELF_MISMATCH = "mandatory_self_mismatch"
SELECTION_STRATIFIED = "stratified"

# Kör kanıt paketinin alanları: mast_labels'ın izin listesinden `interaction_type`
# çıkarılır (üst düzeyde taşınır, kanıt gövdesinde tekrar edilmez).
BLIND_EVIDENCE_FIELDS = tuple(f for f in EVIDENCE_FIELDS if f != "interaction_type")

# Kör payload'da ASLA bulunmaması gereken anahtarlar (testin denetlediği liste).
FORBIDDEN_BLIND_KEYS = (
    "source_run_id", "run_id", "model", "source_model", "provider", "arm",
    "judge_model", "judge_models", "primary_mode", "secondary_modes",
    "majority_label", "agreement_level", "adjudicated_primary_mode", "rationale",
    "confidence", "panel_input_sha256", "full_panel_input_sha256",
    "decision_input_sha256", "external_agreement_level",
    "external_consensus_label", "self_matches_external", "self_judge_model",
)

BLIND_EXPORT_FIELDS = ("human_schema_version", "app_version", "dataset_fingerprint",
                       "package_id", "annotator_id", "phase", "labels")
BLIND_LABEL_FIELDS = ("record_id", "primary_mode", "secondary_modes", "confidence",
                      "rationale", "insufficient_context")
ADJ_EXPORT_FIELDS = ("human_schema_version", "app_version", "dataset_fingerprint",
                     "package_id", "phase", "blind_lock_sha256", "labels")

# İnsan tüketicisi yalnız Grok resume tuple'ına güvenmez; güncel kararın bütün
# provenance zarfını da yeniden doğrular. Hash kimliğin parçasıdır, yerine geçmez.
ADJUDICATION_MATCH_FIELDS = (
    "source_run_id", "experiment", "source_model", "task_set", "task_id", "arm",
    "repeat", "mast_schema_version", "mast_prompt_hash", "adjudicator_model",
    "evidence_sha256", "decision_input_sha256", "decision_rule_version",
    "self_judge_model", "external_judges", "external_agreement_level",
    "reviewed_judges", "interaction_type",
)

# Adjudicator'ın KARARININ kendisi. Snapshot'a girmezse, adjudicator aynı panel
# üzerinde tamamen farklı bir karar verdiğinde paket kimliği değişmez ve eski
# nihai insan kararı yeni AI kararına karşı kabul edilirdi (canlı üretildi).
ADJUDICATED_LABEL_FIELDS = ("adjudicated_primary_mode", "adjudicated_secondary_modes",
                            "adjudicated_confidence", "adjudicated_rationale",
                            "adjudicated_insufficient_context")

MANIFEST_REQUIRED_FIELDS = ("human_schema_version", "app_version", "mast_schema_version",
                            "experiment", "sample_seed", "sample_target",
                            "strata_fields", "annotators", "dataset_fingerprint",
                            "records")


class HumanRoundError(MastPipelineError):
    """İnsan turu güvenle yürütülemez — bozuk paket üretmektense durulur."""


# --- Kanonik hash yardımcıları ----------------------------------------------

def canonical(payload) -> str:
    """Hash'lenecek her yapı için TEK serileştirme.

    Ayrı yerlerde ayrı `json.dumps` çağrısı kullanmak, boşluk ya da anahtar
    sırası farkı yüzünden aynı içeriğin farklı hash vermesine yol açardı —
    kilit doğrulaması sessizce kırılırdı.
    """
    return json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def sha256_of(payload) -> str:
    return hashlib.sha256(canonical(payload).encode("utf-8")).hexdigest()


def _rank(*parts: str) -> str:
    """Deterministik sıralama anahtarı.

    Python'un `hash()`'i, set/dict sırası ve `PYTHONHASHSEED` süreçler arasında
    değişir; örneklem bunlara bağlanırsa "aynı seed aynı örneklem" iddiası
    çöker. Bu yüzden sıralama SHA-256 üzerinden yapılır.
    """
    return hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _task_for(record: dict) -> dict:
    """Görev yüklemeyi mast_labels üzerinden ÇAĞRI ANINDA çözer.

    Doğrudan `from ... import _task_for` yapılsaydı iki modül iki ayrı isme
    bağlanır ve testte biri monkeypatch'lenirken diğeri eski yolu kullanırdı —
    judge'ın gördüğü görevle insanın gördüğü görev sessizce ayrışabilirdi.
    """
    return mast_labels._task_for(record)


# --- Girdi bütünlüğü kapısı --------------------------------------------------

def _mast_dir(exp_dir: Path) -> Path:
    return exp_dir / MAST_LOG_NAMESPACE


def source_fingerprint(mast_manifest: dict) -> str:
    """Kaynak MAST turunun kimliği — `record_id` bunun üzerine kurulur.

    `dataset_fingerprint` KULLANILAMAZ (o, record_id'leri içeren örneklemden
    türer; döngüsel olurdu). Kaynak parmak izi + prompt hash'i, "hangi deneyin
    hangi etiketleme turu" sorusunu tek başına cevaplar.
    """
    return _rank(mast_manifest["source_manifest_fingerprint"],
                 mast_manifest["mast_prompt_hash"],
                 mast_manifest["mast_schema_version"])


def load_mast_round(exp_dir: Path) -> dict:
    """Kaynak deney + MAST turu; İLK hata görülmeden önce hepsi doğrulanır.

    `--allow-missing` bilinçli olarak YOKTUR: insan örneklemi eksik bir hata
    evreninden çekilirse "bütün anlaşmazlıklar incelendi" iddiası savunulamaz.
    """
    mast_dir = _mast_dir(exp_dir)
    manifest_path = mast_dir / "manifest.json"
    if not manifest_path.exists():
        raise HumanRoundError(
            f"MAST turu yok: {manifest_path}\n"
            "önce: uv run python -m eval.mast_labels --exp <exp> --stage all")
    mast_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    sorunlar = []
    if mast_manifest.get("preliminary") is not False:
        sorunlar.append("MAST turu ÖN ETİKETLEME (preliminary=true) — eksik bir "
                        "hata evreninden insan örneklemi çekilemez")
    if mast_manifest.get("source_results_complete") is not True:
        sorunlar.append("kaynak koşu tamamlanmamış (source_results_complete != true)")
    for ad in ("ai_judges.jsonl", "ai_panel.jsonl"):
        if not (mast_dir / ad).exists():
            sorunlar.append(f"eksik MAST çıktısı: {mast_dir / ad}")
    if sorunlar:
        raise HumanRoundError("insan turu durduruldu:\n  - " + "\n  - ".join(sorunlar))

    # load_experiment allow_missing=False: eksik koşu burada da durdurur.
    manifest, records, _report = load_experiment(exp_dir)
    try:
        validate_frozen_panel(mast_manifest.get("judges", ()),
                              mast_manifest.get("adjudicator_model"))
    except MastPipelineError as e:
        raise HumanRoundError(f"MAST manifesti dondurulmuş kadroyla uyuşmuyor: {e}") from e
    beklenen_mast = {
        "experiment": manifest["name"],
        "mast_schema_version": MAST_SCHEMA_VERSION,
        "mast_decision_rule_version": MAST_DECISION_RULE_VERSION,
        "mast_panel_hash_version": MAST_PANEL_HASH_VERSION,
        "mast_prompt_hash": prompt_contract_hash(),
        "source_manifest_fingerprint": source_manifest_fingerprint(manifest),
        "source_model": manifest["model"],
        "source_task_set": manifest["task_set"],
        "llm_call_schema_version": manifest["llm_call_schema_version"],
    }
    farklar = {k: (mast_manifest.get(k), v) for k, v in beklenen_mast.items()
               if mast_manifest.get(k) != v}
    if farklar:
        raise HumanRoundError(
            f"MAST manifesti kaynak deney/güncel sözleşmeyle uyuşmuyor: {farklar}")
    hedef = labelable_records(records)
    if not hedef:
        raise HumanRoundError("etiketlenecek başarısızlık yok.")

    judges = load_jsonl(mast_dir / "ai_judges.jsonl")
    panel = load_jsonl(mast_dir / "ai_panel.jsonl")
    adjudications = load_jsonl(mast_dir / "ai_adjudication.jsonl")
    return {"exp_dir": exp_dir, "mast_dir": mast_dir, "manifest": manifest,
            "mast_manifest": mast_manifest, "records": hedef, "judges": judges,
            "panel": panel, "adjudications": adjudications,
            "experiment": manifest["name"]}


def _panel_index(panel: list[dict]) -> dict[str, dict]:
    by_run: dict[str, dict] = {}
    yinelenen = []
    for row in panel:
        rid = row.get("source_run_id")
        if rid in by_run:
            yinelenen.append(rid)
        by_run[rid] = row
    if yinelenen:
        raise HumanRoundError(
            f"ai_panel.jsonl yinelenen kayıt içeriyor ({len(yinelenen)}): "
            f"{sorted(set(yinelenen))[:3]} — panel türetilmiş bir görünümdür, "
            "judge aşamasını tekrar çalıştırıp yeniden üret.")
    return by_run


def collect_candidates(round_: dict, *, expected_judges=MODEL_JUDGES) -> list[dict]:
    """Örneklem adayları — BÜTÜN girdi kümesi doğrulanmadan hiçbiri döndürülmez.

    Kısmi doğrulama (ilk hatada durup o ana kadarkileri kullanmak) yarım bir
    paket üretirdi; paket bir kez insana verildiğinde geri alınamaz.
    """
    panel_by_run = _panel_index(round_["panel"])
    judges_by_run: dict[str, list[dict]] = {}
    for r in round_["judges"]:
        judges_by_run.setdefault(r.get("source_run_id"), []).append(r)

    experiment = round_["experiment"]
    prompt_hash = prompt_contract_hash()
    kaynak = source_fingerprint(round_["mast_manifest"])

    beklenen_runlar = {r["run_id"] for r in round_["records"]}
    yabanci = sorted(set(panel_by_run) - beklenen_runlar)
    if yabanci:
        raise HumanRoundError(
            f"ai_panel.jsonl'de {len(yabanci)} yabancı kayıt (etiketlenecek "
            f"başarısızlık kümesinde yok): {yabanci[:3]}")

    adaylar, sorunlar = [], []
    for record in round_["records"]:
        run_id = record["run_id"]
        satir = panel_by_run.get(run_id)
        if satir is None:
            sorunlar.append(f"{run_id}: panel kaydı yok")
            continue
        try:
            panel_model = MastPanelVerdict.model_validate(
                {k: satir[k] for k in MastPanelVerdict.model_fields})
        except Exception as e:
            sorunlar.append(f"{run_id}: panel sözleşmesi geçersiz ({e})")
            continue
        # KAPI: "tam üçlü panel" DEĞİL, "dış karar kurulabiliyor mu" (2026-08-03).
        # Self-judge yalnız TANISALDIR (§9.1): kör insan etiketine, dış karara ve
        # nihai insan kararına girmez. Eksik bir self yüzünden kaydı örneklem
        # EVRENİNDEN düşürmek, evreni tanısal bir AI çıktısının başarısına
        # koşullandırırdı. Dış judge eksikliği ise hâlâ fail-closed: onsuz karar
        # girdisi (ve insanlara gösterilecek iki dış etiket) kurulamaz.
        if panel_model.external_agreement_level == "incomplete":
            sorunlar.append(
                f"{run_id}: DIŞ judge eksik ({satir.get('missing_judges')}) — "
                "dış karar girdisi kurulamaz")
            continue
        if not (satir.get("evidence_consistent") and satir.get("prompt_consistent")):
            sorunlar.append(f"{run_id}: panel güncel olmayan kanıt/prompt kullanıyor")
            continue

        evidence = build_evidence(record, _task_for(record))
        digest = evidence_digest(evidence)
        try:
            guncel, _ = current_judges_for_record(
                judges_by_run.get(run_id, []), record, experiment=experiment,
                evidence_sha256=digest, prompt_hash=prompt_hash,
                expected_judges=expected_judges)
            # require_self=False: self yalnız tanısal (yukarıdaki kapı notu).
            self_model, dis_kayitlar, _ = current_external_judges_for_record(
                judges_by_run.get(run_id, []), record, experiment=experiment,
                evidence_sha256=digest, prompt_hash=prompt_hash,
                expected_judges=expected_judges, require_self=False)
        except MastPipelineError as e:
            sorunlar.append(f"{run_id}: {e}")
            continue
        by_model = {r["judge_model"]: r for r in guncel}
        self_record = by_model.get(self_model)
        self_available = self_record is not None
        beklenen_sayi = len(tuple(expected_judges)) - (0 if self_available else 1)
        if len(guncel) != beklenen_sayi:
            sorunlar.append(f"{run_id}: güncel judge kümesi tutarsız "
                            f"({len(guncel)}/{beklenen_sayi})")
            continue
        # Tanısal üçlü hash'i YALNIZ tam panelde tanımlıdır (MastPanelVerdict
        # bunu zaten zorlar); self eksikken panel de None taşır, dolayısıyla
        # aşağıdaki karşılaştırma iki tarafta da None ile eşleşir.
        full_hash = full_panel_input_digest(guncel, expected_judges) if self_available else None
        decision_hash = decision_input_digest(
            dis_kayitlar, expected_judges, source_model=record["model"])
        if satir.get("full_panel_input_sha256") != full_hash:
            sorunlar.append(
                f"{run_id}: panel yazıldıktan sonra tanısal üçlü değişti")
            continue
        if satir.get("decision_input_sha256") != decision_hash:
            sorunlar.append(
                f"{run_id}: panel yazıldıktan sonra dış karar girdisi değişti")
            continue
        if satir.get("evidence_sha256") != digest:
            sorunlar.append(f"{run_id}: panel kanıt hash'i güncel değil")
            continue

        # Panel provenance'ı güncel record + kanıt + prompt'tan yeniden kurulur.
        beklenen_provenance = {
            "source_run_id": run_id, "experiment": experiment,
            "source_model": record["model"], "task_set": record["task_set"],
            "task_id": record["task_id"], "arm": record["arm"],
            "repeat": record["repeat"], "mast_schema_version": MAST_SCHEMA_VERSION,
            "evidence_sha256": digest, "mast_prompt_hash": prompt_hash,
            "interaction_type": interaction_type(record["arm"]),
        }
        farklar = {k: (satir.get(k), v) for k, v in beklenen_provenance.items()
                   if satir.get(k) != v}
        if farklar:
            sorunlar.append(f"{run_id}: panel provenance uyuşmazlığı {farklar}")
            continue

        adaylar.append({
            "record_id": _rank(kaynak, run_id)[:32],
            "source_run_id": run_id,
            "task_id": record["task_id"],
            "task_set": record["task_set"],
            "model": record["model"],
            "arm": record["arm"],
            "repeat": record["repeat"],
            "error_class": record.get("error_class"),
            "interaction_type": interaction_type(record["arm"]),
            "evidence": {k: evidence[k] for k in BLIND_EVIDENCE_FIELDS},
            "evidence_sha256": digest,
            "full_panel_input_sha256": full_hash,
            "decision_input_sha256": decision_hash,
            "external_agreement_level": satir["external_agreement_level"],
            "external_consensus_label": satir.get("external_consensus_label"),
            "self_matches_external": satir.get("self_matches_external"),
            "self_judge_model": self_model,
            "external_judges": list(satir["external_judges"]),
            "adjudicator_required": bool(satir.get("adjudicator_required")),
            "judges": guncel,
            "external_judge_records": dis_kayitlar,
            # Nullable: self yalnız tanısaldır. `self_judge_available` AÇIKÇA
            # taşınır — tüketici "None mu yoksa alan mı yok" diye tahmin etmesin.
            "self_judge_record": self_record,
            "self_judge_available": self_available,
            "external_insufficient_judges": [
                r["judge_model"] for r in dis_kayitlar
                if r.get("insufficient_context")],
        })

    if sorunlar:
        raise HumanRoundError(
            f"insan turu durduruldu — {len(sorunlar)} kayıt kullanılamaz "
            "(kısmi paket YAZILMADI):\n  - " + "\n  - ".join(sorunlar[:8])
            + ("\n  - ..." if len(sorunlar) > 8 else "")
            + "\nönce: uv run python -m eval.mast_labels --exp <exp> --stage all")
    return sorted(adaylar, key=lambda c: c["record_id"])


# --- Deterministik örneklem (§9.3) -------------------------------------------

def select_human_sample(candidates: list[dict], *,
                        target: int = MAST_HUMAN_SAMPLE_TARGET,
                        seed: int = MAST_HUMAN_SAMPLE_SEED,
                        strata_fields=MAST_HUMAN_STRATA_FIELDS) -> list[dict]:
    """Zorunlu küme + tabakalı kalan kontenjan (saf fonksiyon, testte tek başına).

    `target` HEDEF sayıdır, ÜST SINIR DEĞİL: zorunlu küme onu aşarsa hiçbir
    kayıt atılmaz. "Bütün anlaşmazlıklar incelenir" iddiasıyla "en fazla 30"
    birlikte savunulamazdı; önceliği bilimsel iddiaya verdik (§9.3).
    """
    secim: dict[str, dict] = {}
    for c in candidates:
        nedenler, tur = [], None
        if c["external_agreement_level"] != "consensus":
            nedenler.append(f"external_{c['external_agreement_level']}")
            tur = SELECTION_MANDATORY_DISAGREEMENT
        if c["external_insufficient_judges"]:
            nedenler.append("external_judge_insufficient_context:"
                            + ",".join(c["external_insufficient_judges"]))
            tur = tur or SELECTION_MANDATORY_INSUFFICIENT
        if (c["external_agreement_level"] == "consensus"
                and c["self_matches_external"] is False):
            nedenler.append("self_disagrees_with_external_consensus")
            tur = tur or SELECTION_MANDATORY_SELF_MISMATCH
        if tur:
            secim[c["record_id"]] = {**c, "selection_type": tur,
                                     "selection_reasons": nedenler}

    kalan = target - len(secim)
    if kalan > 0:
        havuz = [c for c in candidates if c["record_id"] not in secim]
        tabakalar: dict[tuple, list[dict]] = {}
        for c in havuz:
            tabakalar.setdefault(tuple(c[f] for f in strata_fields), []).append(c)
        # Tabaka İÇİ sıra ve tabakaların BAŞLANGIÇ sırası ayrı ayrı seed'e bağlı:
        # ikisi de sabit değilse "aynı seed aynı örneklem" iddiası tutmaz.
        for anahtar, uyeler in tabakalar.items():
            uyeler.sort(key=lambda c: _rank(seed, anahtar, c["source_run_id"]))
        sira = sorted(tabakalar, key=lambda k: _rank(seed, k))
        tur_no = 0
        while kalan > 0 and any(tabakalar[k] for k in sira):
            for anahtar in sira:
                if not tabakalar[anahtar] or kalan <= 0:
                    continue
                c = tabakalar[anahtar].pop(0)
                secim[c["record_id"]] = {
                    **c, "selection_type": SELECTION_STRATIFIED,
                    "selection_reasons": [f"stratified_round_{tur_no}",
                                          "stratum:" + "|".join(str(x) for x in anahtar)]}
                kalan -= 1
            tur_no += 1

    return sorted(secim.values(), key=lambda c: c["record_id"])


def _manifest_row(c: dict, strata_fields) -> dict:
    return {
        "record_id": c["record_id"], "source_run_id": c["source_run_id"],
        "task_id": c["task_id"], "model": c["model"], "arm": c["arm"],
        "repeat": c["repeat"], "error_class": c["error_class"],
        "evidence_sha256": c["evidence_sha256"],
        "full_panel_input_sha256": c["full_panel_input_sha256"],
        "decision_input_sha256": c["decision_input_sha256"],
        "external_agreement_level": c["external_agreement_level"],
        "external_consensus_label": c["external_consensus_label"],
        "self_matches_external": c["self_matches_external"],
        # Örneklem manifesti "neden bu kayıt" sorusunu cevaplar; self etiketinin
        # bulunup bulunmadığı da o cevabın parçasıdır (self_matches_external=None
        # tek başına "eksik mi, hesaplanamaz mı" ayrımını vermez).
        "self_judge_available": c["self_judge_available"],
        "selection_type": c["selection_type"],
        "selection_reasons": c["selection_reasons"],
        "stratum": [c[f] for f in strata_fields],
    }


def dataset_fingerprint(rows: list[dict], *, seed: int, target: int, strata_fields) -> str:
    """Örneklemin kimliği. Zaman damgası GİRMEZ.

    Girseydi aynı girdiden üretilen iki paket farklı parmak izi taşır, iki
    etiketleyicinin çıktısı birbiriyle eşleştirilemez ve "aynı girdi aynı paket"
    testi imkânsız olurdu.
    """
    return sha256_of({
        "human_schema_version": MAST_HUMAN_SCHEMA_VERSION,
        "app_version": MAST_HUMAN_APP_VERSION,
        "mast_schema_version": MAST_SCHEMA_VERSION,
        "seed": seed, "target": target, "strata_fields": list(strata_fields),
        "records": sorted(rows, key=lambda r: r["record_id"]),
    })


def build_sample_manifest(round_: dict, *, target: int = MAST_HUMAN_SAMPLE_TARGET,
                          seed: int = MAST_HUMAN_SAMPLE_SEED,
                          strata_fields=MAST_HUMAN_STRATA_FIELDS,
                          annotators=MAST_HUMAN_ANNOTATORS,
                          expected_judges=MODEL_JUDGES) -> tuple[dict, list[dict]]:
    adaylar = collect_candidates(round_, expected_judges=expected_judges)
    secilen = select_human_sample(adaylar, target=target, seed=seed,
                                  strata_fields=strata_fields)
    satirlar = [_manifest_row(c, strata_fields) for c in secilen]
    parmak = dataset_fingerprint(satirlar, seed=seed, target=target,
                                 strata_fields=strata_fields)
    zorunlu = sum(1 for r in satirlar if r["selection_type"] != SELECTION_STRATIFIED)
    manifest = {
        "human_schema_version": MAST_HUMAN_SCHEMA_VERSION,
        "app_version": MAST_HUMAN_APP_VERSION,
        "mast_schema_version": MAST_SCHEMA_VERSION,
        "experiment": round_["experiment"],
        "created_ts": _now(),
        "source_manifest_fingerprint": round_["mast_manifest"]["source_manifest_fingerprint"],
        "mast_prompt_hash": round_["mast_manifest"]["mast_prompt_hash"],
        "source_fingerprint": source_fingerprint(round_["mast_manifest"]),
        "sample_seed": seed, "sample_target": target,
        "strata_fields": list(strata_fields),
        "annotators": list(annotators),
        "total_failures": len(adaylar),
        "mandatory_count": zorunlu,
        "selected_count": len(satirlar),
        "dataset_fingerprint": parmak,
        "records": satirlar,
    }
    return manifest, secilen


# --- Kör paket ---------------------------------------------------------------

def _validate_annotators(annotators) -> tuple[str, ...]:
    annotators = tuple(annotators)
    for a in annotators:
        if not ANNOTATOR_ID_PATTERN.match(a):
            raise HumanRoundError(
                f"geçersiz annotator kimliği {a!r} — dosya yolunda kullanılıyor, "
                "yalnız [a-z0-9_-] kabul edilir.")
    if len(set(annotators)) != len(annotators):
        raise HumanRoundError(
            f"annotator kimlikleri farklı olmalı: {list(annotators)} — aynı kimlikli "
            "iki paket 'iki bağımsız etiketleyici' iddiasını çürütür.")
    if len(annotators) != 2:
        raise HumanRoundError(f"tam iki annotator gerekir, verilen: {list(annotators)}")
    return annotators


def package_id(dataset_fp: str, *parts: str) -> str:
    return _rank(dataset_fp, MAST_HUMAN_SCHEMA_VERSION, MAST_HUMAN_APP_VERSION, *parts)


def build_blind_payload(sample: list[dict], dataset_fp: str, annotator_id: str) -> dict:
    """Kör payload — AÇIK İZİN LİSTESİ. Ham kayıt/panel sözlüğü asla girmez."""
    return {
        "human_schema_version": MAST_HUMAN_SCHEMA_VERSION,
        "app_version": MAST_HUMAN_APP_VERSION,
        "mast_schema_version": MAST_SCHEMA_VERSION,
        "dataset_fingerprint": dataset_fp,
        "package_id": package_id(dataset_fp, PHASE_BLIND, annotator_id),
        "annotator_id": annotator_id,
        "phase": PHASE_BLIND,
        "taxonomy": dict(MAST_MODES),
        "confidence_levels": list(MAST_CONFIDENCE_LEVELS),
        "confidence_anchors": dict(MAST_CONFIDENCE_ANCHORS),
        "records": [{
            "record_id": c["record_id"],
            "evidence_sha256": c["evidence_sha256"],
            "interaction_type": c["interaction_type"],
            "evidence": {k: c["evidence"][k] for k in BLIND_EVIDENCE_FIELDS},
        } for c in sample],
    }


def render_package(payload: dict, template: str) -> str:
    """Tek dosyalık statik HTML. Şablona giren TEK değişken base64 payload'dur.

    Güvenilmeyen metin (kod, traceback, mesajlar) HTML üretimine string
    interpolation ile hiç girmez; base64 çözülüp JS tarafında `textContent` ile
    gösterilir.
    """
    html = (TEMPLATE_DIR / template).read_text(encoding="utf-8")
    if PAYLOAD_TOKEN not in html:
        raise HumanRoundError(f"şablonda {PAYLOAD_TOKEN} yok: {template}")
    gomulu = base64.b64encode(canonical(payload).encode("utf-8")).decode("ascii")
    return html.replace(PAYLOAD_TOKEN, gomulu)


def prepare_blind(exp_dir: Path, *, annotators=MAST_HUMAN_ANNOTATORS,
                  target: int = MAST_HUMAN_SAMPLE_TARGET,
                  seed: int = MAST_HUMAN_SAMPLE_SEED,
                  strata_fields=MAST_HUMAN_STRATA_FIELDS,
                  expected_judges=MODEL_JUDGES) -> dict:
    annotators = _validate_annotators(annotators)
    round_ = load_mast_round(exp_dir)
    manifest, secilen = build_sample_manifest(
        round_, target=target, seed=seed, strata_fields=strata_fields,
        annotators=annotators, expected_judges=expected_judges)
    parmak = manifest["dataset_fingerprint"]

    mast_dir = _mast_dir(exp_dir)
    manifest_yolu = mast_dir / "human_sample_manifest.json"
    if manifest_yolu.exists():
        # Örneklem manifesti kör turun kimliğidir: sessizce ezilirse iki
        # etiketleyici FARKLI kayıt kümeleri üzerinde çalışmış olabilir ve
        # kilitlenmiş kör turlar artık hiçbir manifeste ait olmaz.
        mevcut = validate_sample_manifest(
            json.loads(manifest_yolu.read_text(encoding="utf-8")))
        if (mevcut["dataset_fingerprint"] != parmak
                or mevcut["annotators"] != list(annotators)):
            raise HumanRoundError(
                "bu deneyde FARKLI bir insan örneklemi zaten var "
                f"({mevcut['dataset_fingerprint'][:16]} != {parmak[:16]}, "
                f"annotators {mevcut['annotators']} vs {list(annotators)}).\n"
                "  Örneklem/annotator değişimi yeni bir turdur: mevcut "
                "mast/human_sample_manifest.json ve kilitli kör turlar "
                "korunmalı, yeni tur ayrı bir deney adı altında açılmalıdır.")
        manifest = mevcut  # idempotent: created_ts korunur

    paketler = mast_dir / "packages"
    paketler.mkdir(parents=True, exist_ok=True)
    yollar = {}
    for a in annotators:
        payload = build_blind_payload(secilen, parmak, a)
        yol = paketler / f"blind_{a}.html"
        yol.write_text(render_package(payload, "mast_blind.html"), encoding="utf-8")
        yollar[a] = yol
    if not manifest_yolu.exists():
        manifest_yolu.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {"manifest": manifest, "packages": yollar, "sample": secilen}


# --- Kilit -------------------------------------------------------------------

def lock_envelope(content: dict) -> dict:
    """Kilit zarfı: hash İÇERİĞİN dışında tutulur.

    Hash'i içeriğin içine yazmak kendine-referans üretirdi (hash'i hesaplarken
    hash alanı ne olacak?); zarf bu belirsizliği ortadan kaldırır.
    """
    return {"lock_schema_version": MAST_HUMAN_LOCK_SCHEMA_VERSION,
            "locked_ts": _now(), "content_sha256": sha256_of(content),
            "content": content}


def read_lock(path: Path) -> dict:
    zarf = json.loads(path.read_text(encoding="utf-8"))
    for alan in ("lock_schema_version", "content_sha256", "content"):
        if alan not in zarf:
            raise HumanRoundError(f"bozuk kilit dosyası (eksik {alan}): {path}")
    beklenen = sha256_of(zarf["content"])
    if zarf["content_sha256"] != beklenen:
        raise HumanRoundError(
            f"kilit hash doğrulaması BAŞARISIZ: {path}\n"
            f"  saklanan {zarf['content_sha256'][:16]} != hesaplanan {beklenen[:16]}\n"
            "  dosya kilitlendikten sonra düzenlenmiş.")
    return zarf


def write_lock(path: Path, content: dict) -> dict:
    """Kilidi yazar. Aynı içerik idempotent; FARKLI içerik üzerine YAZILMAZ.

    `--force` bilinçli olarak yok: sessizce güncellenebilen bir kilit, kilit
    değildir. Etiket değişmesi gerekiyorsa bu yeni bir turdur.
    """
    if path.exists():
        mevcut = read_lock(path)
        if mevcut["content"] == content:
            return mevcut
        raise HumanRoundError(
            f"kilitli dosya zaten var ve İÇERİĞİ FARKLI: {path}\n"
            "  kilitli kör/nihai etiketler üzerine yazılamaz — değişiklik "
            "gerekiyorsa yeni bir tur/sürüm açılmalıdır.")
    path.parent.mkdir(parents=True, exist_ok=True)
    zarf = lock_envelope(content)
    path.write_text(json.dumps(zarf, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8")
    return zarf


def _label_from(payload: dict, record_id: str, interaction: str) -> MastLabel:
    try:
        label = MastLabel.model_validate(payload)
    except Exception as e:
        raise HumanRoundError(f"{record_id}: geçersiz etiket — {e}") from e
    problems = interaction_problems(label, interaction)
    if problems:
        raise HumanRoundError(f"{record_id}: {'; '.join(problems)}")
    return label


def _check_record_set(labels: list[dict], beklenen: list[str], kaynak: str) -> None:
    gorulen = [entry.get("record_id") for entry in labels]
    yinelenen = sorted({r for r in gorulen if gorulen.count(r) > 1})
    if yinelenen:
        raise HumanRoundError(f"{kaynak}: yinelenen record_id: {yinelenen[:5]}")
    eksik = sorted(set(beklenen) - set(gorulen))
    fazla = sorted(set(gorulen) - set(beklenen))
    if eksik or fazla:
        raise HumanRoundError(
            f"{kaynak}: kayıt kümesi eşleşmiyor — {len(eksik)} eksik "
            f"{eksik[:3]}, {len(fazla)} yabancı {fazla[:3]}")


def validate_blind_export(export: dict, manifest: dict, annotator_id: str) -> dict:
    """Tarayıcı export'unu doğrular ve manifest sırasına normalize eder."""
    if not isinstance(export, dict):
        raise HumanRoundError("export bir JSON nesnesi değil.")
    bilinmeyen = sorted(set(export) - set(BLIND_EXPORT_FIELDS))
    if bilinmeyen:
        raise HumanRoundError(
            f"export bilinmeyen üst düzey alan içeriyor: {bilinmeyen} — "
            "elle düzenlenmiş ya da farklı bir sürümden gelmiş olabilir.")
    eksik = sorted(set(BLIND_EXPORT_FIELDS) - set(export))
    if eksik:
        raise HumanRoundError(f"export eksik alan içeriyor: {eksik}")

    beklenenler = {
        "human_schema_version": MAST_HUMAN_SCHEMA_VERSION,
        "app_version": MAST_HUMAN_APP_VERSION,
        "dataset_fingerprint": manifest["dataset_fingerprint"],
        "annotator_id": annotator_id,
        "phase": PHASE_BLIND,
        "package_id": package_id(manifest["dataset_fingerprint"], PHASE_BLIND,
                                 annotator_id),
    }
    farklar = {k: (export.get(k), v) for k, v in beklenenler.items() if export.get(k) != v}
    if farklar:
        raise HumanRoundError(
            f"export bu pakete ait değil: {farklar}\n"
            "  (başka etiketleyicinin, başka örneklemin ya da başka sürümün dosyası)")

    if not isinstance(export["labels"], list):
        raise HumanRoundError("labels bir liste değil.")
    beklenen_idler = [r["record_id"] for r in manifest["records"]]
    _check_record_set(export["labels"], beklenen_idler, f"{annotator_id} export")

    etkilesim = {r["record_id"]: r for r in manifest["records"]}
    by_id = {}
    for entry in export["labels"]:
        if not isinstance(entry, dict):
            raise HumanRoundError("labels içinde nesne olmayan öğe var.")
        fazla = sorted(set(entry) - set(BLIND_LABEL_FIELDS))
        if fazla:
            raise HumanRoundError(f"{entry.get('record_id')}: bilinmeyen alan {fazla}")
        rid = entry["record_id"]
        interaction = _interaction_for(etkilesim[rid])
        label = _label_from({k: v for k, v in entry.items() if k != "record_id"},
                            rid, interaction)
        by_id[rid] = {"record_id": rid, **label.model_dump()}

    return {**{k: export[k] for k in BLIND_EXPORT_FIELDS if k != "labels"},
            "labels": [by_id[rid] for rid in beklenen_idler]}


def _interaction_for(manifest_row: dict) -> str:
    """Etkileşim tipi manifest satırının KOLUNDAN türetilir.

    Kör payload'da kol adı yok, ama manifest (kör OLMAYAN, yalnız araştırmacıda
    kalan dosya) taşır — doğrulama tarafında tek-ajanlı kayıtta kategori 2
    etiketini reddedebilmek için gereken budur.
    """
    return interaction_type(manifest_row["arm"])


def validate_sample_manifest(manifest: dict) -> dict:
    """Örneklem manifestini şema + parmak izi yeniden hesabıyla doğrular.

    Manifest ham kabul edilseydi, elle düzenlenmiş bir satır (ör. bir kaydın
    `arm`'ı) hem paket kimliğini hem de `interaction_type` doğrulamasını sessizce
    değiştirirdi — kör tur başka bir kayıt kümesi üzerinde yürütülmüş olurdu.
    """
    eksik = sorted(set(MANIFEST_REQUIRED_FIELDS) - set(manifest))
    if eksik:
        raise HumanRoundError(f"örneklem manifesti eksik alan içeriyor: {eksik}")
    surumler = {"human_schema_version": MAST_HUMAN_SCHEMA_VERSION,
                "app_version": MAST_HUMAN_APP_VERSION,
                "mast_schema_version": MAST_SCHEMA_VERSION}
    farklar = {k: (manifest[k], v) for k, v in surumler.items() if manifest[k] != v}
    if farklar:
        raise HumanRoundError(
            f"örneklem manifesti farklı bir sürümle üretilmiş: {farklar} — "
            "sürüm değiştiyse bu yeni bir turdur.")
    _validate_annotators(manifest["annotators"])
    beklenen = dataset_fingerprint(manifest["records"], seed=manifest["sample_seed"],
                                   target=manifest["sample_target"],
                                   strata_fields=tuple(manifest["strata_fields"]))
    if beklenen != manifest["dataset_fingerprint"]:
        raise HumanRoundError(
            "örneklem manifesti kilitlendikten sonra düzenlenmiş: "
            f"saklanan {manifest['dataset_fingerprint'][:16]} != "
            f"hesaplanan {beklenen[:16]}")
    return manifest


def load_sample_manifest(exp_dir: Path) -> dict:
    yol = _mast_dir(exp_dir) / "human_sample_manifest.json"
    if not yol.exists():
        raise HumanRoundError(
            f"örneklem manifesti yok: {yol}\n"
            "önce: uv run python -m eval.mast_human prepare-blind --exp <exp>")
    return validate_sample_manifest(json.loads(yol.read_text(encoding="utf-8")))


def blind_lock_path(exp_dir: Path, annotator_id: str) -> Path:
    return _mast_dir(exp_dir) / f"blind_{annotator_id}.locked.json"


def lock_blind(exp_dir: Path, annotator_id: str, export_path: Path) -> dict:
    manifest = load_sample_manifest(exp_dir)
    if annotator_id not in manifest["annotators"]:
        raise HumanRoundError(
            f"{annotator_id} bu örneklemin etiketleyicisi değil: {manifest['annotators']}")
    export = json.loads(export_path.read_text(encoding="utf-8"))
    icerik = validate_blind_export(export, manifest, annotator_id)
    return write_lock(blind_lock_path(exp_dir, annotator_id), icerik)


# --- AI destekli insan adjudication -----------------------------------------

def ai_snapshot_digest(sample: list[dict], adjudications: dict[str, dict]) -> str:
    """AI panel + adjudication anlık görüntüsünün kimliği.

    Adjudication paketi bu hash'e bağlanır: AI tarafı sonradan yeniden
    etiketlenirse (panel tazelenirse ya da adjudicator başka bir karar verirse)
    eski paketle üretilmiş bir nihai insan kararı artık BAŞKA bir AI görünümüne
    dayanır ve kabul edilmemelidir.

    Full hash self-judge dahil tanısal üçlüyü; decision hash yalnız insanlara
    gösterilen iki dış etiketi kapsar. Split kayıtta Grok kararının İÇERİĞİ de
    hash'e girer. Böylece self-only değişim Grok'u stale yapmaz ama insanın AI
    görünümünü değiştirdiği için insan paketi kimliğini değiştirir.
    """
    return sha256_of([{
        "record_id": c["record_id"],
        "evidence_sha256": c["evidence_sha256"],
        "full_panel_input_sha256": c["full_panel_input_sha256"],
        "decision_input_sha256": c["decision_input_sha256"],
        "external_agreement_level": c["external_agreement_level"],
        "external_consensus_label": c["external_consensus_label"],
        "self_matches_external": c["self_matches_external"],
        "adjudicator_required": c["adjudicator_required"],
        "adjudication": _adjudication_snapshot(adjudications.get(c["source_run_id"])),
    } for c in sorted(sample, key=lambda c: c["record_id"])])


def _adjudication_snapshot(row: dict | None) -> dict | None:
    if row is None:
        return None
    return {f: row.get(f) for f in (*ADJUDICATION_MATCH_FIELDS,
                                    *ADJUDICATED_LABEL_FIELDS)}


def adjudication_identity(candidate: dict, round_: dict) -> dict:
    """Bir kayıt için BEKLENEN AI adjudication kimliği (tam provenance)."""
    return {
        "source_run_id": candidate["source_run_id"],
        "experiment": round_["experiment"],
        "source_model": candidate["model"],
        "task_set": candidate["task_set"],
        "task_id": candidate["task_id"],
        "arm": candidate["arm"],
        "repeat": candidate["repeat"],
        "mast_schema_version": MAST_SCHEMA_VERSION,
        "mast_prompt_hash": prompt_contract_hash(),
        "adjudicator_model": MODEL_ADJUDICATOR,
        "evidence_sha256": candidate["evidence_sha256"],
        "decision_input_sha256": candidate["decision_input_sha256"],
        "decision_rule_version": MAST_DECISION_RULE_VERSION,
        "self_judge_model": candidate["self_judge_model"],
        "external_judges": candidate["external_judges"],
        "external_agreement_level": "split",
        "reviewed_judges": candidate["external_judges"],
        "interaction_type": candidate["interaction_type"],
    }


def _current_adjudications(round_: dict, sample: list[dict]) -> dict[str, dict]:
    """Her kaydın GÜNCEL AI adjudication kararı — TAM kimlik karşılaştırmasıyla.

    Yalnız hash alanlarını karşılaştırmak, 6A'da judge
    katmanında kapatılan açığın insan katmanında tekrarıydı: yanlış experiment,
    üretici model, şema sürümü, prompt sürümü ya da adjudicator modeli taşıyan
    bir kayıt "güncel" sayılıyordu (canlı üretildi). Kanıt paketi kör olduğu için
    iki farklı deneyin aynı görevdeki kanıtı aynı hash'i verebilir — hash kimliğin
    yerine geçemez, ancak parçası olabilir.
    """
    by_run: dict[str, list[dict]] = {}
    for row in round_["adjudications"]:
        if row.get("adjudicator_status") == "ok":
            by_run.setdefault(row.get("source_run_id"), []).append(row)

    guncel, eksik, cakisan = {}, [], []
    for c in sample:
        if not c["adjudicator_required"]:
            continue
        kimlik = adjudication_identity(c, round_)
        uygun = [r for r in by_run.get(c["source_run_id"], [])
                 if {f: r.get(f) for f in ADJUDICATION_MATCH_FIELDS} == kimlik]
        if not uygun:
            eksik.append(c["source_run_id"])
        elif len(uygun) > 1:
            cakisan.append(c["source_run_id"])
        else:
            row = uygun[0]
            eksik_alan = sorted(set(MastAdjudication.model_fields) - set(row))
            if eksik_alan:
                raise HumanRoundError(
                    f"{c['source_run_id']}: güncel AI adjudication sözleşmesi eksik: "
                    f"{eksik_alan}")
            try:
                MastAdjudication.model_validate(
                    {k: row[k] for k in MastAdjudication.model_fields})
            except Exception as e:
                raise HumanRoundError(
                    f"{c['source_run_id']}: güncel AI adjudication geçersiz: {e}") from e
            guncel[c["source_run_id"]] = row

    if cakisan:
        raise HumanRoundError(
            f"{len(cakisan)} kayıtta AYNI tam kimlikte birden fazla başarılı AI "
            f"adjudication: {cakisan[:3]} — hangisinin gösterildiği belirsiz "
            "kalırdı; ai_adjudication.jsonl elle düzenlenmiş olabilir.")
    if eksik:
        raise HumanRoundError(
            f"{len(eksik)} kayıtta güncel AI adjudication yok: {eksik[:3]}\n"
            "önce: uv run python -m eval.mast_labels --exp <exp> --stage adjudicate")
    return guncel


def build_adjudication_payload(sample: list[dict], manifest: dict,
                               blind_locks: dict[str, dict],
                               adjudications: dict[str, dict]) -> dict:
    """AI destekli adjudication payload'u.

    İnsanlara AI MODEL ADLARI verilmez (External Annotator A/B) — otorite yanlılığı
    nihai kararı bağımsız olmaktan çıkarırdı. Üretici model ve deney kolu bu
    pakette de YOKTUR: nihai hata sınıfı kararı için gerekli değiller ve deney
    koşulu yanlılığı üretirlerdi.
    """
    parmak = manifest["dataset_fingerprint"]
    snapshot = ai_snapshot_digest(sample, adjudications)
    kilit_hashleri = {a: z["content_sha256"] for a, z in sorted(blind_locks.items())}
    kor_by_id = {a: {e["record_id"]: e for e in z["content"]["labels"]}
                 for a, z in blind_locks.items()}

    kayitlar = []
    for c in sample:
        # Yalnız iki DIŞ kayıt; self etiketi bu payload'a hiç girmez. CSS/JS ile
        # gizlemek yeterli değildir: kilit öncesi HTML kaynağından da okunamaz.
        sirali = external_annotator_order(
            c["external_judge_records"], c["source_run_id"])
        adj = adjudications.get(c["source_run_id"])
        if c["external_agreement_level"] == "consensus" and adj is not None:
            raise HumanRoundError(
                f"{c['source_run_id']}: dış konsensüste Grok kararı gösterilemez")
        if c["external_agreement_level"] == "split" and adj is None:
            raise HumanRoundError(
                f"{c['source_run_id']}: external split için güncel Grok kararı yok")
        kayitlar.append({
            "record_id": c["record_id"],
            "evidence_sha256": c["evidence_sha256"],
            "interaction_type": c["interaction_type"],
            "evidence": {k: c["evidence"][k] for k in BLIND_EVIDENCE_FIELDS},
            "blind_labels": [{
                "annotator_id": a,
                **{k: kor_by_id[a][c["record_id"]][k] for k in BLIND_LABEL_FIELDS
                   if k != "record_id"},
            } for a in sorted(blind_locks)],
            "external_ai_judges": [{
                "label": harf,
                "primary_mode": r.get("primary_mode"),
                "secondary_modes": r.get("secondary_modes") or [],
                "confidence": r.get("confidence"),
                "rationale": r.get("rationale"),
                "insufficient_context": bool(r.get("insufficient_context")),
            } for harf, r in zip("AB", sirali)],
            "external_agreement_level": c["external_agreement_level"],
            "external_consensus_label": c["external_consensus_label"],
            "full_panel_input_sha256": c["full_panel_input_sha256"],
            "decision_input_sha256": c["decision_input_sha256"],
            "grok_adjudication": None if not adj else {
                "primary_mode": adj.get("adjudicated_primary_mode"),
                "secondary_modes": adj.get("adjudicated_secondary_modes") or [],
                "confidence": adj.get("adjudicated_confidence"),
                "rationale": adj.get("adjudicated_rationale"),
                "insufficient_context": bool(adj.get("adjudicated_insufficient_context")),
            },
        })

    return {
        "human_schema_version": MAST_HUMAN_SCHEMA_VERSION,
        "app_version": MAST_HUMAN_APP_VERSION,
        "mast_schema_version": MAST_SCHEMA_VERSION,
        "dataset_fingerprint": parmak,
        "package_id": package_id(parmak, PHASE_ADJUDICATION, snapshot,
                                 *kilit_hashleri.values()),
        "phase": PHASE_ADJUDICATION,
        "blind_lock_sha256": kilit_hashleri,
        "ai_snapshot_sha256": snapshot,
        "taxonomy": dict(MAST_MODES),
        "confidence_levels": list(MAST_CONFIDENCE_LEVELS),
        "confidence_anchors": dict(MAST_CONFIDENCE_ANCHORS),
        "records": kayitlar,
    }


def _load_blind_locks(exp_dir: Path, manifest: dict) -> dict[str, dict]:
    kilitler, eksik = {}, []
    for a in manifest["annotators"]:
        yol = blind_lock_path(exp_dir, a)
        if not yol.exists():
            eksik.append(a)
            continue
        kilitler[a] = read_lock(yol)
    if eksik:
        raise HumanRoundError(
            f"kilitli kör tur eksik: {eksik} — adjudication İKİ bağımsız kör "
            "turdan sonra açılır.\n"
            "önce: uv run python -m eval.mast_human lock-blind --annotator <id> "
            "--input <export.json>")

    beklenen_idler = [r["record_id"] for r in manifest["records"]]
    for a, zarf in kilitler.items():
        icerik = zarf["content"]
        if icerik.get("annotator_id") != a:
            raise HumanRoundError(
                f"{a} kilidi başka bir etiketleyiciye ait: {icerik.get('annotator_id')}")
        if icerik.get("dataset_fingerprint") != manifest["dataset_fingerprint"]:
            raise HumanRoundError(f"{a} kilidi başka bir örnekleme ait.")
        if icerik.get("package_id") != package_id(manifest["dataset_fingerprint"],
                                                  PHASE_BLIND, a):
            raise HumanRoundError(f"{a} kilidi başka bir pakete ait.")
        _check_record_set(icerik["labels"], beklenen_idler, f"{a} kilidi")
    kimlikler = [z["content"]["annotator_id"] for z in kilitler.values()]
    if len(set(kimlikler)) != len(kimlikler):
        raise HumanRoundError(f"aynı etiketleyiciye ait iki kilit: {kimlikler}")
    return kilitler


def prepare_adjudication(exp_dir: Path, *, expected_judges=MODEL_JUDGES) -> dict:
    round_ = load_mast_round(exp_dir)
    manifest = load_sample_manifest(exp_dir)
    secilen = _sample_from_manifest(round_, manifest, expected_judges=expected_judges)
    kilitler = _load_blind_locks(exp_dir, manifest)
    adjudications = _current_adjudications(round_, secilen)
    payload = build_adjudication_payload(secilen, manifest, kilitler, adjudications)
    yol = _mast_dir(exp_dir) / "packages" / "human_adjudication.html"
    yol.parent.mkdir(parents=True, exist_ok=True)
    yol.write_text(render_package(payload, "mast_human_adjudication.html"),
                   encoding="utf-8")
    return {"payload": payload, "package": yol}


def _sample_from_manifest(round_: dict, manifest: dict, *,
                          expected_judges=MODEL_JUDGES) -> list[dict]:
    """Manifestteki örneklemi güncel MAST çıktısından yeniden kurar.

    Manifest yalnız KİMLİK taşır (kanıt değil); kanıt her zaman güncel
    kaynaktan üretilir ve hash'i manifesttekiyle karşılaştırılır — böylece
    kaynak değişmişse paket üretilemez.
    """
    adaylar = {c["record_id"]: c
               for c in collect_candidates(round_, expected_judges=expected_judges)}
    secilen, sorunlar = [], []
    for satir in manifest["records"]:
        c = adaylar.get(satir["record_id"])
        if c is None:
            sorunlar.append(f"{satir['record_id']}: kayıt güncel MAST turunda yok")
            continue
        if c["evidence_sha256"] != satir["evidence_sha256"]:
            sorunlar.append(f"{satir['record_id']}: kanıt değişmiş")
            continue
        if c["full_panel_input_sha256"] != satir["full_panel_input_sha256"]:
            sorunlar.append(f"{satir['record_id']}: tanısal AI paneli değişmiş")
            continue
        if c["decision_input_sha256"] != satir["decision_input_sha256"]:
            sorunlar.append(f"{satir['record_id']}: dış karar girdisi değişmiş")
            continue
        secilen.append(c)
    if sorunlar:
        raise HumanRoundError(
            "örneklem güncel MAST turuyla uyuşmuyor — bu paket bu pakete ait "
            "olmayan bir AI görünümüne dayanırdı (paket YAZILMADI):\n  - "
            + "\n  - ".join(sorunlar[:8])
            + "\nAI tarafı yeniden etiketlendiyse insan turu da yenidir: "
              "prepare-blind ile yeni bir örneklem/tur açılmalıdır (eski kilitli "
              "kör turlar korunur, üzerine yazılmaz).")
    return secilen


def validate_adjudication_export(export: dict, manifest: dict,
                                 blind_locks: dict[str, dict],
                                 expected_package_id: str) -> dict:
    if not isinstance(export, dict):
        raise HumanRoundError("export bir JSON nesnesi değil.")
    bilinmeyen = sorted(set(export) - set(ADJ_EXPORT_FIELDS))
    if bilinmeyen:
        raise HumanRoundError(f"export bilinmeyen üst düzey alan içeriyor: {bilinmeyen}")
    eksik = sorted(set(ADJ_EXPORT_FIELDS) - set(export))
    if eksik:
        raise HumanRoundError(f"export eksik alan içeriyor: {eksik}")

    beklenenler = {
        "human_schema_version": MAST_HUMAN_SCHEMA_VERSION,
        "app_version": MAST_HUMAN_APP_VERSION,
        "dataset_fingerprint": manifest["dataset_fingerprint"],
        "phase": PHASE_ADJUDICATION,
        "package_id": expected_package_id,
    }
    farklar = {k: (export.get(k), v) for k, v in beklenenler.items() if export.get(k) != v}
    if farklar:
        raise HumanRoundError(
            f"export bu pakete ait değil: {farklar}\n"
            "  (AI paneli ya da kör kilitler değişmişse paket yeniden üretilmelidir)")

    beklenen_kilitler = {a: z["content_sha256"] for a, z in blind_locks.items()}
    if export["blind_lock_sha256"] != beklenen_kilitler:
        raise HumanRoundError(
            "export'taki kör kilit hash'leri diskteki kilitlerle eşleşmiyor — "
            "kör etiketler adjudication'dan sonra değiştirilmiş olabilir.")

    beklenen_idler = [r["record_id"] for r in manifest["records"]]
    _check_record_set(export["labels"], beklenen_idler, "adjudication export")
    satirlar = {r["record_id"]: r for r in manifest["records"]}
    by_id = {}
    for entry in export["labels"]:
        fazla = sorted(set(entry) - {"record_id", "human_adjudicated_label"})
        if fazla:
            # Kör kararlar YALNIZ kilitte yaşar; export'ta taşınan bir kopya
            # "hangisi geçerli" belirsizliği üretir ve değiştirilmelerine kapı açar.
            raise HumanRoundError(
                f"{entry.get('record_id')}: adjudication export'unda yalnız "
                f"human_adjudicated_label düzenlenebilir; fazla alan: {fazla}")
        rid = entry["record_id"]
        label = _label_from(entry["human_adjudicated_label"], rid,
                            _interaction_for(satirlar[rid]))
        by_id[rid] = {"record_id": rid, "human_adjudicated_label": label.model_dump()}

    return {**{k: export[k] for k in ADJ_EXPORT_FIELDS if k != "labels"},
            "labels": [by_id[rid] for rid in beklenen_idler]}


def adjudication_lock_path(exp_dir: Path) -> Path:
    return _mast_dir(exp_dir) / "human_adjudication.locked.json"


def lock_adjudication(exp_dir: Path, export_path: Path, *,
                      expected_judges=MODEL_JUDGES) -> dict:
    round_ = load_mast_round(exp_dir)
    manifest = load_sample_manifest(exp_dir)
    secilen = _sample_from_manifest(round_, manifest, expected_judges=expected_judges)
    kilitler = _load_blind_locks(exp_dir, manifest)
    adjudications = _current_adjudications(round_, secilen)
    payload = build_adjudication_payload(secilen, manifest, kilitler, adjudications)
    export = json.loads(export_path.read_text(encoding="utf-8"))
    icerik = validate_adjudication_export(export, manifest, kilitler,
                                          payload["package_id"])
    return write_lock(adjudication_lock_path(exp_dir), icerik)


def build_diagnostic_payload(sample: list[dict], manifest: dict,
                             final_lock: dict, current_package_id: str,
                             ai_snapshot_sha256: str) -> dict:
    """Nihai kilit SONRASI self-judge görünümü; salt okunur ve ayrı paket.

    Self etiketi adjudication HTML'ine önceden gömülmez. Bu ayrı paketin
    üretilebilmesi için nihai kilidin güncel AI snapshot'ına bağlı package_id'si
    doğrulanmış olmalıdır; böylece HTML kaynağını incelemek kilit öncesi self
    etiketini açığa çıkaramaz.
    """
    content = final_lock["content"]
    if content.get("package_id") != current_package_id:
        raise HumanRoundError(
            "nihai insan kilidi güncel AI snapshot'ına ait değil — diagnostic "
            "paket açılamaz; önce yeni adjudication turu kilitlenmelidir.")
    final_by_id = {r["record_id"]: r["human_adjudicated_label"]
                   for r in content["labels"]}
    expected_ids = [r["record_id"] for r in manifest["records"]]
    _check_record_set(content["labels"], expected_ids, "nihai insan kilidi")

    records = []
    for c in sample:
        self_label = c["self_judge_record"]
        records.append({
            "record_id": c["record_id"],
            "task_id": c["task_id"],
            "human_adjudicated_label": final_by_id[c["record_id"]],
            # Self etiketi ÜRETİLEMEMİŞ olabilir (judge şema uyumsuzluğu). O
            # durumda alan `null`dur ve arayüz "mevcut değil" gösterir; boş bir
            # etiket nesnesi uydurmak "etiket verildi ama boş" izlenimi yaratırdı.
            "self_judge_available": c["self_judge_available"],
            "self_judge_label": None if self_label is None else {
                "primary_mode": self_label.get("primary_mode"),
                "secondary_modes": self_label.get("secondary_modes") or [],
                "confidence": self_label.get("confidence"),
                "rationale": self_label.get("rationale"),
                "insufficient_context": bool(self_label.get("insufficient_context")),
            },
            "external_agreement_level": c["external_agreement_level"],
            "external_consensus_label": c["external_consensus_label"],
            "self_matches_external": c["self_matches_external"],
        })
    return {
        "human_schema_version": MAST_HUMAN_SCHEMA_VERSION,
        "app_version": MAST_HUMAN_APP_VERSION,
        "mast_schema_version": MAST_SCHEMA_VERSION,
        "dataset_fingerprint": manifest["dataset_fingerprint"],
        "package_id": package_id(
            manifest["dataset_fingerprint"], PHASE_DIAGNOSTIC,
            final_lock["content_sha256"], ai_snapshot_sha256),
        "phase": PHASE_DIAGNOSTIC,
        "final_lock_sha256": final_lock["content_sha256"],
        "ai_snapshot_sha256": ai_snapshot_sha256,
        "records": records,
    }


def prepare_diagnostic(exp_dir: Path, *, expected_judges=MODEL_JUDGES) -> dict:
    """Geçerli nihai insan kilidinden sonra salt-okunur self diagnostic paketi."""
    round_ = load_mast_round(exp_dir)
    manifest = load_sample_manifest(exp_dir)
    sample = _sample_from_manifest(round_, manifest, expected_judges=expected_judges)
    blind_locks = _load_blind_locks(exp_dir, manifest)
    adjudications = _current_adjudications(round_, sample)
    current = build_adjudication_payload(sample, manifest, blind_locks, adjudications)
    lock_path = adjudication_lock_path(exp_dir)
    if not lock_path.exists():
        raise HumanRoundError(
            "self-judge diagnostic görünümü yalnız nihai insan kararı "
            "kilitlendikten sonra açılabilir.")
    final_lock = read_lock(lock_path)
    normalized = validate_adjudication_export(
        final_lock["content"], manifest, blind_locks, current["package_id"])
    if normalized != final_lock["content"]:
        raise HumanRoundError(
            "nihai insan kilidi kanonik/güncel adjudication sözleşmesiyle uyuşmuyor")
    payload = build_diagnostic_payload(
        sample, manifest, final_lock, current["package_id"],
        current["ai_snapshot_sha256"])
    path = _mast_dir(exp_dir) / "packages" / "post_lock_diagnostic.html"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_package(payload, "mast_post_lock_diagnostic.html"),
                    encoding="utf-8")
    return {"payload": payload, "package": path}


# --- Betimleyici uyum özeti (§8.5) ------------------------------------------

def cohens_kappa(a: list[str], b: list[str]) -> tuple[float | None, str | None]:
    """Cohen's κ; beklenen uyum 1'e eşitse `None` + açıklama döner.

    Sessiz `NaN` üretmek, 0/0 durumunu "hesaplandı ama tuhaf" gibi gösterirdi.
    Bu uç durum gerçektir: iki etiketleyici de TEK bir kodu kullandığında
    beklenen uyum 1 olur ve κ tanımsızdır.
    """
    n = len(a)
    if n == 0:
        return None, "kayıt yok"
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    kodlar = set(a) | set(b)
    pe = sum((a.count(k) / n) * (b.count(k) / n) for k in kodlar)
    if abs(1.0 - pe) < 1e-12:
        return None, ("beklenen uyum 1'e eşit (her iki etiketleyici de tek bir kod "
                      "kullandı) — κ tanımsız")
    return round((po - pe) / (1.0 - pe), 4), None


def _key(label: dict) -> str:
    return INSUFFICIENT_SENTINEL if label.get("insufficient_context") else label["primary_mode"]


def build_agreement_summary(manifest: dict, blind_locks: dict[str, dict],
                            adjudication: dict, sample: list[dict],
                            adjudications: dict[str, dict]) -> dict:
    idler = [r["record_id"] for r in manifest["records"]]
    kor = {a: {e["record_id"]: e for e in z["content"]["labels"]}
           for a, z in blind_locks.items()}
    nihai = {e["record_id"]: e["human_adjudicated_label"]
             for e in adjudication["content"]["labels"]}
    c_by_id = {c["record_id"]: c for c in sample}

    a_id, b_id = sorted(kor)
    ka = [_key(kor[a_id][r]) for r in idler]
    kb = [_key(kor[b_id][r]) for r in idler]
    kappa, kappa_note = cohens_kappa(ka, kb)

    final = [_key(nihai[r]) for r in idler]

    def _agreement(left: list[str], right: list[str]) -> tuple[float | None, int]:
        n = len(left)
        if n == 0:
            return None, 0
        return round(sum(1 for x, y in zip(left, right) if x == y) / n, 4), n

    consensus_human, consensus_ai = [], []
    grok_human, grok_ai = [], []
    self_ai, consensus_for_self = [], []
    blind_a_consensus, blind_b_consensus = [], []
    blind_a_grok, blind_b_grok = [], []
    external_levels = {"consensus": 0, "split": 0, "incomplete": 0}
    external_insufficient = 0
    grok_insufficient = 0
    self_insufficient = 0
    self_missing = 0
    for index, (rid, human_key) in enumerate(zip(idler, final)):
        c = c_by_id[rid]
        level = c["external_agreement_level"]
        external_levels[level] = external_levels.get(level, 0) + 1
        if any(r.get("insufficient_context") for r in c["external_judge_records"]):
            external_insufficient += 1
        if not c["self_judge_available"]:
            # Self ETİKETİ ÜRETİLEMEDİ. Payda dışında bırakılır ama SAYILIR:
            # sessizce düşürmek, self–dış uyum oranını hiç ölçülmemiş kayıtlar
            # üzerinden hesaplanmış gibi gösterirdi.
            self_missing += 1
        if level == "consensus":
            consensus_human.append(human_key)
            consensus_ai.append(c["external_consensus_label"])
            blind_a_consensus.append(ka[index])
            blind_b_consensus.append(kb[index])
            if c["self_judge_available"]:
                self_ai.append(_key(c["self_judge_record"]))
                consensus_for_self.append(c["external_consensus_label"])
                self_insufficient += int(bool(
                    c["self_judge_record"].get("insufficient_context")))
        elif level == "split":
            adj = adjudications.get(c["source_run_id"])
            if adj is None:
                raise HumanRoundError(
                    f"{c['source_run_id']}: özet için güncel Grok kararı yok")
            blind_a_grok.append(ka[index])
            blind_b_grok.append(kb[index])
            grok_human.append(human_key)
            grok_key = (INSUFFICIENT_SENTINEL
                        if adj.get("adjudicated_insufficient_context")
                        else adj.get("adjudicated_primary_mode"))
            grok_ai.append(grok_key)
            grok_insufficient += int(grok_key == INSUFFICIENT_SENTINEL)

    hh_rate, hh_n = _agreement(ka, kb)
    final_external_rate, final_external_n = _agreement(consensus_human, consensus_ai)
    final_grok_rate, final_grok_n = _agreement(grok_human, grok_ai)
    self_external_rate, self_external_n = _agreement(self_ai, consensus_for_self)
    blind_a_external_rate, _ = _agreement(blind_a_consensus, consensus_ai)
    blind_b_external_rate, _ = _agreement(blind_b_consensus, consensus_ai)
    blind_a_grok_rate, _ = _agreement(blind_a_grok, grok_ai)
    blind_b_grok_rate, _ = _agreement(blind_b_grok, grok_ai)

    dagilim = {}
    for ad, seri in (("annotator_" + a_id, ka), ("annotator_" + b_id, kb),
                     ("human_adjudicated", final)):
        sayim = {}
        for k in seri:
            sayim[str(k)] = sayim.get(str(k), 0) + 1
        dagilim[ad] = dict(sorted(sayim.items()))

    return {
        "human_schema_version": MAST_HUMAN_SCHEMA_VERSION,
        "experiment": manifest["experiment"],
        "dataset_fingerprint": manifest["dataset_fingerprint"],
        "created_ts": _now(),
        "n_records": len(idler),
        "annotators": [a_id, b_id],
        "human_human_exact_primary_agreement": hh_rate,
        "human_human_n_comparable": hh_n,
        "cohens_kappa": kappa,
        "cohens_kappa_note": kappa_note,
        "human_final_vs_external_consensus_agreement": final_external_rate,
        "human_final_vs_external_consensus_n_comparable": final_external_n,
        "human_final_vs_external_consensus_undefined_count": len(idler) - final_external_n,
        "human_final_vs_grok_split_agreement": final_grok_rate,
        "human_final_vs_grok_split_n_comparable": final_grok_n,
        "human_final_vs_grok_split_undefined_count": len(idler) - final_grok_n,
        f"blind_{a_id}_vs_external_consensus_agreement": blind_a_external_rate,
        f"blind_{b_id}_vs_external_consensus_agreement": blind_b_external_rate,
        "blind_human_vs_external_consensus_n_comparable": final_external_n,
        f"blind_{a_id}_vs_grok_split_agreement": blind_a_grok_rate,
        f"blind_{b_id}_vs_grok_split_agreement": blind_b_grok_rate,
        "blind_human_vs_grok_split_n_comparable": final_grok_n,
        "self_vs_external_consensus_agreement": self_external_rate,
        "self_vs_external_consensus_n_comparable": self_external_n,
        "self_vs_external_consensus_undefined_count": len(idler) - self_external_n,
        # Tanımsızlığın İKİ ayrı sebebi vardır ve karıştırılmamalıdır: dış taraf
        # split olduğu için karşılaştırma tanımsızdır (normal), YA DA self etiketi
        # hiç üretilememiştir (ölçüm kaybı). İkincisi ayrıca sayılır.
        "self_judge_missing_count": self_missing,
        "external_agreement_level_counts": external_levels,
        "external_insufficient_context_record_count": external_insufficient,
        "grok_insufficient_context_count": grok_insufficient,
        "self_insufficient_context_consensus_count": self_insufficient,
        "insufficient_context_counts": {
            a_id: sum(1 for r in idler if kor[a_id][r]["insufficient_context"]),
            b_id: sum(1 for r in idler if kor[b_id][r]["insufficient_context"]),
        },
        "changed_from_blind_to_final": {
            a_id: sum(1 for r in idler if _key(kor[a_id][r]) != _key(nihai[r])),
            b_id: sum(1 for r in idler if _key(kor[b_id][r]) != _key(nihai[r])),
        },
        "primary_mode_distributions": dagilim,
        "selection_type_counts": {
            t: sum(1 for r in manifest["records"] if r["selection_type"] == t)
            for t in (SELECTION_MANDATORY_DISAGREEMENT, SELECTION_MANDATORY_INSUFFICIENT,
                      SELECTION_MANDATORY_SELF_MISMATCH, SELECTION_STRATIFIED)},
        # §8.5: bu oranlar HATA EVRENİNE genellenemez.
        "generalization_note": (
            "İnsan örneklemi anlaşmazlık ve düşük bağlam bakımından BİLİNÇLİ olarak "
            "zenginleştirilmiş bir alt kümedir. Buradaki uyum oranları ve Cohen kappa yalnız "
            "BETİMLEYİCİDİR; tüm hata evrenine genellenemez ve hipotez testi olarak "
            "yorumlanamaz."),
    }


def summarize(exp_dir: Path, *, expected_judges=MODEL_JUDGES) -> dict:
    round_ = load_mast_round(exp_dir)
    manifest = load_sample_manifest(exp_dir)
    secilen = _sample_from_manifest(round_, manifest, expected_judges=expected_judges)
    kilitler = _load_blind_locks(exp_dir, manifest)
    yol = adjudication_lock_path(exp_dir)
    if not yol.exists():
        raise HumanRoundError(
            f"nihai adjudication kilidi yok: {yol}\n"
            "önce: uv run python -m eval.mast_human lock-adjudication --input <export.json>")
    adjudication = read_lock(yol)
    ai_adj = _current_adjudications(round_, secilen)
    current_payload = build_adjudication_payload(
        secilen, manifest, kilitler, ai_adj)
    normalized = validate_adjudication_export(
        adjudication["content"], manifest, kilitler, current_payload["package_id"])
    if normalized != adjudication["content"]:
        raise HumanRoundError(
            "nihai insan kilidi kanonik/güncel adjudication sözleşmesiyle uyuşmuyor")
    ozet = build_agreement_summary(manifest, kilitler, adjudication, secilen, ai_adj)
    (_mast_dir(exp_dir) / "agreement_summary.json").write_text(
        json.dumps(ozet, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return ozet


# --- CLI ---------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Kör insan MAST etiketleme turu + AI destekli adjudication")
    alt = parser.add_subparsers(dest="cmd", required=True)

    def _exp(p):
        p.add_argument("--exp", required=True, type=Path,
                       help="deney dizini (logs/exp_<name>)")
        return p

    pb = _exp(alt.add_parser("prepare-blind", help="örneklem + iki kör HTML paketi"))
    pb.add_argument("--annotators", nargs=2, default=list(MAST_HUMAN_ANNOTATORS),
                    metavar="ID", help="iki FARKLI etiketleyici kimliği [a-z0-9_-]")

    lb = _exp(alt.add_parser("lock-blind", help="kör tur export'unu doğrula ve kilitle"))
    lb.add_argument("--annotator", required=True)
    lb.add_argument("--input", required=True, type=Path, help="tarayıcı export JSON'u")

    _exp(alt.add_parser("prepare-adjudication", help="AI destekli adjudication paketi"))

    la = _exp(alt.add_parser("lock-adjudication", help="nihai kararı doğrula ve kilitle"))
    la.add_argument("--input", required=True, type=Path)

    _exp(alt.add_parser(
        "prepare-diagnostic",
        help="nihai kilit sonrası salt-okunur self-judge diagnostic paketi"))

    _exp(alt.add_parser("summarize", help="betimleyici uyum özeti"))

    args = parser.parse_args()

    try:
        if args.cmd == "prepare-blind":
            sonuc = prepare_blind(args.exp, annotators=tuple(args.annotators))
            m = sonuc["manifest"]
            print(f"deney: {m['experiment']} | toplam hata: {m['total_failures']} | "
                  f"zorunlu: {m['mandatory_count']} | seçilen: {m['selected_count']}")
            print(f"dataset_fingerprint: {m['dataset_fingerprint'][:16]}")
            for a, yol in sonuc["packages"].items():
                print(f"  {a}: {yol}")
            print("\nHer etiketleyici KENDİ dosyasını Edge/Chrome'da açar "
                  "(file://), doldurur ve JSON export alır.")
            print("localStorage yalnız kolaylık katmanıdır — otoritatif kayıt "
                  "export + lock-blind hattıdır.")
        elif args.cmd == "lock-blind":
            zarf = lock_blind(args.exp, args.annotator, args.input)
            print(f"kilitlendi: {blind_lock_path(args.exp, args.annotator)}")
            print(f"content_sha256: {zarf['content_sha256']}")
        elif args.cmd == "prepare-adjudication":
            sonuc = prepare_adjudication(args.exp)
            print(f"adjudication paketi: {sonuc['package']}")
            print(f"package_id: {sonuc['payload']['package_id'][:16]}")
        elif args.cmd == "lock-adjudication":
            zarf = lock_adjudication(args.exp, args.input)
            print(f"kilitlendi: {adjudication_lock_path(args.exp)}")
            print(f"content_sha256: {zarf['content_sha256']}")
        elif args.cmd == "prepare-diagnostic":
            sonuc = prepare_diagnostic(args.exp)
            print(f"kilit sonrası diagnostic paket: {sonuc['package']}")
            print(f"package_id: {sonuc['payload']['package_id'][:16]}")
        elif args.cmd == "summarize":
            ozet = summarize(args.exp)
            print(json.dumps({k: v for k, v in ozet.items()
                              if k != "generalization_note"},
                             indent=2, ensure_ascii=False))
            print("\n" + ozet["generalization_note"])
    except (HumanRoundError, MastPipelineError) as e:
        sys.exit(str(e))


if __name__ == "__main__":
    main()
