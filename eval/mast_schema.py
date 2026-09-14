"""MAST etiket sözleşmesi — AI judge, insan turu ve adjudication ORTAK kaynağı.

MAST: "Why Do Multi-Agent LLM Systems Fail?" (Cemri vd., arXiv:2503.13657) —
14 hata modu, 3 kategori. Mod tanımları resmi repo'dan alındı
(github.com/multi-agent-systems-failure-taxonomy/MAST, 2026-07-20'de doğrulandı).

Bu modül üç şeyi tanımlar ve üç tüketici (AI paneli, kör insan arayüzü,
adjudication) bunları PAYLAŞIR — aksi halde "insan şeması" ile "judge şeması"
sessizce ayrışır ve uyum oranı ölçülemez hale gelir:

1. `MastLabel` — bir etiketin ne anlama geldiği ve hangi kombinasyonların
   yasak olduğu.
2. Etiket kaydının provenance zarfı — hangi koşuya, hangi kanıta, hangi
   etiketleyiciye ait (§9). `task_id + arm` YETMEZ: üç tekrar ve iki model
   varken aynı görev-kol çifti altı farklı koşuya karşılık gelir.
3. `panel_verdict()` — üç bağımsız etiketten anlaşma durumunun TÜRETİLMESİ.
"""

import hashlib
import json
import re
from datetime import datetime, timezone
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    field_validator,
    model_validator,
)

from config import (
    MAST_CONFIDENCE_LEVELS,
    MAST_DECISION_RULE_VERSION,
    MAST_PANEL_HASH_VERSION,
    MAST_SCHEMA_VERSION,
    MODEL_ADJUDICATOR,
    MODEL_JUDGES,
    MODEL_PRODUCERS,
)

# Resmi taksonomi tanımları (kısaltılmış; kaynak: MAST repo definitions.txt)
MAST_MODES: dict[str, str] = {
    "1.1": "Disobey Task Specification — fails to adhere to task constraints/requirements",
    "1.2": "Disobey Role Specification — behaves outside its defined role",
    "1.3": "Step Repetition — unnecessarily repeats already-completed steps",
    "1.4": "Loss of Conversation History — disregards recent context, reverts to earlier state",
    "1.5": "Unaware of Termination Conditions — fails to recognize when to stop",
    "2.1": "Conversation Reset — unwarranted restart losing context and progress",
    "2.2": "Fail to Ask for Clarification — proceeds despite unclear/incomplete information",
    "2.3": "Task Derailment — deviates from intended task objectives",
    "2.4": "Information Withholding — has critical information but fails to share it",
    "2.5": "Ignored Other Agent's Input — fails to consider other agents' contributions",
    "2.6": "Action-Reasoning Mismatch — stated reasoning conflicts with actual output",
    "3.1": "Premature Termination — ends before objectives are met",
    "3.2": "Weak Verification — verification exists but misses essential aspects",
    "3.3": "No or Incorrect Verification — outcomes not (or wrongly) checked",
}
NONE_CODE = "none"
VALID_CODES = set(MAST_MODES) | {NONE_CODE}

# Kategori 2 modları AJANLAR ARASI iletişime dairdir; tek-ajanlı baseline'da
# yapısal olarak uygulanamaz (kol adı verilmeden judge'a bu bildirilir).
INTER_AGENT_CODES = {c for c in MAST_MODES if c.startswith("2.")}

# panel_verdict()'in karşılaştırmada "yetersiz bağlam" için kullandığı sentinel.
# primary_mode=None ile taksonomi kodu asla karışmasın diye ayrı bir dize.
INSUFFICIENT_SENTINEL = "__insufficient_context__"

JUDGE_STATUS_OK = "ok"
AGREEMENT_LEVELS = ("unanimous", "majority", "split", "incomplete")
# Karar katmanı yalnız İKİ DIŞ judge'a bakar; bu yüzden ayrı bir seviye kümesi
# (üçlü panelde "majority" mümkündür, iki judge arasında değildir).
EXTERNAL_AGREEMENT_LEVELS = ("consensus", "split", "incomplete")


class MastPipelineError(RuntimeError):
    """MAST hattı güvenle çalıştırılamaz — bozuk panel üretmektense durulur."""


class MastLabel(BaseModel):
    """Bir etiketleyicinin (AI ya da insan) tek karar çıktısı.

    Yasak kombinasyonlar şema düzeyinde reddedilir; "sonra analizde temizleriz"
    yaklaşımı, uyum oranını hesaplarken hangi kaydın geçerli olduğunu belirsiz
    bırakırdı.
    """

    primary_mode: str | None = None
    secondary_modes: list[str] = Field(default_factory=list)
    confidence: str
    rationale: str
    insufficient_context: bool = False

    @field_validator("confidence")
    @classmethod
    def _confidence_gecerli(cls, v: str) -> str:
        if v not in MAST_CONFIDENCE_LEVELS:
            raise ValueError(f"confidence {MAST_CONFIDENCE_LEVELS} içinden olmalı, alınan {v!r}")
        return v

    @field_validator("rationale")
    @classmethod
    def _gerekce_bos_olamaz(cls, v: str) -> str:
        # Boş gerekçe, etiketi denetlenemez kılar: insan adjudication turunda
        # "bu kodu neden verdi" sorusunun cevabı kalmaz.
        if not v.strip():
            raise ValueError("rationale boş olamaz")
        return v.strip()

    @model_validator(mode="after")
    def _kombinasyon_kurallari(self):
        if self.insufficient_context:
            # Yetersiz bağlam BİR ETİKET DEĞİL, etiket verilememesidir. Aksi
            # halde "hem bilmiyorum hem 1.1" gibi bir kayıt uyum hesabında
            # hangi tarafa sayılacağı belirsiz kalırdı.
            if self.primary_mode is not None or self.secondary_modes:
                raise ValueError(
                    "insufficient_context=True iken normal etiket verilemez "
                    "(primary_mode=None, secondary_modes=[])")
            return self

        if self.primary_mode is None:
            raise ValueError("primary_mode zorunlu (yetersiz bağlam ise insufficient_context=True)")
        gecersiz = [c for c in [self.primary_mode, *self.secondary_modes] if c not in VALID_CODES]
        if gecersiz:
            raise ValueError(f"geçersiz MAST kodu: {gecersiz}")
        if len(set(self.secondary_modes)) != len(self.secondary_modes):
            raise ValueError(f"secondary_modes yinelenen kod içeriyor: {self.secondary_modes}")
        if self.primary_mode in self.secondary_modes:
            raise ValueError("primary_mode secondary_modes içinde tekrarlanamaz")
        # "none" = hiçbir taksonomi modu uygulanmıyor; başka bir kodla birlikte
        # kullanılması kendi kendisiyle çelişir.
        if NONE_CODE in [self.primary_mode, *self.secondary_modes] and (
                self.secondary_modes or self.primary_mode != NONE_CODE):
            raise ValueError("'none' başka bir kodla birlikte kullanılamaz")
        return self

    @property
    def all_modes(self) -> list[str]:
        return [] if self.insufficient_context else [self.primary_mode, *self.secondary_modes]

    @property
    def comparison_key(self) -> str:
        """Panel karşılaştırmasında kullanılan primary değeri."""
        return INSUFFICIENT_SENTINEL if self.insufficient_context else self.primary_mode


def interaction_problems(label: MastLabel, interaction_type: str) -> list[str]:
    """Etiketin ETKİLEŞİM TİPİYLE uyumu — judge/adjudicator/insan ORTAK kuralı.

    Kategori 2 modları ajanlar arası iletişime dairdir; tek-ajanlı bir koşuda
    yapısal olarak imkânsızdır. Şema tek başına bunu göremez (2.5 geçerli bir
    koddur), bu yüzden ayrı bir kontrol gerekir. Üç etiketleyici türünde de
    AYNI fonksiyon çağrılır — biri gevşek kalırsa baseline'daki "iletişim
    hatası" etiketleri hata dağılımını doğrudan bozar.
    """
    if interaction_type != "single_agent":
        return []
    yasak = [c for c in label.all_modes if c in INTER_AGENT_CODES]
    if yasak:
        return [f"tek-ajanlı koşuda ajanlar-arası mod kullanılamaz: {yasak}"]
    return []


def evidence_digest(evidence: dict) -> str:
    """Kanıt paketinin SHA-256'sı.

    Üç judge'ın (ve iki insanın) AYNI kanıtı gördüğünü kanıtlar. Etiketler
    farklıysa bunun sebebinin farklı kanıt olmadığı ancak böyle gösterilebilir.
    """
    payload = json.dumps(evidence, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# Bir etiketin PROMPT'TA GÖRÜLEN içeriği. `judge_attempt`/`ts`/`judge_raw` ve
# sağlayıcı/token/maliyet alanları bilinçli olarak YOK: yeniden denenip AYNI
# kararı veren bir judge adjudication'ı geçersiz kılmamalı — adjudicator etiketin
# içeriğine bakar, kaçıncı denemede hangi sağlayıcıdan üretildiğine değil.
# `source_run_id`, kanıt ve prompt hash'i de yok: onlar ÜST resume kimliğinde
# (ADJUDICATION_IDENTITY_FIELDS) ayrı alanlar olarak zaten duruyor; burada
# tekrarlamak aynı bilgiyi iki yerde tutmak olurdu.
PANEL_INPUT_FIELDS = ("primary_mode", "secondary_modes", "confidence", "rationale",
                      "insufficient_context")

HASH_KIND_FULL = "full_panel"
HASH_KIND_DECISION = "decision_input"
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


def _label_payload(judge_record: dict) -> dict:
    return {"judge_model": judge_record.get("judge_model"),
            **{k: judge_record.get(k) for k in PANEL_INPUT_FIELDS}}


def _judges_by_model(judge_records: list[dict], expected) -> dict[str, dict]:
    """Model→etiket haritası; yinelenen ve beklenmeyen judge fail-fast.

    Sessizce hash'lemek en kötü seçenek olurdu: bozuk bir kümenin hash'i de
    64 hex karakterdir, yani kayıt "sağlıklı" görünürken karşılaştırdığı şey
    başka bir panel olurdu.
    """
    by_model: dict[str, dict] = {}
    for r in judge_records:
        m = r.get("judge_model")
        if m in by_model:
            raise MastPipelineError(
                f"hash girdisinde aynı judge için birden fazla etiket: {m!r}")
        by_model[m] = r
    beklenmeyen = sorted(set(by_model) - set(expected))
    if beklenmeyen:
        raise MastPipelineError(
            f"hash girdisinde beklenmeyen judge: {beklenmeyen}; beklenen {list(expected)}")
    return by_model


def _input_digest(by_model: dict[str, dict], order, kind: str) -> str:
    eksik = [m for m in order if m not in by_model]
    if eksik:
        raise MastPipelineError(
            f"{kind} hash'i için eksik judge: {eksik} — yarım bir kümenin hash'i "
            "tamamlanmış bir girdi izlenimi verirdi")
    payload = {
        # Sürüm ve TÜR payload'ın içindedir: aynı üç etiket için full ve decision
        # hash'leri (ör. self eksikken) çakışamaz, ve kanonikleştirme kuralı
        # değişirse eski hash'ler yeni hash'lerle sessizce karşılaştırılamaz.
        "panel_hash_version": MAST_PANEL_HASH_VERSION,
        "hash_kind": kind,
        "labels": [_label_payload(by_model[m]) for m in order],
    }
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def full_panel_input_digest(judge_records: list[dict], expected_judges) -> str:
    """ÜÇ tanısal etiketin kanonik hash'i — yalnız PANEL tazeliği.

    Adjudication resume kimliğine GİRMEZ: kaydı üreten modelin kendi etiketi
    değiştiğinde Grok kararı geçersiz sayılırsa, karara hiç katılmamış bir
    etiket yüzünden ücretli çağrı tekrarlanır ve karar yanlış bir girdiye
    bağımlı gösterilirdi.

    Sıra `expected_judges` (dondurulmuş kadro) sırasıdır; girdi listesinin
    geliş sırasından bağımsızdır.
    """
    expected = validate_expected_judges(expected_judges)
    return _input_digest(_judges_by_model(judge_records, expected), expected,
                         HASH_KIND_FULL)


def decision_input_digest(judge_records: list[dict], expected_judges, *,
                          source_model: str) -> str:
    """KARARA giren iki dış etiketin kanonik hash'i (leave-self-out).

    Adjudication resume kimliğinin parçasıdır: bir dış etiket değişirse eski
    Grok kararı başka bir girdinin ürünüdür ve superseded olmalıdır; self etiketi
    değişirse karar girdisi değişmediği için mevcut karar GÜNCEL kalır.

    `source_model` zorunludur ama self etiketi payload'a GİRMEZ. Dış liste
    burada elle kurulmaz — `judge_role_partition()` kullanılır; iki yerde ayrı
    yazılsaydı karar katmanı ile hash katmanı sessizce ayrışabilirdi.

    Self etiketinin kümede bulunması gerekmez (Parça 4'te prompt'a yalnız iki
    dış etiket gidecek), ama varsa beklenen kümeye ait olmalıdır.
    """
    expected = validate_expected_judges(expected_judges)
    _, external = judge_role_partition(source_model, expected)
    return _input_digest(_judges_by_model(judge_records, expected), external,
                         HASH_KIND_DECISION)


def label_provenance(record: dict, *, experiment: str, evidence_sha256: str,
                     prompt_hash: str | None = None,
                     interaction_type: str | None = None) -> dict:
    """Etiket kaydının kimlik zarfı — her etiketleyici türünde AYNI.

    `source_run_id` zorunlu: `task_id + arm` üç tekrar × iki model altında altı
    farklı koşuya karşılık gelir, yani tek başına kimlik değildir.

    model ve arm burada PROVENANCE olarak kalır (analizde gerekli); kör
    payload'a ve judge prompt'una konulmaz (§9.4, §9.6).
    """
    return {
        "mast_schema_version": MAST_SCHEMA_VERSION,
        "source_run_id": record["run_id"],
        "experiment": experiment,
        "source_model": record["model"],
        "task_set": record["task_set"],
        "task_id": record["task_id"],
        "arm": record["arm"],
        "repeat": record["repeat"],
        "error_class": record.get("error_class"),
        "interaction_type": interaction_type,
        "evidence_sha256": evidence_sha256,
        # Prompt/taksonomi sürümü kimliğin PARÇASIDIR: prompt değişirse eski
        # etiketler yeni prompt'un ürünü sayılamaz (resume onları atlamamalı).
        "mast_prompt_hash": prompt_hash,
    }


def make_judge_record(record: dict, *, experiment: str, evidence_sha256: str,
                      judge_model: str, judge_attempt: int, judge_status: str,
                      prompt_hash: str | None = None, interaction_type: str | None = None,
                      label: MastLabel | None = None,
                      judge_error: str | None = None,
                      judge_raw: str | None = None) -> dict:
    """Tek judge'ın tek kayıt için çıktısı. Başarısız denemeler de SAKLANIR."""
    out = {
        "ts": datetime.now(timezone.utc).isoformat(),
        **label_provenance(record, experiment=experiment, evidence_sha256=evidence_sha256,
                           prompt_hash=prompt_hash, interaction_type=interaction_type),
        "judge_model": judge_model,
        "judge_attempt": judge_attempt,
        "judge_status": judge_status,
    }
    if label is not None:
        out.update(label.model_dump())
    if judge_error:
        out["judge_error"] = judge_error
    # Ham yanıt BAŞARILI durumda da saklanır: etiket sonradan tartışmaya
    # açılırsa modelin ne dediğinin tek kanıtı budur (yeniden üretilemez).
    if judge_raw is not None:
        out["judge_raw"] = judge_raw[:4000]
        out["judge_raw_sha256"] = hashlib.sha256(judge_raw.encode("utf-8")).hexdigest()
    return out


def validate_expected_judges(expected_judges) -> tuple[str, str, str]:
    """Beklenen judge kümesinin YAPISAL geçerliliği (§9.1).

    Rol bölümlemesi ve panel kararı bu kümeye dayandığı için, geçersiz bir küme
    ilk API çağrısından ÖNCE durdurulmalıdır: iki judge'la "leave-self-out"
    yapılamaz (bir self çıkınca tek dış judge kalır, konsensüs kavramı çöker),
    dört judge'la ise hangi ikisinin karar verdiği belirsizleşir.
    """
    expected = tuple(expected_judges)
    if len(expected) != 3:
        raise MastPipelineError(
            f"beklenen judge kümesi tam olarak 3 model içermeli, alınan {len(expected)}: "
            f"{list(expected)}")
    if len(set(expected)) != 3:
        raise MastPipelineError(f"beklenen judge kümesi yinelenen model içeriyor: {list(expected)}")
    if MODEL_ADJUDICATOR in expected:
        raise MastPipelineError(
            f"adjudicator ({MODEL_ADJUDICATOR}) judge panelinde bulunamaz (§9.2)")
    return expected


def validate_frozen_panel(expected_judges, adjudicator: str | None = None) -> tuple[str, str, str]:
    """ÜRETİM kapısı: canlı MAST yolunda kadro DONDURULMUŞ kümeyle birebir aynı.

    `validate_expected_judges()` bilinçli olarak YAPISAL kalır (üç farklı model,
    adjudicator panelde değil) — ters örnekleri ve şema testlerini kurabilmek
    için gerekli. Ama yapısal geçerlilik canlı yol için YETMEZ: programatik
    kullanımda `(MODEL_MAIN, MODEL_SECONDARY, "openrouter/rogue/judge")` üçlüsü
    yapısal olarak kusursuzdur ve ön-kayıtta yeri olmayan bir modele ÜCRETLİ
    çağrı yaptırırdı. Manifest bunu ancak çağrılar bittikten sonra yakalardı.

    SIRA da kimliğin parçasıdır: judge sırası dış judge'ların sırasını
    (`judge_role_partition`) ve Parça 4'te adjudicator'a giden sunum sırasını
    belirler. Doğru modellerin farklı sırası bu yüzden reddedilir.
    """
    expected = validate_expected_judges(expected_judges)
    donmus = tuple(MODEL_JUDGES)
    if expected != donmus:
        neden = ("aynı modeller FARKLI SIRADA" if set(expected) == set(donmus)
                 else "ön-kayıtta olmayan judge")
        raise MastPipelineError(
            f"dondurulmuş judge kadrosu dışında panel ({neden}): {list(expected)}; "
            f"beklenen {list(donmus)} (config.MODEL_JUDGES)")
    if adjudicator is not None and adjudicator != MODEL_ADJUDICATOR:
        raise MastPipelineError(
            f"dondurulmuş adjudicator dışında model: {adjudicator!r}; "
            f"beklenen {MODEL_ADJUDICATOR!r} (config.MODEL_ADJUDICATOR)")
    return expected


def judge_role_partition(source_model: str, expected_judges) -> tuple[str, tuple[str, str]]:
    """(self_judge_model, external_judges) — leave-self-out rol ayrımı (§9.1).

    Rol, etiketin İÇERİĞİNDEN değil kaynak kaydın provenance'ından türer: bir
    kaydı ÜRETEN model, kendi çıktısının hata etiketine oy veremez. Aynı üç
    etiket, kaynak model değiştiğinde farklı bir karar bölümlemesi verir.

    `source_model` üretici allowlist'inde (config.MODEL_PRODUCERS) olmalıdır:
    judge/adjudicator modelleri held-out veri üretmediği için "kendi kaydı"
    kavramı onlar için tanımsızdır. Bölümleme Parça 4'te Grok prompt'una giden
    iki dış judge'ı seçmek için de kullanılacaktır — aynı kural iki yerde ayrı
    yazılırsa karar katmanı ile prompt katmanı sessizce ayrışabilir.

    Dış judge sırası `expected_judges` sırasından türer (deterministik).
    """
    expected = validate_expected_judges(expected_judges)
    if source_model not in MODEL_PRODUCERS:
        raise MastPipelineError(
            f"kaynak model üretici değil: {source_model!r}; beklenen {list(MODEL_PRODUCERS)} "
            "(judge/adjudicator modelleri held-out veri üretmez, §4)")
    if expected.count(source_model) != 1:
        raise MastPipelineError(
            f"kaynak model beklenen judge kümesinde tam bir kez bulunmalı: "
            f"{source_model!r} / {list(expected)}")
    external = tuple(m for m in expected if m != source_model)
    if len(external) != 2 or source_model in external:
        raise MastPipelineError(
            f"dış judge kümesi tam iki farklı model olmalı: {list(external)}")
    return source_model, external


# Bir panel oyunun alabileceği DEĞERLER. Taksonomi kodu ya da "yetersiz bağlam"
# sentineli; başka hiçbir dize (ör. "9.9", "n/a") oy olarak sayılamaz.
PANEL_VOTE_VALUES = VALID_CODES | {INSUFFICIENT_SENTINEL}


def derive_triple_agreement(votes: list[str], missing_judges) -> tuple[str, str | None]:
    """Tam üçlü panelin TANISAL anlaşma durumu (tek türetme noktası).

    `panel_verdict()` ile `MastPanelVerdict` bu fonksiyonu PAYLAŞIR: kayıt üreten
    kod ile onu doğrulayan sözleşme ayrı ayrı yazılsaydı, ikisi zamanla ayrışır
    ve doğrulama gerçekte hiçbir şeyi denetlemez hale gelirdi.
    """
    if list(missing_judges):
        return "incomplete", None
    if len(set(votes)) == 1:
        return "unanimous", votes[0]
    counts = {k: votes.count(k) for k in set(votes)}
    winners = sorted(k for k, n in counts.items() if n >= 2)
    return ("majority", winners[0]) if winners else ("split", None)


def derive_external_decision(votes_by_model: dict[str, str],
                             external_judges) -> tuple[str, str | None]:
    """İki DIŞ judge'ın karar durumu (leave-self-out). Aynı paylaşım gerekçesi."""
    dis = [votes_by_model[m] for m in external_judges if m in votes_by_model]
    if len(dis) < 2:
        # Bir dış judge eksikse konsensüs de split de İDDİA EDİLEMEZ; eksik bir
        # paneli "anlaşmazlık" saymak ölçülen anlaşmazlık oranını şişirirdi.
        return "incomplete", None
    if dis[0] == dis[1]:
        return "consensus", dis[0]
    return "split", None


class MastPanelVerdict(BaseModel):
    """Panel kararının SÖZLEŞMESİ — çapraz alan invariantları burada zorlanır.

    Dağınık bir dict üretip tutarlılığı yalnız testlere bırakmak, "adjudicator
    neden çalıştı" sorusunun cevabını kayıt üreten koda gömerdi. Alanların
    birbiriyle çelişemeyeceği tek noktada güvence altına alınır; `panel_verdict()`
    bu modelin `model_dump()`'ını döndürür (tüketiciler için dict arayüzü aynı).

    Alan bazlı doğrulama YETMEZ: aşağıdaki kayıt her alanında tek başına
    geçerliyken graf olarak imkânsızdır — üç oy da aynıyken `majority`, oy
    kullanmayan bir judge'ın etiketi sayılmış (`valid_judges=3` ama iki model) ve
    hiçbir judge'ın vermediği bir etiket çoğunluk ilan edilmiş:

        valid_judges=3, judge_models=[a, b], primary_modes=["1.1","1.1","1.1"],
        agreement_level="majority", majority_label="9.9", judge_disagreement=False

    Bu yüzden türetilebilir alanların TAMAMI oy dağılımından yeniden hesaplanır
    ve saklananla karşılaştırılır. `judge_models` ile `primary_modes` beklenen
    judge SIRASINDA ve indeks olarak EŞLEŞİK tutulur — aksi halde hangi oyun
    kimden geldiği kaybolur ve dış karar zaten yeniden türetilemezdi.

    `extra="forbid"`: alan adı yanlış yazılmış bir karar alanı (ör.
    `adjudicator_requiered`) sessizce yutulup varsayılanla doldurulamaz.
    """

    model_config = ConfigDict(extra="forbid")

    # --- Tam üçlü panel: TANISAL ---
    expected_judges: list[str]
    valid_judges: int
    missing_judges: list[str]
    judge_models: list[str]
    primary_modes: list[str]
    agreement_level: str
    majority_label: str | None
    judge_disagreement: bool
    panel_complete: bool

    # --- Leave-self-out: KARAR ---
    decision_rule_version: str
    self_judge_model: str
    external_judges: list[str]
    external_agreement_level: str
    external_consensus_label: str | None
    self_matches_external: bool | None
    adjudicator_required: bool

    # --- Girdi hash'leri: tanısal tazelik ile KARAR tazeliği ayrı ---
    panel_hash_version: str
    full_panel_input_sha256: str | None
    decision_input_sha256: str | None

    @model_validator(mode="after")
    def _capraz_invariantlar(self):
        if self.agreement_level not in AGREEMENT_LEVELS:
            raise ValueError(f"geçersiz agreement_level: {self.agreement_level!r}")
        if self.external_agreement_level not in EXTERNAL_AGREEMENT_LEVELS:
            raise ValueError(
                f"geçersiz external_agreement_level: {self.external_agreement_level!r}")
        if self.decision_rule_version != MAST_DECISION_RULE_VERSION:
            raise ValueError(
                f"karar kuralı sürümü uyuşmuyor: {self.decision_rule_version!r} != "
                f"{MAST_DECISION_RULE_VERSION!r}")

        # --- Panel BİLEŞİMİ: beklenen üçlü, eksikler, oy kullananlar ---
        expected = list(self.expected_judges)
        if len(expected) != 3 or len(set(expected)) != 3:
            raise ValueError(
                f"expected_judges tam üç FARKLI model olmalı: {expected}")
        # SIRA da kimliğin parçasıdır (küme eşitliği YETMEZ): dış judge sırası,
        # eksik/oy kullanan listelerin kanonik sırası ve Parça 3'te
        # `decision_input_sha256` bu sıradan türeyecek. Aynı mantıksal panel,
        # yalnız liste sırası farklı diye farklı hash üretemez.
        if expected != list(MODEL_JUDGES):
            raise ValueError(
                f"expected_judges dondurulmuş kadroyla AYNI SIRADA olmalı: {expected} != "
                f"{list(MODEL_JUDGES)}")
        if len(set(self.missing_judges)) != len(self.missing_judges):
            raise ValueError(f"missing_judges yinelenen model içeriyor: {self.missing_judges}")
        if not set(self.missing_judges) <= set(expected):
            raise ValueError(
                f"missing_judges beklenen kümenin alt kümesi olmalı: {self.missing_judges}")
        kanonik_eksik = [m for m in expected if m in set(self.missing_judges)]
        if self.missing_judges != kanonik_eksik:
            raise ValueError(
                f"missing_judges dondurulmuş panel sırasında olmalı: "
                f"{self.missing_judges} != {kanonik_eksik}")
        oy_kullananlar = [m for m in expected if m not in set(self.missing_judges)]
        if self.judge_models != oy_kullananlar:
            raise ValueError(
                f"judge_models, beklenen sırada (beklenen - eksik) olmalı: "
                f"{self.judge_models} != {oy_kullananlar}")
        if not (self.valid_judges == len(self.judge_models) == len(self.primary_modes)):
            raise ValueError(
                f"valid_judges / judge_models / primary_modes uzunlukları eşleşmeli: "
                f"{self.valid_judges} / {len(self.judge_models)} / {len(self.primary_modes)}")

        # --- Oy DEĞERLERİ taksonomiye ait mi ---
        gecersiz = [k for k in self.primary_modes if k not in PANEL_VOTE_VALUES]
        if gecersiz:
            raise ValueError(f"panel oyu taksonomi kodu ya da sentinel olmalı: {gecersiz}")
        for alan, deger in (("majority_label", self.majority_label),
                            ("external_consensus_label", self.external_consensus_label)):
            if deger is not None and deger not in PANEL_VOTE_VALUES:
                raise ValueError(f"{alan} taksonomi kodu ya da sentinel olmalı: {deger!r}")

        # --- Rol bölümlemesi tutarlı mı ---
        # Self-judge yalnız bir ÜRETİCİ olabilir: "kendi kaydı" kavramı held-out
        # veri üretmeyen modeller (MiniMax judge, Grok adjudicator) için tanımsız
        # olduğu gibi, MiniMax'ı self ilan etmek iki üreticiyi birden dış judge
        # yapıp kaydı üreten modelin oyunu karara geri sokardı.
        if self.self_judge_model not in MODEL_PRODUCERS:
            raise ValueError(
                f"self_judge_model üretici olmalı: {self.self_judge_model!r}; "
                f"beklenen {list(MODEL_PRODUCERS)}")
        # Dış judge listesi beklenen sıradan TÜRETİLİR, ayrıca saklanmaz: ters
        # çevrilmiş bir liste aynı mantıksal paneli farklı bir kanonik girdiye
        # dönüştürürdü — `decision_input_sha256` tam olarak bu sıraya bağlıdır.
        kanonik_dis = [m for m in expected if m != self.self_judge_model]
        if self.external_judges != kanonik_dis:
            raise ValueError(
                f"external_judges beklenen sırada (beklenen - self) olmalı: "
                f"{self.external_judges} != {kanonik_dis}")

        # --- TÜRETİLEBİLİR alanlar oy dağılımıyla yeniden hesaplanır ---
        oylar = dict(zip(self.judge_models, self.primary_modes))
        seviye, cogunluk = derive_triple_agreement(self.primary_modes, self.missing_judges)
        if (self.agreement_level, self.majority_label) != (seviye, cogunluk):
            raise ValueError(
                f"tanısal anlaşma alanları oy dağılımıyla uyuşmuyor: saklanan "
                f"({self.agreement_level!r}, {self.majority_label!r}) != "
                f"türetilen ({seviye!r}, {cogunluk!r})")
        if self.judge_disagreement != (seviye != "unanimous"):
            raise ValueError("judge_disagreement, agreement_level'dan türetilmeli")
        if self.panel_complete != (not self.missing_judges):
            raise ValueError("panel_complete, beklenen üç judge'ın tamamlanmasına eşit olmalı")

        dis_seviye, dis_etiket = derive_external_decision(oylar, self.external_judges)
        if (self.external_agreement_level,
                self.external_consensus_label) != (dis_seviye, dis_etiket):
            raise ValueError(
                f"dış karar alanları dış oylarla uyuşmuyor: saklanan "
                f"({self.external_agreement_level!r}, {self.external_consensus_label!r}) "
                f"!= türetilen ({dis_seviye!r}, {dis_etiket!r})")
        # Eksik self + iki dış oy: dış konsensüs KORUNUR (karar verilebilir),
        # ama panel tamamlanmamıştır ve self karşılaştırması tanımsızdır.
        beklenen_uyum = (None if dis_etiket is None or self.self_judge_model not in oylar
                         else oylar[self.self_judge_model] == dis_etiket)
        if self.self_matches_external is not beklenen_uyum:
            raise ValueError(
                f"self_matches_external türetilenle uyuşmuyor: "
                f"{self.self_matches_external!r} != {beklenen_uyum!r}")

        # KARAR: yalnız iki dış judge'ın ayrışması adjudicator gerektirir. Üçlü
        # majority ya da self-judge bu değeri DEĞİŞTİREMEZ.
        if self.adjudicator_required != (dis_seviye == "split"):
            raise ValueError(
                "adjudicator_required yalnız external split'te True olabilir "
                f"(external={self.external_agreement_level!r})")

        # --- Girdi hash'leri ---
        if self.panel_hash_version != MAST_PANEL_HASH_VERSION:
            raise ValueError(
                f"panel hash sürümü uyuşmuyor: {self.panel_hash_version!r} != "
                f"{MAST_PANEL_HASH_VERSION!r}")
        for alan, deger in (("full_panel_input_sha256", self.full_panel_input_sha256),
                            ("decision_input_sha256", self.decision_input_sha256)):
            if deger is not None and not _HEX64.match(deger):
                raise ValueError(f"{alan} 64 karakterlik küçük harf hex olmalı: {deger!r}")
        # Full hash YALNIZ tam üçlü panelde tanımlıdır: yarım bir kümenin hash'i
        # "tamamlanmış panel" izlenimi verirdi.
        if (self.full_panel_input_sha256 is not None) != self.panel_complete:
            raise ValueError(
                "full_panel_input_sha256 yalnız tam üçlü panelde bulunmalı "
                f"(panel_complete={self.panel_complete})")
        # Decision hash, SELF eksik olsa bile iki dış etiket varsa tanımlıdır:
        # karar girdisi tamamdır, eksik olan yalnız tanısal alandır.
        dis_tam = not (set(self.external_judges) & set(self.missing_judges))
        if (self.decision_input_sha256 is not None) != dis_tam:
            raise ValueError(
                "decision_input_sha256 yalnız İKİ dış etiket mevcutken bulunmalı "
                f"(eksik={self.missing_judges})")
        return self


def panel_verdict(judge_records: list[dict], expected_judges,
                  interaction_type: str | None = None, *, source_model: str) -> dict:
    """Üç bağımsız etiketten anlaşma durumu (§9.1-§9.2).

    `expected_judges` bir SAYI DEĞİL, beklenen judge MODEL KÜMESİDİR. Sayı
    saymak üç ayrı bozuk paneli "unanimous" sayardı — üçü de canlı olarak
    üretilebiliyordu:

        [a, a, b]     aynı judge iki kez, beklenen c hiç yok
        [a, b, x]     beklenmeyen judge x, beklenen c hiç yok
        [a, b, c, c]  dört başarılı kayıt (birinin oyu iki kez sayılır)

    Bu yüzden her beklenen judge için TAM BİR başarılı etiket aranır:
      - yinelenen başarılı etiket  -> fail-fast (oy çift sayılır)
      - beklenmeyen judge          -> fail-fast (panel bileşimi değişmiş)
      - beklenen judge eksik       -> incomplete (normal, resume ile tamamlanır)

    Saklanan "başarılı" etiketler ayrıca MastLabel şemasına YENİDEN doğrulanır:
    elle düzenlenmiş ya da eski şemayla yazılmış bir kayıt, judge_status="ok"
    dediği için sorgusuz kabul edilirdi.

    KARAR KURALI (leave-self-out, §9.1): üç etiket de tanısal olarak saklanır,
    fakat kaydı ÜRETEN modelin kendi etiketi karara oy VERMEZ. Karar iki dış
    judge'a dayanır: aynı etiketi verirlerse `external_consensus_label`,
    ayrışırlarsa `adjudicator_required=True`.

    `agreement_level`/`majority_label` yalnız TANISALDIR ve kararı belirleyemez.
    Gerekçe (yöntemsel karşı-örnek): aynı 2/1 majority deseni iki farklı karar
    durumu saklar — çoğunluğu iki dış judge oluşturuyorsa dış konsensüs vardır,
    çoğunluğu self + bir dış judge oluşturuyorsa dış taraf SPLIT'tir. Yani üçlü
    çoğunluk, self-judge etkisini karar katmanından çıkaramaz.

    `source_model` ZORUNLU ve keyword-only: panel kayıtlarından tahmin
    EDİLMEZ — kaynak kaydın provenance'ından açıkça verilir.
    """
    self_judge, external_judges = judge_role_partition(source_model, expected_judges)
    expected = tuple(expected_judges)
    valid = [r for r in judge_records if r.get("judge_status") == JUDGE_STATUS_OK]

    by_model: dict[str, list[dict]] = {}
    for r in valid:
        by_model.setdefault(r.get("judge_model"), []).append(r)

    yinelenen = {m: len(rs) for m, rs in by_model.items() if len(rs) > 1}
    if yinelenen:
        raise MastPipelineError(
            f"aynı judge için birden fazla başarılı etiket (oy çift sayılır): {yinelenen}")
    beklenmeyen = sorted(set(by_model) - set(expected))
    if beklenmeyen:
        raise MastPipelineError(
            f"panelde beklenmeyen judge: {beklenmeyen}; beklenen {list(expected)}")

    keys = []
    key_by_model: dict[str, str | None] = {}
    for model in expected:
        r = by_model.get(model)
        if not r:
            continue
        record = r[0]
        problems = validate_stored_label(record, interaction_type)
        if problems:
            raise MastPipelineError(
                f"judge_status='ok' ama etiket şemaya uymuyor "
                f"({record.get('source_run_id')}, {model}): {problems}")
        key = stored_comparison_key(record)
        keys.append(key)
        key_by_model[model] = key

    # Eksik/oy kullanan listeler DONDURULMUŞ panel sırasında tutulur (alfabetik
    # değil): kanonik sıra, karar girdisinin hash'ini sıralamadan bağımsız kılar.
    eksik = [m for m in expected if m not in by_model]
    # --- Tanısal: tam üçlü panel ---
    level, majority = derive_triple_agreement(keys, eksik)
    # --- Karar: yalnız iki DIŞ judge ---
    dis_seviye, dis_etiket = derive_external_decision(key_by_model, external_judges)

    self_anahtar = key_by_model.get(self_judge)
    self_uyum = (None if dis_etiket is None or self_judge not in key_by_model
                 else self_anahtar == dis_etiket)

    # Hash'ler kararın ÜRETİLDİĞİ yerde, oy kullanan kayıtların TAM AYNISINDAN
    # hesaplanır. Panel dışında hesaplansaydı "hangi etiketler oy kullandı" ile
    # "hangi etiketler hash'lendi" kümeleri sessizce ayrışabilirdi.
    oy_kayitlari = [by_model[m][0] for m in expected if m in by_model]
    tam_hash = (full_panel_input_digest(oy_kayitlari, expected) if not eksik else None)
    dis_hash = (decision_input_digest(oy_kayitlari, expected, source_model=self_judge)
                if all(m in by_model for m in external_judges) else None)

    verdict = MastPanelVerdict(
        expected_judges=list(expected),
        valid_judges=len(keys),
        missing_judges=eksik,
        # `keys` ile İNDEKS EŞLEŞİK: ikisi de beklenen judge sırasındadır, yani
        # hangi oyun kimden geldiği kayıttan yeniden kurulabilir.
        judge_models=[m for m in expected if m in key_by_model],
        primary_modes=keys,
        agreement_level=level,
        majority_label=majority,
        judge_disagreement=level != "unanimous",
        panel_complete=level != "incomplete",
        decision_rule_version=MAST_DECISION_RULE_VERSION,
        self_judge_model=self_judge,
        external_judges=list(external_judges),
        external_agreement_level=dis_seviye,
        external_consensus_label=dis_etiket,
        self_matches_external=self_uyum,
        # Eksik panelde bu alan True olsa bile panel_blockers() adjudication'ı
        # ilk API çağrısından ÖNCE durdurur (§9.2).
        adjudicator_required=dis_seviye == "split",
        panel_hash_version=MAST_PANEL_HASH_VERSION,
        full_panel_input_sha256=tam_hash,
        decision_input_sha256=dis_hash,
    )
    return verdict.model_dump()


ADJUDICATOR_STATUS_OK = "ok"
ADJUDICATOR_STATUS_ERROR = "error"
ADJUDICATOR_STATUSES = (ADJUDICATOR_STATUS_OK, ADJUDICATOR_STATUS_ERROR)
ADJUDICATED_LABEL_KEYS = ("adjudicated_primary_mode", "adjudicated_secondary_modes",
                          "adjudicated_confidence", "adjudicated_rationale",
                          "adjudicated_insufficient_context")


class MastAdjudication(BaseModel):
    """External-only adjudication kararının SÖZLEŞMESİ (§9.2, MAST 3.2).

    Bu alanların tutarlılığı dağınık dict kontrollerine bırakılamaz: "Grok neyi
    gördü ve neye karar verdi" sorusunun cevabı kaydın KENDİSİNDEN okunabilmeli.
    Zorlanan çekirdek invariant, leave-self-out'un adjudication katmanındaki
    karşılığıdır — kaydı ÜRETEN modelin etiketi karar girdisine giremez:

        self_judge_model == source_model
        external_judges  == MODEL_JUDGES - self
        reviewed_judges  == external_judges          (Grok tam olarak bunları gördü)
        annotator_assignment anahtarları {A, B}, değerleri tam external küme

    `external_agreement_level` yalnız `split` olabilir: dış konsensüs varken
    adjudicator çağrılmamalıydı, çağrıldıysa kayıt zaten bozuktur.

    Başarısız denemeler de SAKLANIR (`adjudicator_status="error"`); o durumda
    `adjudicated_*` alanları BULUNMAZ — hata kaydının yarım bir etiket taşıması
    "karar verildi" izlenimi verirdi. Provenance ve karar hash'i her iki durumda
    da korunur.
    """

    model_config = ConfigDict(extra="forbid")

    # Rol/karar zarfı
    decision_rule_version: str
    source_model: str
    self_judge_model: str
    external_judges: list[str]
    external_agreement_level: str
    reviewed_judges: list[str]
    annotator_assignment: dict[str, str]
    decision_input_sha256: str
    interaction_type: Literal["single_agent", "multi_agent"]

    # Adjudicator kimliği ve durumu
    adjudicator_model: str
    adjudicator_attempt: StrictInt
    adjudicator_status: str

    # Karar (yalnız status="ok")
    adjudicated_primary_mode: str | None = None
    adjudicated_secondary_modes: list[str] | None = None
    adjudicated_confidence: str | None = None
    adjudicated_rationale: str | None = None
    adjudicated_insufficient_context: bool | None = None

    @model_validator(mode="after")
    def _capraz_invariantlar(self):
        if self.decision_rule_version != MAST_DECISION_RULE_VERSION:
            raise ValueError(
                f"karar kuralı sürümü uyuşmuyor: {self.decision_rule_version!r}")
        # Sessiz fallback yasak: başka bir modelin verdiği karar "Grok kararı"
        # olarak kaydedilirse panel-dışı adjudicator iddiası çöker.
        if self.adjudicator_model != MODEL_ADJUDICATOR:
            raise ValueError(
                f"adjudicator dondurulmuş model olmalı: {self.adjudicator_model!r} != "
                f"{MODEL_ADJUDICATOR!r}")
        if self.self_judge_model != self.source_model:
            raise ValueError(
                f"self_judge_model kaynak modelle aynı olmalı: "
                f"{self.self_judge_model!r} != {self.source_model!r}")
        if self.source_model not in MODEL_PRODUCERS:
            raise ValueError(f"kaynak model üretici olmalı: {self.source_model!r}")

        kanonik_dis = [m for m in MODEL_JUDGES if m != self.self_judge_model]
        if self.external_judges != kanonik_dis:
            raise ValueError(
                f"external_judges kanonik dış ikili olmalı: {self.external_judges} != "
                f"{kanonik_dis}")
        # Grok'un GERÇEKTEN gördüğü küme, karar kuralının seçtiği kümeyle aynı
        # olmalı; ayrıştıklarında kayıt "iki dış judge'a dayanıyor" derken başka
        # bir girdiye dayanmış olurdu.
        if self.reviewed_judges != self.external_judges:
            raise ValueError(
                f"reviewed_judges dış ikiliyle aynı olmalı: {self.reviewed_judges} != "
                f"{self.external_judges}")
        if self.external_agreement_level != "split":
            raise ValueError(
                "adjudication yalnız external split'te üretilebilir, alınan "
                f"{self.external_agreement_level!r}")

        if sorted(self.annotator_assignment) != ["A", "B"]:
            raise ValueError(
                f"annotator_assignment yalnız A/B taşımalı: "
                f"{sorted(self.annotator_assignment)}")
        atananlar = list(self.annotator_assignment.values())
        if sorted(atananlar) != sorted(self.external_judges):
            raise ValueError(
                f"annotator_assignment tam external kümeyi taşımalı: {atananlar}")

        if not _HEX64.match(self.decision_input_sha256):
            raise ValueError(
                f"decision_input_sha256 64 karakterlik küçük harf hex olmalı: "
                f"{self.decision_input_sha256!r}")
        if self.adjudicator_attempt < 1:
            raise ValueError(f"adjudicator_attempt >= 1 olmalı: {self.adjudicator_attempt}")
        if self.adjudicator_status not in ADJUDICATOR_STATUSES:
            raise ValueError(f"geçersiz adjudicator_status: {self.adjudicator_status!r}")

        etiket_alanlari = {k: getattr(self, k) for k in ADJUDICATED_LABEL_KEYS}
        if self.adjudicator_status == ADJUDICATOR_STATUS_ERROR:
            dolu = {k: v for k, v in etiket_alanlari.items() if v is not None}
            if dolu:
                raise ValueError(
                    f"hata kaydı yarım karar taşıyamaz: {sorted(dolu)}")
            return self

        # status="ok": karar, judge etiketleriyle AYNI şemaya doğrulanır.
        payload = {k.removeprefix("adjudicated_"): v for k, v in etiket_alanlari.items()}
        try:
            label = MastLabel.model_validate(payload)
        except Exception as e:
            raise ValueError(f"adjudication kararı MastLabel şemasına uymuyor: {e}") from e
        problems = interaction_problems(label, self.interaction_type)
        if problems:
            raise ValueError("; ".join(problems))
        return self


def validate_stored_label(record: dict, interaction_type: str | None = None) -> list[str]:
    """Saklanmış bir etiket kaydını şemaya (ve varsa etkileşim tipine) doğrular."""
    payload = {k: record.get(k) for k in
               ("primary_mode", "secondary_modes", "confidence", "rationale",
                "insufficient_context") if k in record}
    payload.setdefault("secondary_modes", [])
    payload.setdefault("insufficient_context", False)
    try:
        label = MastLabel.model_validate(payload)
    except Exception as e:
        return [str(e)]
    return interaction_problems(label, interaction_type) if interaction_type else []


def stored_comparison_key(judge_record: dict) -> str:
    """Kayıttan panel karşılaştırma anahtarı (MastLabel'ınkiyle aynı semantik)."""
    if judge_record.get("insufficient_context"):
        return INSUFFICIENT_SENTINEL
    return judge_record.get("primary_mode")
