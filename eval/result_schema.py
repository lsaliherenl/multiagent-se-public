"""Sonuç kaydı sözleşmesi — runner ile analiz katmanı arasındaki TEK kaynak.

EXPERIMENT_PROTOCOL.md §7. Bu modül alan adlarını, zorunlulukları ve bütünlük
kontrollerini tanımlar. Hem `eval/runner.py` (yazarken doğrular) hem analiz
katmanı (okurken doğrular) BURAYI kullanır — alan adları iki yerde elle
tekrarlanmaz. Analiz fixture'ları da buradaki fabrikadan üretilir, böylece
"testler geçiyor ama gerçek kayıt farklı" durumu yapısal olarak imkânsızlaşır.

Kavramsal ayrımlar (karıştırılırsa analiz sessizce yanlış olur):

- **Kimlik anahtarı** (`IDENTITY_FIELDS`): bir ölçümü evrensel olarak
  belirleyen alanlar. Deney adı dahildir.
- **Resume anahtarı** (`RESUME_KEY_FIELDS`): kesintiden devam ederken "bu koşu
  yapıldı mı" sorusunun anahtarı. MODEL DAHİLDİR — aynı deney adı altında
  farklı modelle koşulmuş bir kayıt, yeni modelin koşusunu atlatmamalı.
- **run_error**: LLM/altyapı hatası. Kayıt SAKLANIR (şeffaflık) ama tamamlanmış
  ölçüm SAYILMAZ; yeniden denenir ve analizde başarı/başarısızlık olarak
  sayılmaz (§7).
"""

from datetime import datetime, timezone
from uuid import uuid4

from config import (
    ALL_ARMS,
    ARM_CONTRACT,
    ARM_STRUCTURED,
    LLM_CALL_SCHEMA_VERSION,
    MAX_PLANNER_ATTEMPTS,
    RESULT_SCHEMA_VERSION,
    STRUCTURED_MODES,
)

RUN_ERROR_STATUS = "run_error"

# Bir ölçümü evrensel olarak belirleyen alanlar.
IDENTITY_FIELDS = ("experiment", "model", "arm", "task_id", "repeat")

# Kesintiden devam anahtarı. experiment DAHİL DEĞİL (zaten deney dizinine
# göre okunuyor); model DAHİL çünkü aynı dizinde farklı model koşulursa
# eski kayıtlar yeni modelin koşusunu sessizce atlatırdı.
RESUME_KEY_FIELDS = ("model", "arm", "task_id", "repeat")

# Her kayıtta (run_error dahil) bulunması gereken alanlar.
REQUIRED_FIELDS = (
    "schema_version", "ts", "experiment", "model", "task_set",
    "arm", "task_id", "repeat", "run_id", "arm_position", "status",
)

# Başarılı (run_error olmayan) kayıtlarda ayrıca bulunması gerekenler.
OUTCOME_FIELDS = (
    "base_status", "base_pass", "base_error_class", "base_duration_s",
    "plus_status", "plus_pass", "plus_error_class", "plus_duration_s",
    "base_plus_available", "error_class",
)

# Kola özgü zorunluluklar: eksilirse ilgili araştırma sorusu ölçülemez hale
# gelir, bu yüzden şema düzeyinde zorunlu tutulur.
ARM_REQUIRED_FIELDS = {
    **{arm: ("handoff_parse_ok", "attempt_count") for arm in STRUCTURED_MODES},
    ARM_CONTRACT: ("handoff_parse_ok", "attempt_count", "handoff_validation"),
}


def resume_key(record: dict) -> tuple:
    return tuple(record[f] for f in RESUME_KEY_FIELDS)


def identity_key(record: dict) -> tuple:
    return tuple(record[f] for f in IDENTITY_FIELDS)


def is_run_error(record: dict) -> bool:
    return record.get("status") == RUN_ERROR_STATUS


def validate_record(record: dict) -> list[str]:
    """Sözleşme ihlallerini döndürür (boş liste = geçerli).

    Exception ATMAZ: çağıran taraf ihlalleri toplu raporlayabilsin diye.
    """
    problems = [f"eksik alan: {f}" for f in REQUIRED_FIELDS if f not in record]
    if problems:
        return problems

    if record["schema_version"] != RESULT_SCHEMA_VERSION:
        problems.append(
            f"şema sürümü uyuşmuyor: {record['schema_version']!r} != {RESULT_SCHEMA_VERSION!r}")
    if record["arm"] not in ALL_ARMS:
        problems.append(f"bilinmeyen kol: {record['arm']!r}")
    if not isinstance(record["repeat"], int) or record["repeat"] < 0:
        problems.append(f"geçersiz repeat: {record['repeat']!r}")

    if is_run_error(record):
        if not record.get("error"):
            problems.append("run_error kaydında 'error' alanı yok")
        return problems

    problems += [f"eksik sonuç alanı: {f}" for f in OUTCOME_FIELDS if f not in record]
    for field in ("base_pass", "plus_pass"):
        if field in record and not isinstance(record[field], bool):
            problems.append(f"{field} bool değil: {record[field]!r}")
    # Plus, base'i kapsar: base düşmüşken plus geçemez. Bu tutarsızlık
    # sessizce geçerse birincil metrik güvenilmez olur.
    if record.get("base_pass") is False and record.get("plus_pass") is True:
        problems.append("tutarsız sonuç: base_pass=False iken plus_pass=True")

    for field in ARM_REQUIRED_FIELDS.get(record["arm"], ()):
        if field not in record:
            problems.append(f"{record['arm']} kolunda eksik alan: {field}")
    problems += _handoff_problems(record)
    return problems


def _is_int(value) -> bool:
    """bool, int'in ALT SINIFI — attempt_count=True'yu geçerli saymayalım."""
    return isinstance(value, int) and not isinstance(value, bool)


def _handoff_problems(record: dict) -> list[str]:
    """RQ2/RQ3 alanlarının TİPİ ve ARALIĞI.

    Varlık kontrolü tek başına yetmez: `attempt_count=None` ya da
    `handoff_validation={"valid": "yes"}` gibi bir değer alan listesinden geçer
    ama uyum metriklerini sessizce çöpe çevirir (None > 1 karşılaştırması
    TypeError, "yes" her zaman truthy). Aralık kontrolü ayrıca graf mantığının
    kaydına da bir tutarlılık testidir:

    - structured_no_validation'da doğrulama kapısı ve retry kenarı YOKTUR, yani
      attempt_count her zaman 1 olmalıdır. 1'den büyük bir değer, kolun sessizce
      contract gibi davrandığı anlamına gelir — iç geçerlilik ihlali.
    - contract'ta 1..MAX_PLANNER_ATTEMPTS dışındaki değer, retry kapısının
      sınırının aşıldığını gösterir.
    """
    arm = record["arm"]
    if arm not in STRUCTURED_MODES:
        return []
    problems = []

    if "handoff_parse_ok" in record and not isinstance(record["handoff_parse_ok"], bool):
        problems.append(f"handoff_parse_ok bool değil: {record['handoff_parse_ok']!r}")

    attempts = record.get("attempt_count")
    if "attempt_count" in record:
        if not _is_int(attempts):
            problems.append(f"attempt_count int değil: {attempts!r}")
        elif arm == ARM_STRUCTURED and attempts != 1:
            problems.append(
                f"structured_no_validation kolunda attempt_count={attempts} — "
                "bu kolda retry kenarı yok, 1 olmalı")
        elif arm == ARM_CONTRACT and not (1 <= attempts <= MAX_PLANNER_ATTEMPTS):
            problems.append(
                f"contract kolunda attempt_count={attempts}, izinli aralık "
                f"1..{MAX_PLANNER_ATTEMPTS}")

    if arm == ARM_CONTRACT and "handoff_validation" in record:
        validation = record["handoff_validation"]
        if not isinstance(validation, dict):
            problems.append(f"handoff_validation dict değil: {validation!r}")
        elif not isinstance(validation.get("valid"), bool):
            problems.append(
                f"handoff_validation.valid bool değil: {validation.get('valid')!r}")
        else:
            problems += _contract_state_problems(record, validation, attempts)
    return problems


def _contract_state_problems(record: dict, validation: dict, attempts) -> list[str]:
    """Contract kolunun grafça İMKÂNSIZ olan son durumları.

    Bunlar tip hatası değil, GRAF MANTIĞI ihlalidir: kayıt tek tek bakıldığında
    makul görünür ama pipeline'ın bu durumu üretmesi mümkün değildir. Böyle bir
    kaydın sessizce geçmesi, retry kapısının fiilen çalışmadığını (erken çıkış)
    gizler ve RQ3'ün ölçtüğü şeyi doğrudan geçersiz kılar.

    1. Doğrulama BAŞARISIZ bittiyse graf denemeleri tüketmiş olmalıdır
       (`_route_after_validation`: geçersizken attempt < MAX ise planner'a döner).
       `valid=False` + `attempt_count < MAX_PLANNER_ATTEMPTS` = erken çıkış.
    2. Doğrulama BAŞARILI ise plan zaten JSON'a ayrıştırılmış demektir; validator
       ancak ayrıştırılmış bir nesneyi doğrulayabilir. `valid=True` +
       `handoff_parse_ok=False` çelişkisi, iki alanın farklı denemelerden
       kalmış olduğunu gösterir.
    """
    problems = []
    valid = validation["valid"]
    if not valid and _is_int(attempts) and attempts != MAX_PLANNER_ATTEMPTS:
        problems.append(
            f"contract erken çıkış: handoff_validation.valid=False iken "
            f"attempt_count={attempts}; graf {MAX_PLANNER_ATTEMPTS} denemeyi "
            "tüketmiş olmalıydı")
    if valid and record.get("handoff_parse_ok") is not True:
        problems.append(
            "contract tutarsız durum: handoff_validation.valid=True iken "
            f"handoff_parse_ok={record.get('handoff_parse_ok')!r}; doğrulama "
            "ancak ayrıştırılmış bir plan üzerinde başarılı olabilir")
    return problems


def stamp_record(record: dict, *, experiment: str, model: str, task_set: str,
                 arm: str, task_id: str, repeat: int, run_id: str,
                 arm_position: int) -> dict:
    """Üretilen ham kayda kimlik + provenance alanlarını basar.

    Kol implementasyonları (baseline / graph) kendi sonuç alanlarını üretir;
    deneye ait kimlik bilgisini runner burada tek noktadan ekler.

    `**record` EN BAŞTA açılır: runner'ın verdiği otoritatif alanların hepsi
    ondan SONRA yazılır ve ham kayıt bunları EZEMEZ. Aksi halde bir kol
    implementasyonunun kendi "model"/"experiment"/"schema_version" alanını
    yazması runner'ın kimliğini sessizce geçersiz kılardı — ve bu, analizde
    yanlış modele atfedilen kayıt olarak ortaya çıkardı.
    """
    return {
        **record,
        "schema_version": RESULT_SCHEMA_VERSION,
        "experiment": experiment,
        "model": model,
        "task_set": task_set,
        "arm": arm,
        "task_id": task_id,
        "repeat": repeat,
        "run_id": run_id,
        "arm_position": arm_position,
    }


def make_run_error_record(*, experiment: str, model: str, task_set: str, arm: str,
                          task_id: str, repeat: int, run_id: str,
                          arm_position: int, error: str) -> dict:
    """LLM/altyapı hatası kaydı. Saklanır ama tamamlanmış ölçüm sayılmaz."""
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "ts": datetime.now(timezone.utc).isoformat(),
        "experiment": experiment, "model": model, "task_set": task_set,
        "arm": arm, "task_id": task_id, "repeat": repeat,
        "run_id": run_id, "arm_position": arm_position,
        "status": RUN_ERROR_STATUS, "error": error,
    }


def make_synthetic_record(*, experiment: str = "sentetik", model: str = "test/model",
                          task_set: str = "heldout", arm: str = "baseline",
                          task_id: str = "t000", repeat: int = 0,
                          run_id: str | None = None, arm_position: int = 0,
                          base_pass: bool = True, plus_pass: bool = True,
                          error_class: str | None = None,
                          **extra) -> dict:
    """Şemaya UYAN sentetik kayıt üretir (analiz fixture'larının kaynağı).

    Analiz testleri alan adlarını elle yazmak yerine bunu çağırır: sözleşme
    değişirse testler de otomatik olarak yeni sözleşmeye göre üretir, böylece
    "testler yeşil ama gerçek kayıtla uyuşmuyor" durumu oluşmaz.

    base_pass=False iken plus_pass otomatik olarak False'a çekilir — Plus,
    base'i kapsadığı için başka türlüsü geçersiz bir kayıt olurdu.
    """
    plus_pass = plus_pass and base_pass
    base_status = "passed" if base_pass else "failed"
    plus_status = "passed" if plus_pass else "failed"
    if not base_pass:
        error_class = error_class or "assertion"

    record = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "ts": datetime.now(timezone.utc).isoformat(),
        "experiment": experiment, "model": model, "task_set": task_set,
        "arm": arm, "task_id": task_id, "repeat": repeat,
        "run_id": run_id or uuid4().hex, "arm_position": arm_position,
        "status": plus_status,
        "error_class": error_class if not plus_pass else None,
        "base_status": base_status, "base_pass": base_pass,
        "base_error_class": error_class if not base_pass else None,
        "base_duration_s": 0.01,
        "plus_status": plus_status, "plus_pass": plus_pass,
        "plus_error_class": error_class if not plus_pass else None,
        "plus_duration_s": 0.02,
        "base_plus_available": True,
        "plus_skipped_base_failed": not base_pass,
    }
    for field in ARM_REQUIRED_FIELDS.get(arm, ()):
        record.setdefault(field, {"valid": True} if field == "handoff_validation"
                          else (True if field == "handoff_parse_ok" else 1))
    record.update(extra)
    return record


# --- Manifest ↔ sonuç provenance ---------------------------------------------
# Manifest olmadan "eksik koşu" ile "hiç planlanmamış koşu" ayırt edilemez.
# llm_call_schema_version DAHİL: çağrı logunun sürümü koddaki güncel sabitten
# değil, deneyin KENDİ manifestinden doğrulanır.
MANIFEST_REQUIRED_FIELDS = ("name", "model", "task_set", "task_ids", "arm_order",
                            "repeats", "result_schema_version", "llm_call_schema_version")

# Sonuç kaydının manifestle EŞLEŞMESİ gereken alanları: (kayıt alanı, manifest alanı).
PROVENANCE_PAIRS = (("model", "model"), ("experiment", "name"),
                    ("task_set", "task_set"), ("schema_version", "result_schema_version"))


def check_provenance(records: list[dict], manifest: dict) -> list[str]:
    """Kayıtların MANİFESTLE eşleştiğini doğrular (boş liste = temiz).

    Kendi içinde tutarlı bir kayıt kümesi, DOĞRU kayıt kümesi demek değildir:
    bütün kayıtlar model/B taşırken manifest model/A diyorsa, yalnız "tek model
    var mı" diye bakan bir kontrol bunu göremez ve sonuçlar yanlış modelin
    verisi olarak doğru manifestin altında raporlanır. Bu, sonradan
    düzeltilmesi imkânsıza yakın bir atıf hatasıdır.

    Analiz VE MAST hattı aynı fonksiyonu kullanır — iki yerde ayrı yazılırsa
    biri güncellenip diğeri unutulur.
    """
    missing = [f for f in MANIFEST_REQUIRED_FIELDS if f not in manifest]
    if missing:
        return [f"manifestte analiz için zorunlu alan(lar) eksik: {missing}"]

    mismatches: dict[str, set] = {}
    for record in records:
        for record_field, manifest_field in PROVENANCE_PAIRS:
            if record.get(record_field) != manifest[manifest_field]:
                mismatches.setdefault(record_field, set()).add(repr(record.get(record_field)))
    return [f"manifest–sonuç uyuşmazlığı: kayıtların {field} alanı "
            f"{sorted(values)}, manifest {manifest[m_field]!r} diyor"
            for field, m_field in PROVENANCE_PAIRS
            for values in [mismatches.get(field)] if values]


# --- Çağrı logu (agents/llm.py) sözleşmesi -----------------------------------
# Analiz, maliyeti/tokenı/gecikmeyi sonuç kaydına BU alan üzerinden bağlar.
LLM_CALL_JOIN_FIELD = "run_id"

# Analizin çağrı logundan fiilen okuduğu alanlar. tests/test_llm.py gerçek bir
# çağrı kaydının bunları taşıdığını doğrular: llm.py'nin log formatı değişirse
# analiz sessizce sıfır maliyet raporlamak yerine testte kırılır.
LLM_CALL_USAGE_FIELDS = (
    "run_id", "model", "arm", "task_id", "repeat", "status",
    "input_tokens", "output_tokens", "reasoning_tokens", "cached_tokens",
    "cost_usd", "latency_s",
)


def make_synthetic_llm_call(*, run_id: str, model: str = "test/model",
                            arm: str = "baseline", task_id: str = "t000",
                            repeat: int = 0, agent_role: str = "coder",
                            status: str = "ok", provider_attempt: int = 1,
                            input_tokens: int = 100, output_tokens: int = 50,
                            reasoning_tokens: int = 0, cached_tokens: int = 0,
                            cost_usd: float = 0.001, latency_s: float = 1.5,
                            **extra) -> dict:
    """Şemaya UYAN sentetik çağrı logu kaydı (analiz fixture'larının kaynağı).

    `status="provider_error"` verildiğinde kullanım alanları BİLİNÇLİ olarak
    boş bırakılır: gerçek bozuk yanıtta da token/maliyet üretilmez ve o
    denemenin gecikmesi başarılı çağrının `latency_s` değerine zaten dahildir.
    """
    record = {
        "schema_version": LLM_CALL_SCHEMA_VERSION,
        "ts": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id, "model": model, "arm": arm, "task_id": task_id,
        "repeat": repeat, "agent_role": agent_role, "agent_attempt": 1,
        "status": status, "provider_attempt": provider_attempt,
    }
    if status == "ok":
        record.update(input_tokens=input_tokens, output_tokens=output_tokens,
                      reasoning_tokens=reasoning_tokens, cached_tokens=cached_tokens,
                      cost_usd=cost_usd, latency_s=latency_s,
                      provider_error_retries=provider_attempt - 1)
    elif status == "error":
        # Taşıma istisnası: token/maliyet yok ama süre ÖLÇÜLMÜŞTÜR
        # (agents/llm.py istisna dalında latency_s yazar).
        record.update(error_type="APIError", error_message="sentetik",
                      latency_s=latency_s)
    elif status == "provider_error":
        # HTTP 200 içine gömülü hata: süre DENEME BAŞINA ve latency_ms adıyla
        # yazılır (llm.py `_provider_error_details`). Bu denemenin süresi,
        # sonunda başarılı olan çağrının latency_s'ine zaten dahildir.
        record.update(error_signature="zero_completion_tokens",
                      latency_ms=round(latency_s * 1000))
    record.update(extra)
    return record


# --- Bütünlük kontrolleri ----------------------------------------------------

def expected_resume_keys(task_ids: list[str], arms: list[str], repeats: int,
                         model: str) -> set[tuple]:
    """Deneyin tamamlanmış sayılması için gereken resume anahtarları."""
    return {(model, arm, task_id, rep)
            for rep in range(repeats) for task_id in task_ids for arm in arms}


def find_duplicates(records: list[dict]) -> dict[tuple, int]:
    """Aynı resume anahtarından birden fazla TAMAMLANMIŞ kayıt.

    run_error'lar hariç: bir koşu hata alıp yeniden denendiğinde aynı anahtarda
    bir run_error + bir gerçek kayıt bulunması NORMALDİR. İki gerçek kayıt ise
    çift sayıma yol açar ve analizi durdurmalıdır.
    """
    counts: dict[tuple, int] = {}
    for r in records:
        if is_run_error(r):
            continue
        key = resume_key(r)
        counts[key] = counts.get(key, 0) + 1
    return {k: n for k, n in counts.items() if n > 1}


def find_missing(records: list[dict], expected: set[tuple]) -> set[tuple]:
    """Beklenip tamamlanmamış koşular (run_error'lar tamamlanmış sayılmaz)."""
    done = {resume_key(r) for r in records if not is_run_error(r)}
    return expected - done


def find_unexpected(records: list[dict], expected: set[tuple]) -> set[tuple]:
    """Beklenen kümede OLMAYAN anahtarlar (yanlış model/görev/kol kaydı).

    Eksik kontrolü tek başına yetmez: yanlış bir modelin/görevin kaydı dosyaya
    girmişken beklenenlerin hepsi de tamamlanmışsa, yalnız `missing`e bakan bir
    kontrol deneyi yanlışlıkla `complete=True` sayardı.
    """
    return {resume_key(r) for r in records} - expected


def integrity_report(records: list[dict], task_ids: list[str], arms: list[str],
                     repeats: int, model: str) -> dict:
    """Analiz öncesi tek bakışta bütünlük özeti."""
    expected = expected_resume_keys(task_ids, arms, repeats, model)
    completed = [r for r in records if not is_run_error(r)]
    invalid = {}
    for r in records:
        problems = validate_record(r)
        if problems:
            invalid[r.get("run_id", "?")] = problems
    missing = find_missing(records, expected)
    duplicates = find_duplicates(records)
    unexpected = find_unexpected(records, expected)
    return {
        "expected_count": len(expected),
        "completed_count": len(completed),
        "run_error_count": sum(is_run_error(r) for r in records),
        "duplicates": duplicates,
        "missing": sorted(missing),
        "unexpected": sorted(unexpected),
        "invalid_records": invalid,
        "complete": not (missing or duplicates or unexpected or invalid),
    }


def verify_resumable(records: list[dict], expected: set[tuple]) -> list[str]:
    """Devam etmeden ÖNCE mevcut sonuç dosyasını denetler.

    Devam etmeye ENGEL olanlar (yeni API çağrısı yapmadan durulmalı — aksi
    halde bozuk bir veri kümesinin üstüne pahalı koşu eklenir):
      - şema ihlali olan kayıt
      - yinelenen tamamlanmış kayıt
      - beklenmeyen anahtar (yanlış model/görev/kol)

    Devam etmeye ENGEL OLMAYANLAR: eksik koşular (zaten devam etmenin sebebi)
    ve geçmiş `run_error` kayıtları (yeniden denenmek üzere saklanır).
    """
    problems = []
    for r in records:
        for p in validate_record(r):
            problems.append(f"geçersiz kayıt {r.get('run_id', '?')}: {p}")
    for key, n in find_duplicates(records).items():
        problems.append(f"yinelenen tamamlanmış kayıt {key}: {n} kez")
    for key in sorted(find_unexpected(records, expected)):
        problems.append(f"beklenmeyen anahtar (yanlış model/görev/kol?): {key}")
    return problems
