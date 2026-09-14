"""EvalPlus tabanlı held-out görev seçiminin SAF (LLM'siz, ağsız) mantığı.

EXPERIMENT_PROTOCOL.md §5. Bu modül yalnız deterministik dönüşüm/filtre/seçim
fonksiyonlarını içerir; indirme ve sandbox çalıştırma scripts/fetch_evalplus.py
tarafındadır. Ayrım bilinçli: buradaki her karar birim testiyle sınanabilmeli.

Oracle yaklaşımı
----------------
EvalPlus veri dosyaları beklenen ÇIKTILARI içermez; yalnız girdileri (base_input
/ plus_input) ve referans çözümü verir. Beklenen çıktılar referans çözüm
çalıştırılarak üretilir — EvalPlus'ın kendi yaptığı da budur.

Girdi deserializasyonu TAHMİN EDİLMEZ, DOĞRULANIR: JSON tuple taşıyamadığı için
EvalPlus bazı görevlerin tuple girdilerini liste olarak saklar. Doğru varyantı
bulmak için EvalPlus'ın kendi `contract` alanı (girdi tiplerini assert eden
kod) referans çözüme enjekte edilir; yanlış varyant AssertionError ile düşer.
Hiçbir varyantın geçmediği görev, makine-okunur nedenle REDDEDİLİR.

Karşılaştırma semantiği
-----------------------
Beklenen çıktı referans çözümden üretildiği için karşılaştırma tam eşitliktir
(atol>0 ise kayan noktada mutlak tolerans). EvalPlus'ın eşitlik-DIŞI oracle
gerektirdiği görevler (küme eşitliği, "None değil", özel matematik) bilinçli
olarak DIŞLANIR — o görevlerde tam eşitlik, doğru ama farklı sıralı bir çözümü
haksız yere eleyebilirdi.

Bu nedenle üretilen `plus_pass` oranları EvalPlus liderlik tablosu sayılarıyla
DOĞRUDAN KIYASLANAMAZ; ölçtüğümüz şey kollar ARASI farktır, mutlak seviye değil.
"""

import ast
import hashlib
import random
import re
from collections import Counter, OrderedDict, deque

# Beklenen çıktıların repr'ı geri çevrilirken tanınan adlar. Hem üretilen test
# kodunun preamble'ı hem parse_groundtruth_output aynı kümeyi kullanır —
# ayrışırlarsa üretim aşamasında kabul edilen bir görev koşu aşamasında
# NameError verirdi.
REPR_NAMESPACE = {
    "nan": float("nan"), "inf": float("inf"),
    "Counter": Counter, "OrderedDict": OrderedDict, "deque": deque,
}

# --- Kaynak sürüm sabitleri (seçim manifestinde dondurulur) ------------------
# Sürüm etiketine sabitlenmiştir; SHA-256 ayrıca bozuk/değişmiş indirmeyi
# yakalar (2026-07-27'de canlı indirilip hesaplandı).
EVALPLUS_SOURCES = {
    "humanevalplus": {
        "version": "v0.1.10",
        "url": "https://github.com/evalplus/humanevalplus_release/releases/"
               "download/v0.1.10/HumanEvalPlus.jsonl.gz",
        "sha256": "272720b90ac375502c8ed23cd791c2a93dfb22a911641a494da74a426c09f101",
        "size_bytes": 925932,
        "prefix": "humanevalplus",
    },
    "mbppplus": {
        "version": "v0.2.0",
        "url": "https://github.com/evalplus/mbppplus_release/releases/"
               "download/v0.2.0/MbppPlus.jsonl.gz",
        "sha256": "af43697e8791c4c149bdfd6b489d8b5412507551ac20e28a439f650b8225db63",
        "size_bytes": 336032,
        "prefix": "mbppplus",
    },
}

# --- EvalPlus'ın eşitlik-dışı oracle kullandığı görevler (DIŞLANIR) ----------
# Kaynak: evalplus/eval/_special_oracle.py (2026-07-27'de doğrulandı).
# Bu görevlerde doğru cevap tek bir değere indirgenemez (sıra bağımsız küme,
# yalnız "None değil" koşulu, ya da göreve özel matematiksel oracle).
SPECIAL_ORACLE_ENTRY_POINTS = frozenset({
    # çıktı yalnız "None değil" diye kontrol edilir
    "check_str", "text_match_three", "text_starta_endb",
    # çıktı küme olarak (sırasız) karşılaştırılır
    "similar_elements", "find_char_long", "common_in_nested_lists",
    "extract_singly", "larg_nnum", "intersection_array", "find_dissimilar", "Diff",
})
SPECIAL_ORACLE_TASK_IDS = frozenset({
    "Mbpp/581",     # _surface_Area: göreve özel geometri oracle'ı
    "Mbpp/558",     # _digit_distance_nums: basamak hizalama oracle'ı
    "HumanEval/32", # _poly: kökün poly(x)≈0 sağlaması yeterli, tek değer değil
})

# --- Harness sözleşmesiyle çakışan adlar ------------------------------------
# eval/harness.py üretilen script'te `check(<entry_point>)` çağırır; entry_point
# bu adlardan biri olursa çağrı kendi kendini gölgeler.
RESERVED_NAMES = frozenset({"check", "candidate"})

# --- İzin verilen standard-library importları --------------------------------
# Harici paket YOK; ayrıca ağ/dosya/saat/randomness taşıyan modüller yok
# (deterministik davranış şartı, §5.3).
ALLOWED_IMPORTS = frozenset({
    "math", "cmath", "re", "collections", "itertools", "functools", "heapq",
    "bisect", "string", "typing", "operator", "copy", "statistics", "fractions",
    "decimal", "numbers", "array", "unicodedata", "textwrap", "enum", "dataclasses",
})
# Determinizmi/izolasyonu bozan çağrı ve adlar (referans çözümde aranır).
FORBIDDEN_PATTERNS = (
    "random", "time", "datetime", "os.", "sys.", "socket", "urllib", "requests",
    "subprocess", "pathlib", "open(", "input(", "eval(", "exec(", "__import__",
    "id(", "hash(",  # nesne kimliği/hash -> süreçler arası değişebilir
)

_IMPORT_RE = re.compile(r"^\s*(?:from\s+([A-Za-z_][\w.]*)|import\s+([A-Za-z_][\w.]*))", re.M)


# --- Referans çözüm inşası ---------------------------------------------------

def reference_solution(record: dict, source: str, *, with_contract: bool) -> str | None:
    """Referans (ground-truth) çözüm kodu.

    with_contract=True: EvalPlus'ın girdi-geçerlilik assert'leri enjekte edilir.
    Bu, beklenen çıktı üretilirken doğru girdi deserializasyonunu DOĞRULAMAK
    için kullanılır (yanlış varyant AssertionError verir).
    with_contract=False: aday koda benzeyen sade referans — harness'ın kendisini
    doğrulamak (referans çözüm testleri geçiyor mu) için kullanılır.

    HumanEval'de contract, prompt (imza + docstring) ile gövde arasına girer.
    MBPP'de canonical_solution tam bir fonksiyon olduğu için def satırından
    hemen sonraya enjekte edilir. Def satırı bulunamazsa None (görev reddedilir).
    """
    contract = record.get("contract", "") if with_contract else ""
    if source == "humanevalplus":
        return record["prompt"] + contract + record["canonical_solution"]

    solution = record["canonical_solution"]
    if not contract:
        return solution
    lines = solution.split("\n")
    for i, line in enumerate(lines):
        if line.lstrip().startswith("def ") and line.rstrip().endswith(":"):
            return "\n".join(lines[: i + 1]) + contract + "\n".join(lines[i + 1:])
    return None


# --- Girdi deserializasyon varyantları ---------------------------------------

def _tuplify(value):
    return tuple(_tuplify(v) for v in value) if isinstance(value, list) else value


def apply_variant(args: list, variant: str) -> list:
    """Tek bir çağrının argüman listesine deserializasyon varyantını uygular."""
    if variant == "asis":
        return list(args)
    if variant == "outer_tuple":
        return [tuple(a) if isinstance(a, list) else a for a in args]
    if variant == "deep_tuple":
        return [_tuplify(a) for a in args]
    raise ValueError(f"bilinmeyen varyant: {variant!r}")


INPUT_VARIANTS = ("asis", "outer_tuple", "deep_tuple")


# --- Statik (çalıştırmasız) uygunluk filtresi --------------------------------

def static_eligibility(record: dict, source: str, excluded_ids: set[str]) -> str | None:
    """Model çağrısı VE kod çalıştırma OLMADAN uygulanabilen filtreler.

    Uygunsa None, değilse makine-okunur ret nedeni döndürür (§5.4 md. 3:
    her ret için neden kaydedilir).
    """
    task_id = record["task_id"]
    if task_id in excluded_ids:
        return "pilot_id_dislandi"
    if task_id in SPECIAL_ORACLE_TASK_IDS:
        return "ozel_oracle_gorevi"
    entry_point = record.get("entry_point") or ""
    if not entry_point:
        return "entry_point_yok"
    if entry_point in SPECIAL_ORACLE_ENTRY_POINTS:
        return "ozel_oracle_gorevi"
    if entry_point in RESERVED_NAMES:
        return "harness_isim_cakismasi"
    if not record.get("base_input") or not record.get("plus_input"):
        return "base_veya_plus_girdi_yok"

    reference = reference_solution(record, source, with_contract=False)
    if reference is None:
        return "referans_cozum_uretilemedi"
    if not re.search(rf"^\s*def\s+{re.escape(entry_point)}\s*\(", reference, re.M):
        return "entry_point_tanimli_degil"

    for module in _imported_modules(reference):
        if module not in ALLOWED_IMPORTS:
            return f"izinsiz_import:{module}"
    for pattern in FORBIDDEN_PATTERNS:
        if pattern in reference:
            return f"determinizm_riski:{pattern.rstrip('(.')}"
    return None


def _imported_modules(code: str) -> set[str]:
    """Kodun import ettiği ÜST düzey modül adları ("a.b" -> "a")."""
    return {(m.group(1) or m.group(2)).split(".")[0] for m in _IMPORT_RE.finditer(code)}


# --- Test kodu üretimi -------------------------------------------------------

# Karşılaştırıcı: atol=0'da tam eşitlik, atol>0'da kayan noktada mutlak
# tolerans. NaN == NaN kabul edilir (aksi halde referans çözüm kendi çıktısıyla
# eşleşmezdi). Liste/tuple ayrımı KORUNUR (tip de sonucun parçası).
_COMPARATOR = '''
import copy as _copy
import math as _math

# Beklenen çıktılar repr ile gömülür; aşağıdaki adlar tanımlı olmazsa repr'ı
# geri çevrilebilen bu değerler NameError verir ve görev haksız yere elenirdi.
# repr(float("nan")) == "nan", repr(Counter(...)) == "Counter({...})" -- ikisi
# de tek başına geçerli kaynak değil. (defaultdict bilerek YOK: repr'ı
# "<class 'int'>" içerir, geri çevrilemez -> o görevler reddedilir.)
from collections import Counter, OrderedDict, deque  # noqa: F401

nan = float("nan")
inf = float("inf")


def _eq(a, b, atol):
    if isinstance(a, float) or isinstance(b, float):
        if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
            return False
        if isinstance(a, bool) or isinstance(b, bool):
            return a is b
        if _math.isnan(a) and _math.isnan(b):
            return True
        return abs(a - b) <= atol if atol else a == b
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return (type(a) is type(b) and len(a) == len(b)
                and all(_eq(x, y, atol) for x, y in zip(a, b)))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_eq(a[k], b[k], atol) for k in a)
    return a == b
'''


def build_test_code(cases: list[tuple], atol: float) -> str:
    """`check(candidate)` sözleşmesine uyan test kodu üretir.

    cases: (argümanlar, beklenen_çıktı) çiftleri. repr ile gömülür — JSON'un
    aksine tuple/list ayrımını korur (bu ayrım karşılaştırmanın parçası).
    Her çağrı öncesi argümanlar deepcopy'lenir: aday kod girdiyi değiştirirse
    sonraki vakalar bozulmasın.
    """
    return (
        f"{_COMPARATOR}\n"
        f"_ATOL = {atol!r}\n"
        f"_CASES = {cases!r}\n\n"
        "def check(candidate):\n"
        "    for _i, (_args, _expected) in enumerate(_CASES):\n"
        "        _out = candidate(*_copy.deepcopy(_args))\n"
        "        assert _eq(_out, _expected, _ATOL), (\n"
        "            'vaka %d: girdi=%r beklenen=%r alinan=%r'\n"
        "            % (_i, _args, _expected, _out))\n"
    )


def build_groundtruth_script(reference: str, entry_point: str,
                             inputs: list, variant: str, marker: str) -> str:
    """Referans çözümü verilen girdilerle koşturup çıktıları repr olarak basar.

    Sandbox'ta çalıştırılmak üzere üretilir (LLM yok, deterministik). Çıktılar
    marker'dan sonra tek satırda repr olarak basılır; okuyan taraf
    ast.literal_eval ile geri alır.
    """
    return (
        "import copy as _copy\n"
        f"{reference}\n\n"
        f"_INPUTS = {inputs!r}\n"
        f"_VARIANT = {variant!r}\n"
        "_OUT = []\n"
        "for _args in _INPUTS:\n"
        f"    _OUT.append({entry_point}(*_copy.deepcopy(_args)))\n"
        f"print({marker!r})\n"
        "print(repr(_OUT))\n"
    )


def parse_groundtruth_output(stdout: str, marker: str) -> list:
    """build_groundtruth_script çıktısını Python değerlerine çevirir.

    Önce ast.literal_eval denenir (yalnız literal veri). Başarısız olursa tek
    kabul edilen genişletme NaN/Inf'tir: repr(float("nan")) == "nan" ve
    repr(float("inf")) == "inf" — ikisi de literal DEĞİL, bu yüzden
    literal_eval onları ayrıştıramaz ve bu değerleri üreten görevler haksız
    yere reddedilirdi. Bu durumda builtins'i TAMAMEN boşaltılmış ve yalnız
    nan/inf adlarını gören bir ortamda değerlendirilir; çağrı/öznitelik erişimi
    yapacak bir ad kalmadığı için literal ayrıştırmadan öteye geçemez. Girdi
    zaten SHA-256 ile doğrulanmış EvalPlus referans çözümünün kendi çıktısıdır.
    Yine de çözülemezse ValueError -> görev reddedilir.
    """
    _, _, tail = stdout.partition(marker)
    tail = tail.strip()
    if not tail:
        raise ValueError("groundtruth marker'ı bulunamadı")
    try:
        return ast.literal_eval(tail)
    except (ValueError, SyntaxError):
        pass
    try:
        return eval(tail, {"__builtins__": {}}, dict(REPR_NAMESPACE))  # noqa: S307
    except Exception as exc:
        # Counter dışı defaultdict, özel sınıflar vb.: repr'ı geri
        # çevrilemeyen çıktılar. Görev REDDEDİLİR (çökme değil) -- neden
        # seçim manifestine yazılır.
        raise ValueError(f"çıktı temel literallerle temsil edilemiyor: {exc}") from exc


# --- Deterministik seçim -----------------------------------------------------

def deterministic_select(eligible_ids: list[str], count: int, seed: int) -> list[str]:
    """Uygun havuzdan sabit seed ile `count` görev seçer.

    Havuz önce sıralanır: çağıranın verdiği sıra (dosya okuma sırası vb.)
    sonucu etkilemesin — aynı havuz, aynı seed, her zaman aynı seçim.
    Seçilenler de sıralı döndürülür (koşu sırası deterministik).
    """
    if len(eligible_ids) < count:
        raise ValueError(f"uygun görev yetersiz: {len(eligible_ids)} < {count}")
    pool = sorted(eligible_ids)
    rng = random.Random(seed)
    return sorted(rng.sample(pool, count))


def task_content_hash(task: dict) -> str:
    """Görev içeriğinin sha256'sı — seçim manifestinde bütünlük kaydı."""
    payload = "\n".join(
        f"{k}={task[k]!r}" for k in sorted(task) if k != "content_sha256"
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
