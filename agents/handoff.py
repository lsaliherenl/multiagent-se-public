"""Planner→coder handoff'unun ortak serileştirme katmanı.

`structured_no_validation` (Kol 3a) ve `contract` (Kol 3b) kolları arasındaki
TEK farkın validator + sınırlı retry olması gerekiyor (EXPERIMENT_PROTOCOL.md §3).
Bu şart, handoff'un BİÇİMİ de iki kolda birebir aynı olduğunda sağlanır — bu
yüzden serileştirme tek bir yardımcıda toplanmıştır: iki kol da aynı
fonksiyonu çağırır, coder'a giden metin byte-for-byte aynıdır.

Parse hatası özel durumu: planner geçerli JSON üretemezse coder'a ÇIPLAK ham
metin gönderilmez — deterministik bir "hata zarfı" (error envelope) canonical
JSON olarak gönderilir. Böylece:
- iki kolun coder handoff biçimi her koşulda canonical JSON kalır (serbest
  metne düşen gizli bir üçüncü davranış yoktur),
- structured kol zarfı retry etmeden coder'a iletir,
- contract kolda aynı zarfı validator (pydantic) reddeder ve planner retry
  edilir; denemeler tükenirse coder yine aynı türden zarfı alır.

Parse başarısı ile sözleşme doğrulaması AYRI kavramlardır ve ayrı loglanır:
- handoff_parse_ok: metin geçerli bir JSON nesnesine dönüştü mü (HER kolda)
- handoff_validation: pydantic sözleşmesine uydu mu (YALNIZ contract kolunda)
"""

import json

HANDOFF_STATUS_KEY = "_handoff_status"
PARSE_ERROR_STATUS = "parse_error"


def canonical_json(plan: dict) -> str:
    """Plan sözlüğünü deterministik JSON metnine çevirir.

    sort_keys: aynı içerik her zaman aynı bayt dizisi üretsin (dict ekleme
    sırası prompt'u değiştirmesin). ensure_ascii=False: Türkçe/Unicode karakter
    \\uXXXX'e kaçırılmasın. separators açıkça verilir (sürümler arası
    varsayılan değişikliğine karşı). indent=2: coder'ın okuyabilmesi için —
    okunabilirlik determinizmden ödün vermeden korunur.
    """
    return json.dumps(plan, sort_keys=True, ensure_ascii=False,
                      indent=2, separators=(",", ": "))


def parse_error_envelope(raw_text: str, error: str) -> dict:
    """Planner geçerli JSON üretemediğinde coder'a/validator'a giden zarf.

    Sabit anahtar kümesi: pydantic PlannerOutput `extra="forbid"` olduğu için
    contract kolunda bu zarf ZORUNLU olarak reddedilir (zaten istenen davranış:
    retry tetiklenir), structured kolunda ise doğrulanmadan iletilir.
    """
    return {
        HANDOFF_STATUS_KEY: PARSE_ERROR_STATUS,
        "_parse_error": error,
        "_raw_text": raw_text,
    }


def is_parse_error_envelope(plan) -> bool:
    return isinstance(plan, dict) and plan.get(HANDOFF_STATUS_KEY) == PARSE_ERROR_STATUS
