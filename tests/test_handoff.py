"""agents/handoff.py birim testleri — canonical serileştirme ve hata zarfı.

Bu katman structured ve contract kollarının handoff BİÇİMİNİ birebir aynı
tutan tek noktadır; determinizmi bozulursa iki kol arasındaki fark
"validator+retry" olmaktan çıkar (RQ3 ölçülemez hale gelir).
"""

import json

from agents.contracts import PlannerOutput
from agents.handoff import (
    canonical_json,
    is_parse_error_envelope,
    parse_error_envelope,
)


def test_anahtar_sirasi_ciktiyi_degistirmez():
    # Aynı içerik, farklı ekleme sırası -> aynı bayt dizisi. Aksi halde iki kol
    # aynı planı farklı metinlerle görebilirdi.
    a = {"task_id": "t1", "steps": [], "function_signature": "def f()"}
    b = {"function_signature": "def f()", "task_id": "t1", "steps": []}
    assert canonical_json(a) == canonical_json(b)


def test_unicode_kacirilmaz():
    assert "ö" in canonical_json({"description": "böl"})
    assert "\\u" not in canonical_json({"description": "böl"})


def test_cikti_gecerli_json():
    plan = {"task_id": "t1", "steps": [{"description": "adım"}]}
    assert json.loads(canonical_json(plan)) == plan


def test_hata_zarfi_sabit_anahtarlara_sahip():
    env = parse_error_envelope("ham model ciktisi", "gecerli JSON yok")
    assert env["_handoff_status"] == "parse_error"
    assert env["_parse_error"] == "gecerli JSON yok"
    assert env["_raw_text"] == "ham model ciktisi"
    assert is_parse_error_envelope(env)


def test_hata_zarfi_sozlesmeyi_kesinlikle_ihlal_eder():
    # Zarfın contract kolunda MUTLAKA retry tetiklemesi gerekiyor. PlannerOutput
    # extra="forbid" olduğu için zarf anahtarları şemayı zorunlu olarak bozar --
    # bu davranış tesadüfi değil, tasarımın dayanağı.
    env = parse_error_envelope("x", "y")
    try:
        PlannerOutput.model_validate(env)
    except Exception:
        return
    raise AssertionError("hata zarfı pydantic doğrulamasından geçmemeliydi")


def test_normal_plan_zarf_sayilmaz():
    assert not is_parse_error_envelope({"task_id": "t1", "steps": []})
    assert not is_parse_error_envelope("serbest metin plan")
    assert not is_parse_error_envelope(None)
