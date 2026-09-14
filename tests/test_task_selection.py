"""eval/task_selection.py birim testleri — held-out seçiminin saf mantığı.

Bu testler ağ/model/sandbox gerektirmez. Seçim boru hattının bozulması, ana
deney görev setinin sessizce değişmesi demektir (EXPERIMENT_PROTOCOL.md §5:
seçim commit edildikten sonra held-out görevlerde pilot yapılmaz) — bu yüzden
her filtre kuralı ayrı ayrı sınanır.
"""

from collections import Counter

import pytest

from eval.task_selection import (
    ALLOWED_IMPORTS,
    EVALPLUS_SOURCES,
    REPR_NAMESPACE,
    apply_variant,
    build_test_code,
    deterministic_select,
    parse_groundtruth_output,
    reference_solution,
    static_eligibility,
    task_content_hash,
)


def _he_record(**over):
    rec = {
        "task_id": "HumanEval/999",
        "prompt": "def add_one(x):\n    '''doc'''\n",
        "contract": "    assert isinstance(x, int)\n",
        "canonical_solution": "    return x + 1\n",
        "entry_point": "add_one",
        "base_input": [[1]],
        "plus_input": [[2]],
        "atol": 0,
    }
    rec.update(over)
    return rec


def _mbpp_record(**over):
    rec = {
        "task_id": "Mbpp/999",
        "prompt": "Write a function.",
        "contract": "\n  assert isinstance(x, tuple)\n",
        "canonical_solution": "\ndef add_one(x):\n  return len(x) + 1\n",
        "entry_point": "add_one",
        "base_input": [[[1, 2]]],
        "plus_input": [[[3]]],
        "atol": 0,
    }
    rec.update(over)
    return rec


# --- Kaynak sabitleri --------------------------------------------------------

def test_kaynaklar_surum_ve_hash_ile_sabitlenmis():
    # Sürüm/hash olmadan "aynı görev seti" iddiası doğrulanamaz.
    for source, meta in EVALPLUS_SOURCES.items():
        assert meta["version"].startswith("v")
        assert len(meta["sha256"]) == 64
        assert meta["version"] in meta["url"], f"{source}: URL sürüme sabitlenmemiş"


# --- Referans çözüm inşası ---------------------------------------------------

def test_humaneval_referansi_prompt_contract_cozum_sirasinda():
    ref = reference_solution(_he_record(), "humanevalplus", with_contract=True)
    assert ref.index("'''doc'''") < ref.index("assert isinstance") < ref.index("return x + 1")


def test_contractsiz_referans_assert_icermez():
    ref = reference_solution(_he_record(), "humanevalplus", with_contract=False)
    assert "assert isinstance" not in ref


def test_mbpp_contracti_def_satirindan_hemen_sonra_girer():
    ref = reference_solution(_mbpp_record(), "mbppplus", with_contract=True)
    lines = [l for l in ref.split("\n") if l.strip()]
    assert lines[0].startswith("def add_one")
    assert "assert isinstance" in lines[1]


def test_mbpp_def_satiri_yoksa_none():
    rec = _mbpp_record(canonical_solution="x = 1\n")
    assert reference_solution(rec, "mbppplus", with_contract=True) is None


# --- Statik uygunluk filtresi ------------------------------------------------

def test_pilot_idleri_dislanir():
    assert static_eligibility(_he_record(), "humanevalplus", {"HumanEval/999"}) == "pilot_id_dislandi"


def test_uygun_gorev_none_dondurur():
    assert static_eligibility(_he_record(), "humanevalplus", set()) is None


def test_ozel_oracle_gorev_idsi_dislanir():
    rec = _he_record(task_id="HumanEval/32")
    assert static_eligibility(rec, "humanevalplus", set()) == "ozel_oracle_gorevi"


def test_ozel_oracle_entry_pointi_dislanir():
    # Küme-eşitliği oracle'ı gerektiren görev; tam eşitlik doğru ama farklı
    # sıralı bir çözümü haksız eleyebilirdi.
    rec = _mbpp_record(entry_point="similar_elements",
                       canonical_solution="\ndef similar_elements(x):\n  return x\n")
    assert static_eligibility(rec, "mbppplus", set()) == "ozel_oracle_gorevi"


def test_harness_ismiyle_cakisan_entry_point_dislanir():
    rec = _he_record(entry_point="check", prompt="def check(x):\n", canonical_solution="    return x\n")
    assert static_eligibility(rec, "humanevalplus", set()) == "harness_isim_cakismasi"


def test_izinsiz_import_dislanir():
    rec = _he_record(canonical_solution="    import numpy\n    return x\n")
    assert static_eligibility(rec, "humanevalplus", set()) == "izinsiz_import:numpy"


def test_izin_verilen_import_kabul_edilir():
    rec = _he_record(canonical_solution="    import math\n    return math.floor(x)\n")
    assert static_eligibility(rec, "humanevalplus", set()) is None
    assert "math" in ALLOWED_IMPORTS


@pytest.mark.parametrize("kod,beklenen", [
    ("    return random.random()\n", "determinizm_riski:random"),
    ("    return time.time()\n", "determinizm_riski:time"),
    ("    return open('f').read()\n", "determinizm_riski:open"),
    ("    return id(x)\n", "determinizm_riski:id"),
])
def test_determinizm_riski_tasiyan_referans_dislanir(kod, beklenen):
    assert static_eligibility(_he_record(canonical_solution=kod), "humanevalplus", set()) == beklenen


def test_girdi_kumesi_eksikse_dislanir():
    assert static_eligibility(_he_record(plus_input=[]), "humanevalplus", set()) == "base_veya_plus_girdi_yok"


def test_entry_point_tanimli_degilse_dislanir():
    rec = _he_record(prompt="def baska_ad(x):\n", canonical_solution="    return x\n")
    assert static_eligibility(rec, "humanevalplus", set()) == "entry_point_tanimli_degil"


# --- Girdi varyantları -------------------------------------------------------

def test_varyantlar_beklenen_donusumu_yapar():
    args = [[1, [2, 3]], "s"]
    assert apply_variant(args, "asis") == [[1, [2, 3]], "s"]
    assert apply_variant(args, "outer_tuple") == [(1, [2, 3]), "s"]
    assert apply_variant(args, "deep_tuple") == [(1, (2, 3)), "s"]


def test_bilinmeyen_varyant_reddedilir():
    with pytest.raises(ValueError):
        apply_variant([[1]], "yok")


# --- Üretilen test kodu ------------------------------------------------------

def _run_check(test_code: str, candidate_src: str) -> bool:
    """Üretilen check()'i gerçekten çalıştırır (kendi ürettiğimiz kod)."""
    ns = {}
    exec(candidate_src, ns)
    exec(test_code, ns)
    try:
        ns["check"](ns["candidate"])
        return True
    except AssertionError:
        return False


def test_dogru_aday_gecer_yanlis_aday_kalir():
    code = build_test_code([([1], 2), ([5], 6)], 0)
    assert _run_check(code, "def candidate(x):\n    return x + 1\n")
    assert not _run_check(code, "def candidate(x):\n    return x + 2\n")


def test_atol_kayan_noktada_tolerans_uygular():
    gevsek = build_test_code([([1.0], 2.0)], 0.01)
    siki = build_test_code([([1.0], 2.0)], 0)
    yaklasik = "def candidate(x):\n    return 2.005\n"
    assert _run_check(gevsek, yaklasik)
    assert not _run_check(siki, yaklasik)


def test_liste_ve_tuple_ayrimi_korunur():
    # Tip de sonucun parçası: tuple bekleyen bir görevde liste dönmek hatadır.
    code = build_test_code([([1], (1, 2))], 0)
    assert _run_check(code, "def candidate(x):\n    return (1, 2)\n")
    assert not _run_check(code, "def candidate(x):\n    return [1, 2]\n")


def test_girdiyi_degistiren_aday_sonraki_vakalari_bozamaz():
    # Her çağrı öncesi deepcopy: aday kod girdiyi mutasyona uğratırsa bile
    # sonraki vakalar orijinal girdiyi görür.
    code = build_test_code([([[1, 2]], 2), ([[1, 2]], 2)], 0)
    mutasyoncu = "def candidate(xs):\n    n = len(xs)\n    xs.clear()\n    return n\n"
    assert _run_check(code, mutasyoncu)


def test_nan_kendisiyle_esit_sayilir():
    # Beklenen çıktı referanstan üretildiği için NaN != NaN olsaydı referans
    # kendi testini geçemezdi.
    code = build_test_code([([1], float("nan"))], 0)
    assert _run_check(code, "def candidate(x):\n    return float('nan')\n")


def test_bool_ile_sayi_karistirilmaz():
    code = build_test_code([([1], 1.0)], 0.5)
    assert not _run_check(code, "def candidate(x):\n    return True\n")


# --- Groundtruth çıktı ayrıştırma --------------------------------------------

def test_groundtruth_ciktisi_ayristirilir():
    out = parse_groundtruth_output("gurultu\n__M__\n[1, (2, 3), 'a']\n", "__M__")
    assert out == [1, (2, 3), "a"]


def test_marker_yoksa_hata():
    with pytest.raises(ValueError):
        parse_groundtruth_output("marker yok", "__M__")


def test_nan_inf_iceren_groundtruth_ayristirilir():
    # repr(nan)=="nan" literal DEĞİL; literal_eval tek başına bunu ayrıştıramaz
    # ve NaN/Inf üreten görevler haksız yere reddedilirdi.
    out = parse_groundtruth_output("__M__\n[nan, inf, -inf, 1.5]\n", "__M__")
    assert out[0] != out[0]          # NaN
    assert out[1] == float("inf")
    assert out[2] == float("-inf")
    assert out[3] == 1.5


def test_groundtruth_ayristirmasi_cagri_yapamaz():
    # Genişletilmiş değerlendirme yalnız REPR_NAMESPACE adlarını görür;
    # builtins boş, dolayısıyla import/dosya erişimi mümkün değil.
    with pytest.raises(ValueError):
        parse_groundtruth_output("__M__\n[__import__('os').getcwd()]\n", "__M__")


def test_counter_ciktisi_ayristirilir():
    out = parse_groundtruth_output("__M__\n[Counter({'a': 2})]\n", "__M__")
    assert out == [Counter({"a": 2})]


def test_repr_i_geri_cevrilemeyen_cikti_reddedilir():
    # defaultdict repr'ı "<class 'int'>" içerir -> geri çevrilemez. Bu ÇÖKME
    # değil, makine-okunur RET olmalı (seçim manifestine yazılabilsin).
    with pytest.raises(ValueError, match="temel literallerle temsil edilemiyor"):
        parse_groundtruth_output("__M__\ndefaultdict(<class 'int'>, {})\n", "__M__")


def test_uretilen_test_kodu_ile_ayristirma_ayni_adlari_taniyor():
    # İki taraf ayrışırsa üretimde kabul edilen görev koşuda NameError verirdi.
    preamble = build_test_code([], 0)
    for ad in REPR_NAMESPACE:
        assert ad in preamble, f"{ad} üretilen test kodunda tanımlı değil"


# --- Deterministik seçim -----------------------------------------------------

POOL = [f"t{i:03d}" for i in range(80)]


def test_ayni_seed_ayni_secim():
    assert deterministic_select(POOL, 30, 20260727) == deterministic_select(POOL, 30, 20260727)


def test_farkli_seed_farkli_secim():
    assert deterministic_select(POOL, 30, 20260727) != deterministic_select(POOL, 30, 1)


def test_havuz_sirasi_secimi_etkilemez():
    # Dosya okuma sırası gibi tesadüfi bir şey seçimi değiştirmemeli.
    assert deterministic_select(POOL, 30, 20260727) == deterministic_select(list(reversed(POOL)), 30, 20260727)


def test_secim_sayisi_ve_sirasi():
    secim = deterministic_select(POOL, 30, 20260727)
    assert len(secim) == 30
    assert len(set(secim)) == 30
    assert secim == sorted(secim)
    assert set(secim) <= set(POOL)


def test_havuz_yetersizse_hata():
    with pytest.raises(ValueError, match="uygun görev yetersiz"):
        deterministic_select(POOL[:10], 30, 20260727)


# --- İçerik hash'i -----------------------------------------------------------

def test_icerik_hashi_kararli_ve_degisime_duyarli():
    task = {"task_id": "t1", "prompt": "p", "base_test_code": "b"}
    h = task_content_hash(task)
    assert h == task_content_hash(dict(reversed(list(task.items()))))  # sıra bağımsız
    assert h != task_content_hash({**task, "prompt": "farkli"})
