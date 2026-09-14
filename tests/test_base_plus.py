"""Base/Plus ayrımı testleri (EXPERIMENT_PROTOCOL.md §5).

Birincil başarı metriği plus_pass, ikincil base_pass. İkisinin AYRI koşup ayrı
alanlara yazılması RQ5'in "base'i geçip Plus'ta elenen çözüm oranı" sorusunun
ön koşulu — tek birleşik koşu bu bilgiyi geri alınamaz şekilde yok ederdi.

Held-out görev dosyaları üretilmişse ayrıca gerçek referans çözümlerin hem base
hem Plus testlerini geçtiği doğrulanır (ölçüm aletinin sürekli regresyon
güvencesi; pilot settekiyle aynı disiplin).
"""

import pytest

from config import HELDOUT_COUNTS, HELDOUT_TASKS_DIR
from eval.harness import evaluate_base_plus, has_base_plus, load_all_tasks
from eval.task_selection import build_test_code

# Sentetik görev: base yalnız pozitif girdi, plus sıfır/negatifi de kapsıyor.
# "Sadece base'i geçen" aday bu ayrımı gösterebilsin diye seçildi.
BASE_PLUS_TASK = {
    "task_id": "sentetik",
    "entry_point": "isaret",
    "base_test_code": build_test_code([([3], 1), ([7], 1)], 0),
    "plus_test_code": build_test_code([([3], 1), ([7], 1), ([0], 0), ([-2], -1)], 0),
}

TAM_DOGRU = "def isaret(x):\n    return (x > 0) - (x < 0)\n"
SADECE_BASE = "def isaret(x):\n    return 1\n"          # negatif/sıfırda yanılır
HIC_GECMEZ = "def isaret(x):\n    return 'yanlis'\n"


def test_tam_dogru_cozum_her_ikisini_gecer():
    r = evaluate_base_plus(BASE_PLUS_TASK, TAM_DOGRU)
    assert r["base_pass"] is True
    assert r["plus_pass"] is True
    assert r["base_plus_available"] is True
    assert r["plus_skipped_base_failed"] is False


def test_base_gecip_plusta_elenen_cozum_ayirt_edilir():
    # RQ5'in keşifsel sorusunun ölçülebilir olmasının kanıtı.
    r = evaluate_base_plus(BASE_PLUS_TASK, SADECE_BASE)
    assert r["base_pass"] is True
    assert r["plus_pass"] is False
    assert r["plus_error_class"] == "assertion"
    assert r["plus_status"] == "failed"


def test_base_duserse_plus_kosulmaz_ama_alanlar_tutarli():
    # Plus base'i kapsadığı için base düştüğünde plus zorunlu olarak düşer;
    # gereksiz sandbox koşusu yapılmaz, plus alanları base'den KOPYALANIR
    # (uydurulmaz).
    r = evaluate_base_plus(BASE_PLUS_TASK, HIC_GECMEZ)
    assert r["base_pass"] is False
    assert r["plus_pass"] is False
    assert r["plus_skipped_base_failed"] is True
    assert r["plus_status"] == r["base_status"]
    assert r["plus_error_class"] == r["base_error_class"]


def test_duz_alanlar_plus_sonucunu_yansitir():
    # Birincil metrik plus olduğu için eski şemadaki düz status/error_class
    # alanları PLUS sonucunu göstermeli (mevcut özet/MAST kodu bunları okuyor).
    r = evaluate_base_plus(BASE_PLUS_TASK, SADECE_BASE)
    assert r["status"] == r["plus_status"]
    assert r["error_class"] == r["plus_error_class"]


def test_pilot_gorevde_tek_kosu_yapilir_ve_sema_ayni_kalir():
    # Base/plus taşımayan pilot görevlerde analiz katmanı yine tek şema görür.
    pilot = {
        "task_id": "dummy",
        "entry_point": "topla",
        "test_code": "def check(candidate):\n    assert candidate(2, 3) == 5\n",
    }
    assert not has_base_plus(pilot)
    r = evaluate_base_plus(pilot, "def topla(a, b):\n    return a + b\n")
    assert r["base_plus_available"] is False
    assert r["base_pass"] is True and r["plus_pass"] is True


# --- Gerçek held-out seti (üretilmişse) --------------------------------------

_heldout = load_all_tasks("heldout") if HELDOUT_TASKS_DIR.exists() else []
_heldout_var = pytest.mark.skipif(
    not _heldout, reason="held-out seti henüz üretilmedi (scripts/fetch_evalplus.py)")


@_heldout_var
def test_heldout_seti_beklenen_buyuklukte():
    assert len(_heldout) == sum(HELDOUT_COUNTS.values())
    for source, count in HELDOUT_COUNTS.items():
        assert sum(t["source"] == source for t in _heldout) == count


@_heldout_var
def test_secim_manifesti_gorev_sanilmaz():
    # Yardımcı belgeler ("_" önekli) görev listesine sızarsa görev sayısı
    # sessizce şişer ve koşu bozulur.
    assert (HELDOUT_TASKS_DIR / "_selection_manifest.json").exists()
    assert all("task_id" in t and "entry_point" in t for t in _heldout)


@_heldout_var
def test_heldout_pilot_ile_kesismiyor():
    # Ana istatistik pilot görevlerden arındırılmış olmalı (§5.1).
    pilot_ids = {t["task_id"] for t in load_all_tasks("pilot")}
    assert not pilot_ids & {t["task_id"] for t in _heldout}


@_heldout_var
@pytest.mark.parametrize("task", _heldout, ids=[t["task_id"] for t in _heldout])
def test_heldout_referans_cozum_base_ve_plus_gecer(task):
    # Ölçüm aletinin sürekli doğrulaması: referans çözüm kendi testlerini
    # geçmiyorsa o görevden gelen hiçbir sonuca güvenilemez.
    r = evaluate_base_plus(task, task["reference_solution"])
    assert r["base_pass"] is True, f"base düştü: {r['base_traceback']}"
    assert r["plus_pass"] is True, f"plus düştü: {r['plus_traceback']}"
