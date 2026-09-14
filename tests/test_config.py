"""config.py birim testleri — model rolleri ve takma ad çözümlemesi.

Bu testler "yanlış model kullanıldı" hatasını erken yakalamak içindir: model
kimliği bu deneyde bir yapılandırma ayrıntısı değil, RQ4'ün (model ailesi
sağlamlığı) bağımsız değişkeni. Yanlış slug'la toplanan veri geri
döndürülemez (EXPERIMENT_PROTOCOL.md §4).
"""

import pytest

import config


# --- Exact slug'lar (EXPERIMENT_PROTOCOL §4) ---------------------------------

def test_dort_rolun_exact_slugu():
    # Slug'lar normatif tablodan birebir kopyalanır; "yakın" bir slug (ör.
    # gemini-2.5 vs 3.5) sessizce başka bir modelin verisini toplar.
    assert config.MODEL_MAIN == "openrouter/google/gemini-3.5-flash-lite"
    assert config.MODEL_SECONDARY == "openrouter/deepseek/deepseek-v4-flash"
    assert config.MODEL_JUDGE_EXTERNAL == "openrouter/minimax/minimax-m3"
    assert config.MODEL_ADJUDICATOR == "openrouter/x-ai/grok-4.3"


def test_iki_uretici_farkli_ailelerden():
    # Aynı aileden iki model, RQ4'ün "etki modelden bağımsız mı" sorusunu
    # cevaplayamaz -- çapraz-model replikasyonun anlamı kalmaz.
    assert config.MODEL_MAIN != config.MODEL_SECONDARY
    assert "google" in config.MODEL_MAIN
    assert "deepseek" in config.MODEL_SECONDARY


def test_pilot_model_replikasyon_modeliyle_ayni():
    # BİLİNÇLİ eşitlik (§4): pilot hattı ile ikinci üretici hattı aynı
    # taşıma/rota davranışını paylaşsın; ayrı bir pilot modeli, pilotta
    # görülmeyen bir rota sorununu ana koşuya taşırdı.
    assert config.MODEL_PILOT == config.MODEL_SECONDARY


# --- Judge paneli ve adjudicator (§9.1-§9.2) ---------------------------------

def test_judge_paneli_tam_olarak_deepseek_gemini_minimax():
    # §9.1: birbirinden habersiz ÜÇ judge; aynı model iki kez sayılırsa
    # dış-konsensüs ölçüsü anlamsızlaşır.
    assert len(config.MODEL_JUDGES) == 3
    assert len(set(config.MODEL_JUDGES)) == 3
    assert set(config.MODEL_JUDGES) == {
        config.MODEL_SECONDARY, config.MODEL_MAIN, config.MODEL_JUDGE_EXTERNAL}


def test_minimax_uretici_degil():
    # §4: MiniMax held-out kod ÜRETMEZ; yalnız üretici-dışı judge'dır.
    assert config.MODEL_JUDGE_EXTERNAL not in (config.MODEL_MAIN, config.MODEL_SECONDARY)
    assert config.MODEL_JUDGE_EXTERNAL != config.MODEL_PILOT


def test_adjudicator_ne_judge_ne_uretici():
    # §9.2: adjudicator, iki dış judge'ın ayrıştığı kayıtları DEĞERLENDİREN
    # panel-dışı taraf; panelde olursa kendi etiketini hakemlemiş olur,
    # üretici olursa kendi çıktısının hatasına karar vermiş olur.
    assert config.MODEL_ADJUDICATOR not in config.MODEL_JUDGES
    assert config.MODEL_ADJUDICATOR not in (config.MODEL_MAIN, config.MODEL_SECONDARY)


def test_varsayilan_tek_judge_panel_uyesi():
    assert config.MODEL_JUDGE in config.MODEL_JUDGES


def test_rol_dogrulamasi_calisma_zamaninda_da_kosuyor():
    # `assert` yerine ValueError: `python -O` altında assert'ler devre dışı
    # kalır ve yanlış bir kadro sessizce ana koşuya girebilirdi.
    config.validate_model_roles()  # mevcut kadro geçerli
    with pytest.raises(ValueError):
        _hatali_kadro_dogrula()


def _hatali_kadro_dogrula():
    """Adjudicator'ı panele koyup doğrulamayı yeniden çalıştırır (geri alır)."""
    orijinal = config.MODEL_ADJUDICATOR
    config.MODEL_ADJUDICATOR = config.MODEL_JUDGES[0]
    try:
        config.validate_model_roles()
    finally:
        config.MODEL_ADJUDICATOR = orijinal


# --- Takma adlar -------------------------------------------------------------

def test_takma_adlar_tam_sluglara_cozulur():
    assert config.resolve_model("main") == config.MODEL_MAIN
    assert config.resolve_model("secondary") == config.MODEL_SECONDARY
    assert config.resolve_model("dev") == config.MODEL_PILOT
    assert config.resolve_model("pilot") == config.MODEL_PILOT


def test_dev_ve_secondary_bilincli_olarak_ayni_modele_cozulur():
    assert config.resolve_model("dev") == config.resolve_model("secondary")
    assert config.resolve_model("dev") == config.MODEL_SECONDARY


def test_upgrade_takma_adi_yok():
    # Koşullu yükseltme adayı 2026-07-29'da kadrodan çıkarıldı; takma ad
    # kalsaydı `--model upgrade` sessizce çözülmeyen bir slug'a düşerdi.
    assert "upgrade" not in config.MODEL_ALIASES
    assert not hasattr(config, "MODEL_UPGRADE_CANDIDATE")


def test_judge_external_uretici_takma_adi_degil():
    # MiniMax bir üretici takma adı olarak sunulursa yanlışlıkla held-out kod
    # üretebilir; yalnız sağlık scriptinin rol konfigürasyonunda bulunur.
    assert config.MODEL_JUDGE_EXTERNAL not in config.MODEL_ALIASES.values()
    assert "judge_external" not in config.MODEL_ALIASES


def test_takma_ad_olmayan_slug_degistirilmeden_gecer():
    # Tanımlı olmayan bir modelle tek seferlik deneme yapılabilsin diye.
    assert config.resolve_model("openrouter/baska/model") == "openrouter/baska/model"


def test_alias_yardim_metni_tek_kaynaktan_turer():
    # Statik "main|secondary|dev" kopyaları kadro değişince sessizce eskir.
    yardim = config.model_alias_help()
    for alias in config.MODEL_ALIASES:
        assert alias in yardim
    assert "upgrade" not in yardim


# --- Held-out üretici kapısı (§4) --------------------------------------------

def test_uretici_listesi_tam_olarak_iki_model():
    assert config.MODEL_PRODUCERS == (config.MODEL_MAIN, config.MODEL_SECONDARY)
    assert len(set(config.MODEL_PRODUCERS)) == 2


def test_judge_ve_adjudicator_uretici_listesinde_yok():
    assert config.MODEL_JUDGE_EXTERNAL not in config.MODEL_PRODUCERS
    assert config.MODEL_ADJUDICATOR not in config.MODEL_PRODUCERS


def test_heldout_yalnizca_uretici_modelleri_kabul_eder():
    for model in (config.MODEL_MAIN, config.MODEL_SECONDARY):
        assert config.validate_model_for_task_set(model, "heldout") == model


def test_heldout_takma_adlar_cozulmus_slug_uzerinden_degerlendirilir():
    # `dev` DeepSeek'e çözülür ve DeepSeek geçerli bir üreticidir: kapı adı
    # değil KİMLİĞİ denetler.
    assert config.validate_model_for_task_set("main", "heldout") == config.MODEL_MAIN
    assert config.validate_model_for_task_set("dev", "heldout") == config.MODEL_SECONDARY


def test_heldout_minimax_reddedilir():
    # MiniMax'in held-out kod üretmemesi, MAST panelindeki üretici-dışılık
    # iddiasının tek dayanağı. Tam slug alias tablosunu atlar -> kapı burada.
    with pytest.raises(ValueError, match="held-out"):
        config.validate_model_for_task_set(config.MODEL_JUDGE_EXTERNAL, "heldout")


def test_heldout_grok_reddedilir():
    # Adjudicator kendi çıktısının hakemi olamaz.
    with pytest.raises(ValueError, match="held-out"):
        config.validate_model_for_task_set(config.MODEL_ADJUDICATOR, "heldout")


def test_heldout_rastgele_slug_reddedilir():
    with pytest.raises(ValueError, match="held-out"):
        config.validate_model_for_task_set("openrouter/baska/model", "heldout")


def test_pilotta_rastgele_tam_slug_kabul_edilir():
    # Uyumluluk smoke'ları tanımsız bir slug'la da yapılabilmeli.
    assert config.validate_model_for_task_set(
        "openrouter/baska/model", "pilot") == "openrouter/baska/model"
    assert config.validate_model_for_task_set(
        config.MODEL_JUDGE_EXTERNAL, "pilot") == config.MODEL_JUDGE_EXTERNAL


# --- Sağlık kapısı konfigürasyonu (§4) ---------------------------------------

def test_saglik_anahtarlari_tam_olarak_uc_rol_ve_sirali():
    # Sıra bilinçlidir: karşı-dengelenmiş blok rotasyonunun başlangıç sırasını
    # belirler.
    assert list(config.HEALTH_GATE_MODELS) == ["main", "secondary", "judge_external"]


def test_saglik_modelleri_dogru_rollere_cozulur():
    assert config.HEALTH_GATE_MODELS["main"] == config.MODEL_MAIN
    assert config.HEALTH_GATE_MODELS["secondary"] == config.MODEL_SECONDARY
    assert config.HEALTH_GATE_MODELS["judge_external"] == config.MODEL_JUDGE_EXTERNAL


def test_dev_ve_upgrade_saglik_anahtari_degil():
    # `dev` ile `secondary` aynı slug'a çözülür; `dev` üzerinden ayrı bir rota
    # ölçmeye çalışmak eski mimariyi yeniden üretirdi.
    assert "dev" not in config.HEALTH_GATE_MODELS
    assert "pilot" not in config.HEALTH_GATE_MODELS
    assert "upgrade" not in config.HEALTH_GATE_MODELS


# --- Ortak üretim parametreleri ----------------------------------------------

def test_ortak_uretim_parametreleri_tanimli():
    # agents/llm.py bunları TEK noktadan uygular; eksik/None olmaları
    # sessizce sağlayıcı varsayılanına düşmek demek olurdu.
    assert isinstance(config.MAX_OUTPUT_TOKENS, int) and config.MAX_OUTPUT_TOKENS > 0
    # [2026-07-30] Gemini 3.5 Flash Lite endpoint'i reasoning'i ZORUNLU kılıyor
    # (p1_compat_20260730_v1: 12/12 HTTP 400). Ortak ayar AÇIK; model/rol/kol
    # bazlı istisna YOK — kısmi uygulama kollar arası gizli parametre farkı olurdu.
    assert config.REASONING_CONFIG == {"enabled": True}
    assert config.OPENROUTER_PROVIDER_ROUTING["require_parameters"] is True


def test_minimax_routing_istisnasi_ucuncu_judge_kimligine_bagli():
    # MiniMax kendi resmi sağlayıcısına sabit (varyans kaynağı kapatıldı).
    minimax = config.provider_routing_for(config.MODEL_JUDGE_EXTERNAL)
    assert minimax["order"] == ["minimax"]
    assert minimax["allow_fallbacks"] is False
    assert minimax["require_parameters"] is True  # json_object garantisi korunuyor


def test_deepseek_minimax_rotasina_zorlanmaz():
    # REGRESYON: istisna eskiden `MODEL_SECONDARY` anahtarındaydı ve o sabit
    # MiniMax'ti. Replikasyon modeli DeepSeek'e çevrilirken satır olduğu gibi
    # bırakılsaydı DeepSeek'in BÜTÜN çağrıları MiniMax rotasına giderdi.
    deepseek = config.provider_routing_for(config.MODEL_SECONDARY)
    assert deepseek.get("order") != ["minimax"]
    assert deepseek is config.OPENROUTER_PROVIDER_ROUTING


def test_uretici_ve_adjudicator_rotalari_varsayilan():
    # Gemini/DeepSeek/Grok için exact provider veya order TAHMİN EDİLMEZ.
    for model in (config.MODEL_MAIN, config.MODEL_SECONDARY, config.MODEL_ADJUDICATOR):
        assert config.provider_routing_for(model) is config.OPENROUTER_PROVIDER_ROUTING


def test_sema_surumleri_tanimli():
    assert config.RESULT_SCHEMA_VERSION
    assert config.LLM_CALL_SCHEMA_VERSION
    assert config.SELF_CONSISTENCY_SCHEMA_VERSION
    assert config.MAST_SCHEMA_VERSION


def test_mast_karar_kurali_ve_semasi_leave_self_out():
    # Şema sürümü, 2.0 (üçlü çoğunluk) artefaktlarının yeni karar sistemine
    # sessizce girmesini engeller; karar kuralı sürümü ise alan listesi
    # değişmeden karar semantiğinin değişebileceğini kayda geçirir.
    assert config.MAST_SCHEMA_VERSION == "3.2"
    assert config.MAST_DECISION_RULE_VERSION == "leave_self_out_v1"


def test_panel_hash_surumu_karar_kuralindan_ayri():
    # Hangi alanların hangi sırada hash'lendiği, karar semantiği hiç değişmeden
    # de değişebilir; tek sürümle izlenirse eski/yeni hash'ler sessizce
    # karşılaştırılamaz hale gelir.
    assert config.MAST_PANEL_HASH_VERSION == "dual_input_v1"
    assert config.MAST_PANEL_HASH_VERSION != config.MAST_DECISION_RULE_VERSION
    assert config.MAST_PANEL_HASH_VERSION != config.MAST_SCHEMA_VERSION


def test_sema_surumleri_bagimsiz_artiyor():
    # Her sürüm AYRI bir tüketici sözleşmesidir ve bağımsız artar: MAST 3.2,
    # insan 2.0, çağrı 2.1 (provenance alanları), analiz 2.2 (2.1'de uzun-kuyruk
    # alanları n/p95/observed_max, 2.2'de RQ5 `base_plus_attrition` bloğu).
    # SONUÇ şeması bunların hiçbirinde değişmedi — 2.0'da kaldı, çünkü koşu
    # kaydının alanları aynı.
    assert config.RESULT_SCHEMA_VERSION == "2.0"
    assert config.LLM_CALL_SCHEMA_VERSION == "2.1"
    assert config.SELF_CONSISTENCY_SCHEMA_VERSION == "1.0"
    assert config.ANALYSIS_SCHEMA_VERSION == "2.2"
    # RQ5/MAST dağılımı AYRI tüketici modülleridir ve kendi sürümlerini taşır:
    # ana analiz sözleşmesi değişmeden bunlar değişebilir (ve tersi).
    assert config.RQ5_SCHEMA_VERSION == "1.0"
    assert config.MAST_DISTRIBUTION_SCHEMA_VERSION == "1.0"
    assert config.RQ5_PRIMARY_ARM == config.ARM_BASELINE
    # İnsan hattı 2.1: self-judge nullable (kapı "tam üçlü panel" yerine "dış
    # karar kurulabiliyor mu"); saklanan veri şekli VE arayüz birlikte değişti.
    assert config.MAST_HUMAN_SCHEMA_VERSION == "2.1"
    assert config.MAST_HUMAN_APP_VERSION == "2.1"
