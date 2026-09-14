"""Merkezi yapılandırma: model adları, yollar, zaman ve hız sınırları.

Deney parametreleri tek yerden yönetilir; agent/pipeline kodu sabit değer içermez.
"""

from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).parent
load_dotenv(ROOT / ".env")

# --- Görev setleri (EXPERIMENT_PROTOCOL.md §5) ---
# İKİ SET AYRI DİZİNDE, bilinçli: pilot görevlerin ana istatistiğe sızması
# (ya da tersi) tek bir yanlış load_all_tasks() çağrısıyla olabilirdi.
TASKS_DIR = ROOT / "tasks"                  # development/pilot seti (20 görev)
HELDOUT_TASKS_DIR = ROOT / "tasks_heldout"  # ana deney seti (50 EvalPlus görevi)
PILOT_TASK_SET = "pilot"
HELDOUT_TASK_SET = "heldout"
TASK_SETS = {PILOT_TASK_SET: TASKS_DIR, HELDOUT_TASK_SET: HELDOUT_TASKS_DIR}

# Held-out seçim parametreleri — seçim yapıldıktan sonra DEĞİŞTİRİLEMEZ
# (değişirse farklı bir görev seti demektir; seçim manifesti bunu korur).
HELDOUT_SELECTION_SEED = 20260727
HELDOUT_COUNTS = {"humanevalplus": 30, "mbppplus": 20}

LOGS_DIR = ROOT / "logs"

# --- Modeller (LiteLLM adlandırması) ---
# Model rolleri EXPERIMENT_PROTOCOL.md §4'te tanımlanmıştır.
# karar); buradaki sabitler o tablonun tek uygulama karşılığıdır. Ana koşu
# öncesi (P1) exact slug + provider routing + temperature + max_tokens +
# reasoning ayarı manifestte dondurulur.
#
# Birincil üretici model — EXPERIMENT_PROTOCOL.md §8'e göre analiz bu
# modelin verisi üzerinde yapılır. Tam tasarım: 50 held-out görev × 3 tekrar ×
# 4 kol = 600 arm-run.
MODEL_MAIN = "openrouter/google/gemini-3.5-flash-lite"
# Tam çapraz-model replikasyon modeli — MODEL_MAIN ile AYNI 50×3×4 tasarımda tam
# çalışır (§4). Ayrı bir aileden olması kasıtlı: RQ4'ün "etki modelden bağımsız
# mı" sorusu için. Sonuçlar MODEL_MAIN ile HAVUZLANMAZ; aynı analiz ayrıca
# uygulanır (§8.4).
#
# Provenance notu (EXPERIMENT_PROTOCOL.md §4):
# DeepSeek repoda düşük hacimli bir API smoke'unda çağrıldı; o smoke YALNIZ
# taşıma/runner/retry/provenance hattını doğruladı. Model çıktıları mimari veya
# prompt ayarlamak için kullanılmadı ve held-out görevleri görmedi — bu yüzden
# "development-exposed" değil, tam çapraz-model replikasyon üreticisidir.
# 2026-07-20 canlı kontrol: 1.05M context, $0.098/$0.196 per 1M tok. litellm
# fiyat haritası bu modeli tanımıyor → maliyet OpenRouter usage accounting'den
# okunur (aşağıda OPENROUTER_USAGE_ACCOUNTING).
MODEL_SECONDARY = "openrouter/deepseek/deepseek-v4-flash"
# Development/pilot koşuları. BİLİNÇLİ olarak replikasyon modeliyle AYNI slug:
# pilot hattı ile ikinci üretici hattı aynı taşıma/rota davranışını paylaşsın
# diye (ayrı bir pilot modeli, pilotta görülmeyen bir rota sorununu ana koşuya
# taşırdı). Held-out görev seti yine yalnız --task-set heldout ile açılır.
MODEL_PILOT = MODEL_SECONDARY

# Held-out veri ÜRETMESİNE izin verilen tam liste (§4). Judge/adjudicator
# modelleri buraya GİREMEZ: MiniMax "held-out kod üretmez" kuralı, MAST
# panelindeki üretici-dışılık iddiasının tek dayanağıdır; Grok'un üretmesi ise
# adjudicator'ı kendi çıktısının hakemi yapardı.
MODEL_PRODUCERS = (MODEL_MAIN, MODEL_SECONDARY)

# --- MAST etiketleme (§9.1 bağımsız AI paneli) ---
# Üretici-DIŞI üçüncü judge. Held-out kod ÜRETMEZ; yalnız etiketler (§4).
# Bir üretici takma adı olarak sunulmaz (bkz. MODEL_ALIASES).
MODEL_JUDGE_EXTERNAL = "openrouter/minimax/minimax-m3"

# Her başarısız kayıt, birbirinden habersiz ÜÇ judge tarafından etiketlenir.
# Üretici model kimliği judge prompt'undan gizlenir (§4). Panelde iki üretici +
# bir üretici-dışı model bulunur; kaydı üreten modelin kendi etiketi tanısal
# olarak saklanır ama karara oy VERMEZ (leave-self-out, §9.1 — karar hattının
# uygulaması P1 Parça 2-4 kapsamındadır).
MODEL_JUDGES = (
    MODEL_SECONDARY,       # DeepSeek V4 Flash
    MODEL_MAIN,            # Gemini 3.5 Flash Lite
    MODEL_JUDGE_EXTERNAL,  # MiniMax M3
)
# İki DIŞ judge ayrışırsa (§9.2) panel-dışı adjudicator çalışır. xAI ailesinden
# seçildi: ne üretici ne panel üyesi — kendi etiketini hakemlemesi yapısal olarak
# imkânsız. Çıktısı AYRI alana yazılır, ilk etiketlerin üzerine yazılmaz; nihai
# karar yine insanlarda (§9.5).
MODEL_ADJUDICATOR = "openrouter/x-ai/grok-4.3"
# Tek-judge (pilot/debug) yolu — eval/mast_labels.py'nin mevcut varsayılanı.
# Ana koşuda MODEL_JUDGES paneli kullanılır.
MODEL_JUDGE = MODEL_JUDGES[0]


def validate_model_roles() -> None:
    """Model rol invariantlarını içe aktarma sırasında doğrular (§4, §9.1-§9.2).

    `assert` yerine açık ValueError: `python -O` altında assert'ler devre dışı
    kalır ve yanlış bir model kadrosu sessizce ana koşuya girebilirdi. Yanlış
    slug'la toplanan veri geri döndürülemez.
    """
    if MODEL_MAIN == MODEL_SECONDARY:
        raise ValueError("MODEL_MAIN ve MODEL_SECONDARY farklı olmalı (RQ4 çapraz-model)")
    if MODEL_PILOT != MODEL_SECONDARY:
        raise ValueError("MODEL_PILOT bilinçli olarak MODEL_SECONDARY ile aynı olmalı")
    if len(MODEL_JUDGES) != 3 or len(set(MODEL_JUDGES)) != 3:
        raise ValueError("MAST paneli tam olarak üç FARKLI judge içermeli (§9.1)")
    for rol, model in (("MODEL_MAIN", MODEL_MAIN), ("MODEL_SECONDARY", MODEL_SECONDARY),
                       ("MODEL_JUDGE_EXTERNAL", MODEL_JUDGE_EXTERNAL)):
        if model not in MODEL_JUDGES:
            raise ValueError(f"{rol} judge panelinde bulunmalı (§9.1)")
    if MODEL_JUDGE_EXTERNAL in (MODEL_MAIN, MODEL_SECONDARY):
        raise ValueError("MODEL_JUDGE_EXTERNAL bir üretici model olamaz (§4)")
    if MODEL_ADJUDICATOR in MODEL_JUDGES or MODEL_ADJUDICATOR in (MODEL_MAIN, MODEL_SECONDARY):
        raise ValueError("MODEL_ADJUDICATOR panel-dışı ve üretici-dışı olmalı (§9.2)")
    if len(MODEL_PRODUCERS) != 2 or len(set(MODEL_PRODUCERS)) != 2:
        raise ValueError("MODEL_PRODUCERS tam olarak iki FARKLI üretici içermeli (§4)")
    if set(MODEL_PRODUCERS) != {MODEL_MAIN, MODEL_SECONDARY}:
        raise ValueError("MODEL_PRODUCERS, MODEL_MAIN + MODEL_SECONDARY ile birebir eşleşmeli")
    if MODEL_JUDGE_EXTERNAL in MODEL_PRODUCERS or MODEL_ADJUDICATOR in MODEL_PRODUCERS:
        raise ValueError("judge/adjudicator modelleri held-out veri ÜRETEMEZ (§4, §9.2)")


validate_model_roles()

# Runner/CLI kısayolları: ana deneyde model ASLA örtük varsayılandan gelmez
# (aşağıdaki resolve_model + agents/llm.py'deki experiment guard), fakat uzun
# slug'ları elle yazmak yazım hatası riski taşır. Takma adlar bu riski kapatır.
# Yalnız ÜRETİCİ/debug rolleri buradadır: `judge_external` bilerek yok — MiniMax
# bir üretici takma adı olarak sunulursa yanlışlıkla held-out kod üretebilir.
# `dev` ve `secondary` aynı slug'a çözülür; bu bilinçli (bkz. MODEL_PILOT).
MODEL_ALIASES = {
    "main": MODEL_MAIN,
    "secondary": MODEL_SECONDARY,
    "dev": MODEL_PILOT,
    "pilot": MODEL_PILOT,
}


def resolve_model(name: str) -> str:
    """Takma adı (main/secondary/dev/pilot) tam slug'a çevirir.

    Takma ad değilse dokunmadan döndürür — tam LiteLLM slug'ı doğrudan
    verilebilsin diye (ör. hiç tanımlanmamış bir modelle tek seferlik deneme).
    """
    return MODEL_ALIASES.get(name, name)


def model_alias_help() -> str:
    """CLI yardım metni — takma ad listesi TEK kaynaktan türetilir.

    Elle yazılmış alias listeleri kadro değiştiğinde sessizce eskir: 2026-07-29
    kadro değişikliğinde kaldırılan bir takma ad, dört ayrı CLI'nın yardım
    metninde kalmıştı. Tek kaynak bu sınıfı hatayı yapısal olarak kapatır.
    """
    return f"takma ad ({'|'.join(MODEL_ALIASES)}) veya tam LiteLLM slug'ı"


def validate_model_for_task_set(model: str, task_set: str) -> str:
    """Modeli çözer ve görev setine göre kapıdan geçirir; çözülmüş slug'ı döner.

    Held-out sette YALNIZ `MODEL_PRODUCERS` kabul edilir. Alias katmanı tek
    başına yeterli DEĞİL: `--model openrouter/minimax/minimax-m3` yazmak alias
    tablosunu tamamen atlar ve "MiniMax held-out kod üretmez" iddiasını sessizce
    çürütürdü (§4). Karar ÇÖZÜLMÜŞ slug üzerinden verilir — `dev` de DeepSeek'e
    çözüldüğü için held-out'ta teknik olarak geçerlidir; kapı adı değil kimliği
    denetler.

    Pilot/development sette kısıt YOKTUR: tek seferlik uyumluluk smoke'ları
    tanımsız bir slug'la da yapılabilmelidir.
    """
    resolved = resolve_model(model)
    if task_set == HELDOUT_TASK_SET and resolved not in MODEL_PRODUCERS:
        raise ValueError(
            f"held-out görev setinde yalnız üretici modeller çalıştırılabilir: "
            f"{list(MODEL_PRODUCERS)}. Verilen: {model!r} -> {resolved!r}. "
            f"Judge/adjudicator modelleri ve tanımsız slug'lar held-out veri üretemez "
            f"(EXPERIMENT_PROTOCOL.md §4)."
        )
    return resolved

# --- Sandbox ---
SANDBOX_TIMEOUT_S = 10
# Plus testleri base'e göre çok daha fazla girdi içerir (HumanEval+'ta ~1000'e
# kadar) -> ayrı, daha geniş bir zaman sınırı.
PLUS_TIMEOUT_S = 30
# Uygunluk şartı: REFERANS çözüm plus testlerini bu süre içinde bitirmeli.
# PLUS_TIMEOUT_S'ten çok daha küçük tutulur ki doğru ama daha yavaş bir aday
# kod yalnız hız yüzünden elenmesin (~10x headroom). Bunu aşan görev reddedilir
# ve nedeni seçim manifestine yazılır.
REFERENCE_PLUS_BUDGET_S = 3.0
SANDBOX_OUTPUT_LIMIT_BYTES = 10_000  # stdout/stderr her biri için üst sınır
TRACEBACK_MAX_LINES = 30             # JSONL'e yazılan kısaltılmış traceback
# Aday koda kalıtılan ortam değişkeni izin listesi (ör. ANTHROPIC_API_KEY/
# OPENROUTER_API_KEY sızmasın diye). Sadece Windows/CPython'ın kendi
# başlatması için gerekli olanlar — "güvenlik" amaçlı genişletilmiş bir
# liste DEĞİL (bkz. eval/sandbox.py docstring, açık risklerin listesi).
SANDBOX_ENV_ALLOWLIST = ("SystemRoot", "SystemDrive", "TEMP", "TMP")

# --- Deney kolları (EXPERIMENT_PROTOCOL.md §3) ---
# Kol adları tek kaynakta: yazım hatası sessizce yeni bir "kol" uydurmasın.
ARM_BASELINE = "baseline"
ARM_NAIVE = "naive"
ARM_STRUCTURED = "structured_no_validation"
ARM_CONTRACT = "contract"
ALL_ARMS = [ARM_BASELINE, ARM_NAIVE, ARM_STRUCTURED, ARM_CONTRACT]

# Graf üzerinden koşan kollar (baseline tek çağrılık ayrı hat).
GRAPH_MODES = (ARM_NAIVE, ARM_STRUCTURED, ARM_CONTRACT)
# Planner'dan canonical JSON isteyen kollar. Bu ikisi AYNI planner prompt'unu,
# AYNI response_format'ı, AYNI parse yolunu ve AYNI canonical serileştirmeyi
# kullanır — aralarındaki TEK fark validator + sınırlı retry'dır (§3).
STRUCTURED_MODES = (ARM_STRUCTURED, ARM_CONTRACT)

# --- Kol 3: sözleşme kapısı ---
# Planner toplam en fazla bu kadar denenir (ilk deneme + retry'ler); sözleşme
# doğrulaması bu sayı boyunca hep başarısız olursa coder yine de en son
# (geçersiz) planla devam eder — bu da MAST'ta Kol 3'e özgü bir hata
# kategorisi olarak loglanacak veri üretir.
MAX_PLANNER_ATTEMPTS = 3

# --- LLM katmanı ---
LLM_NUM_RETRIES = 3        # LiteLLM exponential backoff ile (transport katmanı)
LLM_MIN_INTERVAL_S = 3.0   # istekler arası bekleme (OpenRouter free ~20 istek/dk)
LLM_TIMEOUT_S = 120

# Sağlayıcı-hatası retry'ı (LiteLLM'in transport retry'ından FARKLI katman).
# 2026-07-27 teşhisi: OpenRouter upstream hatasını HTTP 200 GÖVDESİNE gömerek
# döndürüyor — provider_specific_fields.error = {"code": 429, ... upstream rate
# limit}, native_finish_reason="error", usage.completion_tokens=0, içerik
# JSON'un ortasında kesik. LiteLLM native 'error'ı 'stop'a eşlediği için istisna
# ATILMIYOR ve transport retry'ı devreye girmiyordu. Hata çıplak çağrıda da
# görüldü -> bizim yapılandırmamızdan kaynaklanmıyor.
# Gözlenen oran ZAMANA GÖRE KÜMELENİYOR (upstream paylaşımlı kota): bir pencerede
# 4/9 ve 7/25, sonraki pencerelerde 0/7 ve 0/12. Bu yüzden tek bir "oran"
# dondurulamaz; ana koşu öncesi scripts/provider_health.py ile ölçülür.
# Bozuk yanıt agents/llm.py'de tespit edilip yeniden denenir; tükenirse istisna
# atılır (runner run_error yazar, analizde ölçüm sayılmaz).
LLM_PROVIDER_ERROR_RETRIES = 3
LLM_PROVIDER_ERROR_BACKOFF_S = 2.0

# --- Model × sağlayıcı sağlık kapısı (P1, ana koşudan ÖNCE) ---
# Ön-kayıtlı operasyonel eşikler. Evrensel standart DEĞİL; bu projeye özgü,
# sonuçlar görülmeden dondurulmuş kabul kriterleridir. Ölçüm ana koşunun
# GERÇEK temposuyla (aynı throttle) yapılır; tempo hata oranını değiştirir.
#
# Kapıya giren üç OPERASYONEL rota (§4): üç üretici ADAYI değil, deneyde fiilen
# yüksek hacimle kullanılacak rotalar. Sıra bilinçlidir (main → secondary →
# judge_external) ve karşı-dengelenmiş blok rotasyonunun başlangıç sırasını
# belirler. `dev` anahtarı YOK: `dev` ile `secondary` aynı DeepSeek slug'ına
# çözülür, ayrı bir sağlık anahtarı olarak durması eski mimariyi yeniden
# üretirdi. Roller bağımsızdır — biri düşerse diğerine sessiz rol devri olmaz
# (main düşerse Gemini held-out koşusu, secondary düşerse DeepSeek held-out
# koşusu, judge_external düşerse MAST etiketleme başlamaz).
HEALTH_GATE_MODELS = {
    "main": MODEL_MAIN,
    "secondary": MODEL_SECONDARY,
    "judge_external": MODEL_JUDGE_EXTERNAL,
}
HEALTH_GATE_CALLS_PER_CONFIG = 100
# Blok başına ardışık çağrı: blok İÇİNDE tek model art arda koşar (ana koşunun
# gerçek temposu), blok SIRASI bloklar arasında döndürülür. Tek tek dönüşümlü
# çağrıda global throttle modeller arasında bölünür ve rate-limit kaynaklı
# arıza olduğundan DÜŞÜK ölçülür.
HEALTH_GATE_BLOCK_SIZE = 10
HEALTH_GATE_PASS_RATE = 0.05        # ham hata <= %5 ve exhaustion yok -> kullanılabilir
HEALTH_GATE_WARN_RATE = 0.15        # %5-15, exhaustion yok -> uyarıyla kullanılabilir
                                    # > %15 veya tekrarlanan exhaustion -> rota/model değişir
DEFAULT_TEMPERATURE = 0.2
SELF_CONSISTENCY_TEMPERATURE = 0.8  # N aday örneklemesi bu sıcaklıkta yapılır
SELF_CONSISTENCY_N = 5              # görev başına aday sayısı

# --- Ortak model parametreleri (BÜTÜN roller ve BÜTÜN kollar için aynı) ---
# İç geçerlilik şartı: kollar arasındaki tek fark iletişim katmanı olmalı.
# Bu yüzden aşağıdakiler agents/llm.py'de TEK noktadan uygulanır; hiçbir agent
# kendi max_tokens/reasoning/routing değerini geçirmez.

# Çıktı üst sınırı. Görevlerimiz tek fonksiyon + JSON plan ölçeğinde; 8192 bol
# bir tavan (cap olduğu için kullanılmadıkça maliyeti yok). Asıl amaç,
# truncation'ın SESSİZCE olmaması: kesilme olursa finish_reason="length" olarak
# loglanır ve analizde tespit edilebilir.
MAX_OUTPUT_TOKENS = 8192

# Reasoning/thinking ayarı — EXPERIMENT_PROTOCOL.md §4: "Reasoning/thinking ayarı
# kollar arasında aynı olacaktır." OpenRouter'ın "reasoning" gövde alanına gider;
# None verilirse alan hiç gönderilmez (sağlayıcı varsayılanı geçerli olur).
#
# DEĞİŞTİRME KOŞULU: bu değer model BAŞARISINA bakılarak değiştirilmez —
# "reasoning açınca skor arttı" türü bir ayar, sonuç görüldükten sonra parametre
# seçmek olurdu. Yalnız teknik bir UYUMLULUK bulgusu (sağlayıcının ayarı
# reddetmesi/yok sayması gibi) tarihli bir kararla değişikliği gerekçelendirir;
# o durumda değer İKİ model ve DÖRT kol için birden değişir (kısmi uygulama iç
# geçerliliği kırar) ve etkilenen uyumluluk smoke'u ile 3 × 100 sağlık kapısı
# baştan koşar.
#
# [2026-07-30] KAPALI (`{"enabled": False}`) → AÇIK. Gerekçe tam olarak yukarıdaki
# değiştirme koşuludur: `p1_compat_20260730_v1` uyumluluk smoke'unda Gemini 3.5
# Flash Lite 12/12 çağrıda HTTP 400 döndürdü — bu endpoint reasoning'i ZORUNLU
# kılıyor, yani kapalı ayarla model hiç çağrılamıyor. Aynı turda Grok 4.3 12/12
# PASS aldı, dolayısıyla arıza altyapıda değil endpoint sözleşmesindeydi. Karar
# BÜTÜN modeller ve BÜTÜN kollar için ortaktır; model/rol/kol bazlı reasoning
# istisnası YOKTUR (kısmi uygulama, kollar arası gizli bir parametre farkı
# yaratıp iç geçerliliği kırardı). Bu bir performans/prompt ayarı DEĞİL, teknik
# endpoint uyumluluğu düzeltmesidir; smoke'un v1 artefaktı kanıt olarak korunur
# ve yeni tur ayrı bir adla baştan koşar.
# Bedeli bilinçli kabul edildi: reasoning tokenları maliyet/gecikmeyi artırır ve
# çağrıdan çağrıya değişebilir — bu yüzden `reasoning_tokens` her çağrıda
# loglanmaya ve raporlanmaya devam eder.
REASONING_CONFIG = {"enabled": True}

# OpenRouter provider routing politikası (§4: manifestte dondurulacak).
# require_parameters=True KRİTİK: response_format={"type":"json_object"}
# desteklemeyen bir provider'a düşersek Kol 3/structured sessizce serbest metne
# çöker — sözleşme uyum oranı ölçümü anlamsızlaşır. Bu ayar, parametreyi
# desteklemeyen provider'ları eler.
# allow_fallbacks=True bilinçli: tek provider'a kilitlemek uzun koşularda
# kesinti riskini artırır. Gerçekleşen provider her çağrıda loglanır, böylece
# provider değişiminin sonuçlara etkisi analizde kontrol edilebilir.
OPENROUTER_PROVIDER_ROUTING = {"require_parameters": True, "allow_fallbacks": True}

# Model-başına routing istisnaları. Varsayılan politika (yukarıda) çoklu
# sağlayıcıya izin verir; 2026-07-27 probe'unda MiniMax M3 aynı parametrelerle
# dört farklı sağlayıcıya (DeepInfra/Minimax/Morph/Together) yönlendirildi —
# farklı serving yığınları gizli bir varyans kaynağı. MiniMax kendi resmi
# sağlayıcısına sabitlenir: tekrarlanabilirlik, fallback dayanıklılığından
# önce gelir (model sağlayıcısının kendi endpoint'inin düşmesi düşük olasılık;
# düşerse koşu durur ve resume ile devam edilir — sessizce başka bir yığında
# koşmasındansa durması tercih edilir).
#
# ANAHTAR ROLE DEĞİL MODEL KİMLİĞİNE BAĞLI (2026-07-29 düzeltmesi): istisna
# önceden `MODEL_SECONDARY` altında duruyordu ve o sabit MiniMax'ti. Replikasyon
# modeli DeepSeek'e çevrilirken bu satır olduğu gibi bırakılsaydı, DeepSeek'in
# BÜTÜN çağrıları `order=["minimax"]` ile MiniMax sağlayıcı rotasına zorlanırdı.
# Bu yüzden anahtar üçüncü-judge kimliğidir; testte de negatif olarak sınanır.
#
# Bu istisna P0 gözlemine dayalı GEÇİCİ bir aday rotadır; nihai dondurma P1
# uyumluluk smoke'undan sonra yapılacaktır.
# Gemini, DeepSeek ve Grok için exact provider/order TAHMİN EDİLMEZ: varsayılan
# politika (require_parameters + fallback) geçerlidir, gerçekleşen sağlayıcı
# zaten her çağrıda loglanır.
MODEL_PROVIDER_ROUTING = {
    MODEL_JUDGE_EXTERNAL: {"require_parameters": True, "allow_fallbacks": False,
                           "order": ["minimax"]},
}


def provider_routing_for(model: str) -> dict | None:
    """Modele özgü routing politikası; istisna yoksa varsayılan."""
    return MODEL_PROVIDER_ROUTING.get(model, OPENROUTER_PROVIDER_ROUTING)


# OpenRouter usage accounting: yanıtın usage bloğuna GERÇEK maliyeti ekler.
# 2026-07-27 canlı probe: litellm.completion_cost() ÜÇ modelin de fiyatını
# bilmiyor ("This model isn't mapped yet") — yani maliyet takibi statik bir
# fiyat tablosuna bırakılsaydı hem kırılgan hem güncel olmayan olurdu.
# usage.cost ise sağlayıcının kendi faturalandırdığı değer; ayrıca cached_tokens
# görünür hale gelir (prompt cache'i kollar arasında farklı isabet edebilir —
# maliyet karşılaştırmasında kontrol edilmesi gereken bir değişken).
OPENROUTER_USAGE_ACCOUNTING = {"include": True}

# Logprobs ARTIK rutin olarak istenmiyor. Gerekçe (2026-07-27 canlı probe):
# 1. require_parameters=True ile birlikte gönderilince Gemini 2.5 Flash Lite
#    için UYGUN ENDPOINT KALMIYOR (404 "No endpoints found") — hiçbir Google
#    endpoint'i logprobs desteklemiyor. Ana modeli çalıştıramamak, fırsatçı bir
#    yan veriden kat kat ağır basar.
# 2. Logprobs istemek provider seçimini DEĞİŞTİRİYOR (MiniMax: Minimax → Morph)
#    — yani ölçüm aracının kendisi ölçülen sistemi değiştiriyordu.
# 3. Birincil belirsizlik metriği zaten self-consistency proxy'si (sağlayıcıdan
#    bağımsız); logprobs yalnızca ikincil bir korelasyon yan bulgusuydu.
# Gerekirse tek tek çağrılarda call_model(..., logprobs=True) ile açılabilir —
# ama o çağrının provider dağılımı diğerlerinden farklı olacağı için ana deney
# verisiyle karıştırılmamalıdır.
REQUEST_LOGPROBS = False

# --- Analiz (EXPERIMENT_PROTOCOL.md §8; ana veriden önce dondurulur) ---
# Analiz BİRİMİ görevdir. Her (model, görev, kol) için önce tekrarlar üzerinden
# pass_count/repeat_count hesaplanır; 150 tekrar ASLA 150 bağımsız gözlem gibi
# ele alınmaz (aynı görevin tekrarları birbiriyle ilişkilidir — görev zorluğu
# ortak bir etkendir). Kümeleme yok sayılırsa CI olduğundan DAR çıkar.
ANALYSIS_UNIT = "task"

# Birincil estimand: contract - naive, görev-düzeyi EŞLEŞTİRİLMİŞ Plus farkı.
PRIMARY_COMPARISON = (ARM_CONTRACT, ARM_NAIVE)
# İkincil karşılaştırmalar (mekanizma ayrıştırması: RQ2 ve RQ3).
SECONDARY_COMPARISONS = (
    (ARM_STRUCTURED, ARM_NAIVE),     # yapılandırılmış temsilin izole etkisi
    (ARM_CONTRACT, ARM_STRUCTURED),  # validator + retry'ın ilave etkisi
)
# Aynı karşılaştırmalar iki metrik için de üretilir. Plus BİRİNCİL, base ikincil;
# ikisi ASLA aynı sütunda toplanmaz.
ANALYSIS_METRICS = ("plus_pass", "base_pass")
PRIMARY_METRIC = "plus_pass"

# Cluster (görev) bootstrap — seed sonuçlar görülmeden dondurulur.
# Her yinelemede GÖREVLER replacement ile seçilir ve seçilen görevin bütün
# kol/tekrar kümesi birlikte taşınır. Yüzdelik (percentile) %95 CI.
BOOTSTRAP_ITERATIONS = 10_000
BOOTSTRAP_SEED = 20260728
BOOTSTRAP_CI_LEVEL = 0.95

# --- MAST etiketleme protokolü (EXPERIMENT_PROTOCOL.md §9) ---
MAST_CONFIDENCE_LEVELS = ("low", "medium", "high")

# Confidence ankrajları prompt'a GİRER. Tanımsız bırakılırsa "low" her modelde
# başka bir şey demek olur ve §9.3'teki "düşük güvenli kayıtları insana ver"
# kuralı modeller arasında tutarsız bir ölçüte dayanırdı.
MAST_CONFIDENCE_ANCHORS = {
    "low": "evidence is insufficient, or two or more primary modes are about equally likely",
    "medium": "one mode is dominant but a plausible alternative remains",
    "high": "the primary mode is directly and unambiguously supported by the evidence",
}

# Etiketleme bir SINIFLANDIRMA görevidir; örnekleme çeşitliliği istenmez.
# Üretim kollarının DEFAULT_TEMPERATURE'ından ayrı tutulur.
MAST_JUDGE_TEMPERATURE = 0.0

# Judge/adjudicator çağrıları ana performans çağrı loguna KARIŞMAZ:
# logs/exp_<name>/<namespace>/llm_calls.jsonl
MAST_LOG_NAMESPACE = "mast"

# Judge'a kolun ADI (naive/structured/contract) VERİLMEZ; yalnız etkileşim tipi
# verilir. Gerekçe: hata dağılımı kollar arasında karşılaştırılacak; "contract"
# gibi bir ad judge'da beklenti yanlılığı üretir (sözleşme kolunda iletişim
# hatası aramaya yatkınlaşır). Etkileşim tipi yine de gerekli, çünkü tek-ajanlı
# baseline'da kategori 2 (inter-agent) modları YAPISAL OLARAK uygulanamaz.
MAST_INTERACTION_TYPES = {ARM_BASELINE: "single_agent"}  # diğer hepsi multi_agent

# Adjudicator'a giden A/B/C harflerinin sırası kayıt başına DÖNDÜRÜLÜR.
# Sabit bir model→harf eşleşmesi (ör. "A hep DeepSeek") pozisyon yanlılığını
# model yanlılığına çevirirdi: adjudicator sistematik olarak belli bir konumu
# tercih ederse bu, belli bir judge'ı tercih etmesiyle aynı şey olurdu.
# Seed sabit -> aynı koşu aynı prompt'u üretir (tekrarlanabilirlik korunur).
MAST_ANNOTATOR_ORDER_SEED = 20260728

# --- İnsan örneklemi (§9.3) — seçim algoritması sonuçlar görülmeden donduruldu ---
# 30 HEDEF sayıdır, katı üst sınır değil: zorunlu küme (external split/incomplete
# + external yetersiz bağlam + self/external-consensus ayrışması) 30'u aşarsa
# örneklem de aşar. "Bütün anlaşmazlıklar incelenir" iddiası
# ile "en fazla 30" birlikte savunulamazdı; önceliği bilimsel iddiaya verdik.
MAST_HUMAN_SAMPLE_TARGET = 30
MAST_HUMAN_SAMPLE_SEED = 20260728
# Kalan kontenjan bu üçlüye göre tabakalanır (her tabakadan sırayla, fixed-seed
# karıştırmayla) — tek bir kol/model/hata sınıfının örneklemi domine etmemesi için.
MAST_HUMAN_STRATA_FIELDS = ("model", "arm", "error_class")

# Kör insan turu paket sürümleri. İkisi AYRI: şema, saklanan verinin biçimidir
# (kilitli JSON'lar bununla doğrulanır); app sürümü arayüzün kendisidir. İkisi de
# localStorage anahtarına ve paket kimliğine girer; herhangi birinin değişmesi
# yeni insan turudur, eski kilitler tarih olarak korunur ama otomatik kabul edilmez.
# "2.1" (2026-08-03) = self-judge etiketi NULLABLE. Kapı "tam üçlü panel"den
# "dış karar kurulabiliyor mu"ya taşındı: self yalnız TANISAL olduğu için (§9.1)
# eksik bir self, kaydı insan örneklem EVRENİNDEN düşürmez — düşürseydi evren
# tanısal bir AI çıktısının başarısına koşullandırılmış olurdu. Dış judge
# eksikliği hâlâ fail-closed'dır. Saklanan veri şekli değişti (örneklem
# manifestinde `self_judge_available`, diagnostic payload'da nullable
# `self_judge_label`, uyum özetinde `self_judge_missing_count`), app sürümü de
# arayüz "self etiketi mevcut değil" durumunu gösterdiği için arttı.
# 2.0 artefaktları OTOMATİK MİGRATE EDİLMEZ: iki sürüm farklı örneklem evreni
# tanımına dayanır ve `dataset_fingerprint` bu yüzden zaten değişir.
MAST_HUMAN_SCHEMA_VERSION = "2.1"
MAST_HUMAN_APP_VERSION = "2.1"
MAST_HUMAN_LOCK_SCHEMA_VERSION = "1.0"

# İki bağımsız kör etiketleyici. Kimlikler DOSYA YOLUNDA kullanılır — bu yüzden
# yalnız [a-z0-9_-] kabul edilir ve ikisi farklı olmak zorundadır (aynı kimlikli
# iki paket, "iki bağımsız etiketleyici" iddiasını sessizce çürütürdü).
MAST_HUMAN_ANNOTATORS = ("annotator_a", "annotator_b")

# --- Şema sürümleri ---
# Kayıt formatı değiştiğinde artırılır; analiz katmanı beklemediği sürümü
# gördüğünde sessizce yanlış yorumlamak yerine durur.
# "2.0" = final tasarım (4 kol + EvalPlus base/plus + tam provenance).
# Sonuç sözleşmesinin alan listesi Parça 4'te kesinleşir; buradaki sabit tek
# kaynak olarak şimdiden tanımlanır ki loglar ve analiz aynı değeri paylaşsın.
RESULT_SCHEMA_VERSION = "2.0"
# "2.1" = başarılı (status="ok") çağrı kaydına response_id/native_finish_reason/
# requested_provider/actual_provider eklendi (P1 Parça 6A). Bu dört alan daha
# önce YALNIZ provider_error kayıtlarında vardı -- başarılı ve hatalı kayıtlar
# aynı provenance kavramları için farklı alan kümesi taşıyordu. `provider`
# alanı geriye uyumluluk için KORUNDU (actual_provider'la aynı değer); analiz
# katmanı bu alanların hiçbirini ZORUNLU okumaz, yalnız formal uyumluluk
# smoke'u (eval/compatibility_smoke.py) okur.
LLM_CALL_SCHEMA_VERSION = "2.1"
# "1.0" = self-consistency adaylarının deney-bağlı manifest/resume/provenance
# sözleşmesi. Eski global/timestampsiz results_selfcons_*.jsonl dosyaları bu
# sürüme otomatik migrate edilmez; formal RQ5 girdisi yalnız bu sürümlü hattan
# üretilir.
SELF_CONSISTENCY_SCHEMA_VERSION = "1.0"
# "3.0" = leave-self-out karar semantiği (§9.1). 2.0 etiket/panel/manifest
# artefaktları bu turda OTOMATİK MİGRATE EDİLMEZ: 2.0 kayıtları üçlü çoğunluğa
# göre üretilmişti; yeni karar iki DIŞ judge'a dayanıyor. Sessiz kabul, iki farklı
# karar kuralıyla üretilmiş etiketleri aynı analizde toplardı.
# "3.1" = panel kayıt sözleşmesine İKİ zorunlu hash alanı eklendi
# (`full_panel_input_sha256` / `decision_input_sha256`) ve geçici tek hash
# (`panel_input_sha256`) kaldırıldı. 3.0 panelleri 3.1 sayılamaz: tek hash'i olan
# bir kayıt, karar tazeliğini tanısal panel tazeliğinden ayıramaz.
# "3.2" = external-only adjudication. Adjudicator artık ÜÇ değil İKİ etiket
# görür (A/B) ve karar kaydı rol/karar alanlarını taşır. 3.1 adjudication
# kayıtları üç etiketli bir prompt'un ürünüdür — self-judge'ın gerekçesi karara
# girmiştir — ve 3.2 turunda güncel sayılamaz.
MAST_SCHEMA_VERSION = "3.2"
# "2.1" (2026-07-30) = betimleyici blokların (`*_per_run`) alan sözleşmesine
# uzun-kuyruk ölçüleri eklendi: `n`, `p95`, `observed_max`. Mevcut alanlar
# (mean/median/q1/q3/iqr) DEĞİŞMEDİ ve `median` hâlâ P50'dir; yeni bir eşik,
# hipotez testi veya p-değeri EKLENMEDİ. Sürüm yine de artar: çıktı sözleşmesi
# genişledi ve 2.0 özetlerini okuyan bir tüketici bu alanları bulamaz.
# "2.2" (2026-08-03) = RQ5'in üçüncü sorusu ("base geçen çözümlerin ne kadarı
# plus'ta eleniyor", §2 RQ5) `base_plus_attrition` bloğuyla eklendi. Betimleyici,
# birim ARM-RUN; hiçbir ön-kayıtlı estimand, seed, karşılaştırma veya bootstrap
# davranışı değişmedi ve p-değeri yine üretilmiyor. Sürüm artar çünkü çıktı
# sözleşmesi genişledi.
ANALYSIS_SCHEMA_VERSION = "2.2"

# --- RQ5 keşifsel bağlantı katmanı (§8.5) ------------------------------------
# Self-consistency ↔ başarısızlık ilişkisi ve MAST kol dağılımı, ANA performans
# analizinden AYRI modüllerde üretilir (analysis/rq5.py, analysis/mast_distribution.py):
# ana analiz MAST/self-consistency girdisi HİÇ YOKKEN de çalışabilmelidir, aksi
# halde P3'ün etiketleme aşaması birincil estimand'ı rehin alırdı.
RQ5_SCHEMA_VERSION = "1.0"
# Birincil keşifsel eşleştirme kolu. Gerekçe: self-consistency hattı planner ya
# da contract KULLANMAZ — `uncertainty/self_consistency.py` doğrudan
# `pipeline.baseline.SYSTEM_PROMPT` ile tek kodlayıcı adayı üretir. Dolayısıyla
# agreement'ın kavramsal eşi baseline kolunun başarısızlık oranıdır; diğer üç kol
# ve 12 arm-run'lık genel oran yalnız İKİNCİL/betimleyici olarak raporlanır.
RQ5_PRIMARY_ARM = ARM_BASELINE
# Bağ (tie) davranışı ORTALAMA RANK'tır (fractional ranking) ve rho, rank
# vektörlerinin Pearson korelasyonudur. Sürümlenir çünkü bağ kuralı değişirse
# aynı veriden farklı bir rho çıkar.
RQ5_TIE_METHOD = "average_rank_pearson_v1"

# MAST kol dağılımı: İKİ AYRI payda (§8.5 "hataların kollara göre dağılımı").
# Tek payda yanıltıcıdır — farklı sayıda başarısızlığa sahip kollar yalnız
# koşullu yüzdelerle karşılaştırılırsa, az hata yapan bir kolun tek hatası
# "%100 bu mod" görünür.
MAST_DISTRIBUTION_SCHEMA_VERSION = "1.0"

# AI karar kuralının kimliği. Şema sürümünden AYRI: alan listesi değişmeden de
# karar semantiği değişebilir (ör. hangi etiketlerin oy kullandığı). Manifestin
# KRİTİK alanıdır — tur ortasında değişirse aynı dosyada iki farklı kuralla
# üretilmiş kararlar karışırdı.
MAST_DECISION_RULE_VERSION = "leave_self_out_v1"

# Panel girdi hash'lerinin kanonikleştirme sürümü. Karar KURALINDAN da şema
# sürümünden de AYRI: hangi alanların hash'lendiği ve hangi sırada
# kanonikleştirildiği, karar semantiği hiç değişmeden de değişebilir. Manifestin
# KRİTİK alanıdır — tur ortasında değişirse aynı dosyadaki hash'ler
# karşılaştırılamaz hale gelir ve adjudication resume'u sessizce yanlış çalışır.
MAST_PANEL_HASH_VERSION = "dual_input_v1"

# --- Deney çalıştırıcı (eval/runner.py) ---
# Kol sırası karşı-dengeleme formülünün sürümü — manifest'in kritik alanı.
# Formül (task_index + repeat) % len(arms) değişirse bu string de değişmeli;
# aksi halde eski/yeni kayıtlar farklı rotasyon şemasıyla sessizce karışır.
ARM_ROTATION_SCHEME_VERSION = "index_plus_repeat_mod_arms_v1"

# --- P1 Parça 6A: Gemini/Grok uyumluluk smoke altyapısı (OFFLINE) ---
# EXPERIMENT_PROTOCOL.md §4 uyumluluk kapısı (Gemini 3.5
# Flash Lite 10-20 çağrılık smoke) ve P1 Parça 4 öncesi Grok 4.3 MAST-şema/
# taşıma smoke'u. Bu sabitler yalnız ALTYAPIYI dondurur — Parça 6A'da hiçbir
# gerçek/ücretli API çağrısı yapılmadı; formal smoke "tamamlandı" sayılmaz.
COMPATIBILITY_SMOKE_SCHEMA_VERSION = "1.0"
COMPATIBILITY_SMOKE_TASK_SET = PILOT_TASK_SET
COMPATIBILITY_SMOKE_CALLS_PER_TARGET = 12  # normatif 10-20 aralığında (§ "Development kalibrasyon kapısı")
# Gemini matrisi 6 görev × (1 planner + 1 coder) = 12 çağrı kullanır.
COMPATIBILITY_SMOKE_GEMINI_TASK_COUNT = 6
# Grok matrisi held-out KULLANMAZ: 12 sentetik anlaşmazlık senaryosu (6 Gemini +
# 6 DeepSeek kaynaklı), pilot görev metni + sentetik başarısız result kaydı.
COMPATIBILITY_SMOKE_GROK_SCENARIO_COUNT = 12
# Seçim seed'i held-out seçimi (HELDOUT_SELECTION_SEED) ve insan örneklemi
# seed'inden BİLİNÇLİ olarak AYRI: bu smoke hiçbir kimlik alanını onlarla
# paylaşmaz -- aksi halde ayrı kalması gereken iki mekanizma karışabilirdi.
COMPATIBILITY_SMOKE_SELECTION_SEED = 20260730

# Formal smoke CLI'sı YALNIZ bu iki anahtarı kabul eder. DeepSeek/MiniMax/
# main/dev gibi dolaylı takma adlar veya serbest bir LiteLLM slug'ı formal
# hedef olarak KABUL EDİLMEZ -- CLI'da başka argüman/seçenek bilerek YOKTUR.
# Sıra normatiftir (gemini önce): plan/rapor bu sırayla üretilir.
COMPATIBILITY_SMOKE_TARGETS = {
    "gemini": MODEL_MAIN,
    "grok": MODEL_ADJUDICATOR,
}
