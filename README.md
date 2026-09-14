# multiagent-se

`multiagent-se`, yazılım geliştirme görevlerinde dört LLM çalışma biçimini
aynı görevler ve aynı model ayarları altında karşılaştırmak için geliştirilmiş
deneysel bir Python çerçevesidir:

1. `baseline`: tek çağrılık sistem referansı,
2. `naive`: serbest metin planlayıcı-kodlayıcı devri,
3. `structured_no_validation`: yapılandırılmış fakat doğrulanmayan devir,
4. `contract`: şema doğrulaması ve sınırlı planlayıcı yeniden denemesi.

`structured_no_validation` ve `contract` aynı planlayıcı/kodlayıcı yolunu
kullanır. Aralarındaki tek müdahale doğrulayıcı düğümü ve sınırlı yeniden
denemedir; bu invariant `tests/test_arm_equivalence.py` ile korunur.

Bu public dağıtım kaynak kodu, dondurulmuş görev tanımlarını ve deterministik
testleri içerir. Ham model çıktıları, deney sonuçları, makale taslakları,
yayın görselleri ve iç çalışma notları bu repoda dağıtılmaz.

## Kurulum

Gereksinimler: Python 3.12+, Git ve [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/lsaliherenl/multiagent-se-public.git
cd multiagent-se-public
uv sync
```

Canlı model çağrıları için örnek ortam dosyasını kopyalayıp yalnız yerel
`.env` dosyasını doldurun:

```bash
cp .env.example .env
```

```dotenv
OPENROUTER_API_KEY=your_key_here
```

`.env`, loglar, anahtar/sertifika dosyaları ve yayın çalışma alanları Git
tarafından yok sayılır. Anahtarınızı komut satırına, commit mesajına veya hata
çıktısına yapıştırmayın.

## Hızlı doğrulama

Testler ağ erişimi ya da gerçek API anahtarı gerektirmez:

```bash
uv run pytest -q
```

İki görev kümesini kaynaklardan yeniden üretmek için:

```bash
uv run python scripts/fetch_tasks.py
uv run python scripts/fetch_evalplus.py
```

İkinci komut sürüm-sabitli EvalPlus kaynaklarını indirir, SHA-256 değerlerini
doğrular, uygunluk filtrelerini çalıştırır ve `tasks_heldout/` içeriğini
deterministik olarak yeniden üretir.

## Deney çalıştırma

Önce küçük pilot setinde akışı doğrulayın:

```bash
uv run python -m eval.runner \
  --name local_smoke \
  --model dev \
  --task-set pilot \
  --tasks 2 \
  --repeats 1
```

Held-out çalışma bilinçli olarak açık model, görev seti ve tekrar sayısı ister:

```bash
uv run python -m eval.runner \
  --name heldout_run \
  --model main \
  --task-set heldout \
  --repeats 3
```

Çıktılar `logs/exp_<name>/` altında oluşur ve version control'e girmez.
Runner; görev hash'lerini, prompt sözleşmesini, ortamı ve Git commit'ini
manifestte sabitler. Tam deney koşusu yalnız temiz ve commit edilmiş bir
çalışma ağacından başlatılmalıdır.

## Depo yapısı

| Yol | İçerik |
| --- | --- |
| `agents/` | Planlayıcı, kodlayıcı, test edici ve doğrulayıcı rolleri |
| `pipeline/` | Dört kolun LangGraph akışları |
| `eval/` | Sandbox, harness, runner, sonuç sözleşmesi ve MAST araçları |
| `analysis/` | Görev-düzeyi analiz ve keşifsel belirsizlik araçları |
| `uncertainty/` | Self-consistency ölçümü |
| `tasks/` | 20 görevlik development/pilot seti |
| `tasks_heldout/` | 50 görevlik EvalPlus-türevi held-out set |
| `scripts/` | Görev üretimi, sağlık ve uyumluluk yardımcıları |
| `tests/` | Ağsız ve deterministik regresyon testleri |

Deney tasarımı ve yorumlama sınırları için
[`EXPERIMENT_PROTOCOL.md`](EXPERIMENT_PROTOCOL.md), veri seti lisansları için
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) dosyasına bakın.

## Güvenlik uyarısı

Bu proje model tarafından üretilen Python kodunu alt süreçte çalıştırır.
Uygulanan timeout, çıktı sınırı ve ortam değişkeni allowlist'i güçlü bir güvenlik
sandbox'ı değildir; özellikle Windows üzerinde bellek/CPU izolasyonu sağlamaz.
Güvenilmeyen kodu hassas verilerin veya ağ kimlik bilgilerinin bulunduğu bir
makinede çalıştırmayın. Daha güçlü izolasyon için tek kullanımlık container ya
da ayrı bir VM kullanın.

## Sonuçların kapsamı

Bu public kaynak dağıtımı herhangi bir performans sonucu veya etki iddiası
yayımlamaz. Kodu kullanarak elde edilen sonuçlar, iki üretici model için ayrı
analiz edilmeli; tekrarlar bağımsız örnekler gibi sayılmamalı ve yalnız
EvalPlus-türevi equality-compatible alt kümeye genellenmelidir.

## Lisans

Projenin özgün kaynak kodu [Apache License 2.0](LICENSE) altında yayımlanır.
Üçüncü taraf benchmark içeriklerinin ayrı koşulları
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) içinde belirtilmiştir.
