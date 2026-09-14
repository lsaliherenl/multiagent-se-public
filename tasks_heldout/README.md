# Held-out Ana Görev Seti

**Resmi ad:** *EvalPlus-derived equality-compatible held-out subset*
**Birincil metrik:** *Plus-test pass rate* (`plus_pass`), ikincil `base_pass`.

50 görev: 30 HumanEval+ (v0.1.10) + 20 MBPP+ (v0.2.0). Bu, ana deneyin
istatistiksel sonuçlarını üreten settir. `tasks/` altındaki 20 görevlik pilot
seti AYRI bir dizindedir ve nihai istatistiğe dahil edilmez
(bkz. `EXPERIMENT_PROTOCOL.md` §5).

Yeniden üretim:

```bash
uv run python scripts/fetch_evalplus.py
```

Kaynaklar sürüm etiketine sabitlenmiş ve SHA-256 ile doğrulanır; seçim seed'i
`20260727`. Aynı komut aynı 50 görevi bayt-bayt yeniden üretir.

## ⚠️ Adlandırma kuralı

Sonuçlar **"HumanEval+ / MBPP+ skoru" olarak adlandırılmaz.** Aşağıdaki
equality-oracle kısıtı nedeniyle bu sayılar EvalPlus liderlik tablosuyla
**doğrudan kıyaslanamaz**. Doğru ifade:

> EvalPlus-derived equality-compatible held-out subset üzerinde Plus-test
> pass rate

## Format (`*.json`)

| Alan | Açıklama |
|---|---|
| `task_id` | `humanevalplus_NNN` / `mbppplus_NNN` |
| `source` | `humanevalplus` \| `mbppplus` |
| `source_task_id` | Kaynak veri setindeki ID (`HumanEval/2`, `Mbpp/745`) |
| `source_version` | Kaynak release etiketi |
| `prompt` | Modele verilen görev metni |
| `entry_point` | Test edilen fonksiyonun adı |
| `reference_solution` | Doğru çözüm — modele ASLA verilmez; harness doğrulaması ve analiz için |
| `atol` | Kayan nokta karşılaştırma toleransı (0 ise tam eşitlik) |
| `input_variant` | Girdi deserializasyon varyantı (`asis` / `outer_tuple` / `deep_tuple`) |
| `n_base_cases` / `n_plus_cases` | Test vakası sayıları |
| `base_test_code` | Yalnız base girdileriyle `check(candidate)` |
| `plus_test_code` | Base **+** plus girdileriyle `check(candidate)` |
| `content_sha256` | Görev içeriğinin bütünlük hash'i |

Görev dosyaları **bayt-bayt yeniden üretilebilir**: aynı komut aynı içeriği
üretir. Bu yüzden ortama bağlı ölçümler (referans çözümün süreleri) görev
dosyasında DEĞİL, manifestin `reference_timings` alanındadır — dosyaya
yazılsalardı her koşuda değişir ve `content_sha256` anlamsızlaşırdı.

`_` ile başlayan dosyalar (`_selection_manifest.json`, bu README) görev
değildir; `eval.harness.load_all_tasks()` bunları atlar.

## Oracle nasıl kuruldu

EvalPlus veri dosyaları beklenen **çıktıları içermez**; yalnız girdileri
(`base_input` / `plus_input`) ve referans çözümü verir. Beklenen çıktılar
referans çözüm sandbox'ta çalıştırılarak üretilir — EvalPlus'ın kendi yaptığı
da budur.

JSON tuple taşıyamadığı için bazı MBPP girdileri liste olarak saklanır. Doğru
deserializasyon **tahmin edilmez, doğrulanır**: EvalPlus'ın kendi `contract`
alanı (girdi tiplerini assert eden kod) referans çözüme enjekte edilir, yanlış
varyant `AssertionError` ile düşer.

Karşılaştırma tam eşitliktir (`atol > 0` ise kayan noktada mutlak tolerans);
`NaN == NaN` kabul edilir, liste/tuple tip ayrımı korunur, her çağrı öncesi
argümanlar `deepcopy`'lenir.

## Seçim ve uygunluk filtresi

Filtre **hiçbir model çağrısı yapmaz**. Tasarım: `EXPERIMENT_PROTOCOL.md` §5.
Özet:

**Statik:** tekil ve tanımlı entry point · determinizm riski taşıyan desen yok
(`random`, `time`, `open(`, `id(`, `hash(`) · yalnız izin listesindeki
standard-library importları · harness ad çakışması yok · base ve plus girdileri
mevcut · **equality-oracle uyumluluğu**.

**Dinamik (sandbox):** girdi varyantı `contract` ile doğrulanabiliyor · beklenen
çıktılar temel literallerle temsil edilebiliyor · referans çözüm base **ve** plus
testlerini ayrı süreçte geçiyor · referansın plus süresi bütçe içinde.

### Equality-oracle kısıtı (dış geçerlilik sınırı)

EvalPlus'ın eşitlik-**dışı** oracle kullandığı görevler bilinçli olarak
dışlanmıştır (küme-eşitliği, yalnız "None değil", göreve özel matematik; liste
kaynağı: EvalPlus `evalplus/eval/_special_oracle.py`). Bu görevlerde tam
eşitlik, **doğru ama farklı sıralı/biçimli** bir çözümü haksız yere elerdi.

- **İç geçerlilik korunur:** dört kol tam olarak aynı görevlerde, aynı oracle
  ile karşılaştırılır.
- **Dış geçerlilik sınırlıdır:** sonuçlar HumanEval+/MBPP+'ın tamamına
  genellenmez, equality-oracle uyumlu alt kümeye aittir.

## Seçim manifesti

`_selection_manifest.json` denetlenebilirlik için şunları saklar: veri seti adı
ve metrik adı · kaynak sürüm/URL/SHA-256/boyut · toplam, incelenen, dışlanan,
uygun ve seçilen sayıları (+ `counts_consistent` bütünlük özdeşliği) ·
**bütün dışlanan ID'ler** ve **ID başına tam ret nedeni** · nedene göre özet
sayım · uygun havuzun tamamı · seçilenler · dışlanan pilot ID'leri · seed ·
timeout/bütçe değerleri · görev içerik hash'leri · seçilen görevlerin referans
çözüm süreleri (`reference_timings`).

## Regresyon güvencesi

`tests/test_base_plus.py`, 50 görevin **her birinin** referans çözümünün hem
base hem Plus testlerini geçtiğini her `pytest` koşusunda doğrular. Ölçüm
aletine güvenilemezse deney sonuçlarına da güvenilemez.
