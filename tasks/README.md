# Görev Seti

20 görev: 12 HumanEval + 8 sanitized-MBPP. `uv run python scripts/fetch_tasks.py`
ile birebir yeniden üretilir (seçim listeleri scriptte sabittir).

## Format (`*.json`)

| Alan | Açıklama |
|---|---|
| `task_id` | `humaneval_NNN` / `mbpp_NNN` (kaynak veri setindeki numara) |
| `source` | `humaneval` \| `mbpp` |
| `prompt` | Modele verilen görev metni (HumanEval: imza+docstring; MBPP: açıklama + fonksiyon adı + örnek test) |
| `entry_point` | Test edilen fonksiyonun adı |
| `test_code` | `def check(candidate)` tanımlayan test kodu; harness sonuna `check(<entry_point>)` ekler |
| `reference_solution` | Veri setinin doğru çözümü — modele ASLA verilmez, harness'ın kendi doğrulamasında ve analizde kullanılır |

## Seçim kriterleri

- Kolay/orta/zor karışımı: HumanEval 39, 109, 115, 129 zor uçta; MBPP sanitized
  (elle doğrulanmış 427'lik alt küme) kolay-orta ağırlıklı.
- MBPP 56 bilinçli dışarıda: fonksiyon adı `check`, harness sözleşmesiyle çakışıyor.
- İki kaynak da aynı `check(candidate)` sözleşmesine normalize edilir; MBPP
  assert'lerindeki fonksiyon adları `candidate` ile değiştirilir.
- Pilot sonrası taban/tavan etkisi görülürse liste revize edilir; her revizyon
  scriptteki ID listesi üzerinden commit'le izlenir.
