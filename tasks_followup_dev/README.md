# Study 2 Development Görev Seti

**Kaynak:** BigCodeBench-Hard, `v0.1.4` bölmesi, Hugging Face revizyonu
`298d2cc7b96612e15e47313c3603ee124cee0c1f` (dataset SHA-256
`73a4270b43feb81abefad7bb1b592937c768ea45c3263bedc1cabd30a9a6ce79`).
**Birincil metrik:** gizli testlerin geçmesi (`bigcodebench_hidden_test_pass`).

16 görev. Yalnız teknik kalibrasyon içindir; ana istatistiğe GİRMEZ. Seçim özel çalışma alanında yapıldı ve donduruldu; bu
dizindeki dosyalar o seçimin bayt olarak aynısıdır ve yeniden seçilmez
(`EXPERIMENT_PROTOCOL.md` §13). Dosya hash'leri `_selection_manifest.json`
içindeki `task_file_sha256` alanında ve
`reproduction/paper_run_identity.json` içinde kayıtlıdır.

Koşturmak için açık çalışma kimliği gerekir:

```bash
uv run python -m eval.runner --name <ad> --study study2 --model main   --task-set followup_dev --repeats 3
```

Değerlendirme, sabitlenmiş resmî BigCodeBench imajında ağsız yapılır
(`scripts/bigcodebench_runtime.py`).

## Format (`bigcodebench_NNNN.json`)

| Alan | Açıklama |
|---|---|
| `task_id` | Yerel kimlik: `bigcodebench_` + dört haneli kaynak numarası |
| `source_task_id` | Kaynak kimlik (`BigCodeBench/13`); yerel kimlikle iki yönlü eşlenir |
| `prompt` | Modele verilen görev metni (kaynaktaki `instruct_prompt`) |
| `entry_point` | Test edilen fonksiyon (`task_func`) |
| `test_code` | Gizli test kodu — modele ASLA verilmez, sonuç kaydına kopyalanmaz |
| `evaluation_backend` | `bigcodebench_untrusted_check_v1` (harness bu alanla yönlendirir) |
| `prompt_sha256`, `test_sha256` | İçerik hash'leri |
| `provenance` | Seçim girdilerinin parmak izleri ve referans çözüm kapısı sonucu |

Kanonik (altın) çözümler bu dosyalarda **yoktur**. `blind_id` ve
`assignment_position` alanları seçim sırasındaki kör kimlik ve atama
sırasıdır; analizde kullanılmaz.

Lisans: BigCodeBench, Apache License 2.0 (`THIRD_PARTY_NOTICES.md`).
