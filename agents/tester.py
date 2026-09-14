"""Tester düğümü — LLM'siz DETERMİNİSTİK terminal düğüm.

Aday kodu harness'tan geçirir ve yapılandırılmış raporu state'e yazar. Hiçbir
zaman call_model() çağırmaz; bu yüzden sistem "2 LLM rolü (planner,
coder) + 1 deterministik değerlendirme düğümü" olarak adlandırılır, "3 LLM
ajanı" DEĞİL.

Mesaj alıcısı "end": grafta tester'dan sonra END gelir, coder'a dönüş YOKTUR.
Eskiden burada "tester → coder" yazılıyordu — hiç gerçekleşmeyen bir kenardı
ve iletişim analizini yanıltırdı. Tester→coder repair loop'u bilinçli olarak
kapsam dışıdır (EXPERIMENT_PROTOCOL.md §12).
"""

from eval.harness import evaluate_base_plus
from pipeline.state import PipelineState


def tester_node(state: PipelineState) -> dict:
    # evaluate_base_plus: held-out görevlerde base ve Plus testlerini AYRI
    # koşturur (§5.5); base/plus taşımayan pilot görevlerde tek koşuya düşer
    # ama aynı şemayı üretir -> analiz katmanı tek biçim görür.
    report = evaluate_base_plus(state["task"], state["code"])
    return {
        "test_report": report,
        "raw_messages": [{"from": "tester", "to": "end", "content": report}],
    }
