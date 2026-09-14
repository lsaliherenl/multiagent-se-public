"""Paylaşılan LangGraph state şeması — Kol 2 ve Kol 3'ün ortak veri sözleşmesi.

İç geçerlilik şartı: iki kol da AYNI şemayı, AYNI düğümleri ve AYNI prompt
gövdelerini kullanır; tek fark handoff'un serileştirme/doğrulama katmanıdır
(graph.py'deki mode parametresi).
"""

import operator
from typing import Annotated, Any, TypedDict


class PipelineState(TypedDict, total=False):
    task: dict          # tasks/*.json içeriği
    mode: str           # config.GRAPH_MODES üyesi -- agents/llm.py'nin çağrı logunda
    # "arm" olarak kullanılır; ayrı bir "arm" alanı YOK, mode zaten arm ile
    # birebir aynı anlama geliyor (ayrı alan sürüklenebilir gereksizlik olurdu).
    model: str          # üretici model (yoksa config.MODEL_PILOT); runner set eder
    # Çağrı logunun (agents/llm.py) birleştirme anahtarının geri kalanı --
    # runner set eder (pipeline/run_graph.py::run_task üzerinden).
    experiment: str
    run_id: str
    repeat: int
    plan: Any           # naive: serbest metin (str); structured/contract: dict
    code: str           # coder'ın ürettiği aday kod
    test_report: dict | None   # eval.harness.EvalResult alanları
    # İKİ AYRI KAVRAM, bilerek ayrı alanlar:
    # handoff_parse_ok — planner yanıtı geçerli bir JSON NESNESİNE dönüştü mü.
    #   structured VE contract kollarında set edilir (naive'de yok: serbest
    #   metnin parse'ı diye bir şey yok). False ise plan bir hata zarfıdır
    #   (agents/handoff.py::parse_error_envelope).
    handoff_parse_ok: bool
    # handoff_validation — plan pydantic SÖZLEŞMESİNE uydu mu. YALNIZ contract
    #   kolunda set edilir; structured'da pydantic HİÇ çalıştırılmaz (RQ3'ün
    #   ölçtüğü müdahale tam olarak budur), naive'de hep None.
    handoff_validation: dict | None
    attempt_count: int
    # Ham handoff metinleri: Kol 2/3 iletişim farkını analiz edebilmek için
    # her düğüm ürettiği mesajı buraya ekler (operator.add ile birikir).
    raw_messages: Annotated[list[dict], operator.add]
