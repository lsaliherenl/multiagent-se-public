"""Kol 3 sözleşme şemaları (SEMAP-esinli hafif katman).

Tam SEMAP+A2A implementasyonu İDDİA EDİLMİYOR — sadece ön koşul/son koşul +
tipli JSON mesajlaşma ilkeleri alınıyor (EXPERIMENT_PROTOCOL.md). Şemalar donduruldu
(2026-07-21, tam koşu öncesi sıkılaştırma). Bu şema YAPISAL + hafif SEMANTİK
doğrulama yapar (task_id/entry_point eşleşmesi agents/validator.py'de);
planın ALGORİTMİK OLARAK DOĞRU olduğunu ASLA iddia etmez.
"""

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class PlanStep(BaseModel):
    """Planın tek adımı: ne yapılacak + hangi koşullar altında."""

    model_config = ConfigDict(extra="forbid")

    description: NonEmptyStr
    preconditions: list[NonEmptyStr] = Field(default_factory=list)  # kasıtlı opsiyonel:
    # her adıma (özellikle ilk adıma) zorunlu kılmak modeli anlamsız dolgu
    # metne iter, sinyali güçlendirmez.
    postconditions: list[NonEmptyStr] = Field(min_length=1)  # default YOK -> zorunlu:
    # bir adım tanım gereği bir şey garanti eder/değiştirir (asimetrik zorunluluk).


class PlannerOutput(BaseModel):
    """Planner → coder handoff sözleşmesi."""

    model_config = ConfigDict(extra="forbid")

    task_id: str  # NonEmptyStr DEĞİL bilerek -- doğruluğu agents/validator.py semantik kontrol eder
    function_signature: NonEmptyStr = Field(description="Üretilecek fonksiyonun tam imzası")
    steps: list[PlanStep] = Field(min_length=1)
    edge_cases: list[NonEmptyStr] = Field(default_factory=list)


class CoderOutput(BaseModel):
    """Coder → tester handoff sözleşmesi."""

    task_id: str
    code: str
    satisfied_postconditions: list[str] = Field(default_factory=list)


class TestReport(BaseModel):
    """Tester → coder geri bildirim sözleşmesi (eval.harness.EvalResult aynası)."""

    task_id: str
    status: str
    error_class: str | None = None
    traceback: str | None = None
