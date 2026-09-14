"""agents/contracts.py şema birim testleri -- doğrudan Pydantic doğrulaması,
network/graf/validator düğümü YOK (yapısal katılığın kendisini test eder)."""

import pytest
from pydantic import ValidationError

from agents.contracts import PlannerOutput, PlanStep

MINIMAL_PLAN = {
    "task_id": "t1",
    "function_signature": "def f(x: int) -> int",
    "steps": [{"description": "do it", "postconditions": ["returns x+1"]}],
}


def test_minimal_plan_gecerli():
    m = PlannerOutput.model_validate(MINIMAL_PLAN)
    assert m.task_id == "t1"
    assert m.steps[0].postconditions == ["returns x+1"]


def test_preconditions_atlanabilir_bos_listeye_duser():
    m = PlannerOutput.model_validate(MINIMAL_PLAN)
    assert m.steps[0].preconditions == []


def test_edge_cases_atlanabilir_bos_listeye_duser():
    m = PlannerOutput.model_validate(MINIMAL_PLAN)
    assert m.edge_cases == []


def test_bosluk_kirpilir():
    m = PlannerOutput.model_validate({**MINIMAL_PLAN, "function_signature": "  def f(x: int) -> int  "})
    assert m.function_signature == "def f(x: int) -> int"


def test_bosluk_only_description_reddedilir():
    with pytest.raises(ValidationError):
        PlanStep.model_validate({"description": "   ", "postconditions": ["x"]})


def test_bosluk_only_function_signature_reddedilir():
    with pytest.raises(ValidationError):
        PlannerOutput.model_validate({**MINIMAL_PLAN, "function_signature": "   "})


def test_steps_bos_reddedilir():
    with pytest.raises(ValidationError):
        PlannerOutput.model_validate({**MINIMAL_PLAN, "steps": []})


def test_postconditions_alani_tamamen_eksik_reddedilir():
    # En önemli regresyon testi: default_factory=list geri eklenirse bu test
    # kırılır -- postconditions'ın GERÇEKTEN zorunlu olduğunu doğrular.
    with pytest.raises(ValidationError):
        PlanStep.model_validate({"description": "do it"})


def test_postconditions_bos_liste_reddedilir():
    with pytest.raises(ValidationError):
        PlanStep.model_validate({"description": "do it", "postconditions": []})


def test_postconditions_icinde_bosluk_only_reddedilir():
    with pytest.raises(ValidationError):
        PlanStep.model_validate({"description": "do it", "postconditions": ["   "]})


def test_extra_alan_planstep_seviyesinde_reddedilir():
    with pytest.raises(ValidationError):
        PlanStep.model_validate({"description": "do it", "postconditions": ["x"], "surprise": 1})


def test_extra_alan_planneroutput_seviyesinde_reddedilir():
    with pytest.raises(ValidationError):
        PlannerOutput.model_validate({**MINIMAL_PLAN, "surprise": 1})


def test_bos_string_task_id_yapisal_olarak_gecerli():
    # task_id'nin DOĞRULUĞU (gerçek göreve eşleşmesi) bu şemanın işi DEĞİL --
    # agents/validator.py semantik olarak kontrol eder. Bu test o ayrımı
    # belgeliyor: şema seviyesinde boş task_id bile yapısal olarak geçer.
    m = PlannerOutput.model_validate({**MINIMAL_PLAN, "task_id": ""})
    assert m.task_id == ""
