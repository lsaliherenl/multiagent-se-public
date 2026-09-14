"""Coder düğümü.

naive (Kol 2): planner'ın serbest metin planını okuyup kod üretir — bilgi
kaybı (semantic drift) tam olarak burada gerçekleşebilir, çalışmanın ölçmek
istediği şey bu.

structured_no_validation (Kol 3a) ve contract (Kol 3b): İKİSİ DE planı
canonical JSON olarak alır (agents/handoff.py::canonical_json). Coder bu iki
kolda planın doğrulanıp doğrulanmadığını BİLMEZ ve ayırt EDEMEZ; aynı içerik
her iki kolda byte-for-byte aynı metne dönüşür. Daha önce contract kolunda
kullanılan düz metin renderer (_format_structured_plan) KALDIRILDI: JSON'u
düz metne çevirmek, "yapılandırılmış temsil" müdahalesini coder'a ulaşmadan
geri alıyordu (construct hatası) ve structured↔contract farkını
validator+retry dışına taşırıyordu.

Coder'ın kendi talimatı (SYSTEM_PROMPT) ÜÇ kolda da AYNIDIR — deneyin iç
geçerlilik şartı ("tek fark iletişim katmanı olmalı") bu sayede korunur.
"""

from agents.handoff import canonical_json
from agents.llm import call_model
from agents.parsing import extract_code
from config import ARM_NAIVE
from pipeline.state import PipelineState

SYSTEM_PROMPT = (
    "You are an expert Python programmer working from a teammate's plan. "
    "Implement the plan as a single complete Python function. "
    "Respond with a single Python code block only — no explanations."
)


def coder_node(state: PipelineState) -> dict:
    task = state["task"]
    plan = state["plan"]

    # Tek dallanma: serbest metin mi, canonical JSON mu. structured ve contract
    # AYNI daldan geçer -- coder düzeyinde aralarında hiçbir fark yoktur.
    plan_text = plan if state["mode"] == ARM_NAIVE else canonical_json(plan)

    user_content = (
        f"Task:\n{task['prompt']}\n\n"
        f"Plan from your teammate:\n{plan_text}\n\n"
        "Implement this as Python code."
    )
    response = call_model(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        model=state.get("model"),
        task_id=task["task_id"],
        agent_role="coder",
        experiment=state.get("experiment"),
        run_id=state.get("run_id"),
        arm=state["mode"],
        repeat=state.get("repeat"),
    )
    code = extract_code(response.text)

    return {
        "code": code,
        "raw_messages": [{"from": "coder", "to": "tester", "content": code}],
    }
