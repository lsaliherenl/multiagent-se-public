"""Planner düğümü.

naive (Kol 2): serbest doğal dil plan üretir, coder'a düz metin olarak geçer —
planner-coder boşluğunun (arXiv:2510.10460) ölçülmek istendiği asıl kanal
budur; çıktı hiçbir şekilde doğrulanmaz/yapılandırılmaz.

structured_no_validation (Kol 3a) ve contract (Kol 3b): İKİSİ DE bu dosyadaki
AYNI kod yolunu kullanır — aynı CONTRACT_SYSTEM_PROMPT, aynı
response_format={"type":"json_object"}, aynı parse mantığı, aynı hata zarfı.
Aralarındaki TEK fark graf düzeyindedir (pipeline/graph.py): contract'ta
planner'dan sonra validator düğümü ve sınırlı retry kenarı vardır,
structured'da yoktur. Bu ayrım deneyin RQ3'ünün tamamı olduğu için burada
moda göre DALLANMA YAPILMAZ; tek fark, mesajın kime gittiğidir
(contract → validator, structured → coder).

Parse başarısı ile sözleşme doğrulaması ayrı kavramlardır: planner yalnız
"geçerli JSON nesnesi elde edildi mi" sorusunu cevaplar (handoff_parse_ok),
şema uyumunu bilmez — onu agents/validator.py karara bağlar (yalnız contract).
"""

from agents.handoff import parse_error_envelope
from agents.llm import call_model
from agents.parsing import extract_json
from config import ARM_CONTRACT, ARM_NAIVE
from pipeline.state import PipelineState

NAIVE_SYSTEM_PROMPT = (
    "You are a senior software engineer acting as the planner in a small "
    "engineering team. Given a coding task, write a short natural-language "
    "implementation plan for a programmer teammate to follow: your intended "
    "approach, the key steps, and any edge cases to watch for. "
    "Do NOT write code — prose only."
)

CONTRACT_SYSTEM_PROMPT = (
    "You are a senior software engineer acting as the planner in a small "
    "engineering team that uses a strict contract-based handoff protocol. "
    "Given a coding task (with its Task ID), respond with ONLY a JSON object "
    "(no prose, no markdown fences, no fields other than the ones below) with "
    "exactly these fields:\n"
    '  "task_id": string — must exactly match the Task ID given to you\n'
    '  "function_signature": string — the exact function signature to implement '
    "(must include the function name used in the task)\n"
    '  "steps": a non-empty array of objects, each with "description" (non-empty '
    'string), "preconditions" (array of strings, may be empty), "postconditions" '
    '(NON-EMPTY array of strings — every step must state at least one outcome it '
    "guarantees)\n"
    '  "edge_cases": an array of strings describing edge cases to handle\n'
    "Respond with the JSON object only."
)


def planner_node(state: PipelineState) -> dict:
    task = state["task"]

    if state["mode"] == ARM_NAIVE:
        response = call_model(
            [
                {"role": "system", "content": NAIVE_SYSTEM_PROMPT},
                {"role": "user", "content": task["prompt"]},
            ],
            model=state.get("model"),
            task_id=task["task_id"],
            agent_role="planner",
            experiment=state.get("experiment"),
            run_id=state.get("run_id"),
            arm=state["mode"],
            repeat=state.get("repeat"),
        )
        plan = response.text.strip()
        return {
            "plan": plan,
            "raw_messages": [{"from": "planner", "to": "coder", "content": plan}],
        }

    # structured_no_validation + contract: ortak yol
    user_content = f"Task ID: {task['task_id']}\n\n{task['prompt']}"
    validation = state.get("handoff_validation")
    if validation and not validation.get("valid"):
        user_content += (
            "\n\nYour previous JSON response was invalid:\n"
            f"{validation['errors']}\n\n"
            "Fix these issues and respond with a corrected JSON object only."
        )
    attempt = state.get("attempt_count", 0) + 1  # çağrıdan ÖNCE hesaplanır, tek değer iki yerde kullanılır
    response = call_model(
        [
            {"role": "system", "content": CONTRACT_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        model=state.get("model"),
        response_format={"type": "json_object"},
        task_id=task["task_id"],
        agent_role="planner",
        experiment=state.get("experiment"),
        run_id=state.get("run_id"),
        arm=state["mode"],
        repeat=state.get("repeat"),
        agent_attempt=attempt,
    )
    try:
        parsed = extract_json(response.text)
        if not isinstance(parsed, dict):
            raise ValueError("JSON bir nesne değil")
        # task_id ÜZERİNE YAZILMIYOR -- doğruluğunu agents/validator.py kontrol eder.
        plan, parse_ok = parsed, True
    except ValueError:
        # Çıplak ham metne DÜŞÜLMEZ: deterministik hata zarfı da canonical JSON
        # olarak gider, böylece iki kolun handoff biçimi her koşulda aynı kalır.
        plan = parse_error_envelope(response.text, "yanıtta geçerli JSON nesnesi bulunamadı")
        parse_ok = False

    # Mesaj gerçek graf kenarını göstermeli: contract'ta planner'dan sonra
    # validator düğümü var, structured'da doğrudan coder.
    recipient = "validator" if state["mode"] == ARM_CONTRACT else "coder"
    return {
        "plan": plan,
        "attempt_count": attempt,
        "handoff_parse_ok": parse_ok,
        "raw_messages": [{"from": "planner", "to": recipient, "content": plan}],
    }
