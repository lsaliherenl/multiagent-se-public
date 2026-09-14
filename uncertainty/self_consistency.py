"""Self-consistency (öz-tutarlılık) ölçümü — birincil belirsizlik metriği.

Yöntem (oracle-bağımsız, yürütme-tabanlı kümeleme):
1. Aynı görev prompt'undan temp>0 ile N kod adayı üretilir.
2. Görevin KENDİ test girdileri kaydedilir: referans çözüm bir kaydediciyle
   sarılıp check()'ten geçirilir — tüm assert'ler geçtiği için testin
   çağırdığı bütün girdiler yakalanır (kırılgan metin ayrıştırma yok).
   Referans çözüm burada SADECE girdileri keşfetmek için kullanılır.
3. Her aday bu girdilerde çalıştırılır; DÖNEN DEĞERLER birbirleriyle
   karşılaştırılır — oracle'ın beklenen çıktılarına asla bakılmaz. Böylece
   "hepsi birbiriyle tutarlı ama hepsi yanlış" durumu da yakalanabilir
   (güven ile doğruluk ayrışır).
4. Aynı çıktı imzasını üretenler aynı kümeye girer;
   agreement = en büyük küme / N.

Oracle pass/fail'i (harness) her aday için AYRICA loglanır — skora girmez,
sadece "tutarsızlık ↔ başarısızlık korelasyonu" analizinin (H3-hafif) sütunu.

Determinizm notu: set/dict repr'ı süreçler arası hash-seed yüzünden
değişebilir; çıktılar bu yüzden karşılaştırmadan önce kanonikleştirilir
(__canon, aşağıdaki probe şablonunda).

Kullanım:
    uv run python -m uncertainty.self_consistency --name gemini_selfcons \
        --model main --task-set heldout --n 5

Held-out sette yalnız config.MODEL_PRODUCERS çalıştırılabilir (kontrol, API
anahtarı ve görev yüklemesinden ÖNCE).
"""

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from agents.llm import call_model
from agents.parsing import extract_code
from config import (
    LOGS_DIR,
    LLM_CALL_SCHEMA_VERSION,
    LLM_MIN_INTERVAL_S,
    LLM_NUM_RETRIES,
    LLM_PROVIDER_ERROR_BACKOFF_S,
    LLM_PROVIDER_ERROR_RETRIES,
    LLM_TIMEOUT_S,
    MAX_OUTPUT_TOKENS,
    REASONING_CONFIG,
    ROOT,
    SELF_CONSISTENCY_N,
    SELF_CONSISTENCY_SCHEMA_VERSION,
    SELF_CONSISTENCY_TEMPERATURE,
    TASK_SETS,
    model_alias_help,
    provider_routing_for,
    validate_model_for_task_set,
)
from eval.harness import evaluate_base_plus, has_base_plus, load_all_tasks
from eval.sandbox import run_code
from pipeline.baseline import SYSTEM_PROMPT  # Kol 1 ile birebir aynı üretim prompt'u

_INPUTS_MARKER = "__SC_INPUTS__"
_OUTPUTS_MARKER = "__SC_OUTPUTS__"
SELFCONS_ARM = "selfcons"
CANDIDATE_FILE = "candidate_records.jsonl"
RESULT_FILE = "results.jsonl"
MANIFEST_FILE = "manifest.json"
CALL_FILE = "llm_calls.jsonl"


class SelfConsistencyError(RuntimeError):
    """Self-consistency provenance/resume sözleşmesi ihlali."""

# Kanonikleştirici: set/dict sıralaması süreçler arası deterministik olsun
# diye probe scriptine gömülür ({} kaçışlarından kaçınmak için f-string değil).
_CANON_HELPER = '''
def __canon(obj):
    if isinstance(obj, (set, frozenset)):
        return ("set", sorted(repr(__canon(x)) for x in obj))
    if isinstance(obj, dict):
        return ("dict", sorted((repr(k), repr(__canon(v))) for k, v in obj.items()))
    if isinstance(obj, (list, tuple)):
        return (type(obj).__name__, [__canon(x) for x in obj])
    return obj
'''


def _input_discovery_tests(task: dict) -> str:
    """Girdi keşfinde kullanılacak test kodu.

    Held-out (EvalPlus) görevlerinde `test_code` alanı yoktur; BASE testleri
    kullanılır — plus testleri bine kadar girdi taşır ve her adayı o kadar
    girdide koşturmak self-consistency'yi gereksizce pahalılaştırırdı. Ayrıştırma
    gücü için base girdileri yeterli (agreement zaten adaylar ARASI tutarlılığı
    ölçüyor, oracle'a bakmıyor).
    """
    return task["base_test_code"] if has_base_plus(task) else task["test_code"]


def record_test_inputs(task: dict) -> list[str]:
    """Görevin test suite'inin candidate'a geçirdiği tüm girdileri kaydeder.

    Dönen liste: her çağrı için repr((args, kwargs)) metni (çağrı sırasıyla).
    """
    script = (
        f"{task['reference_solution']}\n\n"
        f"__ref = {task['entry_point']}\n"
        "__recorded = []\n"
        "def __recorder(*args, **kwargs):\n"
        "    __recorded.append(repr((args, kwargs)))\n"  # mutasyondan önce kaydet
        "    return __ref(*args, **kwargs)\n\n"
        f"{_input_discovery_tests(task)}\n\n"
        "check(__recorder)\n"
        "import json\n"
        f"print({_INPUTS_MARKER!r} + json.dumps(__recorded))\n"
    )
    result = run_code(script)
    if result.status != "ok":
        raise RuntimeError(
            f"{task['task_id']}: girdi kaydı başarısız ({result.status})\n{result.stderr}"
        )
    for line in reversed(result.stdout.splitlines()):
        if line.startswith(_INPUTS_MARKER):
            return json.loads(line[len(_INPUTS_MARKER):])
    raise RuntimeError(f"{task['task_id']}: girdi kaydı çıktısı bulunamadı")


def output_signature(task: dict, inputs: list[str], candidate_code: str,
                     timeout_s: float | None = None) -> str:
    """Adayın kayıtlı girdilerdeki davranış imzasını döndürür.

    İmza, kanonikleştirilmiş çıktıların (exception'lar dahil) JSON listesidir;
    çalıştırılamayan adaylar ayrı imzalar alır ("__TIMEOUT__" / "__CRASH__").
    """
    script = (
        "import json\n"
        f"{candidate_code}\n"
        f"{_CANON_HELPER}\n"
        f"__inputs = json.loads({json.dumps(json.dumps(inputs))})\n"
        "__outputs = []\n"
        "for __r in __inputs:\n"
        "    __args, __kwargs = eval(__r)\n"
        "    try:\n"
        f"        __outputs.append(repr(__canon({task['entry_point']}(*__args, **__kwargs))))\n"
        "    except Exception as __e:\n"
        "        __outputs.append('__EXC__:' + type(__e).__name__)\n"
        f"print({_OUTPUTS_MARKER!r} + json.dumps(__outputs))\n"
    )
    kwargs = {"timeout_s": timeout_s} if timeout_s is not None else {}
    result = run_code(script, **kwargs)
    if result.status == "timeout":
        return "__TIMEOUT__"
    if result.status != "ok":
        return "__CRASH__"
    for line in reversed(result.stdout.splitlines()):
        if line.startswith(_OUTPUTS_MARKER):
            return line[len(_OUTPUTS_MARKER):]
    return "__NOOUTPUT__"  # aday kod marker satırını bastırdıysa (ör. sys.exit)


def cluster_and_score(signatures: list[str]) -> dict:
    """İmzaları kümeler; agreement = en büyük küme / N."""
    counts = Counter(signatures)
    sizes = sorted(counts.values(), reverse=True)
    return {
        "n": len(signatures),
        "n_clusters": len(counts),
        "cluster_sizes": sizes,
        "agreement": sizes[0] / len(signatures) if signatures else 0.0,
    }


def _canonical_hash(value) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False,
                         separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _prompt_hash() -> str:
    return hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest()


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
            text=True, timeout=10,
        )
        return (result.stdout.strip() or None) if result.returncode == 0 else None
    except Exception:
        return None


def _git_dirty() -> bool | None:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"], cwd=ROOT, capture_output=True,
            text=True, timeout=10,
        )
        return bool(result.stdout.strip()) if result.returncode == 0 else None
    except Exception:
        return None


def require_verified_git_state(manifest: dict | None = None) -> str:
    """Temiz ve çözülebilir git durumunu ücretli çağrıdan önce zorlar.

    `None` temiz demek değildir: git durumu veya HEAD doğrulanamıyorsa formal
    bir tur hangi kaynak koddan üretildiğini kanıtlayamaz ve fail-closed durur.
    Resume sırasında HEAD manifestte dondurulan commit ile de aynı olmalıdır.
    """
    dirty = _git_dirty()
    if dirty is None:
        raise SelfConsistencyError("git çalışma ağacı doğrulanamadı")
    if dirty:
        raise SelfConsistencyError(
            "git çalışma ağacı kirli — self-consistency yalnız temiz committen çalışır")
    commit = _git_commit()
    if not commit:
        raise SelfConsistencyError("git HEAD çözülemedi")
    if manifest is not None and manifest.get("git_commit") != commit:
        raise SelfConsistencyError(
            f"manifest git_commit ({manifest.get('git_commit')!r}) güncel HEAD "
            f"({commit!r}) ile eşleşmiyor — yeni bir --name gerekir")
    return commit


def _task_content_hashes(tasks: list[dict]) -> dict[str, str]:
    return {task["task_id"]: _canonical_hash(task) for task in tasks}


def _task_file_hashes(task_ids: list[str], task_set: str) -> dict[str, str]:
    task_dir = TASK_SETS[task_set]
    hashes = {}
    for task_id in task_ids:
        path = task_dir / f"{task_id}.json"
        if not path.exists():
            raise SelfConsistencyError(f"görev dosyası yok: {path}")
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def _heldout_selection_fingerprint(task_set: str) -> str | None:
    if task_set != "heldout":
        return None
    path = TASK_SETS[task_set] / "_selection_manifest.json"
    if not path.exists():
        raise SelfConsistencyError(f"held-out seçim manifesti yok: {path}")
    try:
        selection = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SelfConsistencyError(f"held-out seçim manifesti okunamadı: {exc}") from exc
    selection.pop("created_ts", None)
    for source in selection.get("sources", {}).values():
        source.pop("reference_timings", None)
    return _canonical_hash(selection)


def _uv_lock_hash() -> str | None:
    path = ROOT / "uv.lock"
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


MANIFEST_CRITICAL_FIELDS = (
    "self_consistency_schema_version", "algorithm_contract", "model", "task_set",
    "task_ids", "task_content_hashes", "task_file_hashes",
    "heldout_selection_fingerprint", "git_commit", "n", "temperature",
    "prompt_hash", "reasoning_config", "provider_routing", "max_tokens",
    "llm_num_retries", "llm_min_interval_s", "llm_timeout_s",
    "llm_provider_error_retries", "llm_provider_error_backoff_s",
    "python_version", "platform", "llm_call_schema_version", "uv_lock_hash",
    "candidate_identity_fields",
)


def build_manifest_snapshot(*, name: str, model: str, task_set: str,
                            tasks: list[dict], n: int, temperature: float,
                            git_commit: str | None = None) -> dict:
    if not name or not name.strip():
        raise SelfConsistencyError("deney adı boş olamaz")
    if n <= 0:
        raise SelfConsistencyError("N pozitif olmalı")
    if not tasks:
        raise SelfConsistencyError(f"{task_set!r} görev seti boş")
    task_ids = [task["task_id"] for task in tasks]
    if len(task_ids) != len(set(task_ids)):
        raise SelfConsistencyError("görev listesinde yinelenen task_id var")
    commit = git_commit if git_commit is not None else _git_commit()
    if not commit:
        raise SelfConsistencyError("git HEAD çözülemedi; manifest üretilemez")
    return {
        "name": name,
        "created_ts": datetime.now(timezone.utc).isoformat(),
        "self_consistency_schema_version": SELF_CONSISTENCY_SCHEMA_VERSION,
        "algorithm_contract": {
            "input_discovery": "base_when_available_v1",
            "canonical_output_signature": "behavioral_canon_v1",
            "clustering": "exact_signature_largest_cluster_v1",
            "oracle_pass_rate": "plus_status_passed_fraction_v1",
        },
        "model": model,
        "task_set": task_set,
        "task_ids": task_ids,
        "task_content_hashes": _task_content_hashes(tasks),
        "task_file_hashes": _task_file_hashes(task_ids, task_set),
        "heldout_selection_fingerprint": _heldout_selection_fingerprint(task_set),
        "git_commit": commit,
        "n": n,
        "temperature": temperature,
        "prompt_hash": _prompt_hash(),
        "reasoning_config": REASONING_CONFIG,
        "provider_routing": provider_routing_for(model),
        "max_tokens": MAX_OUTPUT_TOKENS,
        "llm_num_retries": LLM_NUM_RETRIES,
        "llm_min_interval_s": LLM_MIN_INTERVAL_S,
        "llm_timeout_s": LLM_TIMEOUT_S,
        "llm_provider_error_retries": LLM_PROVIDER_ERROR_RETRIES,
        "llm_provider_error_backoff_s": LLM_PROVIDER_ERROR_BACKOFF_S,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "llm_call_schema_version": LLM_CALL_SCHEMA_VERSION,
        "uv_lock_hash": _uv_lock_hash(),
        "candidate_identity_fields": [
            "model", "task_set", "task_id", "candidate_index"],
    }


def check_or_write_manifest(path: Path, snapshot: dict) -> dict:
    missing = [field for field in MANIFEST_CRITICAL_FIELDS if field not in snapshot]
    if missing:
        raise SelfConsistencyError(f"manifest kritik alanları eksik: {missing}")
    if path.exists():
        try:
            current = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SelfConsistencyError(f"manifest okunamadı: {exc}") from exc
        mismatches = {
            field: (current.get(field), snapshot[field])
            for field in MANIFEST_CRITICAL_FIELDS
            if current.get(field) != snapshot[field]
        }
        if mismatches:
            raise SelfConsistencyError(
                "manifest uyuşmazlığı; aynı isimle farklı commit/config/model/"
                f"görev seti devam ettirilemez: {mismatches}")
        return current
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(snapshot, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8")
    return snapshot


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise SelfConsistencyError(f"{path.name}:{line_no} bozuk JSON: {exc}") from exc
    return rows


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def candidate_key(row: dict) -> tuple:
    return (row.get("model"), row.get("task_set"), row.get("task_id"),
            row.get("candidate_index"))


def candidate_run_id(model: str, task_set: str, task_id: str,
                     candidate_index: int) -> str:
    payload = [model, task_set, task_id, candidate_index]
    return f"selfcons-{_canonical_hash(payload)[:24]}"


def expected_candidate_keys(manifest: dict) -> set[tuple]:
    return {
        (manifest["model"], manifest["task_set"], task_id, index)
        for task_id in manifest["task_ids"]
        for index in range(manifest["n"])
    }


def _candidate_problems(row: dict, manifest: dict) -> list[str]:
    problems = []
    required = {
        "self_consistency_schema_version", "experiment", "arm", "model",
        "task_set", "task_id", "task_content_sha256", "candidate_index",
        "run_id", "temperature", "n", "prompt_hash", "n_test_inputs",
        "code", "signature", "oracle_status",
    }
    missing = sorted(required - set(row))
    if missing:
        return [f"eksik alanlar: {missing}"]
    task_id = row["task_id"]
    expected = {
        "self_consistency_schema_version": manifest["self_consistency_schema_version"],
        "experiment": manifest["name"],
        "arm": SELFCONS_ARM,
        "model": manifest["model"],
        "task_set": manifest["task_set"],
        "temperature": manifest["temperature"],
        "n": manifest["n"],
        "prompt_hash": manifest["prompt_hash"],
    }
    if task_id in manifest["task_content_hashes"]:
        expected["task_content_sha256"] = manifest["task_content_hashes"][task_id]
    for field, value in expected.items():
        if row.get(field) != value:
            problems.append(f"{field}: {row.get(field)!r} != {value!r}")
    index = row.get("candidate_index")
    if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < manifest["n"]:
        problems.append(f"candidate_index geçersiz: {index!r}")
    elif row.get("run_id") != candidate_run_id(
            manifest["model"], manifest["task_set"], task_id, index):
        problems.append("run_id aday kimliğiyle uyuşmuyor")
    if task_id not in manifest["task_ids"]:
        problems.append(f"beklenmeyen task_id: {task_id!r}")
    if isinstance(row.get("n_test_inputs"), bool) or not isinstance(
            row.get("n_test_inputs"), int) or row.get("n_test_inputs", -1) < 0:
        problems.append("n_test_inputs negatif olmayan tamsayı olmalı")
    for field in ("code", "signature", "oracle_status"):
        if not isinstance(row.get(field), str):
            problems.append(f"{field} metin olmalı")
    return problems


def verify_candidate_records(rows: list[dict], manifest: dict) -> dict[tuple, dict]:
    expected = expected_candidate_keys(manifest)
    by_key = {}
    for index, row in enumerate(rows):
        problems = _candidate_problems(row, manifest)
        key = candidate_key(row)
        if problems:
            raise SelfConsistencyError(
                f"stale/geçersiz aday kaydı #{index}: {problems}")
        if key not in expected:
            raise SelfConsistencyError(f"yabancı aday kaydı #{index}: {key}")
        if key in by_key:
            raise SelfConsistencyError(f"yinelenen aday kimliği: {key}")
        by_key[key] = row
    return by_key


def verify_call_records(rows: list[dict], candidates_by_key: dict[tuple, dict],
                        manifest: dict) -> dict[str, int]:
    """Çağrı logunu aday planı ve tamamlanmış adaylarla çapraz doğrular.

    Provider retry satırları (`provider_error`) ve terminal `error` satırları
    aynı run_id altında kalabilir. Tamamlanmış her aday için en az bir `ok`
    zorunludur; bir adayın LLM yanıtından sonra yerel değerlendirme kesildiyse
    resume aynı run_id altında ikinci bir `ok` üretebilir, bu nedenle `ok > 1`
    veri zehirlenmesi değil şeffaf altyapı overhead'idir ve reddedilmez.
    """
    run_to_key = {
        candidate_run_id(model, task_set, task_id, index):
            (model, task_set, task_id, index)
        for model, task_set, task_id, index in expected_candidate_keys(manifest)
    }
    ok_counts = Counter()
    for index, row in enumerate(rows):
        run_id = row.get("run_id")
        if run_id not in run_to_key:
            raise SelfConsistencyError(f"yabancı çağrı run_id #{index}: {run_id!r}")
        key = run_to_key[run_id]
        expected = {
            "schema_version": manifest["llm_call_schema_version"],
            "experiment": manifest["name"],
            "arm": SELFCONS_ARM,
            "model": key[0],
            "task_id": key[2],
            "repeat": key[3],
            "agent_role": "selfcons",
        }
        mismatches = {
            field: (row.get(field), value) for field, value in expected.items()
            if row.get(field) != value
        }
        if mismatches:
            raise SelfConsistencyError(
                f"stale/yanlış bağlamlı çağrı kaydı #{index}: {mismatches}")
        if row.get("status") not in {"ok", "provider_error", "error"}:
            raise SelfConsistencyError(
                f"bilinmeyen çağrı durumu #{index}: {row.get('status')!r}")
        if row["status"] == "ok":
            ok_counts[run_id] += 1
    for key in candidates_by_key:
        run_id = candidate_run_id(*key)
        if ok_counts[run_id] == 0:
            raise SelfConsistencyError(
                f"tamamlanmış adayın başarılı çağrı provenance'ı yok: {key}")
    return dict(ok_counts)


def build_result_record(task_id: str, candidates: list[dict], manifest: dict) -> dict:
    """Yalnız tam N güncel adaydan deterministik görev özeti üretir."""
    ordered = sorted(candidates, key=lambda row: row["candidate_index"])
    expected_indices = list(range(manifest["n"]))
    if [row["candidate_index"] for row in ordered] != expected_indices:
        raise SelfConsistencyError(
            f"{task_id}: sonuç için tam N aday yok; beklenen {expected_indices}")
    for row in ordered:
        problems = _candidate_problems(row, manifest)
        if problems or row["task_id"] != task_id:
            raise SelfConsistencyError(
                f"{task_id}: sonuç girdisi stale/geçersiz: {problems}")
    input_counts = {row["n_test_inputs"] for row in ordered}
    if len(input_counts) != 1:
        raise SelfConsistencyError(
            f"{task_id}: adaylar farklı girdi keşfi sayıları taşıyor: {input_counts}")
    score = cluster_and_score([row["signature"] for row in ordered])
    return {
        "self_consistency_schema_version": manifest["self_consistency_schema_version"],
        "experiment": manifest["name"],
        "arm": SELFCONS_ARM,
        "model": manifest["model"],
        "task_set": manifest["task_set"],
        "task_id": task_id,
        "temperature": manifest["temperature"],
        "n_test_inputs": next(iter(input_counts)),
        **score,
        "oracle_pass_rate": (
            sum(row["oracle_status"] == "passed" for row in ordered) / manifest["n"]),
        "candidates": [
            {field: row[field] for field in
             ("candidate_index", "code", "signature", "oracle_status")}
            for row in ordered
        ],
    }


def verify_result_records(rows: list[dict], candidates_by_key: dict[tuple, dict],
                          manifest: dict) -> dict[str, dict]:
    by_task = {}
    for index, row in enumerate(rows):
        task_id = row.get("task_id")
        if task_id not in manifest["task_ids"]:
            raise SelfConsistencyError(f"yabancı sonuç kaydı #{index}: {task_id!r}")
        if task_id in by_task:
            raise SelfConsistencyError(f"yinelenen sonuç kaydı: {task_id}")
        task_candidates = [
            candidates_by_key[key] for key in sorted(candidates_by_key)
            if key[2] == task_id
        ]
        if len(task_candidates) != manifest["n"]:
            raise SelfConsistencyError(
                f"{task_id}: eksik aday varken sonuç kaydı bulunuyor")
        expected = build_result_record(task_id, task_candidates, manifest)
        if row != expected:
            raise SelfConsistencyError(f"stale/geçersiz sonuç kaydı #{index}: {task_id}")
        by_task[task_id] = row
    return by_task


def generate_candidate(task: dict, model: str, temperature: float, *,
                       experiment: str | None = None, run_id: str | None = None,
                       candidate_index: int | None = None) -> str:
    response = call_model(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": task["prompt"]},
        ],
        model=model,
        temperature=temperature,
        task_id=task["task_id"],
        agent_role="selfcons",
        experiment=experiment,
        run_id=run_id,
        arm=SELFCONS_ARM if experiment else None,
        # LLM çağrı şeması 2.1'de aday indeksine karşılık gelen sürümlü alan
        # `repeat`tir. Deterministik run_id ile birlikte aday-düzeyi join sağlar;
        # yeni bir nullable alan uğruna ana deney/MAST çağrı şemasını kırmayız.
        repeat=candidate_index,
    )
    return extract_code(response.text)


def measure_task(task: dict, n: int, model: str, temperature: float) -> dict:
    """Bir görev için N aday üretir, kümeler ve oracle sonuçlarıyla loglar."""
    inputs = record_test_inputs(task)
    candidates = []
    for _ in range(n):
        code = generate_candidate(task, model, temperature)
        candidates.append({
            "code": code,
            "signature": output_signature(task, inputs, code),
            # Oracle SADECE korelasyon sütunu — agreement skoruna girmez.
            # Held-out görevlerde birincil metrikle (plus) aynı ölçüt kullanılır.
            "oracle_status": evaluate_base_plus(task, code)["plus_status"],
        })
    score = cluster_and_score([c["signature"] for c in candidates])
    return {
        "ts": datetime.now(timezone.utc).isoformat(),
        "arm": "selfcons",
        "task_id": task["task_id"],
        "model": model,
        "temperature": temperature,
        "n_test_inputs": len(inputs),
        **score,
        "oracle_pass_rate": sum(c["oracle_status"] == "passed" for c in candidates) / n,
        "candidates": candidates,
    }


def run_experiment(tasks: list[dict], manifest: dict, out_dir: Path) -> dict:
    """Aday-düzeyi resume ile bekleyen üretimleri çalıştırır.

    Mevcut aday ve sonuç dosyalarının TAMAMI ilk çağrıdan önce doğrulanır.
    Terminal sağlayıcı/transport hatasında aday kaydı yazılmaz; dolayısıyla aynı
    komut o deterministik aday kimliğini yeniden dener. Tamamlanmış aday ise bir
    daha çağrılmaz.
    """
    candidate_path = out_dir / CANDIDATE_FILE
    result_path = out_dir / RESULT_FILE
    call_path = out_dir / CALL_FILE
    candidate_rows = load_jsonl(candidate_path)
    result_rows = load_jsonl(result_path)
    candidates_by_key = verify_candidate_records(candidate_rows, manifest)
    verify_call_records(load_jsonl(call_path), candidates_by_key, manifest)
    results_by_task = verify_result_records(result_rows, candidates_by_key, manifest)
    task_map = {task["task_id"]: task for task in tasks}
    if list(task_map) != manifest["task_ids"]:
        raise SelfConsistencyError("yüklenen görev sırası/kimliği manifestle uyuşmuyor")

    failures = 0
    generated = 0
    for task_position, task_id in enumerate(manifest["task_ids"], 1):
        task = task_map[task_id]
        pending = [
            index for index in range(manifest["n"])
            if (manifest["model"], manifest["task_set"], task_id, index)
            not in candidates_by_key
        ]
        inputs = record_test_inputs(task) if pending else None
        for index in pending:
            run_id = candidate_run_id(
                manifest["model"], manifest["task_set"], task_id, index)
            try:
                code = generate_candidate(
                    task, manifest["model"], manifest["temperature"],
                    experiment=manifest["name"], run_id=run_id,
                    candidate_index=index,
                )
                row = {
                    "self_consistency_schema_version": manifest[
                        "self_consistency_schema_version"],
                    "experiment": manifest["name"],
                    "arm": SELFCONS_ARM,
                    "model": manifest["model"],
                    "task_set": manifest["task_set"],
                    "task_id": task_id,
                    "task_content_sha256": manifest["task_content_hashes"][task_id],
                    "candidate_index": index,
                    "run_id": run_id,
                    "temperature": manifest["temperature"],
                    "n": manifest["n"],
                    "prompt_hash": manifest["prompt_hash"],
                    "n_test_inputs": len(inputs),
                    "code": code,
                    "signature": output_signature(task, inputs, code),
                    "oracle_status": evaluate_base_plus(task, code)["plus_status"],
                }
                problems = _candidate_problems(row, manifest)
                if problems:
                    raise SelfConsistencyError(
                        f"üretilen aday sözleşme ihlali {task_id}/c{index}: {problems}")
                append_jsonl(candidate_path, row)
                candidates_by_key[candidate_key(row)] = row
                generated += 1
            except SelfConsistencyError:
                raise
            except Exception as exc:
                # call_model taşıma hatalarını llm_calls.jsonl'e zaten yazar.
                # Burada tamamlanmış aday uydurmayız: resume aynı kimliği dener.
                failures += 1
                print(f"UYARI {task_id}/c{index}: terminal hata, resume ile "
                      f"yeniden denenecek ({type(exc).__name__}: {exc})")

        task_candidates = [
            candidates_by_key[(manifest["model"], manifest["task_set"], task_id, index)]
            for index in range(manifest["n"])
            if (manifest["model"], manifest["task_set"], task_id, index)
            in candidates_by_key
        ]
        if len(task_candidates) == manifest["n"] and task_id not in results_by_task:
            result = build_result_record(task_id, task_candidates, manifest)
            append_jsonl(result_path, result)
            results_by_task[task_id] = result
        if task_id in results_by_task:
            result = results_by_task[task_id]
            print(f"[{task_position}/{len(tasks)}] {task_id}: "
                  f"agreement={result['agreement']:.2f} "
                  f"kumeler={result['cluster_sizes']} "
                  f"oracle_pass={result['oracle_pass_rate']:.2f}")

    # Diskteki son durum da aynı kapılardan geçsin; özet yalnız tam N'den gelir.
    final_candidates = verify_candidate_records(load_jsonl(candidate_path), manifest)
    verify_call_records(load_jsonl(call_path), final_candidates, manifest)
    final_results = verify_result_records(load_jsonl(result_path), final_candidates, manifest)
    return {
        "expected_candidates": len(expected_candidate_keys(manifest)),
        "completed_candidates": len(final_candidates),
        "completed_tasks": len(final_results),
        "expected_tasks": len(manifest["task_ids"]),
        "generated_this_run": generated,
        "terminal_failures_this_run": failures,
        "complete": (len(final_candidates) == len(expected_candidate_keys(manifest))
                     and len(final_results) == len(manifest["task_ids"])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Self-consistency ölçümü")
    parser.add_argument("--name", required=True,
                        help="deney adı (çıktı: logs/exp_<name>/); resume için aynı ad")
    parser.add_argument("--tasks", type=int, default=None,
                        help="ilk N görevi ölç (varsayılan: tümü)")
    parser.add_argument("--n", type=int, default=SELF_CONSISTENCY_N,
                        help="görev başına aday sayısı")
    # ZORUNLU: self-consistency ana koşu verisidir (EXPERIMENT_PROTOCOL.md §8 —
    # "ilişkilendirildiği ana üretici modelle AYNI model ve görev setinde
    # çalıştırılır"). Örtük bir varsayılan, korelasyonu yanlış modele bağlardı.
    parser.add_argument("--model", required=True, help=model_alias_help())
    parser.add_argument("--temperature", type=float, default=SELF_CONSISTENCY_TEMPERATURE)
    # §8.5: self-consistency, ilişkilendirildiği ana üretici modelle AYNI model
    # ve AYNI görev setinde koşmalı -- görev seti de bu yüzden açıkça istenir.
    parser.add_argument("--task-set", required=True, choices=sorted(TASK_SETS))
    args = parser.parse_args()
    # Üretici kapısı EN ÖNDE (anahtar/görev/çıktı dosyasından önce): §8.5 gereği
    # self-consistency ana koşu verisidir ve üretici-dışı bir modelle held-out
    # sette koşarsa korelasyon yanlış modele bağlanır.
    try:
        model = validate_model_for_task_set(args.model, args.task_set)
    except ValueError as e:
        sys.exit(str(e))
    try:
        commit = require_verified_git_state()
        if not (os.environ.get("OPENROUTER_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")):
            raise SelfConsistencyError(
                "API anahtarı yok — .env.example'ı .env olarak kopyalayıp doldur")
        tasks = load_all_tasks(args.task_set)[: args.tasks]
        snapshot = build_manifest_snapshot(
            name=args.name, model=model, task_set=args.task_set, tasks=tasks,
            n=args.n, temperature=args.temperature, git_commit=commit,
        )
        out_dir = LOGS_DIR / f"exp_{args.name}"
        manifest = check_or_write_manifest(out_dir / MANIFEST_FILE, snapshot)
        require_verified_git_state(manifest)
        report = run_experiment(tasks, manifest, out_dir)
    except SelfConsistencyError as exc:
        sys.exit(str(exc))

    print("\n--- Self-consistency bütünlük ---")
    print(f"aday {report['completed_candidates']}/{report['expected_candidates']} | "
          f"görev {report['completed_tasks']}/{report['expected_tasks']} | "
          f"terminal hata {report['terminal_failures_this_run']}")
    print("TAMAMLANDI" if report["complete"] else
          "EKSİK — aynı komut yalnız eksik adayları yeniden dener")
    print(f"Sonuçlar -> {out_dir / RESULT_FILE}")


if __name__ == "__main__":
    main()
