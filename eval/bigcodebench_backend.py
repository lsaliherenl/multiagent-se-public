"""BigCodeBench evaluation backend — Study 2'nin aday kod değerlendiricisi.

Bu modül `evaluation_backend == "bigcodebench_untrusted_check_v1"` etiketli
görevleri, sabitlenmiş tarifle kurulan resmî BigCodeBench imajı içinde gerçek
``bigcodebench.eval.untrusted_check`` çağrısına verir (EXPERIMENT_PROTOCOL.md
§13). EvalPlus/pilot yolu (`eval.harness.evaluate_base_plus`) bu modülü hiç
görmez. İmaj ve NLTK kaynak cildi `scripts/bigcodebench_runtime.py` ile kurulur.

Tasarım kararları (hepsi bilinçli):

* **Aday kod ve gizli test argv/stdout'a KONMAZ.** Giriş, taze bir named
  volume'a hiç başlatılmayan bir holder container üzerinden ``docker cp`` ile
  konur; sonuç yine dosyadan alınır. stdout'tan sonuç ayrıştırılmaz.
* **Görev prompt'u aday koda İKİNCİ KEZ eklenmez.** Modelin ürettiği kod
  OLDUĞU GİBİ geçer.
* **Değerlendirme ağsız koşar.** NLTK stopwords cildi salt-okunur bağlanır ve
  evaluator başlamadan ÖNCE exact manifest hash'i doğrulanır. Cilt yoksa ağ
  AÇILMAZ; ``RESOURCE_NOT_HYDRATED`` hatasıyla durulur.
* **Aday başarısızlığı ile altyapı arızası ayrı.** Altyapı arızasında istisna
  atılır (runner ``run_error`` yazar), sonuç kaydı ÜRETİLMEZ.
* **Gizli test ve altın çözüm sonuç kaydına kopyalanmaz.** ``details`` yalnız
  hash'iyle kaydedilir; BigCodeBench kayıtlarında ``traceback`` her zaman
  ``None``'dır (bilinen sınırlama).
* **İmaj kimliği kaydedilir ve koşu boyunca değişmemelidir.** Yerel build'in
  kimliği makaledeki koşununkinden farklı olabilir (apt katmanları); bu yüzden
  eşitlik şart koşulmaz, fakat kimlik her kayda yazılır ve makaledeki kimlikle
  eşleşip eşleşmediği ayrıca işaretlenir.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path

from config import (
    BIGCODEBENCH_ADAPTER_CALLABLE,
    BIGCODEBENCH_ADAPTER_MODULE,
    BIGCODEBENCH_EVAL_LIMITS,
    BIGCODEBENCH_EXPECTED_UID,
    BIGCODEBENCH_IMAGE_TAG,
    BIGCODEBENCH_OUTPUT_DIR,
    BIGCODEBENCH_PAPER_IMAGE_ID,
    BIGCODEBENCH_RESOURCE_ENGLISH_SHA256,
    BIGCODEBENCH_RESOURCE_FILE_COUNT,
    BIGCODEBENCH_RESOURCE_MANIFEST_SHA256,
    BIGCODEBENCH_RESOURCE_MOUNT,
    BIGCODEBENCH_RESOURCE_TOTAL_BYTES,
    BIGCODEBENCH_RESOURCE_VOLUME,
    BIGCODEBENCH_RUN_USER,
)

BACKEND_ID = "bigcodebench_untrusted_check_v1"
EVALUATION_METRIC = "bigcodebench_hidden_test_pass"
BACKEND_SCHEMA_VERSION = "1.0"

ADAPTER_MODULE = BIGCODEBENCH_ADAPTER_MODULE
ADAPTER_CALLABLE = BIGCODEBENCH_ADAPTER_CALLABLE
# Değerlendirmede kullanılan imaj referansı. Varsayılan, kurulum betiğinin
# verdiği etikettir; `BIGCODEBENCH_IMAGE` ortam değişkeniyle başka bir yerel
# etiket/kimlik verilebilir.
IMAGE_REF = os.environ.get("BIGCODEBENCH_IMAGE", BIGCODEBENCH_IMAGE_TAG)
RUN_USER = BIGCODEBENCH_RUN_USER
EXPECTED_UID = BIGCODEBENCH_EXPECTED_UID
OUTPUT_DIR = BIGCODEBENCH_OUTPUT_DIR
EVAL_LIMITS = dict(BIGCODEBENCH_EVAL_LIMITS)

RESOURCE_VOLUME = BIGCODEBENCH_RESOURCE_VOLUME
RESOURCE_MOUNT = BIGCODEBENCH_RESOURCE_MOUNT
RESOURCE_MANIFEST_SHA256 = BIGCODEBENCH_RESOURCE_MANIFEST_SHA256
RESOURCE_FILE_COUNT = BIGCODEBENCH_RESOURCE_FILE_COUNT
RESOURCE_TOTAL_BYTES = BIGCODEBENCH_RESOURCE_TOTAL_BYTES
RESOURCE_ENGLISH_SHA256 = BIGCODEBENCH_RESOURCE_ENGLISH_SHA256

VOLUME_PREFIX = "multiagent-se-s2eval-"
CONTAINER_TIMEOUT_S = 1800
DOCKER_CALL_TIMEOUT_S = 300

# Yerel görev kimliğinin dondurulmuş biçimi. Kaynak kimlik `BigCodeBench/655`
# slash taşır ve DOĞRUDAN dosya yolu yapılamaz.
LOCAL_ID_PATTERN = re.compile(r"^bigcodebench_[0-9]{4}$")
SOURCE_ID_PATTERN = re.compile(r"^BigCodeBench/[0-9]{1,4}$")

# Nedensel sınıflandırma sınıfları.
CLASS_PASS = "pass"
CLASS_ASSERTION = "assertion_failure"
CLASS_SYNTAX = "syntax_error"
CLASS_TIMEOUT = "timeout"
CLASS_INFRA = "infrastructure_error"

# Evaluator sınıfı -> kayıt şemasının `error_class` sözlüğü. `infrastructure_error`
# buraya GİRMEZ çünkü o bir aday başarısızlığı değildir.
ERROR_CLASS_MAP = {
    CLASS_PASS: None,
    CLASS_ASSERTION: "assertion",
    CLASS_SYNTAX: "syntax",
    CLASS_TIMEOUT: "timeout",
}

FAILURE_DOCKER = "DOCKER_UNAVAILABLE"
FAILURE_IMAGE = "IMAGE_IDENTITY_MISMATCH"
FAILURE_MUTATION = "IMAGE_MUTATED_DURING_RUN"
FAILURE_RESOURCE = "RESOURCE_NOT_HYDRATED"
FAILURE_STALE = "STALE_OUTPUT_VOLUME"
FAILURE_CHANNEL = "OUTPUT_CHANNEL_FAILURE"
FAILURE_RECIPE = "RECIPE_DEFECT"
FAILURE_TASK = "TASK_PAYLOAD_DEFECT"


class BigCodeBenchBackendError(RuntimeError):
    """Aday başarısızlığı DEĞİL: operasyonel/tarif düzeyinde arıza."""

    def __init__(self, failure_code: str, message: str, *, detail=None):
        super().__init__(f"{failure_code}: {message}")
        self.failure_code = failure_code
        self.detail = detail or {}


def _canonical(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"))


def _fingerprint(value) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------
# Yerel <-> kaynak kimlik kuralı (dondurulmuş, iki yönlü)
# --------------------------------------------------------------------------

def local_task_id(source_task_id: str) -> str:
    """`BigCodeBench/655` -> `bigcodebench_0655`.

    Kaynak kimlikte baştaki sıfır YOKTUR (dataset'te doğrulandı), bu yüzden
    dört haneye doldurma çakışmasız ve tam geri çevrilebilirdir.
    """
    if not isinstance(source_task_id, str) or not SOURCE_ID_PATTERN.match(source_task_id):
        raise BigCodeBenchBackendError(
            FAILURE_TASK, f"kaynak görev kimliği biçimsiz: {source_task_id!r}")
    number = source_task_id.split("/", 1)[1]
    if number != str(int(number)):
        raise BigCodeBenchBackendError(
            FAILURE_TASK, f"kaynak kimlikte baştaki sıfır: {source_task_id!r}")
    return f"bigcodebench_{int(number):04d}"


def source_task_id(local_id: str) -> str:
    """`bigcodebench_0655` -> `BigCodeBench/655`."""
    if not isinstance(local_id, str) or not LOCAL_ID_PATTERN.match(local_id):
        raise BigCodeBenchBackendError(
            FAILURE_TASK, f"yerel görev kimliği biçimsiz: {local_id!r}")
    return f"BigCodeBench/{int(local_id.rsplit('_', 1)[1])}"


def assert_safe_local_id(local_id: str) -> str:
    """Dosya yoluna girmeden önceki zorunlu kapı (path traversal savunması)."""
    if not isinstance(local_id, str) or not LOCAL_ID_PATTERN.match(local_id):
        raise BigCodeBenchBackendError(
            FAILURE_TASK, f"güvenli olmayan yerel kimlik: {local_id!r}")
    # Geri çevrilebilirlik kapının parçasıdır: biçim doğru ama tersi kaynak
    # kimliğe dönmeyen bir ad, manifestteki iki yönlü eşlemeyi bozar.
    if local_task_id(source_task_id(local_id)) != local_id:
        raise BigCodeBenchBackendError(FAILURE_TASK, "kimlik geri çevrilemedi")
    return local_id


# --------------------------------------------------------------------------
# Container içinde koşan ölçüm programı
# --------------------------------------------------------------------------

RESOURCE_PROBE = r'''import hashlib,json,pathlib
root=pathlib.Path("/nltk_data")
files=[{"path":p.relative_to(root).as_posix(),
        "sha256":hashlib.sha256(p.read_bytes()).hexdigest(),
        "bytes":p.stat().st_size}
       for p in sorted(root.rglob("*")) if p.is_file()]
core={"files":files,"file_count":len(files),
      "total_bytes":sum(x["bytes"] for x in files)}
print(json.dumps({"manifest_sha256":hashlib.sha256(
    json.dumps(core,sort_keys=True,separators=(",",":")).encode()).hexdigest(),
    "file_count":core["file_count"],"total_bytes":core["total_bytes"],
    "english_sha256":next(x["sha256"] for x in files
      if x["path"]=="corpora/stopwords/english")},sort_keys=True))
'''


def probe_source() -> str:
    """Tek bir aday değerlendirmesini yürüten, stdout ayrıştırılmayan program."""
    return r'''import hashlib, json, os, socket, sys, time, traceback

payload_path, result_path = sys.argv[1:3]

def canonical(x):
    return json.dumps(x, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

def fingerprint(x):
    return hashlib.sha256(canonical(x).encode("utf-8")).hexdigest()

def atomic(path, payload):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, sort_keys=True, indent=2)
        fh.write("\n"); fh.flush(); os.fsync(fh.fileno())
    os.replace(tmp, path)

def network_witness():
    probes = []
    checks = [
        ("tcp_1.1.1.1_443", lambda: socket.create_connection(("1.1.1.1", 443), 2)),
        ("tcp_8.8.8.8_53", lambda: socket.create_connection(("8.8.8.8", 53), 2)),
        ("dns_pypi", lambda: socket.getaddrinfo("pypi.org", 443)),
    ]
    for name, fn in checks:
        try:
            obj = fn()
            if hasattr(obj, "close"): obj.close()
            probes.append({"probe": name, "denied": False, "error": None})
        except BaseException as exc:
            probes.append({"probe": name, "denied": True,
                           "error_type": type(exc).__name__,
                           "error": str(exc)[:200]})
    interfaces = sorted(os.listdir("/sys/class/net"))
    return {"probes": probes, "network_denied": all(p["denied"] for p in probes),
            "interfaces": interfaces, "only_loopback": interfaces == ["lo"]}

payload = json.load(open(payload_path, encoding="utf-8"))
core = {k: v for k, v in payload.items() if k != "payload_sha256"}
result = {"backend_schema_version": payload["backend_schema_version"],
          "evaluation_id": payload["evaluation_id"],
          "payload_sha256": payload["payload_sha256"],
          "payload_verified": fingerprint(core) == payload["payload_sha256"],
          "completed": False, "uid": os.getuid(), "gid": os.getgid(),
          "status": None, "details_sha256": None, "details_kind": None,
          "pass_constant": None, "fail_constant": None, "timeout_constant": None,
          "error_type": None, "error": None, "traceback": None,
          "duration_s": None, "adapter": None, "network_witness": None}
atomic(result_path, result)
result["network_witness"] = network_witness()
atomic(result_path, result)
if not result["payload_verified"]:
    result["error_type"] = "PayloadHashMismatch"
    result["error"] = "payload hash mismatch"
    atomic(result_path, result); raise SystemExit(0)
try:
    import bigcodebench
    from bigcodebench import eval as bceval
    fn = getattr(bceval, payload["adapter_callable"])
    result["adapter"] = {"module": payload["adapter_module"],
                         "qualname": payload["adapter_callable"],
                         "package_file": bigcodebench.__file__,
                         "adapter_file": bceval.__file__}
    result["pass_constant"] = bceval.PASS
    result["fail_constant"] = bceval.FAIL
    result["timeout_constant"] = getattr(bceval, "TIMEOUT", None)
    lim = payload["limits"]
    started = time.time()
    status, details = fn(
        payload["candidate_code"], payload["test"], payload["entry_point"],
        lim["max_as_limit"], lim["max_data_limit"], lim["max_stack_limit"],
        min_time_limit=lim["min_time_limit"], gt_time_limit=lim["gt_time_limit"])
    result["duration_s"] = round(time.time() - started, 3)
    result["status"] = status
    try:
        json.dumps(details)
        normalized = details
        result["details_kind"] = "json"
    except BaseException:
        normalized = {"repr": repr(details)[:4000]}
        result["details_kind"] = "repr"
    result["details_sha256"] = fingerprint(normalized)
    # Gizli test metni ve altın çözüm ASLA dışarı yazılmaz: yalnız hash ve
    # sınıflandırma için gereken imza taranır.
    blob = canonical(normalized)
    result["details_signals"] = {
        "syntax_error": ("SyntaxError" in blob or "IndentationError" in blob),
        "length": len(blob),
    }
    result["completed"] = True
except BaseException as exc:
    result["error_type"] = type(exc).__name__
    result["error"] = str(exc)[:500]
    result["traceback"] = traceback.format_exc()[-4000:]
atomic(result_path, result)
'''


PROBE_SOURCE_SHA256 = hashlib.sha256(probe_source().encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------
# Docker taşıma katmanı (enjekte edilebilir)
# --------------------------------------------------------------------------

def _default_docker(argv, *, timeout, label):
    try:
        proc = subprocess.run(["docker", *argv], capture_output=True, text=True,
                              encoding="utf-8", errors="replace", timeout=timeout)
    except FileNotFoundError as exc:
        raise BigCodeBenchBackendError(FAILURE_DOCKER, f"docker yok: {exc}") from exc
    except subprocess.TimeoutExpired as exc:
        return {"label": label, "argv": ["docker", *argv], "exit_code": None,
                "timed_out": True, "stdout_tail": "", "stderr_tail": str(exc)[:500]}
    return {"label": label, "argv": ["docker", *argv],
            "exit_code": proc.returncode, "timed_out": False,
            "stdout_tail": (proc.stdout or "")[-8000:],
            "stderr_tail": (proc.stderr or "")[-4000:]}


class DockerChannel:
    """Komut makbuzunu biriktiren ince sarmalayıcı."""

    def __init__(self, runner=None):
        self._runner = runner or _default_docker
        self.receipt: list[dict] = []

    def __call__(self, argv, *, timeout=DOCKER_CALL_TIMEOUT_S, label=None):
        record = self._runner(argv, timeout=timeout, label=label)
        self.receipt.append({k: record.get(k) for k in
                             ("label", "argv", "exit_code", "timed_out")})
        return record


# --------------------------------------------------------------------------
# Kapılar
# --------------------------------------------------------------------------

def _image_identity(docker) -> str:
    """İmajın içerik kimliği (`docker image inspect` Id alanı)."""
    record = docker(["image", "inspect", IMAGE_REF], label="image inspect")
    if record["exit_code"]:
        raise BigCodeBenchBackendError(
            FAILURE_IMAGE, f"imaj bulunamadı: {IMAGE_REF} "
                           "(önce scripts/bigcodebench_runtime.py build)",
            detail={"stderr": record["stderr_tail"]})
    try:
        image_id = json.loads(record["stdout_tail"])[0]["Id"]
    except Exception as exc:
        raise BigCodeBenchBackendError(
            FAILURE_IMAGE, f"imaj kimliği okunamadı: {exc}") from exc
    if not isinstance(image_id, str) or not image_id.startswith("sha256:"):
        raise BigCodeBenchBackendError(FAILURE_IMAGE, f"beklenmeyen imaj kimliği: {image_id!r}")
    return image_id


def verify_resource_volume(docker) -> dict:
    """Stopwords cildini salt-okunur bağlayıp exact kimliğini doğrular.

    Cilt yoksa AĞ AÇILMAZ ve sessiz indirme YAPILMAZ.
    """
    present = docker(["volume", "inspect", RESOURCE_VOLUME],
                     label="resource volume inspect")
    if present["exit_code"]:
        raise BigCodeBenchBackendError(
            FAILURE_RESOURCE,
            f"kaynak cilt yok: {RESOURCE_VOLUME} (ağ açılmadı, indirme yapılmadı)")
    observed_run = docker(
        ["run", "--rm", "--network", "none", "--user", RUN_USER,
         "-v", f"{RESOURCE_VOLUME}:/nltk_data:ro", "--entrypoint", "python",
         IMAGE_REF, "-c", RESOURCE_PROBE],
        label="resource identity")
    try:
        observed = json.loads(observed_run["stdout_tail"].strip())
    except Exception:
        observed = {}
    expected = {"manifest_sha256": RESOURCE_MANIFEST_SHA256,
                "file_count": RESOURCE_FILE_COUNT,
                "total_bytes": RESOURCE_TOTAL_BYTES,
                "english_sha256": RESOURCE_ENGLISH_SHA256}
    if observed_run["exit_code"] or observed != expected:
        raise BigCodeBenchBackendError(
            FAILURE_RESOURCE, "kaynak cilt kimliği uyuşmuyor",
            detail={"observed": observed, "expected": expected})
    return expected


def build_payload(task: dict, candidate_code: str, *, evaluation_id: str) -> dict:
    """Container'a gidecek giriş. Prompt aday koda İKİNCİ KEZ eklenmez."""
    for field in ("task_id", "source_task_id", "entry_point", "test_code"):
        if not isinstance(task.get(field), str) or not task[field]:
            raise BigCodeBenchBackendError(
                FAILURE_TASK, f"görev alanı eksik/biçimsiz: {field}")
    if task.get("evaluation_backend") != BACKEND_ID:
        raise BigCodeBenchBackendError(
            FAILURE_TASK, f"görev bu backend'e ait değil: "
                          f"{task.get('evaluation_backend')!r}")
    assert_safe_local_id(task["task_id"])
    if source_task_id(task["task_id"]) != task["source_task_id"]:
        raise BigCodeBenchBackendError(
            FAILURE_TASK, "yerel ve kaynak görev kimliği eşleşmiyor")
    if not isinstance(candidate_code, str):
        raise BigCodeBenchBackendError(FAILURE_TASK, "aday kod metin değil")
    core = {
        "backend_schema_version": BACKEND_SCHEMA_VERSION,
        "backend_id": BACKEND_ID,
        "evaluation_id": evaluation_id,
        "task_id": task["task_id"],
        "source_task_id": task["source_task_id"],
        "adapter_module": ADAPTER_MODULE,
        "adapter_callable": ADAPTER_CALLABLE,
        "candidate_code": candidate_code,
        "test": task["test_code"],
        "entry_point": task["entry_point"],
        "limits": dict(EVAL_LIMITS),
    }
    return {**core, "payload_sha256": _fingerprint(core)}


def _classify(raw: dict) -> str:
    """Dondurulmuş nedensel sınıflandırma kuralı, details METNİ olmadan.

    `details`in JSON metni `SyntaxError` için container İÇİNDE taranır ve
    yalnız boolean sinyal dışarı çıkar; gizli test metni artefakta hiç girmez.
    """
    if not isinstance(raw, dict):
        return CLASS_INFRA
    if raw.get("error_type") or not raw.get("completed"):
        return CLASS_INFRA
    status = raw.get("status")
    if status is None:
        return CLASS_INFRA
    if status == raw.get("timeout_constant") or status == "timeout":
        return CLASS_TIMEOUT
    if status == raw.get("pass_constant"):
        return CLASS_PASS
    if (raw.get("details_signals") or {}).get("syntax_error"):
        return CLASS_SYNTAX
    if status == raw.get("fail_constant"):
        return CLASS_ASSERTION
    return CLASS_INFRA


def evaluate_bigcodebench(task: dict, candidate_code: str, *,
                          docker_runner=None, keep_receipt: bool = False) -> dict:
    """Bir aday kodu resmî image içinde değerlendirir ve kayıt alanlarını üretir.

    Altyapı/tarif arızasında ``BigCodeBenchBackendError`` ATAR — çağıran taraf
    bunu ``run_error`` olarak kaydeder; aday başarısızlığıyla karıştırılmaz.
    """
    docker = docker_runner if isinstance(docker_runner, DockerChannel) \
        else DockerChannel(docker_runner)
    evaluation_id = uuid.uuid4().hex
    payload = build_payload(task, candidate_code, evaluation_id=evaluation_id)
    volume = f"{VOLUME_PREFIX}{evaluation_id[:16]}"
    holder = f"{volume}-holder"
    channel = Path(tempfile.mkdtemp(prefix="s2eval-"))
    image_before = _image_identity(docker)
    resource = verify_resource_volume(docker)
    try:
        (channel / "payload.json").write_bytes(
            (json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2)
             + "\n").encode("utf-8"))
        (channel / "probe.py").write_bytes(probe_source().encode("utf-8"))

        stale = docker(["volume", "inspect", volume], label="output volume absence")
        if stale["exit_code"] == 0:
            raise BigCodeBenchBackendError(FAILURE_STALE, f"bayat cilt: {volume}")
        created = docker(["volume", "create", volume], label="output volume create")
        if created["exit_code"]:
            raise BigCodeBenchBackendError(
                FAILURE_CHANNEL, "çıktı cildi kurulamadı",
                detail={"stderr": created["stderr_tail"]})
        _inject(docker, holder, volume, channel)
        run = docker(
            ["run", "--rm", "--network", "none", "--user", RUN_USER,
             "-e", f"NLTK_DATA={RESOURCE_MOUNT}",
             "-v", f"{volume}:{OUTPUT_DIR}",
             "-v", f"{RESOURCE_VOLUME}:{RESOURCE_MOUNT}:ro",
             "--entrypoint", "python", IMAGE_REF,
             f"{OUTPUT_DIR}/probe.py", f"{OUTPUT_DIR}/payload.json",
             f"{OUTPUT_DIR}/result.json"],
            timeout=CONTAINER_TIMEOUT_S, label="evaluate candidate")
        if run["timed_out"]:
            raise BigCodeBenchBackendError(
                FAILURE_CHANNEL, "container duvar süresi aşıldı")
        if run["exit_code"]:
            raise BigCodeBenchBackendError(
                FAILURE_CHANNEL, f"container exit {run['exit_code']}",
                detail={"stderr": run["stderr_tail"]})
        raw = _extract(docker, holder, volume, channel)
    finally:
        docker(["rm", "-f", holder], label="cleanup holder")
        docker(["volume", "rm", "-f", volume], label="cleanup output volume")
        shutil.rmtree(channel, ignore_errors=True)

    image_after = _image_identity(docker)
    if image_before != image_after:
        raise BigCodeBenchBackendError(FAILURE_MUTATION,
                                       "image ölçüm sırasında değişti")
    return _to_record(raw, payload=payload, resource=resource,
                      image_identity=image_before,
                      receipt=list(docker.receipt) if keep_receipt else None)


def _inject(docker, holder: str, volume: str, channel: Path) -> None:
    """Hiç BAŞLATILMAYAN holder üzerinden giriş dosyalarını cilde koyar."""
    created = docker(["create", "--name", holder, "--network", "none",
                      "--user", RUN_USER, "-v", f"{volume}:{OUTPUT_DIR}",
                      "--entrypoint", "true", IMAGE_REF],
                     label="input holder")
    if created["exit_code"]:
        raise BigCodeBenchBackendError(
            FAILURE_CHANNEL, "giriş holder'ı kurulamadı",
            detail={"stderr": created["stderr_tail"]})
    for name in ("payload.json", "probe.py"):
        copied = docker(["cp", str(channel / name),
                         f"{holder}:{OUTPUT_DIR}/{name}"], label=f"copy in {name}")
        if copied["exit_code"]:
            raise BigCodeBenchBackendError(
                FAILURE_CHANNEL, f"giriş kopyalanamadı: {name}",
                detail={"stderr": copied["stderr_tail"]})
    docker(["rm", "-f", holder], label="remove input holder")


def _extract(docker, holder: str, volume: str, channel: Path) -> dict:
    created = docker(["create", "--name", holder, "--network", "none",
                      "--user", RUN_USER, "-v", f"{volume}:{OUTPUT_DIR}",
                      "--entrypoint", "true", IMAGE_REF],
                     label="output holder")
    if created["exit_code"]:
        raise BigCodeBenchBackendError(
            FAILURE_CHANNEL, "çıktı holder'ı kurulamadı",
            detail={"stderr": created["stderr_tail"]})
    copied = docker(["cp", f"{holder}:{OUTPUT_DIR}/result.json",
                     str(channel / "result.json")], label="copy out result.json")
    if copied["exit_code"]:
        raise BigCodeBenchBackendError(
            FAILURE_CHANNEL, "sonuç dosyası alınamadı",
            detail={"stderr": copied["stderr_tail"]})
    try:
        return json.loads((channel / "result.json").read_text(encoding="utf-8"))
    except Exception as exc:
        raise BigCodeBenchBackendError(
            FAILURE_CHANNEL, f"sonuç dosyası ayrıştırılamadı: {exc}") from exc


def _to_record(raw: dict, *, payload: dict, resource: dict,
               image_identity: str, receipt) -> dict:
    if raw.get("payload_sha256") != payload["payload_sha256"]:
        raise BigCodeBenchBackendError(FAILURE_RECIPE, "payload hash ayrışması")
    if raw.get("evaluation_id") != payload["evaluation_id"]:
        raise BigCodeBenchBackendError(FAILURE_RECIPE, "evaluation_id ayrışması")
    if not raw.get("payload_verified"):
        raise BigCodeBenchBackendError(FAILURE_RECIPE,
                                       "container payload'ı doğrulayamadı")
    if raw.get("uid") != EXPECTED_UID:
        raise BigCodeBenchBackendError(
            FAILURE_RECIPE, f"container uid {raw.get('uid')!r}, "
                            f"beklenen {EXPECTED_UID}")
    witness = raw.get("network_witness") or {}
    if not witness.get("network_denied") or not witness.get("only_loopback"):
        raise BigCodeBenchBackendError(FAILURE_RECIPE, "ağ witness'ı geçmedi",
                                       detail={"witness": witness})
    klass = _classify(raw)
    if klass == CLASS_INFRA:
        raise BigCodeBenchBackendError(
            FAILURE_RECIPE, "evaluator altyapı hatası verdi",
            detail={"error_type": raw.get("error_type"),
                    "error": raw.get("error")})
    passed = klass == CLASS_PASS
    status = "passed" if passed else ("timeout" if klass == CLASS_TIMEOUT else "failed")
    error_class = ERROR_CLASS_MAP[klass]
    duration = raw.get("duration_s") or 0.0
    fields = {
        "status": status,
        "pass": passed,
        "error_class": error_class,
        # Gizli test metni sızdırmamak için BigCodeBench kayıtlarında
        # traceback her zaman None (bilinen sınırlama).
        "traceback": None,
        "duration_s": duration,
    }
    record = {
        "task_id": payload["task_id"],
        "source_task_id": payload["source_task_id"],
        "primary_status": status,
        "primary_pass": passed,
        "primary_error_class": error_class,
        "primary_duration_s": duration,
        "primary_details_sha256": raw.get("details_sha256"),
        "evaluation_metric": EVALUATION_METRIC,
        "evaluation_backend": BACKEND_ID,
        "evaluator_failure_class": klass,
        # Uyumluluk aynası: mevcut şema base_/plus_ alanları ister. Değerler
        # AYNI ölçümdür; ikinci bir test kümesi YOKTUR.
        **{f"base_{k}": v for k, v in fields.items()},
        **{f"plus_{k}": v for k, v in fields.items()},
        "base_plus_available": False,
        "plus_pass_is_compatibility_alias": True,
        "status": status,
        "error_class": error_class,
        "traceback": None,
        "duration_s": duration,
        "evaluator_provenance": {
            "backend_schema_version": BACKEND_SCHEMA_VERSION,
            "image_ref": IMAGE_REF,
            "image_id": image_identity,
            "matches_paper_image_id": image_identity == BIGCODEBENCH_PAPER_IMAGE_ID,
            "adapter_module": ADAPTER_MODULE,
            "adapter_callable": ADAPTER_CALLABLE,
            "adapter_file": (raw.get("adapter") or {}).get("adapter_file"),
            "probe_source_sha256": PROBE_SOURCE_SHA256,
            "task_payload_sha256": payload["payload_sha256"],
            "limits": dict(EVAL_LIMITS),
            "resource_manifest_sha256": resource["manifest_sha256"],
            "network": "none",
            "uid": raw.get("uid"),
            "network_denied": True,
            "only_loopback": True,
        },
    }
    if receipt is not None:
        record["evaluator_command_receipt"] = receipt
    return record


__all__ = [
    "BACKEND_ID", "BACKEND_SCHEMA_VERSION", "EVALUATION_METRIC",
    "BigCodeBenchBackendError", "DockerChannel", "EVAL_LIMITS",
    "IMAGE_REF", "PROBE_SOURCE_SHA256", "assert_safe_local_id",
    "build_payload", "evaluate_bigcodebench", "local_task_id", "probe_source",
    "source_task_id", "verify_resource_volume",
]
