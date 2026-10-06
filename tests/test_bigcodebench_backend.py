"""BigCodeBench evaluation backend'inin çevrimdışı sözleşme testleri.

Hiçbir test gerçek Docker, ağ veya model çağırmaz: taşıma katmanı enjekte
edilir.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from eval import bigcodebench_backend as backend
from eval import harness, result_schema

ROOT = Path(__file__).resolve().parents[1]
# Makaledeki Study 2 koşusunda container içinde çalışan ölçüm programının hash'i.
PAPER_PROBE_SOURCE_SHA256 = "5226df8a6629fa164bef1ad708f9537fa5c36654b54db0bbfaea179c41f4b6a2"


# --------------------------------------------------------------------------
# Sahte Docker taşıması
# --------------------------------------------------------------------------

def _container_ref(spec: str) -> bool:
    r"""`holder:/path` mı yoksa Windows `C:\path` mi?

    Windows sürücü harfi tek karakterdir; container adı değildir. Bu ayrım
    olmadan çıkış kopyası yanlışlıkla giriş sanılır.
    """
    head = spec.split(":", 1)[0]
    return ":" in spec and len(head) > 1


class FakeDocker:
    """Gerçek `docker` sözleşmesini taklit eden, dosya yazan sahte taşıma."""

    def __init__(self, *, probe_result=None, image_ids=("A", "A"),
                 resource_present=True, resource_payload=None,
                 stale_volume=False, run_exit=0):
        self.probe_result = probe_result
        self.image_ids = list(image_ids)
        self.resource_present = resource_present
        self.resource_payload = resource_payload
        self.stale_volume = stale_volume
        self.run_exit = run_exit
        self.calls: list[list[str]] = []
        self._payload = None

    def _ok(self, label, stdout=""):
        return {"label": label, "argv": ["docker"], "exit_code": 0,
                "timed_out": False, "stdout_tail": stdout, "stderr_tail": ""}

    def _err(self, label, stderr="hata"):
        return {"label": label, "argv": ["docker"], "exit_code": 1,
                "timed_out": False, "stdout_tail": "", "stderr_tail": stderr}

    def __call__(self, argv, *, timeout, label):
        self.calls.append(list(argv))
        head = argv[0]
        if head == "image":
            value = self.image_ids.pop(0) if self.image_ids else "A"
            if value is None:
                return self._err(label, "no such image")
            return self._ok(label, json.dumps([{"Id": _image_id(value)}]))
        if head == "volume" and argv[1] == "inspect":
            if argv[2] == backend.RESOURCE_VOLUME:
                return (self._ok(label, "[]") if self.resource_present
                        else self._err(label, "no such volume"))
            return self._ok(label) if self.stale_volume else self._err(label)
        if head == "volume":                      # create / rm
            return self._ok(label)
        if head == "run":
            payload = self.resource_payload if self.resource_payload is not None \
                else {"manifest_sha256": backend.RESOURCE_MANIFEST_SHA256,
                      "file_count": backend.RESOURCE_FILE_COUNT,
                      "total_bytes": backend.RESOURCE_TOTAL_BYTES,
                      "english_sha256": backend.RESOURCE_ENGLISH_SHA256}
            if "-c" in argv:                       # kaynak kimlik probe'u
                return self._ok(label, json.dumps(payload, sort_keys=True))
            if self.run_exit:
                return self._err(label, f"exit {self.run_exit}")
            return self._ok(label)
        if head in ("create", "rm"):
            return self._ok(label)
        if head == "cp":
            source, target = argv[1], argv[2]
            if _container_ref(target):             # giriş
                if Path(source).name == "payload.json":
                    self._payload = json.loads(Path(source).read_text(encoding="utf-8"))
                return self._ok(label)
            Path(target).write_text(                # çıkış
                json.dumps(self._result(), ensure_ascii=False), encoding="utf-8")
            return self._ok(label)
        raise AssertionError(f"beklenmeyen docker komutu: {argv}")

    def _result(self):
        if self.probe_result is not None:
            return dict(self.probe_result)
        return default_probe_result(self._payload)


def _image_id(tag: str) -> str:
    """Sahte imaj adını `docker image inspect` biçiminde bir içerik kimliğine çevirir."""
    return "sha256:" + hashlib.sha256(tag.encode("utf-8")).hexdigest()


def default_probe_result(payload, **overrides):
    result = {
        "backend_schema_version": backend.BACKEND_SCHEMA_VERSION,
        "evaluation_id": payload["evaluation_id"],
        "payload_sha256": payload["payload_sha256"],
        "payload_verified": True,
        "completed": True,
        "uid": backend.EXPECTED_UID,
        "gid": 1000,
        "status": "pass",
        "pass_constant": "pass",
        "fail_constant": "fail",
        "timeout_constant": "timeout",
        "details_sha256": "0" * 64,
        "details_kind": "json",
        "details_signals": {"syntax_error": False, "length": 2},
        "error_type": None,
        "error": None,
        "traceback": None,
        "duration_s": 0.5,
        "adapter": {"module": backend.ADAPTER_MODULE,
                    "qualname": backend.ADAPTER_CALLABLE,
                    "adapter_file": "/app/bigcodebench/eval/__init__.py"},
        "network_witness": {"network_denied": True, "only_loopback": True,
                            "interfaces": ["lo"], "probes": []},
    }
    result.update(overrides)
    return result


def gorev(**overrides):
    task = {
        "task_id": "bigcodebench_0655",
        "source_task_id": "BigCodeBench/655",
        "entry_point": "task_func",
        "test_code": "import unittest\n",
        "evaluation_backend": backend.BACKEND_ID,
        "prompt": "spec",
    }
    task.update(overrides)
    return task


def _kosur(task=None, code="def task_func():\n    pass\n", **kwargs):
    fake = FakeDocker(**kwargs)
    kanal = backend.DockerChannel(fake)
    return backend.evaluate_bigcodebench(task or gorev(), code,
                                         docker_runner=kanal), fake


# --------------------------------------------------------------------------
# Kimlik kuralı (path traversal dahil)
# --------------------------------------------------------------------------

def test_yerel_kimlik_dolgulu_ve_geri_cevrilebilir():
    assert backend.local_task_id("BigCodeBench/655") == "bigcodebench_0655"
    assert backend.local_task_id("BigCodeBench/13") == "bigcodebench_0013"
    assert backend.local_task_id("BigCodeBench/1129") == "bigcodebench_1129"
    for source in ("BigCodeBench/13", "BigCodeBench/655", "BigCodeBench/1129"):
        assert backend.source_task_id(backend.local_task_id(source)) == source


@pytest.mark.parametrize("kotu", [
    "bigcodebench_0655/../../etc/passwd",
    "../bigcodebench_0655",
    "bigcodebench_0655.json",
    "BigCodeBench/655",
    "bigcodebench_655",
    "bigcodebench_00655",
    "",
    None,
])
def test_guvensiz_yerel_kimlik_reddedilir(kotu):
    """Slash, `..`, uzantı ve yanlış dolgu dosya yoluna GİREMEZ."""
    with pytest.raises(backend.BigCodeBenchBackendError):
        backend.assert_safe_local_id(kotu)


def test_slash_iceren_kaynak_kimlik_dogrudan_yol_yapilmaz():
    yerel = backend.local_task_id("BigCodeBench/655")
    assert "/" not in yerel and "\\" not in yerel
    assert Path(yerel).name == yerel


# --------------------------------------------------------------------------
# Payload sözleşmesi
# --------------------------------------------------------------------------

def test_prompt_aday_koda_ikinci_kez_eklenmez():
    payload = backend.build_payload(gorev(), "KOD", evaluation_id="x")
    assert payload["candidate_code"] == "KOD"
    assert "spec" not in payload["candidate_code"]
    assert "prompt" not in payload


def test_payload_hash_kendi_icerigine_baglidir():
    a = backend.build_payload(gorev(), "KOD", evaluation_id="x")
    b = backend.build_payload(gorev(), "BASKA", evaluation_id="x")
    assert a["payload_sha256"] != b["payload_sha256"]


def test_etiketsiz_gorev_backend_e_giremez():
    with pytest.raises(backend.BigCodeBenchBackendError):
        backend.build_payload(gorev(evaluation_backend=None), "x",
                              evaluation_id="y")


def test_kimlik_ciftinin_tutarsizligi_reddedilir():
    with pytest.raises(backend.BigCodeBenchBackendError):
        backend.build_payload(gorev(source_task_id="BigCodeBench/999"), "x",
                              evaluation_id="y")


# --------------------------------------------------------------------------
# Container tarifi
# --------------------------------------------------------------------------

def test_aday_kod_ve_test_argv_ye_konmaz():
    kod = "GIZLI_ADAY_KODU"
    kayit, fake = _kosur(task=gorev(test_code="GIZLI_TEST_KODU"), code=kod)
    duz = " ".join(" ".join(str(x) for x in argv) for argv in fake.calls)
    assert kod not in duz
    assert "GIZLI_TEST_KODU" not in duz
    assert kayit["primary_pass"] is True


def test_container_bayraklari_ve_kullanici():
    _, fake = _kosur()
    kosu = [a for a in fake.calls
            if a[0] == "run" and "--entrypoint" in a and "-c" not in a]
    assert len(kosu) == 1
    argv = kosu[0]
    assert "--rm" in argv and "--network" in argv and "none" in argv
    assert argv[argv.index("--user") + 1] == backend.RUN_USER
    assert argv[argv.index("--entrypoint") + 1] == "python"
    assert backend.IMAGE_REF in argv
    assert f"{backend.RESOURCE_VOLUME}:{backend.RESOURCE_MOUNT}:ro" in argv


def test_hicbir_build_veya_pull_yapilmaz():
    _, fake = _kosur()
    assert not [a for a in fake.calls if a[0] in ("build", "pull")]


def test_holder_hic_baslatilmaz():
    """Giriş/çıkış kanalı `create` ile kurulur; holder `start` EDİLMEZ."""
    _, fake = _kosur()
    assert not [a for a in fake.calls if a[0] == "start"]
    assert [a for a in fake.calls if a[0] == "create"]


def test_host_bind_mount_yok():
    _, fake = _kosur()
    for argv in fake.calls:
        for i, token in enumerate(argv):
            if token == "-v":
                kaynak = argv[i + 1].split(":")[0]
                assert kaynak.startswith(backend.VOLUME_PREFIX) \
                    or kaynak == backend.RESOURCE_VOLUME


# --------------------------------------------------------------------------
# Negatif kontroller
# --------------------------------------------------------------------------

def test_kaynak_cilt_yoksa_ag_acilmaz():
    with pytest.raises(backend.BigCodeBenchBackendError) as exc:
        _kosur(resource_present=False)
    assert exc.value.failure_code == backend.FAILURE_RESOURCE
    assert "indirme yapılmadı" in str(exc.value)


def test_kaynak_hash_sapmasi_durdurur():
    with pytest.raises(backend.BigCodeBenchBackendError) as exc:
        _kosur(resource_payload={"manifest_sha256": "deadbeef", "file_count": 1,
                                 "total_bytes": 1, "english_sha256": "x"})
    assert exc.value.failure_code == backend.FAILURE_RESOURCE


def test_yanlis_image_durdurur():
    with pytest.raises(backend.BigCodeBenchBackendError) as exc:
        _kosur(image_ids=[None])
    assert exc.value.failure_code == backend.FAILURE_IMAGE


def test_image_mutasyonu_durdurur():
    with pytest.raises(backend.BigCodeBenchBackendError) as exc:
        _kosur(image_ids=["A", "B"])
    assert exc.value.failure_code == backend.FAILURE_MUTATION


def test_bayat_cikti_cildi_durdurur():
    with pytest.raises(backend.BigCodeBenchBackendError) as exc:
        _kosur(stale_volume=True)
    assert exc.value.failure_code == backend.FAILURE_STALE


def test_payload_hash_sapmasi_durdurur():
    fake = FakeDocker()
    kanal = backend.DockerChannel(fake)
    payload_bozuk = default_probe_result(
        {"evaluation_id": "x", "payload_sha256": "y"})
    fake.probe_result = payload_bozuk
    with pytest.raises(backend.BigCodeBenchBackendError) as exc:
        backend.evaluate_bigcodebench(gorev(), "kod", docker_runner=kanal)
    assert exc.value.failure_code == backend.FAILURE_RECIPE


def test_ag_witness_gecmezse_sonuc_uretilmez():
    fake = FakeDocker()
    kanal = backend.DockerChannel(fake)
    orijinal = fake._result

    def bozuk():
        ham = orijinal()
        ham["network_witness"] = {"network_denied": False, "only_loopback": False,
                                  "interfaces": ["lo", "eth0"], "probes": []}
        return ham

    fake._result = bozuk
    with pytest.raises(backend.BigCodeBenchBackendError) as exc:
        backend.evaluate_bigcodebench(gorev(), "kod", docker_runner=kanal)
    assert exc.value.failure_code == backend.FAILURE_RECIPE


def test_yanlis_uid_durdurur():
    fake = FakeDocker()
    kanal = backend.DockerChannel(fake)
    orijinal = fake._result
    fake._result = lambda: {**orijinal(), "uid": 0}
    with pytest.raises(backend.BigCodeBenchBackendError) as exc:
        backend.evaluate_bigcodebench(gorev(), "kod", docker_runner=kanal)
    assert exc.value.failure_code == backend.FAILURE_RECIPE


def test_evaluator_altyapi_hatasi_aday_basarisizligi_sayilmaz():
    """Sıfır olmayan sonuç tek başına aday FAIL'i değildir."""
    fake = FakeDocker()
    kanal = backend.DockerChannel(fake)
    orijinal = fake._result
    fake._result = lambda: {**orijinal(), "completed": False,
                            "error_type": "MemoryError", "status": None}
    with pytest.raises(backend.BigCodeBenchBackendError) as exc:
        backend.evaluate_bigcodebench(gorev(), "kod", docker_runner=kanal)
    assert exc.value.failure_code == backend.FAILURE_RECIPE


# --------------------------------------------------------------------------
# Sonuç eşlemesi
# --------------------------------------------------------------------------

def test_pozitif_sonuc_alanlari():
    kayit, _ = _kosur()
    assert kayit["primary_pass"] is True
    assert kayit["primary_status"] == "passed"
    assert kayit["evaluation_metric"] == backend.EVALUATION_METRIC
    assert kayit["base_plus_available"] is False
    assert kayit["plus_pass_is_compatibility_alias"] is True
    assert kayit["plus_pass"] is True and kayit["base_pass"] is True
    assert kayit["traceback"] is None


def test_negatif_sonuc_kontrollu_assertion():
    fake = FakeDocker()
    kanal = backend.DockerChannel(fake)
    orijinal = fake._result
    fake._result = lambda: {**orijinal(), "status": "fail"}
    kayit = backend.evaluate_bigcodebench(gorev(), "kod", docker_runner=kanal)
    assert kayit["primary_pass"] is False
    assert kayit["primary_status"] == "failed"
    assert kayit["evaluator_failure_class"] == backend.CLASS_ASSERTION
    assert kayit["error_class"] == "assertion"


def test_syntax_sinyali_ayri_sinif_uretir():
    fake = FakeDocker()
    kanal = backend.DockerChannel(fake)
    orijinal = fake._result
    fake._result = lambda: {**orijinal(), "status": "fail",
                            "details_signals": {"syntax_error": True, "length": 9}}
    kayit = backend.evaluate_bigcodebench(gorev(), "kod", docker_runner=kanal)
    assert kayit["evaluator_failure_class"] == backend.CLASS_SYNTAX
    assert kayit["error_class"] == "syntax"


def test_timeout_ayri_sinif_uretir():
    fake = FakeDocker()
    kanal = backend.DockerChannel(fake)
    orijinal = fake._result
    fake._result = lambda: {**orijinal(), "status": "timeout"}
    kayit = backend.evaluate_bigcodebench(gorev(), "kod", docker_runner=kanal)
    assert kayit["primary_status"] == "timeout"
    assert kayit["error_class"] == "timeout"


def test_gizli_test_sonuc_kaydina_kopyalanmaz():
    kayit, _ = _kosur(task=gorev(test_code="assert gizli_test_ifadesi()"))
    duz = json.dumps(kayit, ensure_ascii=False)
    assert "gizli_test_ifadesi" not in duz
    assert "details" not in kayit or kayit.get("primary_details_sha256")
    assert kayit["traceback"] is None


# --------------------------------------------------------------------------
# Probe programı
# --------------------------------------------------------------------------

def test_probe_stdout_yerine_dosya_yazar():
    kaynak = backend.probe_source()
    assert "atomic(result_path" in kaynak
    assert "os.replace" in kaynak and "fsync" in kaynak
    # Sonuç stdout'a YAZILMAZ: probe'da hiç `print` DEYİMİ yoktur. Alt dize
    # araması yanıltıcıdır (`fingerprint(` içinde `print(` geçer), bu yüzden
    # satır başına bakılır.
    satirlar = [s.strip() for s in kaynak.splitlines()]
    assert not [s for s in satirlar if s.startswith("print(")]


def test_probe_gercek_adapter_i_cagirir():
    kaynak = backend.probe_source()
    assert "from bigcodebench import eval as bceval" in kaynak
    assert 'getattr(bceval, payload["adapter_callable"])' in kaynak
    assert backend.ADAPTER_CALLABLE == "untrusted_check"


def test_probe_kaynak_hash_i_sabittir():
    assert backend.PROBE_SOURCE_SHA256 == hashlib.sha256(
        backend.probe_source().encode("utf-8")).hexdigest()


def test_limitler_dondurulmus_degerlerle_ayni():
    assert backend.EVAL_LIMITS == {
        "max_as_limit": 30 * 1024, "max_data_limit": 30 * 1024,
        "max_stack_limit": 10, "min_time_limit": 1.0, "gt_time_limit": 5.0}


def test_probe_kaynagi_makaledeki_koşuyla_ayni():
    """Container içinde koşan ölçüm programı, makaledeki koşununkiyle BAYT olarak aynı."""
    assert backend.PROBE_SOURCE_SHA256 == PAPER_PROBE_SOURCE_SHA256


def test_imaj_kimligi_kayda_yazilir_ve_makaledekiyle_karsilastirilir():
    kayit, _ = _kosur()
    prov = kayit["evaluator_provenance"]
    assert prov["image_ref"] == backend.IMAGE_REF
    assert prov["image_id"] == _image_id("A")
    assert prov["matches_paper_image_id"] is False


# --------------------------------------------------------------------------
# Dispatcher regresyonu
# --------------------------------------------------------------------------

def test_etiketsiz_gorev_eski_evalplus_yolunda_kalir(monkeypatch):
    """Etiketsiz görev sandbox yoluna gider; dispatch HİÇ çalışmaz."""
    cagrilar = []
    monkeypatch.setattr(harness, "_dispatch_backend",
                        lambda *a, **k: pytest.fail("dispatch çalışmamalıydı"))
    monkeypatch.setattr(
        harness, "evaluate",
        lambda task, code, *a, **k: cagrilar.append(task) or harness.EvalResult(
            task["task_id"], "passed", None, None, 0.1))
    sonuc = harness.evaluate_base_plus(
        {"task_id": "humanevalplus_002", "entry_point": "f", "test_code": "t"},
        "kod")
    assert sonuc["plus_pass"] is True
    assert sonuc["base_plus_available"] is False
    assert len(cagrilar) == 1


def test_bigcodebench_etiketi_yeni_adaptore_gider():
    kayit, _ = _kosur()
    assert kayit["evaluation_backend"] == backend.BACKEND_ID
    assert harness.backend_of(gorev()) == harness.BIGCODEBENCH_BACKEND


def test_dispatch_evaluate_base_plus_uzerinden_calisir(monkeypatch):
    gorulen = {}
    monkeypatch.setattr(
        harness, "_dispatch_backend",
        lambda task, code, **k: gorulen.update(task_id=task["task_id"],
                                               code=code) or {"ok": True})
    assert harness.evaluate_base_plus(gorev(), "KOD") == {"ok": True}
    assert gorulen == {"task_id": "bigcodebench_0655", "code": "KOD"}


def test_bilinmeyen_backend_reddedilir():
    with pytest.raises(ValueError):
        harness.backend_of({"evaluation_backend": "e2b_remote_v9"})


def test_cagri_yerleri_backend_bilmez():
    """Dispatch harness'ta: tester/baseline BigCodeBench'i doğrudan bilmez."""
    for rel in ("agents/tester.py", "pipeline/baseline.py"):
        metin = (ROOT / rel).read_text(encoding="utf-8")
        assert "evaluate_base_plus" in metin
        assert "bigcodebench" not in metin.lower()


# --------------------------------------------------------------------------
# Kayıt şeması
# --------------------------------------------------------------------------

def _kayit(**overrides):
    kayit = {
        "schema_version": result_schema.RESULT_SCHEMA_VERSION,
        "ts": "2026-08-21T00:00:00+00:00", "experiment": "e", "model": "m",
        "task_set": "followup_dev", "arm": "baseline",
        "task_id": "bigcodebench_0655", "repeat": 0, "run_id": "r",
        "arm_position": 0, "status": "passed",
        "base_status": "passed", "base_pass": True, "base_error_class": None,
        "base_duration_s": 1.0, "plus_status": "passed", "plus_pass": True,
        "plus_error_class": None, "plus_duration_s": 1.0,
        "base_plus_available": False, "error_class": None,
        "primary_status": "passed", "primary_pass": True,
        "primary_error_class": None, "primary_duration_s": 1.0,
        "evaluation_metric": result_schema.BIGCODEBENCH_METRIC,
        "evaluation_backend": result_schema.BIGCODEBENCH_BACKEND,
        "source_task_id": "BigCodeBench/655",
        "plus_pass_is_compatibility_alias": True,
        "traceback": None,
        "evaluator_provenance": {"image_id": _image_id("A")},
    }
    kayit.update(overrides)
    return kayit


def test_gecerli_bigcodebench_kaydi_dogrulanir():
    assert result_schema.validate_record(_kayit()) == []


def test_alias_bayragi_backend_etiketi_olmadan_gelemez():
    kayit = _kayit()
    del kayit["evaluation_backend"]
    assert any("plus_pass_is_compatibility_alias" in p
               for p in result_schema.validate_record(kayit))


def test_ayna_alanlar_birincil_metrikten_ayrisamaz():
    problems = result_schema.validate_record(_kayit(plus_pass=False,
                                                    plus_status="failed"))
    assert any("plus_pass birincil" in p for p in problems)


def test_kimlik_esleşmezse_reddedilir():
    problems = result_schema.validate_record(
        _kayit(source_task_id="BigCodeBench/999"))
    assert any("source_task_id ile task_id" in p for p in problems)


def test_traceback_sizintisi_reddedilir():
    problems = result_schema.validate_record(_kayit(traceback="gizli test satırı"))
    assert any("traceback None olmalı" in p for p in problems)


def test_eski_kayit_dogrulamasi_degismedi():
    """Study 1A/1B biçimindeki kayıt yeni bloktan HİÇ etkilenmez."""
    kayit = _kayit()
    for alan in ("primary_status", "primary_pass", "primary_error_class",
                 "primary_duration_s", "evaluation_metric",
                 "evaluation_backend", "source_task_id",
                 "plus_pass_is_compatibility_alias", "evaluator_provenance"):
        kayit.pop(alan)
    kayit.update({"task_id": "humanevalplus_002", "base_plus_available": True,
                  "plus_skipped_base_failed": False, "traceback": None})
    assert result_schema.validate_record(kayit) == []


# --------------------------------------------------------------------------
# Model/ağ kullanılmadığının kaynak taraması
# --------------------------------------------------------------------------

def test_backend_modulu_model_veya_ag_kullanmaz():
    kaynak = (ROOT / "eval" / "bigcodebench_backend.py").read_text(encoding="utf-8")
    for yasak in ("litellm", "openai", "anthropic", "httpx", "urllib.request",
                  "call_model", "requests."):
        assert yasak not in kaynak, yasak
