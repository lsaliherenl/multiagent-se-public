"""analysis/rq5.py birim testleri — LLM'siz, tamamen sentetik.

Ana koşu fixture'ları `eval/result_schema` fabrikalarından, self-consistency
fixture'ları ise `uncertainty/self_consistency.py`'nin KENDİ üreticilerinden
(`build_result_record`, `candidate_run_id`) kurulur. Elle yazılmış bir "sonuç
özeti", üretim hattının reddedeceği bir turu testte kabul ettirir ve doğrulamayı
kendi kendini onaylayan hale getirirdi.

Amaç: RQ5'in SESSİZCE yanlış bir sayı üretebileceği noktaları kilitlemek —
yanlış modele bağlanan korelasyon, kaynak zinciri kopuk tur (elle düzenlenmiş
özet, adayı olmayan sonuç, çağrısı olmayan aday), eksik/fazla görev, eski şema,
bağların sıraya bağlı sıralanması ve sıfır varyansta "0 korelasyon" raporlanması.
"""

import json

import pytest

from analysis.rq5 import (
    OVERALL_KEY,
    SELFCONS_ORACLE_KEY,
    Rq5Error,
    average_ranks,
    build_rq5,
    check_selfcons_provenance,
    load_selfcons,
    spearman,
    verify_selfcons,
    write_rq5_outputs,
)
from config import (
    ARM_BASELINE,
    ARM_CONTRACT,
    ARM_NAIVE,
    ARM_STRUCTURED,
    LLM_CALL_SCHEMA_VERSION,
    RESULT_SCHEMA_VERSION,
    RQ5_PRIMARY_ARM,
    SELF_CONSISTENCY_SCHEMA_VERSION,
)
from eval.result_schema import make_run_error_record, make_synthetic_record
from uncertainty.self_consistency import (
    SELFCONS_ARM,
    build_result_record,
    candidate_run_id,
)

ARMS = [ARM_BASELINE, ARM_NAIVE, ARM_STRUCTURED, ARM_CONTRACT]
MODEL = "test/model"
TASKS = ["t00", "t01", "t02", "t03", "t04"]
REPEATS = 3
N = 5


def _main_manifest(task_ids=TASKS, model=MODEL, name="ana", arms=ARMS,
                   task_file_hashes=None):
    return {"name": name, "model": model, "task_set": "heldout",
            "task_ids": list(task_ids), "arm_order": list(arms), "repeats": REPEATS,
            "result_schema_version": RESULT_SCHEMA_VERSION,
            "llm_call_schema_version": LLM_CALL_SCHEMA_VERSION,
            "git_commit": "deadbeef",
            "task_file_hashes": task_file_hashes
            or {f"{t}.json": f"hash-{t}" for t in task_ids}}


def _sc_manifest(task_ids=TASKS, model=MODEL, name="sc", n=N, temperature=0.8,
                 schema=SELF_CONSISTENCY_SCHEMA_VERSION, task_file_hashes=None):
    return {"name": name, "model": model, "task_set": "heldout",
            "task_ids": list(task_ids), "n": n, "temperature": temperature,
            "self_consistency_schema_version": schema,
            "llm_call_schema_version": LLM_CALL_SCHEMA_VERSION,
            "git_commit": "deadbeef", "prompt_hash": "p" * 64,
            "algorithm_contract": {"clustering": "exact_signature_largest_cluster_v1"},
            "task_content_hashes": {t: f"content-{t}" for t in task_ids},
            "task_file_hashes": task_file_hashes
            or {f"{t}.json": f"hash-{t}" for t in task_ids}}


def _records(pass_counts, *, model=MODEL, name="ana", task_ids=TASKS, arms=ARMS):
    """pass_counts: (task_id, arm) -> kaç tekrarda plus geçtiği."""
    out = []
    for rep in range(REPEATS):
        for task_id in task_ids:
            for arm in arms:
                gecti = rep < pass_counts.get((task_id, arm), 0)
                out.append(make_synthetic_record(
                    experiment=name, model=model, arm=arm, task_id=task_id,
                    repeat=rep, base_pass=True, plus_pass=gecti))
    return out


def _signatures(agreement, n=N):
    """agreement = en büyük küme / n olacak imza listesi.

    n=5 için 0.2 adımlarla tam kontrol: k tanesi aynı, kalanların hepsi tekil.
    """
    k = round(agreement * n)
    return ["A"] * k + [f"B{i}" for i in range(n - k)]


def _candidates(manifest, agreements, oracle=None):
    oracle = oracle or {}
    rows = []
    for task_id in manifest["task_ids"]:
        sigs = _signatures(agreements[task_id], manifest["n"])
        gecen = round(oracle.get(task_id, 1.0) * manifest["n"])
        for index in range(manifest["n"]):
            rows.append({
                "self_consistency_schema_version":
                    manifest["self_consistency_schema_version"],
                "experiment": manifest["name"], "arm": SELFCONS_ARM,
                "model": manifest["model"], "task_set": manifest["task_set"],
                "task_id": task_id,
                "task_content_sha256": manifest["task_content_hashes"][task_id],
                "candidate_index": index,
                "run_id": candidate_run_id(manifest["model"], manifest["task_set"],
                                           task_id, index),
                "temperature": manifest["temperature"], "n": manifest["n"],
                "prompt_hash": manifest["prompt_hash"], "n_test_inputs": 4,
                "code": f"def f(): return {index}", "signature": sigs[index],
                "oracle_status": "passed" if index < gecen else "failed",
            })
    return rows


def _calls(manifest, candidates):
    return [{
        "schema_version": manifest["llm_call_schema_version"],
        "experiment": manifest["name"], "arm": SELFCONS_ARM,
        "model": c["model"], "task_id": c["task_id"],
        "repeat": c["candidate_index"], "agent_role": "selfcons",
        "run_id": c["run_id"], "status": "ok",
    } for c in candidates]


def _results(manifest, candidates):
    by_task = {}
    for c in candidates:
        by_task.setdefault(c["task_id"], []).append(c)
    return [build_result_record(t, by_task[t], manifest)
            for t in manifest["task_ids"] if len(by_task.get(t, [])) == manifest["n"]]


def _sc_bundle(agreements, oracle=None, manifest=None):
    manifest = manifest or _sc_manifest()
    candidates = _candidates(manifest, agreements, oracle)
    return manifest, candidates, _results(manifest, candidates), _calls(manifest, candidates)


# --- Spearman çekirdeği -------------------------------------------------------

def test_average_ranks_bagları_ortalama_rank_ile_verir():
    assert average_ranks([10, 20, 30]) == [1.0, 2.0, 3.0]
    assert average_ranks([1, 1, 3, 3]) == [1.5, 1.5, 3.5, 3.5]


def test_average_ranks_veri_sirasindan_bagimsiz():
    """Bağların sıraya göre numaralanması rho'yu satır sırasına bağlardı."""
    x = [5, 5, 1, 9]
    assert average_ranks(x) == list(reversed(average_ranks(list(reversed(x)))))


def test_spearman_mukemmel_ters_iliski():
    assert spearman([1, 2, 3], [3, 2, 1])["rho"] == -1.0


def test_spearman_bilinen_bagli_ornek():
    """Elle doğrulanabilir bağlı örnek: rho = sqrt(0.95).

    x=[1..5], y=[2,2,3,4,5] -> rank_y=[1.5,1.5,3,4,5];
    Sxy=9.5, Sxx=10, Syy=9.5 -> rho = 9.5/sqrt(95) = sqrt(0.95).
    """
    sonuc = spearman([1, 2, 3, 4, 5], [2, 2, 3, 4, 5])
    assert sonuc["rho"] == round(0.95 ** 0.5, 6)
    assert sonuc["tied_groups_y"] == 1 and sonuc["tied_groups_x"] == 0


def test_spearman_sifir_varyansta_tanimsiz_doner_sifir_degil():
    sonuc = spearman([1.0, 1.0, 1.0, 1.0], [0.0, 0.3, 0.6, 1.0])
    assert sonuc["rho"] is None
    assert "sıfır varyans" in sonuc["undefined_reason"]
    assert sonuc["p_value_reported"] is False


def test_spearman_p_degeri_uretmez():
    assert "p_value" not in spearman([1, 2, 3], [1, 2, 3])


def test_spearman_uzunluk_uyusmazliginda_durur():
    with pytest.raises(Rq5Error):
        spearman([1, 2, 3], [1, 2])


# --- Girdi kapıları -----------------------------------------------------------

def test_eski_global_selfcons_dosyasi_reddedilir(tmp_path):
    eski = tmp_path / "results_selfcons_20260731T074529Z.jsonl"
    eski.write_text('{"task_id": "t00", "agreement": 1.0}\n', encoding="utf-8")
    with pytest.raises(Rq5Error, match="DİZİN"):
        load_selfcons(eski)


def test_manifestsiz_dizin_reddedilir(tmp_path):
    (tmp_path / "results.jsonl").write_text("{}\n", encoding="utf-8")
    with pytest.raises(Rq5Error, match="manifesti yok"):
        load_selfcons(tmp_path)


def test_aday_dosyasi_olmayan_tur_reddedilir(tmp_path):
    """Yalnız results.jsonl okumak kaynak zincirini doğrulamaz."""
    manifest, candidates, results, calls = _sc_bundle({t: 1.0 for t in TASKS})
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (tmp_path / "results.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in results), encoding="utf-8")
    (tmp_path / "llm_calls.jsonl").write_text(
        "".join(json.dumps(c) + "\n" for c in calls), encoding="utf-8")
    with pytest.raises(Rq5Error, match="candidate_records.jsonl"):
        load_selfcons(tmp_path)


def test_dort_dosya_da_yuklenir(tmp_path):
    manifest, candidates, results, calls = _sc_bundle({t: 1.0 for t in TASKS})
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    for ad, rows in (("candidate_records.jsonl", candidates),
                     ("results.jsonl", results), ("llm_calls.jsonl", calls)):
        (tmp_path / ad).write_text("".join(json.dumps(r) + "\n" for r in rows),
                                   encoding="utf-8")
    m, c, r, k = load_selfcons(tmp_path)
    assert (len(c), len(r), len(k)) == (len(TASKS) * N, len(TASKS), len(TASKS) * N)
    assert m["name"] == manifest["name"]


def test_eski_sema_surumu_reddedilir():
    sorunlar = check_selfcons_provenance(_sc_manifest(schema="0.9"), _main_manifest())
    assert any("şema sürümü" in s for s in sorunlar)


def test_farkli_model_havuzlanmayi_reddeder():
    sorunlar = check_selfcons_provenance(_sc_manifest(model="baska/model"),
                                         _main_manifest())
    assert any("HAVUZLANMAZ" in s for s in sorunlar)


def test_farkli_gorev_kumesi_reddedilir():
    sorunlar = check_selfcons_provenance(_sc_manifest(task_ids=TASKS[:4]),
                                         _main_manifest())
    assert any("görev kümesi uyuşmuyor" in s for s in sorunlar)


def test_gorev_hash_uyusmazligi_reddedilir():
    bozuk = {f"{t}.json": "farkli" for t in TASKS}
    sorunlar = check_selfcons_provenance(_sc_manifest(task_file_hashes=bozuk),
                                         _main_manifest())
    assert any("hash'leri uyuşmuyor" in s for s in sorunlar)


def test_ayni_manifestler_temiz_gecer():
    assert check_selfcons_provenance(_sc_manifest(), _main_manifest()) == []


def test_selection_fingerprint_karsilastirilmaz():
    """§32 borcu: iki hat aynı içeriği farklı serileştirmeyle hash'liyor."""
    sc = _sc_manifest() | {"heldout_selection_fingerprint": "aaa"}
    ana = _main_manifest() | {"heldout_selection_fingerprint": "bbb"}
    assert check_selfcons_provenance(sc, ana) == []


# --- Kaynak zinciri (aday -> çağrı -> sonuç) ----------------------------------

def test_saglam_zincir_dogrulanir():
    manifest, cand, res, calls = _sc_bundle({t: 1.0 for t in TASKS})
    by_task = verify_selfcons(manifest, cand, res, calls)
    assert set(by_task) == set(TASKS)


def test_elle_degistirilmis_sonuc_ozeti_fail_fast():
    """agreement adaylardan DETERMİNİSTİK türetilir; elle yazılan değer düşer."""
    manifest, cand, res, calls = _sc_bundle({t: 1.0 for t in TASKS})
    res[0] = {**res[0], "agreement": 0.2}
    with pytest.raises(Rq5Error, match="sonuç kaydı"):
        verify_selfcons(manifest, cand, res, calls)


def test_elle_degistirilmis_oracle_pass_rate_fail_fast():
    manifest, cand, res, calls = _sc_bundle({t: 1.0 for t in TASKS})
    res[2] = {**res[2], "oracle_pass_rate": 0.0}
    with pytest.raises(Rq5Error, match="sonuç kaydı"):
        verify_selfcons(manifest, cand, res, calls)


def test_eksik_aday_fail_fast():
    manifest, cand, res, calls = _sc_bundle({t: 1.0 for t in TASKS})
    with pytest.raises(Rq5Error, match="eksik aday|aday"):
        verify_selfcons(manifest, cand[:-1], res, calls)


def test_yinelenen_aday_fail_fast():
    manifest, cand, res, calls = _sc_bundle({t: 1.0 for t in TASKS})
    with pytest.raises(Rq5Error, match="yinelenen aday"):
        verify_selfcons(manifest, [*cand, cand[0]], res, calls)


def test_stale_aday_fail_fast():
    """Görev metni değişmişse aday kaydı o metnin ürünü değildir."""
    manifest, cand, res, calls = _sc_bundle({t: 1.0 for t in TASKS})
    cand[0] = {**cand[0], "task_content_sha256": "bayat"}
    with pytest.raises(Rq5Error, match="stale/geçersiz aday"):
        verify_selfcons(manifest, cand, res, calls)


def test_adayi_olmayan_sonuc_fail_fast():
    manifest, cand, res, calls = _sc_bundle({t: 1.0 for t in TASKS})
    kirpik = [c for c in cand if c["task_id"] != "t04"]
    kirpik_calls = [c for c in calls if c["task_id"] != "t04"]
    with pytest.raises(Rq5Error, match="eksik aday varken sonuç"):
        verify_selfcons(manifest, kirpik, res, kirpik_calls)


def test_basarili_cagrisi_olmayan_aday_fail_fast():
    manifest, cand, res, calls = _sc_bundle({t: 1.0 for t in TASKS})
    with pytest.raises(Rq5Error, match="başarılı çağrı provenance"):
        verify_selfcons(manifest, cand, res, calls[:-1])


def test_yabanci_cagri_fail_fast():
    manifest, cand, res, calls = _sc_bundle({t: 1.0 for t in TASKS})
    yabanci = {**calls[0], "run_id": "selfcons-baska-tur"}
    with pytest.raises(Rq5Error, match="yabancı çağrı"):
        verify_selfcons(manifest, cand, res, [*calls, yabanci])


@pytest.mark.parametrize("alan,deger", [
    ("experiment", "BASKA-DENEY"),
    ("model", "baska/model"),
    ("task_id", "baska-gorev"),
    ("repeat", 99),
    ("agent_role", "coder"),
    ("arm", ARM_BASELINE),
])
def test_yanlis_baglamli_cagri_fail_fast(alan, deger):
    manifest, cand, res, calls = _sc_bundle({t: 1.0 for t in TASKS})
    calls[0] = {**calls[0], alan: deger}
    with pytest.raises(Rq5Error, match="stale/yanlış bağlamlı çağrı"):
        verify_selfcons(manifest, cand, res, calls)


def test_eksik_gorev_sonucu_fail_fast():
    manifest, cand, res, calls = _sc_bundle({t: 1.0 for t in TASKS})
    with pytest.raises(Rq5Error, match="self-consistency sonucu yok"):
        verify_selfcons(manifest, cand, res[:-1], calls)


# --- Uçtan uca ----------------------------------------------------------------

def _kur(agreements, pass_counts, oracle=None):
    ana = _main_manifest()
    sc, cand, res, calls = _sc_bundle(agreements, oracle)
    return ana, _records(pass_counts), sc, cand, res, calls


def test_birincil_eslestirme_baseline_kolu():
    agreements = {"t00": 1.0, "t01": 0.8, "t02": 0.6, "t03": 0.4, "t04": 0.2}
    pass_counts = {(t, a): {"t00": 3, "t01": 3, "t02": 2, "t03": 1, "t04": 0}[t]
                   for t in TASKS for a in ARMS}
    sonuc = build_rq5(*_kur(agreements, pass_counts))

    assert sonuc["primary_pairing"] == RQ5_PRIMARY_ARM == ARM_BASELINE
    assert sonuc["pairings"][ARM_BASELINE]["role"] == "primary"
    assert all(sonuc["pairings"][a]["role"] == "secondary"
               for a in (ARM_NAIVE, ARM_STRUCTURED, ARM_CONTRACT,
                         OVERALL_KEY, SELFCONS_ORACLE_KEY))
    assert sonuc["pairings"][ARM_BASELINE]["spearman"]["rho"] < 0
    assert sonuc["exploratory"] is True and sonuc["p_values_reported"] is False


def test_ozet_kaynak_provenansini_tasir():
    """Sayı artefakt paketine kopyalandığında manifestler yanında olmayabilir."""
    agreements = {t: 0.2 * (i + 1) for i, t in enumerate(TASKS)}
    pass_counts = {(t, a): i % 4 for i, t in enumerate(TASKS) for a in ARMS}
    sonuc = build_rq5(*_kur(agreements, pass_counts))
    p = sonuc["source_provenance"]
    assert p["main_git_commit"] == "deadbeef"
    assert p["selfcons_git_commit"] == "deadbeef"
    assert p["selfcons_schema_version"] == SELF_CONSISTENCY_SCHEMA_VERSION
    assert p["selfcons_prompt_hash"] and p["rq5_schema_version"]
    assert p["main_task_file_hashes_sha256"] == p["selfcons_task_file_hashes_sha256"]
    assert sonuc["self_consistency"]["verified_candidates"] == len(TASKS) * N
    assert sonuc["self_consistency"]["verified_tasks"] == len(TASKS)


def test_cikti_hicbir_yerde_p_degeri_tasimaz(tmp_path):
    agreements = {t: 0.2 * (i + 1) for i, t in enumerate(TASKS)}
    pass_counts = {(t, a): i % 4 for i, t in enumerate(TASKS) for a in ARMS}
    sonuc = build_rq5(*_kur(agreements, pass_counts))
    paths = write_rq5_outputs(sonuc, tmp_path)
    payload = json.loads(paths["rq5_summary"].read_text(encoding="utf-8"))

    def _anahtarlar(node):
        if isinstance(node, dict):
            for k, v in node.items():
                yield k
                yield from _anahtarlar(v)
        elif isinstance(node, list):
            for v in node:
                yield from _anahtarlar(v)

    assert "p_value" not in set(_anahtarlar(payload))
    assert payload["p_values_reported"] is False
    assert all(p["spearman"]["p_value_reported"] is False
               for p in payload["pairings"].values())


def test_ayni_veri_bayt_bayt_ayni_ciktiyi_uretir(tmp_path):
    agreements = {t: 0.2 * (i + 1) for i, t in enumerate(TASKS)}
    pass_counts = {(t, a): i % 4 for i, t in enumerate(TASKS) for a in ARMS}
    girdi = _kur(agreements, pass_counts)
    ilk = write_rq5_outputs(build_rq5(*girdi), tmp_path / "a")
    ikinci = write_rq5_outputs(build_rq5(*girdi), tmp_path / "b")
    for ad in ilk:
        assert ilk[ad].read_bytes() == ikinci[ad].read_bytes()


def test_modeller_havuzlanamaz():
    agreements = {t: 1.0 for t in TASKS}
    pass_counts = {(t, a): 3 for t in TASKS for a in ARMS}
    ana, records, sc, cand, res, calls = _kur(agreements, pass_counts)
    records[0] = {**records[0], "model": "baska/model"}
    from analysis.analyze import AnalysisError
    with pytest.raises(AnalysisError, match="HAVUZLANMAZ"):
        build_rq5(ana, records, sc, cand, res, calls)


def test_eksik_ana_kosu_kaydinda_durur():
    agreements = {t: 1.0 for t in TASKS}
    pass_counts = {(t, a): 3 for t in TASKS for a in ARMS}
    ana, records, sc, cand, res, calls = _kur(agreements, pass_counts)
    kirpik = [r for r in records
              if not (r["task_id"] == "t02" and r["arm"] == ARM_BASELINE)]
    with pytest.raises(Rq5Error, match="eksik koşu"):
        build_rq5(ana, kirpik, sc, cand, res, calls)


def test_yinelenen_ana_kosu_kaydinda_durur():
    agreements = {t: 1.0 for t in TASKS}
    pass_counts = {(t, a): 3 for t in TASKS for a in ARMS}
    ana, records, sc, cand, res, calls = _kur(agreements, pass_counts)
    with pytest.raises(Rq5Error, match="yinelenen kayıt"):
        build_rq5(ana, [*records, records[0]], sc, cand, res, calls)


def test_tavan_etkisinde_rho_tanimsiz_kalir():
    agreements = {t: 1.0 for t in TASKS}
    pass_counts = {(t, a): i % 4 for i, t in enumerate(TASKS) for a in ARMS}
    sonuc = build_rq5(*_kur(agreements, pass_counts))
    s = sonuc["pairings"][ARM_BASELINE]["spearman"]
    assert s["rho"] is None and "sıfır varyans" in s["undefined_reason"]


def test_run_error_paydaya_girmez():
    agreements = {t: 0.2 * (i + 1) for i, t in enumerate(TASKS)}
    pass_counts = {(t, a): 2 for t in TASKS for a in ARMS}
    ana, records, sc, cand, res, calls = _kur(agreements, pass_counts)
    temiz = build_rq5(ana, records, sc, cand, res, calls)
    hatali = make_run_error_record(
        experiment="ana", model=MODEL, task_set="heldout", arm=ARM_BASELINE,
        task_id="t00", repeat=0, run_id="err-1", arm_position=0, error="boom")
    kirli = build_rq5(ana, [*records, hatali], sc, cand, res, calls)
    assert temiz["pairings"][ARM_BASELINE] == kirli["pairings"][ARM_BASELINE]


def test_selfcons_oracle_serisi_ikincil_olarak_raporlanir():
    agreements = {t: 0.2 * (i + 1) for i, t in enumerate(TASKS)}
    oracle = {t: 1.0 - 0.2 * i for i, t in enumerate(TASKS)}
    pass_counts = {(t, a): 3 for t in TASKS for a in ARMS}
    sonuc = build_rq5(*_kur(agreements, pass_counts, oracle))
    p = sonuc["pairings"][SELFCONS_ORACLE_KEY]
    assert p["role"] == "secondary"
    assert p["y_descriptives"]["n"] == len(TASKS)
