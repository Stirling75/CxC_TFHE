"""Validate complete raw trials before exporting arithmetic means."""
import csv
import json
import math
import statistics

from cases import CATALOG, RING, SC, bfv_module, module
from parameter_checks import equal, hybrid_parameters, library_parameters


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_rows(path, warmup, repetitions):
    with path.open() as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames is not None and len(set(reader.fieldnames)) == len(reader.fieldnames),
                "missing or duplicate CSV header")
        rows = list(reader)
    require(len(rows) == warmup + repetitions, "incomplete trial count")
    for index, row in enumerate(rows):
        require(None not in row and all(v is not None for v in row.values()), "malformed CSV row")
        require(row["ok"] == "true", f"failed trial {index}")
        require(row["warmup"] in ("0", "1", "true", "false"), "invalid warmup value")
        require((row["warmup"] in ("1", "true")) == (index < warmup), "warmup mismatch")
        trial_key = "trial" if "trial" in row else "trial_index"
        require(int(row[trial_key]) == index, "duplicate or reordered trial")
    return rows


def check(method, width, threads, args, case, resolved):
    config = CATALOG[method]
    family = config["family"]
    raw = case / "raw"
    if family == "bfv":
        report = bfv_module("audit_run").audit(raw, require_benchmark=True)
        meta = json.loads((raw / "manifest.json").read_text())
        for field, value in (("plaintext_seed", args.seed), ("repetitions", args.repetitions),
                             ("warmup", args.warmup), ("patterns", ["random"]), ("verify_serial", False)):
            equal(meta[field], value, field)
        equal(meta["host"]["evaluation_threads"], threads, "BFV threads")
        require(len(meta["plans"]) == 1 and meta["plans"][0]["width"] == width, "BFV width mismatch")
        require(len(report["summaries"]) == 1, "unexpected BFV result groups")
        summary = report["summaries"][0]
        require(abs(summary["conditional_union_log2"] - resolved["analysis"]["raw_log2_union"]) < 1e-6,
                "BFV runtime noise estimate differs from preflight")
        (case / "audit.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        return summary_row(method, width, threads, summary["seconds"], config)
    rows = read_rows(raw / "timings.csv", args.warmup, args.repetitions)
    key = {"hybrid": "total_ms", "bitwise": "total_s", "st": "total_s",
           "radix": "total_seconds", "parmesan": "total_seconds", "bernard": "total_seconds"}[family]
    if family == "hybrid":
        meta = json.loads((raw / "parameters.json").read_text())
        equal(meta, hybrid_parameters(resolved["plan"], seed=args.seed))
        checker = module("hybrid_runner_checks", RING / "run.py")
        checker.validate_rows(raw / "timings.csv", resolved["plan"], len(rows), threads,
                              analysis=resolved["analysis"], seed=args.seed)
    elif family == "bitwise":
        import bitwise_check
        bitwise_check.validate_run(raw)
        meta = json.loads((raw / "parameters.json").read_text())
        arguments = meta["arguments"]
        equal(meta["parameters"], library_parameters(config["preset"]))
        require(meta["tfhe_rs"] == "1.7.0" and not arguments["fresh_inputs"], "bitwise version/input mismatch")
        require(arguments["repetitions"] == args.repetitions and arguments["warmup"] == args.warmup and
                arguments["patterns"] == ["random"], "bitwise protocol mismatch")
        require(arguments["methods"] == [config["circuit"]] and
                arguments["widths"] == [width] and arguments["threads"] == [threads] and
                arguments["preset"] == config["preset"] and arguments["seed"] == args.seed,
                "bitwise requested/runtime parameters differ")
        require(meta["primitive_log2_p_fail"] == config["primitive_log2_p_fail"], "primitive label mismatch")
        for row in rows:
            require(int(row["seed"]) == args.seed, "bitwise row seed mismatch")
    elif family == "st":
        import st_validate
        meta = json.loads((raw / "parameters.json").read_text())
        st_validate.parameter_roundoff(meta["parameter_set"], resolved["parameters"])
        reference = st_validate.check_plan(raw, width)
        if config.get("reported"):
            require(meta["boundary_delta_log2"] == config["boundary_delta_log2"] and
                    meta["paper_CC2_boundary"] == (config["boundary_delta_log2"] == 52),
                    "reported-parameter CC2 boundary mismatch")
        else:
            require(meta["boundary_delta_log2"] == 52 and meta["paper_CC2_boundary"], "CC2 contract mismatch")
        require(meta["width"] == width and meta["threads"] == threads and meta["seed"] == args.seed,
                "ST requested/runtime dimensions differ")
        require(not meta["inspect"] and not meta["publication_ready"], "unexpected diagnostic state")
        require(meta["repetitions"] == args.repetitions and meta["warmup"] == args.warmup and
                meta["tfhe_rs_version"] == "1.6.1" and meta["pattern"] == "Random", "ST protocol mismatch")
        # Independent check: Python-derived counts from the plan (st_retune.screen).
        python_counts = resolved["analysis"]["expected_runtime_counts"]
        for field, python_field in (("total_blind_rotations", "blind_rotations"), ("key_switches", "key_switches"),
                                    ("ggsw_conversions", "ggsw_conversions"), ("product_cmux", "product_cmux"),
                                    ("terminal_cmux", "terminal_cmux")):
            require(meta["counts"].get(field) == python_counts[python_field],
                    f"ST Rust plan/Python {field} counts differ")
        for row in rows:
            counts = json.loads(row["counts"])
            expected = meta["counts"]
            require(all(counts[k] == python_counts[e] for k, e in (
                        ("blind_rotations", "blind_rotations"), ("key_switches", "key_switches"),
                        ("conversions", "ggsw_conversions"), ("cmux", "cmux"))),
                    "ST runtime counters differ from Python-derived counts")
            require(counts["blind_rotations"] == expected["total_blind_rotations"] and
                    counts["conversions"] == expected["ggsw_conversions"] and
                    counts["cmux"] == expected["product_cmux"] + expected["terminal_cmux"], "ST count mismatch")
            require(row["output_errors"] == "0" and row["post_pbs_errors"] == "0", "ST output errors")
            stages = resolved["analysis"]["stage_counts"]
            require(counts["blind_rotations"] == sum(stages.get(k, 0) for k in
                    ("input_grouped_lift", "compressor", "terminal_binary_lift", "emission_pbs")),
                    "ST analytical/runtime event counts differ")
            # The equality is CSV integrity only (the evaluator writes phase_sum_s
            # as this same sum); phase_sum <= total_s checks timer nesting.
            phase_sum = sum(float(row[k]) for k in
                            ("lift_s", "product_s", "reduction_s", "terminal_s", "emission_s"))
            require(math.isclose(phase_sum, float(row["phase_sum_s"]), rel_tol=1e-12) and
                    phase_sum <= float(row["total_s"]) + 1e-9, "ST phase timers inconsistent")
        n = resolved["parameters"]["polynomial_size"]
        require(meta["counts"]["product_cmux"] == len(reference.products) *
                (65536 // n - 1 + n.bit_length() - 1), "ST analytical/runtime CMux counts differ")
    elif family == "radix":
        meta = json.loads((raw / "parameters.json").read_text())
        expected = library_parameters("m2c2-gaussian")
        preset = "V1_6_PARAM_MESSAGE_2_CARRY_2_KS_PBS_GAUSSIAN_2M128"
        if "ks" in CATALOG[method]:
            expected = dict(expected, ks_base_log=CATALOG[method]["ks"][0], ks_level=CATALOG[method]["ks"][1])
            preset += " with replaced KS decomposition"
        equal(meta["parameters"], expected)
        require(meta["tfhe_rs"] == "1.6.1" and meta["repetitions"] == args.repetitions and
                meta["warmup"] == args.warmup, "radix protocol mismatch")
        require(meta["parameter"] == preset, "radix preset mismatch")
        require(meta["width"] == width and meta["threads"] == threads and meta["seed"] == args.seed,
                "radix requested/runtime parameters differ")
        require(meta["parameters"]["message_modulus"] == 4 and
                meta["parameters"]["carry_modulus"] == 4, "radix moduli mismatch")
        # The binary records the library label of the default set; a replaced KS
        # decomposition carries its own model estimate in the catalogue instead.
        require(meta["primitive_log2_p_fail"] == (-128.597 if "ks" in config else config["primitive_log2_p_fail"]),
                "radix failure label mismatch")
        for row in rows:
            require(int(row["width"]) == width and int(row["threads"]) == threads and
                    row["method"] == "tfhe-rs-1.6.1-gaussian", "radix row configuration mismatch")
    elif family == "bernard":
        meta = json.loads((raw / "parameters.json").read_text())
        schedule = json.loads((case / "schedule.json").read_text())
        for field, expected in (("width", width), ("threads", threads), ("seed", args.seed),
                                ("repetitions", args.repetitions), ("warmup", args.warmup),
                                ("lwe_dimension", 930), ("ks_base_log", config["ks"][0]),
                                ("ks_level", config["ks"][1]), ("phi_max", config["phi"]),
                                ("ticks", len(schedule["ticks"])), ("identical", False)):
            equal(meta[field], expected, field)
        brs = sum(map(len, schedule["ticks"]))
        for row in rows:
            require(int(row["width"]) == width and int(row["threads"]) == threads and
                    row["method"] == "bernard-mvb" and int(row["blind_rotations"]) == brs and
                    row["ok"] == "true", "Bernard et al. row mismatch")
    else:
        parameter_record = json.loads((raw / "parameters.json").read_text())
        equal(parameter_record["parameters"], library_parameters("parmesan"))
        for field, expected in (("tfhe_rs", "0.5.4"), ("width", width), ("threads", threads),
                                ("seed", args.seed), ("repetitions", args.repetitions), ("warmup", args.warmup)):
            equal(parameter_record[field], expected, field)
        meta = (raw / "parameters.txt").read_text()
        require(f"seed={args.seed}\n" in meta and f"rayon_threads={threads}\n" in meta,
                "PARMESAN requested/runtime parameters differ")
        import parmesan_check
        for row in rows:
            require(int(row["width"]) == width and int(row["rayon_threads"]) == threads, "PARMESAN dimension mismatch")
            try:
                parmesan_check.check_row(row, width, args.seed, threads)
            except AssertionError as error:
                raise ValueError(f"PARMESAN output check failed: {error}") from error
    samples = [float(r[key]) / (1000 if family == "hybrid" else 1) for r in rows]
    require(all(math.isfinite(x) and x > 0 for x in samples), "non-finite or nonpositive latency")
    samples = samples[args.warmup:]
    return summary_row(method, width, threads, samples, config)


def summary_row(method, width, threads, samples, config):
    return {"method": method, "width": width, "threads": threads, "repetitions": len(samples),
            "mean_seconds": statistics.mean(samples),
            "sample_sd_seconds": statistics.stdev(samples) if len(samples) > 1 else None,
            "output_contract": config["output"], "failure_status": config["failure_status"],
            "whole_multiplier_approved": False, "correct": True}
