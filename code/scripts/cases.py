"""Resolve one benchmark case, including its analysis and input/output contract."""
import copy
import importlib.util
import json
import math
from pathlib import Path
import sys
from dataclasses import asdict, replace
from functools import lru_cache
from parameter_checks import check_hybrid_model

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
RING = CODE / "ring-variant-multiplier"
SC = CODE / "server-campaign"
BFV = SC / "adapters/bfv-style"
sys.path[:0] = [str(RING), str(RING / "model"), str(SC)]
CATALOG = json.loads((ROOT / "config/methods.json").read_text())
CRATES = {
    "ring": RING / "Cargo.toml",
    "bitwise": SC / "adapters/bitwise/Cargo.toml",
    "parmesan": SC / "adapters/parmesan/Cargo.toml",
    "bfv": BFV / "Cargo.toml",
}
DEFAULT = ["hybrid-grouped-rev-ld", "hybrid-cached-rev-ld", "tfhe-rs-ks28", "st-r2-n688",
           "bernard-mvb-ks36", "trifan", "parmesan", "clot-bfv"]
TARGET_LOG2 = -128
# Fixed estimates; these are not recomputed by packaged screens.
RADIX_PBS_COUNTS = {16: 116, 32: 455, 64: 1772, 128: 6967, 256: 27702}
PARMESAN_LOG2 = {16: -30.72, 32: -28.86}
# clot-bfv W=256: measurements refute the Gaussian model of the dominant quotient
# error term. Calibrated estimate range (low, high), not a certified bound.
CLOT_REFUTED = {256: (-52.0, -41.6)}


def crate_for(method):
    family = CATALOG[method]["family"]
    return family if family in ("bitwise", "parmesan", "bfv") else "ring"


def binary_for(method, build):
    family = CATALOG[method]["family"]
    name = {"hybrid": "ring_variant_hybrid", "radix": "tfhe_radix_baseline",
            "st": "st_revised_reconstruction", "bitwise": "bitwise-campaign",
            "parmesan": "parmesan-benchmark", "bfv": "bfv-style-probe",
            "bernard": "mvb_radix_mul"}[family]
    return build / crate_for(method) / "release" / name


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


@lru_cache(maxsize=None)
def bfv_module(name):
    sys.path.insert(0, str(BFV))
    return module(name, BFV / (name + ".py"))


@lru_cache(maxsize=None)
def resolve(method, width, public_scalar=None, public_cap=64):
    """`public_scalar` (tuple of radix-4 digits) derives the scalar-aware plan of a
    ciphertext-plaintext product-sum lookup."""
    config = CATALOG[method]
    if width not in config["widths"]:
        raise ValueError(f"{method} does not support W={width}")
    result = {"method": method, "width": width, "contract": config,
              "whole_multiplier_approved": False}
    if config["family"] == "hybrid":
        import make_plan
        plan = make_plan.derive(width, config["mode"], chunk_bits=config["chunk_bits"],
            capacity=config.get("group_size", 32), kernel=config.get("kernel", "tree"),
            cbs_level=7, auto_fft="split40", public_scalar=public_scalar, public_cap=public_cap)
        plan["env"].update(CBS_RING_SHAPE=config.get("ring", "2048x1"),
                           CBS_SECRET_LAYOUT="contiguous")
        # A catalogue "cbs" entry overrides the mode's default CBS decomposition;
        # the public partition from make_plan is kept and screened with it.
        for key, prefix in (("auto", "DIRECT_AUTO"), ("ss", "DIRECT_SS"), ("lift_ks", "CBS_LIFT_KS"),
                            ("cbs", "DIRECT_CBS"), ("lift_pbs", "CBS_LIFT_PBS"), ("norm_ks", "KS")):
            if key in config:
                plan["env"][prefix + "_BASE_LOG"], plan["env"][prefix + "_LEVEL"] = map(str, config[key])
        if "shared_ks" in config:
            # Control: share the key switch of a reduction group's digit and carry
            # PBS independently of the product mode.
            plan["env"]["CBS_SHARED_PREPROCESSING"] = str(int(config["shared_ks"]))
        if config.get("phase_threads"):
            plan["env"]["CBS_PHASE_THREADS"] = "1"
        # Old planning variances choose a fixed public schedule, not parameters.
        plan["reference_screen_applies_to_ring"] = False
        plan["reference_screen_log2"] = None
        result["plan"] = plan
        import heterogeneous_screen as screen
        import baseline_study
        n = int(plan["env"]["CBS_RING_SHAPE"].split("x")[0])
        cbs = tuple(int(plan["env"]["DIRECT_CBS" + suffix]) for suffix in ("_BASE_LOG", "_LEVEL"))
        # Catalogue lift-PBS and normalizer-KS overrides enter the model profile.
        original_profile = baseline_study.profile_for
        def profile_for(bits):
            profile = original_profile(bits)
            if "lift_pbs" in config:
                profile = replace(profile, cbs_lift_pbs_base_log=config["lift_pbs"][0],
                                  cbs_lift_pbs_level=config["lift_pbs"][1])
            if "norm_ks" in config:
                profile = replace(profile, ks_base_log=config["norm_ks"][0], ks_level=config["norm_ks"][1])
            return profile
        baseline_study.profile_for = profile_for
        try:
            p, v = screen.primitive(n, config["auto"], config["ss"], config["lift_ks"],
                                    chunk_bits=config["chunk_bits"], cbs=cbs)
            check_hybrid_model(plan, p)
            if "linear_digits" in config:
                # Carry-only reduction: chosen digits stay linear (sum - 4*carry), unrefreshed,
                # and the final addition reads them directly. Chosen for AB and A=B together.
                # The per-digit cap does not bound how many linear digits meet in a later
                # group, so the mid-wave cap is lowered until both screens meet the target.
                cap_mid, cap_last = config["linear_digits"]
                for cap in [c for c in (cap_mid, 30, 20, 10, 0) if c <= cap_mid]:
                    trial = screen.assign_linear(copy.deepcopy(plan), v, cap, cap_last)
                    ab = screen.analyze(trial, p, v)["conditional_union_log2"]
                    aa = screen.analyze(trial, p, v, identical=True)["conditional_union_log2"]
                    if max(ab, aa) < TARGET_LOG2:
                        break
                plan.clear(); plan.update(trial)
            if "mvb" in config:
                # Every group with a retained carry: one multi-value bootstrapping.
                screen.assign_mvb(plan, v, config["mvb"])
            result["analysis_parameters"] = asdict(p)
            result["analysis"] = screen.analyze(plan, p, v)
            result["analysis_identical"] = screen.analyze(plan, p, v, identical=True)
            result["model_variances"] = screen.model_variances(plan, v)
        finally:
            baseline_study.profile_for = original_profile
        estimate = result["analysis"]["conditional_union_log2"]
        if "linear_digits" in config:
            estimate = max(estimate, result["analysis_identical"]["conditional_union_log2"])
        label(result, estimate, "conditional Gaussian screen", True)
    elif config["family"] == "st":
        import st_retune
        params = json.loads((SC / "parameters" / config["parameters"]).read_text())
        if config.get("cc2_big_key"):
            params = dict(params, cc2_big_key=True)
        result["parameters"] = params
        result["analysis"] = st_retune.screen(params, width)
        if config.get("reported"):
            # Paper parameters with unpublished noise: record, never certify or block.
            label(result, float("nan"), "unverifiable: paper Table 2 parameters with assumed noise "
                  f"and CC2 delta 2^{config['boundary_delta_log2']}; screen not applicable", False)
        else:
            label(result, result["analysis"]["conditional_log2_union"], "conditional Gaussian screen", True)
    elif config["family"] == "bfv":
        params = json.loads((BFV / "parameters" / config["parameters"]).read_text())
        verifier = bfv_module("verify_candidate")
        result["parameters"] = params
        result["preflight"] = verifier.verify(params)
        result["analysis"] = next(row for row in result["preflight"]["rows"] if row["width"] == width)
        result["analysis"]["conditional_union_log2"] = result["analysis"]["raw_log2_union"]
        if width in CLOT_REFUTED:
            low, high = CLOT_REFUTED[width]
            label(result, high, "Gaussian screen refuted by encrypted measurement; "
                  f"calibrated correlated-quotient estimate {low}..{high}", True)
        else:
            label(result, result["analysis"]["raw_log2_union"], "conditional Gaussian screen", True)
    elif config["family"] == "bitwise":
        if config["circuit"] == "morshed":
            result["logical_lut_calls"] = "W^2 + 10W(W + min(P,W) - 1)"
            calls = width**2 + 10 * width * (2 * width - 1)  # worst thread budget P >= W
        else:
            result["logical_lut_calls"] = (3 * width**2 if config["circuit"] == "trifan"
                                           else (3 * width**2 + width) // 2)
            calls = result["logical_lut_calls"]
        result["primitive_log2_p_fail"] = config["primitive_log2_p_fail"]
        label(result, config["primitive_log2_p_fail"] + math.log2(calls),
              "per-PBS library label + log2(logical LUT calls), union bound", False)
    elif config["family"] == "radix":
        label(result, config["primitive_log2_p_fail"] + math.log2(RADIX_PBS_COUNTS[width]),
              "per-PBS library label + log2(measured PBS count), union bound", False)
    elif config["family"] == "bernard":
        # Reimplementation of ePrint 2026/2310: the public schedule depends on the
        # thread count, so the label is the worst estimate over the measured budgets.
        sys.path.insert(0, str(RING / "model" / "bernard"))
        import failure as bernard_failure
        rows = {t: bernard_failure.estimate(width, t, tuple(config["ks"]), config["phi"])
                for t in (1, 2, 4, 8, 16, 32, 64)}
        result["analysis"] = {str(t): r for t, r in rows.items()}
        label(result, max(max(r["ab"], r["identical"]) for r in rows.values()),
              "conditional Gaussian screen over the public folding schedule", False)
    elif config["family"] == "parmesan":
        label(result, PARMESAN_LOG2[width], "Gaussian model of TFHE-rs 0.5.4 M4_C0 "
              "(about 2^-40 per PBS) over 725 / 2617 PBS, union bound", False)
    return result


def label(result, estimate, basis, screened):
    """Record the whole-product estimate. Only packaged screens may block a run."""
    result["whole_product_log2_estimate"] = estimate
    result["failure_basis"] = basis
    result["failure_target_met"] = estimate < TARGET_LOG2
    result["screen_failed"] = screened and not result["failure_target_met"]


def prepare(method, width, threads, args, case):
    resolved = copy.deepcopy(resolve(method, width))
    family, config = CATALOG[method]["family"], CATALOG[method]
    output = case / "raw"
    env = {"RAYON_NUM_THREADS": str(threads)}
    command = [str(binary_for(method, args.build_root))]
    common = ["--width", str(width), "--threads", str(threads),
              "--repetitions", str(args.repetitions), "--warmup", str(args.warmup),
              "--seed", str(args.seed), "--output", str(output)]
    if family == "hybrid":
        output.mkdir()
        plan = resolved["plan"]
        plan_path = case / "plan.json"
        plan_path.write_text(json.dumps(plan, indent=2) + "\n")
        env.update(plan["env"])
        env.update(CACHED_MULT_PLAN=str(plan_path), CACHED_MULT_WARMUP=str(args.warmup),
            CACHED_MULT_PATTERN="random", CACHED_MULT_VERIFY_PBS="0",
            CBS_RANDOM_OPERANDS_SEED=str(args.seed), CBS_TIMING_CSV=str(output / "timings.csv"),
            CBS_FAILURE_TARGET_BITS="128", CBS_FAILURE_EXPECTED_WORST_LOG2="")
        command += [str(width), "1", str(args.repetitions + args.warmup)]
    elif family == "bitwise":
        command += ["--preset", config["preset"], "--methods", config["circuit"],
            "--widths", str(width), "--threads", str(threads), "--patterns", "random",
            "--repetitions", str(args.repetitions), "--warmup", str(args.warmup),
            "--seed", str(args.seed), "--output", str(output), "--allow-unverified-parameters"]
    elif family == "bfv":
        command += ["--parameters", str(BFV / "parameters" / config["parameters"]),
            "--benchmark", "--widths", str(width), "--threads", str(threads),
            "--patterns", "random", "--repetitions", str(args.repetitions),
            "--warmup", str(args.warmup), "--seed", str(args.seed), "--output", str(output)]
    elif family == "parmesan":
        command += list(map(str, [width, threads, args.repetitions, args.warmup, output, args.seed]))
    else:
        command += common
        if family == "bernard":
            sys.path.insert(0, str(RING / "model" / "bernard"))
            import schedule as bernard_schedule
            path = case / "schedule.json"
            path.write_text(json.dumps(bernard_schedule.schedule(width, threads, config["phi"])))
            command += ["--schedule", str(path), "--ks", ",".join(map(str, config["ks"]))]
        if family == "radix" and "ks" in config:
            command += ["--ks", ",".join(map(str, config["ks"]))]
        if family == "st" and config.get("reported"):
            command += ["--refined-noise-assumption", "--boundary-delta-log2",
                        str(config["boundary_delta_log2"])]
            if config["boundary_delta_log2"] != 52:
                command += ["--allow-alternate-boundary"]
            if config.get("cc2_big_key"):
                command += ["--cc2-big-key"]
        elif family == "st":
            command += ["--parameters", str(SC / "parameters" / config["parameters"]),
                        "--allow-unverified-parameters"]
    cwd = SC / "sources/parmesan" if family == "parmesan" else BFV if family == "bfv" else RING
    return resolved, command, env, cwd
