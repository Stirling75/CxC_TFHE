"""Independent parameter expectations, transcribed from the locked library presets.

TFHE-rs 1.6.1/1.7.0: shortint/parameters/v1_{6,7}/classic/{gaussian,tuniform}/p_fail_2_minus_128/ks_pbs.rs.
PARMESAN: TFHE-rs 0.5.4 shortint/parameters/mod.rs plus author params.rs M4_C0 override.
Failure labels below are per-library-primitive metadata, never whole-product bounds.
"""
import math


def equal(actual, expected, path="parameters"):
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or actual.keys() != expected.keys():
            raise ValueError(f"{path}: parameter fields differ")
        for key in expected:
            equal(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise ValueError(f"{path}: parameter lengths differ")
        for i, (a, e) in enumerate(zip(actual, expected)):
            equal(a, e, f"{path}[{i}]")
    elif isinstance(expected, float):
        if (isinstance(actual, bool) or not isinstance(actual, (int, float)) or
                not math.isfinite(actual) or abs(actual - expected) > math.ulp(expected)):
            raise ValueError(f"{path}: {actual!r} != {expected!r}")
    elif type(actual) is not type(expected) or actual != expected:
        raise ValueError(f"{path}: {actual!r} != {expected!r}")


def gaussian(sigma):
    return {"Gaussian": {"mean": 0.0, "std": sigma}}


def library_parameters(preset):
    base = {"ciphertext_modulus": {"modulus": 0, "scalar_bits": 64},
            "encryption_key_choice": "Big", "pbs_base_log": 23, "pbs_level": 1}
    if preset == "parmesan":
        return dict(base, lwe_dimension=742, glwe_dimension=1, polynomial_size=2048,
                    lwe_modular_std_dev=7.069849454709433e-6,
                    glwe_modular_std_dev=2.9403601535432533e-16,
                    ks_base_log=3, ks_level=5, message_modulus=32, carry_modulus=1)
    base.update(glwe_noise_distribution=gaussian(2.845267479601915e-15),
                modulus_switch_noise_reduction_params="CenteredMeanNoiseReduction")
    if preset == "m2c2-gaussian":
        return dict(base, lwe_dimension=866, glwe_dimension=1, polynomial_size=2048,
                    lwe_noise_distribution=gaussian(2.046151696979124e-6),
                    ks_base_log=3, ks_level=5, message_modulus=4, carry_modulus=4,
                    max_noise_level=5, log2_p_fail=-128.597)
    if preset not in ("m1c1-gaussian", "m1c1-tuniform"):
        raise ValueError(f"unknown preset {preset}")
    base.update(lwe_dimension=837, glwe_dimension=4, polynomial_size=512,
                lwe_noise_distribution=gaussian(3.3747142481837397e-6),
                ks_base_log=5, ks_level=3, message_modulus=2, carry_modulus=2,
                max_noise_level=3, log2_p_fail=-128.186)
    if preset == "m1c1-tuniform":
        base.update(lwe_dimension=879, log2_p_fail=-144.322,
                    lwe_noise_distribution={"TUniform": {"_phantom": None, "bound_log2": 46}},
                    glwe_noise_distribution={"TUniform": {"_phantom": None, "bound_log2": 17}})
    return base


def hybrid_parameters(plan, seed=None):
    env = plan["env"]
    expected = {"tfhe_rs": "1.6.1", "input": "PBS-refreshed radix-4", "input_scale": 8,
                "message_modulus": 4, "carry_modulus": 4,
                "ciphertext_modulus": {"modulus": 0, "scalar_bits": 64},
                "lwe_noise_distribution": gaussian(2.046151696979124e-6),
                "glwe_noise_distribution": gaussian(2.845267479601915e-15),
                "lwe_dimension": 866, "normalizer_ring": [2048, 1],
                "cbs_ring": list(map(int, env["CBS_RING_SHAPE"].split("x"))),
                "auto_fft": env["DIRECT_AUTO_FFT"],
                "log_lut_count": int(env["DIRECT_LOG_LUT_COUNT"]),
                "whole_multiplier_log2_failure": None,
                "operand_pattern": "random", "random_operands_seed": seed}
    for key, prefix in (("pbs", "PBS"), ("ks", "KS"), ("lift_pbs", "CBS_LIFT_PBS"),
                        ("lift_ks", "CBS_LIFT_KS"), ("auto", "DIRECT_AUTO"),
                        ("ss", "DIRECT_SS"), ("cbs", "DIRECT_CBS")):
        expected[key] = [int(env[prefix + suffix]) for suffix in ("_BASE_LOG", "_LEVEL")]
    return expected


def check_hybrid_model(plan, profile):
    expected = hybrid_parameters(plan)
    equal([profile.poly_n, profile.glwe_k], expected["cbs_ring"], "model CBS ring")
    equal(profile.lwe_n, expected["lwe_dimension"], "model LWE dimension")
    for key, field in (("pbs", "pbs"), ("ks", "ks"), ("lift_pbs", "cbs_lift_pbs"),
                       ("lift_ks", "cbs_lift_ks"), ("auto", "auto"), ("ss", "ss"), ("cbs", "cbs")):
        equal([getattr(profile, field + suffix) for suffix in ("_base_log", "_level")],
              expected[key], f"model {key}")
    for name in ("lwe", "glwe"):
        sigma = expected[name + "_noise_distribution"]["Gaussian"]["std"]
        equal(getattr(profile, name + "_var"), sigma**2, f"model {name} variance")
