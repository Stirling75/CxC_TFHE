import csv
import copy
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from campaign import parse
from cases import CATALOG, binary_for, prepare, resolve
from results import read_rows
from runtime import serializable, source_id
from parameter_checks import equal, hybrid_parameters, library_parameters
import runtime


class CampaignTests(unittest.TestCase):
    def test_clot_bfv_passes_each_width_without_unconditional_approval(self):
        for width in CATALOG["clot-bfv"]["widths"]:
            record = resolve("clot-bfv", width)
            self.assertLess(record["analysis"]["raw_log2_union"], -128)
            self.assertFalse(record["whole_multiplier_approved"])
            self.assertEqual(record["analysis"]["events"],
                             sum(f["count"] for f in record["analysis"]["families"].values()))

    def test_clot_bfv_uses_full_evaluation_and_explicit_parameters(self):
        with patch("sys.argv", ["campaign.py", "bench", "--dry-run", "--output", "unused",
                                "--methods", "clot-bfv"]):
            args = parse()
        with tempfile.TemporaryDirectory() as directory:
            _, command, env, cwd = prepare("clot-bfv", 256, 8, args, Path(directory))
            self.assertIn("--benchmark", command)
            self.assertIn("--parameters", command)
            self.assertNotIn("--kernel-only", command)
            self.assertEqual(command[command.index("--repetitions")+1], "2")
            self.assertEqual(command[command.index("--warmup")+1], "1")
            self.assertEqual(env["RAYON_NUM_THREADS"], "8")
            self.assertEqual(binary_for("clot-bfv", args.build_root).name, "bfv-style-probe")

    def test_benchmark_defaults_to_two_measured_runs(self):
        with patch("sys.argv", ["campaign.py", "bench", "--dry-run", "--output", "unused"]):
            args = parse()
        self.assertEqual((args.repetitions, args.warmup), (2, 1))
        self.assertFalse(args.allow_failing_screen)

    def test_clot_w256_is_labelled_as_missing_the_target(self):
        record = resolve("clot-bfv", 256)
        self.assertLess(record["analysis"]["raw_log2_union"], -128)
        self.assertFalse(record["failure_target_met"])
        self.assertTrue(record["screen_failed"])
        self.assertTrue(resolve("clot-bfv", 128)["failure_target_met"])

    def test_st_reported_is_unverifiable_and_never_blocked(self):
        with patch("sys.argv", ["campaign.py", "bench", "--dry-run", "--output", "unused",
                                "--methods", "st-reported-r2"]):
            args = parse()
        for width in CATALOG["st-reported-r2"]["widths"]:
            record = resolve("st-reported-r2", width)
            self.assertTrue(math.isnan(record["whole_product_log2_estimate"]))
            self.assertFalse(record["failure_target_met"])
            self.assertFalse(record["screen_failed"])
        with tempfile.TemporaryDirectory() as directory:
            _, command, _, _ = prepare("st-reported-r2", 64, 4, args, Path(directory))
        self.assertIn("--refined-noise-assumption", command)
        self.assertEqual(command[command.index("--boundary-delta-log2")+1], "52")
        self.assertNotIn("--allow-alternate-boundary", command)
        self.assertNotIn("--parameters", command)

    def test_library_methods_are_labelled_but_never_blocked(self):
        for method, width in (("tfhe-rs", 16), ("trifan", 256), ("parmesan", 32)):
            record = resolve(method, width)
            self.assertFalse(record["failure_target_met"])
            self.assertFalse(record["screen_failed"])

    def test_failing_screen_blocks_unless_explicitly_allowed(self):
        from campaign import run
        with patch("sys.argv", ["campaign.py", "bench", "--dry-run", "--methods", "clot-bfv",
                                "--widths", "256", "--output", "unused"]):
            args = parse()
        with self.assertRaisesRegex(RuntimeError, "allow-failing-screen"):
            run(args)

    def test_explicit_repetitions_and_smoke_protocol(self):
        with patch("sys.argv", ["campaign.py", "bench", "--dry-run", "--output", "unused",
                                "--repetitions", "20"]):
            self.assertEqual(parse().repetitions, 20)
        with patch("sys.argv", ["campaign.py", "smoke", "--dry-run", "--output", "unused",
                                "--repetitions", "5"]):
            args = parse()
        self.assertEqual((args.repetitions, args.warmup), (1, 0))

    def test_five_run_protocol_requires_warmup_and_all_measurements(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "timings.csv"
            rows = ["ok,trial,warmup", "true,0,true"]
            rows += [f"true,{i},false" for i in range(1, 6)]
            path.write_text("\n".join(rows) + "\n")
            self.assertEqual(len(read_rows(path, 1, 5)), 6)
            path.write_text("\n".join(rows[:-1]) + "\n")
            with self.assertRaisesRegex(ValueError, "incomplete trial count"):
                read_rows(path, 1, 5)

    def test_no_whole_bound_is_implicitly_approved(self):
        for method in CATALOG:
            with self.subTest(method=method):
                record = resolve(method, 16)
                self.assertIs(record["whole_multiplier_approved"], False)

    def test_trifan_pruning_count_and_output(self):
        record = resolve("trifan", 256)
        self.assertEqual(record["logical_lut_calls"], 98432)
        self.assertEqual(record["contract"]["output"], "lower-W bits")

    def test_row_cache_control_changes_only_the_kernel(self):
        cached = resolve("hybrid-grouped-rev-ld", 16)["plan"]
        plain = resolve("hybrid-grouped-rev-no-cache-ld", 16)["plan"]
        self.assertEqual(cached["waves"], plain["waves"])
        diff = {k for k in set(cached["env"]) | set(plain["env"]) if cached["env"].get(k) != plain["env"].get(k)}
        self.assertEqual(diff, {"FUSED_KERNEL"})

    def test_paper_parameter_sets(self):
        # Table 4: refresh (lift PBS), automorphism, scheme switch, CBS decompositions
        expected = {"hybrid-grouped-rev-ld": ([8, 5], [7, 6], [10, 5], [5, 4]),
                    "hybrid-grouped-rev-mvb": ([8, 5], [7, 6], [10, 5], [5, 4]),
                    "hybrid-cached-rev-ld": ([9, 4], [10, 4], [15, 3], [5, 4]),
                    "hybrid-cached-rev-mvb": ([9, 4], [10, 4], [15, 3], [5, 4]),
                    "hybrid-8x8-rev-ld": ([9, 4], [10, 4], [15, 3], [5, 4]),
                    "hybrid-8x8-rev-ld-sks": ([9, 4], [10, 4], [15, 3], [5, 4]),
                    "hybrid-4x4-rev-ld": ([9, 4], [8, 5], [12, 4], [6, 3])}
        for method, (lift_pbs, auto, ss, cbs) in expected.items():
            for width in CATALOG[method]["widths"]:
                with self.subTest(method=method, width=width):
                    record = resolve(method, width)
                    params = hybrid_parameters(record["plan"])
                    self.assertEqual((params["lift_pbs"], params["auto"], params["ss"], params["cbs"]),
                                     (lift_pbs, auto, ss, cbs))
                    self.assertEqual((params["lift_ks"], params["pbs"], params["ks"]), ([7, 2], [23, 1], [2, 8]))
                    self.assertEqual(params["lwe_dimension"], 866)
                    self.assertTrue(record["failure_target_met"], (method, width))
                    self.assertIs(record["whole_multiplier_approved"], False)

    def test_control_gate_rejects_weaker_auto_ss(self):
        for method in ("hybrid-4x4-rev-ld", "hybrid-8x8-rev-ld", "hybrid-cached-rev-ld"):
            resolve.cache_clear()
            try:
                with patch.dict(CATALOG[method], auto=[9, 5], ss=[13, 3]):
                    with self.subTest(method=method):
                        self.assertTrue(resolve(method, 256)["screen_failed"])
            finally:
                resolve.cache_clear()

    def test_cached_screen_retains_shared_gadget_rounding(self):
        import heterogeneous_screen as screen
        record = resolve("hybrid-cached-rev-ld", 32)
        _, prim = screen.primitive(2048, (10, 4), (15, 3), (7, 2), cbs=(5, 4))
        columns = screen.product_columns(record["plan"], prim)
        term = columns[0][0]
        self.assertEqual(term.private_var, 8 * prim.cmux_gadget_var)
        self.assertAlmostEqual(term.shared[("cached-gadget", 0)]**2,
                               8 * prim.cmux_gadget_var, delta=prim.cmux_gadget_var * 1e-12)

    def test_control_screen_rejects_duplicate_partition_term(self):
        import heterogeneous_screen as screen
        plan = copy.deepcopy(resolve("hybrid-8x8-rev-ld", 32)["plan"])
        p, v = screen.primitive(2048, (10, 4), (15, 3), (7, 2), cbs=(5, 4))
        for record in plan["waves"][0]:
            if record["groups"]:
                record["groups"][0].append(record["groups"][0][0])
                break
        with self.assertRaisesRegex(ValueError, "duplicates"):
            screen.analyze(plan, p, v)

    def test_unsupported_width_not_zero_latency(self):
        with self.assertRaises(ValueError):
            resolve("parmesan", 64)

    def test_source_hash_order_and_nonfinite_json(self):
        self.assertEqual(source_id({"a": "1", "b": "2"}), source_id({"b": "2", "a": "1"}))
        self.assertEqual(serializable({"tail": -math.inf}), {"tail": "-inf"})

    def test_vendored_sources_and_cargo_config_are_hashed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dependency = root / "vendor/example-1.0/src/lib.rs"
            config = root / ".cargo/config.toml"
            for path in (dependency, config):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("original\n")
            with patch.object(runtime, "ROOT", root):
                before = runtime.source_record()
                self.assertEqual(set(before), {"vendor/example-1.0/src/lib.rs", ".cargo/config.toml"})
                for path in (dependency, config):
                    path.write_text("changed\n")
                    self.assertNotEqual(source_id(before), source_id(runtime.source_record()))
                    path.write_text("original\n")

    def test_failed_incomplete_or_reordered_trials_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "timings.csv"
            def write(rows):
                with path.open("w", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=["ok", "trial", "warmup"])
                    writer.writeheader()
                    writer.writerows(rows)
            good = [{"ok": "true", "trial": 0, "warmup": "true"},
                    {"ok": "true", "trial": 1, "warmup": "false"}]
            write(good)
            self.assertEqual(len(read_rows(path, 1, 1)), 2)
            for rows in [good[:1], good[::-1], [dict(good[0], ok="false"), good[1]], good + good[:1],
                         [good[0], dict(good[1], warmup="invalid")]]:
                write(rows)
                with self.assertRaises(ValueError):
                    read_rows(path, 1, 1)

    def test_hybrid_trial_index_is_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "timings.csv"
            path.write_text("ok,trial_index,warmup\ntrue,0,0\ntrue,0,0\n")
            with self.assertRaises(ValueError):
                read_rows(path, 0, 2)
            path.write_text("ok,trial_index,warmup,ok\ntrue,0,0,true\n")
            with self.assertRaises(ValueError):
                read_rows(path, 0, 1)

    def test_every_runtime_parameter_leaf_is_checked(self):
        records = [library_parameters(p) for p in
                   ("m1c1-gaussian", "m1c1-tuniform", "m2c2-gaussian", "parmesan")]
        records += [hybrid_parameters(resolve(m, 16)["plan"]) for m in CATALOG
                    if CATALOG[m]["family"] == "hybrid"]
        def leaves(value, path=()):
            if isinstance(value, (dict, list)):
                for key in (value if isinstance(value, dict) else range(len(value))):
                    yield from leaves(value[key], path + (key,))
            else:
                yield path, value
        for expected in records:
            equal(copy.deepcopy(expected), expected)
            for path, value in leaves(expected):
                changed = copy.deepcopy(expected)
                parent = changed
                for key in path[:-1]:
                    parent = parent[key]
                parent[path[-1]] = value + 1 if isinstance(value, (int, float)) else "wrong"
                with self.subTest(path=path), self.assertRaises(ValueError):
                    equal(changed, expected)


if __name__ == "__main__":
    unittest.main()
