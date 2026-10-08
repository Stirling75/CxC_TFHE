import unittest
import random
import make_plan as m
import verify_constructions as c

class PlanTests(unittest.TestCase):
    def test_every_schedule_preserves_the_modular_product(self):
        for width in c.base.WIDTHS:
            for mode in ("baseline","cached"):
                with self.subTest(width=width, mode=mode):
                    plan = m.derive(width,mode)
                    self.assertIsNone(plan["log2_failure"])
                    self.assertLess(plan["reference_screen_log2"],-128)
                    rng = random.Random(width)
                    d = width//2
                    lhs = [rng.randrange(4) for _ in range(d)]
                    rhs = [rng.randrange(4) for _ in range(d)]
                    val = lambda digits: sum(x << (2*i) for i,x in enumerate(digits))
                    expected = val(lhs)*val(rhs) % (1<<width)
                    columns = [[] for _ in range(d)]
                    for u in range(width//8):
                        for v in range(width//8-u):
                            prod = val(lhs[4*u:4*u+4])*val(rhs[4*v:4*v+4])
                            for t in range(8):
                                q=4*(u+v)+t
                                if q<d: columns[q].append((prod>>(2*t))&3)
                    represented = lambda cs: sum(sum(col)<<(2*q) for q,col in enumerate(cs))%(1<<width)
                    self.assertEqual(represented(columns),expected)
                    for wave in plan["waves"]:
                        out = [[] for _ in range(d)]
                        jobs=[]
                        for q,record in enumerate(wave):
                            self.assertEqual(len(record["terms"]),len(columns[q]))
                            self.assertEqual(sorted(i for g in record["groups"] for i in g),list(range(len(columns[q]))))
                            for group in record["groups"]:
                                bound=sum(record["terms"][i]["bound"] for i in group)
                                self.assertLessEqual(bound,15)
                                values=[columns[q][i] for i in group]
                                if len(group)<3: out[q].extend(values)
                                else: jobs.append((q,bound,sum(values)))
                        for q,bound,value in jobs:
                            out[q].append(value%4)
                            if q+1<d and bound//4: out[q+1].append(value//4)
                        self.assertEqual(represented(out),expected)
                        columns=out
                    self.assertTrue(all(len(col)<=2 for col in columns))
                    self.assertEqual(list(map(len,columns)),list(map(len,plan["final_columns"])))

    def test_parameter_mapping(self):
        env=m.parameter_environment("cached",False)
        self.assertEqual((env["DIRECT_CBS_BASE_LOG"],env["DIRECT_CBS_LEVEL"]),("3","6"))
        self.assertEqual((env["CBS_LIFT_PBS_BASE_LOG"],env["CBS_LIFT_PBS_LEVEL"]),("9","4"))
        self.assertEqual((env["CBS_LIFT_KS_BASE_LOG"],env["CBS_LIFT_KS_LEVEL"]),("7","2"))
        self.assertEqual(env["CBS_CENTER_SELECTOR_BOX"],"1")

    def test_legacy_reference_failure_is_a_value_error(self):
        original = c.base.simulate
        class Rejected:
            union_log2_pfail = -100.0
        c.base.simulate = lambda *args, **kwargs: Rejected()
        try:
            with self.assertRaisesRegex(ValueError, "legacy reference screen rejected"):
                m.derive(16, "baseline")
        finally:
            c.base.simulate = original

    def test_normalizer_cell_is_asymmetric(self):
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        import heterogeneous_screen as hs
        self.assertEqual((hs.CELL_LOWER_STEPS, hs.CELL_UPPER_STEPS), (64.5, 63.5))
        variance = 2.0**100
        asymmetric = hs.normalizer_input_log2_pfail(866, variance)
        symmetric = c.est.pbs_input_log2_pfail(866, c.est.Q, 2048, 0, c.est.delta_from_bits(4),
                                               variance, "gaussian", centered_binary_ms=True)
        # The upper side moves in by half a step and the lower side out by half a step.
        self.assertNotAlmostEqual(asymmetric, symmetric, places=3)
        upper_only = c.est.log2_erfc((63.5 * 2.0**52 - (866 / 4 + 0.5)) /
            (2 * (variance + c.est.centered_binary_ms_decision_noise(866, c.est.Q, 4096)[0])) ** 0.5) - 1
        self.assertGreater(asymmetric, upper_only)
        self.assertLess(asymmetric, upper_only + 1)

    def test_runner_counts_match_screen_events(self):
        import sys
        from pathlib import Path
        root = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(root))
        import heterogeneous_screen as hs
        import importlib.util
        spec = importlib.util.spec_from_file_location("hybrid_run", root / "run.py")
        run = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(run)
        for width in (16, 64):
            for mode, chunk in (("baseline", 4), ("baseline", 8), ("cached", 8)):
                plan = m.derive(width, mode, chunk_bits=chunk)
                p, v = hs.primitive(2048, (3, 16), (13, 4), (7, 2), chunk_bits=chunk,
                                    cbs=tuple(int(plan["env"]["DIRECT_CBS" + s]) for s in ("_BASE_LOG", "_LEVEL")))
                analysis = hs.analyze(plan, p, v)
                counts = run.normalization_counts(plan, 1)
                self.assertEqual(counts["reduction_pbs"], analysis["reduction_pbs"])
                self.assertEqual(counts["row_refresh"], analysis["row_refresh"])

    def test_squaring_is_outside_this_experiment(self):
        for mode in ("baseline", "cached", "fused"):
            with self.assertRaises(AssertionError):
                m.derive(256, mode, True)

if __name__=="__main__":
    unittest.main()
