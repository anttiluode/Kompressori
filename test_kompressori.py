import unittest
import numpy as np

import kompressori as k
import gate2_crossvalidate as g2
import gate3_transfer as g3
import gate5_unbiased_scan as g5
import gate6_lowrank_update as g6
import gate7_update_composition as g7
import gate8_operator_overlap as g8


class KompressoriTests(unittest.TestCase):
    def test_step_stays_finite(self):
        p = k.Params()
        phi, old = k.initial_gaussian(24)
        for _ in range(20):
            phi, old = k.step(phi, old, p)
        self.assertTrue(np.isfinite(phi).all())

    def test_mask_fraction_is_exact_to_rounding(self):
        target, _ = k.initial_gaussian(20)
        rng = np.random.default_rng(1)
        m = k.make_mask("random", target, 0.10, rng)
        self.assertEqual(int(m.sum()), 40)

    def test_gate2_random_mask_budget(self):
        m = g2.random_mask(20, 0.05, np.random.default_rng(2))
        self.assertEqual(int(m.sum()), 20)
        self.assertEqual(m.dtype, np.bool_)

    def test_small_probe_is_nearly_linear(self):
        p = k.Params()
        phi, old = k.evolve_state(24, p, 40)
        ref = k.rollout(phi, old, p, 8)
        d, _ = k.torus_packet(24, np.random.default_rng(3))
        eps = 1e-4
        a = k.rollout(phi + eps*d, old + eps*d, p, 8) - ref
        b = k.rollout(phi + 0.5*eps*d, old + 0.5*eps*d, p, 8) - ref
        rel = np.linalg.norm(a - 2*b) / (np.linalg.norm(a) + 1e-15)
        self.assertLess(rel, 0.02)

    def test_gate3_torus_distance_wraps(self):
        self.assertAlmostEqual(g3.torus_distance((1.0, 5.0), (19.0, 5.0), 20), 2.0)

    def test_gate3_zero_preparation_gives_unit_transfer(self):
        p = k.Params()
        target, old = k.evolve_state(20, p, 30)
        base_phi, base_old = g3.advance(target.copy(), old.copy(), p, 4)
        base_future = k.rollout(base_phi.copy(), base_old.copy(), p, 5)
        row = g3.evaluate_pair(
            pair=0,
            epsilon_a=0.0,
            epsilon_b=1e-4,
            target=target,
            old=old,
            base_delayed=(base_phi, base_old),
            base_future=base_future,
            p=p,
            delay=4,
            horizon=5,
        )
        self.assertAlmostEqual(row["transfer_gain"], 1.0, places=10)
        self.assertAlmostEqual(row["response_corr"], 1.0, places=10)

    def test_gate5_locations_are_deterministic_and_unique(self):
        a = g5.deterministic_a_locations(40, 30)
        b = g5.deterministic_a_locations(40, 30)
        self.assertEqual(a, b)
        self.assertEqual(len(a), 30)
        self.assertEqual(len(set(a)), 30)
        self.assertEqual(a[0], (7, 3))

    def test_gate5_summary_counts_thresholds(self):
        s = g5.summarize([0.8, 1.0, 1.2, 1.4])
        self.assertEqual(s["count"], 4)
        self.assertAlmostEqual(s["mean"], 1.1)
        self.assertAlmostEqual(s["fraction_gain_gt_1_1"], 0.5)
        self.assertAlmostEqual(s["fraction_gain_lt_0_9"], 0.25)

    def test_gate6_whitening_makes_input_gram_identity(self):
        x = np.array([[1.0, 0.2], [0.3, 1.0], [0.4, 0.5]])
        w = g6.whitening_from_inputs(x)
        gram = w @ (x.T @ x) @ w
        self.assertTrue(np.allclose(gram, np.eye(2), atol=1e-10))

    def test_gate6_spectrum_summary_has_expected_rank(self):
        s = g6.spectrum_summary(np.array([9.0, 1.0, 0.0]))
        self.assertEqual(s["rank90"], 1)
        self.assertEqual(s["rank95"], 2)
        self.assertLess(s["effective_rank"], 2.0)

    def test_gate7_torus_distance_wraps(self):
        self.assertAlmostEqual(g7.torus_distance((1, 1), (39, 1), 40), 2.0)

    def test_gate8_pair_protocol_is_geometry_only_and_deterministic(self):
        sites = g5.deterministic_a_locations(40, 30)
        a = g8.selected_pairs(sites, 40)
        b = g8.selected_pairs(sites, 40)
        self.assertEqual(a, b)
        self.assertEqual(len(a), 34)
        counts = {}
        for d2, _, _, _ in a:
            counts[d2] = counts.get(d2, 0) + 1
        self.assertEqual(counts[32], 8)
        self.assertEqual(counts[800], 8)
        for d2 in g8.INTERMEDIATE_DISTANCE_SQUARED:
            self.assertEqual(counts[d2], 3)

    def test_gate8_identical_low_rank_views_have_unit_subspace_overlap(self):
        rng = np.random.default_rng(8)
        delta = rng.normal(size=(20, 4))
        view = g8.low_rank_view(delta, np.eye(4), energy_fraction=.999999)
        metrics = g8.overlap_metrics(view, view)
        self.assertAlmostEqual(metrics["operator_cosine"], 1.0, places=12)
        self.assertAlmostEqual(metrics["input_subspace_overlap"], 1.0, places=12)
        self.assertAlmostEqual(metrics["output_subspace_overlap"], 1.0, places=12)


if __name__ == "__main__":
    unittest.main()
