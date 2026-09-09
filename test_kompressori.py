import unittest
import numpy as np

import kompressori as k


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


if __name__ == "__main__":
    unittest.main()
