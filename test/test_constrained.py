import sys
import os
import unittest
import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from src.constrained_min import InteriorPointOptimizer
from examples import quadratic_programming, qp_first_inequality_constrait, qp_second_inequality_constrait, qp_third_inequality_constrait, linear_programming, lp_first_inequality_constraint, lp_second_inequality_constraint, lp_third_inequality_constraint, lp_fourth_inequality_constraint
from src.utils import plot_iterations, plot_feasible_regions_3d, plot_feasible_set_2d


class TestInteriorPointOptimizer(unittest.TestCase):
    starting_point_qp = np.array([0.1, 0.2, 0.7], dtype=np.float64)
    starting_point_lp = np.array([0.5, 0.75], dtype=np.float64)
    minimizer = InteriorPointOptimizer()

    def test_qp(self):
        eq_constraint_mat = np.array([[1, 1, 1]]).reshape(1, -1)
        self._run_minimizer_test(
            func=quadratic_programming,
            x0=self.starting_point_qp,
            ineq_constraints=[qp_first_inequality_constrait, qp_second_inequality_constrait, qp_third_inequality_constrait],
            eq_constraints_mat=eq_constraint_mat,
            eq_constraints_rhs=np.array([1]),
            plot_3d=True
        )

    def test_lp(self):
        self._run_minimizer_test(
            func=linear_programming,
            x0=self.starting_point_lp,
            ineq_constraints=[lp_first_inequality_constraint, lp_second_inequality_constraint, lp_third_inequality_constraint, lp_fourth_inequality_constraint],
            eq_constraints_mat=np.array([]),
            eq_constraints_rhs=np.array([]),
            plot_3d=False
        )

    def _run_minimizer_test(self, func, x0, ineq_constraints, eq_constraints_mat, eq_constraints_rhs, plot_3d):
        x_s, obj_values, outer_x_s, outer_obj_values = self.minimizer.interior_pt(
            func=func,
            x0=x0,
            ineq_constraints=ineq_constraints,
            eq_constraints_mat=eq_constraints_mat,
            eq_constraints_rhs=eq_constraints_rhs,
        )

        print(f"Point of convergence: {x_s[-1]}")
        print(f"Objective value at point of convergence: {func(x_s[-1])[0]:.4f}")
        for ineq_constraint in ineq_constraints:
            print(
                f"{ineq_constraint.__name__} value at point of convergence: {ineq_constraint(x_s[-1])[0]:.4f}"
            )
        if eq_constraints_mat.size > 0:
            print(
                f"x + y + z value at point of convergence: {np.sum(x_s[-1]):.6f}"
            )

        if plot_3d:
            plot_feasible_regions_3d(x_s)
        else:
            plot_feasible_set_2d(x_s)

        plot_iterations(
            f"Objective function values of {func.__name__} function",
            outer_obj_values,
            obj_values,
            "Outer objective values",
            "Objective values",
        )


if __name__ == "__main__":
    unittest.main()
