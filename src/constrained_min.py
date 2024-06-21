import numpy as np
import math

class InteriorPointOptimizer:
    t_param = 1
    mu_param = 10

    def interior_pt(self, func, x0, ineq_constraints, eq_constraints_mat, eq_constraints_rhs):
        current_x = x0
        f_current, g_current, h_current = func(current_x)
        f_phi, g_phi, h_phi = self.phi(ineq_constraints, current_x)
        t_param = self.t_param

        x_history = [x0]
        obj_values_history = [f_current]
        outer_x_history = [x0]
        outer_obj_values_history = [f_current]

        f_current, g_current, h_current = self.update_fgh(f_current, g_current, h_current, f_phi, g_phi, h_phi, t_param)

        outer_iteration = 0
        while outer_iteration < 20:
            block_matrix = self.construct_block_matrix(eq_constraints_mat, h_current)
            equality_vector = self.construct_equality_vector(g_current, block_matrix.shape[0])

            previous_x = current_x
            previous_f = f_current

            inner_iteration = 0
            while inner_iteration < 100:
                if self.convergence_check(inner_iteration, current_x, previous_x, f_current, previous_f):
                    break

                search_direction = self.compute_search_direction(block_matrix, equality_vector, len(current_x))
                lambda_param = self.compute_lambda(search_direction, h_current)
                if 0.5 * (lambda_param**2) < 10e-12:
                    break

                step_size = self.wolfe(func, search_direction, current_x)

                previous_x, previous_f = current_x, f_current

                current_x = current_x + step_size * search_direction
                f_current, g_current, h_current = func(current_x)
                f_phi, g_phi, h_phi = self.phi(ineq_constraints, current_x)

                x_history.append(current_x)
                obj_values_history.append(f_current)

                f_current, g_current, h_current = self.update_fgh(f_current, g_current, h_current, f_phi, g_phi, h_phi, t_param)

                inner_iteration += 1

            outer_x_history.append(current_x)
            outer_obj_values_history.append((f_current - f_phi) / t_param)

            if len(ineq_constraints) / t_param < 10e-10:
                return x_history, obj_values_history, outer_x_history, outer_obj_values_history

            t_param = self.mu_param * t_param
            outer_iteration += 1

        return x_history, obj_values_history, outer_x_history, outer_obj_values_history

    def wolfe(self, func, search_direction, current_x, wolfe_constant=0.01, backtrack_constant=0.5):
        step_size = 1
        current_func_result = func(current_x)
        current_func_val, current_grad = current_func_result[0], current_func_result[1]

        while not self.satisfies_wolfe_condition(func, current_x, step_size, search_direction, wolfe_constant, current_func_val, current_grad):
            step_size *= backtrack_constant

        return step_size

    def satisfies_wolfe_condition(self, func, current_x, step_size, search_direction, wolfe_constant, current_func_val, current_grad):
        new_x = current_x + step_size * search_direction
        new_func_val = func(new_x)[0]
        rhs = current_func_val + wolfe_constant * step_size * np.dot(current_grad.T, search_direction)
        return new_func_val <= rhs

    def phi(self, ineq_constraints, current_x):
        total_f, total_g, total_h = 0, 0, 0

        for constraint_func in ineq_constraints:
            f_current, g_current, h_current = constraint_func(current_x)
            total_f += self.compute_f(f_current)
            total_g += self.compute_g(f_current, g_current)
            total_h += self.compute_h(f_current, g_current, h_current)

        return -total_f, -total_g, -total_h

    def compute_f(self, f_current):
        return math.log(-f_current)

    def compute_g(self, f_current, g_current):
        return g_current / f_current

    def compute_h(self, f_current, g_current, h_current):
        gradient = g_current / f_current
        gradient_mesh = self.compute_gradient_mesh(gradient)
        return (h_current * f_current - gradient_mesh) / f_current**2

    def compute_gradient_mesh(self, gradient):
        gradient_reshaped = gradient.reshape(gradient.shape[0], -1)
        return np.tile(gradient_reshaped, (1, gradient.shape[0])) * np.tile(gradient_reshaped.T, (gradient.shape[0], 1))

    def update_fgh(self, f_current, g_current, h_current, f_phi, g_phi, h_phi, t_param):
        f_current = t_param * f_current + f_phi
        g_current = t_param * g_current + g_phi
        h_current = t_param * h_current + h_phi
        return f_current, g_current, h_current

    def construct_block_matrix(self, eq_constraints_mat, h_current):
        if eq_constraints_mat.size:
            upper_block = np.concatenate([h_current, eq_constraints_mat.T], axis=1)
            size_zeros = (eq_constraints_mat.shape[0], eq_constraints_mat.shape[0])
            lower_block = np.concatenate([eq_constraints_mat, np.zeros(size_zeros)], axis=1)
            return np.concatenate([upper_block, lower_block], axis=0)
        else:
            return h_current

    def construct_equality_vector(self, g_current, total_size):
        return np.concatenate([-g_current, np.zeros(total_size - len(g_current))])

    def convergence_check(self, inner_iteration, current_x, previous_x, f_current, previous_f):
        if inner_iteration != 0 and sum(abs(current_x - previous_x)) < 10e-8:
            return True
        if inner_iteration != 0 and (previous_f - f_current < 10e-12):
            return True
        return False

    def compute_search_direction(self, block_matrix, equality_vector, length_x):
        return np.linalg.solve(block_matrix, equality_vector)[:length_x]

    def compute_lambda(self, search_direction, h_current):
        return np.matmul(search_direction.transpose(), np.matmul(h_current, search_direction)) ** 0.5
