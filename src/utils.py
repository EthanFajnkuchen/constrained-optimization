import matplotlib.pyplot as plt
import numpy as np

def plot_iterations(title, obj_values_1=None, obj_values_2=None, label_1=None, label_2=None):
    _ , ax = plt.subplots()

    def plot_obj_values(obj_values, label, color):
        if obj_values is not None:
            ax.plot(range(len(obj_values)), obj_values, label=label, color=color)

    plot_obj_values(obj_values_1, label_1, color='red')
    plot_obj_values(obj_values_2, label_2, color='blue')

    ax.legend()
    ax.set_title(title)
    ax.set_xlabel("Number of iterations")
    ax.set_ylabel("Objective function value")
    plt.show()


def plot_feasible_set_2d(path_points=None):
    def plot_feasible_region():
        d = np.linspace(-2, 4, 300)
        x, y = np.meshgrid(d, d)
        plt.imshow(
            ((y >= -x + 1) & (y <= 1) & (x <= 2) & (y >= 0)).astype(int),
            extent=(x.min(), x.max(), y.min(), y.max()),
            origin="lower",
            cmap="Blues",
            alpha=0.3,
        )

    def plot_constraints():
        x = np.linspace(0, 4, 2000)
        plt.plot(x, -x + 1, label=r"$y \geq -x + 1$", color='green')
        plt.plot(x, np.ones(x.size), label=r"$y \leq 1$", color='red')
        plt.plot(x, np.zeros(x.size), label=r"$y \geq 0$", color='orange')
        plt.plot(np.ones(x.size) * 2, x, label=r"$x \leq 2$", color='purple')

    def plot_path(path_points):
        if path_points is not None:
            x_path, y_path = zip(*path_points)
            plt.plot(x_path, y_path, label="algorithm's path", color="black", marker=".", linestyle="--")

    plot_feasible_region()
    plot_constraints()
    plot_path(path_points)

    plt.xlim(0, 3)
    plt.ylim(0, 2)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
    plt.xlabel(r"$x$")
    plt.ylabel(r"$y$")
    plt.suptitle('Feasible region and path')
    plt.show()


def plot_feasible_regions_3d(path):
    def plot_feasible_region(ax):
        ax.plot_trisurf([1, 0, 0], [0, 1, 0], [0, 0, 1], color='blue', alpha=0.5)

    def plot_path(ax, path):
        path = np.array(path)
        ax.plot(path[:, 0], path[:, 1], path[:, 2], label='Path', color="red")
        ax.scatter(path[-1][0], path[-1][1], path[-1][2], s=50, c='purple', marker='o', label='Final candidate')

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    plot_feasible_region(ax)
    plot_path(ax, path)

    ax.set_title("Feasible Regions and Path")
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.legend()
    ax.view_init(50, 50)
    plt.show()

