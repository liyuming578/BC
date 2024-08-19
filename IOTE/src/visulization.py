import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata


def show(param_names, results_df, save_path=True):
    if len(param_names) == 1:
        # When there is only one hyperparameter, plot a 2D line graph.
        param = param_names[0]
        plt.figure(figsize=(10, 8))
        plt.plot(results_df[param], results_df['mae_avg'], marker='o', linestyle='-', color='b')
        plt.xlabel(param)
        plt.ylabel('MAE')
        plt.title(f'{param} vs. MAE')

        if save_path:
            plt.savefig('D:\\Users\\allmi\\PycharmProjects\\IOTE\\results\\hyperparameters_figure.png')  # 保存图像
        plt.show()

    elif len(param_names) == 2:
        # When there are two hyperparameters, plot a 3D surface graph.
        param1, param2 = param_names

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Create a grid to interpolate the surface.
        param1_values = np.linspace(min(results_df[param1]), max(results_df[param1]), 100)
        param2_values = np.linspace(min(results_df[param2]), max(results_df[param2]), 100)
        param1_grid, param2_grid = np.meshgrid(param1_values, param2_values)
        mae_grid = griddata(
            (results_df[param1], results_df[param2]), results_df['mae_avg'],
            (param1_grid, param2_grid), method='linear'
        )

        # Plot the surface graph.
        ax.plot_surface(param1_grid, param2_grid, mae_grid, cmap='viridis', edgecolor='none')
        # ax.zaxis.set_major_formatter(FormatStrFormatter('%.9e'))
        ax.set_xlabel(param1)
        ax.set_ylabel(param2)
        ax.set_zlabel('MAE')
        ax.set_title(f'{param1} and {param2} vs. MAE')

        ax.tick_params(axis='z')

        if save_path:
            plt.savefig('D:\\Users\\allmi\\PycharmProjects\\IOTE\\results\\hyperparameters_figure.png')  # save image
        plt.show()

    else:
        print("This feature only supports visualization of one or two hyperparameters.")
