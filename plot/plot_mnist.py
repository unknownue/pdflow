
import matplotlib.pyplot as plt


def plot_mnist(imgs, matrix_shape, save_path=None, is_show=False):
    w, h = matrix_shape[0] * 3, matrix_shape[1] * 5
    fig = plt.figure(figsize=(w, h))
    num_img = len(imgs)

    for i in range(num_img):
        ax = plt.subplot(matrix_shape[0], matrix_shape[1], i + 1)

        x, y = imgs[i].transpose(0, 1)
        scatter = plt.scatter(x, y)

        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        scatter.axes.invert_yaxis()

    if save_path is not None:
        plt.savefig(save_path)
    if is_show:
        plt.show()
