import numpy as np
from matplotlib import pyplot as plt
from convolutional_vae import CVAE


def visualize_results(model, test_data, images_per_row=10):
    """Visualize the given test images and their encode/decode reconstruction

    Args:
        model (CVAE):
        test_data (ndarray): Numpy array of N images, each with shape `model.img_shape`
        images_per_row (int, optional): Number of images per row. Must be a submultiple of `test_data.shape[0]`

    """
    encoded = model.encode(test_data)
    decoded = model.decode(encoded)

    f, axes = plt.subplots(int(test_data.shape[0] / images_per_row) * 2, images_per_row)
    ax_idx = 0
    for img_idx in range(test_data.shape[0]):
        if img_idx % images_per_row == 0 and img_idx > 0:
            ax_idx += 2
        axes[ax_idx, img_idx % images_per_row].imshow(test_data[img_idx])
        axes[ax_idx + 1, img_idx % images_per_row].imshow(decoded[img_idx].numpy())
    plt.show()


def visualize_frames(model, test_data, fps=1.):
    """Visualize original images and their encode/decode reconstruction at the given frame-rate

    Args:
        model (CVAE): Trained CVAE model
        test_data (np.ndarray): Array of images
        fps (float): Images displayed per second

    """

    plt.ion()
    f, axes = plt.subplots(1, 2)
    encoded = model.encode(test_data)
    decoded = model.decode(encoded)
    for i in range(test_data.shape[0]):
        axes[0].imshow(test_data[i])
        axes[1].imshow(decoded[i])
        plt.pause(1. / fps)
