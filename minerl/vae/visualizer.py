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
    decoded = model.decode(encoded, apply_sigmoid=True)

    f, axes = plt.subplots(int(test_data.shape[0] / images_per_row) * 2, images_per_row)
    ax_idx = 0
    for img_idx in range(test_data.shape[0]):
        if img_idx % images_per_row == 0 and img_idx > 0:
            ax_idx += 2
        axes[ax_idx, img_idx % images_per_row].imshow(test_data[img_idx])
        axes[ax_idx + 1, img_idx % images_per_row].imshow(decoded[img_idx].numpy())
    plt.show()
