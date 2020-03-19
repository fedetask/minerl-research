import numpy as np
import os
import random
import math
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from convolutional_vae import CVAE


def get_random_dataset(dataset_dir='images/', n_datasets=1):
    dataset_names = [f for f in os.listdir(dataset_dir)]
    return random.sample(dataset_names, n_datasets)


def load_dataset(dataset_name, dataset_dir='images/'):
    path = os.path.join(dataset_dir, dataset_name)
    return np.load(path, mmap_mode='r')


def get_datasets(dataset_dir='images/'):
    dataset_paths = [os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir)]
    datasets = [np.load(path, mmap_mode='r') for path in dataset_paths]
    return datasets


def get_random_samples(datasets, sample_size, dataset_weights):
    """Returns the given number of samples from the datasets. A number of samples is taken from each dataset
    according to its weight.

    Args:
        datasets (list of ndarray): List of ndarrays of which only the first dimension can differ in size.
        sample_size (int): Total number of samples to return
        dataset_weights (list of float): List of weights that sum to 1.

    Returns:
        ndarray of sample_size data points

    """
    res = np.empty(shape=((sample_size, ) + datasets[0].shape[1:]))
    pos = 0
    for i in range(len(datasets)):
        dataset = datasets[i]
        n_samples = math.floor(dataset.shape[0] * dataset_weights[i])  # We want to be sure to not take more samples
        indices = np.random.choice(range(dataset.shape[0]))
        res[pos: pos + n_samples] = dataset[indices]
        pos += n_samples

    if pos < sample_size:  # Rounding error in n_samples led to less samples
        remaining = sample_size - pos
        res[pos: pos + remaining] = datasets[0][0:remaining]  # Fast fix, remaining should be low so it shouldn't matter
    return res


def train(model, dataset_loader, dataset_names=None, dataset_dir='images/', optimizer=None, lr=1e-5, epochs=10,
          batch_size=100, dataset_fractions=0.3, random_datasets=True, dataset_weights=None,
          checkpoint_dir='checkpoints/', checkpoint_freq=50000):
    """Train the given model by switching datasets and randomly sampling batches.

    The training is performed for the given number of epochs as follows:
        1. A given fraction of the dataset is sampled depending on the dataset_fractions parameter
        2. The sampled data is divided in batches of batch_size
        3. The CVAE is trained on the given batches
        4. The next dataset is loaded -in the given order or randomly sampled according to the use_random_datasets param

    Args:
        model (CVAE): The CVAE to be trained
        dataset_loader (callable): A callable with the following parameters in the given order:
                                    dataset_name: name of the npy file to be loaded
                                    dataset_dir: path to the directory containing the npy files. Defaults to 'images/'
                                    Must return a ndarray of shape N x model.img_shape
        dataset_names (list of str): List of dataset names that will be used in training. If None, all datasets in
            dataset_dir will be used.
        dataset_dir (str): Path in which datasets are stored.
        optimizer (tf.keras.optimizers.Optimizer): Any implementation of the Optimizer class. Defaults to Adam()
        lr (float): Learning rate set to Adam optimizer if optimizer is None.
        epochs (int): Number of training epochs as defined above. Defaults to 10.
        batch_size (int): Size of a batch used in ech training step. Defaults to 100 and shouldn't be lower.
        dataset_fractions (float or list): The fraction of data points of a dataset that will be used before switching
            to another dataset. Can be a list with a value for each dataset or a single value that will be used for all.
        random_datasets (bool): Whether to use datasets in the given order or chose a random one when switching.
        dataset_weights (list of floats): List of probabilities used when randomly switching dataset.
        checkpoint_dir (str): Where to save model checkpoints.
        checkpoint_freq (int): Frequency, in batches, of checkpoint saving. If -1 then checkpoint saving is disabled.
    """

    if optimizer is None:
        optimizer = Adam(lr)
    if dataset_weights is None:
        dataset_probs = [1. / len(dataset_names) for name in dataset_names]  # Uniform probability of datasets
    if not isinstance(dataset_fractions, list):
        dataset_frac = [dataset_fractions for dataset in dataset_names]  # Same sampled fraction for all datasets

    for epoch in range(epochs):
        print('Epoch '+str(epoch + 1)+'/'+str(epochs))

        next_dataset_idx = 0
        for i in range(len(dataset_names)):  # Iterate over all datasets
            print('Dataset step '+str(i + 1)+'/'+str(len(dataset_names)))
            # Selecting index of dataset to be used
            if random_datasets:
                dataset_idx = np.random.choice(range(len(dataset_names)), p=dataset_probs)
            else:
                dataset_idx = next_dataset_idx
                next_dataset_idx = (next_dataset_idx + 1) % len(dataset_names)

            # Load dataset
            dataset = dataset_loader(dataset_name=dataset_names[dataset_idx], dataset_dir=dataset_dir)
            print('Loaded dataset ' + str(dataset_idx) + ': ' + str(dataset_names[dataset_idx]))

            # Sample the desired proportion, floor to nearest multiple of batch_size, and scale to [0, 1] float32
            n_sampled = int(batch_size * math.floor(dataset_frac[dataset_idx] * dataset.shape[0] / batch_size))
            sampled_indices = np.random.choice(range(dataset.shape[0]), n_sampled)
            dataset = dataset[sampled_indices].astype(np.float32) / 255.
            print('Sampled subset of shape ' + str(dataset.shape))

            # Train on sampled data
            rec_train_metr = tf.keras.metrics.Mean()
            kl_train_metr = tf.keras.metrics.Mean()
            stateful_metrics = ['train_reconstruction_loss', 'train_kl_loss']
            bar = tf.keras.utils.Progbar(target=dataset.shape[0], stateful_metrics=stateful_metrics)
            for batch_start in range(0, dataset.shape[0], batch_size):
                batch = dataset[batch_start: batch_start + batch_size]
                rec_loss, kl_loss, tot_loss = model.compute_apply_gradients(batch, optimizer)
                rec_train_metr.update_state(rec_loss)
                kl_train_metr.update_state(kl_loss)
                bar.add(batch_size, [('train_reconstruction_loss', rec_train_metr.result()),
                                     ('train_kl_loss', kl_train_metr.result())])


if __name__ == "__main__":
    dataset_dir = 'images/'
    datasets = [f for f in os.listdir(dataset_dir)]
    cvae = CVAE(img_shape=(64, 64, 3), latent_dim=128, beta=2.)
    train(model=cvae, dataset_loader=load_dataset, dataset_names=datasets, dataset_dir=dataset_dir, optimizer=None,
          lr=1e-5, epochs=10, batch_size=100, dataset_fractions=0.3, random_datasets=True, dataset_weights=None,
          checkpoint_dir='checkpoints/', checkpoint_freq=50000)
