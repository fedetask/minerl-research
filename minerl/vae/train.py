import numpy as np
import os
import random
import math
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from convolutional_vae import CVAE
from visualizer import visualize_results
from matplotlib import pyplot as plt


def get_random_dataset(dataset_dir='images/', n_datasets=1):
    dataset_names = [f for f in os.listdir(dataset_dir)]
    return random.sample(dataset_names, n_datasets)


def load_dataset(dataset_name, dataset_dir='images/'):
    path = os.path.join(dataset_dir, dataset_name)
    return np.load(path, mmap_mode='r')


def get_datasets(dataset_dir='images/'):
    dataset_paths = [os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir)]
    dataset_list = [np.load(path, mmap_mode='r') for path in dataset_paths]
    return dataset_list


def get_random_samples(datasets, sample_size, dataset_weights=None, normalize=True):
    """Returns the given number of samples from the datasets. A number of samples is taken from each dataset
    according to its weight.

    Args:
        datasets (list of ndarray): List of ndarrays of which only the first dimension can differ in size.
        sample_size (int): Total number of samples to return
        dataset_weights (list of float): List of weights that sum to 1. If None then use uniform
        normalize (bool): If true images are rescaled into [0., 1.] by dividing by 255.

    Returns:
        ndarray of sample_size data points, as np.float32, scaled in [0., 1.]

    """
    if dataset_weights is None:
        dataset_weights = [1. / len(datasets) for dataset in datasets]
    res = np.empty(shape=((sample_size, ) + datasets[0].shape[1:]))
    pos = 0
    for i in range(len(datasets)):
        dataset = datasets[i]
        n_samples = math.floor(sample_size * dataset_weights[i])  # We want to be sure to not take more samples
        indices = np.random.choice(range(dataset.shape[0]), n_samples)
        res[pos: pos + n_samples] = dataset[indices]
        pos += n_samples

    if pos < sample_size:  # Rounding error in n_samples led to less samples
        remaining = sample_size - pos
        res[pos: pos + remaining] = datasets[0][0:remaining]  # Fast fix, remaining should be low so it shouldn't matter
    return res.astype(np.float32) / 255.


def train(model, dataset_list, optimizer, epochs=10, sample_per_epoch=50000, resample_freq=20, batch_size=100,
          dataset_weights=None, checkpoint_dir='checkpoints/', checkpoint_freq=-1, checkpoint_prefix='cpk'):
    """Train the given model by switching datasets and randomly sampling batches.

    The training is performed for the given number of epochs as follows:
        1. For each epoch
        2. A sample_per_epoch number of samples is taken from the datasets according to dataset_weights
        3. The model is trained in batches of batch_size
        4. A checkpoint is stored if checkpoint_freq > 0

    Args:
        model (CVAE): The CVAE to be trained
        dataset_list (list of ndarray): List of datasets to use
        optimizer (tf.keras.optimizers.Optimizer): Any implementation of the Optimizer class.
        epochs (int): Number of training epochs as defined above. Defaults to 10.
        sample_per_epoch (int): Number of samples to be loaded from datasets in each epoch
        resample_freq (int): Frequency (in epochs) of the resampling.
        batch_size (int): Size of a batch used in ech training step. Defaults to 100 and shouldn't be lower.
        dataset_weights (list of floats): List of weights used when deciding how many samples to take from each dataset.
            Must sum to 1. Example: if dataset 3 has weight 0.2, then 0.2 * sample_per_epoch will be taken from it.
            If None then uniform weights will be given.
        checkpoint_dir (str): Where to save model checkpoints.
        checkpoint_freq (int): Frequency, in epochs, of checkpoint saving. If <= 0 then checkpoint saving is disabled.
        checkpoint_prefix (str): Prefix of the checkpoint file names.
    """

    if checkpoint_freq > 0:
        checkpoint_saver = tf.train.Checkpoint(optimizer=optimizer, model=model)

    for epoch in range(epochs):
        print('Epoch '+str(epoch + 1)+'/'+str(epochs))

        if epoch % resample_freq == 0:
            samples = get_random_samples(dataset_list, sample_per_epoch, dataset_weights, normalize=True)

        rec_train_metr = tf.keras.metrics.Mean()
        kl_train_metr = tf.keras.metrics.Mean()
        tot_train_metr = tf.keras.metrics.Mean()
        stateful_metrics = ['train_reconstruction_loss', 'train_kl_loss','train_total_loss']
        bar = tf.keras.utils.Progbar(target=samples.shape[0], stateful_metrics=stateful_metrics)
        for batch_start in range(0, samples.shape[0], batch_size):
            batch = samples[batch_start: batch_start + batch_size]
            rec_loss, kl_loss, tot_loss = model.compute_apply_gradients(batch, optimizer)
            rec_train_metr.update_state(rec_loss)
            kl_train_metr.update_state(kl_loss)
            tot_train_metr.update_state(rec_loss + kl_loss)
            bar.add(batch_size, [('train_reconstruction_loss', rec_train_metr.result()),
                                 ('train_kl_loss', kl_train_metr.result()),
                                 ('train_total_loss', tot_train_metr.result())])

        # Save checkpoint
        if epoch % checkpoint_freq == 0:
            checkpoint_saver.save(file_prefix=os.path.join(checkpoint_dir, checkpoint_prefix))


if __name__ == "__main__":
    datasets = get_datasets()

    cvae = CVAE(img_shape=(64, 64, 3), latent_dim=128, beta=2.)
    adam = Adam()

    checkpoint = tf.train.Checkpoint(optimizer=adam, model=cvae)
    status = checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir='checkpoints/'))

    train(model=cvae, dataset_list=datasets, optimizer=adam, epochs=10, sample_per_epoch=50000, resample_freq=10,
          batch_size=100,dataset_weights=None, checkpoint_dir='checkpoints/', checkpoint_freq=1,
          checkpoint_prefix='latent_128')

    #imgs = get_random_samples(datasets=datasets, sample_size=20, normalize=True)
    # visualize_results(model=cvae, test_data=imgs)
