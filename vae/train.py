import numpy as np
import os
import random
import math
import tensorflow as tf
import time
from tensorflow.keras.optimizers import Adam
from convolutional_vae import CVAE
from visualizer import visualize_results, visualize_frames


class TrainMetrics:
    """
    Contains a collection of metrics and allows multiple updates an resets.
    """
    def __init__(self, names):
        self.names = names
        self.metrics = [tf.keras.metrics.Mean() for name in names]

    def update(self, values):
        for i, val in enumerate(values):
            self.metrics[i].update_state(val)

    def reset(self):
        for metric in self.metrics:
            metric.reset_states()


def get_datasets(dataset_dir='images/'):
    """Returns a list of datasets loaded from the given directory.

    Args:
        dataset_dir (str): Directory containing the numpy files -e.g. .npy files.

    Returns:
        A list of numpy.ndarray lazy-loaded from the given directory
    """
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
        indices = np.random.choice(range(dataset.shape[0]), size=n_samples, replace=False)
        res[pos: pos + n_samples] = dataset[indices]
        pos += n_samples

    if pos < sample_size:  # Rounding error in n_samples can lead to collect less than sample_size
        remaining = sample_size - pos
        res[pos: pos + remaining] = datasets[0][0:remaining]  # Fast fix, remaining should be low so it shouldn't matter

    res = res.astype(np.float32)
    if normalize:
        res = res / 255.
    np.random.shuffle(res)
    return res


def train(model, dataset_list, optimizer, epochs=10, sample_per_epoch=50000, resample_freq=10, batch_size=100,
          beta_half_cycle=3, beta_freeze=2, dataset_weights=None, checkpoint_manager=None, tensorboard_dir='log_dir'):
    """Train the given model by switching datasets and randomly sampling batches.

    The training is performed for the given number of epochs as follows:
        1. For each epoch
        2. A sample_per_epoch number of samples is taken from the datasets according to dataset_weights
        3. The model is trained in batches of batch_size
        4. A checkpoint is stored if checkpoint_freq > 0
    During the training process, the beta parameter of the CVAE model is annealed in cycles as in Li et al. (2019)
    https://www.microsoft.com/en-us/research/blog/less-pain-more-gain-a-simple-method-for-vae-training-with-less-of-that-kl-vanishing-agony/

    Args:
        model (CVAE): The CVAE to be trained
        dataset_list (list of ndarray): List of datasets to use
        optimizer (tf.keras.optimizers.Optimizer): Any implementation of the Optimizer class.
        epochs (int): Number of training epochs as defined above. Defaults to 10.
        sample_per_epoch (int): Number of samples to be loaded from datasets in each epoch
        resample_freq (int): Frequency (in epochs) of the resampling.
        batch_size (int): Size of a batch used in ech training step. Defaults to 100 and shouldn't be lower.
        beta_half_cycle (int): Half-cycle duration (in epochs) of the beta annealing. E.g. if beta_cycle = 3 then beta is
            annealed in cycles of 3 epochs in which it is increased and 3 epochs in which it is kept fixed at its
            maximum. If beta_cycle is None then no annealing is performed.
        beta_freeze (int): Number of initial epochs that are performed with beta fixed to 0.
        dataset_weights (list of floats): List of weights used when deciding how many samples to take from each dataset.
            Must sum to 1. Example: if dataset 3 has weight 0.2, then 0.2 * sample_per_epoch will be taken from it.
            If None then uniform weights will be given.
        checkpoint_manager (tf.train.CheckpointManager): Checkpoint manager initialized on this model and optimizer.
            If None then no checkpoints are saved.
        tensorboard_dir (str): Directory for tensorboard output files. If None, Tensorboard is disabled.
    """

    assert sample_per_epoch % batch_size == 0
    assert beta_half_cycle is None or beta_half_cycle > 0

    if tensorboard_dir is not None:
        tf_summary_freq = 1  # Frequency of tensorboard logging (in batches)
        tf_summary_writer = tf.summary.create_file_writer(os.path.join(tensorboard_dir, 'train'))

    if beta_half_cycle is not None:
        half_cycle_steps = int(beta_half_cycle * sample_per_epoch / batch_size)
        print('Annealing every ' + str(beta_half_cycle) + ' epochs which means ' + str(half_cycle_steps) + ' batches')

    steps = 0
    for epoch in range(epochs):
        print('Epoch '+str(epoch + 1)+'/'+str(epochs))

        if epoch % resample_freq == 0:
            samples = get_random_samples(dataset_list, sample_per_epoch, dataset_weights, normalize=True)

        # Beta annealing cycles
        if beta_half_cycle is not None:
            if epoch % (2 * beta_half_cycle) == 0:  # We completed a full cycle
                model.beta.assign(0.)
                increase_beta = True
            elif epoch % beta_half_cycle == 0:  # We completed the first half of the cycle
                increase_beta = False

        if epoch < beta_freeze:
            increase_beta = False
            model.beta.assign(0.)

        metrics = TrainMetrics(names=['train_reconstruction_loss', 'train_kl_loss', 'train_total_loss'])
        bar = tf.keras.utils.Progbar(target=samples.shape[0], stateful_metrics=metrics.names)
        for batch_start in range(0, samples.shape[0], batch_size):
            batch = samples[batch_start: batch_start + batch_size]
            rec_loss, kl_loss, tot_loss = model.compute_apply_gradients(batch, optimizer)
            if increase_beta:
                model.beta.assign(model.beta + model.fixed_beta / half_cycle_steps)

            # Update bar and tensorboard metrics
            metrics.update([rec_loss, kl_loss, rec_loss + model.beta.numpy() * kl_loss])
            bar.add(batch_size, [(name, metric.result()) for name, metric in zip(metrics.names, metrics.metrics)])
            if tensorboard_dir is not None and steps % tf_summary_freq == 0:
                with tf_summary_writer.as_default():
                    for name, metric in zip(metrics.names, metrics.metrics):
                        tf.summary.scalar(name, metric.result(), step=steps)
                    tf.summary.scalar('beta', model.beta.numpy(), step=steps)
            steps += 1

        # Save checkpoint
        if checkpoint_manager is not None:
            checkpoint_manager.save()


if __name__ == "__main__":
    dataset_list = get_datasets()

    cvae = CVAE(img_shape=(64, 64, 3), latent_dim=256, beta=2., scale_losses=True)
    adam = Adam()

    # Checkpoint management for saving and restoring model
    cpk = tf.train.Checkpoint(optimizer=adam, model=cvae)
    cpk_manager = tf.train.CheckpointManager(cpk, directory='checkpoints/', max_to_keep=2)
    status = cpk.restore(cpk_manager.latest_checkpoint)

    # Train the model
    train(model=cvae, dataset_list=dataset_list, optimizer=adam, epochs=50, sample_per_epoch=25600, resample_freq=1,
          batch_size=128, beta_half_cycle=3, beta_freeze=5, dataset_weights=None, checkpoint_manager=cpk_manager)

    # Visualize the encode/decode reconstruction of some random images
    imgs = get_random_samples(datasets=dataset_list, sample_size=20, normalize=True)
    # visualize_results(model=cvae, test_data=imgs)
    visualize_frames(cvae, imgs, fps=1.)