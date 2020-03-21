# Convolutional Variational Autoencoder

This directory contains an implementation of a Convolutional Variational Autoencoer that I will
use to encode the large dataset of Minerl images into a smaller latent space, improving the
performances of the RL agent. I use a VAE instead of a more simple autoencoder because I believe
the disentanglement of the latent space will be essential for the Reinforcement Learning agent.

## The Convolutional Model
The model is contained in `convolutional_vae.py`. A few things to note about it:
 - The model works with any image of the shape (n, n, k) where n is a power of 2 (32, 64,
   128 ...) and k the number of channels, but can easily be extended to any shape by tuning
   the layers.
 - The first convolutional layer does not have an activation function. This is purely
   experimental.
 - The reconstruction loss is a Mean Squared Error, the KL loss is computed with respect to an
   isotropic Gaussian vector. The losses can be used as they are or scaled by setting the
   `scale_losses` parameter to True. In this case, the reconstruction loss is then divided by
   the number of pixels in the image, while the KL loss is divided by the size of the latent
   space. The rationale behind this is that the order of magnitude of the reconstruction loss
   depends on the dimensionality of the image while the KL loss depends only on the size of the
   latent space. Thus, for big images the reconstruction loss would weight orders of magnitude
   more than the KL divergence, while the opposite happens for small images. This is, however,
   in no way a definitive technique, and I'll research more about it.
 - The `beta` parameter of [beta-VAE: Learning Basic Visual Concepts with a Constrained Variational
   Framework](https://openreview.net/forum?id=Sy2fzU9gl) is implemented and can be set on
   instantiation. The `CVAE.fixed_beta` is a constant that keeps track of the desired `beta`
   value, and the `CVAE.beta` is the actual value used and can be modified during training
   (see below).

## The training
The training is performed in `train.py`. It uses Tensorflow 2.1 `GradientTape` in order to have
full control over each training step.

### Loading the data
The training is tought for a large dataset -possibly saved in multiple numpy files- that does
not fit into RAM. The main elements of data loading are the followings:
 - `get_datasets(dataset_dir='images/')` Returns a list of **lazy-loaded** numpy arrays from 
   all the files in the given directory. These array can be used as any other numpy array, but
   their content is read from disk only when accessed.
 - `get_random_samples(datasets, sample_size, dataset_weights=None, normalize=True)`
   Returns a random sample taken from all the given datasets, with optional weights.
   For example, if `sample_size` is `10000` and we have `4` datasets, `2500` samples will be
   taken from each dataset, shuffled, scaled in `[0, 1]` if `normalize` is `True`, and returned.
   If `dataset_weights` is, for example, `[0.4, 0.2, 0.2, 0.2]` then `4000` samples will be
   taken from the first dataset and `2000` from the others.
   
### The training loop
The `train()` function performs the training loop. It implements the [Cyclical Annealing
Schedule: A Simple Approach to Mitigate KL Vanishing](https://www.microsoft.com/en-us/research/publication/cyclical-annealing-schedule-a-simple-approach-to-mitigate-kl-vanishing/),
where the Cyclical Annealing Schedule: A Simple Approach to Mitigate KL Vanishing
`beta` parameter is cyclically annealed. The training loop works as follows:
1. A super-batch of images is sampled from the datasets.
2. A smaller batch of size `batch_size` is taken from the super batch.
3. A gradient step is performed with `CVAE.compute_apply_gradients()`.
4. Repeat 3 for all batches in the super batch.
5. Repeat from 1.

During this process, `beta` is kept to `0` for the first `batch_freeze` epochs. Then, it is
cyclically increased from `0.` to `CVAE.fixed_beta` in `half_beta_cycle` epochs, and kept at
`CVAE.fixed_beta` for other `half_beta_cycle` epochs.