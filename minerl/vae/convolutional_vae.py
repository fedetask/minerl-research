from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Flatten, Conv2DTranspose, Dense, Reshape


class CVAE(tf.keras.Model):

    def __init__(self, img_shape, latent_dim, beta=1.):
        """Create the Convolutional Variational Autoencoder model.

        The model consists of 4 convolutional layers, each one downscaling the image by a factor of 2, the encoded layer
        and 4 transpose convolutional layers that upscale the decoded image to the right shape. Input/output shape
        consistency is ensured as long as it is n x n x k with n power of 2

        Args:
            img_shape (tuple): The shape of input images (height, width, channels)
            latent_dim (int): Size of encoding
            beta (float, optional):

        """
        super(CVAE, self).__init__()
        self.img_shape = img_shape
        self.latent_dim = tf.constant(latent_dim)
        self.fixed_beta = tf.constant(float(beta))  # Keep the beta value of the model
        self.beta = tf.Variable(float(beta), trainable=False)  # Beta value used (can be modified in annealing)

        encoder_input = Input(shape=self.img_shape)
        x = Conv2D(filters=64, kernel_size=5, activation='relu', padding='same')(encoder_input)
        x = Conv2D(filters=128, kernel_size=3, strides=2, activation='relu', padding='same')(x)
        x = Conv2D(filters=256, kernel_size=3, strides=2, activation='relu', padding='same')(x)
        x = Conv2D(filters=512, kernel_size=3, strides=2, activation='relu', padding='same')(x)
        conv_shape = x.shape
        x = Flatten()(x)
        x = Dense(latent_dim * 2)(x)
        self.inference_net = tf.keras.Model(encoder_input, x)
        self.inference_net.summary()

        decoder_input = Input(shape=(latent_dim,))
        x = Dense(units=conv_shape[1] * conv_shape[2] * conv_shape[3])(decoder_input)
        x = Reshape(target_shape=(conv_shape[1], conv_shape[2], conv_shape[3]))(x)
        x = Conv2DTranspose(filters=512, kernel_size=3, strides=2, padding='same', activation='relu')(x)
        x = Conv2DTranspose(filters=256, kernel_size=3, strides=2, padding='same', activation='relu')(x)
        x = Conv2DTranspose(filters=128, kernel_size=3, strides=2, padding='same', activation='relu')(x)
        x = Conv2DTranspose(filters=self.img_shape[2], kernel_size=3, padding='same', activation='sigmoid')(x)
        self.generative_net = tf.keras.Model(decoder_input, x)
        self.generative_net.summary()

    def encode(self, x):
        """Encode the given batch of images in latent space.

        Args:
            x (ndarray): Batch of images of shape N x `self.img_shape`

        Returns:
            mean, logvar: N Tensors with mean and log variance of the latent space distribution of the given image batch

        """
        mean, logvar = tf.split(self.inference_net(x), num_or_size_splits=2, axis=1)
        return mean, logvar

    def reparametrize(self, mean, logvar):
        """Sample from a normal distribution with the given means and log variances using the reparametrization trick

        Args:
            mean (Tensor): N mean vectors
            logvar (Tensor): N log-variance vectors

        Returns:
            N random vectors (as Tensors) sampled from a normal distribution with the given means and variances

        """
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean

    def decode(self, z, apply_sigmoid=False):
        """Decode the given vectors in latent space into an image of shape `self.img_shape`

        Args:
            z (Tensor): N latent vectors (as Tensors)
            apply_sigmoid (int, optional): Whether to apply a sigmoid activation on the decoded images

        Returns:
            N images decoded from the given latent space representations. Pixel values are ensured to be in [0, 1] only
            if `apply_sigmoid` is True

        """
        logits = self.generative_net(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs

        return logits

    def compute_loss(self, x):
        """Compute the VAE loss (reconstruction + KL divergence) for the given batch of images

        For a batch of N images, the loss is computed as reconstruction_loss + KL_loss, where
        reconstruction_loss: Mean (over the N images) of the sum squared errors between images and their reconstructions
        KL_loss: KL divergence between the encoder predicted distribution and N(0, 1)

        Args:
            x (Tensor): A batch of N images with shape `model.img_shape`

        Returns:
            Loss of the batch as described above

        """
        mean, logvar = self.encode(x)
        z = self.reparametrize(mean, logvar)
        decoded = self.decode(z)

        npixesl = float(self.img_shape[0] * self.img_shape[1] * self.img_shape[2])
        reconstruction_loss = tf.reduce_sum(tf.square(x - decoded), axis=[1, 2, 3]) / npixesl
        kl_loss = -0.5 * tf.reduce_sum(1 + logvar - tf.pow(mean, 2) - tf.exp(logvar), axis=1) / float(self.latent_dim)
        loss = tf.reduce_mean(reconstruction_loss + self.beta * kl_loss)
        return reconstruction_loss, kl_loss, loss

    def compute_apply_gradients(self, x, opt):
        """Compute the gradients of the model variables with respect to the loss batch and update the model

        Args:
            x (Tensor): Batch of N images with shape `model.img_shape`
            opt (tf.keras.optimizers.Optimizer): Keras Optimizer used for applying gradients

        """
        with tf.GradientTape() as tape:
            reconstruction_loss, kl_loss, loss = self.compute_loss(x)
        gradients = tape.gradient(loss, self.trainable_variables)
        opt.apply_gradients(zip(gradients, self.trainable_variables))
        return reconstruction_loss, kl_loss, loss
