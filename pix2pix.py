import numpy as np
from keras.models import Model
from keras.layers import Conv2D, BatchNormalization, LeakyReLU, Activation, Conv2DTranspose, Dropout
from keras.layers import Input, Concatenate
from keras.initializers import RandomNormal
from keras.optimizers import Adam
from matplotlib import pyplot as plt
from data import load_batch
from model import MODEL_DIR


class Pix2Pix:
    kernel_init = RandomNormal(0.0, 0.02)
    gamma_init = RandomNormal(1.0, 0.02)
    optimizer = Adam(0.0002, 0.5)

    @classmethod
    def batch_norm(cls):
        def func(t):
            """
            Layer of batch normalization.
            :param t: tensor, input tensor
            :return b: tensor, tensor after batch normalization
            """
            b = BatchNormalization(momentum=0.9, axis=-1, epsilon=1.01e-5, gamma_initializer=cls.gamma_init)(t)
            return b
        return func

    @classmethod
    def conv2d(cls, filters, kernel_size=4, strides=2, bn=True, alpha=0.2):
        """
        Layers for 2d convolution.
        :param filters: int, number of filters
        :param kernel_size: int or tuple, kernel size
        :param strides: int or tuple, strides
        :param bn: bool, whether batch normalize
        :param alpha: float, negative slope coefficient
        :return func: function, layers for 2d convolution
        """
        def func(t):
            """
            :param t: tensor, input tensor
            :return b: tensor, output tensor
            """
            c = Conv2D(filters, kernel_size=kernel_size, strides=strides, use_bias=not bn,
                       kernel_initializer=cls.kernel_init, padding="same")(t)
            if bn:
                c = cls.batch_norm()(c)
            if alpha > 0:
                c = LeakyReLU(alpha=0.2)(c)
            return c
        return func

    @classmethod
    def deconv2d(cls, filters, kernel_size=4, strides=2, bn=True, dropout_rate=0.0):
        """
        Layers for 2d deconvolution.
        :param filters: int, number of filters
        :param kernel_size: int or tuple, kernel size
        :param strides: int or tuple, strides
        :param bn: bool, whether batch normalize
        :param dropout_rate: float, dropout rate
        :return func: function, layers for 2d deconvolution
        """
        def func(t):
            """
            :param t: tensor, input tensor
            :return b: tensor, output tensor
            """
            d = Activation("relu")(t)
            d = Conv2DTranspose(filters, kernel_size=kernel_size, strides=strides, use_bias=not bn,
                                kernel_initializer=cls.kernel_init, padding="same")(d)
            if bn:
                d = cls.batch_norm()(d)
            if dropout_rate > 0:
                d = Dropout(0.5)(d)
            return d
        return func

    @classmethod
    def unet_deconv2d(cls, filters, kernel_size=4, strides=2, bn=True, dropout_rate=0.0):
        """
        Layers for 2d deconvolution.
        :param filters: int, number of filters
        :param kernel_size: int or tuple, kernel size
        :param strides: int or tuple, strides
        :param bn: bool, whether batch normalize
        :param dropout_rate: float, dropout rate
        :return func: function, layers for 2d deconvolution
        """
        def func(t1, t2):
            """
            :param t1: tensor, input tensor from last deconv2d layer
            :param t2: tensor, input tensor from skip conv2d layer, skip connection
            :return b: tensor, output tensor
            """
            d = Concatenate(axis=-1)([t1, t2])
            d = cls.deconv2d(filters, kernel_size, strides, bn, dropout_rate)(d)
            return d
        return func

    @classmethod
    def build_generator(cls):
        """
        Build generator.
        :return generator: Model, given (n, 256, 256, 3) ndarray of normalized conditional images,
            return (n, 256, 256, 3) ndarray of normalized fake (generated) images
        """
        cond = Input((256, 256, 3))
        # conv
        c1 = cls.conv2d(64, 4, 2, bn=False)(cond)
        c2 = cls.conv2d(128, 4, 2)(c1)
        c3 = cls.conv2d(256, 4, 2)(c2)
        c4 = cls.conv2d(512, 4, 2)(c3)
        c5 = cls.conv2d(512, 4, 2)(c4)
        c6 = cls.conv2d(512, 4, 2)(c5)
        c7 = cls.conv2d(512, 4, 2)(c6)
        c8 = cls.conv2d(512, 4, 2, bn=False)(c7)
        # deconv
        d8 = cls.deconv2d(512, 4, 2, dropout_rate=0.5)(c8)
        d7 = cls.unet_deconv2d(512, 4, 2, dropout_rate=0.5)(d8, c7)
        d6 = cls.unet_deconv2d(512, 4, 2, dropout_rate=0.5)(d7, c6)
        d5 = cls.unet_deconv2d(512, 4, 2)(d6, c5)
        d4 = cls.unet_deconv2d(256, 4, 2)(d5, c4)
        d3 = cls.unet_deconv2d(128, 4, 2)(d4, c3)
        d2 = cls.unet_deconv2d(64, 4, 2)(d3, c2)
        d1 = cls.unet_deconv2d(3, 4, 2, bn=False)(d2, c1)
        fake = Activation("tanh")(d1)
        generator = Model(cond, fake)
        return generator

    @classmethod
    def build_discriminator(cls):
        """
        Build generator.
        :return discriminator: Model, given normalized real or fake images, return normalized fake (generated) images
        """
        real_or_fake = Input((256, 256, 3))
        cond = Input((256, 256, 3))
        v = Concatenate(axis=-1)([real_or_fake, cond])
        v = cls.conv2d(64, 4, 2, bn=False)(v)
        v = cls.conv2d(128, 4, 2)(v)
        v = cls.conv2d(256, 4, 2)(v)
        v = cls.conv2d(512, 4, 1)(v)
        v = cls.conv2d(1, 4, 1, bn=False)(v)
        valid = Activation("sigmoid")(v)
        discriminator = Model([real_or_fake, cond], valid)
        return discriminator

    def __init__(self):
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss="binary_crossentropy", optimizer=self.optimizer, metrics=["acc"])
        self.generator = self.build_generator()
        real = Input((256, 256, 3))
        cond = Input((256, 256, 3))
        fake = self.generator(cond)
        self.discriminator.trainable = False
        valid = self.discriminator([fake, cond])
        self.gan = Model([real, cond], [valid, fake])
        self.gan.compile(loss=["binary_crossentropy", "mae"], loss_weights=[1, 100], optimizer=self.optimizer)

    def train(self, dataset, batch_size=1, epochs=100, save=True):
        """
        Train the gan.
        :param dataset: str, name of dataset, such as "facade"
        :param batch_size: int, batch size
        :param epochs: int, training epochs
        :param save: bool, whether save model after training
        """
        for ei in range(epochs):
            if ei % 10 == 0 and ei > 0:
                h5_file = "%s/%s_%d.h5" % (MODEL_DIR, dataset, ei)
                self.gan.save_weights(h5_file)
            is_valid = np.ones((batch_size, 32, 32, 1))
            is_invalid = np.zeros((batch_size, 32, 32, 1))
            for bi, (real, cond) in enumerate(load_batch(dataset, "train", 256, batch_size)):
                fake = self.generator.predict(cond)
                # train discriminator
                d_bc_real, d_acc_real = self.discriminator.train_on_batch([real, cond], is_valid)
                d_bc_fake, d_acc_fake = self.discriminator.train_on_batch([fake, cond], is_invalid)
                d_bc = 0.5 * (d_bc_real + d_bc_fake)
                d_acc = 0.5 * (d_acc_real + d_acc_fake)
                # train generator
                g_metrics = self.gan.train_on_batch([real, cond], [is_valid, real])
                g_loss, g_bc, g_mae = g_metrics
                print("epoch-%d batch-%d | d_bc:%.4f d_acc:%.4f | g_loss: %.4f g_bc:%.4f g_mae:%.4f" %
                      (ei + 1, bi + 1, d_bc, d_acc, g_loss, g_bc, g_mae))
        h5_file = "%s/%s_%d.h5" % (MODEL_DIR, dataset, epochs)
        self.gan.save_weights(h5_file)
        if save:
            h5_file = "%s/%s.h5" % (MODEL_DIR, dataset)
            self.gan.save_weights(h5_file)

    def load_model(self, model):
        """
        Load model.
        :param model: str, model name, such as "facade"
        """
        h5_file = "%s/%s.h5" % (MODEL_DIR, model)
        self.gan.load_weights(h5_file)


def train():
    p2p = Pix2Pix()
    p2p.train("facade", epochs=200, save=True)


def test():
    def imshow(ax, normalize_image, title):
        img = ((normalize_image + 1) * 127.5).astype(np.uint8)[0]
        ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        ax.set_title(title)

    epochs = [50, 100, 150, 200]
    models = ["facade_%d" % epoch for epoch in epochs]
    gans = []
    for model in models:
        p2p = Pix2Pix()
        p2p.load_model(model)
        gans.append(p2p)
    for real, cond in load_batch("facade", "test", 256, 1):
        ax = plt.subplot(231)
        imshow(ax, cond, "conditional")
        ax = plt.subplot(232)
        imshow(ax, real, "real")
        for gi, (epoch, gan) in enumerate(zip(epochs, gans)):
            fake = gan.generator.predict(cond)
            ax = plt.subplot(2, 3, gi + 3)
            imshow(ax, fake, "epoch %d" % epoch)
        plt.show()


if __name__ == "__main__":
    # train()
    test()
