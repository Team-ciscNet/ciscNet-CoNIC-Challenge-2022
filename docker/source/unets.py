import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import tanh


def build_unet(act_fun, pool_method, normalization, device, num_gpus, ch_in=1, ch_out=1, filters=(64, 1024)):
    """ Build U-net architecture.

    :param act_fun: 'relu', 'leakyrelu', 'elu', 'mish' (not in the output layer).
        :type act_fun: str
    :param pool_method: 'max' (maximum pooling), 'conv' (convolution with stride 2).
        :type pool_method: str
    :param normalization: 'bn' (batch normalization), 'gn' (group norm., 8 groups), 'in' (instance norm.).
        :type normalization: str
    :param device: 'cuda' or 'cpu'.
        :type device: device
    :param num_gpus: Number of GPUs to use.
        :type num_gpus: int
    :param ch_in: Number of channels of the input image.
        :type ch_in: int
    :param ch_out: Number of channels of the prediction.
        :type ch_out: int
    :param filters: depth of the encoder (and decoder reversed) and number of feature maps used in a block.
        :type filters: list
    :return: model
    """

    model = MCUNet(ch_in=ch_in,
                   ch_out=ch_out,
                   pool_method=pool_method,
                   filters=filters,
                   act_fun=act_fun,
                   normalization=normalization)

    # Use multiple GPUs if available
    if num_gpus > 1:
        model = nn.DataParallel(model)

    # Move model to used device (GPU or CPU)
    model = model.to(device)

    return model


class Mish(nn.Module):
    """ Mish activation function. """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x * (tanh(F.softplus(x)))
        return x


class ConvBlock(nn.Module):
    """ Basic convolutional block of a U-net. """

    def __init__(self, ch_in, ch_out, act_fun, normalization):
        """

        :param ch_in: Number of channels of the input image.
            :type ch_in: int
        :param ch_out: Number of channels of the prediction.
            :type ch_out: int
        :param act_fun: 'relu', 'leakyrelu', 'elu', 'mish' (not in the output layer)
            :type act_fun: str
        :param normalization: 'bn' (batch normalization), 'gn' (group norm., 8 groups), 'in' (instance norm.)
            :type normalization: str
        """

        super().__init__()
        self.conv = list()

        # 1st convolution
        self.conv.append(nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True))

        # 1st activation function
        if act_fun == 'relu':
            self.conv.append(nn.ReLU(inplace=True))
        elif act_fun == 'leakyrelu':
            self.conv.append(nn.LeakyReLU(inplace=True))
        elif act_fun == 'elu':
            self.conv.append(nn.ELU(inplace=True))
        elif act_fun == 'mish':
            self.conv.append(Mish())
        else:
            raise Exception('Unsupported activation function: {}'.format(act_fun))

        # 1st normalization
        if normalization == 'bn':
            self.conv.append(nn.BatchNorm2d(ch_out))
        elif normalization == 'gn':
            self.conv.append(nn.GroupNorm(num_groups=8, num_channels=ch_out))
        elif normalization == 'in':
            self.conv.append(nn.InstanceNorm2d(num_features=ch_out))
        else:
            raise Exception('Unsupported normalization: {}'.format(normalization))

        # 2nd convolution
        self.conv.append(nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True))

        # 2nd activation function
        if act_fun == 'relu':
            self.conv.append(nn.ReLU(inplace=True))
        elif act_fun == 'leakyrelu':
            self.conv.append(nn.LeakyReLU(inplace=True))
        elif act_fun == 'elu':
            self.conv.append(nn.ELU(inplace=True))
        elif act_fun == 'mish':
            self.conv.append(Mish())
        else:
            raise Exception('Unsupported activation function: {}'.format(act_fun))

        # 2nd normalization
        if normalization == 'bn':
            self.conv.append(nn.BatchNorm2d(ch_out))
        elif normalization == 'gn':
            self.conv.append(nn.GroupNorm(num_groups=8, num_channels=ch_out))
        elif normalization == 'in':
            self.conv.append(nn.InstanceNorm2d(num_features=ch_out))
        else:
            raise Exception('Unsupported normalization: {}'.format(normalization))

        self.conv = nn.Sequential(*self.conv)

    def forward(self, x):
        """

        :param x: Block input (image or feature maps).
            :type x:
        :return: Block output (feature maps).
        """
        for i in range(len(self.conv)):
            x = self.conv[i](x)

        return x


class ConvPool(nn.Module):

    def __init__(self, ch_in, act_fun, normalization):
        """

        :param ch_in: Number of channels of the input image.
            :type ch_in: int
        :param act_fun: 'relu', 'leakyrelu', 'elu', 'mish' (not in the output layer).
            :type act_fun: str
        :param normalization: 'bn' (batch normalization), 'gn' (group norm., 8 groups), 'in' (instance norm.).
            :type normalization: str
        """

        super().__init__()
        self.conv_pool = list()

        self.conv_pool.append(nn.Conv2d(ch_in, ch_in, kernel_size=3, stride=2, padding=1, bias=True))

        if act_fun == 'relu':
            self.conv_pool.append(nn.ReLU(inplace=True))
        elif act_fun == 'leakyrelu':
            self.conv_pool.append(nn.LeakyReLU(inplace=True))
        elif act_fun == 'elu':
            self.conv_pool.append(nn.ELU(inplace=True))
        elif act_fun == 'mish':
            self.conv_pool.append(Mish())
        else:
            raise Exception('Unsupported activation function: {}'.format(act_fun))

        if normalization == 'bn':
            self.conv_pool.append(nn.BatchNorm2d(ch_in))
        elif normalization == 'gn':
            self.conv_pool.append(nn.GroupNorm(num_groups=8, num_channels=ch_in))
        elif normalization == 'in':
            self.conv_pool.append(nn.InstanceNorm2d(num_features=ch_in))
        else:
            raise Exception('Unsupported normalization: {}'.format(normalization))

        self.conv_pool = nn.Sequential(*self.conv_pool)

    def forward(self, x):
        """

        :param x: Block input (image or feature maps).
            :type x:
        :return: Block output (feature maps).
        """
        for i in range(len(self.conv_pool)):
            x = self.conv_pool[i](x)

        return x


class TranspConvBlock(nn.Module):
    """ Upsampling block of a unet (with transposed convolutions). """

    def __init__(self, ch_in, ch_out, normalization):
        """

        :param ch_in: Number of channels of the input image.
            :type ch_in: int
        :param ch_out: Number of channels of the prediction.
            :type ch_out: int
        :param normalization: 'bn' (batch normalization), 'gn' (group norm., 8 groups), 'in' (instance norm.).
            :type normalization: str
        """
        super().__init__()

        self.up = nn.Sequential(nn.ConvTranspose2d(ch_in, ch_out, kernel_size=2, stride=2))
        if normalization == 'bn':
            self.norm = nn.BatchNorm2d(ch_out)
        elif normalization == 'gn':
            self.norm = nn.GroupNorm(num_groups=8, num_channels=ch_out)
        elif normalization == 'in':
            self.norm = nn.InstanceNorm2d(num_features=ch_out)
        else:
            raise Exception('Unsupported normalization: {}'.format(normalization))

    def forward(self, x):
        """

        :param x: Block input (image or feature maps).
            :type x:
        :return: Block output (upsampled feature maps).
        """
        x = self.up(x)
        x = self.norm(x)

        return x


class MCUNet(nn.Module):
    """ U-net for combined segmentation and classification (multi-class unet). """

    def __init__(self, ch_in=1, ch_out=1, pool_method='conv', act_fun='relu', normalization='bn', filters=(64, 1024)):
        """

        :param ch_in: Number of channels of the input image.
            :type ch_in: int
        :param ch_out: Number of channels of the prediction.
            :type ch_out: int
        :param pool_method: 'max' (maximum pooling), 'conv' (convolution with stride 2).
            :type pool_method: str
        :param act_fun: 'relu', 'leakyrelu', 'elu', 'mish' (not in the output layer).
            :type act_fun: str
        :param normalization: 'bn' (batch normalization), 'gn' (group norm., 8 groups), 'in' (instance norm.).
            :type normalization: str
        :param filters: depth of the encoder (and decoder reversed) and number of feature maps used in a block.
            :type filters: list
        """

        super().__init__()

        self.ch_in = ch_in
        self.filters = filters
        self.pool_method = pool_method

        # Encoder
        self.encoderConv = nn.ModuleList()

        if self.pool_method == 'max':
            self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)
        elif self.pool_method == 'conv':
            self.pooling = nn.ModuleList()

        # First encoder block
        n_featuremaps = filters[0]
        self.encoderConv.append(ConvBlock(ch_in=self.ch_in,
                                          ch_out=n_featuremaps,
                                          act_fun=act_fun,
                                          normalization=normalization))
        if self.pool_method == 'conv':
            self.pooling.append(ConvPool(ch_in=n_featuremaps, act_fun=act_fun, normalization=normalization))

        # Remaining encoder blocks
        while n_featuremaps < filters[1]:

            self.encoderConv.append(ConvBlock(ch_in=n_featuremaps,
                                              ch_out=(n_featuremaps*2),
                                              act_fun=act_fun,
                                              normalization=normalization))

            if n_featuremaps * 2 < filters[1] and self.pool_method == 'conv':
                self.pooling.append(ConvPool(ch_in=n_featuremaps*2, act_fun=act_fun, normalization=normalization))

            n_featuremaps *= 2

        # Decoder 1 (borders, seeds) and Decoder 2 (cells)
        self.decoderUpconv = nn.ModuleList()
        self.decoderConv = nn.ModuleList()

        while n_featuremaps > filters[0]:
            self.decoderUpconv.append(TranspConvBlock(ch_in=n_featuremaps,
                                                      ch_out=(n_featuremaps // 2),
                                                      normalization=normalization))
            self.decoderConv.append(ConvBlock(ch_in=n_featuremaps,
                                              ch_out=(n_featuremaps // 2),
                                              act_fun=act_fun,
                                              normalization=normalization))
            n_featuremaps //= 2

        # Last 1x1 convolutions (2nd path has always 1 channel: binary or dist)
        self.decoderConv.append(nn.Conv2d(n_featuremaps, ch_out, kernel_size=1, stride=1, padding=0))

    def forward(self, x):
        """

        :param x: Model input.
            :type x:
        :return: Model output / prediction.
        """

        x_temp = list()

        # Encoder
        for i in range(len(self.encoderConv) - 1):
            x = self.encoderConv[i](x)
            x_temp.append(x)
            if self.pool_method == 'max':
                x = self.pooling(x)
            elif self.pool_method == 'conv':
                x = self.pooling[i](x)
        x = self.encoderConv[-1](x)

        # Intermediate results for concatenation
        x_temp = list(reversed(x_temp))

        # Decoder
        for i in range(len(self.decoderConv) - 1):
            x = self.decoderUpconv[i](x)
            x = torch.cat([x, x_temp[i]], 1)
            x = self.decoderConv[i](x)
        x = self.decoderConv[-1](x)

        # Sum everything up
        x_all = torch.sum(x, 1)  # sum over channel dimension
        x = torch.cat((x_all[:, None, ...], x), dim=1)  # get channel dimension back

        return x
