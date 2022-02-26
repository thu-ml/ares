import torch
import torch.nn as nn
import math


class VGG(nn.Module):
    def __init__(self, features, last_conv_channels_len, linear_layer, num_classes=10):
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((2, 2))
        self.classifier = nn.Sequential(
            linear_layer(last_conv_channels_len * 2 * 2, 256),
            nn.ReLU(True),
            linear_layer(256, 256),
            nn.ReLU(True),
            linear_layer(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def initialize_weights(model, init_type):
    print(f"Initializing model with {init_type}")
    assert init_type in ["kaiming_normal", "kaiming_uniform", "signed_const"]
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if init_type == "signed_const":
                n = math.sqrt(
                    2.0 / (m.kernel_size[0] * m.kernel_size[1] * m.in_channels)
                )
                m.weight.data = m.weight.data.sign() * n
            elif init_type == "kaiming_uniform":
                nn.init.kaiming_uniform_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.01)
            m.bias.data.zero_()
            if init_type == "signed_const":
                n = math.sqrt(2.0 / m.in_features)
                m.weight.data = m.weight.data.sign() * n
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()


def make_layers(cfg, conv_layer, batch_norm=True):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = conv_layer(in_channels, v, kernel_size=3, padding=1, bias=False)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs = {
    "2": [64, "M", 64, "M"],
    "4": [64, 64, "M", 128, 128, "M"],
    "6": [64, 64, "M", 128, 128, "M", 256, 256, "M"],
    "8": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M"],
    "11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512],
    "13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512],
    "16": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
    ],
}


def vgg2(conv_layer, linear_layer, init_type, **kwargs):
    n = [i for i in cfgs["2"] if isinstance(i, int)][-1]
    model = VGG(
        make_layers(cfgs["2"], conv_layer, batch_norm=False), n, linear_layer, **kwargs
    )
    initialize_weights(model, init_type)
    return model


def vgg2_bn(conv_layer, linear_layer, init_type, **kwargs):
    n = [i for i in cfgs["2"] if isinstance(i, int)][-1]
    model = VGG(
        make_layers(cfgs["2"], conv_layer, batch_norm=True), n, linear_layer, **kwargs
    )
    initialize_weights(model, init_type)
    return model


def vgg4(conv_layer, linear_layer, init_type, **kwargs):
    n = [i for i in cfgs["4"] if isinstance(i, int)][-1]
    model = VGG(
        make_layers(cfgs["4"], conv_layer, batch_norm=False), n, linear_layer, **kwargs
    )
    initialize_weights(model, init_type)
    return model


def vgg4_bn(conv_layer, linear_layer, init_type, **kwargs):
    n = [i for i in cfgs["4"] if isinstance(i, int)][-1]
    model = VGG(
        make_layers(cfgs["4"], conv_layer, batch_norm=True), n, linear_layer, **kwargs
    )
    initialize_weights(model, init_type)
    return model


def vgg6(conv_layer, linear_layer, init_type, **kwargs):
    n = [i for i in cfgs["6"] if isinstance(i, int)][-1]
    model = VGG(
        make_layers(cfgs["6"], conv_layer, batch_norm=False), n, linear_layer, **kwargs
    )
    initialize_weights(model, init_type)
    return model


def vgg6_bn(conv_layer, linear_layer, init_type, **kwargs):
    n = [i for i in cfgs["6"] if isinstance(i, int)][-1]
    model = VGG(
        make_layers(cfgs["6"], conv_layer, batch_norm=True), n, linear_layer, **kwargs
    )
    initialize_weights(model, init_type)
    return model


def vgg8(conv_layer, linear_layer, init_type, **kwargs):
    n = [i for i in cfgs["8"] if isinstance(i, int)][-1]
    model = VGG(
        make_layers(cfgs["8"], conv_layer, batch_norm=False), n, linear_layer, **kwargs
    )
    initialize_weights(model, init_type)
    return model


def vgg8_bn(conv_layer, linear_layer, init_type, **kwargs):
    n = [i for i in cfgs["8"] if isinstance(i, int)][-1]
    model = VGG(
        make_layers(cfgs["8"], conv_layer, batch_norm=True), n, linear_layer, **kwargs
    )
    initialize_weights(model, init_type)
    return model


def vgg11(conv_layer, linear_layer, init_type, **kwargs):
    n = [i for i in cfgs["11"] if isinstance(i, int)][-1]
    model = VGG(
        make_layers(cfgs["11"], conv_layer, batch_norm=False), n, linear_layer, **kwargs
    )
    initialize_weights(model, init_type)
    return model


def vgg11_bn(conv_layer, linear_layer, init_type, **kwargs):
    n = [i for i in cfgs["11"] if isinstance(i, int)][-1]
    model = VGG(
        make_layers(cfgs["11"], conv_layer, batch_norm=True), n, linear_layer, **kwargs
    )
    initialize_weights(model, init_type)
    return model


def vgg13(conv_layer, linear_layer, init_type, **kwargs):
    n = [i for i in cfgs["13"] if isinstance(i, int)][-1]
    model = VGG(
        make_layers(cfgs["13"], conv_layer, batch_norm=False), n, linear_layer, **kwargs
    )
    initialize_weights(model, init_type)
    return model


def vgg13_bn(conv_layer, linear_layer, init_type, **kwargs):
    n = [i for i in cfgs["13"] if isinstance(i, int)][-1]
    model = VGG(
        make_layers(cfgs["13"], conv_layer, batch_norm=True), n, linear_layer, **kwargs
    )
    initialize_weights(model, init_type)
    return model


def vgg16(conv_layer, linear_layer, init_type, **kwargs):
    n = [i for i in cfgs["16"] if isinstance(i, int)][-1]
    model = VGG(
        make_layers(cfgs["16"], conv_layer, batch_norm=False), n, linear_layer, **kwargs
    )
    initialize_weights(model, init_type)
    return model


def vgg16_bn(conv_layer, linear_layer, init_type, **kwargs):
    n = [i for i in cfgs["16"] if isinstance(i, int)][-1]
    model = VGG(
        make_layers(cfgs["16"], conv_layer, batch_norm=True), n, linear_layer, **kwargs
    )
    initialize_weights(model, init_type)
    return model
