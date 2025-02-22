import torch
from torch import nn


def convs(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels,
                  out_channels=out_channels,
                  kernel_size=3,
                  padding=1),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(inplace=True),
        nn.Conv2d(in_channels=out_channels,
                  out_channels=out_channels,
                  kernel_size=1),
        nn.LeakyReLU(inplace=True)
    )


def down_irblock(in_channels, out_channels):
    return nn.Sequential(
        InvertedResidualBlock(in_channels, in_channels),
        nn.LeakyReLU(inplace=True),
        nn.Conv2d(in_channels, out_channels, kernel_size=1),
        nn.LeakyReLU(inplace=True)
    )


class ConvNormReLU(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            groups=1,
            activation=None,
            norm_layer=None):

        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if activation is None:
            activation = nn.LeakyReLU()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            groups=groups,
            bias=False)
        self.norm = norm_layer(out_channels)
        self.activate = activation

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        return self.activate(x)


class InvertedResidualBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            t=6,
            stride=1,
            norm_layer=None):

        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        hidden_dim = int(in_channels * t)
        self.down_sample = stride == 2

        modules = [ConvNormReLU(in_channels, hidden_dim,
                                kernel_size=1, norm_layer=norm_layer)]
        if t > 1:
            modules.append(ConvNormReLU(hidden_dim,
                                        hidden_dim,
                                        stride=stride,
                                        groups=hidden_dim,
                                        norm_layer=norm_layer))
            modules.append(
                nn.Conv2d(hidden_dim,
                          out_channels,
                          kernel_size=1,
                          bias=False))
            modules.append(norm_layer(out_channels))
        self.irblock = nn.Sequential(*modules)

    def forward(self, x):
        if self.down_sample:
            return self.irblock(x)
        else:
            return self.irblock(x) + x


class UNet_Based(nn.Module):
    def __init__(self, in_channels=3, based_dim=64):
        super().__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv1 = convs(in_channels, based_dim)
        self.sconv1 = nn.Conv2d(in_channels=based_dim,
                                out_channels=based_dim,
                                kernel_size=3,
                                stride=2,
                                padding=1)
        self.conv2 = convs(based_dim, based_dim*2)
        self.sconv2 = nn.Conv2d(in_channels=based_dim*2,
                                out_channels=based_dim*2,
                                kernel_size=3,
                                stride=2,
                                padding=1)
        self.conv3 = convs(based_dim*2, based_dim*4)
        self.sconv3 = nn.Conv2d(in_channels=based_dim*4,
                                out_channels=based_dim*4,
                                kernel_size=3,
                                stride=2,
                                padding=1)
        self.conv4 = convs(based_dim*4, based_dim*8)
        self.sconv4 = nn.Conv2d(in_channels=based_dim*8,
                                out_channels=based_dim*8,
                                kernel_size=3,
                                stride=2,
                                padding=1)
        self.conv5 = convs(based_dim*8, based_dim*16)

        self.tconv1 = nn.ConvTranspose2d(
            in_channels=based_dim*16,
            out_channels=based_dim*8,
            kernel_size=2,
            stride=2
        )
        self.upconv1 = convs(based_dim*16, based_dim*8)
        self.tconv2 = nn.ConvTranspose2d(
            in_channels=based_dim*8,
            out_channels=based_dim*4,
            kernel_size=2,
            stride=2
        )
        self.upconv2 = convs(based_dim*8, based_dim*4)
        self.tconv3 = nn.ConvTranspose2d(
            in_channels=based_dim*4,
            out_channels=based_dim*2,
            kernel_size=2,
            stride=2
        )
        self.upconv3 = convs(based_dim*4, based_dim*2)
        self.tconv4 = nn.ConvTranspose2d(
            in_channels=based_dim*2,
            out_channels=based_dim,
            kernel_size=2,
            stride=2
        )
        self.upconv4 = convs(based_dim*2, based_dim)

        self.sep = nn.Conv2d(
            in_channels=based_dim,
            out_channels=in_channels*2,
            kernel_size=1
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, combined_image):
        # encoder
        # bs, c, h, w

        # maxpool
        # x1 = self.conv1(combined_image)
        # x2 = self.maxpool(x1)
        # x3 = self.conv2(x2)
        # x4 = self.maxpool(x3)
        # x5 = self.conv3(x4)
        # x6 = self.maxpool(x5)
        # x7 = self.conv4(x6)
        # x8 = self.maxpool(x7)
        # x9 = self.conv5(x8)

        x1 = self.conv1(combined_image)
        x = self.sconv1(x1)
        x2 = self.conv2(x)
        x = self.sconv2(x2)
        x3 = self.conv3(x)
        x = self.sconv3(x3)
        x4 = self.conv4(x)
        x = self.sconv4(x4)
        x5 = self.conv5(x)

        # decoder
        x = self.tconv1(x5)
        x = self.upconv1(torch.cat([x, x4], dim=1))
        x = self.tconv2(x)
        x = self.upconv2(torch.cat([x, x3], dim=1))
        x = self.tconv3(x)
        x = self.upconv3(torch.cat([x, x2], dim=1))
        x = self.tconv4(x)
        x = self.upconv4(torch.cat([x, x1], dim=1))
        x = self.sep(x)
        return x


class MobileUnet(nn.Module):
    def __init__(self, in_channels, based_dim=32):
        super().__init__()
        self.conv1 = convs(in_channels, based_dim)
        self.sconv1 = InvertedResidualBlock(based_dim,
                                            based_dim,
                                            stride=2)
        self.conv2 = InvertedResidualBlock(based_dim, based_dim)
        self.sconv2 = InvertedResidualBlock(based_dim,
                                            based_dim*2,
                                            stride=2)
        self.conv3 = InvertedResidualBlock(based_dim*2, based_dim*2)
        self.sconv3 = InvertedResidualBlock(based_dim*2,
                                            based_dim*4,
                                            stride=2)
        self.conv4 = InvertedResidualBlock(based_dim*4, based_dim*4)
        self.sconv4 = InvertedResidualBlock(based_dim*4,
                                            based_dim*8,
                                            stride=2)
        self.conv5 = InvertedResidualBlock(based_dim*8, based_dim*8)
        self.sconv5 = InvertedResidualBlock(based_dim*8,
                                            based_dim*16,
                                            stride=2)

        self.tconv1 = nn.ConvTranspose2d(
            in_channels=based_dim*16,
            out_channels=based_dim*8,
            kernel_size=2,
            stride=2
        )
        self.upconv1 = down_irblock(based_dim*16, based_dim*8)
        self.tconv2 = nn.ConvTranspose2d(
            in_channels=based_dim*8,
            out_channels=based_dim*4,
            kernel_size=2,
            stride=2
        )
        self.upconv2 = down_irblock(based_dim*8, based_dim*4)
        self.tconv3 = nn.ConvTranspose2d(
            in_channels=based_dim*4,
            out_channels=based_dim*2,
            kernel_size=2,
            stride=2
        )
        self.upconv3 = down_irblock(based_dim*4, based_dim*2)
        self.tconv4 = nn.ConvTranspose2d(
            in_channels=based_dim*2,
            out_channels=based_dim,
            kernel_size=2,
            stride=2
        )
        self.upconv4 = down_irblock(based_dim*2, based_dim)
        self.tconv5 = nn.ConvTranspose2d(
            in_channels=based_dim,
            out_channels=based_dim,
            kernel_size=2,
            stride=2
        )

        self.sep = nn.Conv2d(
            in_channels=based_dim,
            out_channels=in_channels*2,
            kernel_size=1
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, combined_image):
        # encoder
        # bs, c, h, w
        x = self.conv1(combined_image)
        x1 = self.sconv1(x)
        x = self.conv2(x1)
        x2 = self.sconv2(x)
        x = self.conv3(x2)
        x3 = self.sconv3(x)
        x = self.conv4(x3)
        x4 = self.sconv4(x)
        x = self.conv5(x4)
        x = self.sconv5(x)

        # decoder
        x = self.tconv1(x)
        x = self.upconv1(torch.cat([x, x4], dim=1))
        x = self.tconv2(x)
        x = self.upconv2(torch.cat([x, x3], dim=1))
        x = self.tconv3(x)
        x = self.upconv3(torch.cat([x, x2], dim=1))
        x = self.tconv4(x)
        x = self.upconv4(torch.cat([x, x1], dim=1))
        x = self.tconv5(x)
        x = self.sep(x)
        return x


class MobileUnet1DUp(nn.Module):
    def __init__(self, in_channels, based_dim=32):
        super().__init__()
        self.conv1 = convs(in_channels, based_dim)
        self.sconv1 = InvertedResidualBlock(based_dim,
                                            based_dim,
                                            stride=2)
        self.conv2 = InvertedResidualBlock(based_dim, based_dim)
        self.sconv2 = InvertedResidualBlock(based_dim,
                                            based_dim*2,
                                            stride=2,)
        self.conv3 = InvertedResidualBlock(based_dim*2, based_dim*2)
        self.sconv3 = InvertedResidualBlock(based_dim*2,
                                            based_dim*4,
                                            stride=2)
        self.conv4 = InvertedResidualBlock(based_dim*4, based_dim*4)
        self.sconv4 = InvertedResidualBlock(based_dim*4,
                                            based_dim*8,
                                            stride=2)
        self.conv5 = InvertedResidualBlock(based_dim*8, based_dim*8)
        self.sconv5 = InvertedResidualBlock(based_dim*8,
                                            based_dim*16,
                                            stride=2)

        self.tconv1 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=based_dim*16,
                out_channels=based_dim*16,
                kernel_size=2,
                stride=2
            ),
            nn.LeakyReLU(),
            down_irblock(
                in_channels=based_dim*16,
                out_channels=based_dim*8
            )
        )
        self.upconv1 = down_irblock(based_dim*16, based_dim*8)
        self.tconv2 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=based_dim*8,
                out_channels=based_dim*8,
                kernel_size=2,
                stride=2
            ),
            nn.LeakyReLU(),
            down_irblock(
                in_channels=based_dim*8,
                out_channels=based_dim*4
            )
        )
        self.upconv2 = down_irblock(based_dim*8, based_dim*4)
        self.tconv3 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=based_dim*4,
                out_channels=based_dim*4,
                kernel_size=2,
                stride=2
            ),
            nn.LeakyReLU(),
            down_irblock(
                in_channels=based_dim*4,
                out_channels=based_dim*2
            )
        )
        self.upconv3 = down_irblock(based_dim*4, based_dim*2)
        self.tconv4 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=based_dim*2,
                out_channels=based_dim*2,
                kernel_size=2,
                stride=2
            ),
            nn.LeakyReLU(),
            down_irblock(
                in_channels=based_dim*2,
                out_channels=based_dim
            )
        )
        self.upconv4 = down_irblock(based_dim*2, based_dim)
        self.tconv5 = nn.ConvTranspose2d(
            in_channels=based_dim,
            out_channels=based_dim,
            kernel_size=2,
            stride=2
        )

        self.sep = down_irblock(
            in_channels=based_dim,
            out_channels=in_channels*2
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, combined_image):
        # encoder
        # bs, c, h, w
        x = self.conv1(combined_image)
        x1 = self.sconv1(x)
        x = self.conv2(x1)
        x2 = self.sconv2(x)
        x = self.conv3(x2)
        x3 = self.sconv3(x)
        x = self.conv4(x3)
        x4 = self.sconv4(x)
        x = self.conv5(x4)
        x = self.sconv5(x)

        # decoder
        x = self.tconv1(x)
        x = self.upconv1(torch.cat([x, x4], dim=1))
        x = self.tconv2(x)
        x = self.upconv2(torch.cat([x, x3], dim=1))
        x = self.tconv3(x)
        x = self.upconv3(torch.cat([x, x2], dim=1))
        x = self.tconv4(x)
        x = self.upconv4(torch.cat([x, x1], dim=1))
        x = self.tconv5(x)
        x = self.sep(x)
        return x


class MobileUnet2Heads(nn.Module):
    def __init__(self, in_channels, based_dim=32):
        super().__init__()
        self.conv1 = convs(in_channels, based_dim)
        self.sconv1 = InvertedResidualBlock(based_dim,
                                            based_dim,
                                            stride=2)
        self.conv2 = InvertedResidualBlock(based_dim, based_dim)
        self.sconv2 = InvertedResidualBlock(based_dim,
                                            based_dim*2,
                                            stride=2,)
        self.conv3 = InvertedResidualBlock(based_dim*2, based_dim*2)
        self.sconv3 = InvertedResidualBlock(based_dim*2,
                                            based_dim*4,
                                            stride=2)
        self.conv4 = InvertedResidualBlock(based_dim*4, based_dim*4)
        self.sconv4 = InvertedResidualBlock(based_dim*4,
                                            based_dim*8,
                                            stride=2)
        self.conv5 = InvertedResidualBlock(based_dim*8, based_dim*8)
        self.sconv5 = InvertedResidualBlock(based_dim*8,
                                            based_dim*16,
                                            stride=2)

        self.tconv1h1 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=based_dim*16,
                out_channels=based_dim*16,
                kernel_size=2,
                stride=2
            ),
            nn.LeakyReLU(),
            down_irblock(
                in_channels=based_dim*16,
                out_channels=based_dim*8
            )
        )
        self.tconv1h2 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=based_dim*16,
                out_channels=based_dim*16,
                kernel_size=2,
                stride=2
            ),
            nn.LeakyReLU(),
            down_irblock(
                in_channels=based_dim*16,
                out_channels=based_dim*8
            )
        )
        self.upconv1h1 = down_irblock(based_dim*16, based_dim*8)
        self.tconv2h1 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=based_dim*8,
                out_channels=based_dim*8,
                kernel_size=2,
                stride=2
            ),
            nn.LeakyReLU(),
            down_irblock(
                in_channels=based_dim*8,
                out_channels=based_dim*4
            )
        )
        self.upconv1h2 = down_irblock(based_dim*16, based_dim*8)
        self.tconv2h2 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=based_dim*8,
                out_channels=based_dim*8,
                kernel_size=2,
                stride=2
            ),
            nn.LeakyReLU(),
            down_irblock(
                in_channels=based_dim*8,
                out_channels=based_dim*4
            )
        )
        self.upconv2h1 = down_irblock(based_dim*8, based_dim*4)
        self.tconv3h1 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=based_dim*4,
                out_channels=based_dim*4,
                kernel_size=2,
                stride=2
            ),
            nn.LeakyReLU(),
            down_irblock(
                in_channels=based_dim*4,
                out_channels=based_dim*2
            )
        )
        self.upconv2h2 = down_irblock(based_dim*8, based_dim*4)
        self.tconv3h2 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=based_dim*4,
                out_channels=based_dim*4,
                kernel_size=2,
                stride=2
            ),
            nn.LeakyReLU(),
            down_irblock(
                in_channels=based_dim*4,
                out_channels=based_dim*2
            )
        )
        self.upconv3h1 = down_irblock(based_dim*4, based_dim*2)
        self.tconv4h1 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=based_dim*2,
                out_channels=based_dim*2,
                kernel_size=2,
                stride=2
            ),
            nn.LeakyReLU(),
            down_irblock(
                in_channels=based_dim*2,
                out_channels=based_dim
            )
        )
        self.upconv3h2 = down_irblock(based_dim*4, based_dim*2)
        self.tconv4h2 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=based_dim*2,
                out_channels=based_dim*2,
                kernel_size=2,
                stride=2
            ),
            nn.LeakyReLU(),
            down_irblock(
                in_channels=based_dim*2,
                out_channels=based_dim
            )
        )
        self.upconv4h1 = down_irblock(based_dim*2, based_dim)
        self.tconv5h1 = nn.ConvTranspose2d(
            in_channels=based_dim,
            out_channels=based_dim,
            kernel_size=2,
            stride=2
        )
        self.upconv4h2 = down_irblock(based_dim*2, based_dim)
        self.tconv5h2 = nn.ConvTranspose2d(
            in_channels=based_dim,
            out_channels=based_dim,
            kernel_size=2,
            stride=2
        )

        self.seph1 = down_irblock(
            in_channels=based_dim,
            out_channels=in_channels
        )
        self.seph2 = down_irblock(
            in_channels=based_dim,
            out_channels=in_channels
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, combined_image):
        # encoder
        # bs, c, h, w
        x = self.conv1(combined_image)
        x1 = self.sconv1(x)
        x = self.conv2(x1)
        x2 = self.sconv2(x)
        x = self.conv3(x2)
        x3 = self.sconv3(x)
        x = self.conv4(x3)
        x4 = self.sconv4(x)
        x = self.conv5(x4)
        x = self.sconv5(x)

        # decoder
        xh1, xh2 = self.tconv1h1(x), self.tconv1h2(x)
        xh1, xh2 = (self.upconv1h1(torch.cat([xh1, x4], dim=1)),
                    self.upconv1h2(torch.cat([xh2, x4], dim=1)))
        xh1, xh2 = self.tconv2h1(xh1), self.tconv2h2(xh2)
        xh1, xh2 = (self.upconv2h1(torch.cat([xh1, x3], dim=1)),
                    self.upconv2h2(torch.cat([xh2, x3], dim=1)))
        xh1, xh2 = self.tconv3h1(xh1), self.tconv3h2(xh2)
        xh1, xh2 = (self.upconv3h1(torch.cat([xh1, x2], dim=1)),
                    self.upconv3h2(torch.cat([xh2, x2], dim=1)))
        xh1, xh2 = self.tconv4h1(xh1), self.tconv4h2(xh2)
        xh1, xh2 = (self.upconv4h1(torch.cat([xh1, x1], dim=1)),
                    self.upconv4h2(torch.cat([xh2, x1], dim=1)))
        xh1, xh2 = self.tconv5h1(xh1), self.tconv5h2(xh2)
        xh1, xh2 = self.seph1(xh1), self.seph2(xh2)
        x = torch.cat([xh1, xh2], dim=1)
        return x


class MobileUnet2HeadsNoPrev(MobileUnet2Heads):
    def __init__(self, in_channels, based_dim=32):
        super().__init__(in_channels, based_dim)

    def forward(self, combined_image):
        # encoder
        # bs, c, h, w
        x = self.conv1(combined_image)
        x = self.sconv1(x)
        x = self.conv2(x)
        x = self.sconv2(x)
        x = self.conv3(x)
        x = self.sconv3(x)
        x = self.conv4(x)
        x = self.sconv4(x)
        x = self.conv5(x)
        x = self.sconv5(x)

        # decoder
        xh1, xh2 = self.tconv1h1(x), self.tconv1h2(x)
        xh1, xh2 = (self.upconv1h1(torch.cat([xh1, xh2], dim=1)),
                    self.upconv1h2(torch.cat([xh2, xh1], dim=1)))
        xh1, xh2 = self.tconv2h1(xh1), self.tconv2h2(xh2)
        xh1, xh2 = (self.upconv2h1(torch.cat([xh1, xh2], dim=1)),
                    self.upconv2h2(torch.cat([xh2, xh1], dim=1)))
        xh1, xh2 = self.tconv3h1(xh1), self.tconv3h2(xh2)
        xh1, xh2 = (self.upconv3h1(torch.cat([xh1, xh2], dim=1)),
                    self.upconv3h2(torch.cat([xh2, xh1], dim=1)))
        xh1, xh2 = self.tconv4h1(xh1), self.tconv4h2(xh2)
        xh1, xh2 = (self.upconv4h1(torch.cat([xh1, xh2], dim=1)),
                    self.upconv4h2(torch.cat([xh2, xh1], dim=1)))
        xh1, xh2 = self.tconv5h1(xh1), self.tconv5h2(xh2)
        xh1, xh2 = self.seph1(xh1), self.seph2(xh2)
        x = torch.cat([xh1, xh2], dim=1)
        return x


class MobileUnetNoCat(nn.Module):
    def __init__(self, in_channels, based_dim=32):
        super().__init__()
        self.conv1 = convs(in_channels, based_dim)
        self.sconv1 = InvertedResidualBlock(based_dim,
                                            based_dim,
                                            stride=2)
        self.conv2 = InvertedResidualBlock(based_dim, based_dim)
        self.sconv2 = InvertedResidualBlock(based_dim,
                                            based_dim*2,
                                            stride=2,)
        self.conv3 = InvertedResidualBlock(based_dim*2, based_dim*2)
        self.sconv3 = InvertedResidualBlock(based_dim*2,
                                            based_dim*4,
                                            stride=2)
        self.conv4 = InvertedResidualBlock(based_dim*4, based_dim*4)
        self.sconv4 = InvertedResidualBlock(based_dim*4,
                                            based_dim*8,
                                            stride=2)
        self.conv5 = InvertedResidualBlock(based_dim*8, based_dim*8)
        self.sconv5 = InvertedResidualBlock(based_dim*8,
                                            based_dim*16,
                                            stride=2)
        self.encoder = nn.Sequential(
            self.conv1,
            self.sconv1,
            self.conv2,
            self.sconv2,
            self.conv3,
            self.sconv3,
            self.conv4,
            self.sconv4,
            self.conv5,
            self.sconv5
        )

        self.tconv1 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=based_dim*16,
                out_channels=based_dim*16,
                kernel_size=2,
                stride=2
            ),
            nn.LeakyReLU(),
            down_irblock(
                in_channels=based_dim*16,
                out_channels=based_dim*8
            )
        )
        self.tconv2 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=based_dim*8,
                out_channels=based_dim*8,
                kernel_size=2,
                stride=2
            ),
            nn.LeakyReLU(),
            down_irblock(
                in_channels=based_dim*8,
                out_channels=based_dim*4
            )
        )
        self.tconv3 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=based_dim*4,
                out_channels=based_dim*4,
                kernel_size=2,
                stride=2
            ),
            nn.LeakyReLU(),
            down_irblock(
                in_channels=based_dim*4,
                out_channels=based_dim*2
            )
        )
        self.tconv4 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=based_dim*2,
                out_channels=based_dim*2,
                kernel_size=2,
                stride=2
            ),
            nn.LeakyReLU(),
            down_irblock(
                in_channels=based_dim*2,
                out_channels=based_dim
            )
        )
        self.tconv5 = nn.ConvTranspose2d(
            in_channels=based_dim,
            out_channels=based_dim,
            kernel_size=2,
            stride=2
        )

        self.sep = down_irblock(
            in_channels=based_dim,
            out_channels=in_channels*2
        )
        self.decoder = nn.Sequential(
            self.tconv1,
            self.tconv2,
            self.tconv3,
            self.tconv4,
            self.tconv5,
            nn.LeakyReLU(),
            self.sep
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, combined_image):
        # encoder
        # bs, c, h, w
        x = self.encoder(combined_image)

        # decoder
        x = self.decoder(x)
        return x


class MobileUnetNoCat2Heads(nn.Module):
    def __init__(self, in_channels, based_dim=32):
        super().__init__()
        self.conv1 = convs(in_channels, based_dim)
        self.sconv1 = InvertedResidualBlock(based_dim,
                                            based_dim,
                                            stride=2)
        self.conv2 = InvertedResidualBlock(based_dim, based_dim)
        self.sconv2 = InvertedResidualBlock(based_dim,
                                            based_dim*2,
                                            stride=2,)
        self.conv3 = InvertedResidualBlock(based_dim*2, based_dim*2)
        self.sconv3 = InvertedResidualBlock(based_dim*2,
                                            based_dim*4,
                                            stride=2)
        self.conv4 = InvertedResidualBlock(based_dim*4, based_dim*4)
        self.sconv4 = InvertedResidualBlock(based_dim*4,
                                            based_dim*8,
                                            stride=2)
        self.conv5 = InvertedResidualBlock(based_dim*8, based_dim*8)
        self.sconv5 = InvertedResidualBlock(based_dim*8,
                                            based_dim*16,
                                            stride=2)
        self.encoder = nn.Sequential(
            self.conv1,
            self.sconv1,
            self.conv2,
            self.sconv2,
            self.conv3,
            self.sconv3,
            self.conv4,
            self.sconv4,
            self.conv5,
            self.sconv5
        )

        self.tconv1h1 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=based_dim*16,
                out_channels=based_dim*16,
                kernel_size=2,
                stride=2
            ),
            nn.LeakyReLU(),
            down_irblock(
                in_channels=based_dim*16,
                out_channels=based_dim*8
            )
        )
        self.tconv1h2 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=based_dim*16,
                out_channels=based_dim*16,
                kernel_size=2,
                stride=2
            ),
            nn.LeakyReLU(),
            down_irblock(
                in_channels=based_dim*16,
                out_channels=based_dim*8
            )
        )
        self.tconv2h1 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=based_dim*8,
                out_channels=based_dim*8,
                kernel_size=2,
                stride=2
            ),
            nn.LeakyReLU(),
            down_irblock(
                in_channels=based_dim*8,
                out_channels=based_dim*4
            )
        )
        self.tconv2h2 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=based_dim*8,
                out_channels=based_dim*8,
                kernel_size=2,
                stride=2
            ),
            nn.LeakyReLU(),
            down_irblock(
                in_channels=based_dim*8,
                out_channels=based_dim*4
            )
        )
        self.tconv3h1 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=based_dim*4,
                out_channels=based_dim*4,
                kernel_size=2,
                stride=2
            ),
            nn.LeakyReLU(),
            down_irblock(
                in_channels=based_dim*4,
                out_channels=based_dim*2
            )
        )
        self.tconv3h2 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=based_dim*4,
                out_channels=based_dim*4,
                kernel_size=2,
                stride=2
            ),
            nn.LeakyReLU(),
            down_irblock(
                in_channels=based_dim*4,
                out_channels=based_dim*2
            )
        )
        self.tconv4h1 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=based_dim*2,
                out_channels=based_dim*2,
                kernel_size=2,
                stride=2
            ),
            nn.LeakyReLU(),
            down_irblock(
                in_channels=based_dim*2,
                out_channels=based_dim
            )
        )
        self.tconv4h2 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=based_dim*2,
                out_channels=based_dim*2,
                kernel_size=2,
                stride=2
            ),
            nn.LeakyReLU(),
            down_irblock(
                in_channels=based_dim*2,
                out_channels=based_dim
            )
        )
        self.tconv5h1 = nn.ConvTranspose2d(
            in_channels=based_dim,
            out_channels=based_dim,
            kernel_size=2,
            stride=2
        )
        self.tconv5h2 = nn.ConvTranspose2d(
            in_channels=based_dim,
            out_channels=based_dim,
            kernel_size=2,
            stride=2
        )

        self.seph1 = down_irblock(
            in_channels=based_dim,
            out_channels=in_channels
        )
        self.seph2 = down_irblock(
            in_channels=based_dim,
            out_channels=in_channels
        )
        self.decoderh1 = nn.Sequential(
            self.tconv1h1,
            self.tconv2h1,
            self.tconv3h1,
            self.tconv4h1,
            self.tconv5h1,
            nn.LeakyReLU(),
            self.seph1
        )

        self.decoderh2 = nn.Sequential(
            self.tconv1h2,
            self.tconv2h2,
            self.tconv3h2,
            self.tconv4h2,
            self.tconv5h2,
            nn.LeakyReLU(),
            self.seph2
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, combined_image):
        # encoder
        # bs, c, h, w
        x = self.encoder(combined_image)

        # decoder
        xh1, xh2 = self.decoderh1(x), self.decoderh2(x)
        x = torch.cat([xh1, xh2], dim=1)
        return x


class DeConvNormReLU(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2,
            groups=1,
            activation=None,
            norm_layer=None):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if activation is None:
            activation = nn.LeakyReLU()
        self.conv = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            groups=groups,
            bias=False)
        self.norm = norm_layer(out_channels)
        self.activate = activation

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        return self.activate(x)


class InvertedResidualTransBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            stride=2,
            t=6,
            norm_layer=None):

        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        hidden_dim = int(in_channels * t)
        modules = [ConvNormReLU(in_channels, hidden_dim,
                                kernel_size=1, norm_layer=norm_layer)]
        modules.append(DeConvNormReLU(hidden_dim,
                                      hidden_dim,
                                      stride=stride,
                                      groups=hidden_dim,
                                      norm_layer=norm_layer))
        modules.append(
            nn.Conv2d(hidden_dim,
                      out_channels,
                      kernel_size=1,
                      bias=False))
        modules.append(norm_layer(out_channels))
        self.irtblock = nn.Sequential(*modules)

    def forward(self, x):
        return self.irtblock(x)


class DistractionConv(nn.Module):
    """Inspired by SKNet"""

    def __init__(self, in_channels, M=3, r=16, stride=1, L=32, norm=None):
        super().__init__()
        d = max(in_channels//r, L)
        if norm is None:
            norm = nn.BatchNorm2d
        self.in_channels, self.M = in_channels, M
        self.covsA, self.covsB = nn.ModuleList([]), nn.ModuleList([])
        for _ in range(M):
            self.covsA.append(InvertedResidualBlock(
                in_channels,
                in_channels,
                stride=stride,
            ))
            self.covsB.append(InvertedResidualBlock(
                in_channels,
                in_channels,
                stride=stride,
            ))

        self.gapA = nn.AdaptiveAvgPool2d((1, 1))
        self.gapB = nn.AdaptiveAvgPool2d((1, 1))
        self.fcA = nn.Sequential(
            nn.Conv2d(in_channels, d, kernel_size=1, stride=1, bias=False),
            norm(d),
            nn.LeakyReLU(inplace=True)
        )
        self.fcB = nn.Sequential(
            nn.Conv2d(in_channels, d, kernel_size=1, stride=1, bias=False),
            norm(d),
            nn.LeakyReLU(inplace=True)
        )
        self.fcsA = nn.ModuleList(
            [nn.Conv2d(
                d, in_channels, kernel_size=1, stride=1
            ) for _ in range(M)])
        self.fcsB = nn.ModuleList(
            [nn.Conv2d(
                d, in_channels, kernel_size=1, stride=1
            ) for _ in range(M)])
        # self.softmin = nn.Softmin(dim=1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, A, B):
        batch_size = A.size(0)

        featsA, featsB = ([deconv(A) for deconv in self.covsA],
                          [deconv(B) for deconv in self.covsB])
        featsA, featsB = torch.cat(featsA, dim=1), torch.cat(featsB, dim=1)
        featsA, featsB = (featsA.view(
            batch_size,
            self.M,
            self.in_channels,
            featsA.shape[2],
            featsA.shape[3]),
            featsB.view(
            batch_size,
            self.M,
            self.in_channels,
            featsB.shape[2],
            featsB.shape[3]))

        featsA_U, featsB_U = torch.sum(featsA, dim=1), torch.sum(featsB, dim=1)
        featsA_S, featsB_S = self.gapA(featsA_U), self.gapB(featsB_U)
        featsA_Z, featsB_Z = self.fcA(featsA_S), self.fcB(featsB_S)

        distractionA, distractionB = ([fc(featsA_Z) for fc in self.fcsA],
                                      [fc(featsB_Z) for fc in self.fcsB])
        distractionA, distractionB = (torch.cat(distractionA, dim=1),
                                      torch.cat(distractionB, dim=1))
        distractionA, distractionB = (distractionA.view(
            batch_size,
            self.M,
            self.in_channels, 1, 1
        ),
            distractionB.view(
            batch_size,
            self.M,
            self.in_channels, 1, 1)
        )
        distractionA, distractionB = (self.softmax(distractionA),
                                      self.softmax(distractionB))

        featsA_V, featsB_V = (torch.sum(featsA*distractionA, dim=1),
                              torch.sum(featsB*distractionB, dim=1))
        return featsA_V, featsB_V


class MobileUnet2HeadsDistraction(MobileUnetNoCat2Heads):
    def __init__(self, in_channels, based_dim=32):
        super().__init__(in_channels, based_dim)
        self.distract1 = DistractionConv(based_dim*8)
        self.tconv1h1 = InvertedResidualTransBlock(based_dim*16, based_dim*8)
        self.tconv1h2 = InvertedResidualTransBlock(based_dim*16, based_dim*8)
        self.distract2 = DistractionConv(based_dim*4)
        self.tconv2h1 = InvertedResidualTransBlock(based_dim*8, based_dim*4)
        self.tconv2h2 = InvertedResidualTransBlock(based_dim*8, based_dim*4)
        self.distract3 = DistractionConv(based_dim*2)
        self.tconv3h1 = InvertedResidualTransBlock(based_dim*4, based_dim*2)
        self.tconv3h2 = InvertedResidualTransBlock(based_dim*4, based_dim*2)
        self.distract4 = DistractionConv(based_dim)
        self.tconv4h1 = InvertedResidualTransBlock(based_dim*2, based_dim)
        self.tconv4h2 = InvertedResidualTransBlock(based_dim*2, based_dim)
        self.distract5 = DistractionConv(in_channels)
        self.tconv5h1 = InvertedResidualTransBlock(based_dim, in_channels)
        self.tconv5h2 = InvertedResidualTransBlock(based_dim, in_channels)

        self._initialize_weights()

    def forward(self, combined_image):
        # bs, c, h, w
        # encoder
        A = B = self.encoder(combined_image)

        # decoder
        A, B = self.tconv1h1(A), self.tconv1h2(B)
        A, B = self.distract1(A, B)
        A, B = self.tconv2h1(A), self.tconv2h2(B)
        A, B = self.distract2(A, B)
        A, B = self.tconv3h1(A), self.tconv3h2(B)
        A, B = self.distract3(A, B)
        A, B = self.tconv4h1(A), self.tconv4h2(B)
        A, B = self.distract4(A, B)
        A, B = self.tconv5h1(A), self.tconv5h2(B)
        A, B = self.distract5(A, B)
        x = torch.cat((A, B), dim=1)
        return x
