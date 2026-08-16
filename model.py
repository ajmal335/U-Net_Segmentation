import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=1):
        super().__init__()

        # Encoder
        self.encoder1 = DoubleConv(in_channels, 64)
        self.encoder2 = DoubleConv(64, 128)
        self.encoder3 = DoubleConv(128, 256)
        self.encoder4 = DoubleConv(256, 512)

        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        # Bottleneck
        self.bottleneck = DoubleConv(512, 1024)

        # Decoder
        self.up4 = nn.ConvTranspose2d(
            1024,
            512,
            kernel_size=2,
            stride=2
        )

        self.decoder4 = DoubleConv(
            1024,
            512
        )

        self.up3 = nn.ConvTranspose2d(
            512,
            256,
            kernel_size=2,
            stride=2
        )

        self.decoder3 = DoubleConv(
            512,
            256
        )

        self.up2 = nn.ConvTranspose2d(
            256,
            128,
            kernel_size=2,
            stride=2
        )

        self.decoder2 = DoubleConv(
            256,
            128
        )

        self.up1 = nn.ConvTranspose2d(
            128,
            64,
            kernel_size=2,
            stride=2
        )

        self.decoder1 = DoubleConv(
            128,
            64
        )

        # Final segmentation layer
        self.final = nn.Conv2d(
            64,
            num_classes,
            kernel_size=1
        )

    def forward(self, x):

        # Encoder
        enc1 = self.encoder1(x)

        enc2 = self.encoder2(
            self.pool(enc1)
        )

        enc3 = self.encoder3(
            self.pool(enc2)
        )

        enc4 = self.encoder4(
            self.pool(enc3)
        )

        # Bottleneck
        bottleneck = self.bottleneck(
            self.pool(enc4)
        )

        # Decoder + skip connections
        dec4 = self.up4(bottleneck)

        dec4 = torch.cat(
            [dec4, enc4],
            dim=1
        )

        dec4 = self.decoder4(dec4)

        dec3 = self.up3(dec4)

        dec3 = torch.cat(
            [dec3, enc3],
            dim=1
        )

        dec3 = self.decoder3(dec3)

        dec2 = self.up2(dec3)

        dec2 = torch.cat(
            [dec2, enc2],
            dim=1
        )

        dec2 = self.decoder2(dec2)

        dec1 = self.up1(dec2)

        dec1 = torch.cat(
            [dec1, enc1],
            dim=1
        )

        dec1 = self.decoder1(dec1)

        # Final prediction
        output = self.final(dec1)

        return output


if __name__ == "__main__":

    model = UNet()

    x = torch.randn(2, 3, 224, 224)

    output = model(x)

    print("Input shape :", x.shape)
    print("Output shape:", output.shape)