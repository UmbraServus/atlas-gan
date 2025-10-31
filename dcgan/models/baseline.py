# Models


# Generator Network
class Generator(nn.Module):
    def __init__(self, latent_dim, img_ch):
        super(Generator, self).__init__()

        # Starting size will be 7x7 after first layer
        self.init_size = 7

        # Linear layer to expand latent vector
        self.fc = nn.Linear(latent_dim, 128 * self.init_size**2)

        self.conv_blocks = nn.Sequential(
          nn.BatchNorm2d(128),

          # Upsample to 14x14
          nn.Upsample(scale_factor=2),
          nn.Conv2d(128, 64, c.kernel, stride=1, padding=1),
          nn.BatchNorm2d(64),
          nn.ReLU(inplace=True),

          # Upsample to 28x28
          nn.Upsample(scale_factor=2),
          nn.Conv2d(64, 32, c.kernel, stride=1, padding=1),
          nn.BatchNorm2d(32),
          nn.ReLU(inplace=True),

          # Output layer
          nn.Conv2d(32, c.img_ch, c.kernel, stride=1, padding=1),
          nn.Tanh()
          # Output range [-1, 1]
        )
    def forward(self, z):
        # z shape: (batch_size, latent_dim)
        out = self.fc(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

        # Discriminator Network
class Discriminator(nn.Module):
    def __init__(self, img_ch):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            # Input: 1x28x28
            nn.Conv2d(img_ch, 64, kernel, stride=2, padding= 1),
            nn.LeakyReLU(0.2, inplace=True),


            nn.Conv2d(64, 128, kernel, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),


            nn.Conv2d(128, 256, kernel, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            )
        self.adv_layer = nn.Sequential(
        # Flatten and output single value
            nn.Flatten(),
            nn.Linear(256 * 3 * 3, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        # print(f"Input shape: {img.shape}")  # (128, 1, 28, 28)

        out = self.model(img)
        # print(f"After conv layers: {out.shape}")  # (128, 128, ?, ?)

        # out_flat = out.view(out.size(0), -1)  # Flatten manually
        # print(f"After flatten: {out_flat.shape}")  # (128, ???)

        validity = self.adv_layer(out)
        return validity