# Training function
# Weight initialization (important for GANs)
def weights_init(m):
  classname = m.__class__.__name__
  if classname.find('Conv') != -1:
    nn.init.normal_(m.weight.data, 0.0, 0.02)
  elif classname.find('BatchNorm') != -1:
    nn.init.normal_(m.weight.data, 1.0, 0.02)
    nn.init.constant_(m.bias.data, 0)

def train_dcgan(dataloader, num_epochs, batch_size):

  # Initialize models
  generator = Generator(latent_dim, img_ch).to(device)
  discriminator = Discriminator(img_ch).to(device)

  # Apply weight initialization
  generator.apply(weights_init)
  discriminator.apply(weights_init)
  # Loss function
  bce_loss = nn.BCELoss()
  # Optimizers
  optimizer_G = optim.Adam(generator.parameters(), lr=c.lr_G, betas=(Beta1, 0.999))
  optimizer_D = optim.Adam(discriminator.parameters(), lr=c.lr_D, betas=(Beta1, 0.999))
  # update WandB
  wandb.config.update({
      "latent_dim": c.latent_dim,
      "batch_size": c.batch_size,
      "img_ch": c.img_ch,
      "img_size": c.img_size,
      "kernel": c.kernel,
      "Beta1": c.Beta1,
      "optimizer": "Adam",
      "lr_G": c.lr_G,
      "lr_D": c.lr_D,
      "num_epochs": c.num_epochs,
      "architecture": "Baseline DCGAN"
      })
  # wandb.watch(generator)
  # wandb.watch(discriminator)

  # Fixed noise for visualization
  fixed_noise = torch.randn(64, latent_dim).to(device)
  print(f"Generator parameters: {sum(p.numel() for p in generator.parameters()):,}")
  print(f"Discriminator parameters: {sum(p.numel() for p in discriminator.parameters()):,}")

  for epoch in range(num_epochs):
    for i, (real_imgs,_) in enumerate(dataloader):
      curr_batch_size = real_imgs.size(0) # Get current batch size
      real_imgs = real_imgs.to(device)

      # Labels for real and fake images, use curr_batch_size
      real_labels = torch.ones(curr_batch_size, 1).to(device)
      fake_labels = torch.zeros(curr_batch_size, 1).to(device)

      # ---------------------
      #  Train Discriminator
      # ---------------------

      optimizer_D.zero_grad()

      # Add Gaussian noise to real images
      noise_std = 0.3  # You can adjust this
      real_imgs_noisy = real_imgs + torch.randn_like(real_imgs) * noise_std

      # Loss on real images
      real_out = discriminator(real_imgs_noisy)
      d_real_loss = bce_loss(real_out, real_labels)

      # Generate fake images
      z = torch.randn(curr_batch_size, c.latent_dim).to(device) # Use curr_batch_size for fake images
      fake_imgs = generator(z)

      # Add Gaussian noise to fake images
      fake_imgs_noisy = fake_imgs.detach() + torch.randn_like(fake_imgs) * noise_std

      # Loss on fake images
      fake_out = discriminator(fake_imgs_noisy)
      d_fake_loss = bce_loss(fake_out, fake_labels)

      # Total discriminator loss
      d_loss = d_real_loss + d_fake_loss
      d_loss.backward()
      # nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
      optimizer_D.step()

      # -----------------
      #  Train Generator
      # -----------------
      optimizer_G.zero_grad()

      # Generate fake images
      z = torch.randn(curr_batch_size, c.latent_dim).to(device) # Use curr_batch_size for fake images
      fake_imgs = generator(z)

      # Loss on fake images
      fake_out = discriminator(fake_imgs)
      g_loss = bce_loss(fake_out, real_labels)

      # Generator tries to fool discriminator
      g_loss.backward()
      # nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
      optimizer_G.step()

      # ---------------------
      #  Log metrics
      # ---------------------

      # Print progress at every batch
      if (i+1) % 469 == 0:
        print(f"[Epoch {epoch+1}/{num_epochs}]\n"
        f"[Batch {i+1}/{len(dataloader)}]\n"
        f"D_loss: {d_loss.item():.4f}\n"
        f"G_loss: {g_loss.item():.4f}\n"
        f"G_acc: {(fake_out >= 0.5).float().mean().item():.4f}\n"
        f"D_real_acc: {(real_out >= 0.5).float().mean().item():.4f}\n"
        f"D_fake_acc: {(fake_out < 0.5).float().mean().item():.4f}")
    # Log to WandB at end of epoch
    log_interval = 1  # log every 10 batches
    if (i + 1) % log_interval == 0:
      wandb.log({
          "epoch": epoch,
          "d_loss": d_loss.item(),
          "g_loss": g_loss.item(),
          "d_real_acc": (real_out >= 0.5).float().mean().item(),
          "d_fake_acc": (fake_out < 0.5).float().mean().item(),
          "g_acc": (fake_out >= 0.5).float().mean().item(),
          })

    # Generate and log images every 5 epochs
    if epoch % 5 == 0: # Generate and log images every 5 epochs and every 100 batches
      with torch.no_grad():
        fake_imgs = generator(fixed_noise).detach().cpu()

        # Denormalize images from [-1, 1] to [0, 1]
        fake_imgs = (fake_imgs + 1) / 2
        # Create grid of images
        grid = torchvision.utils.make_grid(fake_imgs,
                                           nrow=8,
                                           padding=2,
                                           normalize=False)
        # Log to WandB
        wandb.log({"generated_images": wandb.Image(grid)})
        save_image(fake_imgs[:64],f'generated_epoch_{epoch}.png', nrow=8, padding=2, normalize=False)


# Save final models
  torch.save(generator.state_dict(), "generator.pth")
  torch.save(discriminator.state_dict(), "discriminator.pth")
  wandb.save("generator.pth")
  wandb.save("discriminator.pth")

  return generator, discriminator