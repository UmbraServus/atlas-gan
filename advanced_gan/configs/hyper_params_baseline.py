# hyperparameters

latent_dim = 128
lr_G = 0.0009
lr_D = 0.00005
img_ch = 1
img_size = 28
kernel = 4
Beta1 = 0.5
batch_size = 128
epochs = 50
device = "cuda"
if torch.cuda.is_available():
  print("Using GPU")
else:
  print("Using CPU")