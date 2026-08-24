# Initial handwriting
CORR_DIR = "fonts_samples/scrivener_words_GloriousFree-dBR6"
# Pretty handwriting
UNDIR = "fonts_samples/scrivener_words_ArianaVioleta-dz2K"
# Experiment path:
EXP_DIR = "fakes/experiment_5"
# Experiment name:
EXP_NAME = "exp_5"

# Number of epochs
N_EPOCHS = 2000
# Learning rate
LR = 0.00001
# Noise vector dimension
Z_DIM = 256
# Weight of the gradient penalty
C_LAMBDA = 10
# The number of images per forward/backward pass
BATCH_SIZE = 2

# Number of times to update the critic per generator update
CRIT_REPEATS = 5

# Image weight
IMAGE_W = 100
# Image height
IMAGE_H = 40
# How often to display/visualize the images
DISPLAY_STEP = 10
# Step to change the pretty style handwriting
CHANGE_IMG_REF = 1000
