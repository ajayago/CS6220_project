# Model architecture (WGAN-GP conditioned with tissue embeddings) and training hyperparameters
EPOCHS_TO_TRACK = []
PWD = '../src/baselines/gan'
CHECKPOINT_DIR = PWD+'/checkpoints'
LOG_DIR = PWD+'/logs/'
FIG_DIR = PWD+'/figures'

# BEST config 
CONFIG_WGAN = dict()
# Architecture
CONFIG_WGAN['latent_dim'] = 128
CONFIG_WGAN['x_dim'] = 7776
CONFIG_WGAN['embedded_dim'] = 2
CONFIG_WGAN['numerical_dim'] = 3
CONFIG_WGAN['hidden_dim1_g'] = 256
CONFIG_WGAN['hidden_dim2_g'] = 512
CONFIG_WGAN['hidden_dim3_g'] = 1024
CONFIG_WGAN['hidden_dim1_d'] = 512
CONFIG_WGAN['hidden_dim2_d'] = 256
CONFIG_WGAN['hidden_dim3_d'] = None
CONFIG_WGAN['output_dim'] = 1

CONFIG_WGAN['vocab_size'] = 24
CONFIG_WGAN['categorical'] = 'tissue_type'
# Training
CONFIG_WGAN['activation'] = 'leaky_relu'
CONFIG_WGAN['negative_slope'] = 0.05
CONFIG_WGAN['optimizer'] = 'adam'
CONFIG_WGAN['lr_g'] = 0.0001
CONFIG_WGAN['lr_d'] = 0.001
CONFIG_WGAN['batch_size'] = 256
CONFIG_WGAN['epochs'] = 800
CONFIG_WGAN['iters_critic'] = 5
CONFIG_WGAN['lambda_penalty'] = 10
CONFIG_WGAN['nb_principal_components'] = 2000
CONFIG_WGAN['prob_success'] = 0
CONFIG_WGAN['norm_scale'] = 0.5
# Logs
CONFIG_WGAN['epochs_checkpoints'] = EPOCHS_TO_TRACK
CONFIG_WGAN['checkpoint_dir'] = CHECKPOINT_DIR
CONFIG_WGAN['log_dir'] = LOG_DIR
CONFIG_WGAN['fig_dir'] = FIG_DIR
CONFIG_WGAN['step'] = 100
CONFIG_WGAN['pca_applied'] = True
