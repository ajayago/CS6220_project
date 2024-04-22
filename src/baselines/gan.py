# From https://forge.ibisc.univ-evry.fr/alacan/GANs-for-transcriptomics/blob/86918cf88eb1e546282c48380c8d98d647d937bc/src/models/gan/model.py
# Imports
import os
import time as t
import datetime
import random
import torch
import torch.nn as nn
import torch.optim
import torch.nn.functional as F
from utils import *

# SEED
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)
random.seed(42)


class Generator(nn.Module):
    """
    Generator class
    """

    def __init__(self, latent_dim: int,
                 embedd_dim: int,
                 numerical_dim: int,
                 hidden_dim1: int,
                 hidden_dim2: int,
                 output_dim: int = None,
                 hidden_dim3: int = None,
                 vocab_size: int = None,
                 activation_func: str = 'relu',
                 negative_slope: float = 0.1):
        super().__init__()

        """Parameters:
            latent_dim (int): dimension of latent noise vector z.
            embedd_dim (int): dimension of categorical embedded variables (cancer types or tissue types).
            numerical_dim (int): dimension of numerical variables.
            hidden_dim1 (int): dimension of 1st hidden layer.
            hidden_dim2 (int): dimension of 2nd hidden layer.
            hidden_dim3 (int): dimension of 3rd hidden layer.
            output_dim (int): dimension of generated data (nb of genes).
            vocab_size (int): size of vocabulary for cancer/tissue embeddings.
            activation_func (str): activation function between hidden layers (either 'relu' or 'leaky_relu'). Default 'relu'.
            negative_slope (float): slope of leaky relu activation. Default 0.1.
        """

        # Dimensions
        self.vocab_size = vocab_size
        self.embedd_dim = embedd_dim
        self.latent_dim = latent_dim
        # We concatenate conditional covariates with gene expression variables
        self.input_dim = latent_dim #+ numerical_dim + self.embedd_dim * self.vocab_size
        # Layers params
        self.output_dim = output_dim
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        self.hidden_dim3 = hidden_dim3

        # Embedding layers for tissue type/cancer types
        self.embedding = nn.Embedding(self.vocab_size, self.embedd_dim)

        # Activation function
        self.negative_slope = negative_slope
        if activation_func.lower() == 'leaky_relu':
            self.activation_func = nn.LeakyReLU(self.negative_slope)
        elif activation_func.lower() == 'relu':
            self.activation_func = nn.ReLU()

        # Linear layers with batch norm in GAN
        self.proj1 = nn.Linear(self.input_dim, self.hidden_dim1)
        self.bn1 = nn.BatchNorm1d(self.hidden_dim1)
        self.proj2 = nn.Linear(self.hidden_dim1, self.hidden_dim2)
        self.bn2 = nn.BatchNorm1d(self.hidden_dim2)
        self.proj3 = nn.Linear(self.hidden_dim2, self.hidden_dim3)
        self.bn3 = nn.BatchNorm1d(self.hidden_dim3)
        # Output (unconstrained, i.e no TanH/Sigmoid)
        self.proj_output = nn.Linear(self.hidden_dim3, self.output_dim)

    def forward(self, x: torch.tensor):
        """ Main function to generate from input noise vector.
        ----
        Parameters:
            x (torch.tensor): input noise vector.
            categorical (torch.tensor): input categorical conditions to embed.
        Returns:
            (torch.tensor): generated data.
        """
        # cat = self.embedding(
        #     categorical)  # Embedding for tissue types/cancer types
        # # Concatenate all variables (expression data and conditions)
        # x = torch.cat((x, cat.flatten(start_dim=1)), 1)
        # Linear + activation function
        x = x.float()
        x = self.activation_func(self.bn1(self.proj1(x)))
        x = self.activation_func(self.bn2(self.proj2(x)))
        x = self.activation_func(self.bn3(self.proj3(x)))
        # Output (unconstrained)
        x = self.proj_output(x)

        return x


class Discriminator(nn.Module):
    """ Critic class
    """

    def __init__(self, x_dim: int,
                 embedd_dim: int,
                 numerical_dim: int,
                 hidden_dim1: int,
                 hidden_dim2: int,
                 output_dim: int,
                 vocab_size: int = None,
                 activation_func: str = 'relu',
                 negative_slope: float = 0.1):
        super().__init__()
        """Parameters:
            x_dim (int): dimension of data (nb of genes).
            embedd_dim (int): dimension of categorical embedded variables.
            numerical_dim (int): dimension of numerical variables.
            hidden_dim1 (int): dimension of 1st hidden layer.
            hidden_dim2 (int): dimension of 2nd hidden layer.
            output_dim (int): output dimension.
            vocab_size (int): vocabulary size for cancer/tissue embeddings.
            activation_func (str): activation function between hidden layers (either 'relu' or 'leaky_relu'). Default 'relu'.
            negative_slope (float): slope of leaky relu activation. Default 0.1.
        """
        # Layers params
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        self.output_dim = output_dim
        # Dimensions
        self.vocab_size = vocab_size
        self.embedd_dim = embedd_dim
        # We concatenate conditional covariates with gene expression variables
        self.input_dim = x_dim #+ numerical_dim + self.embedd_dim * self.vocab_size
        # Embedding layers for tissue type/cancer types
        self.embedding = nn.Embedding(self.vocab_size, self.embedd_dim)

        # Activation function
        self.negative_slope = negative_slope
        if activation_func.lower() == 'leaky_relu':
            self.activation_func = nn.LeakyReLU(self.negative_slope)
        elif activation_func.lower() == 'relu':
            self.activation_func = nn.ReLU()

        # Linear layers
        self.proj1 = nn.Linear(self.input_dim, self.hidden_dim1)
        self.proj2 = nn.Linear(self.hidden_dim1, self.hidden_dim2)
        # Output layer
        self.activation_output = nn.Sigmoid()

    def forward(self, x: torch.tensor):
        """ Main function to discriminate input data.
        ----
        Parameters:
            x (torch.tensor): input data.
            categorical (torch.tensor): input conditional categorical data.
        Returns:
            (torch.tensor): predicted class (true = 1, fake = 0)
        """
        # cat = self.embedding(
        #     categorical)  # Embedding for tissue types/cancer types
        # # Concatenate all variables (expression data and conditions)
        # x = torch.cat((x, cat.flatten(start_dim=1)), 1)
        x = x.float()
        x = self.activation_func(self.proj1(x))
        x = self.activation_func(self.proj2(x))
        return self.activation_output(x)


class GAN(object):
    """
    Vanilla Conditional GAN class.
    """

    def __init__(self, config: dict):
        """ Parameters:
            config (dict): model architecture dictionary.
        """
        # Set architecture
        self.latent_dim = config['latent_dim']
        self.x_dim = config['x_dim']
        self.embedded_dim = config['embedded_dim']
        self.numerical_dim = config['numerical_dim']
        self.hidden_dim1_g = config['hidden_dim1_g']
        self.hidden_dim2_g = config['hidden_dim2_g']
        self.hidden_dim3_g = config['hidden_dim3_g']
        self.hidden_dim1_d = config['hidden_dim1_d']
        self.hidden_dim2_d = config['hidden_dim2_d']
        self.output_dim = config['output_dim']
        self.vocab_size = config['vocab_size']
        self.categorical = config['categorical']
        self.epochs_checkpoints = config['epochs_checkpoints']
        self.nb_principal_components = config["nb_principal_components"]

        #  use gpu if available
        self.device = config['device']

        # Set generator and critic models with init arguments
        self.G = Generator(
            self.latent_dim,
            self.embedded_dim,
            self.numerical_dim,
            self.hidden_dim1_g,
            self.hidden_dim2_g,
            output_dim=self.x_dim,
            vocab_size=self.vocab_size,
            hidden_dim3=self.hidden_dim3_g,
            activation_func=config['activation'],
            negative_slope=config['negative_slope']).to(
            self.device)

        self.D = Discriminator(
            self.x_dim,
            self.embedded_dim,
            self.numerical_dim,
            self.hidden_dim1_d,
            self.hidden_dim2_d,
            self.output_dim,
            vocab_size=self.vocab_size,
            activation_func=config['activation'],
            negative_slope=config['negative_slope']).to(
            self.device)

        self.checkpoint_prefix = None
        self.G_path = None
        self.D_path = None

        # Callbacks objects
        self.LossTracker = TrackLoss()

        # Init One Hot encoder
        self.tissues_one_hot_encoder = []

        # Whether PCA reduction has been applied on data
        self.pca_applied = config['pca_applied']

    def train(self, TrainDataLoader,
              ValDataLoader,
              z_dim: int,
              epochs: int,
              categorical: str = None,
              step: int = 5,
              verbose: bool = True,
              checkpoint_dir: str = './checkpoints',
              log_dir: str = './logs/',
              fig_dir: str = './figures',
              prob_success: int = 0,
              norm_scale: float = 0.5,
              optimizer: str = 'rms_prop',
              lr_g: float = 5e-4,
              lr_d: float = 5e-4,
              nb_principal_components: int = 2000,
              config: dict = None,
              hyperparameters_search: bool = False):
        """
        Main train function to train full model.
        ----
        Parameters:
            TrainDataLoader (pytorch loader): train data loader with expression data, covariates and labels.
            ValDataLoader (pytorch loader): validation data loader.
            z_dim (int): latent noise dimension.
            epochs (int): number of training epochs.
            categorical (str): whether condition categorical variables are 'tissue_type' or 'cancer_type'.
            step (int): each step to compute metrics.
            verbose (bool): print training callbacks (default True).
            checkpoint_dir (str): where to save model weights.
            log_dir (str): path where to save logs.
            fig_dir (str): path where to store figures.
            patience (int): number of epochs without improving after which the training is halted (default 30).
            prob_success (int): default 0.
            norm_scale (float): default 0.5.
            optimizer (str): either 'rms_prop' or 'adam'. Default 'rms_prop'.
            lr_g (float): generator learning rate. Default 5e-4.
            lr_d (float): discriminator learning rate. Default 5e-4.
            nb_principal_components (int): dimension of principal components reduction for analysis. Default 2000.
            config (dict): dictionnary of model configuration.
            hyperparameters_search (bool): whether training is performed for a search and weights should be saved in same path every run. Default False.
        """
        # Safety check
        assert z_dim == self.latent_dim, "Please assign the correct latent dim for 'z_dim' argument."

        # Log training duration
        self.t_begin = t.time()
        self.epoch_times = []

        # Init optimizers and directories
        self.init_train(log_dir,
                        checkpoint_dir,
                        fig_dir,
                        optimizer=optimizer,
                        lr_g=lr_g,
                        lr_d=lr_d,
                        epochs=epochs,
                        trainloader=TrainDataLoader,
                        nb_principal_components=nb_principal_components)

        # Write configuration and architecture in log
        write_config(file_path=self.path_log, config=config)

        # Init training metrics history
        all_epochs_val_score = []
        all_epochs_fid_score = []
        all_epochs_prec_recall = []
        all_epochs_prec_recall_train = []
        all_aats = []
        all_aats_train = []
        all_aats_train_PC = []
        all_discrep = []
        all_discrep_real = []
        all_epochs_prec_recall_train_PC = []

        # Set categorical variable (indicates if we perform embeddings of
        # tissue type or cancer type)
        self.categorical = categorical

        for epoch in range(1, epochs + 1):
            self.epoch = epoch

            self.start_epoch_time = t.time()

            # Init epoch losses
            epoch_disc_loss = 0
            epoch_gen_loss = 0

            # Loss
            criterion = nn.BCELoss()

            # Load batches of expression data, numerical covariates and encoded
            # categorical covariates
            # for batch, age, gender, _, encoded_tissue_types, labels in TrainDataLoader:
            for batch, _ in TrainDataLoader:

                # To GPU, else CPU
                batch = batch.to(self.device)

                # Numerical and categorical covariates to GPU, else CPU
                # age = age.to(self.device)
                # gender = gender.to(self.device)

                # if self.categorical == 'tissue_type':
                #     _ = []  # Save memory
                #     batch_categorical = encoded_tissue_types.to(self.device)
                #     labels = labels.to(self.device)
                #     batch_numerical = torch.cat(
                #         (age.reshape(
                #             batch.shape[0], 1), gender.reshape(
                #             batch.shape[0], 1), labels.reshape(
                #             batch.shape[0], 1)), 1)

                ################ Train discriminator ################
                # Reset gradients back to 0
                self.D_optimizer.zero_grad()

                # Get random latent variables z
                batch_z = torch.normal(
                    0,
                    1,
                    size=(
                        batch.shape[0],
                        z_dim),
                    device=self.device)

                # Generator forward pass with concatenated variables
                # gen_outputs = self.G(
                #     torch.cat(
                #         (batch_z, batch_numerical), 1), batch_categorical)
                gen_outputs = self.G(batch_z)

                # Perform random augmentations for stability
                NB_GENES = gen_outputs.shape[1]
                BATCH_SIZE = batch_z.shape[0]
                augmentations = torch.distributions.binomial.Binomial(
                    total_count=1, probs=prob_success).sample(
                    torch.tensor(
                        [BATCH_SIZE])).to(
                    gen_outputs.device)
                gen_outputs = gen_outputs + augmentations[:, None] * torch.normal(
                    0, norm_scale, size=(NB_GENES,), device=gen_outputs.device)

                # Forward pass on discriminator with concatenated variables
                # disc_real = self.D(
                #     torch.cat(
                #         (batch, batch_numerical), 1), batch_categorical)
                disc_real = self.D(batch)

                # Loss on true data
                true_disc_loss = criterion(
                    disc_real, torch.ones_like(disc_real))

                # Loss on fake data -add .detach() here (we train only the
                # discriminator)
                # disc_fake = self.D(
                #     torch.cat(
                #         (gen_outputs.detach(),
                #          batch_numerical),
                #         1),
                #     batch_categorical)
                disc_fake = self.D(gen_outputs)
                fake_disc_loss = criterion(
                    disc_fake, torch.zeros_like(disc_fake))

                # Total loss
                disc_loss = true_disc_loss + fake_disc_loss
                disc_loss.backward()
                self.D_optimizer.step()

                epoch_disc_loss += disc_loss.detach().item()

                # Track loss components
                self.LossTracker({"disc_loss_batch": disc_loss.detach().item(),
                                  "real_loss": true_disc_loss.detach().item(),
                                  "fake_loss": fake_disc_loss.detach().item()})

                ################ Train Generator ################
                # Reset gradients back to 0
                self.G_optimizer.zero_grad()

                # Same forward pass on discriminator
                # disc_outputs = self.D(
                #     torch.cat(
                #         (gen_outputs, batch_numerical), 1), batch_categorical)
                gen_outputs = gen_outputs.detach()
                disc_outputs = self.D(gen_outputs)

                # Gen loss with fake labels as ones to force the discriminator
                # to classify them as true
                gen_loss = criterion(
                    disc_outputs, torch.ones_like(disc_outputs))

                # Backpropagate
                gen_loss.backward()

                # Track loss components
                self.LossTracker({"g_loss_batch": gen_loss.detach().item()})

                # Update parameters (could add clipping)
                self.G_optimizer.step()

                epoch_gen_loss += gen_loss.detach().item()

            # Store all losses
            self.LossTracker({"disc_loss_epoch": epoch_disc_loss /
                              len(TrainDataLoader), "g_loss_epoch": epoch_gen_loss /
                              len(TrainDataLoader)})

            self.end_epoch_time = t.time()
            self.epoch_time = self.end_epoch_time - self.start_epoch_time

            # Check whether to print UMAP
            if self.epoch in self.epochs_checkpoints:
                # Generate data according to validation covariates
                x_real, x_gen, _, _, tissues, cancer_labels = self.real_fake_data(
                    ValDataLoader, z_dim, categorical, return_labels=True)
                self.UMAP_Printer(
                    x_real,
                    x_gen,
                    tissues_labels=tissues,
                    cancer_labels=cancer_labels,
                    epoch=self.epoch)

            ########## Epochs checkpoints ####################
        #     if epoch % step == 0:

        #         # Validation
        #         x_real, x_gen = self.real_fake_data(
        #             ValDataLoader, z_dim, categorical)
        #         all_epochs_val_score, all_epochs_prec_recall, all_aats, all_discrep, all_discrep_real = epoch_checkpoint_val(x_real, x_gen,
        #                                                                                                                      list_val=all_epochs_val_score,
        #                                                                                                                      list_prec_recall=all_epochs_prec_recall,
        #                                                                                                                      list_aats=all_aats,
        #                                                                                                                      list_discrep=all_discrep,
        #                                                                                                                      list_discrep_real=all_discrep_real,
        #                                                                                                                      epoch=epoch,
        #                                                                                                                      pca_applied=self.pca_applied,
        #                                                                                                                      nb_principal_components=self.nb_principal_components,
        #                                                                                                                      pca_obj=self.pca)

        #         # Train
        #         all_epochs_fid_score, all_epochs_prec_recall_train, all_epochs_prec_recall_train_PC, all_aats_train, all_aats_train_PC, all_discrep, all_discrep_real = epoch_checkpoint_train(x_real, x_gen,
        #                                                                                                                                                                                        device_pretrained_cls=self.device,
        #                                                                                                                                                                                        list_fid=all_epochs_fid_score,
        #                                                                                                                                                                                        list_prec_recall_train=all_epochs_prec_recall_train,
        #                                                                                                                                                                                        list_prec_recall_train_PC=all_epochs_prec_recall_train_PC,
        #                                                                                                                                                                                        list_aats_train=all_aats_train,
        #                                                                                                                                                                                        list_aats_train_pc=all_aats_train_PC,
        #                                                                                                                                                                                        list_discrep=all_discrep,
        #                                                                                                                                                                                        list_discrep_real=all_discrep_real,
        #                                                                                                                                                                                        pca_applied=self.pca_applied,
        #                                                                                                                                                                                        nb_principal_components=self.nb_principal_components,
        #                                                                                                                                                                                        pca_obj=self.pca)

        #         # Epoch duration
        #         self.epoch_times.append(self.epoch_time)

        #         # Write logs
        #         watch_dict = {'epoch': epoch,
        #                       'gen_loss': epoch_gen_loss,
        #                       'disc_loss': epoch_disc_loss,
        #                       'val_score': all_epochs_val_score[-1],
        #                       'precision_train': round(all_epochs_prec_recall_train[-1][0], 3),
        #                       'recall_train': round(all_epochs_prec_recall_train[-1][1], 3),
        #                       'precision_val': round(all_epochs_prec_recall[-1][0], 3),
        #                       'recall_val': round(all_epochs_prec_recall[-1][1], 3),
        #                       'precision_train_pc': round(all_epochs_prec_recall_train_PC[-1][0], 3),
        #                       'recall_train_pc': round(all_epochs_prec_recall_train_PC[-1][1], 3),
        #                       'AAts_val': round(all_aats[-1], 3),
        #                       'AAts_train': round(all_aats_train[-1], 3),
        #                       'AAts_train_pc': round(all_aats_train_PC[-1], 3),
        #                       'discrepancy': round(all_discrep[-1], 3),
        #                       'discrepancy_real': round(all_discrep_real[-1], 3),
        #                       'nb_samples': len(TrainDataLoader)}

        #         write_log(file_path=self.path_log, metrics_dict=watch_dict)

        #         if verbose:
        #             print_func(watch_dict)

        # ############### End of training ##############################

        # Print training time
        print_training_time(self.t_begin)

        # Save last weigths for G and D
        save_weights(
            self.G,
            self.D,
            self.G_path,
            self.D_path,
            hyperparameters_search)

        # Save history
        save_history(all_epochs_val_score, self.history_val_score_path)
        save_history(all_epochs_fid_score, self.history_fid_path)
        save_history(all_epochs_prec_recall, self.history_prec_recall_path)
        save_history(all_epochs_prec_recall_train, self.history_prec_recall_train_path)
        save_history( all_epochs_prec_recall_train_PC, self.history_prec_recall_train_PC_path)
        save_history(all_aats, self.history_aats_path)
        save_history(all_aats_train, self.history_aats_train_path)
        save_history(all_aats_train_PC, self.history_aats_train_PC_path)
        save_history(all_discrep, self.history_discrepancy_path)
        save_history(self.epoch_times, self.epoch_times_path)

    def init_train(
            self,
            log_dir: str,
            checkpoint_dir: str,
            fig_dir: str,
            optimizer: str = 'rms_prop',
            lr_g: float = 5e-4,
            lr_d: float = 5e-4,
            epochs: int = None,
            trainloader=None,
            nb_principal_components: int = None):
        """
        Training initialization: init directories, callbacks and PCA.
        ----
        Parameters:
            log_dir (str): path where to save logs.
            checkpoint_dir (str): where to save model weights.
            fig_dir (str): path where to store figures.
            optimizer (str): either 'rms_prop' or 'adam'. Default 'rms_prop'.
            lr_g (float): generator learning rate. Default 5e-4.
            lr_d (float): discriminator learning rate. Default 5e-4.
            epochs (int): number of training epochs.
            trainloader (pytorch loader): train data loader with expression data, covariates and labels.
            nb_principal_components (int): dimension of principal components reduction for analysis. Default 2000.
        """
        # Optimizers
        if optimizer.lower() == 'rms_prop':
            # Use the RMSProp version of gradient descent with small learning
            # rate and no momentum (e.g. 0.00005).
            self.G_optimizer = torch.optim.RMSprop(
                self.G.parameters(), lr=lr_g)
            self.D_optimizer = torch.optim.RMSprop(
                self.D.parameters(), lr=lr_d)
        elif optimizer.lower() == 'adam':
            self.G_optimizer = torch.optim.Adam(
                self.G.parameters(), lr=lr_g, betas=(.9, .99))
            self.D_optimizer = torch.optim.Adam(
                self.D.parameters(), lr=lr_d, betas=(.9, .99))

        # Set up logs and checkpoints
        current_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        self.log_dir = log_dir + current_time

        # Make dir if it does not exist
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        if not os.path.exists(self.log_dir):
            os.mkdir(self.log_dir)
            print("Directory '%s' created" % self.log_dir)

        # training history path
        self.history_gen_path = self.log_dir + '/train_gen_loss.npy'
        self.history_disc_path = self.log_dir + '/train_disc_loss.npy'
        self.history_val_score_path = self.log_dir + '/val_score.npy'
        self.history_fid_path = self.log_dir + '/fid_scores.npy'
        self.history_prec_recall_path = self.log_dir + '/precision_recall.npy'
        self.history_prec_recall_train_path = self.log_dir + '/precision_recall_train.npy'
        self.history_prec_recall_train_PC_path = self.log_dir + \
            '/precision_recall_train_pc.npy'
        self.history_aats_path = self.log_dir + '/aats.npy'
        self.history_aats_train_path = self.log_dir + '/aats_train.npy'
        self.history_aats_train_PC_path = self.log_dir + '/aats_train_pc.npy'
        self.history_discrepancy_path = self.log_dir + '/discrepancy.npy'
        self.epoch_times_path = self.log_dir + '/epoch_durations.npy'

        # Init log path
        self.path_log = self.log_dir + '/logs.txt'

        # Define checkpoints path
        self.checkpoint_dir = os.path.join(checkpoint_dir, current_time)
        # Make dir if it does not exist
        if not os.path.exists(checkpoint_dir):
            os.mkdir(checkpoint_dir)
        if not os.path.exists(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)
            print("Directory '%s' created" % self.checkpoint_dir)

        # Create figures folder
        self.fig_dir = os.path.join(fig_dir, current_time)
        # Make dir if it does not exist
        if not os.path.exists(fig_dir):
            os.mkdir(fig_dir)
        if not os.path.exists(self.fig_dir):
            os.mkdir(self.fig_dir)
            print("Directory '%s' created" % self.fig_dir)

        # Initialize the paths where to store models
        self.D_path = self.checkpoint_dir + '/_disc.pt'
        self.G_path = self.checkpoint_dir + '/_gen.pt'

        # Initialize the tracking object for loss components
        self.LossTracker = TrackLoss(
            path=self.log_dir +
            '/train_history.npy',
            nb_epochs=epochs)

        # Initialize wieghts tracker to save generator weights every 5 epochs
        #self.WeightsTracker = TrackWeights(path =self.checkpoint_dir, epochs_to_track=epochs_to_track)

        # Initialize UMAP printer
        self.UMAP_Printer = VisuUMAPEpoch(
            path=self.fig_dir, on_projections=False)

        # Init PCA on real training data
        if not self.pca_applied:
            self.nb_principal_components = nb_principal_components
            self.init_pca(trainloader, nb_components=nb_principal_components)

    def init_pca(self, trainloader=None, nb_components: int = 2000):
        """
        Init PCA on training data.
        ----
        Parameters:
            trainloader (pytorch loader): train data loader with expression data, covariates and labels.
            nb_components (int): dimension of principal components reduction for analysis. Default 2000.
        """
        # Initialize eigenvectors
        print("Performing PCA on training data")
        # Retrieve data to fit the algorithm
        train_data = []
        for x, _, _, _, _, _ in trainloader:
            train_data.append(x)
        train_data = torch.vstack((train_data)).numpy()
        # Save PCA object with singular vectors and transform function
        self.pca = perform_pca(
            data=train_data,
            nb_principal_components=nb_components,
            return_eigen_vectors=False)
        # Init projections errors
        self.err_proj = {i: [] for i in range(nb_components)}
        print("PCA finished. Init singular vectors saved.")

    def real_fake_data(
            self,
            ValDataLoader,
            z_dim: int,
            categorical: str = 'tissue_type',
            return_labels: bool = False):
        """
        Returns real data and generated data.
        ----
        Parameters:
            ValDataLoader (pytorch loader): data loader with gene expression data, covariates and labels.
            z_dim (int): latent noise vector dimension.
            categorical (str): condition categorcial variables, whether 'tissue_type' or 'cancer_type'. Default 'tissue_type'.
            return_labels (bool): whether to return real and generated data with respective condition labels. Default False.
        Returns:
            tuple of real and generated data as numpy arrays.
        """

        x_gen = []
        x_real = []
        all_age = []
        all_gender = []
        all_tissues = []
        all_labels = []
        # self.G.eval() # Evaluation mode

        with torch.no_grad():
            for batch, _ in ValDataLoader:
                # To GPU, else CPU
                batch = batch.to(self.device)
                # age = age.to(self.device)
                # gender = gender.to(self.device)

                # if categorical == 'tissue_type':
                #     _ = []  # Save memory
                #     labels = labels.to(self.device)
                #     batch_categorical = encoded_tissue_types.to(self.device)
                #     batch_numerical = torch.cat(
                #         (age.reshape(
                #             batch.shape[0], 1), gender.reshape(
                #             batch.shape[0], 1), labels.reshape(
                #             batch.shape[0], 1)), 1)

                # Get random latent variables z
                batch_z = torch.normal(
                    0,
                    1,
                    size=(
                        batch.shape[0],
                        z_dim),
                    device=self.device)

                # Generator forward pass with concatenated variables
                gen_inputs = batch_z #torch.cat((batch_z, batch_numerical), 1)
                gen_outputs = self.G(gen_inputs)#, batch_categorical
                x_gen.append(gen_outputs.cpu())
                x_real.append(batch.cpu())

                # if return_labels:
                #     all_age.append(age.detach().cpu())
                #     all_gender.append(gender.detach().cpu())
                #     all_tissues.append(encoded_tissue_types.detach().cpu())
                #     all_labels.append(labels.detach().cpu())

        # Concatenate and to array
        x_gen = torch.cat(x_gen, 0).detach().numpy()
        # Loader returns tensors (but on CPU directly)
        x_real = torch.cat(x_real, 0).detach().numpy()

        if return_labels:
            all_age = torch.cat((all_age), axis=0)
            all_gender = torch.cat((all_gender), axis=0)
            all_tissues = torch.cat((all_tissues), axis=0)
            all_labels = torch.cat((all_labels), axis=0)
            # Back to tissues names
            all_tissues = self.tissues_one_hot_encoder.inverse_transform(
                all_tissues.numpy())
            all_tissues = all_tissues.reshape(len(all_tissues))

            return x_real, x_gen, all_age.numpy(
            ), all_gender.numpy(), all_tissues, all_labels.numpy()

        elif not return_labels:
            return x_real, x_gen
