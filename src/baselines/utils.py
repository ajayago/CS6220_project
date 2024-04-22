# https://forge.ibisc.univ-evry.fr/alacan/GANs-for-transcriptomics/blob/86918cf88eb1e546282c48380c8d98d647d937bc/src/models/utils.py
"""

Tool box for training.

"""

# Imports
import os
import sys
import numpy as np
import torch
import time as t
import copy
from sklearn.decomposition import PCA
#from src.models.utils_visualization import *

# Metrics
#sys.path.append(os.path.abspath("../metrics/"))
#from correlation_score import gamma_coeff_score
#from frechet import compute_frechet_distance_score
#from aats import compute_AAts
#from precision_recall import get_precision_recall

def wasserstein_loss(y_true: torch.tensor, y_pred: torch.tensor):
    """
    Returns Wasserstein loss (product of real/fake labels and critic scores on real or fake data)
    ----
    Parameters:
        y_true (torch.tensor): true labels (either real or fake)
        y_pred (torch.tensor): critic scores on real or fake data
    Returns:
        (torch.tensor): mean product of real labels and critic scores
    """
    return torch.mean(y_true * y_pred)


def generator_loss(fake_score: torch.tensor):
    """
    Returns generator loss i.e the negative scores of the critic on fake data.
    ----
    Parameters:
        fake_score (torch.tensor): critic scores on fake data
    Returns:
        (torch.tensor): generator loss"""

    return wasserstein_loss(-torch.ones_like(fake_score), fake_score)


def discriminator_loss(real_score: torch.tensor, fake_score: torch.tensor):
    """
    Compute and return the wasserstein loss of critic scores on real and fake data i.e: wassertstein_loss = mean(-score_real) + mean(score_fake)
    ----
    Parameters:
        real_score (torch.tensor): critic scores on real data
        fake_score (torch.tensor): critic scores on fake data
    Returns:
        (torch.tensor): wasserstein loss
    """
    real_loss = wasserstein_loss(-torch.ones_like(real_score), real_score)
    fake_loss = wasserstein_loss(torch.ones_like(fake_score), fake_score)

    return real_loss, fake_loss


def load_model(
        model: torch.nn.Module,
        path: str = None,
        location: str = "cuda:0"):
    """
    Loading previously saved model.
    ----
    Parameters:
        model (torch.nn.module): Pytorch model object to load.
        path (str): path to retrieve model weights.
        location (str): device (GPU/CPU) where to load model.
    Return:
        Loaded Pytorch model."""

    assert path is not None, "Please provide a path to load the Generator from."
    try:
        # Load model
        model.load_state_dict(torch.load(path, map_location=location))
        print('Model loaded.')
        return model
    except FileNotFoundError:  # if no model saved at given path
        print(f"No previously saved model at given path {path}.")


def print_training_time(t_begin):
    """
    Compute and print training time in seconds.
    """
    t_end = t.time()
    time_sec = t_end - t_begin
    print(
        f'Time of training: {round(time_sec, 4)} sec = {round(time_sec/60, 4)} minute(s) = {round(time_sec/3600, 4)} hour(s)')


def save_weights(G, D, G_path: str, D_path: str,
                 hyperparameters_search: bool = False):
    """
    Save current weights at model checkpoint path when called.
    ----
    Parameters:
        G: generator.
        D: dicriminator.
        G_path (str): path to store generator.
        D_path (str): path to store discriminator.
        hyperparameters_search (bool): whether current training is performed for a search.
    """
    if hyperparameters_search:
        # The same storing path is used for a search.
        G_path = './checkpoints/search_gen.pt'
        D_path = './checkpoints/search_disc.pt'
    torch.save(G.state_dict(), G_path)
    torch.save(D.state_dict(), D_path)

    print(
        f"--------------------\nDiscriminator saved at {D_path} and generator saved at {G_path}.")


def save_history(metric_history, metric_path: str):
    """
    Save metric history as numpy object.
    ----
    Parameters:
        metric_history: list/dict/np.array of training metric history
        metric path (str): path where to store metric history
    """
    # Save metrics history
    np.save(metric_path, metric_history)


def perform_pca(
        nb_principal_components: int = 2000,
        data=None,
        return_eigen_vectors=False,
        solver: str = 'full'):
    """
    Perform PCA on input data. Built on scikit-learn PCA class object.
    ----
    Parameters:
        nb_principal_components (int): number of principal components to compute
        data: input data used to perform PCA
        return_eigen_vectors (bool): whether to return eigen vectors of the first principal components (default False)
        solver (str): solver used by Scikit-learn (default `full`)
    Returns:
        eigen vectors if return_eigen_vectors on True, else the pca object
    """
    # Perform PCA on training data
    pca = PCA(
        svd_solver=solver,
        n_components=nb_principal_components,
        random_state=10)
    # Fit
    pca.fit(data)

    if return_eigen_vectors:
        return pca.components_

    else:
        return pca


def PC_projections(data: torch.tensor, e_=None, pca=None, as_tensors=True):
    """
    Return projections on given eigen vectors e_sampled.
    ----
    Parameters:
        data (torch.tensor): batch of data
        e_(torch.tensor): principal component number to project on (int type)
        pca (scikit-learn object): PCA object with precomputed singular vectors
    Returns:
        proj (torch.tensor): projection of data on principal eigen vectors of interest
    """
    # Center data and dot product with eigen vector to get projection (as done
    # in Scikit-learn docu)
    means = pca.mean_
    if as_tensors:
        proj = torch.matmul(
            data -
            torch.from_numpy(means).to(
                data.device),
            torch.from_numpy(
                pca.components_[
                    e_,
                    :].T).to(
                device=data.device))

    else:
        proj = np.matmul(data - means, pca.components_[e_, :].T)

    return proj


def metrics_checkpoints_val(
        x_real,
        x_fake,
        list_val,
        list_prec_recall,
        list_discrep_real,
        list_discrep,
        list_aats):
    """
    Compute metrics of interest at given epochs checkpoints.
    ----
    Parameters:
        x_real (torch.tensor): real data
        x_gen (torch.tensor): generated data
        list_val (list): list of validation score values at each checkpoint. Default None.
        list_prec_recall (list): list of precision and recall values at each checkpoint. Default None.
        list_discrep (list): list of discrepancy values at each checkpoint. Default None.
        list_discrep_real (list): list of discrepancy values at each checkpoint. Default None.
        list_aats (list): list of adversarial accuracy values at each checkpoint. Default None.

    Returns:
        Updated lists of metrics.
    """
    # Compute validation score
    list_val.append(gamma_coeff_score(x_real, x_fake))

    # Precision/recall on validation data
    prec, recall = get_precision_recall(
        torch.from_numpy(x_real), torch.from_numpy(x_fake))
    list_prec_recall.append((prec, recall))

    # AAts (adversarial accuracy) on validation data
    discrepancy_tr_tr, discrepancy_tr_fk, adversarial = compute_AAts(
        real_data=x_real, fake_data=x_fake)
    list_discrep_real.append(discrepancy_tr_tr)
    list_discrep.append(discrepancy_tr_fk)
    list_aats.append(adversarial)

    return list_val, list_prec_recall, list_discrep_real, list_discrep, list_aats


def epoch_checkpoint_val(x_real, x_gen,
                         fig_dir: str,
                         list_val: list = None,
                         list_prec_recall: list = None,
                         list_aats: list = None,
                         list_discrep: list = None,
                         list_discrep_real: list = None,
                         epoch: int = None,
                         pca_applied: bool = False,
                         nb_principal_components: int = None,
                         pca_obj=None):
    """
    Compute metrics of interest and plot of PCA projections at given epochs checkpoints.
    ----
    Parameters:
        x_real (torch.tensor): real data
        x_gen (torch.tensor): generated data
        list_val (list): list of validation score values at each checkpoint. Default None.
        list_prec_recall (list): list of precision and recall values at each checkpoint. Default None.
        list_aats (list): list of adversarial accuracy values at each checkpoint. Default None.
        list_discrep (list): list of discrepancy values at each checkpoint. Default None.
        list_discrep_real (list): list of discrepancy values at each checkpoint. Default None.
        epoch (int): current epoch iteration. Default None.
        pca_applied (bool): whether PCA reduction was applied on input data. Default False.
        nb_principal_components (int): number of principal components to perform reduction. Default None.
        pca_obj: sklearn PCA object previously trained on train data. Default None.


    Returns:
        Updated lists of metrics.
    """

    # Compute epoch checkpoint metrics
    list_val, list_prec_recall, list_discrep_real, list_discrep, list_aats = metrics_checkpoints_val(
        x_real, x_gen, list_val, list_prec_recall, list_discrep_real, list_discrep, list_aats)

    return list_val, list_prec_recall, list_aats, list_discrep, list_discrep_real


def epoch_checkpoint_train(x_real, x_gen,
                           device_pretrained_cls: str = None,
                           list_fid: list = None,
                           list_prec_recall_train: list = None,
                           list_prec_recall_train_pc: list = None,
                           list_aats_train: list = None,
                           list_aats_train_pc: list = None,
                           list_discrep: list = None,
                           list_discrep_real: list = None,
                           pca_applied: bool = False,
                           nb_principal_components: int = None,
                           pca_obj=None):
    """
    Compute metrics of interest at given epochs checkpoints.
    ----
    Parameters:
        x_real (torch.tensor): real data
        x_gen (torch.tensor): generated data
        device_pretrained_cls (str): device where to load the pretrained classifier used for frechet distance scores.
        list_fid (lsit): list of frechet distances at each checkpoint. Default None.
        list_prec_recall_train (list): list of precision and recall values at each checkpoint on train data. Default None.
        list_prec_recall_train_pc (list): list of precision and recall values at each checkpoint on train data reduced by PCA. Default None.
        list_aats_train (list): list of adversarial accuracy values at each checkpoint on train data. Default None.
        list_aats_train_pc (list): list of adversarial accuracy values at each checkpoint on train data reduced by PCA. Default None.
        list_discrep (list): list of discrepancy values at each checkpoint. Default None.
        list_discrep_real (list): list of discrepancy values at each checkpoint. Default None.
        pca_applied (bool): whether PCA reduction was applied on input data. Default False.
        nb_principal_components (int): number of principal components to perform reduction. Default None.
        pca_obj: sklearn PCA object previously trained on train data. Default None.


    Returns:
        Updated lists.
    """

    # Compute FD score
    fid_bin = compute_frechet_distance_score(
        x_real, x_gen, task='binary', device=device_pretrained_cls)
    fid_tissue = compute_frechet_distance_score(
        x_real, x_gen, task='tissue_type', device=device_pretrained_cls)
    
    list_fid.append((fid_bin, fid_tissue))

    # Precision/recall on training data
    prec, recall = get_precision_recall(
        torch.from_numpy(x_real), torch.from_numpy(x_gen))
    list_prec_recall_train.append((prec, recall))

    # AAts (adversarial accuracy) on train data
    _, _, adversarial = compute_AAts(
        real_data=x_real[:2000], fake_data=x_gen[:2000])
    list_aats_train.append(adversarial)

    if not pca_applied:
        # Real and fake PC projections
        e = np.arange(nb_principal_components)
        x_real = PC_projections(
            x_real,
            e_sampled=e,
            pca=pca_obj,
            as_tensors=False)
        x_gen = PC_projections(
            x_gen,
            e_sampled=e,
            pca=pca_obj,
            as_tensors=False)

        # Precision/recall on PC projections
        prec, recall = get_precision_recall(
            torch.from_numpy(x_real), torch.from_numpy(x_gen))
        list_prec_recall_train_pc.append((prec, recall))

        # AAts (adversarial accuracy) on train data reduced by PCA
        _, _, adversarial = compute_AAts(
            real_data=x_real[:2000], fake_data=x_gen[:2000])
        list_aats_train_pc.append(adversarial)

    else:
        list_prec_recall_train_pc.append((0., 0.))
        list_aats_train_pc.append(0.0)

    return list_fid, list_prec_recall_train, list_prec_recall_train_pc, list_aats_train, list_aats_train_pc, list_discrep, list_discrep_real


def print_func(metrics_dict: dict = None):
    """
    Print current epoch losses and validation scores.
    ----
    Parameters:
        metrics_dict (dict): dictionnary with metrics values to print. Default None.
    """

    print(
        '-------------\n Epoch {}. Gen loss: {:.2f}. Disc loss: {:.2f}. Val score: {:.2f}. Binary FID: decile {}. Tissue type FID: decile {}.'.format(
            metrics_dict['epoch'],
            round(
                metrics_dict['gen_loss'] /
                metrics_dict['nb_samples'],
                4),
            round(
                metrics_dict['disc_loss'] /
                metrics_dict['nb_samples'],
                4),
            round(
                metrics_dict['val_score'],
                4),
            metrics_dict['fid_decile_bin'],
            metrics_dict['fid_decile_tissue']))

    print(
        'Precision train: {}. Recall train: {}. Precision train PC: {}. Recall train PC: {}. Precision val: {}. Recall val: {}. AAts train: {}. AAts val: {} AAts train PC: {} Discrepancy: {} Discrepancy real: {} \n ------------------\n'.format(
            metrics_dict['precision_train'],
            metrics_dict['recall_train'],
            metrics_dict['precision_train_pc'],
            metrics_dict['recall_train_pc'],
            metrics_dict['precision_val'],
            metrics_dict['recall_val'],
            metrics_dict['AAts_train'],
            metrics_dict['AAts_val'],
            metrics_dict['AAts_train_pc'],
            metrics_dict['discrepancy'],
            metrics_dict['discrepancy_real']))


def write_log(file_path: str = None, metrics_dict: dict = None):
    """ 
    Write training logs in given path.
    ----
    Parameters:
        file_path (str): path where to write log. Default None.
        metrics_dict (dict): dictionary with information to write. Default None.
    """

    # Open and append
    with open(file_path, 'a') as f:
        f.write(
            '\n Epoch {}. \t Gen loss: {:.2f} \t Disc loss: {:.2f} \t Val score: {:.2f} \t Binary FID: decile {} \t Tissue type FID: decile {} \n'.format(
                metrics_dict['epoch'],
                round(
                    metrics_dict['gen_loss'] /
                    metrics_dict['nb_samples'],
                    4),
                round(
                    metrics_dict['disc_loss'] /
                    metrics_dict['nb_samples'],
                    4),
                round(
                    metrics_dict['val_score'],
                    4),
                metrics_dict['fid_decile_bin'],
                metrics_dict['fid_decile_tissue']))

        f.write(
            '\t Precision train: {} \t Recall train: {} \t Precision val: {} \t Recall val: {} \t AAts val: {} \t Discrepancy val: {} \t Discrepancy real val: {} \n ------------------\n'.format(
                metrics_dict['precision_train'],
                metrics_dict['recall_train'],
                metrics_dict['precision_val'],
                metrics_dict['recall_val'],
                metrics_dict['AAts_val'],
                metrics_dict['discrepancy'],
                metrics_dict['discrepancy_real']))


def write_config(file_path: str = None, config: dict = None):
    """ 
    Write model config in given path
    ----
    Parameters:
        file_path (str): path where to write config file. Default None.
        config (dict): dictionary with configuration information. Default None.
    """

    # Open and append file
    with open(file_path, 'a') as f:
        f.write('CONFIGURATION: \n \t Z latent dim: {} \t Nb Epochs: {} \t Batch size: {} \t Nb iters critic: {} \t Lambda penalty: {} \n \t Prob success: {} \t Norm scale: {} \t Optimizer: {} \t LR Gen: {}, \t LR Disc: {}\n \t Activation func: {} \t Negative_slope: {} \t Hidden_dim1_g: {} \t Hidden_dim2_g:{} \t Hidden_dim3_g:{} \n \t Hidden_dim1_d:{} \t Hidden_dim2_d:{}\n -------------------------------------'.format(
            config['latent_dim'], config['epochs'], config['batch_size'], config['iters_critic'], config['lambda_penalty'], config['prob_success'], config['norm_scale'], config['optimizer'], config['lr_g'], config['lr_d'], config['activation'], config['negative_slope'], config['hidden_dim1_g'], config['hidden_dim2_g'], config['hidden_dim3_g'], config['hidden_dim1_d'], config['hidden_dim2_d']))


class VisuUMAPEpoch(object):
    """
    Object plotting UMAPs of real and fake data when called.
    ----
    Parameters:
        path (str): directory path where to store all figures. Default None.
        on_projections (bool): whether UMAPs are performed on data reduced in PC space. Default False.
    """

    def __init__(self, path: str = None, on_projections: bool = False):
        self.path = path
        if on_projections:
            self.path_suffixe = '/umap_projs_epoch_{}.png'
            self.marker_type = 'x'
        elif not on_projections:
            self.path_suffixe = '/umap_epoch_{}.png'
            self.marker_type = '.'

    def __call__(
            self,
            real,
            fake,
            tissues_labels=None,
            cancer_labels=None,
            epoch=None):
        """ 
        Perform 2D UMAP projections and save figure for given epoch.

        """
        full_data = np.concatenate((real, fake))
        labels = np.array(["real"] * len(real) + ["fake"] * len(fake))

        tissues_labels = np.concatenate(
            (tissues_labels.ravel(), tissues_labels.ravel()))
        # Cancer labels
        cancer_labels = np.array(['normal' if cancer_labels.ravel(
        )[q] == 0 else 'cancer' for q in range(len(cancer_labels.ravel()))])
        cancer_labels = np.concatenate((cancer_labels, cancer_labels))

        # Perform UMAP
        umap_proj = perform_umap(full_data, n_neighbors=300)

        # Count fake/real points distribution in projection
        nb_points = dist_points_in_hyper_rectangles(umap_proj[:len(real), 0], umap_proj[len(
            real):, 0], umap_proj[:len(real), 1], umap_proj[len(real):, 1], SIZE_X=10, SIZE_Y=10, nb_iterations=10000)
        nb_points = np.asarray(nb_points)

        # Plot and save fig
        subplots_umaps(
            nb_points,
            umap_proj,
            labels1=labels,
            labels2=tissues_labels,
            labels3=cancer_labels,
            save_to=self.path +
            self.path_suffixe.format(epoch),
            marker=self.marker_type)

class TrackLoss:
    """
    Callback tracking all components of the loss.
    ----
    Parameters:
        verbose (bool): whether to print information when callback is called. Default False.
        path (str): path where to store loss history dictionary as numpy object. Default 'loss_history.npy'.
        nb_epochs (int): total number of epochs in training. Default 0.
    """

    def __init__(
            self,
            verbose: bool = False,
            path: str = 'loss_history.npy',
            nb_epochs: int = 0):

        self.verbose = verbose
        self.path = path
        self.nb_epochs = nb_epochs
        self.history = {}

    def __call__(self, hist_dict: dict):
        """
        Main function to call to store current training history. History is saved as a numpy object at the final epoch.
        ----
        Parameters:
            hist_dict (dict): current epoch training history dictionary.
        """
        for key in hist_dict.keys():
            if key not in self.history:
                self.history[key] = []
            self.history[key].append(hist_dict[key])

        if "disc_loss_epoch" in self.history.keys():
            if len(self.history["disc_loss_epoch"]) == self.nb_epochs:
            # Save current training history
                self.save_history()

    def save_history(self):
        """ Saves training history"""
        if self.verbose:
            print(f'All losses components tracked and saved.')
        np.save(self.path, self.history)
