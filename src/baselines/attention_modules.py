# Imports
import random 
import numpy as np
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import softmin

# SEED
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

    
class Attention_Head(nn.Module):
    """
    Self-attention head with domain knowledge on genes of interest: either from protein-protein interactions and genes correlations.
    """

    def __init__(self, genes_of_interest, nb_attention_head:int=1):
        """
        """
        super().__init__()

        # Genes of interest
        self.genes_of_interest = genes_of_interest
        self.nb_genes_of_interest = len(self.genes_of_interest)
        
        # K, Q, V layers for PPI: We compute keys and values only for genes with correlations/interactions
        self.query_layer = nn.Linear(self.nb_genes_of_interest, self.nb_genes_of_interest, bias=False)
        self.key_layer = nn.Linear(self.nb_genes_of_interest, self.nb_genes_of_interest, bias=False)
        self.value_layer = nn.Linear(self.nb_genes_of_interest, self.nb_genes_of_interest, bias=False)
        
        # Multi-head attention
        self.nb_attention_head = nb_attention_head
        if self.nb_attention_head==2:
            self.query_layer2 = nn.Linear(self.nb_genes_of_interest, self.nb_genes_of_interest, bias=False)
            self.key_layer2 = nn.Linear(self.nb_genes_of_interest, self.nb_genes_of_interest, bias=False)
            self.value_layer2 = nn.Linear(self.nb_genes_of_interest, self.nb_genes_of_interest, bias=False)    
            # Concatenation parameters between 2 heads. Random Init + convex combination.
            params = torch.softmax(torch.randn((2, self.nb_genes_of_interest)), 0)
            self.concat_weights1 = nn.Parameter(params[0])
            self.concat_weights2 = nn.Parameter(params[1])

        
    def forward_2nd_head(self, x, goi_dict:dict=None):
        """
        Pass data through second attention head.
        ----
        Parameters:
            x (torch.tensor): input batch.
            goi_dict (dict): genes of interest dictionary. Default None.
        Returns:
            out (torch.tensor): attention values for genes of interest.
        """
        batch_size, _ = x.size() # B * W

        # Initialize lookup table
        self.GenesTable = self.LookUpTable(genes_interactions_dict=goi_dict)

        # Store keys, queries and values for genes of interest only and add a dimension because we handle batches        
        Q = self.query_layer2(x[:, self.genes_of_interest]).view(batch_size, -1, self.nb_genes_of_interest) # Batch_size x 1 x Nb_genes_of_interest
        K = self.key_layer2(x[:, self.genes_of_interest]).view(batch_size, -1, self.nb_genes_of_interest) # Batch_size x 1 x Nb_genes_of_interest
        V = self.value_layer2(x[:, self.genes_of_interest]).view(batch_size, -1, self.nb_genes_of_interest) # Batch_size x 1 x Nb_genes_of_interest
        
        # Compute attention weighted values for all genes of interest
        out = self.compute_attention(K, Q, V)
        
        return out

        
    def forward(self, x, goi_dict:dict=None):
        """ Main function: pass data through attention module.
        ----
        Parameters:
            x (torch.tensor): input batch.
            goi_dict (dict): genes of interest dictionary. Default None.
        Returns:
            (torch.tensor): attention values for genes of interest.
        """
        batch_size, _ = x.size() # B * W

        # Initialize lookup table
        self.GenesTable = self.LookUpTable(genes_interactions_dict=goi_dict)

        # Store keys, queries and values for genes of interest only and add a dimension because we handle batches        
        Q = self.query_layer(x[:, self.genes_of_interest]).view(batch_size, -1, self.nb_genes_of_interest) # Batch_size x 1 x Nb_genes_of_interest
        K = self.key_layer(x[:, self.genes_of_interest]).view(batch_size, -1, self.nb_genes_of_interest) # Batch_size x 1 x Nb_genes_of_interest
        V = self.value_layer(x[:, self.genes_of_interest]).view(batch_size, -1, self.nb_genes_of_interest) # Batch_size x 1 x Nb_genes_of_interest

        # Compute attention weighted values for all genes of interest 
        out = self.compute_attention(K, Q, V)
        
        if self.nb_attention_head==2:
            out2= self.forward_2nd_head(x, goi_dict)
            # Concatenate multi head attention
            out = self.concat_weights1*out + self.concat_weights2*out2
        
        return out
    
    
    def attention_map(self, x, goi_dict:dict=None):
        """
        Compute attention values for genes of interest.
        ----
        Parameters:
            x (torch.tensor): input batch.
            goi_dict (dict): genes of interest dictionary. Default None.
        Returns:
            out (torch.tensor): Attention weighted values for all genes of interest (size Batch_size x Nb_genes_of_interest)
        """
        
        batch_size, _ = x.size() # B * W

        # Initialize lookup table
        self.GenesTable = self.LookUpTable(genes_interactions_dict=goi_dict)

        # Store keys, queries and values for genes of interest only and add a dimension because we handle batches        
        Q = self.query_layer(x[:, self.genes_of_interest]).view(batch_size, -1, self.nb_genes_of_interest) # Batch_size x 1 x Nb_genes_of_interest
        K = self.key_layer(x[:, self.genes_of_interest]).view(batch_size, -1, self.nb_genes_of_interest) # Batch_size x 1 x Nb_genes_of_interest
        V = self.value_layer(x[:, self.genes_of_interest]).view(batch_size, -1, self.nb_genes_of_interest) # Batch_size x 1 x Nb_genes_of_interest

        # Compute attention weighted values for all genes of interest 
        out = self.get_attention_matrix(K, Q, V)
        
        return out


    # Using the generator pattern (an iterable)
    class LookUpTable(object):
        """
        Generator of tuples of current gene index and corresponding interacting genes indices
        """
        def __init__(self, genes_interactions_dict:dict=None):
            self.genes_interactions_dict = genes_interactions_dict
            # Init with first key i.e first gene of interest index
            self.batch_idx = list(self.genes_interactions_dict.keys())[0] 
            self.idx = 0
            # Last key i.e maximum last batch index (-1 for indexing from 0)
            self.max_idx = len(self.genes_interactions_dict.keys())-1

        def __iter__(self):
            return self

        # Python 3 compatibility
        def __next__(self):
            return self.next()

        def next(self):
            if self.idx < self.max_idx: 
                # Update
                self.batch_idx = list(self.genes_interactions_dict.keys())[self.idx] 
                self.idx = self.idx+1

                # Use gene index key to return interaction indexes
                # Returns tuple
                return self.genes_interactions_dict[self.batch_idx]

            elif self.idx == self.max_idx:
                # Update
                self.batch_idx = list(self.genes_interactions_dict.keys())[self.idx]
                self.idx = self.idx+1
                # Use gene index key to return interaction indexes 
                return self.genes_interactions_dict[self.batch_idx]

            raise StopIteration()

    @staticmethod
    def compute_similarity_score(q, k):
        """
        Similarity score function (absolute difference between keys (K) and queries (Q)).
        ----
        Parameters:
            k (torch.tensor): size Batch_size x 1 x Nb_current_genes x Nb_genes_interactions
            q (torch.tensor): size Batch_size x 1 x Nb_current_genes
        Returns:
            similarity_score (torch.tensor): absolute difference between key and query.
        """

        # Compute distance as similarity score
        return torch.abs(q-k) 

    @staticmethod
    def compute_softmin(score):
        """
        Normalization function. (Softmax in general attention frameworks with images/text embeddings. Softmin in our case.)
        ----
        Parameters:
            score (torch.tensor): size Batch_size x 1 x Nb_current_genes x Nb_genes_interactions
        Returns:
            softmin (torch.tensor)
        """
        return softmin(score, dim=-1)

    @staticmethod
    def compute_weighted_values(attention_scores, values):
        """
        Weighting attention scores with values (dot product between attention scores and values (V)).
        Tensors are handled in 4D for correct matrix transposition.
        ----
        Parameters:
            attention_scores (torch.tensor): Batch_size x 1 x Nb_current_genes x Nb_genes_interactions
            values (torch.tensor): size Batch_size x 1 x Nb_current_genes x Nb_genes_interactions
        Returns:
            dot product VA.T
        """
        # Need to reshape the tensors for computations
        batch_size, nb_current_genes = values.shape[0], values.shape[2]
        # Transpose attention matrix: A.V =V.A^T
        return values.view(batch_size, nb_current_genes, 1, -1) @ attention_scores.view(batch_size, nb_current_genes, 1, -1).permute(0, 1, 3, 2) # Same as torch.bmm() but 4D here

    def attention_on_genes(self, k, q, v):
        """
        Compute attention weighted value on current gene.
        ----
        Parameters:
            k (torch.tensor): Keys; size Batch_size x 1 x Nb_genes_interactions
            q (torch.tensor): Queries; size Batch_size x 1 x Nb_current_genes x 1
            v (torch.tensor): Values; size Batch_size x 1 x Nb_genes_interactions
        Returns:
            out (torch.tensor): size Batch_size x Nb_current_genes
        """
        score = self.compute_similarity_score(q, k) # Shape: Batch_size x 1 x Nb_current_genes x Nb_genes_interactions
        att = self.compute_softmin(score) # Shape: Batch_size x 1 x Nb_current_genes x Nb_genes_interactions
        out = self.compute_weighted_values(att, v) # Shape: Batch_size x Nb_current_genes x 1 x 1
        out = out.view(out.shape[0], -1) # Shape: Batch_size x Nb_current_genes
        return out


    def compute_attention(self, K, Q, V):
        """
        Compute attention values for all genes of interest.
        ----
        Parameters:
            K (torch.tensor): Keys; size Batch_size x 1 x Nb_genes_of_interest
            Q (torch.tensor): Queries; size Batch_size x 1 x Nb_genes_of_interest
            V (torch.tensor): Values; size Batch_size x 1 x Nb_genes_of_interest
        Returns:
            Attention weighted values for all genes of interest (torch.tensor): size Batch_size x Nb_genes_of_interest
        """
        batch_size = K.shape[0]

        # Output tensor of size Batch_size x Nb_genes_of_interest
        output = torch.zeros_like(Q).view(batch_size, -1).to(Q.device)

        # Compute attention for all genes but with keys and values only for genes with interactions 
        table = self.GenesTable
        print(table)

        # for i, term in enumerate(table):
        #     print(i)
        #     print(term)
        #     break
            
        # Loop over lookup table generator
        for self.ITER, (current_genes, interactions_idx) in enumerate(table):                
            nb_current_genes = len(current_genes)
            weighted_attention = self.attention_on_genes(K[:, :, interactions_idx], Q[:, :, current_genes].view(batch_size, -1, nb_current_genes, 1), V[:, :, interactions_idx]) 
            # Rebuild attention values in original dimension (with attention value of 0 for genes not included in genes of interest)
            output[:, current_genes] += weighted_attention

        return output
    
    def attention_matrix_on_genes(self, k, q, v):
        """
        Compute attention weighted value on current gene.
        ----
        Parameters:
            k (torch.tensor): Keys; size Batch_size x 1 x Nb_genes_interactions
            q (torch.tensor): Queries; size Batch_size x 1 x Nb_current_genes x 1
            v (torch.tensor): Values; size Batch_size x 1 x Nb_genes_interactions
        Returns:
            out (torch.tensor): size Batch_size x Nb_current_genes
        """
        score = self.compute_similarity_score(q, k) # Shape: Batch_size x 1 x Nb_current_genes x Nb_genes_interactions
        att = self.compute_softmin(score) # Shape: Batch_size x 1 x Nb_current_genes x Nb_genes_interactions
        att = att.view(att.shape[0], att.shape[2], -1) # Shape: Batch_size x Nb_current_genes x Nb_genes_interactions
        return att
    
    def get_attention_matrix(self, K, Q, V):
        """
        Compute attention values for all genes of interest.
        ----
        Parameters:
            K (torch.tensor): size Batch_size x 1 x Nb_genes_of_interest
            Q (torch.tensor): size Batch_size x 1 x Nb_genes_of_interest
            V (torch.tensor): size Batch_size x 1 x Nb_genes_of_interest
        Returns:
            Attention weighted values for all genes of interest (torch.tensor): size Batch_size x Nb_genes_of_interest
        """
        batch_size = K.shape[0]

        # Output tensor of size Batch_size x Nb_genes_of_interest
        output = list()

        # Compute attention for all genes but with keys and values only for genes with interactions 
        table = self.GenesTable
        
        # Loop over lookup table generator
        for self.ITER, (current_genes, interactions_idx) in enumerate(table):                
            nb_current_genes = len(current_genes)
            att_matrix = self.attention_matrix_on_genes(K[:, :, interactions_idx], Q[:, :, current_genes].view(batch_size, -1, nb_current_genes, 1), V[:, :, interactions_idx]) 
            # Rebuild attention values in original dimension (with attention value of 0 for genes not included in genes of interest)
            output.append(att_matrix)

        return output

class Attention_DK_Multi(nn.Module):
    """
    Self-attention module based on domain knowledge: either 1 head on PPI/CoExp or 2 heads combining them.
    """

    def __init__(self, config):
        """
        """
        super().__init__()

        self.attn_on_ppi = False
        self.attn_on_corr =  False
        self.config = config

        if config['ppi_threshold'] != -1:
            self.attn_on_ppi = True
            self.head_ppi = Attention_Head(config['genes_of_interest_ppi'], len(config['genes_of_interest_ppi']))
            # Total unique genes of interest
            self.unique_genes_of_interest = config['genes_of_interest_ppi']
        if config['corr_threshold'] != -1:
            self.attn_on_corr = True
            self.head_corr = Attention_Head(config['genes_of_interest_corr'], len(config['genes_of_interest_corr']))
            # Total unique genes of interest
            self.unique_genes_of_interest = config['genes_of_interest_corr']

        if self.attn_on_ppi and self.attn_on_corr:
            self.dk_multi = True 
            print("MultiHead attention module")
        else:
            self.dk_multi = False
            print("1-head attention module")


        # Multiple domain knowledge
        if self.dk_multi:
            # Overlapping genes of interest
            self.overlaps_genes_of_interest = config['overlaps_genes_of_interest']
            self.nb_overlaps_genes_of_interest = len(self.overlaps_genes_of_interest)
            self.overlaps_ppi_idx_reduced = config['overlaps_ppi_reduced']
            self.overlaps_corr_idx_reduced = config['overlaps_corr_reduced']
            
            # Non overlapping genes
            self.non_overlaps_ppi = config['non_overlaps_ppi']
            self.non_overlaps_corr = config['non_overlaps_corr']
            # Non overlapping genes reduced
            self.non_overlaps_ppi_reduced = config['non_overlaps_ppi_reduced']
            self.non_overlaps_corr_reduced = config['non_overlaps_corr_reduced']
                    
            # Concatenation parameter between PPI and corr attention outputs.  Random Init + convex combination.
            params_concat = torch.softmax(torch.randn((2, self.nb_overlaps_genes_of_interest)), 0)
            self.concat_weights_ppi = nn.Parameter(params_concat[0])
            self.concat_weights_corr = nn.Parameter(params_concat[1])
        
        # Gamma parameter
        if config['is_gamma_fixed']:
            self.gamma = torch.tensor([config['gamma_init']]).to(config['device'])
        elif not config['is_gamma_fixed']:
            self.gamma = nn.Parameter(torch.zeros(1, requires_grad=True))


    def forward(self, x, goi_dict_ppi:dict=None, goi_dict_corr:dict=None):
        """
        Compute attention values for genes of interest in PPIs and/or CoExpression interactions. Values are weighted by the attention gamma parameter before output.
        ----
        Parameters:
            x (torch.tensor): input batch.
            goi_dict_ppi (dict): genes of interest dictionary for PPI. Default None.
            goi_dict_corr (dict): genes of interest dictionary for CoExpression. Default None.
        Returns:
            (torch.tensor): attention values for genes of interest, others 0.
        """

        if self.dk_multi:
            # Compute self-attention for PPI
            out_ppi = self.head_ppi(x.clone(), goi_dict_ppi)
            # Compute self-attention for correlations
            out_corr = self.head_corr(x.clone(), goi_dict_corr)
            # Concatenate multi attention
            out = self.concat_weights_ppi*out_ppi[:, self.overlaps_ppi_idx_reduced] + self.concat_weights_corr*out_corr[:, self.overlaps_corr_idx_reduced]

            # Rebuild attention values in original dimension. 
            # We replace genes of interest with attention outputs, other genes are left unmodified (i.e it is like their attention value is 0)
            # Trick to avoid inplace operations not allowed by pytorch and concatenate attention values to keep gradient info:
            output = torch.zeros_like(x).to(x.device)
            
            # Add attention of unique genes
            output[:, self.non_overlaps_ppi] += out_ppi[:, self.non_overlaps_ppi_reduced]
            output[:, self.non_overlaps_corr] += out_corr[:, self.non_overlaps_corr_reduced]
            output[:, self.overlaps_genes_of_interest] += out

        else:
            if self.attn_on_corr:
                out = self.head_corr(x, goi_dict_corr)
            elif self.attn_on_ppi:
                out = self.head_ppi(x, goi_dict_ppi)

            # Add attention of unique genes
            output = torch.zeros_like(x, requires_grad=False).to(x.device)
            output[:, self.config["genes_of_interest_ppi"]] += out
 
        return self.gamma*output
