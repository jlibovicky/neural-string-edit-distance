"""
Copyright (C) 2019 University of Massachusetts Amherst.
This file is part of "stance"
http://github.com/iesl/stance
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import math

import numpy as np
import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss


def batch_sinkhorn_loss(C, C_mask, epsilon=1, niter=100):
    """
    :param C: Batch size by MSL by MSL
    :param C_mask: Batch size by MSL by MSL
    :param epsilon:
    :param n:
    :param niter:
    :return:
    """
    # B by MSL
    mu = C_mask[:,:,0]
    mu = mu / mu.sum(dim=1, keepdim=True)

    nu = C_mask[:,0,:]
    nu = nu / nu.sum(dim=1, keepdim=True)

    # Parameters of the Sinkhorn algorithm.
    rho = 1  # (.5) **2          # unbalanced transport
    tau = -.8  # nesterov-like acceleration
    lam = rho / (rho + epsilon)  # Update exponent
    thresh = 10**(-1)  # stopping criterion

    def ave(u, u1):
        "Barycenter subroutine, used by kinetic acceleration through extrapolation."
        return tau * u + (1 - tau) * u1

    def M(u, v):
        "Modified cost for logarithmic updates"
        "$M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$"
        return (-C + u.unsqueeze(2) + v.unsqueeze(1)) / epsilon

    def lse(A,dim):
        "log-sum-exp"
        return torch.log(torch.exp(A).sum(dim=dim, keepdim=True) + 1e-6)  # add 10^-6 to prevent NaN
    batch_size = C_mask.size(0)
    u, v, err = 0. * mu, 0. * nu, 0.
    actual_nits = 0  # to check if algorithm terminates because of threshold or max iterations reached
    for i in range(niter):
        u1 = u  # useful to check the update
        u = epsilon * (
            torch.log(mu) # B by MSL
            - lse(M(u, v), # M = B by MSL by MSL, lse should sum along the columns
                  dim=2).squeeze()) \
            + u
        v = epsilon * (torch.log(nu) - lse(M(u, v),dim=1).squeeze()) + v
        # accelerated unbalanced iterations
        # u = ave( u, lam * ( epsilon * ( torch.log(mu) - lse(M(u,v)).squeeze()   ) + u ) )
        # v = ave( v, lam * ( epsilon * ( torch.log(nu) - lse(M(u,v).t()).squeeze() ) + v ) )
        err = (u - u1).abs().sum() / batch_size

        actual_nits += 1
        if (err < thresh).data.cpu().numpy():
            break
    U, V = u, v
    pi = torch.exp(M(U, V))  # Transport plan pi = diag(a)*K*diag(b)
    return pi


class CNN(torch.nn.Module):
    def __init__(self, is_increasing, num_layers, filter_counts, max_len_token=100):
        """
        params is_increasing: whether the filter size is increasing or decreasing
        params num_layers: number of layers in the CNN
        params filter_counts: dictionary of filter index to filter size
        params max_len_token: maximum number of tokens in sentence
        """
        super(CNN, self).__init__()
        decreasing = 0
        if not is_increasing:
            decreasing = 1
        assert 1 <= num_layers <= 4
        self.num_layers = num_layers

        map_conv_layer_to_filter_size = {
            4: [[3, 5, 5, 7], [7, 5, 5, 3]],
            3: [[5, 5, 7], [7, 5, 5]],
            2: [[5, 3],[5, 3]],
            1: [[7],[7]]}
        pool_output_height = int(np.floor(max_len_token/2.0))


        for i in range(1, self.num_layers+1):
            filter_size = map_conv_layer_to_filter_size[self.num_layers][decreasing][i-1]
            padding_size = math.floor(filter_size / 2)
            prev_filter_count = 1
            if i > 1:
                prev_filter_count = filter_counts[i-2]
            convlyr = nn.Conv2d(prev_filter_count, filter_counts[i-1], filter_size, padding = padding_size, stride=1)
            if i == 1:
                self.add_module("cnn_1", convlyr)
            elif i == 2:
                self.add_module("cnn_2", convlyr)
            elif i == 3:
                self.add_module("cnn_3", convlyr)
            elif i == 4:
                self.add_module("cnn_4", convlyr)

        self.align_weights = nn.Parameter(
            torch.randn(filter_counts[num_layers - 1],
                pool_output_height,
                pool_output_height).cuda(),requires_grad=True)
        self.final_dense = nn.Conv2d(filter_counts[num_layers - 1], 1, 1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d((2, 2), stride=2)

    def forward(self, src_tgt_sim, src_tgt_mask):
        """Run CNN over input.

        :params src_tgt_sim: tensor representing similarity between source and target
        :return: scores for similarity
        """

        # Needs num channels
        convd = self.cnn_1(src_tgt_sim.unsqueeze(1))
        if self.num_layers > 1:
            convd = self.relu(convd)
            convd = self.cnn_2(convd)
        if self.num_layers > 2:
            convd = self.relu(convd)
            convd = self.cnn_3(convd)
        if self.num_layers > 3:
            convd = self.relu(convd)
            convd = self.cnn_4(convd)

        convd_after_pooling = self.pool(convd)
        pooled_mask = self.pool(src_tgt_mask)

        output = self.final_dense(convd_after_pooling).squeeze(1)

        return (output * pooled_mask).sum(2).sum(1) / pooled_mask.sum(2).sum(1)


class Stance(torch.nn.Module):
    """"STANCE first gets character embeddings. Next, LSTM runs over char
    embeddings to get char representations. Then, similarity matrix created
    where all LSTM embeddings are scored for similarity using dot product.
    Optimal Transport is then run over the similarity matrix to align weights.
    Finally, CNN detects features in aligned matrix and outputs similarity
    score."""

    def __init__(self, encoder, cnn_increasing, cnn_num_layers, cnn_filter_counts, tgt_encoder=None):
        """
        param config: config object
        param vocab: vocab object
        param max_len_token: max number of tokens
        """
        super(Stance, self).__init__()
        self.src_encoder = encoder
        if tgt_encoder is None:
            self.tgt_encoder = encoder
        else:
            self.tgt_encoder = tgt_encoder

        self.CNN = CNN(cnn_increasing, cnn_num_layers, cnn_filter_counts)

        # Vector of ones (used for loss)
        self.loss = BCEWithLogitsLoss()

    def compute_loss(self, query, query_mask, pos, pos_mask, neg, neg_mask):
        """Compute loss for batch of query positive negative triplets.

        param qry: query tokens (batch size of list of tokens)
        param pos: positive mention lookup (batch size of list of tokens)
        param neg: negative mention lookup (batch size of list of tokens)
        return: loss (batch_size)
        """

        pos_score = self.score_pair(query, pos, query_mask, pos_mask)
        neg_score = self.score_pair(query, neg, query_mask, neg_mask)

        diff_loss = self.loss(pos_score - neg_score, torch.ones_like(pos_score))
        pos_loss = self.loss(pos_score, torch.ones_like(pos_score))
        neg_loss = self.loss(neg_score, torch.zeros_like(pos_score))

        return diff_loss + pos_loss + neg_loss


    def score_pair(self, qry_emb, cnd_emb, qry_msk, cnd_msk):
        """Score the batch of query candidate pair.

        Take the dot product of all pairs of embeddings (with bmm) to get similarity matrix
        Uses optimal transport to align the weights
        Then runs CNN over the similarity matrix

        param qry: query mention embedding (batch_size * max_len_token * hidden_state_output_size)
        param cnd: candidate mention embedding (batch_size * max_len_token * hidden_state_output_size)
        param qry_msk: query mention mask (batch_size * max_len_token)
        param cnd_mask: candidate mention mask (batch_size * max_len_token)
        return: score for query candidate pairs (batch_size * 1)
        """

        qry_cnd_sim = torch.bmm(qry_emb, torch.transpose(cnd_emb, 2, 1))

        qry_mask = qry_msk.unsqueeze(dim=2).float()
        cnd_msk = cnd_msk.unsqueeze(dim=1).float()
        qry_cnd_mask = torch.bmm(qry_mask, cnd_msk)

        qry_cnd_dist = torch.cuda.FloatTensor(qry_cnd_sim.size()).fill_(torch.max(qry_cnd_sim)) - qry_cnd_sim + 1e-6
        qry_cnd_pi = batch_sinkhorn_loss(qry_cnd_dist, qry_cnd_mask)
        qry_cnd_sim_aligned = torch.mul(qry_cnd_sim, qry_cnd_pi)
        qry_cnd_sim_aligned = torch.mul(qry_cnd_sim_aligned, qry_cnd_mask)

        return self.CNN(qry_cnd_sim_aligned, qry_cnd_mask)
