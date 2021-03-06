from __future__ import absolute_import, division, print_function

import sys
import os
import argparse
from collections import namedtuple
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Categorical

from easydict import EasyDict as edict

from utils import *
 
SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])



class KnowledgeEmbedding_memory_graph(nn.Module):
    def __init__(self, args):
        super(KnowledgeEmbedding_memory_graph, self).__init__()
        self.embed_size = args.embed_size

        self.device = args.device
        self.l2_lambda = args.l2_lambda
        dataset = load_dataset(args, args.dataset, logger = args.logger)


        # Initialize entity embeddings.
        self.embeds = load_embed(args.dataset, logger = args.logger)

        
        
        if args.load_pt_emb_size == True:
            self.embeds = load_embed_dim(args.dataset, args.embed_size, logger = args.logger)
            print('self.embeds = load_embed(load_embed_dim) = ')
            args.logger.info('self.embeds = load_embed(load_embed_dim')
            # input()
            # except:
            #     pass

        self.entities = edict()

        for et in dataset.entity_list:
            # print('et in dataset ', et, dataset.entity_list[et].vocab_size)
            print('et = ',  et, int(dataset.entity_list[et].vocab_size))
            self.entities[et] = edict(vocab_size= int(dataset.entity_list[et].vocab_size))


        for e in self.entities:
            embed = self._entity_embedding(e, self.entities[e].vocab_size)
            setattr(self, e, embed)

        self.relations = edict()
        for r in dataset.rela_list:
            self.relations[r] = edict(
                et_distrib=self._make_distrib([float(1) for _ in range(6000)]))

        for r in dataset.rela_list:
            embed = self._relation_embedding(r)
            setattr(self, r, embed)
            bias = self._relation_bias(188047)
            setattr(self, r + '_bias', bias)


    def _entity_embedding(self, key, vocab_size):
        """Create entity embedding of size [vocab_size+1, embed_size].
            Note that last dimension is always 0's.
        """
        embed = nn.Embedding(vocab_size + 1, self.embed_size, padding_idx=-1, sparse=False).requires_grad_(False)
        embed.weight.data = torch.from_numpy(np.concatenate((self.embeds[key], np.zeros_like(self.embeds[key])[:1,:]), axis=0))[:,:self.embed_size]
        print('key = ', key)
        print('self.embeds[key] = ', (torch.from_numpy(np.concatenate((self.embeds[key], np.zeros_like(self.embeds[key])[:1,:]), axis=0))[:,:self.embed_size]).shape)
        print('vocab_size + 1 = ', vocab_size + 1)
        return embed

    def _relation_embedding(self):
        """Create relation vector of size [1, embed_size]."""
        # initrange = 0.5 / self.embed_size
        weight = torch.randn(1, self.embed_size,  requires_grad=False)

        embed = nn.Parameter(weight)
        return embed

    def _relation_embedding(self, key): 
        """Create relation vector of size [1, embed_size]."""
        # initrange = 0.5 / self.embed_size
        weight = torch.randn(1, self.embed_size,  requires_grad=False)
        print('weight = ', weight.shape)
        try:
            embed = nn.Parameter(torch.from_numpy(self.embeds[key][0:])[:,:self.embed_size])
            print('torch.from_numpy(self.embeds[key][0:])[:,:self.embed_size] = ', (torch.from_numpy(self.embeds[key][0:])[:,:self.embed_size]).shpae)
        except:
            embed = nn.Parameter(weight[:,:self.embed_size])
        return embed

    def _relation_bias(self, vocab_size):
        """Create relation bias of size [vocab_size+1]."""
        bias = nn.Embedding(vocab_size + 1, 1, padding_idx=-1, sparse=False)
        bias.weight = nn.Parameter(torch.zeros(vocab_size + 1, 1))
        return bias

    def _make_distrib(self, distrib):
        """Normalize input numpy vector to distribution."""
        distrib = np.power(np.array(distrib, dtype=np.float), 0.75)
        distrib = distrib / distrib.sum()
        distrib = torch.FloatTensor(distrib).to(self.device)
        return distrib

    def lookup_emb(self, node_type, type_index):
        embedding_file = getattr(self, node_type)
        entity_vec = embedding_file(type_index)
        return entity_vec

    def lookup_rela_emb(self, node_type):
        relation_vec = getattr(self, node_type)
        return relation_vec


class KnowledgeEmbedding_memory(nn.Module):
    def __init__(self, args):
        super(KnowledgeEmbedding_memory, self).__init__()


        dataset = load_dataset(args.dataset, logger = args.logger)

        self.embed_size = args.embed_size
        self.device = args.device

        self.embeds = load_embed(args.dataset, logger = args.logger)

        if args.load_pt_emb_size == True:
            try:
                self.embeds = load_embed_dim(args.dataset, args.embed_size, logger = args.logger)
                print('self.embeds = load_embed(load_embed_dim) = ')
                args.logger.info('self.embeds = load_embed(load_embed_dim')
            except:
                pass

        # Initialize entity embeddings.
        self.entities = edict(
            user=edict(vocab_size=dataset.user.vocab_size),
            product=edict(vocab_size=dataset.product.vocab_size),
            word=edict(vocab_size=dataset.word.vocab_size),
            related_product=edict(vocab_size=dataset.related_product.vocab_size),
            brand=edict(vocab_size=dataset.brand.vocab_size),
            category=edict(vocab_size=dataset.category.vocab_size),
        )

        for e in self.entities:
            embed = self._entity_embedding(e, self.entities[e].vocab_size)
            setattr(self, e, embed)

        # Initialize relation embeddings and relation biases.
        self.relations = edict(
            self_loop=edict(
                et='self_loop',
                et_distrib=self._make_distrib(dataset.review.product_uniform_distrib)),
            purchase=edict(
                et='product',
                et_distrib=self._make_distrib(dataset.review.product_uniform_distrib)),
            mentions=edict(
                et='word',
                et_distrib=self._make_distrib(dataset.review.word_distrib)),
            described_as=edict(
                et='word',
                et_distrib=self._make_distrib(dataset.review.word_distrib)),
            produced_by=edict(
                et='brand',
                et_distrib=self._make_distrib(dataset.produced_by.et_distrib)),
            belongs_to=edict(
                et='category',
                et_distrib=self._make_distrib(dataset.belongs_to.et_distrib)),
            also_bought=edict(
                et='related_product',
                et_distrib=self._make_distrib(dataset.also_bought.et_distrib)),
            also_viewed=edict(
                et='related_product',
                et_distrib=self._make_distrib(dataset.also_viewed.et_distrib)),
            bought_together=edict(
                et='related_product',
                et_distrib=self._make_distrib(dataset.bought_together.et_distrib)),
            padding=edict(
                et='padding',
                et_distrib=self._make_distrib(dataset.bought_together.et_distrib)),
        )
        for r in self.relations:
            # print('r = ', 'setup', r)
            embed = self._relation_embedding(r)
            setattr(self, r, embed)
            bias = self._relation_bias(len(self.relations[r].et_distrib))
            setattr(self, r + '_bias', bias)


    def _entity_embedding(self, key, vocab_size):
        """Create entity embedding of size [vocab_size+1, embed_size].
            Note that last dimension is always 0's.
        """
        embed = nn.Embedding(vocab_size + 1, self.embed_size, padding_idx=-1, sparse=False).requires_grad_(False)
        embed.weight.data = torch.from_numpy(np.concatenate((self.embeds[key], np.zeros_like(self.embeds[key])[:1,:]), axis=0))[:,:self.embed_size]
        print('key = ', key)
        print('self.embeds[key] = ', (torch.from_numpy(np.concatenate((self.embeds[key], np.zeros_like(self.embeds[key])[:1,:]), axis=0))[:,:self.embed_size]).shape)
        print('vocab_size + 1 = ', vocab_size + 1)
        return embed

    def _relation_embedding(self, key): 
        """Create relation vector of size [1, embed_size]."""
        # initrange = 0.5 / self.embed_size
        weight = torch.randn(1, self.embed_size,  requires_grad=False)
        print('key = ', key)
        try:
            embed = nn.Parameter(torch.from_numpy(self.embeds[key][0:])[:,:self.embed_size])
        except:
            embed = nn.Parameter(weight[:,:self.embed_size])
        return embed

    def _relation_bias(self, vocab_size):
        """Create relation bias of size [vocab_size+1]."""
        bias = nn.Embedding(vocab_size + 1, 1, padding_idx=-1, sparse=False)
        bias.weight = nn.Parameter(torch.zeros(vocab_size + 1, 1))
        return bias

    def _make_distrib(self, distrib):
        """Normalize input numpy vector to distribution."""
        distrib = np.power(np.array(distrib, dtype=np.float), 0.75)
        distrib = distrib / distrib.sum()
        distrib = torch.FloatTensor(distrib).to(self.device)
        return distrib

    def lookup_emb(self, node_type, type_index):
        embedding_file = getattr(self, node_type)
        entity_vec = embedding_file(type_index)
        return entity_vec

    def lookup_rela_emb(self, node_type):
        relation_vec = getattr(self, node_type)
        return relation_vec
