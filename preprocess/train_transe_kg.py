from __future__ import absolute_import, division, print_function

import os
import sys
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from utils import *
from dataset import *

logger = None


class KG_KGemb_dataloader(object):
    """This class acts as the dataloader for training knowledge graph embeddings."""

    def __init__(self, kg, datadir, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size

        rating_file = datadir + '/ratings_final'
        rating_np = np.load(rating_file + '.npy')
        n_user = max(set(rating_np[:, 0])) + 1
        n_item = max(set(rating_np[:, 1])) + 1

        self.data = pd.read_csv(f'{datadir}/train_pd.csv',index_col=None)
        self.data = self.data.drop(self.data.columns[0], axis=1)
        self.data = self.data[['user','item','like']].values

        kg_file = datadir + '/kg_final'
        self.kg_np = np.load(kg_file + '.npy')

        self.data_entity_triplets = []
        for row in self.data:
            if row[2] == 1:
                user = row[0]
                item = row[1] + n_user
                self.data_entity_triplets.append([self.dataset.et_idx2ty[user], PURCHASE,
                     self.dataset.et_idx2ty[item], user, item])

        print('len(dataset.entity_list[USER]) = ', dataset.entity_list[USER].vocab_size)
        print('len(dataset.entity_list[PRODUCT]) = ', dataset.entity_list[PRODUCT].vocab_size)
        self.reset()

    def reset(self):

        self.entity_triplets = self.data_entity_triplets
        
        self.start_index = 0
        self.end_index = self.batch_size
        random.shuffle(self.entity_triplets)  
        self._has_next = True

    def get_batch(self):
        """Return a matrix of [batch_size x 8], where each row contains
        (u_id, p_id, w_id, b_id, c_id, rp_id, rp_id, rp_id).
        """

        batch = self.entity_triplets[self.start_index:self.end_index]

        self.start_index += self.batch_size
        self.end_index += self.batch_size
        if self.end_index >= len(self.entity_triplets):
            self._has_next = False

        return batch


    def has_next(self):
        """Has next batch."""
        return self._has_next

class KnowledgeEmbedding(nn.Module):
    def __init__(self, kg, dataset, args):
        super(KnowledgeEmbedding, self).__init__()
        self.embed_size = args.embed_size
        self.num_neg_samples = args.num_neg_samples
        self.device = args.device
        self.l2_lambda = args.l2_lambda

        self.entities = edict()

        for et in dataset.entity_list:
            self.entities[et] = edict(vocab_size= int(dataset.entity_list[et].vocab_size))

        for e in self.entities:
            embed = self._entity_embedding(self.entities[e].vocab_size)
            setattr(self, e, embed)

        self.relations = edict()
        for r in dataset.rela_list:

            if PURCHASE == r:
                tmp_1 = [float(0) for _ in range(dataset.entity_list[USER].vocab_size)]
                tmp_2 = [float(1) for _ in range(dataset.entity_list[USER].vocab_size,dataset.entity_list[PRODUCT].vocab_size,1)]
                self.relations[r] = edict(
                    et_distrib=self._make_distrib(tmp_1 + tmp_2))
            else:
                self.relations[r] = edict(
                    et_distrib=self._make_distrib([float(1) for _ in range(6000)]))

        for r in dataset.rela_list:
            embed = self._relation_embedding()
            setattr(self, r, embed)
            bias = self._relation_bias(188047)

            setattr(self, r + '_bias', bias)

    def _entity_embedding(self, vocab_size):
        """Create entity embedding of size [vocab_size+1, embed_size].
            Note that last dimension is always 0's.
        """
        embed = nn.Embedding(vocab_size + 1, self.embed_size, padding_idx=-1, sparse=False)
        initrange = 0.5 / self.embed_size
        weight = torch.FloatTensor(vocab_size + 1, self.embed_size).uniform_(-initrange, initrange)
        embed.weight = nn.Parameter(weight)
        return embed

    def _relation_embedding(self):
        """Create relation vector of size [1, embed_size]."""
        initrange = 0.5 / self.embed_size
        weight = torch.FloatTensor(1, self.embed_size).uniform_(-initrange, initrange)
        embed = nn.Parameter(weight)
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

    def forward(self, batch):
        loss = self.compute_loss(batch)
        return loss

    def compute_loss(self, batch):
        """Compute knowledge graph negative sampling loss.
        batch_idxs: batch_size * 8 array, where each row is
                (u_id, p_id, w_id, b_id, c_id, rp_id, rp_id, rp_id).
        """

        regularizations, loss = [], 0

        for b_i in range(len(batch)):
            row = batch[b_i]
            row[3] = torch.from_numpy(np.array([int(row[3])])).to(self.device)
            row[4] = torch.from_numpy(np.array([int(row[4])])).to(self.device)
            up_loss, up_embeds = self.neg_loss(row[0], row[1], row[2], 
                    row[3], row[4])
            regularizations.extend(up_embeds)
            loss += up_loss

        # l2 regularization
        if self.l2_lambda > 0:
            l2_loss = 0.0
            for term in regularizations:
                l2_loss += torch.norm(term)
            loss += self.l2_lambda * l2_loss

        return loss

    def neg_loss(self, entity_head, relation, entity_tail, entity_head_idxs, entity_tail_idxs):
        # Entity tail indices can be -1. Remove these indices. Batch size may be changed!

        fixed_entity_head_idxs = entity_head_idxs
        fixed_entity_tail_idxs = entity_tail_idxs

        entity_head_embedding = getattr(self, entity_head)  # nn.Embedding
        entity_tail_embedding = getattr(self, entity_tail)  # nn.Embedding
        relation_vec = getattr(self, relation)  # [1, embed_size]
        relation_bias_embedding = getattr(self, relation + '_bias')  # nn.Embedding
        entity_tail_distrib = self.relations[relation].et_distrib  # [vocab_size] 

        return kg_neg_loss(entity_head_embedding, entity_tail_embedding,
                           fixed_entity_head_idxs, fixed_entity_tail_idxs,
                           relation_vec, relation_bias_embedding, self.num_neg_samples, entity_tail_distrib)


def kg_neg_loss(entity_head_embed, entity_tail_embed, entity_head_idxs, entity_tail_idxs,
                relation_vec, relation_bias_embed, num_samples, distrib):
    """Compute negative sampling loss for triple (entity_head, relation, entity_tail).

    Args:
        entity_head_embed: Tensor of size [batch_size, embed_size].
        entity_tail_embed: Tensor of size [batch_size, embed_size].
        entity_head_idxs:
        entity_tail_idxs:
        relation_vec: Parameter of size [1, embed_size].
        relation_bias: Tensor of size [batch_size]
        num_samples: An integer.
        distrib: Tensor of size [vocab_size].

    Returns:
        A tensor of [1].
    """

    entity_head_vec = entity_head_embed(entity_head_idxs)  # [batch_size, embed_size]
    example_vec = entity_head_vec + relation_vec  # [batch_size, embed_size]
    example_vec = example_vec.unsqueeze(2)  # [batch_size, embed_size, 1]

    entity_tail_vec = entity_tail_embed(entity_tail_idxs)  # [batch_size, embed_size]
    pos_vec = entity_tail_vec.unsqueeze(1)  # [batch_size, 1, embed_size]
    relation_bias = relation_bias_embed(entity_tail_idxs).squeeze(1)  # [batch_size]
    pos_logits = torch.bmm(pos_vec, example_vec).squeeze() + relation_bias  # [batch_size]
    pos_loss = -pos_logits.sigmoid().log()  # [batch_size]

    neg_sample_idx = torch.multinomial(distrib, num_samples, replacement=True).view(-1)

    neg_vec = entity_tail_embed(neg_sample_idx)  # [num_samples, embed_size]
    neg_logits = torch.mm(example_vec.squeeze(2), neg_vec.transpose(1, 0).contiguous())
    neg_logits += relation_bias.unsqueeze(1)  # [batch_size, num_samples]
    neg_loss = -neg_logits.neg().sigmoid().log().sum(1)  # [batch_size]

    loss = (pos_loss + neg_loss).mean()
    return loss, [entity_head_vec, entity_tail_vec, neg_vec]


def train(args):
    dataset = load_dataset(args.dataset)
    kg = load_kg(args.dataset)
    model = KnowledgeEmbedding(kg, dataset, args).to(args.device)

    dataloader = KG_KGemb_dataloader(kg, DATA_DIR[args.dataset], dataset, args.batch_size)
    logger.info('Parameters:' + str([i[0] for i in model.named_parameters()]))
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    steps = 0 
    for epoch in range(1, args.epochs + 1):
        dataloader.reset()
        while dataloader.has_next():
            lr = args.lr
            for pg in optimizer.param_groups:
                pg['lr'] = lr

            batch = dataloader.get_batch()
            optimizer.zero_grad()
            train_loss = model(batch)
            train_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()

            steps += 1
            if steps % args.steps_per_checkpoint == 0:
                logger.info('Epoch: {:02d} | '.format(epoch) +
                            'Lr: {:.5f} | '.format(lr) )

        torch.save(model.state_dict(), '{}/transe_model_sd_epoch_{}.ckpt'.format(args.log_dir, epoch))

def extract_embeddings(args):
    """Note that last entity embedding is of size [vocab_size+1, d]."""
    model_file = '{}/transe_model_sd_epoch_{}.ckpt'.format(args.log_dir, args.epochs)
    print('Load embeddings', model_file)
    dataset = load_dataset(args.dataset)
    state_dict = torch.load(model_file, map_location=lambda storage, loc: storage)
    embeds = {}
    for et in dataset.entity_list:
        embeds[et] = state_dict[et+'.weight'].cpu().data.numpy()[:-1]
    for r in dataset.rela_list:
        embeds[r] =(state_dict[r].cpu().data.numpy()[0],
                state_dict[r+'_bias.weight'].cpu().data.numpy())

    save_embed(args.dataset, embeds)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=BEAUTY_CORE, help='One of {beauty, cd, cell, clothing}.')
    parser.add_argument('--name', type=str, default='train_transe_model', help='model name.')
    parser.add_argument('--seed', type=int, default=123, help='random seed.')
    parser.add_argument('--gpu', type=str, default='0', help='gpu device.')
    parser.add_argument('--epochs', type=int, default=1, help='number of epochs to train.')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size.')
    parser.add_argument('--sub_batch_size', type=int, default=5, help='sub batch size.')
    parser.add_argument('--lr', type=float, default=0.5, help='learning rate.')
    parser.add_argument('--weight_decay', type=float, default=0, help='weight decay for adam.')
    parser.add_argument('--l2_lambda', type=float, default=0, help='l2 lambda')
    parser.add_argument('--max_grad_norm', type=float, default=5.0, help='Clipping gradient.')
    parser.add_argument('--embed_size', type=int, default=50, help='knowledge embedding size.')
    parser.add_argument('--num_neg_samples', type=int, default=5, help='number of negative samples.')
    parser.add_argument('--gradient_plot',  type=str, default='gradient_plot/', help='number of negative samples.')
    parser.add_argument('--steps_per_checkpoint', type=int, default=200, help='Number of steps for checkpoint.')
    parser.add_argument('--att_core', type=int, default=0, help='core number')
    parser.add_argument('--item_core', type=int, default=10, help='core number')
    parser.add_argument('--user_core', type=int, default=300, help='core number')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    args.device = torch.device('cuda:0') 
    
    args.log_dir = '{}/{}'.format(DATA_DIR[args.dataset], args.name)
    if not os.path.isdir(args.log_dir):
        os.makedirs(args.log_dir)

    global logger
    logger = get_logger(args.log_dir + '/train_log.txt')
    logger.info(args)

    set_random_seed(args.seed)
    train(args)
    extract_embeddings(args)

if __name__ == '__main__':
    main()

