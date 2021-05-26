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

class Review_KGemb_dataloader(object):
    """This class acts as the dataloader for training knowledge graph embeddings."""

    def __init__(self, args, dataset, batch_size):
        
        self.dataset = dataset
        self.batch_size = batch_size
        self.total_trn_review = self.dataset.review.size
        self.product_relations = ['produced_by', 'belongs_to', 'also_bought', 'also_viewed', 'bought_together']
        self.reset()

    def reset(self):
        # Shuffle reviews order
        self.random_idx = np.random.permutation(self.total_trn_review)
        self.index = 0
        self.sub_index = 0
        self._has_next = True

    def get_batch(self):
        """Return a matrix of [batch_size x 8], where each row contains
        (u_id, p_id, w_id, b_id, c_id, rp_id, rp_id, rp_id).
        """
        batch = []
        while len(batch) < self.batch_size:
            # 1) Sample the word
            user_idx, product_idx, text_list = self.dataset.review.data[self.random_idx[self.index]]
            word_idx = text_list[self.sub_index]

            data = [user_idx, product_idx, word_idx]
            for pr in self.product_relations:
                # -1 mask remove
                if len(getattr(self.dataset, pr).data[product_idx]) <= 0: data.append(-1)
                else: data.append(random.choice(getattr(self.dataset, pr).data[product_idx]))
            batch.append(data)

            self.sub_index += 1
            if self.sub_index >= len(text_list):
                self.sub_index = 0
                
                self.index += 1
                if self.index >= self.total_trn_review: self._has_next = False; break

        return np.array(batch)

    def has_next(self):
        """Has next batch."""
        return self._has_next


class KnowledgeEmbedding(nn.Module):
    def __init__(self, dataset, args):
        super(KnowledgeEmbedding, self).__init__()
        self.embed_size = args.embed_size
        self.num_neg_samples = args.num_neg_samples
        self.device = args.device
        self.l2_lambda = args.l2_lambda

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
            embed = self._entity_embedding(self.entities[e].vocab_size)
            setattr(self, e, embed)

        self.relations = edict(
            purchase=edict(
                et='product',
                et_distrib=self._make_distrib(dataset.review.product_uniform_distrib)),
            mentions=edict(
                et='word',
                et_distrib=self._make_distrib(dataset.review.word_distrib)),
            describe_as=edict(
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
        )
        for r in self.relations:
            embed = self._relation_embedding()
            setattr(self, r, embed)
            bias = self._relation_bias(len(self.relations[r].et_distrib))
            setattr(self, r + '_bias', bias)

    def _entity_embedding(self, vocab_size):
        embed = nn.Embedding(vocab_size + 1, self.embed_size, padding_idx=-1, sparse=False)
        initrange = 0.5 / self.embed_size
        weight = torch.FloatTensor(vocab_size + 1, self.embed_size).uniform_(-initrange, initrange)
        embed.weight = nn.Parameter(weight)
        return embed

    def _relation_embedding(self):
        initrange = 0.5 / self.embed_size
        weight = torch.FloatTensor(1, self.embed_size).uniform_(-initrange, initrange)
        embed = nn.Parameter(weight)
        return embed

    def _relation_bias(self, vocab_size):
        bias = nn.Embedding(vocab_size + 1, 1, padding_idx=-1, sparse=False)
        bias.weight = nn.Parameter(torch.zeros(vocab_size + 1, 1))
        return bias

    def _make_distrib(self, distrib):
        distrib = np.power(np.array(distrib, dtype=np.float), 0.75)
        distrib = distrib / distrib.sum()
        distrib = torch.FloatTensor(distrib).to(self.device)
        return distrib

    def forward(self, batch_idxs):
        loss = self.compute_loss(batch_idxs)
        return loss

    def compute_loss(self, batch_idxs):
        user_idxs = batch_idxs[0, 0]
        product_idxs = batch_idxs[0, 1]
        word_idxs = batch_idxs[0, 2]
        brand_idxs = batch_idxs[0, 3]
        category_idxs = batch_idxs[0, 4]
        rproduct1_idxs = batch_idxs[0, 5]
        rproduct2_idxs = batch_idxs[0, 6]
        rproduct3_idxs = batch_idxs[0, 7]

        regularizations = []

        # user + purchase -> product
        up_loss, up_embeds = self.neg_loss('user', 'purchase', 'product', user_idxs, product_idxs)
        regularizations.extend(up_embeds)
        loss = up_loss

        # user + mentions -> word
        uw_loss, uw_embeds = self.neg_loss('user', 'mentions', 'word', user_idxs, word_idxs)
        regularizations.extend(uw_embeds)
        loss += uw_loss

        # product + describe_as -> word
        pw_loss, pw_embeds = self.neg_loss('product', 'describe_as', 'word', product_idxs, word_idxs)
        regularizations.extend(pw_embeds)
        loss += pw_loss

        # product + produced_by -> brand
        pb_loss, pb_embeds = self.neg_loss('product', 'produced_by', 'brand', product_idxs, brand_idxs)
        if pb_loss is not None:
            regularizations.extend(pb_embeds)
            loss += pb_loss

        # product + belongs_to -> category
        pc_loss, pc_embeds = self.neg_loss('product', 'belongs_to', 'category', product_idxs, category_idxs)
        if pc_loss is not None:
            regularizations.extend(pc_embeds)
            loss += pc_loss

        # product + also_bought -> related_product1
        pr1_loss, pr1_embeds = self.neg_loss('product', 'also_bought', 'related_product', product_idxs, rproduct1_idxs)
        if pr1_loss is not None:
            regularizations.extend(pr1_embeds)
            loss += pr1_loss

        # product + also_viewed -> related_product2
        pr2_loss, pr2_embeds = self.neg_loss('product', 'also_viewed', 'related_product', product_idxs, rproduct2_idxs)
        if pr2_loss is not None:
            regularizations.extend(pr2_embeds)
            loss += pr2_loss

        # product + bought_together -> related_product3
        pr3_loss, pr3_embeds = self.neg_loss('product', 'bought_together', 'related_product', product_idxs, rproduct3_idxs)
        if pr3_loss is not None:
            regularizations.extend(pr3_embeds)
            loss += pr3_loss

        # l2 regularization
        if self.l2_lambda > 0:
            l2_loss = 0.0
            for term in regularizations:
                l2_loss += torch.norm(term)
            loss += self.l2_lambda * l2_loss

        return loss

    def neg_loss(self, entity_head, relation, entity_tail, entity_head_idxs, entity_tail_idxs):
        # Entity tail indices can be -1. Remove these indices. Batch size may be changed!
        mask = entity_tail_idxs >= 0
        fixed_entity_head_idxs = entity_head_idxs[mask]
        fixed_entity_tail_idxs = entity_tail_idxs[mask]
        if fixed_entity_head_idxs.size(0) <= 0:
            return None, []

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

    batch_size = entity_head_idxs.size(0)
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



def extract_embeddings(args):
    """Note that last entity embedding is of size [vocab_size+1, d]."""
    model_file = '{}/transe_model_sd_epoch_{}.ckpt'.format(args.log_dir, args.epochs)
    print('Load embeddings', model_file)
    state_dict = torch.load(model_file, map_location=lambda storage, loc: storage)
    embeds = {
        USER: state_dict['user.weight'].cpu().data.numpy()[:-1],  # Must remove last dummy 'user' with 0 embed.
        PRODUCT: state_dict['product.weight'].cpu().data.numpy()[:-1],
        WORD: state_dict['word.weight'].cpu().data.numpy()[:-1],
        BRAND: state_dict['brand.weight'].cpu().data.numpy()[:-1],
        CATEGORY: state_dict['category.weight'].cpu().data.numpy()[:-1],
        RPRODUCT: state_dict['related_product.weight'].cpu().data.numpy()[:-1],

        PURCHASE: (
            state_dict['purchase'].cpu().data.numpy()[0],
            state_dict['purchase_bias.weight'].cpu().data.numpy()
        ),
        MENTION: (
            state_dict['mentions'].cpu().data.numpy()[0],
            state_dict['mentions_bias.weight'].cpu().data.numpy()
        ),
        DESCRIBED_AS: (
            state_dict['describe_as'].cpu().data.numpy()[0],
            state_dict['describe_as_bias.weight'].cpu().data.numpy()
        ),
        PRODUCED_BY: (
            state_dict['produced_by'].cpu().data.numpy()[0],
            state_dict['produced_by_bias.weight'].cpu().data.numpy()
        ),
        BELONG_TO: (
            state_dict['belongs_to'].cpu().data.numpy()[0],
            state_dict['belongs_to_bias.weight'].cpu().data.numpy()
        ),
        ALSO_BOUGHT: (
            state_dict['also_bought'].cpu().data.numpy()[0],
            state_dict['also_bought_bias.weight'].cpu().data.numpy()
        ),
        ALSO_VIEWED: (
            state_dict['also_viewed'].cpu().data.numpy()[0],
            state_dict['also_viewed_bias.weight'].cpu().data.numpy()
        ),
        BOUGHT_TOGETHER: (
            state_dict['bought_together'].cpu().data.numpy()[0],
            state_dict['bought_together_bias.weight'].cpu().data.numpy()
        ),
    }

    save_embed(args.dataset, embeds)

def train(args):
    dataset = load_dataset(args.dataset)
    KG_dataloader = Review_KGemb_dataloader(args, dataset, args.batch_size)

    model = KnowledgeEmbedding(dataset, args).to(args.device)
    logger.info('Parameters:' + str([i[0] for i in model.named_parameters()]))
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    steps = 0 

    for epoch in range(1, args.epochs + 1):
        KG_dataloader.reset()
        while KG_dataloader.has_next():

            lr = args.lr
            for pg in optimizer.param_groups:
                pg['lr'] = lr

            # Get training batch.
            batch_idxs = KG_dataloader.get_batch()
            batch_idxs = torch.from_numpy(batch_idxs).to(args.device)

            # Train model.
            optimizer.zero_grad()
            train_loss = model(batch_idxs)
            train_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()

            steps += 1
            if steps % args.steps_per_checkpoint == 0:
                logger.info('Epoch: {:02d} | '.format(epoch) +
                            'Lr: {:.5f} | '.format(lr))

        torch.save(model.state_dict(), '{}/transe_model_sd_epoch_{}.ckpt'.format(args.log_dir, epoch))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=BEAUTY_CORE, help='One of {beauty, cd, cell, clothing}.')
    parser.add_argument('--name', type=str, default='train_transe_model', help='model name.')
    parser.add_argument('--seed', type=int, default=123, help='random seed.')
    parser.add_argument('--gpu', type=str, default='0', help='gpu device.')
    parser.add_argument('--epochs', type=int, default=1, help='number of epochs to train.')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size.')
    parser.add_argument('--sub_batch_size', type=int, default=5, help='sub batch size.')
    parser.add_argument('--lr', type=float, default=0.5, help='learning rate.')
    parser.add_argument('--weight_decay', type=float, default=0, help='weight decay for adam.')
    parser.add_argument('--l2_lambda', type=float, default=0, help='l2 lambda')
    parser.add_argument('--max_grad_norm', type=float, default=5.0, help='Clipping gradient.')
    parser.add_argument('--embed_size', type=int, default=100, help='knowledge embedding size.')
    parser.add_argument('--num_neg_samples', type=int, default=5, help='number of negative samples.')
    parser.add_argument('--gradient_plot',  type=str, default='gradient_plot/', help='number of negative samples.')
    parser.add_argument('--steps_per_checkpoint', type=int, default=200, help='Number of steps for checkpoint.')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    args.device = torch.device('cuda:0') if torch.cuda.is_available() else 'cpu'

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

