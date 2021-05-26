from __future__ import absolute_import, division, print_function

import os
import numpy as np
import gzip
import pickle
from easydict import EasyDict as edict
import random
from utils import *
import pandas as pd
import random

class RW_based_dataset(object):
    """This class is used to load data files and save in the instance."""

    def __init__(self, args, data_dir, set_name='train'):
        self.data_dir = data_dir
        if not self.data_dir.endswith('/'):
            self.data_dir += '/'
        self.review_file = 'review_' + set_name + '.txt.gz'
        self.user_core_th = args.user_core_th

        self.load_entities()
        self.load_kg()

        self.load_reviews()
        self.select_user_th()

    def select_user_th(self):
        user_counter_dict = {}
        self.item_fre = {}

        for rid, data in enumerate(self.review.data):
            uid, pid, review = data
            if uid not in user_counter_dict: user_counter_dict[uid] = 0
            user_counter_dict[uid] += 1
            if pid not in self.item_fre: self.item_fre[pid] = 0
            self.item_fre[pid] += 1

        user_counter_dict = sorted(user_counter_dict.items(), key=lambda x: x[1], reverse=True)
        self.user_counter_dict = user_counter_dict

        self.total_user_list = [user_set[0] for user_set in user_counter_dict]

        self.core_user_list = [user_set[0] for user_set in user_counter_dict if user_set[1] >= self.user_core_th]
        print('self.core_user_list = ', len(self.core_user_list))

    def load_reviews(self):
        """Load user-product reviews from train/test data files.
        Create member variable `review` associated with following attributes:
        - `data`: list of tuples (user_idx, product_idx, [word_idx...]).
        - `size`: number of reviews.
        - `product_distrib`: product vocab frequency among all eviews.
        - `product_uniform_distrib`: product vocab frequency (all 1's)
        - `word_distrib`: word vocab frequency among all reviews.
        - `word_count`: number of words (including duplicates).
        - `review_distrib`: always 1.
        """
        print('load review ')

        review_data = []  # (user_idx, product_idx, [word1_idx,...,wordn_idx])

        # for training kg emb
        product_distrib = np.zeros(self.product.vocab_size)
        word_distrib = np.zeros(self.word.vocab_size)

        for line in self._load_file(self.review_file):
            arr = line.split('\t')

            user_idx, product_idx = int(arr[0]), int(arr[1])
            word_indices = [int(i) for i in arr[2].split(' ')]  # list of word idx

            review_data.append((user_idx, product_idx, word_indices))

            for wi in word_indices:
                word_distrib[wi] += 1
            product_distrib[product_idx] += 1

        self.review = edict(
                data=review_data,
                size=len(review_data),
                word_distrib=word_distrib,
                product_distrib=product_distrib,
                product_uniform_distrib=np.ones(self.product.vocab_size))

        for revi_tst in self.review.data[:5]:
            print('review tst = ', revi_tst)

        print('Load review of size', self.review.size)


    def load_entities(self):
        print('load entities ')
        entity_files = edict(
                user='users.txt.gz',
                product='product.txt.gz',
                word='vocab.txt.gz',
                related_product='related_product.txt.gz',
                brand='brand.txt.gz',
                category='category.txt.gz')

        for name in entity_files:
            vocab = self._load_file(entity_files[name])
            setattr(self, name, edict(vocab=vocab, vocab_size=len(vocab)))

            print(name, 'enti voc = ', getattr(self, name).vocab[:10])

    def load_kg(self):
        product_relations = edict(
                produced_by=('brand_p_b.txt.gz', self.brand),  # (filename, entity_tail)
                belongs_to=('category_p_c.txt.gz', self.category),
                also_bought=('also_bought_p_p.txt.gz', self.related_product),
                also_viewed=('also_viewed_p_p.txt.gz', self.related_product),
                bought_together=('bought_together_p_p.txt.gz', self.related_product))

        for name in product_relations:            
            head_map = edict(
                    data=[],
                    et_vocab=product_relations[name][1].vocab, #copy of brand, catgory ... 's vocab 
                    et_distrib=np.zeros(product_relations[name][1].vocab_size) #[1] means self.brand ..
                    )

            for line in self._load_file(product_relations[name][0]): #[0] means brand_p_b.txt.gz ..
                tails = []
                for x in line.split(' '):  # some lines may be empty
                    if len(x) > 0:
                        tails.append(int(x))
                        head_map.et_distrib[int(x)] += 1
                head_map.data.append(tails)

            setattr(self, name, head_map)
            print('Load', name, 'of size', len(head_map.data))

            head_map_tmp = getattr(self, name)
            print(name, 'et_vocab = ',  head_map_tmp.et_vocab[:10], 'data = ', head_map_tmp.data[:10])

    def _load_file(self, filename):
        with gzip.open(self.data_dir + filename, 'r') as f:
            return [line.decode('utf-8').strip() for line in f]


class KG_based_dataset(object):
    """This class is used to load data files and save in the instance."""

    def __init__(self, args, data_dir, set_name='train', word_sampling_rate=1e-4):
        
        self.data_dir = data_dir
        self.user_top_k = args.user_top_k

        self.load_entities()
        self.select_user_th()


    def load_rating_kg(self):

        rating_file = self.data_dir + '/ratings_final'
        self.rating_np = np.load(rating_file + '.npy')

        kg_file = self.data_dir + '/kg_final'
        self.kg_np = np.load(kg_file + '.npy')

    def load_entities(self):

        self.load_rating_kg()

        n_user = max(set(self.rating_np[:, 0])) + 1
        n_item = max(set(self.rating_np[:, 1])) + 1
        n_attribute = max(set([row[0] for row in self.kg_np] + \
                        [row[2] for row in self.kg_np])) + 1

        print('n_user = ', n_user)
        print('n_item = ', n_item)
        print('n_attribute = ', n_attribute)

        self.entity_list = {}
        self.rela_list = {}

        self.entity_list[USER] = edict(vocab_size=n_user)
        self.entity_list[PRODUCT] = edict(vocab_size= n_user + n_item)
        self.entity_list['attribute'] =  edict(vocab_size= n_user + n_attribute)

        print('self.entity_list = ', self.entity_list)

        setattr(self, USER, edict(vocab_size=n_user))
        setattr(self, PRODUCT, edict(vocab_size=n_user + n_item))
        setattr(self, 'attribute', edict(vocab_size= n_user + n_attribute))

        self.rela_list = set([PURCHASE, PADDING, SELF_LOOP])

        self.et_idx2ty = {}

        for row in self.rating_np:
            self.et_idx2ty[row[0]] = USER
            self.et_idx2ty[row[1] + n_user] = PRODUCT

        for row in self.kg_np:
            if (row[0] + n_user) not in self.et_idx2ty:
                self.et_idx2ty[row[0] + n_user] = 'attribute'
            if (row[2] + n_user) not in self.et_idx2ty:
                self.et_idx2ty[row[2] + n_user] = 'attribute'
            self.rela_list.add(str(row[1]))

    def select_user_th(self):

        trn_data = pd.read_csv(f'{self.data_dir}/train_pd.csv',index_col=None)
        trn_data = trn_data.drop(trn_data.columns[0], axis=1)
        trn_data = trn_data[['user','item','like']].values

        user_counter_dict = {}
        self.item_fre = {}

        for row in trn_data:
            if row[2] == 1:
                if row[0] not in user_counter_dict: user_counter_dict[row[0]] = 0
                user_counter_dict[row[0]] += 1
                if row[1] not in self.item_fre: self.item_fre[row[1]] = 0
                self.item_fre[row[1]] += 1

        user_counter_dict = sorted(user_counter_dict.items(), key=lambda x: x[1], reverse=True)
        self.user_counter_dict = user_counter_dict

        self.core_user_list = [user_set[0] for user_set in user_counter_dict[:self.user_top_k]]
        self.total_user_list = [user_set[0] for user_set in user_counter_dict]



