from __future__ import absolute_import, division, print_function

import os
import numpy as np
import gzip
import pickle
from easydict import EasyDict as edict
import random


class AmazonDataset(object):
    """This class is used to load data files and save in the instance."""

    def __init__(self, data_dir, set_name='train', word_sampling_rate=1e-4):
        self.data_dir = data_dir
        if not self.data_dir.endswith('/'):
            self.data_dir += '/'
        self.review_file = set_name + '.txt.gz'
        self.load_entities()
        self.load_product_relations()
        self.load_reviews()
        self.create_word_sampling_rate(word_sampling_rate)

    def _load_file(self, filename):
        with gzip.open(self.data_dir + filename, 'r') as f:
            # In Python 3, must use decode() to convert bytes to string!
            return [line.decode('utf-8').strip() for line in f]

    def load_entities(self):
        """Load 6 global entities from data files:
        `user`, `product`, `word`, `related_product`, `brand`, `category`.
        Create a member variable for each entity associated with attributes:
        - `vocab`: a list of string indicating entity values.
        - `vocab_size`: vocabulary size.
        """
        entity_files = edict(
                user='users.txt.gz',
                product='product.txt.gz',
                word='vocab.txt.gz',
                related_product='related_product.txt.gz',
                brand='brand.txt.gz',
                category='category.txt.gz',
        )
        for name in entity_files:
            vocab = self._load_file(entity_files[name])
            # print('vocab = ', vocab)
            # input()
            setattr(self, name, edict(vocab=vocab, vocab_size=len(vocab)))
            # print('Load', name, 'of size', len(vocab))

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
        review_data = []  # (user_idx, product_idx, [word1_idx,...,wordn_idx])
        product_distrib = np.zeros(self.product.vocab_size)
        word_distrib = np.zeros(self.word.vocab_size)
        word_count = 0
        for line in self._load_file(self.review_file):
            arr = line.split('\t')
            user_idx = int(arr[0])
            product_idx = int(arr[1])
            word_indices = [int(i) for i in arr[2].split(' ')]  # list of word idx
            review_data.append((user_idx, product_idx, word_indices))
            product_distrib[product_idx] += 1
            for wi in word_indices:
                word_distrib[wi] += 1
            word_count += len(word_indices)
        self.review = edict(
                data=review_data,
                size=len(review_data),
                product_distrib=product_distrib,
                product_uniform_distrib=np.ones(self.product.vocab_size),
                word_distrib=word_distrib,
                word_count=word_count,
                review_distrib=np.ones(len(review_data)) #set to 1 now
        )
        print('Load review of size', self.review.size, 'word count=', word_count)

    def load_product_relations(self):
        """Load 5 product -> ? relations:
        - `produced_by`: product -> brand,
        - `belongs_to`: product -> category,
        - `also_bought`: product -> related_product,
        - `also_viewed`: product -> related_product,
        - `bought_together`: product -> related_product,
        Create member variable for each relation associated with following attributes:
        - `data`: list of list of entity_tail indices (can be empty).
        - `et_vocab`: vocabulary of entity_tail (copy from entity vocab).
        - `et_distrib`: frequency of entity_tail vocab.
        """
        product_relations = edict(
                produced_by=('brand_p_b.txt.gz', self.brand),  # (filename, entity_tail)
                belongs_to=('category_p_c.txt.gz', self.category),
                also_bought=('also_bought_p_p.txt.gz', self.related_product),
                also_viewed=('also_viewed_p_p.txt.gz', self.related_product),
                bought_together=('bought_together_p_p.txt.gz', self.related_product),
        )
        for name in product_relations:
            # We save information of entity_tail (et) in each relation.
            # Note that `data` variable saves list of entity_tail indices.
            # The i-th record of `data` variable is the entity_tail idx (i.e. product_idx=i).
            # So for each product-relation, there are always |products| records.
            # print('name = ', name)
            # input()
            # print('product_relations[name][1].vocab, = ', product_relations[name][1].vocab)
            # input()
            
            relation = edict(
                    data=[],
                    et_vocab=product_relations[name][1].vocab, #copy of brand, catgory ... 's vocab 
                    et_distrib=np.zeros(product_relations[name][1].vocab_size) #[1] means self.brand ..
            )
            for line in self._load_file(product_relations[name][0]): #[0] means brand_p_b.txt.gz ..
                # print('line = ', line)
                # input()

                knowledge = []
                for x in line.split(' '):  # some lines may be empty
                    if len(x) > 0:
                        x = int(x)
                        knowledge.append(x)
                        relation.et_distrib[x] += 1
                relation.data.append(knowledge)
            setattr(self, name, relation)
            print('Load', name, 'of size', len(relation.data))

    def create_word_sampling_rate(self, sampling_threshold):
        print('Create word sampling rate')
        self.word_sampling_rate = np.ones(self.word.vocab_size)
        if sampling_threshold <= 0:
            return
        threshold = sum(self.review.word_distrib) * sampling_threshold
        for i in range(self.word.vocab_size):
            if self.review.word_distrib[i] == 0:
                continue
            self.word_sampling_rate[i] = min((np.sqrt(float(self.review.word_distrib[i]) / threshold) + 1) * threshold / float(self.review.word_distrib[i]), 1.0)



class AmazonDataset_userth(object):
    """This class is used to load data files and save in the instance."""

    def __init__(self, args, data_dir, set_name='train', word_sampling_rate=1e-4):
        self.data_dir = data_dir
        if not self.data_dir.endswith('/'):
            self.data_dir += '/'
        self.review_file = set_name + '.txt.gz'
        # self.kg_user_theroshold = args.user_core
        self.user_core_th = args.user_core_th
        self.load_entities()
        self.load_product_relations()
        self.load_reviews()
        self.create_word_sampling_rate(word_sampling_rate)

        self.record_fre_user_enti()

    def record_fre_user_enti(self):
        user_counter_dict = {}
        self.item_fre = {}

        for rid, data in enumerate(self.review.data):
            uid, pid, review = data
            if uid not in user_counter_dict: user_counter_dict[uid] = 0
            user_counter_dict[uid] += 1
            if pid not in self.item_fre: self.item_fre[pid] = 0
            self.item_fre[pid] += 1

        # user_counter_dict = {user:len(train_record[user]) for user in user_list}
        user_counter_dict = sorted(user_counter_dict.items(), key=lambda x: x[1], reverse=True)
        self.user_counter_dict = user_counter_dict
        # user_counter_dict_3000 = user_counter_dict[:4000]

        # for user_set in user_counter_dict:
        #     print('user = ', user_set[0], user_set[1])

        self.select_user_list = [user_set[0] for user_set in user_counter_dict if user_set[1] >= self.user_core_th]
        print('self.select_user_lis = ', len(self.select_user_list))
        # input()
        # user_counter_dict_user_theroshold = user_counter_dict[:self.kg_user_theroshold]
        # self.select_user_list = [user_set[0] for user_set in user_counter_dict_user_theroshold]

        user_counter_dict_query_user_select = user_counter_dict
        self.query_kg_user_filter = [user_set[0] for user_set in user_counter_dict_query_user_select]


    def _load_file(self, filename):
        with gzip.open(self.data_dir + filename, 'r') as f:
            # In Python 3, must use decode() to convert bytes to string!
            return [line.decode('utf-8').strip() for line in f]

    def load_entities(self):
        """Load 6 global entities from data files:
        `user`, `product`, `word`, `related_product`, `brand`, `category`.
        Create a member variable for each entity associated with attributes:
        - `vocab`: a list of string indicating entity values.
        - `vocab_size`: vocabulary size.
        """
        entity_files = edict(
                user='users.txt.gz',
                product='product.txt.gz',
                word='vocab.txt.gz',
                related_product='related_product.txt.gz',
                brand='brand.txt.gz',
                category='category.txt.gz',
        )
        for name in entity_files:
            vocab = self._load_file(entity_files[name])
            # print('vocab = ', vocab)
            # input()
            setattr(self, name, edict(vocab=vocab, vocab_size=len(vocab)))
            # print('Load', name, 'of size', len(vocab))

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
        review_data = []  # (user_idx, product_idx, [word1_idx,...,wordn_idx])
        product_distrib = np.zeros(self.product.vocab_size)
        word_distrib = np.zeros(self.word.vocab_size)
        word_count = 0
        for line in self._load_file(self.review_file):
            arr = line.split('\t')
            user_idx = int(arr[0])
            product_idx = int(arr[1])
            word_indices = [int(i) for i in arr[2].split(' ')]  # list of word idx
            review_data.append((user_idx, product_idx, word_indices))
            product_distrib[product_idx] += 1
            for wi in word_indices:
                word_distrib[wi] += 1
            word_count += len(word_indices)
        self.review = edict(
                data=review_data,
                size=len(review_data),
                product_distrib=product_distrib,
                product_uniform_distrib=np.ones(self.product.vocab_size),
                word_distrib=word_distrib,
                word_count=word_count,
                review_distrib=np.ones(len(review_data)) #set to 1 now
        )
        print('Load review of size', self.review.size, 'word count=', word_count)

    def load_product_relations(self):
        """Load 5 product -> ? relations:
        - `produced_by`: product -> brand,
        - `belongs_to`: product -> category,
        - `also_bought`: product -> related_product,
        - `also_viewed`: product -> related_product,
        - `bought_together`: product -> related_product,
        Create member variable for each relation associated with following attributes:
        - `data`: list of list of entity_tail indices (can be empty).
        - `et_vocab`: vocabulary of entity_tail (copy from entity vocab).
        - `et_distrib`: frequency of entity_tail vocab.
        """
        product_relations = edict(
                produced_by=('brand_p_b.txt.gz', self.brand),  # (filename, entity_tail)
                belongs_to=('category_p_c.txt.gz', self.category),
                also_bought=('also_bought_p_p.txt.gz', self.related_product),
                also_viewed=('also_viewed_p_p.txt.gz', self.related_product),
                bought_together=('bought_together_p_p.txt.gz', self.related_product),
        )
        for name in product_relations:
            # We save information of entity_tail (et) in each relation.
            # Note that `data` variable saves list of entity_tail indices.
            # The i-th record of `data` variable is the entity_tail idx (i.e. product_idx=i).
            # So for each product-relation, there are always |products| records.
            # print('name = ', name)
            # input()
            # print('product_relations[name][1].vocab, = ', product_relations[name][1].vocab)
            # input()
            
            relation = edict(
                    data=[],
                    et_vocab=product_relations[name][1].vocab, #copy of brand, catgory ... 's vocab 
                    et_distrib=np.zeros(product_relations[name][1].vocab_size) #[1] means self.brand ..
            )
            for line in self._load_file(product_relations[name][0]): #[0] means brand_p_b.txt.gz ..
                knowledge = []
                for x in line.split(' '):  # some lines may be empty
                    if len(x) > 0:
                        x = int(x)
                        knowledge.append(x)
                        relation.et_distrib[x] += 1
                relation.data.append(knowledge)
            setattr(self, name, relation)
            print('Load', name, 'of size', len(relation.data))

    def create_word_sampling_rate(self, sampling_threshold):
        print('Create word sampling rate')
        self.word_sampling_rate = np.ones(self.word.vocab_size)
        if sampling_threshold <= 0:
            return
        threshold = sum(self.review.word_distrib) * sampling_threshold
        for i in range(self.word.vocab_size):
            if self.review.word_distrib[i] == 0:
                continue
            self.word_sampling_rate[i] = min((np.sqrt(float(self.review.word_distrib[i]) / threshold) + 1) * threshold / float(self.review.word_distrib[i]), 1.0)

class AmazonDataLoader(object):
    """This class acts as the dataloader for training knowledge graph embeddings."""

    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size

        # print('self.dataset.review = ', self.dataset.review)
        # input()

        self.review_size = self.dataset.review.size
        self.product_relations = ['produced_by', 'belongs_to', 'also_bought', 'also_viewed', 'bought_together']
        self.finished_word_num = 0
        self.reset()

    def reset(self):
        # Shuffle reviews order
        self.review_seq = np.random.permutation(self.review_size)
        self.cur_review_i = 0
        self.cur_word_i = 0
        self._has_next = True

    def get_batch(self):
        """Return a matrix of [batch_size x 8], where each row contains
        (u_id, p_id, w_id, b_id, c_id, rp_id, rp_id, rp_id).
        """
        batch = []
        review_idx = self.review_seq[self.cur_review_i]
        user_idx, product_idx, text_list = self.dataset.review.data[review_idx]
        product_knowledge = {pr: getattr(self.dataset, pr).data[product_idx] for pr in self.product_relations}

        # print('product_knowledge = ', product_knowledge)
        # input()

        while len(batch) < self.batch_size:
            # 1) Sample the word
            word_idx = text_list[self.cur_word_i]
            if random.random() < self.dataset.word_sampling_rate[word_idx]:
                data = [user_idx, product_idx, word_idx]
                for pr in self.product_relations:
                    if len(product_knowledge[pr]) <= 0:
                        data.append(-1)
                    else:
                        data.append(random.choice(product_knowledge[pr]))
                batch.append(data)

            # 2) Move to next word/review
            self.cur_word_i += 1
            self.finished_word_num += 1
            if self.cur_word_i >= len(text_list):
                self.cur_review_i += 1
                if self.cur_review_i >= self.review_size:
                    self._has_next = False
                    break
                self.cur_word_i = 0
                review_idx = self.review_seq[self.cur_review_i]
                user_idx, product_idx, text_list = self.dataset.review.data[review_idx]
                product_knowledge = {pr: getattr(self.dataset, pr).data[product_idx] for pr in self.product_relations}

        return np.array(batch)

    def has_next(self):
        """Has next batch."""
        return self._has_next


class AmazonDataset_graph(object):
    """This class is used to load data files and save in the instance."""

    def __init__(self, data_dir, set_name='train', word_sampling_rate=1e-4):
        self.data_dir = data_dir

        rating_file = self.data_dir + '/ratings_final'
        self.rating_np = np.load(rating_file + '.npy')
        
        kg_file = self.data_dir + '/kg_final'
        self.kg_np = np.load(kg_file + '.npy')

        self.entity_list = {}
        self.rela_list = {}
        # print('kg_file = ', self.kg_np)
        # print('self.rating_np = ', self.rating_np)
        # input()

        # self.review_file = set_name + '.txt.gz'
        self.load_entities()
        # self.load_product_relations()
        # self.load_reviews()
        # self.create_word_sampling_rate(word_sampling_rate)

    def _load_file(self, filename):
        with gzip.open(self.data_dir + filename, 'r') as f:
            # In Python 3, must use decode() to convert bytes to string!
            return [line.decode('utf-8').strip() for line in f]

    def load_entities(self):
        """Load 6 global entities from data files:
        `user`, `product`, `word`, `related_product`, `brand`, `category`.
        Create a member variable for each entity associated with attributes:
        - `vocab`: a list of string indicating entity values.
        - `vocab_size`: vocabulary size.
        """

        n_user = max(set(self.rating_np[:, 0])) + 1
        n_item = max(set(self.rating_np[:, 1])) + 1


        setattr(self, USER, edict(vocab_size=n_user))
        setattr(self, PRODUCT, edict(vocab_size=n_user + n_item))

        self.entity_list[USER] = edict(vocab_size=n_user)
        self.entity_list[PRODUCT] = edict(vocab_size= n_user + n_item)

        n_attribute = max(set([row[0] for row in self.kg_np] + \
                        [row[2] for row in self.kg_np])) + 1
        print('n_attribute = ', n_attribute)
        setattr(self, 'attribute', edict(vocab_size= n_user + n_attribute))
        self.entity_list['attribute'] =  edict(vocab_size= n_user + n_attribute)

        # self.entity_list = [USER, PRODUCT, 'attribute']

        print('self.entity_list = ', self.entity_list)
        self.rela_list = set([PURCHASE])

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
        # row[0] >= n_item-1: tmp_dict[row[1]].add(row[0])
# 
        # tmp_dict = {}
        # for row in self.kg_np:

        #     if row[1] not in tmp_dict: tmp_dict[row[1]] = set()
        #     if row[0] >= n_item-1: tmp_dict[row[1]].add(row[0]) 
        #     if row[2] >= n_item-1: tmp_dict[row[1]].add(row[2]) 

        # for rela, sets in tmp_dict.items():
        #     print('rela = ', rela)
        #     self.entity_list[str(rela)] = edict(vocab_size=max(sets)+1)
        #     setattr(self, str(rela), edict(vocab_size=max(sets)+1))

        # for row in self.rating_np:
        #     self.rela_list[row[1]] = 1


        # print('self.kg_np = ', self.kg_np)
        # entity_files = edict(
        #         user='users.txt.gz',
        #         product='product.txt.gz',
        #         word='vocab.txt.gz',
        #         related_product='related_product.txt.gz',
        #         brand='brand.txt.gz',
        #         category='category.txt.gz',
        # )
        # for name in entity_files:
        #     vocab = self._load_file(entity_files[name])
        #     setattr(self, name, edict(vocab=vocab, vocab_size=len(vocab)))

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
        review_data = []  # (user_idx, product_idx, [word1_idx,...,wordn_idx])
        product_distrib = np.zeros(self.product.vocab_size)
        word_distrib = np.zeros(self.word.vocab_size)
        word_count = 0
        for line in self._load_file(self.review_file):
            arr = line.split('\t')
            user_idx = int(arr[0])
            product_idx = int(arr[1])
            word_indices = [int(i) for i in arr[2].split(' ')]  # list of word idx
            review_data.append((user_idx, product_idx, word_indices))
            product_distrib[product_idx] += 1
            for wi in word_indices:
                word_distrib[wi] += 1
            word_count += len(word_indices)
        self.review = edict(
                data=review_data,
                size=len(review_data),
                product_distrib=product_distrib,
                product_uniform_distrib=np.ones(self.product.vocab_size),
                word_distrib=word_distrib,
                word_count=word_count,
                review_distrib=np.ones(len(review_data)) #set to 1 now
        )
        print('Load review of size', self.review.size, 'word count=', word_count)

    def load_product_relations(self):
        """Load 5 product -> ? relations:
        - `produced_by`: product -> brand,
        - `belongs_to`: product -> category,
        - `also_bought`: product -> related_product,
        - `also_viewed`: product -> related_product,
        - `bought_together`: product -> related_product,
        Create member variable for each relation associated with following attributes:
        - `data`: list of list of entity_tail indices (can be empty).
        - `et_vocab`: vocabulary of entity_tail (copy from entity vocab).
        - `et_distrib`: frequency of entity_tail vocab.
        """
        product_relations = edict(
                produced_by=('brand_p_b.txt.gz', self.brand),  # (filename, entity_tail)
                belongs_to=('category_p_c.txt.gz', self.category),
                also_bought=('also_bought_p_p.txt.gz', self.related_product),
                also_viewed=('also_viewed_p_p.txt.gz', self.related_product),
                bought_together=('bought_together_p_p.txt.gz', self.related_product),
        )
        for name in product_relations:
            # We save information of entity_tail (et) in each relation.
            # Note that `data` variable saves list of entity_tail indices.
            # The i-th record of `data` variable is the entity_tail idx (i.e. product_idx=i).
            # So for each product-relation, there are always |products| records.
            # print('name = ', name)
            # input()
            # print('product_relations[name][1].vocab, = ', product_relations[name][1].vocab)
            # input()
            
            relation = edict(
                    data=[],
                    et_vocab=product_relations[name][1].vocab, #copy of brand, catgory ... 's vocab 
                    et_distrib=np.zeros(product_relations[name][1].vocab_size) #[1] means self.brand ..
            )
            for line in self._load_file(product_relations[name][0]): #[0] means brand_p_b.txt.gz ..
                knowledge = []
                for x in line.split(' '):  # some lines may be empty
                    if len(x) > 0:
                        x = int(x)
                        knowledge.append(x)
                        relation.et_distrib[x] += 1
                relation.data.append(knowledge)
            setattr(self, name, relation)
            print('Load', name, 'of size', len(relation.data))

    def create_word_sampling_rate(self, sampling_threshold):
        print('Create word sampling rate')
        self.word_sampling_rate = np.ones(self.word.vocab_size)
        if sampling_threshold <= 0:
            return
        threshold = sum(self.review.word_distrib) * sampling_threshold
        for i in range(self.word.vocab_size):
            if self.review.word_distrib[i] == 0:
                continue
            self.word_sampling_rate[i] = min((np.sqrt(float(self.review.word_distrib[i]) / threshold) + 1) * threshold / float(self.review.word_distrib[i]), 1.0)


import pandas as pd
import random

class AmazonDataLoader_graph(object):
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
            user = row[0]
            item = row[1] + n_user
            self.data_entity_triplets.append([self.dataset.et_idx2ty[user], PURCHASE,
                 self.dataset.et_idx2ty[item], user, item])
        
        self.kg_entity_triplets =  []
        for row in self.kg_np:
            start, end = row[0] + n_user, row[2] + n_user
            if start < n_user + n_item:
                self.kg_entity_triplets.append([self.dataset.et_idx2ty[start], str(row[1]),
                    self.dataset.et_idx2ty[end], start, end])
            if end < n_user + n_item:
                self.kg_entity_triplets.append([self.dataset.et_idx2ty[end], str(row[1]),
                    self.dataset.et_idx2ty[start], end, start])

        print('len(self.data_entity_triplets) = ', len(self.data_entity_triplets))
        print('len(self.kg_entity_triplets) = ', len(self.kg_entity_triplets))
        # input()
        self.finished_word_num = 0
        self.reset()

    def reset(self):

        random.shuffle(self.kg_entity_triplets) 

        # print('self.kg_entity_triplets[:len(self.data_entity_triplets) * 3] = ', self.kg_entity_triplets[:len(self.data_entity_triplets) * 3])
        # input()
        self.entity_triplets = self.data_entity_triplets + self.kg_entity_triplets[:len(self.data_entity_triplets) * 3]

        self.start_index = 0
        self.end_index = self.batch_size
        random.shuffle(self.entity_triplets)  
        self._has_next = True

    def get_batch(self):
        """Return a matrix of [batch_size x 8], where each row contains
        (u_id, p_id, w_id, b_id, c_id, rp_id, rp_id, rp_id).
        """
        batch = []
        
        # while len(self.entity_triplets) >= self.end_index:
        batch = self.entity_triplets[self.start_index:self.end_index]
        # print('batch = ', batch, batch[0], len(batch))
        # input()
        self.start_index += self.batch_size
        self.end_index += self.batch_size
        if self.end_index >= len(self.entity_triplets):
            self._has_next = False

        return batch


    def has_next(self):
        """Has next batch."""
        return self._has_next

