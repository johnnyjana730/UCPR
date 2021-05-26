from __future__ import absolute_import, division, print_function

import os
import sys
import argparse
from math import log
from tqdm import tqdm
from copy import deepcopy
import pandas as pd
import numpy as np
import gzip
import pickle
import random
from datetime import datetime
# import matplotlib.pyplot as plt
import torch

from utils import *


class RW_based_KG(object):

    def __init__(self, args, dataset):

        self.KG = dict()
        # self.KG_relation = dict()

        self.KG_rela = {}
        self.KG_rela_count = {}
        for relation in [PURCHASE, MENTION, DESCRIBED_AS, PRODUCED_BY, BELONG_TO, ALSO_BOUGHT, ALSO_VIEWED, BOUGHT_TOGETHER]:
            self.KG_rela[relation] = []
            self.KG_rela_count[relation] = 0

        self.att_th_lower = args.att_th_lower
        self.att_th_upper = args.att_th_upper

        self.kg_fre_dict = {}
        for et in [USER, WORD, PRODUCT, BRAND, CATEGORY, RPRODUCT]:
            self.kg_fre_dict[et] = {}

        # precalculated frequency
        self._fre_pur_men_des(dataset)
        self._fre_pro_be_bou(dataset)

        self._load_entities(dataset)
        self._kg_pur_men_des(dataset)
        self._kg_pro_be_bou(dataset)

        self._clean()

        self.check_kg_statistic()

    def _fre_pur_men_des(self, dataset, word_tfidf_threshold=0.1, word_freq_threshold=5000):
        print('Load reviews...')
        # (1) Filter words by both tfidf and frequency.

        reviews = [d[2] for d in dataset.review.data]
        review_tfidf = compute_tfidf_fast(dataset.word.vocab, reviews)

        for rid, data in enumerate(dataset.review.data):
            uid, pid, review = data

            doc_tfidf = review_tfidf[rid].toarray()[0]
            remained_words = [wid for wid in set(review) if doc_tfidf[wid] >= word_tfidf_threshold and dataset.review.word_distrib[wid] <= word_freq_threshold]

            self.update_dic(uid, self.kg_fre_dict[USER])
            self.update_dic(pid, self.kg_fre_dict[PRODUCT])

            for wid in remained_words:
                self.update_dic(wid, self.kg_fre_dict[WORD])

    def _fre_pro_be_bou(self, dataset):
        for relation in [PRODUCED_BY, BELONG_TO, ALSO_BOUGHT, ALSO_VIEWED, BOUGHT_TOGETHER]:
            print('Load knowledge {}...'.format(relation))
            data = getattr(dataset, relation).data
            num_edges = 0
            for pid, eids in enumerate(data):
                if len(eids) <= 0: continue 

                for eid in set(eids):
                    if relation == PRODUCED_BY:
                        self.update_dic(pid, self.kg_fre_dict[PRODUCT])
                        self.update_dic(eid, self.kg_fre_dict[BRAND])
                    elif relation == BELONG_TO:
                        self.update_dic(pid, self.kg_fre_dict[PRODUCT])
                        self.update_dic(eid, self.kg_fre_dict[CATEGORY])
                    elif relation == ALSO_BOUGHT:
                        self.update_dic(pid, self.kg_fre_dict[PRODUCT])
                        self.update_dic(eid, self.kg_fre_dict[RPRODUCT])
                    elif relation == ALSO_VIEWED:
                        self.update_dic(pid, self.kg_fre_dict[PRODUCT])
                        self.update_dic(eid, self.kg_fre_dict[RPRODUCT])
                    elif relation == BOUGHT_TOGETHER:
                        self.update_dic(pid, self.kg_fre_dict[PRODUCT])
                        self.update_dic(eid, self.kg_fre_dict[RPRODUCT])

    def _load_entities(self, dataset):
        print('Load entities...')
        num_nodes = 0
        for entity in [USER, WORD, PRODUCT, BRAND, CATEGORY, RPRODUCT]:
            self.KG[entity] = {}
            vocab_size = getattr(dataset, entity).vocab_size

            for eid in range(vocab_size):
                self.KG[entity][eid] = {r: set() for r in list(KG_RELATION[entity].keys())}

            num_nodes += vocab_size
        print('Total {:d} nodes.'.format(num_nodes))

    def _kg_pur_men_des(self, dataset, word_tfidf_threshold=0.1, word_freq_threshold=5000):

        print('Load reviews...')

        reviews = [d[2] for d in dataset.review.data]
        review_tfidf = compute_tfidf_fast(dataset.word.vocab, reviews)


        for rid, data in enumerate(dataset.review.data):
            uid, pid, review = data

            doc_tfidf = review_tfidf[rid].toarray()[0]
            remained_words = [wid for wid in set(review) if doc_tfidf[wid] >= word_tfidf_threshold and dataset.review.word_distrib[wid] <= word_freq_threshold]

            # (2) Add edges.

            self._add_edge(USER, uid, PURCHASE, PRODUCT, pid)

            for wid in remained_words:
                if self.kg_fre_dict[WORD][wid] <= self.att_th_upper and self.kg_fre_dict[WORD][wid] > self.att_th_lower:
                    self._add_edge(USER, uid, MENTION, WORD, wid)
                    self._add_edge(PRODUCT, pid, DESCRIBED_AS, WORD, wid)


    def _kg_pro_be_bou(self, dataset):
        for relation in [PRODUCED_BY, BELONG_TO, ALSO_BOUGHT, ALSO_VIEWED, BOUGHT_TOGETHER]:
            print('Load knowledge {}...'.format(relation))
            data = getattr(dataset, relation).data

            for pid, eids in enumerate(data):
                if len(eids) <= 0: continue 
                for eid in set(eids):
                    if relation == PRODUCED_BY and self.kg_fre_dict[BRAND][eid] >= self.att_th_upper: continue
                    elif relation == BELONG_TO and self.kg_fre_dict[CATEGORY][eid] >= self.att_th_upper: continue
                    elif relation in [ALSO_BOUGHT, ALSO_VIEWED, BOUGHT_TOGETHER] and self.kg_fre_dict[RPRODUCT][eid] >= self.att_th_upper: continue

                    et_type = get_entity_tail(PRODUCT, relation)
                    self._add_edge(PRODUCT, pid, relation, et_type, eid)

    def check_kg_statistic(self):

        print('relation = ', self.KG_rela)
        print('relation count = ', self.KG_rela_count)

        varyious_dict = {}
        for e_type in [USER, PRODUCT, WORD, RPRODUCT, BRAND, CATEGORY]:
            varyious_dict[e_type] = {}
            for e1, v_ in self.KG[e_type].items():
                for rla, v in v_.items():
                    if len(v) > 0:
                        if rla not in varyious_dict[e_type]: varyious_dict[e_type][rla] = [0, 0]
                        varyious_dict[e_type][rla][0] += len(v)
                        varyious_dict[e_type][rla][1] += 1

            for rla, v_ in varyious_dict[e_type].items():
                varyious_dict[e_type][rla] = v_[0] / v_[1]

        print('rela avg link number = ', varyious_dict)

    def _add_edge(self, etype1, eid1, relation, etype2, eid2):

        self.KG[etype1][eid1][relation].add(eid2)
        self.KG[etype2][eid2][relation].add(eid1)

        self.KG_rela_count[relation] += 2

        if len(self.KG_rela[relation]) < 30:
            self.KG_rela[relation].append([etype1, eid1, relation, eid2])

    def _clean(self):
        print('Remove duplicates...')
        for etype in self.KG:
            for eid in self.KG[etype]:
                for r in self.KG[etype][eid]:
                    data = self.KG[etype][eid][r]
                    data = tuple(sorted(set(data)))
                    self.KG[etype][eid][r] = data

    def get(self, eh_type, eh_id=None, relation=None):
        data = self.KG
        if eh_type is not None:
            data = data[eh_type]
        if eh_id is not None:
            data = data[eh_id]
        if relation is not None:
            data = data[relation]
        return data

    def __call__(self, eh_type, eh_id=None, relation=None):
        return self.get(eh_type, eh_id, relation)


    def update_dic(self, key, data_dic):
        if key not in data_dic:
            data_dic[key] = 0
        data_dic[key] += 1



class KG_based_KG(object):

    def __init__(self, args, dataset):
        self.data_dir = DATA_DIR[args.dataset]
        self.dataset = dataset

        self.KG = dict()

        self.KG_rela = {}
        self.KG_rela_count = {}
        for relation in list(dataset.rela_list):
            self.KG_rela[relation] = []
            self.KG_rela_count[relation] = 0

        self.att_th_lower = args.att_th_lower
        self.att_th_upper = args.att_th_upper
        self.att_th_upper = 1000

        self.load_rating_kg()
        self._fre_kg()
        self._kg_pur()
        self._kg_oth_rela()

        self._clean()
        self.check_kg_statistic()

    def load_rating_kg(self):

        rating_file = self.data_dir + '/ratings_final'
        self.rating_np = np.load(rating_file + '.npy')

        kg_file = self.data_dir + '/kg_final'
        self.kg_np = np.load(kg_file + '.npy')

        self.n_user = max(set(self.rating_np[:, 0])) + 1
        self.n_item = max(set(self.rating_np[:, 1])) + 1

    def _fre_kg(self):

        self.kg_fre_dict = {}
        for row in self.kg_np:
            if row[0] not in self.kg_fre_dict:
                self.kg_fre_dict[row[0] + self.n_user] = 0
            if row[2] not in self.kg_fre_dict:
                self.kg_fre_dict[row[2] + self.n_user] = 0
            self.kg_fre_dict[row[0] + self.n_user] += 1
            self.kg_fre_dict[row[2] + self.n_user] += 1


    def _kg_pur(self):

        tst_data = pd.read_csv(f'{self.data_dir}/test_pd.csv',index_col=None)
        tst_data = tst_data.drop(tst_data.columns[0], axis=1)
        tst_data = tst_data[['user','item','like']].values

        tst_user_product = {}
        for row in tst_data:
            user, item, like = row[0], row[1] + self.n_user, row[2]
            if like == 1:
                if user not in tst_user_product:
                    tst_user_product[user] = []
                tst_user_product[user].append(item)


        trn_data = pd.read_csv(f'{self.data_dir}/train_pd.csv',index_col=None)
        trn_data = trn_data.drop(trn_data.columns[0], axis=1)
        trn_data = trn_data[['user','item','like']].values

        for row in trn_data:
            user, item, like = row[0], row[1] + self.n_user, row[2]
            if like == 1:
                # avoid tst product
                if user in tst_user_product and item in tst_user_product[user]: continue
                self.add_new_triplet(user,PURCHASE,item)
                self.add_new_triplet(item,PURCHASE,user)

                if user not in self.kg_fre_dict:
                    self.kg_fre_dict[user] = 0
                if item not in self.kg_fre_dict:
                    self.kg_fre_dict[item] = 0
                self.kg_fre_dict[user] += 1
                self.kg_fre_dict[item] += 1

    def _kg_oth_rela(self):

        for row in self.kg_np:
            if self.kg_fre_dict[row[0] + self.n_user] < self.att_th_upper and self.kg_fre_dict[row[2] + self.n_user] < self.att_th_upper:
                head, rela, tail = row[0] + self.n_user , str(row[1]), row[2] + self.n_user
                head += self.n_user
                tail += self.n_user
                self.add_new_triplet(head,rela,tail)
                self.add_new_triplet(tail,rela,head)

    def add_new_triplet(self, head, relation, tail):
        if self.dataset.et_idx2ty[head] not in self.KG:
            self.KG[self.dataset.et_idx2ty[head]] = {}
        if head not in self.KG[self.dataset.et_idx2ty[head]]:
            self.KG[self.dataset.et_idx2ty[head]][head] = {}
        if relation not in self.KG[self.dataset.et_idx2ty[head]][head]:
            self.KG[self.dataset.et_idx2ty[head]][head][relation] = set()
        self.KG[self.dataset.et_idx2ty[head]][head][relation].add(tail)

        if SELF_LOOP not in self.KG[self.dataset.et_idx2ty[head]][head]:
            self.KG[self.dataset.et_idx2ty[head]][head][SELF_LOOP] = set()
        self.KG[self.dataset.et_idx2ty[head]][head][SELF_LOOP].add(head)

        self.KG_rela_count[relation] += 1

        if len(self.KG_rela[relation]) < 30:
            self.KG_rela[relation].append([self.dataset.et_idx2ty[head], head, relation, tail])


    def check_kg_statistic(self):

        print('relation = ', self.KG_rela)
        print('relation count = ', self.KG_rela_count)

        # self.statistic = {}
        # for k, v in self.kg_fre_dict.items():
        #     if v not in self.statistic:
        #         self.statistic[v] = 0
        #     self.statistic[v] += 1

        # print('self.statistic = ', self.statistic)

        varyious_dict = {}
        for e_type in [USER, PRODUCT, 'attribute']:
            varyious_dict[e_type] = {}
            for e1, v_ in self.KG[e_type].items():
                for rla, v in v_.items():
                    if len(v) > 0:
                        if rla not in varyious_dict[e_type]: varyious_dict[e_type][rla] = [0, 0]
                        varyious_dict[e_type][rla][0] += len(v)
                        varyious_dict[e_type][rla][1] += 1

            for rla, v_ in varyious_dict[e_type].items():
                varyious_dict[e_type][rla] = v_[0] / v_[1]

        print('rela avg link number = ', varyious_dict)


    def _clean(self):
        print('Remove duplicates...')
        for etype in self.KG:
            for eid in self.KG[etype]:
                for r in self.KG[etype][eid]:
                    data = self.KG[etype][eid][r]
                    data = tuple(sorted(set(data)))
                    self.KG[etype][eid][r] = data


    def get(self, eh_type, eh_id=None, relation=None):
        data = self.KG
        if eh_type is not None:
            data = data[eh_type]
        if eh_id is not None:
            data = data[eh_id]
        if relation is not None:
            data = data[relation]
        return data

    def __call__(self, eh_type, eh_id=None, relation=None):
        return self.get(eh_type, eh_id, relation)
