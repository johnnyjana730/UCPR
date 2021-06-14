from __future__ import absolute_import, division, print_function

import os
import sys
import argparse
from math import log
from tqdm import tqdm
from copy import deepcopy
import numpy as np
import gzip
import pickle
import random
from datetime import datetime
# import matplotlib.pyplot as plt
import torch

from utils import *
from data_utils import AmazonDataset


class KnowledgeGraph(object):

    def __init__(self, args, dataset):
        self.G = dict()

        self._load_entities(dataset)

        self.att_core = args.att_core
        self.item_core = args.item_core

        self.user_purchase_dict = {}
        self.item_interact_dict = {}
        self.knowledge_dict = {}
        self.knowledge_dict[USER] = {}
        self.knowledge_dict[WORD] = {}
        self.knowledge_dict[PRODUCT] = {}
        self.knowledge_dict[BRAND] = {}
        self.knowledge_dict[CATEGORY] = {}
        self.knowledge_dict[RPRODUCT] = {}

        self._load_reviews(dataset)

        # print('self.word_core_review_dict = ', self.word_core_review_dict)
        # total = 0
        # for k, v in self.word_core_review_dict.items():
        #     total += v
        # print('total/self.word_core_review_dict = ', total/len(self.word_core_review_dict))

        self._core_reviews(dataset)
        # self.knowledge_dict = {}
        self._load_knowledge(dataset)
        
        # for relation in [PRODUCED_BY, BELONG_TO]:
        #     total = 0
        #     for k, v in self.knowledge_dict[relation].items():
        #         total += v
        #     print('relation = ', relation)
        #     print('total/self.core_review_dict = ', total/len(self.knowledge_dict[relation]))
        #     # input()

        # for relation in [ALSO_BOUGHT, ALSO_VIEWED, BOUGHT_TOGETHER]:
        #     total = 0
        #     for k, v in self.knowledge_dict[relation].items():
        #         total += v
        #     print('relation = ', relation)
        #     print('total/self.core_review_dict = ', total/len(self.knowledge_dict[relation]))
            # input()

        self._core_load_knowledge(dataset)
        self._clean()
        self.top_matches = None

    def _load_entities(self, dataset):
        print('Load entities...')
        num_nodes = 0
        for entity in get_entities():
            self.G[entity] = {}
            vocab_size = getattr(dataset, entity).vocab_size
            for eid in range(vocab_size):
                self.G[entity][eid] = {r: set() for r in get_relations(entity)}
            num_nodes += vocab_size
        print('Total {:d} nodes.'.format(num_nodes))

    def _load_reviews(self, dataset, word_tfidf_threshold=0.1, word_freq_threshold=5000):
        print('Load reviews...')
        # (1) Filter words by both tfidf and frequency.
        vocab = dataset.word.vocab
        reviews = [d[2] for d in dataset.review.data]
        review_tfidf = compute_tfidf_fast(vocab, reviews)
        distrib = dataset.review.word_distrib

        num_edges = 0
        all_removed_words = []

        for rid, data in enumerate(dataset.review.data):
            uid, pid, review = data
            doc_tfidf = review_tfidf[rid].toarray()[0]
            remained_words = [wid for wid in set(review)
                              if doc_tfidf[wid] >= word_tfidf_threshold
                              and distrib[wid] <= word_freq_threshold]
            removed_words = set(review).difference(remained_words)  # only for visualize
            removed_words = [vocab[wid] for wid in removed_words]
            all_removed_words.append(removed_words)
            # if len(remained_words) <= 0:
            #     continue

            # (2) Add edges.
            # self._add_edge(USER, uid, PURCHASE, PRODUCT, pid)
            self.update_dic(pid, self.knowledge_dict[PRODUCT])

            # num_edges += 2
            for wid in remained_words:
                # self._add_edge(USER, uid, MENTION, WORD, wid)
                # self._add_edge(PRODUCT, pid, DESCRIBED_AS, WORD, wid)
                # num_edges += 4

                self.update_dic(wid, self.knowledge_dict[WORD])

        for rid, data in enumerate(dataset.review.data):
            uid, pid, review = data
            doc_tfidf = review_tfidf[rid].toarray()[0]
            remained_words = [wid for wid in set(review)
                              if doc_tfidf[wid] >= word_tfidf_threshold
                              and distrib[wid] <= word_freq_threshold]
            removed_words = set(review).difference(remained_words)  # only for visualize
            removed_words = [vocab[wid] for wid in removed_words]
            all_removed_words.append(removed_words)
            # if len(remained_words) <= 0:
            #     continue
            # print('self.item_interact_dict[pid] = ', self.item_interact_dict[pid])
            # if self.item_interact_dict[pid] <= self.item_core:
            #     continue

            self.update_dic(uid, self.knowledge_dict[USER])

        # print('Total {:d} review edges.'.format(num_edges))

        # with open('./tmp/review_removed_words.txt', 'w') as f:
        #     f.writelines([' '.join(words) + '\n' for words in all_removed_words])

    def _core_reviews(self, dataset, word_tfidf_threshold=0.1, word_freq_threshold=5000):

        print('Load reviews...')
        # (1) Filter words by both tfidf and frequency.
        vocab = dataset.word.vocab
        reviews = [d[2] for d in dataset.review.data]
        review_tfidf = compute_tfidf_fast(vocab, reviews)
        distrib = dataset.review.word_distrib

        num_edges = 0
        all_removed_words = []

        for rid, data in enumerate(dataset.review.data):
            uid, pid, review = data
            # if dataset.item_fre[pid] <= self.item_core:
            #     continue

            doc_tfidf = review_tfidf[rid].toarray()[0]
            remained_words = [wid for wid in set(review)
                              if doc_tfidf[wid] >= word_tfidf_threshold
                              and distrib[wid] <= word_freq_threshold]
            removed_words = set(review).difference(remained_words)  # only for visualize
            removed_words = [vocab[wid] for wid in removed_words]
            all_removed_words.append(removed_words)
            # if len(remained_words) <= 0:
            #     continue

            # (2) Add edges.

            self._add_edge(USER, uid, PURCHASE, PRODUCT, pid)
            num_edges += 2
            for wid in remained_words:
                if self.knowledge_dict[WORD][wid] <= 3000 and self.knowledge_dict[WORD][wid] > self.item_core:
                    self._add_edge(USER, uid, MENTION, WORD, wid)
                    self._add_edge(PRODUCT, pid, DESCRIBED_AS, WORD, wid)
                    num_edges += 4

        print('Total {:d} review edges.'.format(num_edges))

        # with open('./tmp/review_removed_words.txt', 'w') as f:
        #     f.writelines([' '.join(words) + '\n' for words in all_removed_words])

    def _load_knowledge(self, dataset):
        for relation in [PRODUCED_BY, BELONG_TO, ALSO_BOUGHT, ALSO_VIEWED, BOUGHT_TOGETHER]:
            print('Load knowledge {}...'.format(relation))

            data = getattr(dataset, relation).data
            num_edges = 0
            for pid, eids in enumerate(data):

                if len(eids) <= 0:
                    continue 
                for eid in set(eids):
                    # if eid not in self.knowledge_dict[relation]:
                    #     self.knowledge_dict[relation][eid] = 0
                    # self.knowledge_dict[relation][eid] += 1
                    # print('eid = ', eid)
                    if relation == PRODUCED_BY:
                        self.update_dic(pid, self.knowledge_dict[PRODUCT])
                        self.update_dic(eid, self.knowledge_dict[BRAND])
                    elif relation == BELONG_TO:
                        self.update_dic(pid, self.knowledge_dict[PRODUCT])
                        self.update_dic(eid, self.knowledge_dict[CATEGORY])
                    elif relation == ALSO_BOUGHT:
                        self.update_dic(pid, self.knowledge_dict[PRODUCT])
                        self.update_dic(eid, self.knowledge_dict[RPRODUCT])
                    elif relation == ALSO_VIEWED:
                        self.update_dic(pid, self.knowledge_dict[PRODUCT])
                        self.update_dic(eid, self.knowledge_dict[RPRODUCT])
                    elif relation == BOUGHT_TOGETHER:
                        self.update_dic(pid, self.knowledge_dict[PRODUCT])
                        self.update_dic(eid, self.knowledge_dict[RPRODUCT])
                
            print('Total {:d} {:s} edges.'.format(num_edges, relation))

    def _core_load_knowledge(self, dataset):
        for relation in [PRODUCED_BY, BELONG_TO]:
            print('Load knowledge {}...'.format(relation))
            data = getattr(dataset, relation).data
            num_edges = 0
            for pid, eids in enumerate(data):
                if len(eids) <= 0:
                    continue 
                for eid in set(eids):
                    if relation == PRODUCED_BY and self.knowledge_dict[BRAND][eid] >= 3000: continue
                    elif relation == BELONG_TO and self.knowledge_dict[CATEGORY][eid] >= 3000: continue
                    et_type = get_entity_tail(PRODUCT, relation)
                    self._add_edge(PRODUCT, pid, relation, et_type, eid)
                    num_edges += 2
            print('Total {:d} {:s} edges.'.format(num_edges, relation))

        for relation in [ALSO_BOUGHT, ALSO_VIEWED, BOUGHT_TOGETHER]:
            print('Load knowledge {}...'.format(relation))
            data = getattr(dataset, relation).data
            num_edges = 0
            for pid, eids in enumerate(data):
                if len(eids) <= 0:
                    continue
                for eid in set(eids):
                    if self.knowledge_dict[RPRODUCT][eid] >= 3000: continue
                    et_type = get_entity_tail(PRODUCT, relation)
                    self._add_edge(PRODUCT, pid, relation, et_type, eid)
                    num_edges += 2
            print('Total {:d} {:s} edges.'.format(num_edges, relation))

    def _add_edge(self, etype1, eid1, relation, etype2, eid2):
        # if eid2 in self.G[etype1][eid1][relation]:
        #     # if relation == ALSO_VIEWED:
        #     print('already in eid2', etype1, eid1, relation, eid2)
        # if eid1 in self.G[etype2][eid2][relation]:
        #     # if relation == ALSO_VIEWED:
        #     print('already in eid1', etype2, eid2, relation, eid1)

        self.G[etype1][eid1][relation].add(eid2)
        self.G[etype2][eid2][relation].add(eid1)

    def _clean(self):
        print('Remove duplicates...')
        for etype in self.G:
            for eid in self.G[etype]:
                for r in self.G[etype][eid]:
                    data = self.G[etype][eid][r]
                    data = tuple(sorted(set(data)))
                    self.G[etype][eid][r] = data

    def compute_degrees(self):
        print('Compute node degrees...')
        self.degrees = {}
        self.max_degree = {}
        for etype in self.G:
            self.degrees[etype] = {}
            for eid in self.G[etype]:
                count = 0
                for r in self.G[etype][eid]:
                    count += len(self.G[etype][eid][r])
                self.degrees[etype][eid] = count

    def get(self, eh_type, eh_id=None, relation=None):
        data = self.G
        if eh_type is not None:
            data = data[eh_type]
        if eh_id is not None:
            data = data[eh_id]
        if relation is not None:
            data = data[relation]
        return data

    def __call__(self, eh_type, eh_id=None, relation=None):
        return self.get(eh_type, eh_id, relation)

    def get_tails(self, entity_type, entity_id, relation):
        return self.G[entity_type][entity_id][relation]

    def get_tails_given_user(self, entity_type, entity_id, relation, user_id):
        """ Very important!
        :param entity_type:
        :param entity_id:
        :param relation:
        :param user_id:
        :return:
        """
        tail_type = KG_RELATION[entity_type][relation]
        tail_ids = self.G[entity_type][entity_id][relation]
        if tail_type not in self.top_matches:
            return tail_ids
        top_match_set = set(self.top_matches[tail_type][user_id])
        top_k = len(top_match_set)
        if len(tail_ids) > top_k:
            tail_ids = top_match_set.intersection(tail_ids)
        return list(tail_ids)

    def trim_edges(self):
        degrees = {}
        for entity in self.G:
            degrees[entity] = {}
            for eid in self.G[entity]:
                for r in self.G[entity][eid]:
                    if r not in degrees[entity]:
                        degrees[entity][r] = []
                    degrees[entity][r].append(len(self.G[entity][eid][r]))

        for entity in degrees:
            for r in degrees[entity]:
                tmp = sorted(degrees[entity][r], reverse=True)
                print(entity, r, tmp[:10])

    def set_top_matches(self, u_u_match, u_p_match, u_w_match):
        self.top_matches = {
            USER: u_u_match,
            PRODUCT: u_p_match,
            WORD: u_w_match,
        }

    def heuristic_search(self, uid, pid, pattern_id, trim_edges=False):
        if trim_edges and self.top_matches is None:
            raise Exception('To enable edge-trimming, must set top_matches of users first!')
        if trim_edges:
            _get = lambda e, i, r: self.get_tails_given_user(e, i, r, uid)
        else:
            _get = lambda e, i, r: self.get_tails(e, i, r)

        pattern = PATH_PATTERN[pattern_id]
        paths = []
        if pattern_id == 1:  # OK
            wids_u = set(_get(USER, uid, MENTION))  # USER->MENTION->WORD
            wids_p = set(_get(PRODUCT, pid, DESCRIBED_AS))  # PRODUCT->DESCRIBE->WORD
            intersect_nodes = wids_u.intersection(wids_p)
            paths = [(uid, x, pid) for x in intersect_nodes]
        elif pattern_id in [11, 12, 13, 14, 15, 16, 17]:
            pids_u = set(_get(USER, uid, PURCHASE))  # USER->PURCHASE->PRODUCT
            pids_u = pids_u.difference([pid])  # exclude target product
            nodes_p = set(_get(PRODUCT, pid, pattern[3][0]))  # PRODUCT->relation->node2
            if pattern[2][1] == USER:
                nodes_p.difference([uid])
            for pid_u in pids_u:
                relation, entity_tail = pattern[2][0], pattern[2][1]
                et_ids = set(_get(PRODUCT, pid_u, relation))  # USER->PURCHASE->PRODUCT->relation->node2
                intersect_nodes = et_ids.intersection(nodes_p)
                tmp_paths = [(uid, pid_u, x, pid) for x in intersect_nodes]
                paths.extend(tmp_paths)
        elif pattern_id == 18:
            wids_u = set(_get(USER, uid, MENTION))  # USER->MENTION->WORD
            uids_p = set(_get(PRODUCT, pid, PURCHASE))  # PRODUCT->PURCHASE->USER
            uids_p = uids_p.difference([uid])  # exclude source user
            for uid_p in uids_p:
                wids_u_p = set(_get(USER, uid_p, MENTION))  # PRODUCT->PURCHASE->USER->MENTION->WORD
                intersect_nodes = wids_u.intersection(wids_u_p)
                tmp_paths = [(uid, x, uid_p, pid) for x in intersect_nodes]
                paths.extend(tmp_paths)

        return paths


    def update_dic(self, key, data_dic):
        if key not in data_dic:
            data_dic[key] = 0
        data_dic[key] += 1


def check_test_path(dataset_str, kg):
    # Check if there exists at least one path for any user-product in test set.
    test_user_products = load_labels(dataset_str, 'test')
    for uid in test_user_products:
        for pid in test_user_products[uid]:
            count = 0
            for pattern_id in [1, 11, 12, 13, 14, 15, 16, 17, 18]:
                tmp_path = kg.heuristic_search(uid, pid, pattern_id)
                count += len(tmp_path)
            if count == 0:
                print(uid, pid)

