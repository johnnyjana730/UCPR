from __future__ import absolute_import, division, print_function

import os
import sys
from tqdm import tqdm
import pickle
import random
import torch
from datetime import datetime

from knowledge_graph import KnowledgeGraph
from utils import *

import torch
import torch.nn as nn
import torch.nn.functional as F


import collections
from collections import defaultdict
from collections import Counter 
import time
import multiprocessing
import itertools
from multiprocessing import Pool, cpu_count
import multiprocessing as mp
from functools import partial
import random


# class KGState(object):
#     def __init__(self, embed_size, history_len=1):
#         self.embed_size = embed_size
#         self.history_len = history_len  # mode: one of {full, current}
#         if history_len == 0:
#             self.dim = 2 * embed_size
#         elif history_len == 1:
#             self.dim = 4 * embed_size
#         elif history_len == 2:
#             self.dim = 6 * embed_size
#         else:
#             raise Exception('history length should be one of {0, 1, 2}')

#     def __call__(self, user_embed, node_embed, last_node_embed, last_relation_embed, older_node_embed,
#                  older_relation_embed):
#         if self.history_len == 0:
#             return np.concatenate([user_embed, node_embed])
#         elif self.history_len == 1:
#             return np.concatenate([user_embed, node_embed, last_node_embed, last_relation_embed])
#         elif self.history_len == 2:
#             return np.concatenate([user_embed, node_embed, last_node_embed, last_relation_embed, older_node_embed,
#                                    older_relation_embed])
#         else:
#             raise Exception('mode should be one of {full, current}')

def get_user_triplet_set(kg, user_list, p_hop, n_memory):
    args = {'p_hop': p_hop, 'n_memory': n_memory}
    # print('KG_RELATION = ', KG_RELATION)
    # input()
    # user -> [(hop_0_heads, hop_0_relations, hop_0_tails), (hop_1_heads, hop_1_relations, hop_1_tails), ...]
    user_triplet_set = collections.defaultdict(list)
    # entity_interaction_dict = collections.defaultdict(list)
    user_history_dict = {}

    for user in user_list:
        if user not in user_history_dict:
            user_history_dict[user] = [[USER, user]]

            # if len(kg(USER,user)['purchase']) < 1 or len(kg(USER,user)['mentions']) < 1:
            #     print('USER,user = ', USER,user)
            #     print('kg(USER,user) = ', kg(USER,user), len(kg(USER,user)))
            #     print(len(kg(USER,user)['mentions']), len(kg(USER,user)['purchase']))
            #     input()

    global g_kg
    g_kg = kg
    with mp.Pool(processes=min(mp.cpu_count(), 12)) as pool:
        job = partial(_get_user_triplet_set, p_hop=max(1,args['p_hop']), KG_RELATION = KG_RELATION, n_memory=args['n_memory'], n_neighbor=16)
        for u, u_r_set in pool.starmap(job, user_history_dict.items()):
            # print('u_r_set = ', u_r_set)
            user_triplet_set[u] = u_r_set
            # entity_interaction_dict[u] = u_interaction_list
    del g_kg
    return user_triplet_set

def _get_user_triplet_set(user, history, p_hop=2, KG_RELATION = None, n_memory=32, n_neighbor=16):
    ret = []
    entity_interaction_list = []

    # relations_nodes = self.kg(curr_node_type, curr_node_id)
    # candidate_acts = []  # list of tuples of (relation, node_type, node_id)
    # visited_nodes = set([(v[1], v[2]) for v in path])
    # for r in relations_nodes:
    #     next_node_type = KG_RELATION[curr_node_type][r]
    #     next_node_ids = relations_nodes[r]
    #     next_node_ids = [n for n in next_node_ids if (next_node_type, n) not in visited_nodes]  # filter
    #     candidate_acts.extend(zip([r] * len(next_node_ids), next_node_ids))
    # print('user, before ret = ', user, ret)
    # input() 

    for h in range(max(1,p_hop)):
        memories_h = []
        memories_r = []
        memories_t = []

        if h == 0:
            tails_of_last_hop = history
        else:
            tails_of_last_hop = ret[-1][2]

        for entity_type, entity in tails_of_last_hop:
            # if entity_type == 'product':
            # print('entity_type = ', entity_type)
            # print('g_kg(entity_type,entity).items() = ', g_kg(entity_type,entity).items())
            # input()
            tmp_list = []
            for k_, v_set in g_kg(entity_type,entity).items():
                # print('k_, v_set = ', k_, v_set)
                if k_ == 'mentions' or k_ == 'described_as' or k_ == 'also_bought' or \
                    k_ == 'also_viewed' or k_ == 'belongs_to': continue
                # print('k_, v_set = ', k_, v_set)
                # if k_ == 'mention' or k_ == 'described_as' or : continue
                for v_ in v_set:
                    tmp_list.append([k_,v_])
            for tail_and_relation in random.sample(tmp_list, min(len(tmp_list), n_neighbor)):
                memories_h.append([entity_type,entity])
                memories_r.append(tail_and_relation[0])
                memories_t.append([KG_RELATION[entity_type][tail_and_relation[0]],tail_and_relation[1]])

        # if the current ripple set of the given user is empty, we simply copy the ripple set of the last hop here
        # this won't happen for h = 0, because only the items that appear in the KG have been selected
        # this only happens on 154 users in Book-Crossing dataset (since both BX dataset and the KG are sparse)
        if len(memories_h) == 0:
            ret.append(ret[-1])
        else:
            # sample a fixed-size 1-hop memory for each user
            replace = len(memories_h) < n_memory
            indices = np.random.choice(len(memories_h), size=n_memory, replace=replace)
            memories_h = [memories_h[i] for i in indices]
            memories_r = [memories_r[i] for i in indices]
            memories_t = [memories_t[i] for i in indices]
            # entity_interaction_list += zip(memories_h, memories_r, memories_t)
            ret.append([memories_h, memories_r, memories_t])
    # print('user, ret = ', user, ret)
    # input() 
    return user, ret

class PATH_PTN(object):
    def __init__(self):
        self.patterns = []
        for pattern_id in [1, 11, 12, 13, 14, 15, 16, 17, 18]:
            pattern = PATH_PATTERN[pattern_id]
            pattern = [SELF_LOOP] + [v[0] for v in pattern[1:]]  # pattern contains all relations
            if pattern_id == 1:
                pattern.append(SELF_LOOP)
            self.patterns.append(tuple(pattern))

    def _has_pattern(self, path):
        pattern = tuple([v[0] for v in path])
        return pattern in self.patterns

    def _batch_has_pattern(self, batch_path):
        return [self._has_pattern(path) for path in batch_path]


class BatchKGEnvironment(object):
    def __init__(self, args, dataset_str, max_acts, max_path_len=3, state_history=1):
        super(BatchKGEnvironment, self).__init__()
        # print('SELF_LOOP = ', SELF_LOOP)
        # input()
        self.sub_batch_size = args.sub_batch_size
        self.max_acts = max_acts
        self.act_dim = max_acts + 1  # Add self-loop action, whose act_idx is always 0.
        self.max_num_nodes = max_path_len + 1  # max number of hops (= #nodes - 1)
        self.user_core = args.user_core
        self.training = args.training
        self.args = args

        self.kg = load_kg(dataset_str)
        self.PATH_PTN = PATH_PTN()

        self.p_hop = args.p_hop
        self.n_memory = args.n_memory
        # print('self.n_memory = ', self.n_memory)
        # input()
        self.embeds = load_embed(dataset_str)
        self.embed_size = self.embeds[USER].shape[1]
        self.embeds[SELF_LOOP] = (np.zeros(self.embed_size), 0.0)
        self._done = False

        self.load_selective_user()

        self.user_triplet_list = [user for user in list(self.kg(USER).keys())if user in self.select_user_list]
        print('self.user_triplet_list = ', len(self.user_triplet_list))        
        self.reset_path(0)

        self.rela_2_index = {}
        for k, v in self.embeds.items():
            if k not in self.rela_2_index:
                self.rela_2_index[k] = len(self.rela_2_index)

        if self.args.reward_hybrid == True:
            self.reward_u_p_score()

    def load_selective_user(self):
        train_labels = load_labels(self.args.dataset, 'train')
        self.select_user_list = list(train_labels.keys())

    def reward_u_p_score(self):
        u_p_scores = np.dot(self.embeds[USER] + self.embeds[PURCHASE][0], self.embeds[PRODUCT].T)
        self.u_p_scales = np.max(u_p_scores, axis=1)

    def reset_path(self, epoch):
        if self.training == True:
            self.user_triplet_set_path = '{}/triplet_set_{}_ep_{}.pickle'.format(self.args.save_model_dir, self.args.name, epoch)
            print('save user_triplet_set_path = ', self.user_triplet_set_path)
            self.user_triplet_set = get_user_triplet_set(self.kg, self.user_triplet_list, self.p_hop, self.n_memory)
            with open(self.user_triplet_set_path, 'wb') as fp:
                pickle.dump(self.user_triplet_set, fp)
        else:
            self.user_triplet_set_path = '{}/triplet_set_{}_ep_{}.pickle'.format(self.args.save_model_dir, self.args.name, self.args.eva_epochs)   
            print('load user_triplet_set_path = ', self.user_triplet_set_path)            
            with open(self.user_triplet_set_path, 'rb') as fp:
                self.user_triplet_set = pickle.load(fp)

    def reset(self, epoch, uids=None, training = False):
        if uids is None:
            all_uids = list(self.kg(USER).keys())
            uids = [random.choice(all_uids)]

        if training == True:
            self._batch_path = [[(SELF_LOOP, USER, uid)] for uid in uids for _ in range(self.sub_batch_size)]
        else:
            self._batch_path = [[(SELF_LOOP, USER, uid)] for uid in uids]
        self._done = False
        # self._batch_curr_state = self._batch_get_state(self._batch_path)
        self._batch_curr_actions = self._batch_get_actions(self._batch_path, self._done)
        self._batch_curr_reward = self._batch_get_reward(self._batch_path)
        # self.reset_path(epoch)

    def batch_step(self, batch_act_idx):
        """
        Args:
            batch_act_idx: list of integers.
        Returns:
            batch_next_state: numpy array of size [bs, state_dim].
            batch_reward: numpy array of size [bs].
            done: True/False
        """
        assert len(batch_act_idx) == len(self._batch_path)

        # Execute batch actions.
        for i in range(len(batch_act_idx)):
            act_idx = batch_act_idx[i]
            _, curr_node_type, curr_node_id = self._batch_path[i][-1]
            relation, next_node_id = self._batch_curr_actions[i][act_idx]
            if relation == SELF_LOOP:
                next_node_type = curr_node_type
            else:
                next_node_type = KG_RELATION[curr_node_type][relation]
            self._batch_path[i].append((relation, next_node_type, next_node_id))

        self._done = self._done or len(self._batch_path[0]) >= self.max_num_nodes  # must run before get actions, etc.
        self._batch_curr_actions = self._batch_get_actions(self._batch_path, self._done)
        self._batch_curr_reward = self._batch_get_reward(self._batch_path)

        return None, self._batch_curr_reward

    def _batch_get_actions(self, batch_path, done):
        def _get_actions(index, path, done):
            """Compute actions for current node."""
            _, curr_node_type, curr_node_id = path[-1]
            actions = [(SELF_LOOP, curr_node_id)]  # self-loop must be included.

            # (1) If game is finished, only return self-loop action.
            if done:
                return actions

            # (2) Get all possible edges from original knowledge graph.
            # [CAVEAT] Must remove visited nodes!
            relations_nodes = self.kg(curr_node_type, curr_node_id)
            candidate_acts = []  # list of tuples of (relation, node_type, node_id)
            candidate_node_types = []
            visited_nodes = set([(v[1], v[2]) for v in path])
            for r in relations_nodes:
                # print('KG_RELATION = ', KG_RELATION)
                # input()
                next_node_type = KG_RELATION[curr_node_type][r]
                next_node_ids = relations_nodes[r]
                next_node_ids = [n for n in next_node_ids if (next_node_type, n) not in visited_nodes]  # filter
                candidate_node_types.extend([next_node_type] * len(next_node_ids))
                candidate_acts.extend(zip([r] * len(next_node_ids), next_node_ids))

            # (3) If candidate action set is empty, only return self-loop action.
            if len(candidate_acts) == 0:
                return actions

            # (4) If number of available actions is smaller than max_acts, return action sets.
            if len(candidate_acts) <= self.max_acts:
                candidate_acts = sorted(candidate_acts, key=lambda x: (x[0], x[1]))
                actions.extend(candidate_acts)
                return actions

            # (5) If there are too many actions, do some deterministic trimming here!
            user_embed = self.embeds[USER][path[0][-1]]

            # user_embed = self.user_emb_memory[index,:].detach().numpy()

            scores = []
            for r, next_node_id in candidate_acts:
                next_node_type = KG_RELATION[curr_node_type][r]
                if next_node_type == USER:
                    src_embed = user_embed
                elif next_node_type == PRODUCT:
                    src_embed = user_embed + self.embeds[PURCHASE][0]
                elif next_node_type == WORD:
                    src_embed = user_embed + self.embeds[MENTION][0]
                else:  # BRAND, CATEGORY, RELATED_PRODUCT
                    src_embed = user_embed + self.embeds[PURCHASE][0] + self.embeds[r][0]
                score = np.matmul(src_embed, self.embeds[next_node_type][next_node_id])
                # This trimming may filter out target products!
                # Manually set the score of target products a very large number.
                 #    score = 99999.0
                scores.append(score)
            candidate_idxs = np.argsort(scores)[-self.max_acts:]  # choose actions with larger scores
            candidate_acts = sorted([candidate_acts[i] for i in candidate_idxs], key=lambda x: (x[0], x[1]))
            random.shuffle(candidate_acts)
            actions.extend(candidate_acts)
            return actions

        return [_get_actions(index, path, done) for index, path in enumerate(batch_path)]

    def _batch_get_reward(self, batch_path):
        def _get_reward(path):
            # If it is initial state or 1-hop search, reward is 0.
            if len(path) <= 2:
                return 0.0
            if not self.PATH_PTN._has_pattern(path):
                return 0.0
            target_score = 0.0
            _, curr_node_type, curr_node_id = path[-1]
            if curr_node_type == PRODUCT:
                # Give soft reward for other reached products.
                uid = path[0][-1]
                if curr_node_id in self.kg(USER, uid)[PURCHASE]: target_score = 1.0
                else: target_score = 0.0


                # print('path = ', path)
                # input()

                # u_vec = self.embeds[USER][uid] + self.embeds[PURCHASE][0]
                # p_vec = self.embeds[PRODUCT][curr_node_id]
                # score = np.dot(u_vec, p_vec) / self.u_p_scales[uid]
                # target_score = max(score, 0.0)
            return target_score

        def _get_reward_ori(path):
            # If it is initial state or 1-hop search, reward is 0.
            if len(path) <= 2:
                return 0.0
            if not self.PATH_PTN._has_pattern(path):
                return 0.0
            target_score = 0.0
            _, curr_node_type, curr_node_id = path[-1]
            if curr_node_type == PRODUCT:
                # Give soft reward for other reached products.

                uid = path[0][-1]
                u_vec = self.embeds[USER][uid] + self.embeds[PURCHASE][0]
                p_vec = self.embeds[PRODUCT][curr_node_id]
                score = np.dot(u_vec, p_vec) / self.u_p_scales[uid]

                # print('reward_hybrid score = ', score)
                if curr_node_id in self.kg(USER, uid)[PURCHASE]: score += 0.8
                else: score += 0.0

                # print('after reward_hybrid score = ', score)

                target_score = max(score, 0.0)

            return target_score

        if self.args.reward_hybrid == True:
            # print('reward_hybrid')
            batch_reward = [_get_reward_ori(path) for path in batch_path]
        else:
            batch_reward = [_get_reward(path) for path in batch_path]

        return np.array(batch_reward)


    def batch_action_mask(self, dropout=0.0):
        """Return action masks of size [bs, act_dim]."""
        batch_mask = []
        for actions in self._batch_curr_actions:
            act_idxs = list(range(len(actions)))
            if dropout > 0 and len(act_idxs) >= 5:
                keep_size = int(len(act_idxs[1:]) * (1.0 - dropout))
                # keep_size = int(len(act_idxs[1:]) * (1.0))
                tmp = np.random.choice(act_idxs[1:], keep_size, replace=False).tolist()
                act_idxs = [act_idxs[0]] + tmp
            # act_mask = np.zeros(self.act_dim, dtype=np.uint8)
            act_mask = np.zeros(self.act_dim)
            act_mask[act_idxs] = 1
            batch_mask.append(act_mask)

        return np.vstack(batch_mask)

    def output_valid_user(self):
        return [user  for user in list(self.kg(USER).keys()) if user in self.select_user_list] 


# class BatchKGEnvironment_metapath(BatchKGEnvironment):
#     def __init__(self, args, dataset_str, max_acts, max_path_len=3, state_history=1):
#         super().__init__(args, dataset_str, max_acts, max_path_len, state_history)


#         self.meta_path = load_metapath_core(args.dataset)


#     def reset_path(self, epoch):
#         if self.training == True:
#             self.user_triplet_set_path = '{}/triplet_set_{}_ep_{}.pickle'.format(self.args.save_model_dir, self.args.name, epoch)
            

#             self.user_triplet_set = get_user_triplet_set(self.kg, self.et_idx2ty, self.user_triplet_list, self.p_hop, self.n_memory)
#             with open(self.user_triplet_set_path, 'wb') as fp:
#                 pickle.dump(self.user_triplet_set, fp)
#         else:
#             self.user_triplet_set_path = '{}/triplet_set_{}_ep_{}.pickle'.format(self.args.save_model_dir, self.args.name, self.args.eva_epochs)
#             print('load user_triplet_set_path = ', self.user_triplet_set_path)
#             # self.user_triplet_set_path = '{}/triplet_set_{}.pickle'.format(self.args.save_model_dir, self.args.name)   
#             with open(self.user_triplet_set_path, 'rb') as fp:
#                 self.user_triplet_set = pickle.load(fp)



def _reset_meta_path(user, _, mp_simple_size=2):
    # print('user_triplet_list = ', len(user_triplet_list))
    # for user in user_triplet_list:
    #     print('user = ', user)
    # print('user = ', user)
    tmp_user_triplet_set = []
    for key, metapaths in META_path[user].items():
        replace = len(metapaths) < mp_simple_size
        indices = np.random.choice(len(metapaths), size=mp_simple_size, replace=replace)
        metapaths_tmp = [metapaths[i] for i in indices]
        tmp_user_triplet_set += metapaths_tmp
    replace = len(tmp_user_triplet_set) < 32
    indices = np.random.choice(len(tmp_user_triplet_set), size=32, replace=replace)
    tmp_user_triplet_set = [tmp_user_triplet_set[i] for i in indices]
        # print("self.user_triplet_set[user] = ", len(self.user_triplet_set[user]))
        # input()
    # print('user = ', user, len(tmp_user_triplet_set))
    return user, tmp_user_triplet_set
 
class BatchKGEnvironment_metapath(BatchKGEnvironment):
    def __init__(self, args, dataset_str, max_acts, max_path_len=3, state_history=1):
        super().__init__(args, dataset_str, max_acts, max_path_len, state_history)
        self.reset_path = self.reset_meta_path
        # print('reset reset_path')
        self.meta_path = load_metapath_core(args.dataset)
        self.key_path =  self.meta_path[0].keys()
        # print('self.key_path = ', self.key_path)
        # input()
        self.mp_simple_size = 15
        self.reset_path(0)

    # def reset_meta_path(self, epoch):
    #     # print('reset_path stop')
    #     # input()

    #     if self.training == True:
    #         self.user_triplet_set_path = '{}/triplet_set_{}_ep_{}.pickle'.format(self.args.save_model_dir, self.args.name, epoch)
    #         self.user_triplet_set = {}

    #         for user in self.user_triplet_list:
    #             self.user_triplet_set[user] = []
    #             for key, metapaths in self.meta_path[user].items():
    #                 replace = len(metapaths) < self.mp_simple_size
    #                 indices = np.random.choice(len(metapaths), size=self.mp_simple_size, replace=replace)
    #                 metapaths_tmp = [metapaths[i] for i in indices]
    #                 self.user_triplet_set[user] += metapaths_tmp

    #             replace = len(self.user_triplet_set[user]) < 32
    #             indices = np.random.choice(len(self.user_triplet_set[user]), size=32, replace=replace)
    #             self.user_triplet_set[user] = [self.user_triplet_set[user][i] for i in indices]
    #             # print('self.user_triplet_set[user] = ', len(self.user_triplet_set[user]))
    #             # input()
    #         with open(self.user_triplet_set_path, 'wb') as fp:
    #             pickle.dump(self.user_triplet_set, fp)
    #     else:
    #         self.user_triplet_set_path = '{}/triplet_set_{}_ep_{}.pickle'.format(self.args.save_model_dir, self.args.name, self.args.eva_epochs)
    #         print('load user_triplet_set_path = ', self.user_triplet_set_path)
    #         with open(self.user_triplet_set_path, 'rb') as fp:
    #             self.user_triplet_set = pickle.load(fp)

    def reset_meta_path(self, epoch):
        # print("reset_path stop")
        # input()
        if self.training == True:
            self.user_triplet_set_path = '{}/triplet_set_{}_ep_{}.pickle'.format(self.args.save_model_dir, self.args.name, epoch)
            self.user_triplet_set = {}
            g_user_triplet_list = {u:1 for u in self.user_triplet_list}

            global META_path 
            META_path = self.meta_path
            print('reset_meta_path start')
            print('self.user_triplet_list = ', len(self.user_triplet_list))
            with mp.Pool(processes=min(mp.cpu_count(), 12)) as pool:
                job = partial(_reset_meta_path, mp_simple_size = self.mp_simple_size)
                for u, u_r_set in pool.starmap(job, g_user_triplet_list.items()):
                    # print('u = ', u, 'u_r_set = ', u_r_set)
                    self.user_triplet_set[u] = u_r_set
            # input()
            print('save')
            with open(self.user_triplet_set_path, 'wb') as fp:
                pickle.dump(self.user_triplet_set, fp)
        else:
            self.user_triplet_set_path = "{}/triplet_set_{}_ep_{}.pickle".format(self.args.save_model_dir, self.args.name, self.args.eva_epochs)
            print("load user_triplet_set_path = ", self.user_triplet_set_path)
            with open(self.user_triplet_set_path, "rb") as fp:
                self.user_triplet_set = pickle.load(fp)