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

import numpy as np
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

def get_user_triplet_set(args, kg, et_idx2ty, user_list, p_hop, n_memory):
    args_tmp = {'p_hop': p_hop, 'n_memory': n_memory}

    # user -> [(hop_0_heads, hop_0_relations, hop_0_tails), (hop_1_heads, hop_1_relations, hop_1_tails), ...]
    user_triplet_set = collections.defaultdict(list)
    # entity_interaction_dict = collections.defaultdict(list)
    user_history_dict = {}

    for user in user_list:
        if user not in user_history_dict:
            user_history_dict[user] = [[USER, user]]

    global g_kg, g_et_idx2ty, g_args
    g_args = args
    g_kg = kg
    g_et_idx2ty = et_idx2ty
    with mp.Pool(processes=min(mp.cpu_count(), 12)) as pool:
        job = partial(_get_user_triplet_set, p_hop=max(1,args_tmp['p_hop']), KG_RELATION = KG_RELATION, n_memory=args_tmp['n_memory'], n_neighbor=16)
        for u, u_r_set in pool.starmap(job, user_history_dict.items()):
            user_triplet_set[u] = u_r_set

    del g_kg, g_et_idx2ty, g_args
    return user_triplet_set

def _get_user_triplet_set(user, history, p_hop=2, KG_RELATION = None, n_memory=32, n_neighbor=16):
    ret = []
    entity_interaction_list = []

    for h in range(max(1,p_hop)):
        memories_h = []
        memories_r = []
        memories_t = []

        if h == 0:
            tails_of_last_hop = history
        else:
            tails_of_last_hop = ret[-1][2]

        for entity_type, entity in tails_of_last_hop:
            # print('entity_type, entity = ', entity_type, entity)
            tmp_list = []
            for k_, v_set in g_kg(entity_type,entity).items():
                # if h != 0 and k_ == PURCHASE: continue
                if h != 0 and k_ == PURCHASE:
                    for v_ in v_set:
                        if v_ in g_args.kg_user_filter:
                            tmp_list.append([k_,v_])    
                else:
                    for v_ in v_set:
                        tmp_list.append([k_,v_])
            for tail_and_relation in random.sample(tmp_list, min(len(tmp_list), n_neighbor)):
                memories_h.append([entity_type,entity])
                memories_r.append(tail_and_relation[0])
                memories_t.append([g_et_idx2ty[tail_and_relation[1]],tail_and_relation[1]])

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
    return user, ret

class BatchKGEnvironment_graph(object):
    def __init__(self, args, dataset_str, max_acts, max_path_len=3, state_history=1):
        super(BatchKGEnvironment_graph, self).__init__()
        # print('SELF_LOOP = ', SELF_LOOP)
        # input()
        self.sub_batch_size = args.sub_batch_size
        self.max_acts = max_acts
        self.act_dim = max_acts + 1  # Add self-loop action, whose act_idx is always 0.
        self.max_num_nodes = max_path_len + 1  # max number of hops (= #nodes - 1)
        self.user_core = args.user_core
        self.training = args.training
        self.args = args

        dataset = load_dataset_core(args, args.dataset)
        self.et_idx2ty = dataset.et_idx2ty
        self.entity_list = dataset.entity_list
        self.rela_list = dataset.rela_list
        self.select_user_list = dataset.select_user_list
        self.args.kg_user_filter = dataset.kg_user_filter

        # print('self.args.kg_user_filter = ', self.args.kg_user_filter)

        self.kg = load_kg_core(args, dataset_str)
        self.p_hop = args.p_hop
        self.n_memory = args.n_memory
        self.embeds = load_embed(dataset_str)
        self.embed_size = self.embeds[USER].shape[1]
        self.embeds[SELF_LOOP] = (np.zeros(self.embed_size), 0.0)
        self._done = False

        self.user_triplet_list = [user for user in list(self.kg(USER).keys())if user in self.select_user_list]
        print('self.user_triplet_list = ', len(self.user_triplet_list))        
        self.reset_path(0)        

        self.rela_2_index = {}
        for k, v in self.embeds.items():
            if k not in self.rela_2_index:
                self.rela_2_index[k] = len(self.rela_2_index)

        if self.args.reward_hybrid == True:
            self.reward_u_p_score()

        self.record_path_info = {}

    def reward_u_p_score(self):
        u_p_scores = np.dot(self.embeds[USER] + self.embeds[PURCHASE][0], self.embeds[PRODUCT].T)
        self.u_p_scales = np.max(u_p_scores, axis=1)

    def reset_path(self, epoch):
        if self.training == True:
            self.user_triplet_set_path = '{}/triplet_set_{}_ep_{}.pickle'.format(self.args.save_model_dir, self.args.name, epoch)
            self.user_triplet_set = get_user_triplet_set(self.args, self.kg, self.et_idx2ty, self.user_triplet_list, self.p_hop, self.n_memory)
            with open(self.user_triplet_set_path, 'wb') as fp:
                pickle.dump(self.user_triplet_set, fp)
        else:
            self.user_triplet_set_path = '{}/triplet_set_{}_ep_{}.pickle'.format(self.args.save_model_dir, self.args.name, self.args.eva_epochs)
            print('load user_triplet_set_path = ', self.user_triplet_set_path)
            # self.user_triplet_set_path = '{}/triplet_set_{}.pickle'.format(self.args.save_model_dir, self.args.name)   
            with open(self.user_triplet_set_path, 'rb') as fp:
                self.user_triplet_set = pickle.load(fp)

    def reset(self, epoch, uids=None, training = False):
        if uids is None:
            # print('ddfsdfdsfds')
            all_uids = list(self.kg(USER).keys())
            uids = [random.choice(all_uids)]
        # print('uids = ',uids)
        # input()
        if training == True:
            self._batch_path = [[(SELF_LOOP, USER, uid)] for uid in uids for _ in range(self.sub_batch_size)]
        else:
            self._batch_path = [[(SELF_LOOP, USER, uid)] for uid in uids]
        
        # print('self._batch_path = ,', self._batch_path, uids)
        # input()
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
                next_node_type = self.et_idx2ty[next_node_id]
            self._batch_path[i].append((relation, next_node_type, next_node_id))

        self._done = self._done or len(self._batch_path[0]) >= self.max_num_nodes  # must run before get actions, etc.
        self._batch_curr_actions = self._batch_get_actions(self._batch_path, self._done)
        self._batch_curr_reward = self._batch_get_reward(self._batch_path)

        return None, self._batch_curr_reward

    def _batch_has_pattern(self, batch_path):
        def _has_pattern(path):
            pattern = tuple([v[0] for v in path])
            if pattern[0] == SELF_LOOP and pattern[1] == PURCHASE  \
                and pattern[2] != SELF_LOOP and pattern[3] != SELF_LOOP:
                return True
            else:
                return False
        return [_has_pattern(path) for path in batch_path]


    def _batch_get_actions(self, batch_path, done):
        def _get_actions(index, path, done):
            def queru_next_possible_action(curr_node_type, curr_node_id):
                relations_nodes = self.kg(curr_node_type, curr_node_id)
                candidate_acts = []  # list of tuples of (relation, node_type, node_id)
                candidate_node_types = []

                for r in relations_nodes:
                    # next_node_type = KG_RELATION[curr_node_type][r]
                    next_node_ids = relations_nodes[r]
                    next_node_set = []
                    for n_id in next_node_ids:
                        if self.et_idx2ty[n_id] == USER:
                            if n_id in self.args.kg_user_filter: next_node_set.append([self.et_idx2ty[n_id],n_id])    
                        else: next_node_set.append([self.et_idx2ty[n_id],n_id])
                    next_node_ids = [n_set[1] for n_set in next_node_set if (n_set[0], n_set[1])]  # filter
                    candidate_acts.extend(zip([r] * len(next_node_ids), next_node_ids))

                # (3) If candidate action set is empty, only return self-loop action.
                if len(candidate_acts) == 0:
                    return []

                # (4) If number of available actions is smaller than max_acts, return action sets.
                if len(candidate_acts) <= self.max_acts:
                    return candidate_acts

                # user_embed = self.embeds[USER][path[0][-1]]

                user_embed = self.embeds[USER][self.args.kg_user_filter[0]]

                scores = []
                for r, next_node_id in candidate_acts:
                    next_node_type = self.et_idx2ty[next_node_id]
                    if next_node_type == USER:
                        src_embed = user_embed
                    elif next_node_type == PRODUCT:
                        src_embed = user_embed + self.embeds[PURCHASE][0]
                    elif next_node_type == WORD:
                        src_embed = user_embed + self.embeds[MENTION][0]
                    else:  # BRAND, CATEGORY, RELATED_PRODUCT
                        src_embed = user_embed + self.embeds[PURCHASE][0] + self.embeds[r][0]
                    score = np.matmul(src_embed, self.embeds[next_node_type][next_node_id])
                    scores.append(score)

                candidate_idxs = np.argsort(scores)[-self.max_acts:]  # choose actions with larger scores
                candidate_acts = [candidate_acts[i] for i in candidate_idxs]
                return candidate_acts

            _, curr_node_type, curr_node_id = path[-1]

            actions = [(SELF_LOOP, curr_node_id)]
            if done:
                return actions
            
            cur_st_key = 'type_'+ str(curr_node_type) + 'id_' + str(curr_node_id)
            candidate_acts = queru_next_possible_action(curr_node_type,curr_node_id)
            # print('cur_st_key = ', cur_st_key, '\r')
            # if cur_st_key not in self.record_path_info:
            #     candidate_acts = queru_next_possible_action(curr_node_type,curr_node_id)
            #     # if cur_st_key == 'type_userid_1327': print('candidate_acts = ', candidate_acts)
            #     self.record_path_info[cur_st_key] = candidate_acts
            #     self.args.logger.info('cur_st_key not in = ' + str(cur_st_key))
            # else:
            #     candidate_acts = self.record_path_info[cur_st_key]

            visited_nodes = set([(v[1], v[2]) for v in path])            
            candidate_acts = [(r, n_id) for r, n_id in candidate_acts if (self.et_idx2ty[n_id], n_id) not in visited_nodes]
            
            random.shuffle(candidate_acts)
            actions.extend(candidate_acts)
            return actions

        return [_get_actions(index, path, done) for index, path in enumerate(batch_path)]


    def _batch_get_reward(self, batch_path):
        def _has_pattern(path):
            pattern = tuple([v[0] for v in path])
            if pattern[0] == SELF_LOOP and pattern[1] == PURCHASE  \
                and pattern[2] != SELF_LOOP and pattern[3] != SELF_LOOP:
                return True
            else:
                return False

        def _get_reward(path):
            # If it is initial state or 1-hop search, reward is 0.
            if len(path) <= 3:
                # print('len path == ', len(path))
                return 0.0

            if not _has_pattern(path):
                # print(' not _has pattern')
                # print('pattern = ', tuple([v[0] for v in path]))
                return 0.0

            target_score = 0.0
            _, curr_node_type, curr_node_id = path[-1]
            if curr_node_type == PRODUCT:
                # Give soft reward for other reached products.
                uid = path[0][-1]
                if curr_node_id in self.kg(USER, uid)[PURCHASE]: target_score = 1.0
                else: target_score = 0.0

            # if len(path) == 4:
                # print('curr_node_id , curr_node_type, target_score = ', curr_node_id, curr_node_type, target_score)
                # print('USER, self.kg(USER, uid)[PURCHASE] = ', path[0][-1], self.kg(USER, path[0][-1])[PURCHASE])

            return target_score

        def _get_reward_ori(path):
            # If it is initial state or 1-hop search, reward is 0.
            if len(path) <= 3:
                return 0.0

            if not _has_pattern(path):
                return 0.0

            target_score = 0.0
            _, curr_node_type, curr_node_id = path[-1]
            if curr_node_type == PRODUCT:
                # Give soft reward for other reached products.
                uid = path[0][-1]

                u_vec = self.embeds[USER][uid] + self.embeds[PURCHASE][0]
                p_vec = self.embeds[PRODUCT][curr_node_id]
                score = np.dot(u_vec, p_vec) / self.u_p_scales[uid]

                # print('reward_hybrid = ', score)

                if curr_node_id in self.kg(USER, uid)[PURCHASE]: score += 0.8
                else: score += 0.0

                # print('after reward_hybrid = ', score)

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
 
class BatchKGEnvironment_graph_metapath(BatchKGEnvironment_graph):
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
