from __future__ import absolute_import, division, print_function

import os
import sys
sys.path.insert(0,'../preprocess')

from tqdm import tqdm
import pickle
import random
import torch
from datetime import datetime

from utils import *
# from preprocess.knowledge_graph import RW_based_KG, KG_based_KG
# from preprocess import knowledge_graph
# from preprocess import dataset

# input()

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


# import collections
# from collections import defaultdict
# from collections import Counter 

from env.sp_user_tri_set import kg_based_get_user_triplet_set, rw_get_user_triplet_set

class PATH_PTN(object):
    def __init__(self):
        self.patterns = []
        for pattern_id in [1, 11, 12, 13, 14, 15, 16, 17, 18]:
            pattern = PATH_PATTERN[pattern_id]
            pattern = [SELF_LOOP] + [v[0] for v in pattern[1:]]  # pattern contains all relations
            if pattern_id == 1:
                pattern.append(SELF_LOOP)
            self.patterns.append(tuple(pattern))

    def _rw_has_pattern(self, path):
        pattern = tuple([v[0] for v in path])
        return pattern in self.patterns

    def _kg_has_pattern(self, path):
        pattern = tuple([v[0] for v in path])
        if pattern[0] == SELF_LOOP and pattern[1] == PURCHASE  \
            and pattern[2] != SELF_LOOP and pattern[3] != SELF_LOOP:
            return True
        else:
            return False

class BatchKGEnvironment(object):
    def __init__(self, args, dataset_str, max_acts, max_path_len=3, state_history=1):
        super(BatchKGEnvironment, self).__init__()
        self.args = args
        self.max_acts = max_acts
        self.act_dim = max_acts + 1  # Add self-loop action, whose act_idx is always 0.
        self.max_num_nodes = max_path_len + 1  # max number of hops (= #nodes - 1)
        self.p_hop = args.p_hop
        self.n_memory = args.n_memory
        self._done = False

        self.load_kg_dataset_emb(args, dataset_str)

        if self.args.envir == 'p1':
            self._get_reward = self._rw_get_reward
            self.next_type = self.rw_based_next_type
            self.get_user_triplet = rw_get_user_triplet_set

        elif self.args.envir == 'p2':
            self.et_idx2ty = self.dataset.et_idx2ty
            self._get_reward = self._kg_get_reward
            self.next_type = self.kg_based_next_type
            self.get_user_triplet = kg_based_get_user_triplet_set

                
        self.set_UC_view()  
        self.PATH_PTN = PATH_PTN()

    def load_kg_dataset_emb(self, args, dataset_str):

        self.kg = load_kg(dataset_str)
        self.dataset = load_dataset(dataset_str)
        self.args.core_user_list = self.dataset.core_user_list

        self.args.kg_fre_dict = self.kg.kg_fre_dict  
        print('len(self.core_user_list) = ', len(self.args.core_user_list))
        self.user_list = [user for user in list(self.kg(USER).keys())if user in self.args.core_user_list]

        print('self.args.core_user_list = ', len(self.args.core_user_list))
        print('self.user_triplet_list = ', len(self.user_list))
        # input()

        self.embeds = load_embed(dataset_str)
        self.embed_size = self.embeds[USER].shape[1]
        self.embeds[SELF_LOOP] = (np.zeros(self.embed_size), 0.0)



        self.rela_2_index = {}
        for k, v in self.embeds.items():
            if k not in self.rela_2_index:
                self.rela_2_index[k] = len(self.rela_2_index)


    def set_UC_view(self):
        if self.args.non_sampling == True:
            self.user_triplet_set = [user for user in list(self.kg(USER).keys())if user in self.args.core_user_list]
            return

        user_triplet_path = '{}/triplet_set_{}.pickle'.format(self.args.save_model_dir, self.args.name)
        if os.path.exists(user_triplet_path):
            with open(user_triplet_path, 'rb') as fp:
                self.user_triplet_set = pickle.load(fp)
            print('load user_triplet_path = ', user_triplet_path)
        else:
            self.user_triplet_set = self.get_user_triplet(self.args, self.kg, self.user_list, self.p_hop, self.n_memory)
            with open(user_triplet_path, 'wb') as fp:
                pickle.dump(self.user_triplet_set, fp)
            print('user_triplet_set_path save = ', user_triplet_path)


    def reset(self, epoch, uids=None, training = False):
        if uids is None:
            # print('ddfsdfdsfds')
            all_uids = list(self.kg(USER).keys())
            uids = [random.choice(all_uids)]

        self._batch_path = [[(SELF_LOOP, USER, uid)] for uid in uids]

        self._done = False
        self._batch_curr_actions = self._batch_get_actions(self._batch_path, self._done)
        self._batch_curr_reward = self._batch_get_reward(self._batch_path)

    def kg_based_next_type(self, curr_node_type, relation, next_node_id):
        return self.et_idx2ty[next_node_id]

    def rw_based_next_type(self, curr_node_type, relation, next_node_id):
        return KG_RELATION[curr_node_type][relation]

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
                next_node_type = self.next_type(curr_node_type, relation, next_node_id)
            self._batch_path[i].append((relation, next_node_type, next_node_id))

        self._done = self._done or len(self._batch_path[0]) >= self.max_num_nodes  # must run before get actions, etc.
        self._batch_curr_actions = self._batch_get_actions(self._batch_path, self._done)
        self._batch_curr_reward = self._batch_get_reward(self._batch_path)

        return None, self._batch_curr_reward

    def _batch_get_actions(self, batch_path, done):
        return [self._get_actions(path, done) for path in batch_path]

    def _get_actions(self, path, done):
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
        visited_nodes = set([(v[1], v[2]) for v in path])
        for r in relations_nodes:
            next_node_ids = relations_nodes[r]
            next_node_set = []
            for n_id in next_node_ids:
                next_node_set.append([self.next_type(curr_node_type, r, n_id),n_id])
            next_node_ids = [n_set[1] for n_set in next_node_set if (n_set[0], n_set[1]) not in visited_nodes]
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
        scores = []
        for r, next_node_id in candidate_acts:
            next_node_type = self.next_type(curr_node_type, r, next_node_id)
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
            # if next_node_type == PRODUCT and next_node_id in self._target_pids:
            #    score = 99999.0
            scores.append(score)
        candidate_idxs = np.argsort(scores)[-self.max_acts:]  # choose actions with larger scores
        candidate_acts = sorted([candidate_acts[i] for i in candidate_idxs], key=lambda x: (x[0], x[1]))
        actions.extend(candidate_acts)
        return actions

    def _kg_get_reward(self, path):
        # If it is initial state or 1-hop search, reward is 0.
        if len(path) <= 3:
            return 0.0
        if not self.PATH_PTN._kg_has_pattern(path):
            return 0.0
        target_score = 0.0
        _, curr_node_type, curr_node_id = path[-1]
        if curr_node_type == PRODUCT:
            # Give soft reward for other reached products.

            score = 0
            uid = path[0][-1]

            if curr_node_id in self.kg(USER, uid)[PURCHASE]: score += 1
            else: score += 0.0

            target_score = max(score, 0.0)

        return target_score

    def _rw_get_reward(self, path):
        # If it is initial state or 1-hop search, reward is 0.
        if len(path) <= 2:
            return 0.0
        if not self.PATH_PTN._rw_has_pattern(path):
            return 0.0
        target_score = 0.0
        _, curr_node_type, curr_node_id = path[-1]
        if curr_node_type == PRODUCT:
            # Give soft reward for other reached products.
            uid = path[0][-1]
            score = 0
            if curr_node_id in self.kg(USER, uid)[PURCHASE]: score += 1
            else: score += 0.0
            target_score = max(score, 0.0)

        return target_score

    def _batch_get_reward(self, batch_path):

        batch_reward = [self._get_reward(path) for path in batch_path]

        return np.array(batch_reward)

    def batch_action_mask(self, dropout=0.0):
        """Return action masks of size [bs, act_dim]."""
        batch_mask = []
        for actions in self._batch_curr_actions:
            act_idxs = list(range(len(actions)))
            if dropout > 0 and len(act_idxs) >= 5:
                keep_size = int(len(act_idxs[1:]) * (1.0 - dropout))
                tmp = np.random.choice(act_idxs[1:], keep_size, replace=False).tolist()
                act_idxs = [act_idxs[0]] + tmp
            # act_mask = np.zeros(self.act_dim, dtype=np.uint8)
            act_mask = np.zeros(self.act_dim)
            act_mask[act_idxs] = 1
            batch_mask.append(act_mask)
        return np.vstack(batch_mask)

    def output_valid_user(self):
        return [user  for user in list(self.kg(USER).keys()) if user in self.user_list] 
