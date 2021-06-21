from __future__ import absolute_import, division, print_function

import os
import sys
from tqdm import tqdm
import pickle
import random
import torch
from datetime import datetime

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

from utils import *

def kg_based_get_user_triplet_set(args, kg, et_idx2ty, user_list, p_hop, n_memory):
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
    with mp.Pool(processes=min(mp.cpu_count(), 5)) as pool:
        job = partial(_kg_based_get_user_triplet_set, p_hop=max(1,args_tmp['p_hop']), KG_RELATION = KG_RELATION, n_memory=args_tmp['n_memory'], n_neighbor=16)
        for u, u_r_set in pool.starmap(job, user_history_dict.items()):
            user_triplet_set[u] = u_r_set

    del g_kg, g_et_idx2ty, g_args
    return user_triplet_set

def _kg_based_get_user_triplet_set(user, history, p_hop=2, KG_RELATION = None, n_memory=32, n_neighbor=16):
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
            tmp_list = []
            for k_, v_set in g_kg(entity_type,entity).items():
                if k_ == SELF_LOOP: continue
                for v_ in v_set:
                    if g_et_idx2ty[v_] == USER:
                        if v_ in g_args.sp_user_filter: tmp_list.append([k_,v_])
                    else:
                        if g_args.kg_fre_dict[v_] >= g_args.kg_fre_lower:
                            tmp_list.append([k_,v_])
                if h != 0 and len(tmp_list) >= 30: break
                

            if len(tmp_list) == 0:
                for k_, v_set in g_kg(entity_type,entity).items():
                    for v_ in v_set:
                        tmp_list.append([k_,v_])
                    if h != 0 and len(tmp_list) >= 30: break

            for tail_and_relation in random.sample(tmp_list, min(len(tmp_list), n_neighbor)):
                memories_h.append([entity_type,entity])
                memories_r.append(tail_and_relation[0])
                memories_t.append([g_et_idx2ty[tail_and_relation[1]],tail_and_relation[1]])

        if len(memories_h) == 0:
            ret.append(ret[-1])
        else:
            replace = len(memories_h) < n_memory
            indices = np.random.choice(len(memories_h), size=n_memory, replace=replace)
            memories_h = [memories_h[i] for i in indices]
            memories_r = [memories_r[i] for i in indices]
            memories_t = [memories_t[i] for i in indices]
            ret.append([memories_h, memories_r, memories_t])
    return user, ret

def rw_get_user_triplet_set(args, kg, user_list, p_hop, n_memory):
    args_tmp = {'p_hop': p_hop, 'n_memory': n_memory}

    user_triplet_set = collections.defaultdict(list)
    user_history_dict = {}

    for user in user_list:
        if user not in user_history_dict:
            user_history_dict[user] = [[USER, user]]

    global g_kg, g_args
    g_kg = kg
    g_args = args
    with mp.Pool(processes=min(mp.cpu_count(), 4)) as pool:
        job = partial(_rw_get_user_triplet_set, p_hop=max(1,args_tmp['p_hop']), KG_RELATION = KG_RELATION, n_memory=args_tmp['n_memory'], n_neighbor=16)
        for u, u_r_set in pool.starmap(job, user_history_dict.items()):
            # print(' u, u_r_set  = ',  u, u_r_set)
            user_triplet_set[u] = u_r_set


    del g_kg
    return user_triplet_set

def _rw_get_user_triplet_set(user, history, p_hop=2, KG_RELATION = None, n_memory=32, n_neighbor=16):

    def add_list_condition(next_entities, h, entity_type, entity, kg_fre_upper, kg_fre_lower, remove_type, skip_self_loop = True):
        for k_, v_set in g_kg(entity_type,entity).items():
            if k_ == SELF_LOOP and skip_self_loop == True: continue
            next_node_type = KG_RELATION[entity_type][k_]
            if next_node_type in remove_type: print('next_node_type remove == ', remove_type, end='\r'); continue

            if WORD in remove_type:
                if k_ == 'mentions' or k_ == 'described_as' or  k_ == 'also_viewed': 
                    print('next_node_type remove == ', k_, end='\r')
                    continue

            if next_node_type == USER:
                for v_ in v_set:
                    if v_ in g_args.sp_user_filter:  next_entities.append([k_,v_])
            else:
                for v_ in v_set:
                    if g_args.kg_fre_dict[next_node_type][v_] < kg_fre_upper and \
                        g_args.kg_fre_dict[next_node_type][v_] >= kg_fre_lower:
                        next_entities.append([k_,v_])
            if h != 0 and len(next_entities) >= 20: break

        return next_entities

    if g_args.tri_wd_rm == True: 
        remove_type = [WORD]
    elif g_args.tri_pro_rm == True: 
        remove_type = [PRODUCT]
    else: 
        remove_type = []

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

            next_entities = []

            next_entities = add_list_condition(next_entities, h, entity_type, entity,  g_args.kg_fre_upper, g_args.kg_fre_lower, remove_type, skip_self_loop = True)
            if len(next_entities) == 0:
                next_entities = add_list_condition(next_entities, h, entity_type, entity,   g_args.kg_fre_upper, g_args.kg_fre_lower, [], skip_self_loop = False)
            # print('len(next_entities)  = ', len(next_entities))
            for tail_and_relation in random.sample(next_entities, min(len(next_entities), n_neighbor)):
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
            ret.append([memories_h, memories_r, memories_t])

    return user, ret
