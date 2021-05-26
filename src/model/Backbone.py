from __future__ import absolute_import, division, print_function

import sys
import os
import argparse
from collections import namedtuple
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Categorical

from easydict import EasyDict as edict
from model.lstm_base.model_lstm_mf_emb import AC_lstm_mf_dummy

from utils import *

SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

class Backbone_1(AC_lstm_mf_dummy):
    def __init__(self, args, user_triplet_set, rela_2_index, act_dim, gamma=0.99, hidden_sizes=[512, 256]):
        super().__init__(args, user_triplet_set, rela_2_index, act_dim, gamma, hidden_sizes)

    def bulid_model_reasoning(self):

        self.reasoning_step = self.args.reasoning_step
        self.rn_state_tr_query = []
        self.update_rn_state = []
        self.rn_query_st_tr = []
        self.rh_query = []
        self.o_r_query = []
        self.v_query = []
        self.t_u_query = []
        for i in range(self.reasoning_step):
            self.rn_state_tr_query.append(nn.Linear(self.embed_size * 2, self.embed_size).cuda())
            self.update_rn_state.append(nn.Linear(self.embed_size * 2, self.embed_size).cuda())
            self.rn_query_st_tr.append(nn.Linear(self.embed_size * (self.p_hop), self.embed_size).cuda())

            self.rh_query.append(nn.Linear(self.embed_size, self.embed_size, bias=False).cuda())
            self.o_r_query.append(nn.Linear(self.embed_size, self.embed_size, bias=False).cuda())
            self.v_query.append(nn.Linear(self.embed_size, self.embed_size, bias=False).cuda())
            self.t_u_query.append(nn.Linear(self.embed_size, self.embed_size, bias=False).cuda())

        self.rn_state_tr_query = nn.ModuleList(self.rn_state_tr_query)
        self.update_rn_state = nn.ModuleList(self.update_rn_state)
        self.rn_query_st_tr = nn.ModuleList(self.rn_query_st_tr)

        self.rh_query = nn.ModuleList(self.rh_query)
        self.o_r_query = nn.ModuleList(self.o_r_query)
        self.v_query = nn.ModuleList(self.v_query)
        self.t_u_query = nn.ModuleList(self.t_u_query)

        self.rn_cal_state_prop = nn.Linear(self.embed_size, 1, bias=False).cuda()

    def bulid_mode_user(self):

        self.relation_emb = nn.Embedding(len(self.rela_2_index), self.embed_size * self.embed_size)
        
        # if self.h0_embbed == True:
        #     self.transform_matrix_ = nn.Linear(self.embed_size * (self.p_hop + 1), self.embed_size, bias=True)
        # else:
        #     self.transform_matrix_ = nn.Linear(self.embed_size * (self.p_hop), self.embed_size, bias=True)

        self.update_us_tr = []
        for hop in range(self.p_hop):
            self.update_us_tr.append(nn.Linear(self.embed_size * 2, self.embed_size).cuda())
        self.update_us_tr = nn.ModuleList(self.update_us_tr)

        self.cal_state_prop = nn.Linear(self.embed_size * 3, 1)

    
    def update_query_embedding(self, selc_entitiy):
        # update before query
        selc_entitiy = selc_entitiy.repeat(1, self.memories_t[0].shape[1], 1)
        for hop in range(self.p_hop):
            tmp_memories_t = th.cat([self.memories_t[hop], selc_entitiy], -1)
            self.memories_t[hop] = self.update_us_tr[hop](tmp_memories_t)

    def update_path_info_memories(self, up_date_hop):
        new_memories_h = {}
        new_memories_r = {}
        new_memories_t = {}

        for i in range(max(1,self.p_hop)):
            new_memories_h[i] = []
            new_memories_r[i] = []
            new_memories_t[i] = []

            for row in up_date_hop:
                new_memories_h[i].append(self.memories_h[i][row,:,:].unsqueeze(0))
                new_memories_r[i].append(self.memories_r[i][row,:,:,:].unsqueeze(0))
                new_memories_t[i].append(self.memories_t[i][row,:,:].unsqueeze(0))

            self.memories_h[i] = th.cat(new_memories_h[i], 0).to(self.device)
            self.memories_r[i] = th.cat(new_memories_r[i], 0).to(self.device)
            self.memories_t[i] = th.cat(new_memories_t[i], 0).to(self.device)

class Backbone_2(Backbone_1):
    def __init__(self, args, user_triplet_set, rela_2_index, act_dim, gamma=0.99, hidden_sizes=[512, 256]):
        super().__init__(args, user_triplet_set, rela_2_index, act_dim, gamma, hidden_sizes)

    def reset(self, uids=None):
        
        self.lstm_state_cache = []

        self.uids = [uid for uid in uids for _ in range(1)]
        self.memories_h = {}
        self.memories_r = {}
        self.memories_t = {}

        for i in range(max(1,self.p_hop)):

            self.memories_h[i] = th.cat([th.cat([self.kg_emb.lookup_emb(u_set[0], type_index = torch.LongTensor([u_set[1]]).to(self.device))
                                 for u_set in self.user_triplet_set[user][i][0]], 0).unsqueeze(0) for user in self.uids], 0)
            
            self.memories_r[i] = th.cat([self.relation_emb(torch.LongTensor([self.rela_2_index[relation] 
                                    for relation in self.user_triplet_set[user][i][1]]).to(self.device)).unsqueeze(0) for user in self.uids], 0)
            
            self.memories_r[i] = self.memories_r[i].view(-1, self.n_memory, self.embed_size, self.embed_size)

            self.memories_t[i] = th.cat([th.cat([self.kg_emb.lookup_emb(u_set[0], type_index = torch.LongTensor([u_set[1]]).to(self.device))
                                 for u_set in self.user_triplet_set[user][i][2]], 0).unsqueeze(0) for user in self.uids], 0)

        self.prev_state_h, self.prev_state_c = self.state_lstm.set_up_hidden_state(len(self.uids))
    

    def select_action(self, batch_state, batch_next_action_emb, batch_act_mask, device):
        
        act_mask = torch.BoolTensor(batch_act_mask).to(device)  # Tensor of [bs, act_dim]
        
        state_output, res_user_emb = batch_state[0], batch_state[1]
        next_enti_emb, next_action_emb = batch_next_action_emb[0], batch_next_action_emb[1]
        probs, value = self((state_output, res_user_emb, next_enti_emb, next_action_emb, act_mask))  # act_probs: [bs, act_dim], state_value: [bs, 1]

        m = Categorical(probs)
        acts = m.sample()  # Tensor of [bs, ], requires_grad=False

        # [CAVEAT] If sampled action is out of action_space, choose the first action in action_space.
        valid_idx = act_mask.gather(1, acts.view(-1, 1)).view(-1)
        acts[valid_idx == 0] = 0

        self.saved_actions.append(SavedAction(m.log_prob(acts), value))
        self.entropy.append(m.entropy())

        return acts.cpu().numpy().tolist()

    def generate_act_emb(self, batch_path, batch_curr_actions):
        all_action_set = [self._get_actions(index, actions_sets[0], actions_sets[1]) 
                    for index, actions_sets in enumerate(zip(batch_path, batch_curr_actions))]
        enti_emb = th.cat([action_set[0].unsqueeze(0) for action_set in all_action_set], 0)
        next_action_state = th.cat([action_set[1].unsqueeze(0) for action_set in all_action_set], 0)

        return [enti_emb, next_action_state]

    def _get_actions(self, index, curr_path, curr_actions):

        last_relation, curr_node_type, curr_node_id = curr_path[-1]
        entities_embs = []
        relation_embs = []

        for action_set in curr_actions:
            if action_set[0] == SELF_LOOP: next_node_type = curr_node_type
            else: next_node_type = self._get_next_node_type(curr_node_type, action_set[0], action_set[1])
            enti_emb = self.kg_emb.lookup_emb(next_node_type,
                            type_index = torch.LongTensor([action_set[1]]).to(self.device))
            entities_embs.append(enti_emb)
            rela_emb = self.kg_emb.lookup_rela_emb(action_set[0])
            relation_embs.append(rela_emb)

        pad_emb = self.kg_emb.lookup_rela_emb(PADDING)
        for _ in range(self.act_dim - len(entities_embs)):
            entities_embs.append(pad_emb)
            relation_embs.append(pad_emb)

        enti_emb = th.cat(entities_embs, 0)
        rela_emb = th.cat(relation_embs, 0)

        next_action_state = th.cat([enti_emb, rela_emb], -1)
        
        return [enti_emb, next_action_state]

class Backbone(Backbone_2):
    def __init__(self, args, user_triplet_set, rela_2_index, act_dim, gamma=0.99, hidden_sizes=[512, 256]):
        super().__init__(args, user_triplet_set, rela_2_index, act_dim, gamma, hidden_sizes)

        self.bulid_mode_user()
        self.bulid_model_rl()
        self.bulid_model_reasoning()

        self.scalar = nn.Parameter(torch.Tensor([args.lambda_num]), requires_grad=True)
        self.l2_weight = args.l2_weight

        ones_dummy_rela = torch.ones(max(self.user_triplet_set) * 2 + 1, 1, self.embed_size)
        self.dummy_rela = nn.Parameter(ones_dummy_rela, requires_grad=True).to(self.device)
        self.dummy_rela_emb = nn.Embedding(max(self.user_triplet_set) * 2 + 1, self.embed_size * self.embed_size).to(self.device)
