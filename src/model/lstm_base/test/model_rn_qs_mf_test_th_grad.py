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
# from model_refine.lstm_base.model_kg import KnowledgeEmbedding_memory, KnowledgeEmbedding_memory_graph
from model_refine.lstm_base.model_kg import KnowledgeEmbedding_memory, KnowledgeEmbedding_memory_graph
from model_refine.lstm_base.backbone_lstm import EncoderRNN, EncoderRNN_batch, KGState_LSTM, KGState_LSTM_no_rela
from model_refine.lstm_base.model_lstm import ActorCritic_lstm_base
from model_refine.lstm_query_state.model_rn_qs import *
from model_refine.lstm_query_state.model_rn_qs_sp import lstm_query_st_us_rn_up_sp_sum
from utils import *

SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])


class lstm_query_usrn_con_mf_test(lstm_query_st_us_rn_up):
    def __init__(self, args, user_triplet_set, rela_2_index, act_dim, gamma=0.99, hidden_sizes=[512, 256]):
        super().__init__(args, user_triplet_set, rela_2_index, act_dim, gamma, hidden_sizes)
    
        self.cast_st_save = args.cast_st_save

    def bulid_model_rl(self):
        self.state_lstm = KGState_LSTM(self.args, history_len=1)

        self.l1 = nn.Linear(2 * self.embed_size, self.hidden_sizes[1])
        self.l2 = nn.Linear(self.hidden_sizes[0], self.hidden_sizes[1])

        self.actor = nn.Linear(self.hidden_sizes[1], self.act_dim)
        self.critic = nn.Linear(self.hidden_sizes[1], 1)

        self.saved_actions = []
        self.rewards = []
        self.entropy = []


    def bulid_model_reasoning(self):

        self.reasoning_step = self.args.reasoning_step
        self.rn_state_tr_query = []
        self.rn_cal_state_prop = []
        self.update_rn_state = []
        self.rn_query_st_tr = []
        for i in range(self.reasoning_step):
            self.rn_state_tr_query.append(nn.Linear(self.embed_size * 2, self.embed_size).cuda())
            self.rn_cal_state_prop.append(nn.Linear(self.embed_size * 3, 1).cuda())
            self.update_rn_state.append(nn.Linear(self.embed_size * 3, self.embed_size * 2).cuda())

            if i == self.reasoning_step - 1:
                self.rn_query_st_tr.append(nn.Linear(self.embed_size * (self.p_hop), self.embed_size * 2).cuda())
            else:    
                self.rn_query_st_tr.append(nn.Linear(self.embed_size * (self.p_hop), self.embed_size).cuda())

        self.rn_state_tr_query = nn.ModuleList(self.rn_state_tr_query)
        self.rn_cal_state_prop = nn.ModuleList(self.rn_cal_state_prop)
        self.update_rn_state = nn.ModuleList(self.update_rn_state)
        self.rn_query_st_tr = nn.ModuleList(self.rn_query_st_tr)

    def reset(self, uids=None):
        
        self.lstm_state_cache = []

        self.uids = [uid for uid in uids for _ in range(self.sub_batch_size)]
        self.memories_h = {}
        self.memories_r = {}
        self.memories_t = {}


        self.memories_h_index = {}
        self.memories_r_index = {}
        self.memories_t_index = {}

        for i in range(max(1,self.p_hop)):
            self.memories_h[i] = th.cat([th.cat([self.kg_emb.lookup_emb(u_set[0], type_index = torch.LongTensor([u_set[1]]).to(self.device))
                                 for u_set in self.user_triplet_set[user][i][0]], 0).unsqueeze(0) for user in self.uids], 0)
            
            self.memories_r[i] = th.cat([self.relation_emb(torch.LongTensor([self.rela_2_index[relation] 
                                    for relation in self.user_triplet_set[user][i][1]]).to(self.device)).unsqueeze(0) for user in self.uids], 0)
            
            self.memories_r[i] = self.memories_r[i].view(-1, self.n_memory, self.embed_size, self.embed_size)
            
            self.memories_t[i] = th.cat([th.cat([self.kg_emb.lookup_emb(u_set[0], type_index = torch.LongTensor([u_set[1]]).to(self.device))
                                 for u_set in self.user_triplet_set[user][i][2]], 0).unsqueeze(0) for user in self.uids], 0)

            self.memories_h_index[i] = [[[u_set[0], self.args.asin2title[self.args.data_entity_ind_id[u_set[0]][u_set[1]]] if 
              self.args.data_entity_ind_id[u_set[0]][u_set[1]] in self.args.asin2title else self.args.data_entity_ind_id[u_set[0]][u_set[1]]] 
                    for u_set in self.user_triplet_set[user][i][0]] for user in self.uids]

                    
            # print('self.memories_r[i] = ', self.memories_r[i].shape)
            self.memories_r_index[i] = [[relation for relation in self.user_triplet_set[user][i][1]] for user in self.uids]

            self.memories_t_index[i] = [[[u_set[0], self.args.asin2title[self.args.data_entity_ind_id[u_set[0]][u_set[1]]] if 
              self.args.data_entity_ind_id[u_set[0]][u_set[1]] in self.args.asin2title else self.args.data_entity_ind_id[u_set[0]][u_set[1]]] 
                    for u_set in self.user_triplet_set[user][i][2]]  for user in self.uids]

        self.prev_state_h, self.prev_state_c = self.state_lstm.set_up_hidden_state(len(self.uids))

    def bulid_mode_user(self):

        self.relation_emb = nn.Embedding(len(self.rela_2_index), self.embed_size * self.embed_size)
        
        if self.h0_embbed == True:
            self.transform_matrix_ = nn.Linear(self.embed_size * (self.p_hop + 1), self.embed_size, bias=True)
        else:
            self.transform_matrix_ = nn.Linear(self.embed_size * (self.p_hop), self.embed_size, bias=True)

        self.update_us_tr = []
        for hop in range(self.p_hop):
            self.update_us_tr.append(nn.Linear(self.embed_size * 2, self.embed_size).cuda())
        self.update_us_tr = nn.ModuleList(self.update_us_tr)

        self.cal_state_prop = nn.Linear(self.embed_size * 3, 1)

    def forward(self, inputs):
        state, res_user_emb, batch_next_action_emb,  act_mask = inputs  # state: [bs, state_dim], act_mask: [bs, act_dim]
        # state = state.squeeze()
        # res_user_emb = res_user_emb.squeeze()

        state_tr = state.unsqueeze(1).repeat(1, batch_next_action_emb.shape[1], 1)
        probs_st = state_tr * batch_next_action_emb 

        res_user_emb = res_user_emb.unsqueeze(1).repeat(1, batch_next_action_emb.shape[1], 1)
        
        probs_user = res_user_emb * batch_next_action_emb

        probs = probs_st + probs_user

        probs = probs.sum(-1)
        probs = probs.masked_fill(~act_mask, value=torch.tensor(-1e10))
        act_probs = F.softmax(probs, dim=-1)

        x = self.l1(state)
        x = F.dropout(F.elu(x), p=0.4)

        state_values = self.critic(x)  # Tensor of [bs, 1]
        return act_probs, state_values

    def select_action(self, batch_state, batch_next_action_emb, 
                                        batch_act_mask, device):
        
        act_mask = torch.BoolTensor(batch_act_mask).to(device)  # Tensor of [bs, act_dim]
        
        state_output, res_user_emb = batch_state[0], batch_state[1]
        probs, value = self((state_output, res_user_emb, batch_next_action_emb, act_mask))  # act_probs: [bs, act_dim], state_value: [bs, 1]

        m = Categorical(probs)
        acts = m.sample()  # Tensor of [bs, ], requires_grad=False

        # [CAVEAT] If sampled action is out of action_space, choose the first action in action_space.
        valid_idx = act_mask.gather(1, acts.view(-1, 1)).view(-1)
        acts[valid_idx == 0] = 0

        self.saved_actions.append(SavedAction(m.log_prob(acts), value))
        self.entropy.append(m.entropy())

        return acts.cpu().numpy().tolist()

    def rn_query_st(self, state, T_RN):

        user_embeddings = self.memories_h[0][:,0]

        # state = th.cat([state.squeeze(), user_embeddings], -1)
        state = th.cat([state.squeeze().unsqueeze(0)], -1)
        query = self.rn_state_tr_query[T_RN](state)

        o_list = []

        self.by_hop_query = {}

        for hop in range(self.p_hop):
            self.by_hop_query[hop] = []
            
            h_expanded = torch.unsqueeze(self.memories_t[hop], dim=3)

            Rh = torch.squeeze(torch.matmul(self.memories_r[hop], h_expanded)).unsqueeze(0)
            # [batch_size, dim, 1]
            v =  query.unsqueeze(1).repeat(1, self.memories_t[0].shape[1], 1)
            t_u = user_embeddings.unsqueeze(1).repeat(1, self.memories_t[0].shape[1], 1)

            t_state = th.cat([Rh, v, t_u], -1)

            probs = torch.squeeze(self.rn_cal_state_prop[T_RN](t_state)).unsqueeze(0)

            # [batch_size, n_memory]
            probs_normalized = F.softmax(probs, dim=1)

            # [batch_size, n_memory, 1]
            probs_expanded = torch.unsqueeze(probs_normalized, dim=2)

            probs_expanded_to_list = probs_expanded.tolist()
            for t_ , r_, p_ in zip(self.memories_t_index[hop], self.memories_r_index[hop], probs_expanded_to_list[0]):
                state_trp = ', '.join([t_[0][:2], str(t_[1]), r_[:2]])
                if len(state_trp) < 15: 
                    state_trp += ' ' * (15 - len(state_trp))
                state_trp = [state_trp,  str(round(p_[0], 5))]
                state_trp = ', prb = '.join(state_trp)
                if len(state_trp) < 25: 
                    state_trp += ' ' * (25 - len(state_trp))
                self.by_hop_query[hop].append(state_trp)


            # [batch_size, dim]
            o = (self.memories_t[hop] * probs_expanded).sum(dim=1)

            o_list.append(o)

        o_list = torch.cat(o_list, -1)

        user_o = self.rn_query_st_tr[T_RN](o_list)

        return user_o


    def generate_st_emb(self, batch_path, up_date_hop = None):
        if up_date_hop != None:
            self.update_path_info(up_date_hop)
            self.update_path_info_memories(up_date_hop)

        self.current_step = {}

        tmp_state = [self._get_state_update(index, path) for index, path in enumerate(batch_path)]
        
        all_state = th.cat([ts[1].unsqueeze(0) for ts in tmp_state], 0)

        state_output, self.prev_state_h, self.prev_state_c = self.state_lstm(all_state, 
                    self.prev_state_h,  self.prev_state_c)

        state_tmp = state_output
        for rn_step in range(self.reasoning_step):
            query_state = self.rn_query_st(state_tmp, rn_step)
            self.current_step[rn_step] = self.by_hop_query
            if rn_step < self.reasoning_step - 1: 
                state_tmp_ = th.cat([query_state.unsqueeze(1), state_tmp], -1)
                state_tmp = self.update_rn_state[rn_step](state_tmp_)
        res_user_emb = query_state

        state_output = state_output.squeeze().unsqueeze(0)
        res_user_emb = res_user_emb.squeeze().unsqueeze(0)

        return [state_output, res_user_emb]


    def generate_act_emb(self, batch_path, batch_curr_actions):
        self.current_step['user'] = str(self.uids[0])
        b_path = [','.join([str(uni) for uni in bpa]) for bpa in batch_path[0]]        
        b_action = [' id = '.join([str(uni) for uni in bpa]) for bpa in batch_curr_actions[0]]

        self.current_step["path"] = 'path = ' + ', next = '.join(b_path)
        self.current_step["actions"] = b_action
        # input()
        return th.cat([self._get_actions(index, actions_sets[0], 
            actions_sets[1]).unsqueeze(0) for index, actions_sets in enumerate(zip(batch_path, batch_curr_actions))], 0)

    def _record_case_study(self):

        # print('self.cast_st_save = ', self.cast_st_save)

        eva_file = open(self.cast_st_save, "a")
        eva_file.write("*" * 50)
        eva_file.write('\n')
        eva_file.write('user = ' + self.current_step['user'])
        eva_file.write('\n')
        eva_file.write(self.current_step['path'])
        eva_file.write('\n')
        eva_file.write("*" * 50)
        eva_file.write('\n')
        eva_file.write("querying result")
        eva_file.write('\n')
        for hop in range(self.p_hop):
            eva_file.write("*" * 50)
            eva_file.write('\n')
            eva_file.write("hop = " + str(hop))
            eva_file.write('\n')
            tmp_list = []
            for rn_step in range(self.reasoning_step):  
                tmp_list.append(self.current_step[rn_step][hop])
            for state_s in zip(*tmp_list):
                # print('state_s = ', list(state_s))
                eva_file.write(' n_hop = '.join(list(state_s)))
                eva_file.write('\n')
        eva_file.write("*" * 50)
        eva_file.write('\n')
        for ac_pro in self.current_step['actions_pro']:
            # print(ac_pro)
            eva_file.write(ac_pro)
            eva_file.write('\n')
        eva_file.write("next_action = " + self.current_step['next_acts'])
        eva_file.write('\n')
        eva_file.write("*" * 50)
        eva_file.close()

    def _record_case_study_az(self):

        # print('self.cast_st_save = ', self.cast_st_save)

        eva_file = open(self.cast_st_save, "a")
        eva_file.write("*" * 50)
        eva_file.write('\n')
        eva_file.write('user = ' + self.current_step['user'])
        eva_file.write('\n')
        eva_file.write(self.current_step['path'])
        eva_file.write('\n')
        eva_file.write("*" * 50)
        eva_file.write('\n')
        eva_file.write("querying result")
        eva_file.write('\n')
        for hop in range(self.p_hop):
            eva_file.write("*" * 50)
            eva_file.write('\n')
            eva_file.write("hop = " + str(hop))
            eva_file.write('\n')
            tmp_list = []
            for rn_step in range(self.reasoning_step):  
                tmp_list.append(self.current_step[rn_step][hop])
            for state_s in zip(*tmp_list):
                # print('state_s = ', list(state_s))
                eva_file.write(' n_hop = '.join(list(state_s)))
                eva_file.write('\n')
        eva_file.write("*" * 50)
        eva_file.write('\n')
        for ac_pro in self.current_step['actions_pro']:
            # print(ac_pro)
            eva_file.write(ac_pro)
            eva_file.write('\n')
        eva_file.write("next_action = " + self.current_step['next_acts'])
        eva_file.write('\n')
        eva_file.write("*" * 50)
        eva_file.close()

class lstm_query_usrn_con_et_mf_v28_test_th(lstm_query_usrn_con_mf_test):
    def __init__(self, args, user_triplet_set, rela_2_index, act_dim, gamma=0.99, hidden_sizes=[512, 256]):
        super().__init__(args, user_triplet_set, rela_2_index, act_dim, gamma, hidden_sizes)
        self.scalar = Variable(torch.Tensor([0.5]), requires_grad=True).cuda().to(self.device)
        self.scalar = nn.Parameter(self.scalar)

        self.dummy_rela = torch.ones(max(self.user_triplet_set) * 2 + 1, 1, self.embed_size)
        self.dummy_rela = nn.Parameter(self.dummy_rela, requires_grad=True).to(self.device)
        self.dummy_rela_emb = nn.Embedding(max(self.user_triplet_set) * 2 + 1, self.embed_size * self.embed_size).to(self.device)

        self.record_path_info = {}

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
        
        if self.h0_embbed == True:
            self.transform_matrix_ = nn.Linear(self.embed_size * (self.p_hop + 1), self.embed_size, bias=True)
        else:
            self.transform_matrix_ = nn.Linear(self.embed_size * (self.p_hop), self.embed_size, bias=True)

        self.update_us_tr = []
        for hop in range(self.p_hop):
            self.update_us_tr.append(nn.Linear(self.embed_size * 2, self.embed_size).cuda())
        self.update_us_tr = nn.ModuleList(self.update_us_tr)

        self.cal_state_prop = nn.Linear(self.embed_size * 3, 1)

    def forward(self, inputs):
        state, res_user_emb, next_enti_emb, next_action_emb,  act_mask = inputs  # state: [bs, state_dim], act_mask: [bs, act_dim]

        state_tr = state.unsqueeze(1).repeat(1, next_action_emb.shape[1], 1)
        probs_st = state_tr * next_action_emb 

        res_user_emb = res_user_emb.unsqueeze(1).repeat(1, next_enti_emb.shape[1], 1)
        probs_user = res_user_emb * next_enti_emb

        probs_st = probs_st.sum(-1)

        probs_user = probs_user.sum(-1)

        scalar = self.scalar.unsqueeze(1).repeat(probs_user.shape[0],1)

        probs = probs_st + self.scalar * probs_user
        probs = probs.masked_fill(~act_mask, value=torch.tensor(-1e10))
        act_probs = F.softmax(probs, dim=-1)

        x = self.l1(state)
        x = F.dropout(F.elu(x), p=0.4)

        state_values = self.critic(x)  # Tensor of [bs, 1]
        return act_probs, state_values

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

    def rn_query_st(self, state, relation_embed_dual, rn_step):

        user_embeddings = self.memories_h[0][:,0]

        # state = th.cat([state.squeeze(), user_embeddings], -1)
        state = th.cat([state.squeeze()], -1)
        relation_embed_dual = th.cat([relation_embed_dual.squeeze()], -1)

        self.by_hop_query = {}
        self.by_hop_query_record = {}

        o_list = []
        for hop in range(self.p_hop):
            self.by_hop_query[hop] = []
            self.by_hop_query_record[hop] = []

            h_expanded = torch.unsqueeze(self.memories_t[hop], dim=3)

            Rh = torch.squeeze(torch.matmul(self.memories_r[hop], h_expanded))
            # [batch_size, dim, 1]
            v =  state.unsqueeze(1).repeat(1, self.memories_t[0].shape[1], 1)

            r_v = relation_embed_dual.unsqueeze(1).repeat(1, self.memories_t[0].shape[1], 1, 1)

            r_vh = torch.squeeze(torch.matmul(r_v, h_expanded))

            t_u = user_embeddings.unsqueeze(1).repeat(1, self.memories_t[0].shape[1], 1)

            q_Rh = self.rh_query[rn_step](Rh)
            q_v = self.v_query[rn_step](v)
            t_u = self.t_u_query[rn_step](t_u)
            o_r = self.o_r_query[rn_step](r_vh)

            t_state = torch.tanh(q_Rh + q_v + t_u + o_r)
            # print('t_state = ', t_state.shape)
            probs = torch.squeeze(self.rn_cal_state_prop(t_state))

            probs_normalized = F.softmax(probs, dim=1)

            probs_expanded = torch.unsqueeze(probs_normalized, dim=2)

            probs_expanded_to_list = probs_expanded.tolist()

            for prob_index in range(len(probs_expanded_to_list)):
                by_hop_query_tmp = []
                by_hop_query_dict = []

                for t_ , r_, p_ in zip(self.memories_t_index[hop][prob_index], self.memories_r_index[hop][prob_index], probs_expanded_to_list[prob_index]):
                    by_hop_query_dict.append([t_, r_,  p_])

                    state_trp = ', '.join([t_[0][:2], str(t_[1]), r_[:2]])
                    if len(state_trp) < 15: 
                        state_trp += ' ' * (15 - len(state_trp))
                    state_trp = [state_trp,  str(round(p_[0], 5))]
                    state_trp = ', prb = '.join(state_trp)
                    if len(state_trp) < 25: 
                        state_trp += ' ' * (25 - len(state_trp))
                    by_hop_query_tmp.append(state_trp)

                by_hop_query_highlight_list = sorted(by_hop_query_dict, key = lambda s: s[2])[::-1][:8]

                self.by_hop_query[hop].append(by_hop_query_tmp)

                self.by_hop_query_record[hop].append(by_hop_query_highlight_list)

            # [batch_size, dim]
            o = (self.memories_t[hop] * probs_expanded).sum(dim=1).unsqueeze(1)

            o_list.append(o)

        o_list = torch.cat(o_list, 1)

        user_o = o_list.sum(1)

        return user_o

    def update_query_embedding(self, selc_entitiy):
        # update before query
        selc_entitiy = selc_entitiy.repeat(1, self.memories_t[0].shape[1], 1)
        
        for hop in range(self.p_hop):
            tmp_memories_t = th.cat([self.memories_t[hop], selc_entitiy], -1)
            self.memories_t[hop] = self.update_us_tr[hop](tmp_memories_t)

    # def update_path_info_memories(self, up_date_hop):

    #     new_memories_h = {}
    #     new_memories_r = {}
    #     new_memories_t = {}

    #     for i in range(max(1,self.p_hop)):
    #         new_memories_h[i] = []
    #         new_memories_r[i] = []
    #         new_memories_t[i] = []

    #         for row in up_date_hop:

    #             new_memories_h[i].append(self.memories_h[i][row,:,:].unsqueeze(0))
    #             new_memories_r[i].append(self.memories_r[i][row,:,:,:].unsqueeze(0))
    #             new_memories_t[i].append(self.memories_t[i][row,:,:].unsqueeze(0))

    #         self.memories_h[i] = th.cat(new_memories_h[i], 0).to(self.device)
    #         self.memories_r[i] = th.cat(new_memories_r[i], 0).to(self.device)
    #         self.memories_t[i] = th.cat(new_memories_t[i], 0).to(self.device)

    #     new_memories_h_index = {}
    #     new_memories_r_index = {}
    #     new_memories_t_index = {}

    #     for i in range(max(1,self.p_hop)):
    #         new_memories_h_index[i] = []
    #         new_memories_r_index[i] = []
    #         new_memories_t_index[i] = []

    #         for row in up_date_hop:
    #             new_memories_h_index[i].append(self.memories_h_index[i][row])
    #             new_memories_r_index[i].append(self.memories_r_index[i][row])
    #             new_memories_t_index[i].append(self.memories_t_index[i][row])

    #         self.memories_h_index[i] = new_memories_h_index[i]
    #         self.memories_r_index[i] = new_memories_r_index[i]
    #         self.memories_t_index[i] = new_memories_t_index[i]


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

        new_memories_h_index = {}
        new_memories_r_index = {}
        new_memories_t_index = {}

        for i in range(max(1,self.p_hop)):
            new_memories_h_index[i] = []
            new_memories_r_index[i] = []
            new_memories_t_index[i] = []

            for row in up_date_hop:
                new_memories_h_index[i].append(self.memories_h_index[i][row])
                new_memories_r_index[i].append(self.memories_r_index[i][row])
                new_memories_t_index[i].append(self.memories_t_index[i][row])

            self.memories_h_index[i] = new_memories_h_index[i]
            self.memories_r_index[i] = new_memories_r_index[i]
            self.memories_t_index[i] = new_memories_t_index[i]

    def generate_st_emb(self, batch_path, up_date_hop = None):
        if up_date_hop != None:
            self.update_path_info(up_date_hop)
            self.update_path_info_memories(up_date_hop)

        self.current_step = {}

        tmp_state = [self._get_state_update(index, path) for index, path in enumerate(batch_path)]

        all_state = th.cat([ts[3].unsqueeze(0) for ts in tmp_state], 0)

        if len(batch_path[0]) != 1:
            selc_entitiy = th.cat([ts[0].unsqueeze(0) for ts in tmp_state], 0)
            self.update_query_embedding(selc_entitiy)

        state_output, self.prev_state_h, self.prev_state_c = self.state_lstm(all_state, 
                    self.prev_state_h,  self.prev_state_c)

        curr_node_embed = th.cat([ts[0].unsqueeze(0) for ts in tmp_state], 0)
        relation_embed = th.cat([ts[1].unsqueeze(0) for ts in tmp_state], 0)
        relation_embed_dual = th.cat([ts[2].unsqueeze(0) for ts in tmp_state], 0)

        state_tmp = relation_embed

        for rn_step in range(self.reasoning_step):
            query_state = self.rn_query_st(state_tmp, relation_embed_dual, rn_step)
            self.current_step[rn_step] = self.by_hop_query
            self.current_step[str(rn_step) + 'record'] = self.by_hop_query_record
            if rn_step < self.reasoning_step - 1: 
                state_tmp_ = th.cat([query_state, state_tmp], -1)
                state_tmp = self.update_rn_state[rn_step](state_tmp_)
        # input()
        res_user_emb = query_state

        state_output = state_output.squeeze()
        res_user_emb = res_user_emb.squeeze()

        return [state_output, res_user_emb]


    def generate_act_emb(self, batch_path, batch_curr_actions):
        self.current_step['user'] = [str(ui) for ui in self.uids]
        b_path = [[','.join([str(uni) for uni in bpa])  for bpa in ba_path] for ba_path in batch_path]        
        b_action = [[' id = '.join([str(uni) for uni in bpa]) for bpa in batch_actions] for batch_actions in batch_curr_actions]

        n_batch_path = []
        list_batch_path = []
        for ba_path in batch_path:
            n_ba_path = []
            list_ba_path = []
            for bpa in ba_path:
                bpa = list(bpa)
                bpa[2] = self.args.data_entity_ind_id[bpa[1]][bpa[2]]
                if bpa[2] in self.args.asin2title:
                    bpa[2] = self.args.asin2title[bpa[2]]
                n_ba_path.append(','.join([str(uni) for uni in bpa]))
                list_ba_path.append(bpa)
            n_batch_path.append(n_ba_path)
            list_batch_path.append(list_ba_path)

        n_batch_curr_actions = []
        list_batch_curr_actions = []
        for index, actions_sets in enumerate(zip(batch_path, batch_curr_actions)):
            curr_path, curr_actions = actions_sets[0], actions_sets[1]
            last_relation, curr_node_type, curr_node_id = curr_path[-1]
            n_curr_actions = []
            for action_set in curr_actions:
                action_set = list(action_set)
                if action_set[0] == SELF_LOOP: next_node_type = curr_node_type
                else: next_node_type = self._get_next_node_type(curr_node_type, action_set[0], action_set[1])
                action_set[1] = self.args.data_entity_ind_id[next_node_type][action_set[1]]
                if action_set[1] in self.args.asin2title:
                    action_set[1] = self.args.asin2title[action_set[1]]
                n_curr_actions.append(' id = '.join([str(uni) for uni in action_set]))
            n_batch_curr_actions.append(n_curr_actions)

        self.current_step["path"] = ['path = ' + ', next = '.join(cn_b_path) for cn_b_path in n_batch_path]
        self.current_step["path_list"] = list_batch_path
        self.current_step["actions"] = n_batch_curr_actions

        all_action_set = [self._get_actions(index, actions_sets[0], actions_sets[1]) 
                    for index, actions_sets in enumerate(zip(batch_path, batch_curr_actions))]
        enti_emb = th.cat([action_set[0].unsqueeze(0) for action_set in all_action_set], 0)
        next_action_state = th.cat([action_set[1].unsqueeze(0) for action_set in all_action_set], 0)

        return [enti_emb, next_action_state]

    def _get_state_update(self, index, path):
        """Return state of numpy vector: [user_embed, curr_node_embed, last_node_embed, last_relation]."""
        if len(path) == 1:
            if self.user_o == True:
                user_embed = self.global_user[index,:].unsqueeze(0)
            else:
                # print('user embedd only kg')
                user_embed = self.kg_emb.lookup_emb(USER, type_index = 
                        torch.LongTensor([path[0][-1]]).to(self.device))[0].unsqueeze(0)
            curr_node_embed = user_embed
            last_relation_embed = self.dummy_rela[path[0][-1], :, :]
            relation_embed_dual = self.dummy_rela_emb(torch.LongTensor([path[0][-1]]).to(self.device))
            relation_embed_dual = relation_embed_dual.view(self.embed_size, self.embed_size)

            st_emb = self.action_encoder(last_relation_embed, user_embed)
        else:
            last_relation, curr_node_type, curr_node_id = path[-1]
            # print('last_relation, curr_node_type, curr_node_id  = ', last_relation, curr_node_type, curr_node_id )
            curr_node_embed = self.kg_emb.lookup_emb(curr_node_type, 
                    type_index = torch.LongTensor([curr_node_id]).to(self.device))[0].unsqueeze(0)
            last_relation_embed = self.kg_emb.lookup_rela_emb(last_relation)[0].unsqueeze(0)

            relation_embed_dual = self.relation_emb(torch.LongTensor([self.rela_2_index[last_relation]]).to(self.device))
            relation_embed_dual = relation_embed_dual.view(self.embed_size, self.embed_size)

            st_emb = self.action_encoder(last_relation_embed, curr_node_embed)

        return [curr_node_embed, last_relation_embed.squeeze(), relation_embed_dual.squeeze(), st_emb]

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

    def _record_case_study(self):
        eva_file = open(self.cast_st_save, "a")
        for ind in range(len(self.current_step['user'])):

            rd_uid = self.current_step["user"][ind]
            if rd_uid not in self.record_path_info:
                self.record_path_info[rd_uid] = {}

            re_path = self.current_step["path"][ind]
            if re_path not in self.record_path_info[rd_uid]:
                self.record_path_info[rd_uid][re_path] = {}

            self.record_path_info[rd_uid][re_path]['path_list'] =  self.current_step["path_list"][ind]

            self.record_path_info[rd_uid][re_path]['action'] = self.current_step["actions"][ind]

            for hop in range(self.p_hop):
                tmp_list = []
                for rn_step in range(self.reasoning_step):  
                    tmp_list.append(self.current_step[rn_step][hop][ind])
                re_record_string = ''
                for state_s in zip(*tmp_list):
                    re_record_string += ' n_hop = '.join(list(state_s))

                self.record_path_info[rd_uid][re_path]['hop_' + str(hop)] = self.current_step[str(rn_step) + 'record'][hop][ind]

            self.record_path_info[rd_uid][re_path]['next_acts'] =  self.current_step['next_acts'][ind]

            eva_file.write("*" * 50)
            eva_file.write('\n')
            eva_file.write('user = ' + self.current_step['user'][ind])
            eva_file.write('\n')
            eva_file.write(self.current_step['path'][ind])
            eva_file.write('\n')
            eva_file.write("*" * 50)
            eva_file.write('\n')
            eva_file.write("querying result")
            eva_file.write('\n')
            for hop in range(self.p_hop):
                eva_file.write("*" * 50)
                eva_file.write('\n')
                eva_file.write("hop = " + str(hop))
                eva_file.write('\n')
                tmp_list = []
                for rn_step in range(self.reasoning_step):  
                    tmp_list.append(self.current_step[rn_step][hop][ind])
                for state_s in zip(*tmp_list):                    # print('state_s = ', list(state_s))
                    eva_file.write(', = '.join(list(state_s)))
                    eva_file.write('\n')
            eva_file.write("*" * 50)
            eva_file.write('\n')
            for ac_pro in self.current_step['actions_pro'][ind]:
                eva_file.write(ac_pro)
                eva_file.write('\n')
            eva_file.write("next_action = " + ', '.join([str(k) for k in self.current_step['next_acts'][ind]]))
            eva_file.write('\n')
            eva_file.write("*" * 50)
        eva_file.close()

    def save_path_record_dict(self, record_path):
        with open(record_path, 'wb') as handle:
            pickle.dump(self.record_path_info, handle, protocol=pickle.HIGHEST_PROTOCOL)



class lstm_query_us_rn_up_sp_con_mf_v28_up_test_th(lstm_query_usrn_con_mf_test):
    def __init__(self, args, user_triplet_set, rela_2_index, act_dim, gamma=0.99, hidden_sizes=[512, 256]):
        super().__init__(args, user_triplet_set, rela_2_index, act_dim, gamma, hidden_sizes)
    
        # self.scalar = torch.Tensor([args.lambda_num]), requires_grad=True).cuda().to(self.device)
        self.scalar = nn.Parameter(torch.Tensor([args.lambda_num]), requires_grad=True)

        print('args.lambda_num = ', args.lambda_num)

        self.dummy_rela = torch.ones(max(self.user_triplet_set) * 2 + 1, 1, self.embed_size)
        self.dummy_rela = nn.Parameter(self.dummy_rela, requires_grad=True).to(self.device)
        self.dummy_rela_emb = nn.Embedding(max(self.user_triplet_set) * 2 + 1, self.embed_size * self.embed_size).to(self.device)


        self.record_path_info = {}

    def bulid_model_rl(self):
        self.state_lstm = KGState_LSTM(self.args, history_len=1)

        self.l1 = nn.Linear(2 * self.embed_size, self.hidden_sizes[1])
        self.l2 = nn.Linear(self.hidden_sizes[0], self.hidden_sizes[1])

        self.actor = nn.Linear(self.hidden_sizes[1], self.act_dim)
        self.critic = nn.Linear(self.hidden_sizes[1], 1)

        self.saved_actions = []
        self.rewards = []
        self.entropy = []

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

        new_memories_h_index = {}
        new_memories_r_index = {}
        new_memories_t_index = {}

        for i in range(max(1,self.p_hop)):
            new_memories_h_index[i] = []
            new_memories_r_index[i] = []
            new_memories_t_index[i] = []

            for row in up_date_hop:
                new_memories_h_index[i].append(self.memories_h_index[i][row])
                new_memories_r_index[i].append(self.memories_r_index[i][row])
                new_memories_t_index[i].append(self.memories_t_index[i][row])

            self.memories_h_index[i] = new_memories_h_index[i]
            self.memories_r_index[i] = new_memories_r_index[i]
            self.memories_t_index[i] = new_memories_t_index[i]


    def bulid_model_reasoning(self):

        self.reasoning_step = self.args.reasoning_step
        self.rn_cal_state_prop = []
        self.update_rn_state = []
        self.rh_query = []
        self.v_query = []
        self.o_r_query = []
        self.t_u_query = []

        for i_r in range(self.reasoning_step):
            for i_p in range(self.p_hop):
                self.rh_query.append(nn.Linear(self.embed_size, self.embed_size).cuda())
                self.v_query.append(nn.Linear(self.embed_size, self.embed_size).cuda())
                self.t_u_query.append(nn.Linear(self.embed_size, self.embed_size).cuda())
                self.o_r_query.append(nn.Linear(self.embed_size, self.embed_size).cuda())
                self.update_rn_state.append(nn.Linear(self.embed_size * 2, self.embed_size).cuda())
                self.rn_cal_state_prop.append(nn.Linear(self.embed_size * 1, 1).cuda())

        self.rh_query = nn.ModuleList(self.rh_query)
        self.v_query = nn.ModuleList(self.v_query)
        self.t_u_query = nn.ModuleList(self.t_u_query)
        self.o_r_query = nn.ModuleList(self.o_r_query)
        self.update_rn_state = nn.ModuleList(self.update_rn_state)
        self.rn_cal_state_prop = nn.ModuleList(self.rn_cal_state_prop)

    def forward(self, inputs):
        state, res_user_emb, next_enti_emb, next_action_emb,  act_mask = inputs  # state: [bs, state_dim], act_mask: [bs, act_dim]

        state_tr = state.unsqueeze(1).repeat(1, next_action_emb.shape[1], 1)
        probs_st = state_tr * next_action_emb 

        res_user_emb = res_user_emb.unsqueeze(1).repeat(1, next_enti_emb.shape[1], 1)
        probs_user = res_user_emb * next_enti_emb

        probs_st = probs_st.sum(-1)

        probs_user = probs_user.sum(-1)

        scalar = self.scalar.unsqueeze(1).repeat(probs_user.shape[0],1)

        probs = probs_st + self.scalar * probs_user
        probs = probs.masked_fill(~act_mask, value=torch.tensor(-1e10))
        act_probs = F.softmax(probs, dim=-1)

        x = self.l1(state)
        x = F.dropout(F.elu(x), p=0.4)

        state_values = self.critic(x)  # Tensor of [bs, 1]
        return act_probs, state_values

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

    def rn_query_st_sp_v2(self, hop, state, relation_embed_dual, T_RN):
        cur_index = hop * self.reasoning_step + T_RN

        user_embeddings = self.memories_h[0][:,0]

        # state = th.cat([state.squeeze(), user_embeddings], -1)
        state = th.cat([state.squeeze()], -1)
        relation_embed_dual = th.cat([relation_embed_dual.squeeze()], -1)

        t_expanded = torch.unsqueeze(self.memories_t[hop], dim=3)
        # print('hop = ', hop)
        # print('t_expanded = ', t_expanded.shape)

        Rt = torch.squeeze(torch.matmul(self.memories_r[hop], t_expanded))
        v =  state.unsqueeze(1).repeat(1, self.memories_t[0].shape[1], 1)

        r_v = relation_embed_dual.unsqueeze(1).repeat(1, self.memories_t[0].shape[1], 1, 1)
        r_vh = torch.squeeze(torch.matmul(r_v, t_expanded))
            
        t_u = user_embeddings.unsqueeze(1).repeat(1, self.memories_t[0].shape[1], 1)

        q_Rt = self.rh_query[cur_index](Rt)
        q_v = self.v_query[cur_index](v)
        t_u = self.t_u_query[cur_index](t_u)
        o_r = self.o_r_query[cur_index](r_vh)

        t_state = torch.tanh(q_Rt + q_v + t_u + o_r)
        probs = torch.squeeze(self.rn_cal_state_prop[cur_index](t_state))

        # [batch_size, n_memory]
        probs_normalized = F.softmax(probs, dim=1)

        # [batch_size, n_memory, 1]
        probs_expanded = torch.unsqueeze(probs_normalized, dim=2)


        probs_expanded_to_list = probs_expanded.tolist()


        for prob_index in range(len(probs_expanded_to_list)):
            by_hop_query_tmp = []

            by_hop_query_dict = []

            # print('prob_index = ', prob_index)

            for t_, r_, p_ in zip(self.memories_t_index[hop][prob_index], self.memories_r_index[hop][prob_index], probs_expanded_to_list[prob_index]):
                
            #     print('t_ = ', t_)
            #     print('r_ = ', r_)
            #     print('p_ = ', p_)
            #     # input()

            #     print('t_[0][:2] = ', t_[0][:2])
            #     print(' str(t_[1]) = ',  str(t_[1]))
            #     print(' r_[:2] = ',  r_[:2])
                # input()

                by_hop_query_dict.append([t_, r_,  p_])
                state_trp = ', '.join([t_[0][:2], str(t_[1]), r_[:2]])
                if len(state_trp) < 15: 
                    state_trp += ' ' * (15 - len(state_trp))
                state_trp = [state_trp,  str(round(p_[0], 5))]
                state_trp = ', prb = '.join(state_trp)
                if len(state_trp) < 25: 
                    state_trp += ' ' * (25 - len(state_trp))
                by_hop_query_tmp.append(state_trp)

            by_hop_query_highlight_list = sorted(by_hop_query_dict, key = lambda s: s[2])[::-1][:8]

            self.by_hop_query[hop].append(by_hop_query_tmp)

            self.by_hop_query_record[hop].append(by_hop_query_highlight_list)

        # [batch_size, dim]
        response = (self.memories_t[hop] * probs_expanded).sum(dim=1)

        return response

    def generate_st_emb(self, batch_path, up_date_hop = None):
        if up_date_hop != None:
            self.update_path_info(up_date_hop)
            self.update_path_info_memories(up_date_hop)
        
        self.current_step = {}

        tmp_state = [self._get_state_update(index, path) for index, path in enumerate(batch_path)]
        # input()

        
        all_state = th.cat([ts[3].unsqueeze(0) for ts in tmp_state], 0)

        if len(batch_path[0]) != 1:
            selc_entitiy = th.cat([ts[0].unsqueeze(0) for ts in tmp_state], 0)
            self.update_query_embedding(selc_entitiy)

        state_output, self.prev_state_h, self.prev_state_c = self.state_lstm(all_state, 
                    self.prev_state_h,  self.prev_state_c)

        curr_node_embed = th.cat([ts[0].unsqueeze(0) for ts in tmp_state], 0)
        relation_embed = th.cat([ts[1].unsqueeze(0) for ts in tmp_state], 0)
        relation_embed_dual = th.cat([ts[2].unsqueeze(0) for ts in tmp_state], 0)

        self.by_hop_query = {}
        self.by_hop_query_record = {}

        o_list = []
        for hop in range(self.p_hop):

            state_tmp = relation_embed
            for rn_step in range(self.reasoning_step):
                self.by_hop_query[hop] = []
                self.by_hop_query_record[hop] = []
                query_state = self.rn_query_st_sp_v2(hop, state_tmp, relation_embed_dual, rn_step)

                self.current_step[rn_step] = self.by_hop_query
                self.current_step[str(rn_step) + 'record'] = self.by_hop_query_record

                state_tmp_ = th.cat([query_state, state_tmp], -1)
                state_tmp = self.update_rn_state[hop * self.reasoning_step + rn_step](state_tmp_)
            o_list.append(query_state.unsqueeze(1))

        o_list = torch.cat(o_list, 1)
        res_user_emb = o_list.sum(1)

        state_output = state_output.squeeze()
        res_user_emb = res_user_emb.squeeze()

        return [state_output, res_user_emb]

    def generate_act_emb(self, batch_path, batch_curr_actions):

        self.current_step['user'] = [str(ui) for ui in self.uids]
        b_path = [[','.join([str(uni) for uni in bpa])  for bpa in ba_path] for ba_path in batch_path]        
        b_action = [[' id = '.join([str(uni) for uni in bpa]) for bpa in batch_actions] for batch_actions in batch_curr_actions]

        n_batch_path = []
        list_batch_path = []
        for ba_path in batch_path:
            n_ba_path = []
            list_ba_path = []
            for bpa in ba_path:
                bpa = list(bpa)
                bpa[2] = self.args.data_entity_ind_id[bpa[1]][bpa[2]]
                if bpa[2] in self.args.asin2title:
                    bpa[2] = self.args.asin2title[bpa[2]]
                n_ba_path.append(','.join([str(uni) for uni in bpa]))
                list_ba_path.append(bpa)
            n_batch_path.append(n_ba_path)
            list_batch_path.append(list_ba_path)

        n_batch_curr_actions = []
        list_batch_curr_actions = []
        for index, actions_sets in enumerate(zip(batch_path, batch_curr_actions)):
            curr_path, curr_actions = actions_sets[0], actions_sets[1]
            last_relation, curr_node_type, curr_node_id = curr_path[-1]
            n_curr_actions = []
            for action_set in curr_actions:
                action_set = list(action_set)
                if action_set[0] == SELF_LOOP: next_node_type = curr_node_type
                else: next_node_type = self._get_next_node_type(curr_node_type, action_set[0], action_set[1])
                action_set[1] = self.args.data_entity_ind_id[next_node_type][action_set[1]]
                if action_set[1] in self.args.asin2title:
                    action_set[1] = self.args.asin2title[action_set[1]]
                n_curr_actions.append(' id = '.join([str(uni) for uni in action_set]))
            n_batch_curr_actions.append(n_curr_actions)

        self.current_step["path"] = ['path = ' + ', next = '.join(cn_b_path) for cn_b_path in n_batch_path]
        self.current_step["path_list"] = list_batch_path
        self.current_step["actions"] = n_batch_curr_actions

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

    def _get_state_update(self, index, path):
        """Return state of numpy vector: [user_embed, curr_node_embed, last_node_embed, last_relation]."""
        if len(path) == 1:
            if self.user_o == True:
                user_embed = self.global_user[index,:].unsqueeze(0)
            else:
                # print('user embedd only kg')
                user_embed = self.kg_emb.lookup_emb(USER, type_index = 
                        torch.LongTensor([path[0][-1]]).to(self.device))[0].unsqueeze(0)
            curr_node_embed = user_embed
            last_relation_embed = self.dummy_rela[path[0][-1], :, :]
            relation_embed_dual = self.dummy_rela_emb(torch.LongTensor([path[0][-1]]).to(self.device))
            relation_embed_dual = relation_embed_dual.view(self.embed_size, self.embed_size)

            st_emb = self.action_encoder(last_relation_embed, user_embed)
        else:
            last_relation, curr_node_type, curr_node_id = path[-1]
            # print('last_relation, curr_node_type, curr_node_id  = ', last_relation, curr_node_type, curr_node_id )
            curr_node_embed = self.kg_emb.lookup_emb(curr_node_type, 
                    type_index = torch.LongTensor([curr_node_id]).to(self.device))[0].unsqueeze(0)
            last_relation_embed = self.kg_emb.lookup_rela_emb(last_relation)[0].unsqueeze(0)

            relation_embed_dual = self.relation_emb(torch.LongTensor([self.rela_2_index[last_relation]]).to(self.device))
            relation_embed_dual = relation_embed_dual.view(self.embed_size, self.embed_size)

            st_emb = self.action_encoder(last_relation_embed, curr_node_embed)

        return [curr_node_embed, last_relation_embed.squeeze(), relation_embed_dual.squeeze(), st_emb]


    def _record_case_study(self):
        eva_file = open(self.cast_st_save, "a")
        for ind in range(len(self.current_step['user'])):

            rd_uid = self.current_step["user"][ind]
            if rd_uid not in self.record_path_info:
                self.record_path_info[rd_uid] = {}

            re_path = self.current_step["path"][ind]
            if re_path not in self.record_path_info[rd_uid]:
                self.record_path_info[rd_uid][re_path] = {}

            self.record_path_info[rd_uid][re_path]['path_list'] =  self.current_step["path_list"][ind]

            self.record_path_info[rd_uid][re_path]['action'] = self.current_step["actions"][ind]

            for hop in range(self.p_hop):
                tmp_list = []
                for rn_step in range(self.reasoning_step):  
                    tmp_list.append(self.current_step[rn_step][hop][ind])
                re_record_string = ''
                for state_s in zip(*tmp_list):
                    re_record_string += ' n_hop = '.join(list(state_s))

                self.record_path_info[rd_uid][re_path]['hop_' + str(hop)] = self.current_step[str(rn_step) + 'record'][hop][ind]

            self.record_path_info[rd_uid][re_path]['next_acts'] =  self.current_step['next_acts'][ind]

            eva_file.write("*" * 50)
            eva_file.write('\n')
            eva_file.write('user = ' + self.current_step['user'][ind])
            eva_file.write('\n')
            eva_file.write(self.current_step['path'][ind])
            eva_file.write('\n')
            eva_file.write("*" * 50)
            eva_file.write('\n')
            eva_file.write("querying result")
            eva_file.write('\n')
            for hop in range(self.p_hop):
                eva_file.write("*" * 50)
                eva_file.write('\n')
                eva_file.write("hop = " + str(hop))
                eva_file.write('\n')
                tmp_list = []
                for rn_step in range(self.reasoning_step):  
                    tmp_list.append(self.current_step[rn_step][hop][ind])
                for state_s in zip(*tmp_list):                    # print('state_s = ', list(state_s))
                    eva_file.write(', = '.join(list(state_s)))
                    eva_file.write('\n')
            eva_file.write("*" * 50)
            eva_file.write('\n')
            for ac_pro in self.current_step['actions_pro'][ind]:
                eva_file.write(ac_pro)
                eva_file.write('\n')
            eva_file.write("next_action = " + ', '.join([str(k) for k in self.current_step['next_acts'][ind]]))
            eva_file.write('\n')
            eva_file.write("*" * 50)
        eva_file.close()

    def save_path_record_dict(self, record_path):
        with open(record_path, 'wb') as handle:
            pickle.dump(self.record_path_info, handle, protocol=pickle.HIGHEST_PROTOCOL)