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
from model.Backbone import Backbone
from utils import *


class UCPR(Backbone):
    def __init__(self, args, user_triplet_set, rela_2_index, act_dim, gamma=0.99, hidden_sizes=[512, 256]):
        super().__init__(args, user_triplet_set, rela_2_index, act_dim, gamma, hidden_sizes)

    def rn_query_st(self, state, relation_embed_dual, rn_step):

        user_embeddings = self.memories_h[0][:,0]

        state = th.cat([state.squeeze()], -1)
        relation_embed_dual = th.cat([relation_embed_dual.squeeze()], -1)

        o_list = []
        for hop in range(self.p_hop):
            
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

            probs = torch.squeeze(self.rn_cal_state_prop(t_state))

            probs_normalized = F.softmax(probs, dim=1)

            probs_expanded = torch.unsqueeze(probs_normalized, dim=2)

            # [batch_size, dim]
            o = (self.memories_t[hop] * probs_expanded).sum(dim=1).unsqueeze(1)

            o_list.append(o)

        o_list = torch.cat(o_list, 1)

        user_o = o_list.sum(1)

        return user_o

    def generate_st_emb(self, batch_path, up_date_hop = None):
        if up_date_hop != None:
            self.update_path_info(up_date_hop)
            self.update_path_info_memories(up_date_hop)

        tmp_state = [self.get_state_update(index, path) for index, path in enumerate(batch_path)]

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
            if rn_step < self.reasoning_step - 1: 
                state_tmp_ = th.cat([query_state, state_tmp], -1)
                state_tmp = self.update_rn_state[rn_step](state_tmp_)

        res_user_emb = query_state

        state_output = state_output.squeeze()
        res_user_emb = res_user_emb.squeeze()

        return [state_output, res_user_emb]

    def get_state_update(self, index, path):
        """Return state of numpy vector: [user_embed, curr_node_embed, last_node_embed, last_relation]."""
        if len(path) == 1:
            user_embed = self.kg_emb.lookup_emb(USER, type_index = torch.LongTensor([path[0][-1]]).to(self.device))[0].unsqueeze(0)
            curr_node_embed = user_embed
            last_relation_embed = self.dummy_rela[path[0][-1], :, :]
            relation_embed_dual = self.dummy_rela_emb(torch.LongTensor([path[0][-1]]).to(self.device))
            relation_embed_dual = relation_embed_dual.view(self.embed_size, self.embed_size)

            st_emb = self.action_encoder(last_relation_embed, user_embed)
        else:
            last_relation, curr_node_type, curr_node_id = path[-1]
            curr_node_embed = self.kg_emb.lookup_emb(curr_node_type, 
                    type_index = torch.LongTensor([curr_node_id]).to(self.device))[0].unsqueeze(0)
            last_relation_embed = self.kg_emb.lookup_rela_emb(last_relation)[0].unsqueeze(0)

            relation_embed_dual = self.relation_emb(torch.LongTensor([self.rela_2_index[last_relation]]).to(self.device))
            relation_embed_dual = relation_embed_dual.view(self.embed_size, self.embed_size)

            st_emb = self.action_encoder(last_relation_embed, curr_node_embed)

        return [curr_node_embed, last_relation_embed.squeeze(), relation_embed_dual.squeeze(), st_emb]

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

    def update(self, optimizer, env_model, device, ent_weight, step):
        if len(self.rewards) <= 0:
            del self.rewards[:]
            del self.saved_actions[:]
            del self.entropy[:]
            return 0.0, 0.0, 0.0

        batch_rewards = np.vstack(self.rewards).T  # numpy array of [bs, #steps]
        batch_rewards = torch.FloatTensor(batch_rewards).to(device)
        num_steps = batch_rewards.shape[1]
        for i in range(1, num_steps):
            batch_rewards[:, num_steps - i - 1] += self.gamma * batch_rewards[:, num_steps - i]

        actor_loss = 0
        critic_loss = 0
        entropy_loss = 0
        for i in range(0, num_steps):
            log_prob, value = self.saved_actions[i]  # log_prob: Tensor of [bs, ], value: Tensor of [bs, 1]
            advantage = batch_rewards[:, i] - value.squeeze(1)  # Tensor of [bs, ]
            actor_loss += -log_prob * advantage.detach()  # Tensor of [bs, ]
            critic_loss += advantage.pow(2)  # Tensor of [bs, ]
            entropy_loss += -self.entropy[i]  # Tensor of [bs, ]

        l2_loss = 0
        l2_reg = 0
        for name, param in self.named_parameters():
            if 'weight' in name:
                l2_reg += torch.norm(param)
        l2_loss += self.l2_weight * l2_reg

        actor_loss = actor_loss.mean()
        critic_loss = critic_loss.mean()
        entropy_loss = entropy_loss.mean()
        loss = actor_loss + critic_loss + ent_weight * entropy_loss + l2_loss
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
        del self.rewards[:]
        del self.saved_actions[:]
        del self.entropy[:]

        return loss.item(), actor_loss.item(), critic_loss.item(), entropy_loss.item()
