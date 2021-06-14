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
from model.lstm_base.model_kg import KG_KGE, RW_KGE
from model.lstm_base.model_kg_pre import KG_KGE_pretrained, RW_KGE_pretrained
from model.lstm_base.backbone_lstm import EncoderRNN_batch, KGState_LSTM
from utils import *

SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])


class AC_lstm_mf_dummy(nn.Module):
    def __init__(self, args, user_triplet_set, rela_2_index, act_dim, gamma=0.99, hidden_sizes=[512, 256]):
        super(AC_lstm_mf_dummy, self).__init__()
        self.args = args        
        self.act_dim = act_dim
        self.device = args.device
        self.gamma = gamma
        self.p_hop = args.p_hop
        self.hidden_sizes = hidden_sizes
        self.n_memory = args.n_memory
        self.rela_2_index = rela_2_index

        self.l2_weight = args.l2_weight
        self.embed_size = args.embed_size
 
        self.user_triplet_set = user_triplet_set

        if self.args.envir == 'p1':
            self._get_next_node_type = self._get_next_node_type_meta
            if self.args.KGE_pretrained == True: self.kg_emb = RW_KGE_pretrained(args)
            else:  self.kg_emb = RW_KGE(args)

        elif self.args.envir == 'p2':
            self._get_next_node_type = self._get_next_node_type_graph
            if self.args.KGE_pretrained == True: 
                # if  self.args.dataset == MOVIE_CORE:
                #     self.kg_emb = KnowledgeEmbedding_memory_graph(args)
                # else:
                    # self.kg_emb = KG_KGE_pretrained(args)
                self.kg_emb = KG_KGE_pretrained(args)
            else:  self.kg_emb = KG_KGE(args)

            dataset = load_dataset(args.dataset)
            self.et_idx2ty = dataset.et_idx2ty
            self.entity_list = dataset.entity_list
            self.rela_list = dataset.rela_list

        self.bulid_model_rl()

        self.dummy_rela = torch.ones(max(self.user_triplet_set) * 2+ 1, 1, self.embed_size)
        self.dummy_rela = nn.Parameter(self.dummy_rela, requires_grad=True).to(self.device)

    def _get_next_node_type_meta(self, curr_node_type, next_relation, next_entity_id):
        return KG_RELATION[curr_node_type][next_relation]

    def _get_next_node_type_graph(self, curr_node_type, next_relation, next_entity_id):
        return self.et_idx2ty[next_entity_id]


    def bulid_model_rl(self):
        self.state_lstm = KGState_LSTM(self.args, history_len=1)

        self.transfor_state = nn.Linear(2 * self.embed_size, 2 * self.embed_size)
        self.state_tr_query = nn.Linear(self.embed_size * 3, self.embed_size)

        self.l1 = nn.Linear(2 * self.embed_size, self.hidden_sizes[1])
        self.l2 = nn.Linear(self.hidden_sizes[0], self.hidden_sizes[1])
        self.actor = nn.Linear(self.hidden_sizes[1], self.act_dim)
        self.critic = nn.Linear(self.hidden_sizes[1], 1)

        self.saved_actions = []
        self.rewards = []
        self.entropy = []

    def forward(self, inputs):
        state, batch_next_action_emb,  act_mask = inputs  # state: [bs, state_dim], act_mask: [bs, act_dim]
        state = state.squeeze()

        state_tr = state.unsqueeze(1).repeat(1, batch_next_action_emb.shape[1], 1)
        probs = state_tr * batch_next_action_emb

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
        probs, value = self((batch_state, batch_next_action_emb, act_mask))  # act_probs: [bs, act_dim], state_value: [bs, 1]

        m = Categorical(probs)
        acts = m.sample()  # Tensor of [bs, ], requires_grad=False

        # [CAVEAT] If sampled action is out of action_space, choose the first action in action_space.
        valid_idx = act_mask.gather(1, acts.view(-1, 1)).view(-1)
        acts[valid_idx == 0] = 0

        self.saved_actions.append(SavedAction(m.log_prob(acts), value))
        self.entropy.append(m.entropy())

        return acts.cpu().numpy().tolist()

    def reset(self, uids=None):
        self.uids = [uid for uid in uids]

        self.prev_state_h, self.prev_state_c = self.state_lstm.set_up_hidden_state(len(self.uids))
        # print(' self.uids  = ', len(self.uids))

    def update_path_info(self, up_date_hop):

        new_uids = []

        for row in up_date_hop:
            new_uids.append(self.uids[row])

        self.uids = new_uids

        new_prev_state_h = []
        new_prev_state_c = []

        for row in up_date_hop:
            new_prev_state_h.append(self.prev_state_h[:,row,:].unsqueeze(1))
            new_prev_state_c.append(self.prev_state_c[:,row,:].unsqueeze(1))

        self.prev_state_h = th.cat(new_prev_state_h, 1).to(self.device)
        self.prev_state_c = th.cat(new_prev_state_c, 1).to(self.device)

    def generate_st_emb(self, batch_path, up_date_hop = None):
        if up_date_hop != None:
            self.update_path_info(up_date_hop)
        all_state = th.cat([self._get_state_update(index, path).unsqueeze(0)
            for index, path in enumerate(batch_path)], 0)

        # print('all_state = ', all_state.shape)
        # print('prev_state_h = ', self.prev_state_h.shape)
        # print('prev_state_c = ', self.prev_state_c.shape)
        # input()
        # print('all_state = ', all_state[0,:])
        # print('self.prev_state_h = ', self.prev_state_h[:,0,:])
        # print('self.prev_state_c = ', self.prev_state_c[:,0,:])
        # input()

        state_output, self.prev_state_h, self.prev_state_c = self.state_lstm(all_state, self.prev_state_h,  self.prev_state_c)
        return state_output

    def action_encoder(self, relation_emb, entitiy_emb):
        action_embedding = th.cat([relation_emb, entitiy_emb], -1)
        return action_embedding

    def _get_state_update(self, index, path):
        """Return state of numpy vector: [user_embed, curr_node_embed, last_node_embed, last_relation]."""

        if len(path) == 1:
            user_embed = self.kg_emb.lookup_emb(USER, type_index = 
                        torch.LongTensor([path[0][-1]]).to(self.device))[0].unsqueeze(0)
            curr_node_embed = user_embed
            dummy_rela = self.dummy_rela[path[0][-1], :, :]
            st_emb = self.action_encoder(dummy_rela, user_embed)

        else:
            last_relation, curr_node_type, curr_node_id = path[-1]
            curr_node_embed = self.kg_emb.lookup_emb(curr_node_type, 
                    type_index = torch.LongTensor([curr_node_id]).to(self.device))[0].unsqueeze(0)
            last_relation_embed = self.kg_emb.lookup_rela_emb(last_relation)[0].unsqueeze(0)
            st_emb = self.action_encoder(last_relation_embed, curr_node_embed)
        return st_emb

    def generate_act_emb(self, batch_path, batch_curr_actions):
        return th.cat([self._get_actions(index, actions_sets[0], 
            actions_sets[1]).unsqueeze(0) for index, actions_sets in enumerate(zip(batch_path, batch_curr_actions))], 0)
    
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

            # print('rela_emb = ', rela_emb.shape)

        pad_emb = self.kg_emb.lookup_rela_emb(PADDING)
        for _ in range(self.act_dim - len(entities_embs)):
            entities_embs.append(pad_emb)
            relation_embs.append(pad_emb)

        #     print('pad_emb = ', pad_emb.shape)
        # input()

        enti_emb = th.cat(entities_embs, 0)
        rela_emb = th.cat(relation_embs, 0)

        next_action_state = th.cat([enti_emb, rela_emb], -1)
        
        return next_action_state

    def _get_next_node_type(self, curr_node_type, next_relation, next_entity_id):
        pass

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


        l2_reg = 0
        for name, param in self.named_parameters():
            if 'weight' in name:
                l2_reg += torch.norm(param)
        l2_loss = self.l2_weight * l2_reg

        actor_loss = 0
        critic_loss = 0
        entropy_loss = 0
        for i in range(0, num_steps):
            log_prob, value = self.saved_actions[i]  # log_prob: Tensor of [bs, ], value: Tensor of [bs, 1]
            advantage = batch_rewards[:, i]
            actor_loss += -log_prob * advantage.detach()  # Tensor of [bs, ]
            critic_loss += advantage.pow(2)  # Tensor of [bs, ]
            entropy_loss += -self.entropy[i]  # Tensor of [bs, ]
        actor_loss = actor_loss.mean()
        critic_loss = critic_loss.mean()
        entropy_loss = entropy_loss.mean()
        loss = actor_loss + critic_loss + ent_weight * entropy_loss + l2_loss
        optimizer.zero_grad()
        loss.backward()

        if self.args.grad_check == True and step % 50 == 0:
            plot_grad_flow_v2(self.named_parameters(), self.args.log_dir, step)
            print('grad_cherck')

        optimizer.step()
        del self.rewards[:]
        del self.saved_actions[:]
        del self.entropy[:]

        return loss.item(), actor_loss.item(), critic_loss.item(), entropy_loss.item()


