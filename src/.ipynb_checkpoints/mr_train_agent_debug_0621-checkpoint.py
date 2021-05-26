from __future__ import absolute_import, division, print_function

import sys
import os
import argparse
from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Categorical
from utils import *
from model_refine.get_model import *
from parser import parse_args

import time

logger = None

SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

class ACDataLoader(object):
    def __init__(self, uids, batch_size):
        self.uids = np.array(uids)
        self.num_users = len(uids)
        self.batch_size = batch_size
        self.reset()

    def reset(self):
        self._rand_perm = np.random.permutation(self.num_users)
        self._start_idx = 0
        self._has_next = True

    def has_next(self):
        return self._has_next

    def get_batch(self):
        if not self._has_next:
            return None
        # Multiple users per batch
        end_idx = min(self._start_idx + self.batch_size, self.num_users)
        batch_idx = self._rand_perm[self._start_idx:end_idx]
        batch_uids = self.uids[batch_idx]
        self._has_next = self._has_next and end_idx < self.num_users
        self._start_idx = end_idx
        return batch_uids.tolist()

def train(args):
    global logger
    args.logger =  get_logger(args.log_dir + '/train_log_rd.txt')

    env = KGEnvironment(args, args.dataset, args.max_acts, max_path_len=args.max_path_len, state_history=args.state_history)
    dataloader = ACDataLoader(env.output_valid_user(), args.batch_size)

    if args.load_pretrain_model == True:
        print('args.load_pretrain_model == True')
        policy_file = args.save_model_dir + '/policy_model_epoch_{}.ckpt'.format(args.pretrained_st_epoch)
        pretrain_sd = torch.load(policy_file)
        model = Memory_Model(args, env.user_triplet_set, env.rela_2_index, 
                                env.act_dim, gamma=args.gamma, hidden_sizes=args.hidden).to(args.device)
        model_sd = model.state_dict()
        model_sd.update(pretrain_sd)
        model.load_state_dict(model_sd)
        start_epoch = args.pretrained_st_epoch + 1
        logger = get_logger(args.log_dir + '/train_log_pretrain.txt')
        # args_tmp = {k:v for k, v in args.items() if k != 'kg_user_filter'}

        kg_user_filter_tmp = args.kg_user_filter
        kgquery_frequency_dict_tmp = args.kgquery_frequency_dict
        query_enti_frequency_tmp = args.query_enti_frequency

        args.kg_user_filter = ''
        args.kgquery_frequency_dict = ''
        args.query_enti_frequency = ''
        logger.info(args)
        args.kg_user_filter = kg_user_filter_tmp
        args.kgquery_frequency_dict = kgquery_frequency_dict_tmp
        args.query_enti_frequency = query_enti_frequency_tmp
        del kg_user_filter_tmp, kgquery_frequency_dict_tmp, query_enti_frequency_tmp
    else:
        model = Memory_Model(args, env.user_triplet_set, env.rela_2_index, env.act_dim, gamma=args.gamma, hidden_sizes=args.hidden).to(args.device)
        start_epoch = 1
        logger = get_logger(args.log_dir + '/train_log.txt')

        kg_user_filter_tmp = args.kg_user_filter
        kgquery_frequency_dict_tmp = args.kgquery_frequency_dict
        query_enti_frequency_tmp = args.query_enti_frequency

        args.kg_user_filter = ''
        args.kgquery_frequency_dict = ''
        args.query_enti_frequency = ''
        logger.info(args)
        args.kg_user_filter = kg_user_filter_tmp
        args.kgquery_frequency_dict = kgquery_frequency_dict_tmp
        args.query_enti_frequency = query_enti_frequency_tmp
        del kg_user_filter_tmp, kgquery_frequency_dict_tmp, query_enti_frequency_tmp

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    logger.info('Parameters:' + str([i[0] for i in model.named_parameters()]))
    total_losses, total_plosses, total_vlosses, total_entropy, total_rewards = [], [], [], [], []
    step = 0
    model.train()
    for epoch in range(start_epoch, args.epochs + 1):
        ### Start epoch ###
        if epoch % 2 == 0:
            env.reset_path(epoch)
        dataloader.reset()
        while dataloader.has_next():
            batch_uids = dataloader.get_batch()
            ### Start batch episodes ###
            env.reset(epoch, batch_uids, training = True)  # numpy array of [bs, state_dim]
            model.user_triplet_set = env.user_triplet_set
            model.reset(batch_uids)

            while not env._done:
                batch_act_mask = env.batch_action_mask(dropout=args.act_dropout)  # numpy array of size [bs, act_dim]
                batch_emb_state = model.generate_st_emb(env._batch_path)
                batch_next_action_emb = model.generate_act_emb(env._batch_path, env._batch_curr_actions)
                batch_act_idx = model.select_action(batch_emb_state, batch_next_action_emb, batch_act_mask, args.device)  # int
                batch_state, batch_reward = env.batch_step(batch_act_idx)
                model.rewards.append(batch_reward)
            ### End of episodes ###

            lr = args.lr
             # * max(1e-4, 1.0 - float(step) / (args.epochs * 5000 / args.batch_size))
            for pg in optimizer.param_groups:
                pg['lr'] = lr

            # Update policy
            total_rewards.append(np.sum(model.rewards))
            loss, ploss, vloss, eloss = model.update(optimizer, env, args.device, args.ent_weight, step)
            total_losses.append(loss)
            total_plosses.append(ploss)
            total_vlosses.append(vloss)
            total_entropy.append(eloss)
            step += 1

            # Report performance
            if step > 0 and step % 100 == 0:
                avg_reward = np.mean(total_rewards) / args.batch_size
                avg_loss = np.mean(total_losses)
                avg_ploss = np.mean(total_plosses)
                avg_vloss = np.mean(total_vlosses)
                avg_entropy = np.mean(total_entropy)
                total_losses, total_plosses, total_vlosses, total_entropy, total_rewards = [], [], [], [], []
                logger.info(
                        'epoch/step={:d}/{:d}'.format(epoch, step) +
                        ' | loss={:.5f}'.format(avg_loss) +
                        ' | ploss={:.5f}'.format(avg_ploss) +
                        ' | vloss={:.5f}'.format(avg_vloss) +
                        ' | entropy={:.5f}'.format(avg_entropy) +
                        ' | reward={:.5f}'.format(avg_reward) +
                        ' | lr={:.5f}'.format(lr))
        ### END of epoch ###

        policy_file = '{}/policy_model_epoch_{}.ckpt'.format(args.save_model_dir, epoch)
        logger.info("Save model to " + policy_file)
        torch.save(model.state_dict(), policy_file)

    cur_tim = time.strftime("%Y%m%d-%H%M%S")
    logger.info("current time = " + str(cur_tim))

def main():

    args = parse_args()

    args.training = 1
    args.training = (args.training == 1)

    args.att_core = 0
    args.item_core = 0
    args.user_core = 6000
    args.user_query_threshold = 800
    args.query_threshold = 15
    args.query_threshold_maximum = 500
    args.max_acts = 50
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    args.device = torch.device('cuda:0') if torch.cuda.is_available() else 'cpu'

    print('torch.cuda.current_device() = ', torch.cuda.current_device())
    
    if args.h0_embbed == True:
        log_dir_fodder = f"no_shu_qy_uam_{args.user_query_threshold}_{args.query_threshold}_{args.query_threshold_maximum}_lr_{args.lr}_bs_{args.batch_size}_sb_{args.sub_batch_size}_ma_{args.max_acts}_p0{args.p_hop}_{args.name}_g_aiu_{args.att_core}_{args.item_core}_{args.user_core}_qy_uam_{args.user_query_threshold}_{args.query_threshold}_{args.query_threshold_maximum}"
    else:
        log_dir_fodder = f'no_shu_qy_uam_{args.user_query_threshold}_{args.query_threshold}_{args.query_threshold_maximum}_lr_{args.lr}_bs_{args.batch_size}_sb_{args.sub_batch_size}_ma_{args.max_acts}_p1{args.p_hop}_{args.name}_g_aiu_{args.att_core}_{args.item_core}_{args.user_core}_qy_uam_{args.user_query_threshold}_{args.query_threshold}_{args.query_threshold_maximum}'
    
    if args.reward_hybrid == True:
        log_dir_fodder = 'rh_' + log_dir_fodder
    
    args.log_dir = '{}/{}/{}'.format(EVALUATION_debug_3[args.dataset], args.use_men, log_dir_fodder)
    if not os.path.isdir(args.log_dir):
        os.makedirs(args.log_dir)

    args.save_model_dir = '{}/{}/{}'.format(SAVE_MODEL_DIR_debug[args.dataset], args.use_men, log_dir_fodder)
    if not os.path.isdir(args.save_model_dir):
        os.makedirs(args.save_model_dir)

    try:
        args.gradient_plot_save = os.path.join(args.gradient_plot,  args.use_men, log_dir_fodder)
        if not os.path.isdir(args.gradient_plot_save): 
            os.makedirs(args.gradient_plot_save)
    except:
        pass

    set_random_seed(args.seed)
    train(args)


if __name__ == '__main__':
    main()

