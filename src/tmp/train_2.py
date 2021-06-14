from __future__ import absolute_import, division, print_function

import sys
sys.path.insert(0,'/home/johnnyjana730/github/UCPR/')
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
from model.get_model.get_model import *
from parser import parse_args
from para_setting import parameter_path, parameter_path_th
from data_loader import ACDataLoader
import time
import json

SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])


# print('args.load_pretrain_model == True')
# policy_file = args.save_model_dir_tmp + '/policy_model_epoch_{}.ckpt'.format(args.pretrained_st_epoch)
# pretrain_sd = torch.load(policy_file)
# model = Memory_Model(args, env.user_triplet_set, env.rela_2_index, 
#                         env.act_dim, gamma=args.gamma, hidden_sizes=args.hidden).to(args.device)
# model_sd = model.state_dict()
# pretrain_sd = {k: v for k, v in pretrain_sd.items() if k in model_sd}
# para_meter = [k.split('.')[0] for k, v in pretrain_sd.items()]

# model_sd.update(pretrain_sd)
# model.load_state_dict(model_sd)

def pretrain_set(args, env):
    if args.load_pretrain_model == True:    
        logger = get_logger(args.log_dir + '/train_log_pretrain.txt')
        args.logger = logger

        print(args.pretrained_dir + '/' + args.sort_by + '_pretrained_md_json_' + args.topk_string + '.txt')
        with open(args.pretrained_dir + '/' + args.sort_by + '_pretrained_md_json_' + args.topk_string + '.txt') as json_file:
            best_model_json = json.load(json_file)


        policy_file = best_model_json['pretrained_file']
        pretrain_sd = torch.load(policy_file)
        model = Memory_Model(args, env.user_triplet_set, env.rela_2_index, env.act_dim, gamma=args.gamma, hidden_sizes=args.hidden).to(args.device)
        model_sd = model.state_dict()

        pretrain_sd = {k: v for k, v in pretrain_sd.items() if k in model_sd}
        para_meter = [k.split('.')[0] for k, v in pretrain_sd.items()]

        model_sd.update(pretrain_sd)
        model.load_state_dict(model_sd)


        for name, child in model.named_children():
            print('name = ', name)
            for param in child.parameters():
                param.requires_grad = True

        if args.dataset in [BEAUTY_CORE, CELL_CORE, CLOTH_CORE]:         
            for name, child in model.named_children():
                print('name = ', name)
                if 'kg' in name:
                    print('name = ', name)
                    for param in child.parameters():
                        param.requires_grad = False
        elif args.dataset in [MOVIE_CORE, AZ_BOOK_CORE]:
            for name, child in model.named_children():
                print('name = ', name)
                if name in para_meter and 'kg' not in name and 'actor' not in name and 'critic' not in name:
                    print('name = ', name)
                    for param in child.parameters():
                        param.requires_grad = False

        grad_string = ''
        for name, child in model.named_children():
            print('name = ', name)
            for param in child.parameters():
                print(param.requires_grad)
            grad_string += ' name = ' + name  + ' ' + str(param.requires_grad)

        logger.info(grad_string)
        # start_epoch = args.pretrained_st_epoch + 1

    else:
        logger = get_logger(args.log_dir + '/train_log.txt')
        args.logger = logger

        model = Memory_Model(args, env.user_triplet_set, env.rela_2_index, env.act_dim, gamma=args.gamma, hidden_sizes=args.hidden).to(args.device)

    core_user_list = args.core_user_list
    kg_fre_dict = args.kg_fre_dict

    args.core_user_list = ''
    args.kg_fre_dict = ''
    logger.info(args)
    args.core_user_list = core_user_list
    args.kg_fre_dict = kg_fre_dict
    del core_user_list, kg_fre_dict

    return model, logger

def train(args):

    env = KG_Env(args, args.dataset, args.max_acts, max_path_len=args.max_path_len, state_history=args.state_history)
    print('env.output_valid_user() = ', len(env.output_valid_user()))
    print('args.batch_size = ', args.batch_size)
    # input()
    dataloader = ACDataLoader(env.output_valid_user(), args.batch_size)

    model, logger = pretrain_set(args, env)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    logger.info('Parameters:' + str([i[0] for i in model.named_parameters()]))
    step, total_losses, total_plosses, total_vlosses, total_entropy, total_rewards = 0, [], [], [], [], []
    model.train()
    for epoch in range(0, args.epochs + 1):

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

            for pg in optimizer.param_groups:
                lr = pg['lr']

            # Update policy
            total_rewards.append(np.sum(model.rewards))
            loss, ploss, vloss, eloss = model.update(optimizer, env, args.device, args.ent_weight, step)
            total_losses.append(loss)
            total_plosses.append(ploss)
            total_vlosses.append(vloss)
            total_entropy.append(eloss)
            step += 1

            # print('step = ', step, 'batch_reward = ', batch_reward, end='\r')

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
                        ' | reward={:.5f}'.format(avg_reward))
        ### END of epoch ###

        if epoch % 5 == 0:
            policy_file = '{}/policy_model_epoch_{}.ckpt'.format(args.save_model_dir, epoch)
            logger.info("Save model to " + policy_file)
            torch.save(model.state_dict(), policy_file)

    cur_tim = time.strftime("%Y%m%d-%H%M%S")
    logger.info("current time = " + str(cur_tim))

if __name__ == '__main__':
    args = parse_args()

    args.training = 1
    args.training = (args.training == 1)
        
    if args.envir == 'p1': 
        para_env = parameter_path_th
        KG_Env = KGEnvironment

    elif args.envir == 'p2':
        para_env = parameter_path
        KG_Env = KGEnvironment

    para_env(args)
    train(args)
