from __future__ import absolute_import, division, print_function

import sys
import os
import argparse
from math import log
from datetime import datetime
from tqdm import tqdm
from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Categorical
import threading
from functools import reduce
import time

import gc
# from kg_env_memory_graph import BatchKGEnvironment
# from train_agent import ActorCritic
from utils import *
from model_refine.get_model import *
from parser import parse_args
from parameter_setting import parameter_env_th

def evaluate(topk_matches, test_user_products, no_skip_user):
    """Compute metrics for predicted recommendations.
    Args:
        topk_matches: a list or dict of product ids in ascending order.
    """
    cum_k = 0
    invalid_users = []
    # Compute metrics
    precisions, recalls, ndcgs, hits = [], [], [], []
    test_user_idxs = list(test_user_products.keys())
    for uid in test_user_idxs:
        if uid not in topk_matches:
            invalid_users.append(uid)
            continue

        pred_list, rel_set = topk_matches[uid][::-1], test_user_products[uid]
        if uid not in no_skip_user:
            continue

        if len(pred_list) == 0:
            cum_k += 1
            ndcgs.append(0)
            recalls.append(0)
            precisions.append(0)
            hits.append(0)
            continue

        dcg = 0.0
        hit_num = 0.0
        for i in range(len(pred_list)):
            if pred_list[i] in rel_set:
                dcg += 1. / (log(i + 2) / log(2))
                hit_num += 1
        # idcg
        idcg = 0.0
        for i in range(min(len(rel_set), len(pred_list))):
            idcg += 1. / (log(i + 2) / log(2))
        ndcg = dcg / idcg

        recall = hit_num / len(rel_set)

        precision = hit_num / len(pred_list)

        hit = 1.0 if hit_num > 0.0 else 0.0

        
        # if len(pred_list) == 0:
        #     ndcg = 0
        # else:
            # ndcg = dcg / idcg


        # if len(pred_list) == 0:
        #     precision = 0
        # else:
        #     precision = hit_num / len(pred_list)

        ndcgs.append(ndcg)
        recalls.append(recall)
        precisions.append(precision)
        hits.append(hit)

    avg_precision = np.mean(precisions) * 100
    avg_recall = np.mean(recalls) * 100
    avg_ndcg = np.mean(ndcgs) * 100
    avg_hit = np.mean(hits) * 100
    print('NDCG={:.3f} |  Recall={:.3f} | HR={:.3f} | Precision={:.3f} | Invalid users={}'.format(
            avg_ndcg, avg_recall, avg_hit, avg_precision, len(invalid_users)))
    print('cum_k == 0 ',  cum_k)
    return avg_precision, avg_recall, avg_ndcg, avg_hit, invalid_users, cum_k

def batch_beam_search(args, env, model, uids, device, topk=[25, 5, 1], topk_list= [1,25,125,125]):
    def _batch_acts_to_masks(batch_acts):
        batch_masks = []
        for acts in batch_acts:
            num_acts = len(acts)
            act_mask = np.zeros(model.act_dim, dtype=np.uint8)
            act_mask[:num_acts] = 1
            batch_masks.append(act_mask)
        return np.vstack(batch_masks)

    state_pool = env.reset(args.epochs,uids)  # numpy of [bs, dim]
    
    model.reset(uids)

    path_pool = env._batch_path  # list of list, size=bs
    probs_pool = [[] for _ in uids]
    index_ori_list = [_ for _ in range(len(uids))]
    idx_list = [i for i in range(len(uids))]
    # print('idx_list = ', idx_list)

    model.eval()
    for hop in range(3):

        acts_pool = env._batch_get_actions(path_pool, False)  # list of list, size=bs
        actmask_pool = _batch_acts_to_masks(acts_pool)  # numpy of [bs, dim]

        if args.test_lstm_up == True:
            state_tensor = model.generate_st_emb(path_pool, up_date_hop = idx_list) 
        else:
            state_tensor = model.generate_st_emb(path_pool, test_hop = index_ori_list)

        batch_next_action_emb = model.generate_act_emb(path_pool, acts_pool)
        
        # print('batch_next_action_emb = ', batch_next_action_emb)

        actmask_tensor = torch.BoolTensor(actmask_pool).to(device)

      
        try:
            next_enti_emb, next_action_emb = batch_next_action_emb[0], batch_next_action_emb[1]
            probs, _ = model((state_tensor[0],state_tensor[1], next_enti_emb, next_action_emb, actmask_tensor))
        except:
            probs, _ = model((state_tensor, batch_next_action_emb, actmask_tensor))  # Tensor of [bs, act_dim]
      
        # try:
        #     next_enti_emb, next_action_emb = batch_next_action_emb[0], batch_next_action_emb[1]
        #     probs, _ = model((state_tensor[0],state_tensor[1], next_enti_emb, next_action_emb, actmask_tensor))
        # except:
        #     probs, _ = model((state_tensor, batch_next_action_emb, actmask_tensor))  # Tensor of [bs, act_dim]

            # if args.use_men == 'mr_qs_us_rn_con_et_mf' or args.use_men == 'mr_qs_us_rn_con_et_mf_v2' 
                # or args.use_men == "mr_qn_us_rn_con_et_mf" or args.use_men == 'mr_qs_us_rn_con_et_mf_v3':  
        probs = probs + actmask_tensor.float()  # In order to differ from masked actions
        
        model.current_step['actions_pro'] = []
        prob_tmp = probs.tolist()[0]
        for act_, prob_ in zip(model.current_step["actions"], prob_tmp):
            prob_ = str(round(prob_, 5))
            action_prob_list = ' prob = '.join([act_, prob_])
            model.current_step['actions_pro'].append(action_prob_list)
        
        topk_probs, topk_idxs = torch.topk(probs, topk[hop], dim=1)  # LongTensor of [bs, k]

        model.current_step['next_acts'] = str(topk_idxs.tolist()[0][0])
        # print('topk_idxs = ', topk_idxs)
        model._record_case_study()

        topk_idxs = topk_idxs.detach().cpu().numpy()
        topk_probs = topk_probs.detach().cpu().numpy()

        new_path_pool, new_probs_pool, new_index_pool, new_idx = [], [], [], []
        for row in range(topk_idxs.shape[0]):
            path = path_pool[row]
            probs = probs_pool[row]
            index_ori = index_ori_list[row]

            for idx, p in zip(topk_idxs[row], topk_probs[row]):
                if idx >= len(acts_pool[row]):  # act idx is invalid
                    continue
                relation, next_node_id = acts_pool[row][idx]  # (relation, next_node_id)
                if relation == SELF_LOOP:
                    next_node_type = path[-1][1]
                else:
                    if args.envir == 'p1':
                        next_node_type = KG_RELATION[path[-1][1]][relation]
                    else:
                        next_node_type = env.et_idx2ty[next_node_id]
                    # next_node_type = KG_RELATION[path[-1][1]][relation]
                new_path = path + [(relation, next_node_type, next_node_id)]
                new_path_pool.append(new_path)
                new_probs_pool.append(probs + [p])
                new_index_pool.append(index_ori)
                new_idx.append(row)


        path_pool = new_path_pool
        probs_pool = new_probs_pool
        index_ori_list = new_index_pool
        idx_list = new_idx

        # del new_path_pool, new_probs_pool, new_index_pool, new_idx
        # print('new_idx = ', new_idx)
        # input()

        # if hop < 2:
        #     state_pool = env._batch_get_state(path_pool)
    # del index_ori_list, idx_list
    gc.collect()

    return path_pool, probs_pool



def predict_paths(policy_file, path_file, args):
    print('Predicting paths...')
        
    env = KGEnvironment(args, args.dataset, args.max_acts, max_path_len=args.max_path_len, state_history=args.state_history)
    # env.reset_path(args.eva_epochs) already in init

    pretrain_sd = torch.load(policy_file)

    model = Memory_Model(args, env.user_triplet_set, env.rela_2_index, 
                            env.act_dim, gamma=args.gamma, hidden_sizes=args.hidden).to(args.device)

    model_sd = model.state_dict()
    model_sd.update(pretrain_sd)
    model.load_state_dict(model_sd)

    trn_labels = load_labels_core_th(args,args.dataset, 'train')

    test_labels = load_labels_core_th(args,args.dataset, 'test')
    test_uids = list(test_labels.keys())
    test_uids = [uid for uid in test_uids if uid in trn_labels and uid in env.user_triplet_list]


    batch_size = 1
    start_idx = 0
    all_paths, all_probs = [], []

    times = 0
    # pbar = tqdm(total=len(test_uids))
    while start_idx < len(test_uids):
        # print(' bar state/text_uid = ', start_idx, '/', len(test_uids), end = '\r')
        end_idx = min(start_idx + batch_size, len(test_uids))
        batch_uids = test_uids[start_idx:end_idx]
        # print('batch_uids = ', batch_uids)
        # input()
        paths, probs = batch_beam_search(args, env, model, batch_uids, args.device, topk=args.topk)
        all_paths.extend(paths)
        all_probs.extend(probs)
        start_idx = end_idx

        times += 1
        if times % 100 == 0:
            print('start_idx, end_idx = ', start_idx, end_idx, end = '\r')
        # if times == 800:
        #     break

        if times == 250:
            break
        # print('start_idx, times = 
        # pbar.update(batch_size)
    predicts = {'paths': all_paths, 'probs': all_probs}
    pickle.dump(predicts, open(path_file, 'wb'))


def evaluate_paths(top_k, path_file, eva_file, train_labels, test_labels):
    embeds = load_embed(args.dataset)
    user_embeds = embeds[USER]
    purchase_embeds = embeds[PURCHASE][0]
    product_embeds = embeds[PRODUCT]
    scores = np.dot(user_embeds + purchase_embeds, product_embeds.T)

    # 1) Get all valid paths for each user, compute path score and path probability.
    results = pickle.load(open(path_file, 'rb'))
    pred_paths = {uid: {} for uid in test_labels if uid in train_labels}

    total_pre_user_num = {}

    no_skip_user = {}

    for path, probs in zip(results['paths'], results['probs']):
        uid = path[0][2]
        no_skip_user[uid] = 1

    for path, probs in zip(results['paths'], results['probs']):
        if path[-1][1] != PRODUCT:
            continue
        uid = path[0][2]
        if uid not in total_pre_user_num:
            total_pre_user_num[uid] = len(total_pre_user_num)
        if uid not in pred_paths:
            continue
        pid = path[-1][2]
        if pid not in pred_paths[uid]:
            pred_paths[uid][pid] = []

        # print('path, probs = ', path, probs)
        path_score = scores[uid][pid]
        path_prob = reduce(lambda x, y: x * y, probs)
        pred_paths[uid][pid].append((path_score, path_prob, path))

        # print('pred_paths[uid][pid] = ', pred_paths[uid][pid])
        # input()
    # 2) Pick best path for each user-product pair, also remove pid if it is in train set.
    best_pred_paths = {}
    for uid in pred_paths:
        train_pids = set(train_labels[uid])
        best_pred_paths[uid] = []
        for pid in pred_paths[uid]:
            if pid in train_pids:
                continue
            # print('pred_paths[uid][pid] = ', pred_paths[uid][pid])
            # Get the path with highest probability
            sorted_path = sorted(pred_paths[uid][pid], key=lambda x: x[1], reverse=True)
            best_pred_paths[uid].append(sorted_path[0])

            # print('sorted_path = ', sorted_path)
            # input()

    # 3) Compute top 10 recommended products for each user.
    sort_by = 'prob'
    pred_labels = {}

    total_pro_num = 0
    for uid in best_pred_paths:
        if sort_by == 'score':
            sorted_path = sorted(best_pred_paths[uid], key=lambda x: (x[0], x[1]), reverse=True)
        elif sort_by == 'prob':
            sorted_path = sorted(best_pred_paths[uid], key=lambda x: (x[1], x[0]), reverse=True)
        top_k_pids = [p[-1][2] for _, _, p in sorted_path[:top_k]]  # from largest to smallest
        # add up to 10 pids if not enough
        # end of add

        # print('uid = ', uid)
        # print('top10_pids = ', top10_pids)
        # input()

        pred_labels[uid] = top_k_pids[::-1]  # change order to from smallest to largest!
        total_pro_num += len(top_k_pids)

    avg_precision, avg_recall, avg_ndcg, avg_hit, invalid_users, cum_k = evaluate(pred_labels, test_labels, no_skip_user)

    eva_file = open(eva_file, "a")

    eva_file.write('Epoch = ' + str(args.eva_epochs) + ', top_k = ' + str(top_k) + 'no skip sort_by prob' + 'topk = 1, 10, 150, 150' + '\n')
    eva_result = 'NDCG={:.3f} |  Recall={:.3f} | HR={:.3f} | Precision={:.3f} | Invalid users={} | cum_k={}'.format(
            avg_ndcg, avg_recall, avg_hit, avg_precision, len(invalid_users), cum_k)
    eva_file.write(eva_result)
    eva_file.write('\n')
    
    eva_disturibution = 'total_pro_num = ' + str(total_pro_num) + ' len(total_pre_user_num) = ' + str(len(total_pre_user_num)) \
                         + ' avg product num = ' + str(total_pro_num/len(total_pre_user_num))
    eva_file.write(eva_disturibution)
    eva_file.write('\n')
    
    eva_user_num = 'total user num = ' + str(len(test_labels)) + ' total total_pre_user_num num  = ' +  str(len(total_pre_user_num))
    eva_file.write(eva_user_num)
    eva_file.write('\n')
    eva_file.close()


def test(args):
    if args.pretrained_st_epoch < args.eva_epochs:
        print('start predict')

        policy_file = args.save_model_dir + '/policy_model_epoch_{}.ckpt'.format(args.eva_epochs)
        path_file = args.save_model_dir + '/policy_paths_epoch_test{}.pkl'.format(args.eva_epochs)
        eva_file = args.log_dir + '/eva_noskip_top_k.txt'

        train_labels = load_labels_core_th(args,args.dataset, 'train')
        test_labels = load_labels_core_th(args,args.dataset, 'test')

        if args.run_path:
            predict_paths(policy_file, path_file, args)
        if args.run_eval:
            for top_k in [5,10, 25,50]:
                evaluate_paths(top_k, path_file, eva_file, train_labels, test_labels)

            eva_file = open(eva_file, "a")
            eva_file.write('*' * 50)
            eva_file.write('\n')
            eva_file.close()

    # eva_file = open(eva_file, "a")
    # cur_tim = time.strftime("%Y%m%d-%H%M%S")
    # eva_file.write("current time = " + str(cur_tim))


if __name__ == '__main__':
    args = parse_args()

    args.training = 0
    args.training = (args.training == 1)
    args.user_core_th_setting = True

    parameter_env_th(args)
    
    args.topk = [10, 15, 1]
    args.topk_list = [1, 10, 150, 150]

    if args.h0_embbed == True:
        log_dir_fodder = f"th_qu_{args.user_query_threshold}_{args.query_threshold}_{args.query_threshold_maximum}_lr_{args.lr}_bs_{args.batch_size}_ma_{args.max_acts}_p0{args.p_hop}_{args.name}_g_aiu_{args.att_core}_{args.item_core}_{args.user_core_th}"
    else:
        log_dir_fodder = f'th_qu_{args.user_query_threshold}_{args.query_threshold}_{args.query_threshold_maximum}_lr_{args.lr}_bs_{args.batch_size}_ma_{args.max_acts}_p1{args.p_hop}_{args.name}_g_aiu_{args.att_core}_{args.item_core}_{args.user_core_th}'
    
    args.sub_batch_size = 1
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    args.device = torch.device('cuda:0') if torch.cuda.is_available() else 'cpu'

    if args.reward_hybrid == True:
        log_dir_fodder = 'rh_' + log_dir_fodder

    args.cast_st_dir = '{}/{}/{}'.format(CASE_ST[args.dataset], args.use_men, log_dir_fodder)
    if not os.path.isdir(args.cast_st_dir):
        os.makedirs(args.cast_st_dir)

    if args.reward_rh == 'hybrid':
        args.log_dir = '{}/{}/{}'.format(EVALUATION_debug_4[args.dataset], args.use_men + '_rh_hb', log_dir_fodder)
        if not os.path.isdir(args.log_dir):
            os.makedirs(args.log_dir)
    elif args.reward_rh == 'pgpr':
        args.log_dir = '{}/{}/{}'.format(EVALUATION_debug_4[args.dataset], args.use_men + '_rh_pgpr', log_dir_fodder)
        if not os.path.isdir(args.log_dir):
            os.makedirs(args.log_dir)
    else:
        args.log_dir = '{}/{}/{}'.format(EVALUATION_debug_4[args.dataset], args.use_men +'_rh_no', log_dir_fodder)
        if not os.path.isdir(args.log_dir):
            os.makedirs(args.log_dir)

    if args.reward_rh == 'hybrid':
        args.save_model_dir = '{}/{}/{}'.format(SAVE_MODEL_DIR_debug[args.dataset], args.use_men + '_rh_hb', log_dir_fodder)
        if not os.path.isdir(args.save_model_dir):
            os.makedirs(args.save_model_dir)
    elif args.reward_rh == 'pgpr':
        args.save_model_dir = '{}/{}/{}'.format(SAVE_MODEL_DIR_debug[args.dataset], args.use_men + '_rh_pgpr', log_dir_fodder)
        if not os.path.isdir(args.save_model_dir):
            os.makedirs(args.save_model_dir)
    else:
        args.save_model_dir = '{}/{}/{}'.format(SAVE_MODEL_DIR_debug[args.dataset], args.use_men +'_rh_no', log_dir_fodder)
        if not os.path.isdir(args.save_model_dir):
            os.makedirs(args.save_model_dir)

    try:
        args.gradient_plot_save = os.path.join(args.gradient_plot,  args.use_men, log_dir_fodder)
        if not os.path.isdir(args.gradient_plot_save): 
            os.makedirs(args.gradient_plot_save)
    except:
        pass
    args.logger =  get_logger(args.log_dir + '/test_log_rd.txt')


    try:
        args.cast_st_save = args.cast_st_dir + '/case_st.txt'
        args.topk_list = [1, 1, 1, 1]
        args.topk = [1,1,1]

        args.eva_epochs = 40
        test(args)
    except:

        args.log_dir = '{}/{}/{}'.format(EVALUATION_debug_3[args.dataset], args.use_men, log_dir_fodder)
        if not os.path.isdir(args.log_dir):
            os.makedirs(args.log_dir)

        args.save_model_dir = '{}/{}/{}'.format(SAVE_MODEL_DIR_debug[args.dataset], args.use_men, log_dir_fodder)
        if not os.path.isdir(args.save_model_dir):
            os.makedirs(args.save_model_dir)

        args.cast_st_save = args.cast_st_dir + '/case_st.txt'
        args.topk_list = [1, 1, 1, 1]
        args.topk = [1,1,1]

        args.eva_epochs = 40
        test(args)

    # args.cast_st_save = args.cast_st_dir + '/case_st.txt'
    # args.topk_list = [1, 1, 1, 1]
    # args.topk = [1,1,1]

    # args.eva_epochs = 40
    # test(args)