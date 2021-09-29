from __future__ import absolute_import, division, print_function

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
import json
from utils import *
from model.get_model.get_model import *
from parser import parse_args
from para_setting import parameter_path, parameter_path_th

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
            print('uid not in topk_matches = ',uid)
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
        
        state_tensor = model.generate_st_emb(path_pool, up_date_hop = idx_list) 

        # if args.test_lstm_up == True:
        #     state_tensor = model.generate_st_emb(path_pool, up_date_hop = idx_list) 
        # else:
        #     state_tensor = model.generate_st_emb(path_pool, test_hop = index_ori_list)

        batch_next_action_emb = model.generate_act_emb(path_pool, acts_pool)
        
        actmask_tensor = torch.BoolTensor(actmask_pool).to(device)
      
        try:
            next_enti_emb, next_action_emb = batch_next_action_emb[0], batch_next_action_emb[1]
            probs, _ = model((state_tensor[0],state_tensor[1], next_enti_emb, next_action_emb, actmask_tensor))
        except:
            probs, _ = model((state_tensor, batch_next_action_emb, actmask_tensor))  # Tensor of [bs, act_dim]
 
        probs = probs + actmask_tensor.float()  # In order to differ from masked actions
        
        topk_probs, topk_idxs = torch.topk(probs, topk[hop], dim=1)  # LongTensor of [bs, k]
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

    gc.collect()

    return path_pool, probs_pool

def predict_paths(args, policy_file, path_file, trn_labels, test_labels, pretest):
    print('Predicting paths...')
        
    env = KG_Env(args, args.dataset, args.max_acts, max_path_len=args.max_path_len, state_history=args.state_history)

    pretrain_sd = torch.load(policy_file)
    model = Memory_Model(args, env.user_triplet_set, env.rela_2_index, 
                            env.act_dim, gamma=args.gamma, hidden_sizes=args.hidden).to(args.device)
    model_sd = model.state_dict()
    model_sd.update(pretrain_sd)
    model.load_state_dict(model_sd)

    test_uids = list(test_labels.keys())
    test_uids = [uid for uid in test_uids if uid in trn_labels and uid in env.user_list]

    batch_size = 16
    start_idx = 0
    all_paths, all_probs = [], []

    times = 0
    pbar = tqdm(total=len(test_uids))
    while start_idx < len(test_uids):
        # print(' bar state/text_uid = ', start_idx, '/', len(test_uids), end = '\r')
        end_idx = min(start_idx + batch_size, len(test_uids))
        batch_uids = test_uids[start_idx:end_idx]

        paths, probs = batch_beam_search(args, env, model, batch_uids, args.device, topk=args.topk)
        all_paths.extend(paths)
        all_probs.extend(probs)
        start_idx = end_idx
        times += 1

        if times % 50 == 0:
            str_batch_uids = [str(st) for st in batch_uids]
            fail_uids = ",".join(str_batch_uids)
            fail_batch = f"'batch_uids = ', {fail_uids}, {str(start_idx)}, {str(end_idx)}"
            args.logger.info(fail_batch)

        if pretest == 1 and  times >= 100: break

        pbar.update(batch_size)
    predicts = {'paths': all_paths, 'probs': all_probs}
    pickle.dump(predicts, open(path_file, 'wb'))


def evaluate_paths(top_k, path_file, eva_file, train_labels, test_labels, sort_by = 'score'):
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

        path_score = scores[uid][pid]
        path_prob = reduce(lambda x, y: x * y, probs)
        pred_paths[uid][pid].append((path_score, path_prob, path))

    # 2) Pick best path for each user-product pair, also remove pid if it is in train set.
    best_pred_paths = {}
    for uid in pred_paths:
        train_pids = set(train_labels[uid])
        best_pred_paths[uid] = []
        for pid in pred_paths[uid]:
            if pid in train_pids:
                continue
            sorted_path = sorted(pred_paths[uid][pid], key=lambda x: x[1], reverse=True)
            best_pred_paths[uid].append(sorted_path[0])

    pred_labels = {}

    total_pro_num = 0
    for uid in best_pred_paths:
        if sort_by == 'score':
            sorted_path = sorted(best_pred_paths[uid], key=lambda x: (x[0], x[1]), reverse=True)
        elif sort_by == 'prob':
            sorted_path = sorted(best_pred_paths[uid], key=lambda x: (x[1], x[0]), reverse=True)
        top_k_pids = [p[-1][2] for _, _, p in sorted_path[:top_k]]  # from largest to smallest

        pred_labels[uid] = top_k_pids[::-1]  # change order to from smallest to largest!
        total_pro_num += len(top_k_pids)

    avg_precision, avg_recall, avg_ndcg, avg_hit, invalid_users, cum_k = evaluate(pred_labels, test_labels, no_skip_user)

    # recall file
    eva_file = open(eva_file, "a")

    eva_file.write('Epoch = ' + str(args.eva_epochs) + ', top_k = ' + str(top_k) + ' sort by ' + str(sort_by) + args.topk_string + '\n')
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

    if args.dataset in [BEAUTY_CORE, CELL_CORE, CLOTH_CORE, MOVIE_CORE]:
        return avg_recall
    else:
        return avg_ndcg

def test(args, train_labels, test_labels, best_recall, pretest = 1):

    print('start predict')


    policy_file = args.save_model_dir + '/policy_model_epoch_{}.ckpt'.format(args.eva_epochs)
    path_file = args.save_model_dir + '/' + 'pre' + str(pretest) + 'policy_paths_epoch{}_{}.pkl'.format(args.eva_epochs, args.topk_string)

    if os.path.exists(path_file) == False:
        predict_paths(args, policy_file, path_file, train_labels, test_labels, pretest)

    eva_file = args.log_dir + '/' + 'pre' + str(pretest) + args.sort_by + '_eva'+ '_' + args.topk_string + '.txt'

    for top_k in [10, 20, 25,50]:
        recall = evaluate_paths(top_k, path_file, eva_file, train_labels, test_labels, args.sort_by)
        if top_k == 10: recall_eva = recall
        
    eva_file = open(eva_file, "a")
    eva_file.write('*' * 50)
    eva_file.write('\n')
    eva_file.close()

    try:
        if args.dataset in [BEAUTY_CORE, CELL_CORE, CLOTH_CORE]: 
            sort_by_2 = 'prob'

            eva_file_2 = args.log_dir + '/' + 'pre' + str(pretest) + sort_by_2 + '_eva'+ '_' + args.topk_string + '.txt'

            for top_k in [10, 20, 25,50]:
                _ = evaluate_paths(top_k, path_file, eva_file_2, train_labels, test_labels, sort_by_2)

            eva_file_2 = open(eva_file_2, "a")
            eva_file_2.write('*' * 50)
            eva_file_2.write('\n')
            eva_file_2.close()
    except:
        pass

    if args.eva_epochs >= 20 and recall_eva >= best_recall:
        args.best_save_model_dir = args.save_model_dir + '/policy_model_epoch_{}.ckpt'.format(args.eva_epochs)
        best_recall = recall_eva
        args.best_model_epoch = args.eva_epochs



    return best_recall

if __name__ == '__main__':
    args = parse_args()

    args.training = 0
    args.training = (args.training == 1)

    args.att_evaluation = False

    if args.envir == 'p1': 
        para_env = parameter_path_th
        KG_Env = KGEnvironment

    elif args.envir == 'p2':
        para_env = parameter_path
        KG_Env = KGEnvironment

    para_env(args)

    train_labels = load_labels(args.dataset, 'train')
    test_labels = load_labels(args.dataset, 'test')

    best_recall = 0

    if args.dataset in [BEAUTY_CORE, CELL_CORE, CLOTH_CORE]: 

        epo = 10
        try:
            while True:
                args.eva_epochs = epo
                best_recall = test(args, train_labels, test_labels, best_recall)  
                epo += 20
        except:
            pass
    else:
        epo = 5
        try:
            while True:
                args.eva_epochs = epo
                best_recall = test(args, train_labels, test_labels, best_recall)  
                epo += 10
        except:
            pass

    args.eva_epochs = args.best_model_epoch
    test(args, train_labels, test_labels, best_recall, pretest = 0)

    # save pretrained md
    if args.model == 'lstm' and args.save_pretrain_model == True:
        best_model_json = {}
        best_model_json['pretrained_file'] = args.best_save_model_dir
        print('best_model_json = ', best_model_json)
        print(args.pretrained_dir + '/' + args.sort_by + '_pretrained_md_json_' + args.topk_string + '.txt')
        with open(args.pretrained_dir + '/' + args.sort_by + '_pretrained_md_json_' + args.topk_string + '.txt', 'w') as outfile:
            json.dump(best_model_json, outfile)
