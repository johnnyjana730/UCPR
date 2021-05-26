from __future__ import absolute_import, division, print_function

import os
import pickle
import gzip
import argparse

from utils import *
from dataset import RW_based_dataset, KG_based_dataset
from knowledge_graph import RW_based_KG, KG_based_KG
import pandas as pd

def labels_filter(core_user_list, dataset, mode='train'):

    review_file = '{}/{}/review_{}.txt.gz'.format(DATA_DIR[dataset], 'review_data', mode)
    user_products = {}  # {uid: [pid,...], ...}

    print('len(core_user_list) = ', len(core_user_list))

    count = 0
    with gzip.open(review_file, 'r') as f:
        for line in f:

            line = line.decode('utf-8').strip()
            arr = line.split('\t')
            user_idx = int(arr[0])
            product_idx = int(arr[1])

            if user_idx in core_user_list:
                if user_idx not in user_products:
                    user_products[user_idx] = []
                user_products[user_idx].append(product_idx)
                count += 1

    print(mode + ', avg user product = ', count/len(user_products))

    return user_products

def preprocess_rw_based(args):

    print('Load', args.dataset, 'dataset from file...')
    dataset = RW_based_dataset(args, DATA_DIR[args.dataset] + '/review_data/')
        
    print('generate filter label', args.dataset, 'knowledge graph from dataset...')
    core_user_list = dataset.core_user_list
    trn_label = labels_filter(core_user_list, args.dataset, 'train')
    tst_label = labels_filter(core_user_list, args.dataset, 'test')

    print('build', args.dataset, 'knowledge graph from dataset...')
    kg = RW_based_KG(args, dataset)


    print(args.dataset, ' save dataset, trn tst label, kg')
    save_dataset(args.dataset, dataset)
    save_labels(args.dataset, trn_label, mode='train')
    save_labels(args.dataset, tst_label, mode='test')
    save_kg(args.dataset, kg)


def kg_labels_filter(core_user_list, dataset, mode='train'):

    rating_file = DATA_DIR[dataset] + '/ratings_final'
    rating_np = np.load(rating_file + '.npy')
    n_user = max(set(rating_np[:, 0])) + 1
    data = pd.read_csv(f'{DATA_DIR[dataset]}/{mode}_pd.csv',index_col=None)
    data = data.drop(data.columns[0], axis=1)
    data = data[['user','item','like']].values

    user_products = {}  # {uid: [pid,...], ...}
    for row in data:
        user_idx, product_idx, like = row[0], row[1]  + n_user, row[2]
        if like == 0: continue

        if user_idx in core_user_list:
            if user_idx not in user_products:
                user_products[user_idx] = []
            user_products[user_idx].append(product_idx)

    return user_products

def preprocess_kg_based(args):

    print('Load', args.dataset, 'dataset from file...')
    dataset = KG_based_dataset(args, DATA_DIR[args.dataset])


    print('generate filter label', args.dataset, 'knowledge graph from dataset...')
    core_user_list = dataset.core_user_list
    trn_label = kg_labels_filter(core_user_list, args.dataset, 'train')
    tst_label = kg_labels_filter(core_user_list, args.dataset, 'test')

    print('build', args.dataset, 'knowledge graph from dataset...')
    kg = KG_based_KG(args, dataset)

    print(args.dataset, ' save dataset, trn tst label, kg')
    save_dataset(args.dataset, dataset)
    save_labels(args.dataset, trn_label, mode='train')
    save_labels(args.dataset, tst_label, mode='test')
    save_kg(args.dataset, kg)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=BEAUTY_CORE, help='One of {BEAUTY, CELL, CD, CLOTH}.')
    parser.add_argument('--att_th_lower', type=int, default=0, help='core number')
    parser.add_argument('--att_th_upper', type=int, default=3000, help='core number')
    parser.add_argument('--user_top_k', type=int, default=6000, help='core number')
    parser.add_argument('--user_core_th', type=int, default=6, help='core number')
    args = parser.parse_args()
            
    if not os.path.isdir(DATA_DIR[args.dataset]):
        os.makedirs(DATA_DIR[args.dataset])

    if args.dataset in [BEAUTY_CORE, CELL_CORE, CLOTH_CORE]: preprocess_rw_based(args)
    else: preprocess_kg_based(args)

if __name__ == '__main__':
    main()

