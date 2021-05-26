from __future__ import absolute_import, division, print_function

import sys


import random
import pickle
import logging
import logging.handlers
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfTransformer

# from kg_memory_reasoning_model.knowledge_graph_core import KnowledgeGraph
import torch


# Dataset names.
BEAUTY_CORE = 'beauty_core'
CELL_CORE = 'cell_core'
CLOTH_CORE = 'cloth_core'
MOVIE_CORE = 'MovieLens-1M_core'
AZ_BOOK_CORE = 'amazon-book_20core'


# Model result directories.
DATA_DIR = {
    BEAUTY_CORE: '../data/Amazon_Beauty_Core',
    CELL_CORE: '../data/Amazon_Cellphones_Core',
    CLOTH_CORE: '../data/Amazon_Clothing_Core',
    MOVIE_CORE: '../data/MovieLens-1M_Core',
    AZ_BOOK_CORE: '../data/amazon-book_20core',
}

SAVE_MODEL_DIR = {
    BEAUTY_CORE: '../save_model/Amazon_Beauty_Core',
    CELL_CORE: '../save_model/Amazon_Cellphones_Core',
    CLOTH_CORE: '../save_mmodel/Amazon_Clothing_Core',
    MOVIE_CORE: '../save_model/MovieLens-1M_Core',
    AZ_BOOK_CORE: '../save_model/amazon-book_20core'
}


EVALUATION = {
    BEAUTY_CORE: '../eva/Amazon_Beauty_Core',
    CELL_CORE: '../eva/Amazon_Cellphones_Core',
    CLOTH_CORE: '../eva/Amazon_Clothing_Core',
    MOVIE_CORE: '../eva/MovieLens-1M_Core',
    AZ_BOOK_CORE: '../eva/amazon-book_20core'
}



CASE_ST = {
    BEAUTY_CORE: '../cast_st/Amazon_Beauty_Core',
    CELL_CORE: '../cast_st/Amazon_Cellphones_Core',
    CLOTH_CORE: '../cast_st/Amazon_Clothing_Core',
    MOVIE_CORE: '../cast_st/MovieLens-1M_Core',
    AZ_BOOK_CORE: '../cast_st/amazon-book_20core'
}


# Label files.
LABELS = {
    BEAUTY_CORE: (DATA_DIR[BEAUTY_CORE] + '/train_label.pkl', DATA_DIR[BEAUTY_CORE] + '/test_label.pkl'),
    CLOTH_CORE: (DATA_DIR[CLOTH_CORE] + '/train_label.pkl', DATA_DIR[CLOTH_CORE] + '/test_label.pkl'),
    CELL_CORE: (DATA_DIR[CELL_CORE] + '/train_label.pkl', DATA_DIR[CELL_CORE] + '/test_label.pkl'),
    MOVIE_CORE: (DATA_DIR[MOVIE_CORE] + '/train_label.pkl', DATA_DIR[MOVIE_CORE] + '/test_label.pkl'),
    AZ_BOOK_CORE: (DATA_DIR[AZ_BOOK_CORE] + '/train_label.pkl', DATA_DIR[AZ_BOOK_CORE] + '/test_label.pkl')
}



# Entities
USER = 'user'
PRODUCT = 'product'
WORD = 'word'
RPRODUCT = 'related_product'
BRAND = 'brand'
CATEGORY = 'category'


# Relations
PURCHASE = 'purchase'
MENTION = 'mentions'
DESCRIBED_AS = 'described_as'
PRODUCED_BY = 'produced_by'
BELONG_TO = 'belongs_to'
ALSO_BOUGHT = 'also_bought'
ALSO_VIEWED = 'also_viewed'
BOUGHT_TOGETHER = 'bought_together'
SELF_LOOP = 'self_loop'  # only for kg env
PADDING = 'padding'


KG_RELATION = {
    USER: {
        PURCHASE: PRODUCT,
        MENTION: WORD,
    },
    WORD: {
        MENTION: USER,
        DESCRIBED_AS: PRODUCT,
    },
    PRODUCT: {
        PURCHASE: USER,
        DESCRIBED_AS: WORD,
        PRODUCED_BY: BRAND,
        BELONG_TO: CATEGORY,
        ALSO_BOUGHT: RPRODUCT,
        ALSO_VIEWED: RPRODUCT,
        BOUGHT_TOGETHER: RPRODUCT,
    },
    BRAND: {
        PRODUCED_BY: PRODUCT,
    },
    CATEGORY: {
        BELONG_TO: PRODUCT,
    },
    RPRODUCT: {
        ALSO_BOUGHT: PRODUCT,
        ALSO_VIEWED: PRODUCT,
        BOUGHT_TOGETHER: PRODUCT,
    }
}


PATH_PATTERN = {
    # length = 3
    1: ((None, USER), (MENTION, WORD), (DESCRIBED_AS, PRODUCT)), 
    # length = 4
    11: ((None, USER), (PURCHASE, PRODUCT), (PURCHASE, USER), (PURCHASE, PRODUCT)),
    12: ((None, USER), (PURCHASE, PRODUCT), (DESCRIBED_AS, WORD), (DESCRIBED_AS, PRODUCT)),
    13: ((None, USER), (PURCHASE, PRODUCT), (PRODUCED_BY, BRAND), (PRODUCED_BY, PRODUCT)),
    14: ((None, USER), (PURCHASE, PRODUCT), (BELONG_TO, CATEGORY), (BELONG_TO, PRODUCT)),
    15: ((None, USER), (PURCHASE, PRODUCT), (ALSO_BOUGHT, RPRODUCT), (ALSO_BOUGHT, PRODUCT)),
    16: ((None, USER), (PURCHASE, PRODUCT), (ALSO_VIEWED, RPRODUCT), (ALSO_VIEWED, PRODUCT)),
    17: ((None, USER), (PURCHASE, PRODUCT), (BOUGHT_TOGETHER, RPRODUCT), (BOUGHT_TOGETHER, PRODUCT)),
    18: ((None, USER), (MENTION, WORD), (MENTION, USER), (PURCHASE, PRODUCT)),
}


PATH_PATTERN_GRAPH = {
    1: ((None, USER), (PURCHASE, PRODUCT), ('all', 'attribute'), ('all', PRODUCT))
}


def get_entities():
    return list(KG_RELATION.keys())


def get_entities_graph():
    return list(KG_RELATION_GRAPH.keys())


def get_relations(entity_head):
    return list(KG_RELATION[entity_head].keys())


def get_entity_tail(entity_head, relation):
    return KG_RELATION[entity_head][relation]


def compute_tfidf_fast(vocab, docs):
    """Compute TFIDF scores for all vocabs.

    Args:
        docs: list of list of integers, e.g. [[0,0,1], [1,2,0,1]]

    Returns:
        sp.csr_matrix, [num_docs, num_vocab]
    """
    # (1) Compute term frequency in each doc.
    data, indices, indptr = [], [], [0]
    for d in docs:
        term_count = {}
        for term_idx in d:
            if term_idx not in term_count:
                term_count[term_idx] = 1
            else:
                term_count[term_idx] += 1
        indices.extend(term_count.keys())
        data.extend(term_count.values())
        indptr.append(len(indices))
    tf = sp.csr_matrix((data, indices, indptr), dtype=int, shape=(len(docs), len(vocab)))

    # (2) Compute normalized tfidf for each term/doc.
    transformer = TfidfTransformer(smooth_idf=True)
    tfidf = transformer.fit_transform(tf)
    return tfidf


def get_logger(logname):
    logger = logging.getLogger(logname)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(levelname)s]  %(message)s')
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    fh = logging.handlers.RotatingFileHandler(logname, mode='w')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_dataset(dataset, dataset_obj):
    dataset_file = DATA_DIR[dataset] + '/dataset.pkl'
    with open(dataset_file, 'wb') as f:
        pickle.dump(dataset_obj, f)


def load_dataset(dataset):
    dataset_file = DATA_DIR[dataset] + '/dataset.pkl'
    dataset_obj = pickle.load(open(dataset_file, 'rb'))
    return dataset_obj


def save_dataset_test_eval(args, dataset, dataset_obj):
    dataset_file = DATA_DIR[dataset] + '/dataset'
    dataset_file += '_eval_test'+ '.pkl'
    print('save dataset test eval file = ', dataset_file)
    with open(dataset_file, 'wb') as f:
        pickle.dump(dataset_obj, f)


def save_labels(dataset, labels, mode='train'):
    if mode == 'train':
        label_file = LABELS[dataset][0]
    elif mode == 'test':
        label_file = LABELS[dataset][1]
    else:
        raise Exception('mode should be one of {train, test}.')
    with open(label_file, 'wb') as f:
        pickle.dump(labels, f)


def load_labels(dataset, mode='train'):
    if mode == 'train':
        label_file = LABELS[dataset][0]
    elif mode == 'test':
        label_file = LABELS[dataset][1]
    else:
        raise Exception('mode should be one of {train, test}.')
    print('label_file = ', label_file)

    user_products = pickle.load(open(label_file, 'rb'))
    return user_products


def save_embed(dataset, embed):
    embed_file = '{}/transe_embed.pkl'.format(DATA_DIR[dataset])
    pickle.dump(embed, open(embed_file, 'wb'))


def load_embed(dataset):
    embed_file = '{}/transe_embed.pkl'.format(DATA_DIR[dataset])
    print('Load embedding:', embed_file)
    embed = pickle.load(open(embed_file, 'rb'))
    return embed


def save_kg(dataset, kg):
    kg_file = DATA_DIR[dataset] + '/kg.pkl'
    pickle.dump(kg, open(kg_file, 'wb'))  


def load_kg(dataset):
    kg_file = DATA_DIR[dataset] + '/kg.pkl'
    kg = pickle.load(open(kg_file, 'rb'))
    return kg
