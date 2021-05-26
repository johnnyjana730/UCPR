import os
from utils import *
import argparse
import random

def parse_args():
    boolean = lambda x: (str(x).lower() == 'true')
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dataset', type=str, default=BEAUTY_CORE, help='One of {clothing, cell, beauty, cd}')
    parser.add_argument('--name', type=str, default='train_agent_enti_emb', help='directory name.')
    parser.add_argument('--model', type=str, default='UCPR', help='directory name.')

    parser.add_argument('--seed', type=int, default=52, help='random seed.')
    parser.add_argument('--p_hop', type=int, default=1, help='random seed.')
    parser.add_argument('--gpu', type=int, default=0, help='gpu device.')
    parser.add_argument('--epochs', type=int, default=38, help='Max number of epochs.')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size.')
    parser.add_argument('--sub_batch_size', type=int, default=1, help='sub batch size.')
    parser.add_argument('--n_memory', type=int, default=32, help='sub batch size.')
    # parser.add_argument('--user_core', type=int, default=5, help='sub batch size.')
    parser.add_argument('--lr', type=float, default=7e-5, help='learning rate.')
    parser.add_argument('--l2_lambda', type=float, default=0, help='l2 lambda')
    parser.add_argument('--max_acts', type=int, default=50, help='Max number of actions.')
    parser.add_argument('--max_path_len', type=int, default=3, help='Max path length.')
    parser.add_argument('--gamma', type=float, default=0.99, help='reward discount factor.')
    parser.add_argument('--ent_weight', type=float, default=1e-3, help='weight factor for entropy loss')
    parser.add_argument('--reasoning_step', type=int, default=3, help='weight factor for entropy loss')
    parser.add_argument('--pretrained_st_epoch', type=int, default=0, help='h0_embbed')

    parser.add_argument('--att_core', type=int, default=0, help='h0_embbed')
    parser.add_argument('--user_core_th', type=int, default=6, help='h0_embbed')


    parser.add_argument('--embed_size', type=int, default=50, help='knowledge embedding size.')
    parser.add_argument('--act_dropout', type=float, default=0.5, help='action dropout rate.')
    parser.add_argument('--state_history', type=int, default=1, help='state history length')

    parser.add_argument('--hidden', type=int, nargs='*', default=[64, 32], help='number of samples')
    parser.add_argument('--gradient_plot',  type=str, default='gradient_plot/', help='number of negative samples.')

    parser.add_argument('--best_save_model_dir',  type=str, default='', help='best_save_model_dir')

    parser.add_argument('--reward_hybrid', type=int, default=0, help='weight factor for entropy loss')
    parser.add_argument('--reward_rh', type=str, default='', help='number of negative samples.')

    parser.add_argument('--test_lstm_up', type=int, default=1, help='test_lstm_up')
    parser.add_argument('--h0_embbed', type=int, default=0, help='h0_embbed')
    parser.add_argument('--training', type=int, default=0, help='h0_embbed')
    parser.add_argument('--load_pretrain_model', type=int, default=0, help='h0_embbed')
    parser.add_argument('--att_evaluation', type=int, default=0, help='att_evaluation')
    parser.add_argument('--state_rg', type=int, default=0, help='state_require_gradient')
    parser.add_argument('--kg_emb_grad', type=int, default=0, help='if kg_emb_grad')
    parser.add_argument('--save_pretrain_model', type=int, default=0, help='save_pretrain_model')

    parser.add_argument('--tri_wd_rm', type=int, default=0, help='tri_wd_rm')
    parser.add_argument('--tri_pro_rm', type=int, default=0, help='tri_pro_rm')

    parser.add_argument('--l2_weight', type=float, default=1e-6, help='weight of the l2 regularization term')

    parser.add_argument('--sam_type',  type=str, default='alet', help='number of negative samples.')
    parser.add_argument('--add_products', type=boolean, default=False, help='add_products')
    parser.add_argument('--topk', type=int, nargs='*', default=[10, 10, 1], help='number of samples')
    parser.add_argument('--topk_list', type=int, nargs='*', default=[1, 10, 100, 100], help='number of samples')
    parser.add_argument('--run_path', type=boolean, default=True, help='Generate predicted path? (takes long time)')
    parser.add_argument('--run_eval', type=boolean, default=True, help='Run evaluation?')

    parser.add_argument('--pretest', type=int, default=0, help='pretest')

    parser.add_argument('--item_core', type=int, default=10, help='core number')
    parser.add_argument('--user_core', type=int, default=300, help='core number')
    parser.add_argument('--best_model_epoch', type=int, default=0, help='core number')

    parser.add_argument('--kg_fre_lower', type=int, default=5, help='core number')
    parser.add_argument('--kg_fre_upper', type=int, default=500, help='core number')

    parser.add_argument('--lambda_num', type=float, default=0.5, help='core number')

    parser.add_argument('--non_sampling', type=boolean, default=False, help='core number')
    parser.add_argument('--gp_setting', type=str, default='6000_800_15_500_250', help='core number')
    parser.add_argument('--kg_no_grad', type=boolean, default=False, help='core number')

    parser.add_argument('--sort_by', type=str, default='score', help='score or prob')
    parser.add_argument('--eva_epochs', type=int, default=0, help='core number')
    parser.add_argument('--KGE_pretrained', type=int, default=0, help='KGE_pretrained')
    parser.add_argument('--load_pt_emb_size', type=int, default=0, help='core number')

    args = parser.parse_args()
    args.gpu = str(args.gpu)


    args.KGE_pretrained = (args.KGE_pretrained == 1)
    args.reward_hybrid = (args.reward_hybrid == 1)
    args.test_lstm_up = (args.test_lstm_up == 1)
    args.load_pretrain_model = (args.load_pretrain_model == 1)
    args.att_evaluation = (args.att_evaluation == 1)
    args.state_rg = (args.state_rg == 1)
    args.load_pt_emb_size = (args.load_pt_emb_size == 1)
    args.kg_emb_grad = (args.kg_emb_grad == 1)

    args.pretest = (args.pretest == 1)
    args.save_pretrain_model = (args.save_pretrain_model == 1)

    args.tri_wd_rm = (args.tri_wd_rm == 1)
    args.tri_pro_rm = (args.tri_pro_rm == 1)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    args.device = torch.device('cuda:0') if torch.cuda.is_available() else 'cpu'


    if args.dataset in [BEAUTY_CORE, CELL_CORE, CLOTH_CORE]: 
        args.envir = 'p1'
        args.sort_by = 'score'
        # if args.KGE_pretrained == True: args.embed_size = 100
    else: 
        args.envir = 'p2'
        args.sort_by = 'prob'
        # if args.KGE_pretrained == True: args.embed_size = 50

    if args.dataset in [BEAUTY_CORE, CELL_CORE, CLOTH_CORE]: 
        args.topk = [10, 10, 1]
        args.topk_list = [1, 10, 100, 100]

    elif args.dataset == MOVIE_CORE:
        args.topk = [8, 3, 4]
        args.topk_list = [1, 8, 24, 96]        

    elif args.dataset == AZ_BOOK_CORE:
        args.topk = [8, 2, 6]
        args.topk_list =  [1, 8, 16, 96]


    args.topk_string = ', '.join([str(k) for k in args.topk])

    if args.model in ['lstm', 'state_history']:
        args.non_sampling = True

    return args