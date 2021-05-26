from utils import *
import argparse

def parse_args():
    boolean = lambda x: (str(x).lower() == 'true')
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dataset', type=str, default=BEAUTY_CORE, help='One of {clothing, cell, beauty, cd}')
    parser.add_argument('--name', type=str, default='train_agent_enti_emb', help='directory name.')
    parser.add_argument('--use_men', type=str, default='use', help='directory name.')
    parser.add_argument('--seed', type=int, default=123, help='random seed.')
    parser.add_argument('--p_hop', type=int, default=1, help='random seed.')
    parser.add_argument('--gpu', type=int, default=1, help='gpu device.')
    parser.add_argument('--epochs', type=int, default=38, help='Max number of epochs.')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size.')
    parser.add_argument('--sub_batch_size', type=int, default=3, help='sub batch size.')
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

    parser.add_argument('--embed_size', type=int, default=20, help='knowledge embedding size.')
    parser.add_argument('--act_dropout', type=float, default=0.5, help='action dropout rate.')
    parser.add_argument('--state_history', type=int, default=1, help='state history length')
    parser.add_argument('--hidden', type=int, nargs='*', default=[64, 32], help='number of samples')
    parser.add_argument('--gradient_plot',  type=str, default='gradient_plot/', help='number of negative samples.')

    parser.add_argument('--reward_hybrid', type=int, default=0, help='weight factor for entropy loss')
    parser.add_argument('--reward_rh', type=str, default='', help='number of negative samples.')

    parser.add_argument('--test_lstm_up', type=int, default=1, help='user_o')
    parser.add_argument('--env_meta_path', type=int, default=0, help='user_o')
    parser.add_argument('--user_o', type=int, default=0, help='user_o')
    parser.add_argument('--h0_embbed', type=int, default=0, help='h0_embbed')
    parser.add_argument('--training', type=int, default=0, help='h0_embbed')
    parser.add_argument('--load_pretrain_model', type=int, default=0, help='h0_embbed')
    parser.add_argument('--att_evaluation', type=int, default=0, help='att_evaluation')
    parser.add_argument('--state_rg', type=int, default=0, help='state_require_gradient')


    parser.add_argument('--add_products', type=boolean, default=False, help='add_products')
    parser.add_argument('--topk', type=int, nargs='*', default=[10, 10, 1], help='number of samples')
    parser.add_argument('--topk_list', type=int, nargs='*', default=[1, 10, 100, 100], help='number of samples')
    parser.add_argument('--run_path', type=boolean, default=True, help='Generate predicted path? (takes long time)')
    parser.add_argument('--run_eval', type=boolean, default=True, help='Run evaluation?')

    parser.add_argument('--att_core', type=int, default=0, help='core number')
    parser.add_argument('--item_core', type=int, default=10, help='core number')
    parser.add_argument('--user_core', type=int, default=300, help='core number')

    parser.add_argument('--query_threshold', type=int, default=5, help='core number')
    parser.add_argument('--query_threshold_maximum', type=int, default=500, help='core number')

    parser.add_argument('--non_sampling', type=boolean, default=False, help='core number')
    parser.add_argument('--gp_setting', type=str, default='6000_800_15_500_250', help='core number')
    parser.add_argument('--kg_no_grad', type=boolean, default=False, help='core number')

    parser.add_argument('--user_core_th_setting', type=int, default=1, help='core number')

    args = parser.parse_args()
    args.gpu = str(args.gpu)
    # print(args.max_acts)
    # input()

    # args.name = args.name + '_nw_gh'

    args.user_o = (args.user_o == 1)
    args.h0_embbed = (args.h0_embbed == 1)
    args.reward_hybrid = (args.reward_hybrid == 1)
    args.test_lstm_up = (args.test_lstm_up == 1)
    args.env_meta_path = (args.env_meta_path == 1)
    args.load_pretrain_model = (args.load_pretrain_model == 1)
    args.att_evaluation = (args.att_evaluation == 1)
    args.state_rg = (args.state_rg == 1)
    args.user_core_th_setting = (args.user_core_th_setting == 1)

    if args.dataset in [BEAUTY_CORE, CELL_CORE, CD_CORE, CLOTH_CORE]: args.envir = 'p1'
    else: args.envir = 'p2'


    if args.use_men in ['lstm_mf', 'state_history','state_history_no_emb', 'state_history_no_grad']:
        args.non_sampling = True

    if args.use_men == "state_history_no_grad":
        args.non_sampling = True

    # # args.hidden = [512, 256]

    # if args.dataset == LAST_FM_CORE or args.dataset == AZ_BOOK_CORE:
    #     args.lr = 1e-4

    # if args.dataset == BEAUTY_CORE or args.dataset == CELL_CORE or args.dataset == MOVIE_CORE:
    #     args.user_core = 5
    # elif args.dataset == CD_CORE:
    #     args.user_core = 8
    #     args.batch_size = 256

    return args