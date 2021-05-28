import os
import numpy as np
from utils import EVALUATION, SAVE_MODEL_DIR, get_logger

def parameter_path(args):
    if args.gp_setting == '6000_800_15_500_50':
        args.att_core = 0
        args.item_core = 0
        args.user_core = 6000
        args.kg_fre_lower = 15
        args.kg_fre_upper = 500
        args.max_acts = 50
    elif args.gp_setting == '6000_100_15_500_50':
        args.att_core = 0
        args.item_core = 0
        args.user_core = 6000
        args.kg_fre_lower = 15
        args.kg_fre_upper = 500
        args.max_acts = 50
    elif args.gp_setting == '6000_800_8_500_50':
        args.att_core = 0
        args.item_core = 0
        args.user_core = 6000
        args.kg_fre_lower = 8
        args.kg_fre_upper = 500
        args.max_acts = 50
    elif args.gp_setting == '6000_800_4_500_50':
        args.att_core = 0
        args.item_core = 0
        args.user_core = 6000
        args.kg_fre_lower = 4
        args.kg_fre_upper = 500
        args.max_acts = 50
    elif args.gp_setting == '6000_800_0_500_50':
        args.att_core = 0
        args.item_core = 0
        args.user_core = 6000
        args.kg_fre_lower = 0
        args.kg_fre_upper = 500
        args.max_acts = 50
    elif args.gp_setting == '6000_50_20_500_50':
        args.att_core = 0
        args.item_core = 0
        args.user_core = 6000
        args.kg_fre_lower = 20
        args.kg_fre_upper = 500
        args.max_acts = 50
    elif args.gp_setting == '60000_150_15_1500_50':
        args.att_core = 0
        args.item_core = 0
        args.user_core = 6000
        args.kg_fre_lower = 15
        args.kg_fre_upper = 1500
        args.max_acts = 50
    elif args.gp_setting == '60000_150_15_500_50':
        args.att_core = 0
        args.item_core = 0
        args.user_core = 6000
        args.kg_fre_lower = 15
        args.kg_fre_upper = 500
        args.max_acts = 50

    print('args.gp_setting = ', args.gp_setting, 'args.att_core = ', args.att_core,
         'args.item_core = ' ,args.item_core , 'args.user_core = ', args.user_core, 
         'args.kg_fre_upper = ', args.kg_fre_upper, 'args.max_acts = ', args.max_acts)        

    log_dir_fodder = f'{args.name}_g_aiu_{args.att_core}_{args.item_core}_{args.user_core}'

    args.log_dir = '{}/{}/{}'.format(EVALUATION[args.dataset], args.model, log_dir_fodder)
    if not os.path.isdir(args.log_dir):
        os.makedirs(args.log_dir)

    args.save_model_dir = '{}/{}/{}'.format(SAVE_MODEL_DIR[args.dataset], args.model, log_dir_fodder)
    if not os.path.isdir(args.save_model_dir):
        os.makedirs(args.save_model_dir)

    log_dir_fodder_rw = f'g_aiu_0_{args.item_core}_{args.user_core}'

    args.pretrained_dir = '{}/{}/{}'.format(EVALUATION[args.dataset], 'pretrained', log_dir_fodder_rw)
    if not os.path.isdir(args.pretrained_dir):
        os.makedirs(args.pretrained_dir)

    args.logger =  get_logger(args.log_dir + '/test_log_rd.txt')

def parameter_path_th(args):
    if args.gp_setting == '6_800_15_500_50':
        args.att_core = 0
        args.item_core = 0
        args.user_core_th = 6
        args.kg_fre_lower = 15
        args.kg_fre_upper = 500
        args.max_acts = 50

    elif args.gp_setting == '6_100_15_500_50':
        args.att_core = 0
        args.item_core = 0
        args.user_core_th = 6
        args.kg_fre_lower = 15
        args.kg_fre_upper = 500
        args.max_acts = 50

    print('args.gp_setting = ', args.gp_setting, 'args.att_core = ', args.att_core,
         'args.item_core = ' ,args.item_core , 'args.user_core_th = ', args.user_core_th,
        'args.kg_fre_upper = ', args.kg_fre_upper, 'args.max_acts = ', args.max_acts)  

    log_dir_fodder = f'{args.name}_g_aiu_{args.att_core}_{args.item_core}_{args.user_core}'

    args.log_dir = '{}/{}/{}'.format(EVALUATION[args.dataset], args.model, log_dir_fodder)
    if not os.path.isdir(args.log_dir):
        os.makedirs(args.log_dir)

    args.save_model_dir = '{}/{}/{}'.format(SAVE_MODEL_DIR[args.dataset], args.model, log_dir_fodder)
    if not os.path.isdir(args.save_model_dir):
        os.makedirs(args.save_model_dir)

    # log_dir_fodder_rw = f'g_aiu_{args.att_core}_{args.item_core}_{args.user_core}'

    args.pretrained_dir = '{}/{}/{}'.format(EVALUATION[args.dataset], 'pretrained', 'emb_szie_' + str(args.embed_size))
    if not os.path.isdir(args.pretrained_dir):
        os.makedirs(args.pretrained_dir)

    args.logger =  get_logger(args.log_dir + '/test_log_rd.txt')