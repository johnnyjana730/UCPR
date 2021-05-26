
from parser import parse_args

from model_refine.lstm_base.test.model_lstm_test_az import *
from model_refine.lstm_base.model_lstm_mf_emb_nograd import ActorCritic_lstm_mf_dummy_emn_nograd

from model_refine.lstm_base.model_hista_base import *
from model_refine.lstm_base.model_hista_noemb import *

from model_refine.lstm_base.model_lstm_mf import *
from model_refine.lstm_query_entity.model_qe import *
from model_refine.lstm_query_mp.model_mp import *
from model_refine.lstm_query_state.model_qs import *
from model_refine.lstm_query_state.test.model_qs_test import *
from model_refine.lstm_query_state.model_rn_qs import *
from model_refine.lstm_query_state.model_rn_qs_sp import *
from model_refine.lstm_query_state.test.model_rn_qs_test import *
from model_refine.lstm_query_state.model_ku_rn_qs import *
from model_refine.lstm_query_mp.model_mp import *
from model_refine.lstm_query_mp.model_mp_up import *


from model_refine.lstm_query_state_mf.model_rn_qs_mf import *
from model_refine.lstm_query_state_mf.model_rn_qs_mf_2 import *
from model_refine.lstm_query_state_mf.model_rn_qs_mf_3 import *
from model_refine.lstm_query_state_mf.model_rn_qs_mf_4 import *
from model_refine.lstm_query_state_mf.model_rn_qs_mf_5 import lstm_query_usrn_con_et_mf_v11_up, lstm_query_usrn_con_et_mf_v11_up_wonl, lstm_query_usrn_con_et_mf_v12_up, lstm_query_usrn_con_et_mf_v13_up, lstm_query_usrn_con_et_mf_v14_up, lstm_query_usrn_con_et_mf_v15_up
from model_refine.lstm_query_state_mf.model_rn_qs_mf_6_3 import lstm_query_usrn_con_et_mf_v27, lstm_query_usrn_con_et_mf_v28, lstm_query_usrn_con_et_mf_v29, lstm_query_usrn_con_et_mf_v30
from model_refine.lstm_query_state.model_rn_qn_mf import lstm_query_ed_usrn_con_et_mf
from model_refine.lstm_query_state_mf.model_rn_qs_mf_sp import *

from model_refine.lstm_query_state_mf.model_rn_qs_mf_6_3_gd import lstm_query_usrn_con_et_mf_v28_gd, lstm_query_usrn_con_et_mf_v28_gd_up2, lstm_query_usrn_con_et_mf_v28_gd_noup

from model_refine.lstm_query_state_mf.model_rn_qs_mf_6_3_nogd import lstm_query_usrn_con_et_mf_v28_nogd, lstm_query_usrn_con_et_mf_v28_nogd_up2, lstm_query_usrn_con_et_mf_v28_nogd_noup, lstm_query_usrn_con_et_mf_v28_nogd_noft, lstm_query_usrn_con_et_mf_v35_nogd, lstm_query_usrn_con_et_mf_v35_nogd_up2


from model_refine.lstm_query_state_mf.test.model_rn_qs_mf_test_az import lstm_query_usrn_con_et_mf_v2_gate_up_test, lstm_query_usrn_con_et_mf_v2_gate_up_wonl_test, lstm_query_usrn_con_et_mf_v3_up_wonl_test, lstm_query_usrn_con_et_mf_v3_up_test, lstm_query_usrn_con_et_mf_v23_test, lstm_query_usrn_con_et_mf_v28_test, \
                                lstm_query_us_rn_up_sp_con_mf_v28_test_az

# from model.model_kg_mem_up import ActorCritic_memory_up
# from env.kg_env_memory_path import *

from env.kg_env_memory_path import *
from env.kg_env_memory_graph_path import *

args = parse_args()

# ********************* model select *****************************
# if args.use_men == 'use':
    # print('args.use_men = ', args.use_men, 'men')
    # Memory_Model = ActorCritic_memory
if args.use_men == 'lstm': 
    if args.att_evaluation == True:
        Memory_Model = ActorCritic_lstm_base_test
    else: Memory_Model = ActorCritic_lstm_base

elif args.use_men == 'lstm_mf_dummy_no_grad':
    Memory_Model = ActorCritic_lstm_mf_dummy_emn_nograd
    
elif args.use_men == 'mr_qs_rn':
    Memory_Model = lstm_query_st_rn
elif args.use_men == 'mr_qs_rn_up':
    Memory_Model = lstm_query_st_rn_up
elif args.use_men == 'mr_qs_utit_rn_up':
    Memory_Model = lstm_query_st_utit_rn_up
elif args.use_men == 'mr_qs_allst_utit_rn_up':
    Memory_Model = lstm_query_allst_utit_rn_up
elif args.use_men == 'mr_qs_ku_rn_up':
    Memory_Model = lstm_query_ku_st_rn_up
elif args.use_men == "mr_qs_us_up":
    if args.att_evaluation == True:
        print('args.use_men = ', args.use_men, 'lstm lstm_qs_us_up', args.envir) 
        Memory_Model = lstm_query_st_us_up_test
    else: Memory_Model = lstm_query_st_us_up
elif args.use_men == "mr_qs_us_uiup":
    Memory_Model = lstm_query_st_us_uiup
elif args.use_men == 'mp_qs_mpst_rn':
    Memory_Model = lstm_query_st_mpst_rn
elif args.use_men == 'mp_qs_mpst_rn_up':
    Memory_Model = lstm_query_st_mpst_rn_up
elif args.use_men == 'mr_qs_us_rn_up':
    if args.att_evaluation == True:
        print('args.use_men = ', args.use_men, 'lstm lstm_us_rn_up_test', args.envir) 
        Memory_Model = lstm_query_st_us_rn_up_test
    else: Memory_Model = lstm_query_st_us_rn_up
elif args.use_men == 'mr_qs_us_rn_up_sp_sum':
    Memory_Model = lstm_query_st_us_rn_up_sp_sum
elif args.use_men == 'mr_qs_us_rn_up_sp_con':
    Memory_Model = lstm_query_st_us_rn_up_sp_con
elif args.use_men == 'mr_qs_ku_us_rn_up':
    Memory_Model = lstm_query_ku_st_us_rn_up
elif args.use_men == 'mp_qs_utit_mpst_rn':
    Memory_Model = lstm_query_st_utit_mpst_rn
elif args.use_men == 'mp_qs_utit_mpst_rn_up':
    Memory_Model = lstm_query_st_utit_mpst_rn_up


elif args.use_men == 'mr_qs_us':
    Memory_Model = lstm_query_us_st

# elif args.use_men == 'lstm_mf':
#     Memory_Model = ActorCritic_lstm_mf

elif args.use_men == 'lstm_mf':
    if args.att_evaluation == True:
        Memory_Model =  ActorCritic_lstm_mf_base_test
    else:
        Memory_Model = ActorCritic_lstm_mf

elif args.use_men == 'state_history' or args.use_men ==  "state_history_no_grad":
    Memory_Model = ActorCritic_lstm_histat

elif args.use_men == 'lstm_alst_mf':
    Memory_Model = ActorCritic_lstm_allst_mf
elif args.use_men == 'mr_qs_us_rn_con_mf':
    Memory_Model = lstm_query_usrn_con_mf
elif args.use_men == 'mr_qs_us_rn_con_et_mf':
    Memory_Model = lstm_query_usrn_con_et_mf
elif args.use_men == 'mr_qs_us_rn_con_et_mf_v2':
    Memory_Model = lstm_query_usrn_con_et_mf_v2
elif args.use_men == 'mr_qs_us_rn_con_et_mf_v2_up':
    Memory_Model = lstm_query_usrn_con_et_mf_v2_up
elif args.use_men == 'mr_qs_us_rn_con_et_mf_v2_up_lamda':
    Memory_Model = lstm_query_usrn_con_et_mf_v2_up_wonl

elif args.use_men == 'mr_qs_us_rn_con_et_mf_v2_gate_up':
    if args.att_evaluation == True:
        print('args.use_men = ', args.use_men, 'lstm_query_usrn_con_et_mf_v2_gate_up_test', args.envir) 
        Memory_Model = lstm_query_usrn_con_et_mf_v2_gate_up_test
    else:
        Memory_Model = lstm_query_usrn_con_et_mf_v2_gate_up
elif args.use_men == 'mr_qs_us_rn_con_et_mf_v2_gate_up_lamda':
    if args.att_evaluation == True:
        print('args.use_men = ', args.use_men, 'lstm_query_usrn_con_et_mf_v2_gate_up_wonl_test', args.envir) 
        Memory_Model = lstm_query_usrn_con_et_mf_v2_gate_up_wonl_test
    else:
        Memory_Model = lstm_query_usrn_con_et_mf_v2_gate_up_wonl
elif args.use_men == 'mr_qs_us_rn_con_et_mf_v3':
    Memory_Model = lstm_query_usrn_con_et_mf_v3

elif args.use_men == 'mr_qs_us_rn_con_et_mf_v3_up':
    if args.att_evaluation == True:
        print('args.use_men = ', args.use_men, 'lstm_query_usrn_con_et_mf_v3_gate_up_wonl_test', args.envir) 
        Memory_Model = lstm_query_usrn_con_et_mf_v3_up_test
    else:
        Memory_Model = lstm_query_usrn_con_et_mf_v3_up

elif args.use_men == 'mr_qs_us_rn_con_et_mf_v3_up_lamda':
    if args.att_evaluation == True:
        print('args.use_men = ', args.use_men, 'lstm_query_usrn_con_et_mf_v3_gate_up_wonl_test', args.envir) 
        Memory_Model = lstm_query_usrn_con_et_mf_v3_up_wonl_test
    else:
        Memory_Model = lstm_query_usrn_con_et_mf_v3_up_wonl

elif args.use_men == 'mr_qs_us_rn_con_et_mf_v4':
    Memory_Model = lstm_query_usrn_con_et_mf_v4
elif args.use_men == 'mr_qs_us_rn_con_et_mf_v5':
    Memory_Model = lstm_query_usrn_con_et_mf_v5
elif args.use_men == 'mr_qs_us_rn_con_et_mf_v6':
    Memory_Model = lstm_query_usrn_con_et_mf_v6
elif args.use_men == 'mr_qs_us_rn_con_et_mf_v7':
    Memory_Model = lstm_query_usrn_con_et_mf_v7
elif args.use_men == 'mr_qs_us_rn_con_et_mf_v8':
    Memory_Model = lstm_query_usrn_con_et_mf_v8
elif args.use_men == 'mr_qs_us_rn_con_et_mf_v9':
    Memory_Model = lstm_query_usrn_con_et_mf_v9
elif args.use_men == 'mr_qs_us_rn_con_et_mf_init':
    Memory_Model = lstm_query_usrn_con_init_mf
elif args.use_men == 'mr_qs_us_rn_con_et_mf_v10':
    Memory_Model = lstm_query_usrn_con_et_mf_v10
elif args.use_men == 'mr_qs_us_rn_con_et_mf_v10_lamda':
    Memory_Model = lstm_query_usrn_con_et_mf_v10_wonl

elif args.use_men == 'mr_qs_us_rn_con_et_mf_v11':
    Memory_Model = lstm_query_usrn_con_et_mf_v11_up
elif args.use_men == 'mr_qs_us_rn_con_et_mf_v11_lamda':
    Memory_Model = lstm_query_usrn_con_et_mf_v11_up_wonl

elif args.use_men == 'mr_qs_us_rn_con_et_mf_v12':
    Memory_Model = lstm_query_usrn_con_et_mf_v12_up
elif args.use_men == 'mr_qs_us_rn_con_et_mf_v13':
    Memory_Model = lstm_query_usrn_con_et_mf_v13_up
elif args.use_men == 'mr_qs_us_rn_con_et_mf_v14':
    Memory_Model = lstm_query_usrn_con_et_mf_v14_up
elif args.use_men == 'mr_qs_us_rn_con_et_mf_v15':
    Memory_Model = lstm_query_usrn_con_et_mf_v15_up

elif args.use_men == 'mr_qs_us_rn_con_et_mf_v23_lamda':
    if args.att_evaluation == True:
        print('args.use_men = ', args.use_men, 'lstm_query_usrn_con_et_mf_v23 ', args.envir) 
        Memory_Model = lstm_query_usrn_con_et_mf_v23_test
    else:
        Memory_Model = lstm_query_usrn_con_et_mf_v23

elif args.use_men == 'mr_qs_us_rn_con_et_mf_v28_lamda':
    if args.att_evaluation == True:
        print('args.use_men = ', args.use_men, 'lstm_query_usrn_con_et_mf_v22 ', args.envir) 
        Memory_Model = lstm_query_usrn_con_et_mf_v28_test
    else:
        Memory_Model = lstm_query_usrn_con_et_mf_v28


elif args.use_men == 'mr_qs_us_rn_con_et_mf_v28_lamda_gd':
    Memory_Model = lstm_query_usrn_con_et_mf_v28_gd

elif args.use_men == 'mr_qs_us_rn_con_et_mf_v28_lamda_nogd':
    if args.att_evaluation == True:
        print('args.use_men = ', args.use_men, 'lstm_query_usrn_con_et_mf_v28 ', args.envir) 
        if args.envir == 'p1':
            Memory_Model = lstm_query_usrn_con_et_mf_v28_test_no_grad_th
        elif args.envir == 'p2':
            Memory_Model = lstm_query_usrn_con_et_mf_v28_test
    else:
        Memory_Model = lstm_query_usrn_con_et_mf_v28_nogd

elif args.use_men == 'mr_qs_us_rn_up_sp_con_mf_v28':
    if args.att_evaluation == True:
        if args.envir == 'p1':
            Memory_Model = lstm_query_us_rn_up_sp_con_mf_v28_up_test_th
        elif args.envir == 'p2':
            Memory_Model = lstm_query_us_rn_up_sp_con_mf_v28_test_az
    else:
        Memory_Model = lstm_query_us_rn_up_sp_con_mf_v28_up

elif args.use_men == 'mr_qs_us_rn_con_et_all_mf':
    Memory_Model = lstm_query_usrn_con_all_mf

elif args.use_men == 'mr_qn_us_rn_con_et_mf':
    Memory_Model = lstm_query_ed_usrn_con_et_mf

elif args.use_men == 'mr_qs_us_rn_up_sp_con_mf_v2':
    Memory_Model = lstm_query_us_rn_up_sp_con_mf_v2
elif args.use_men == 'mr_qs_us_rn_up_sp_con_mf_v3':
    Memory_Model = lstm_query_us_rn_up_sp_con_mf

# ********************* model select *****************************




# ********************** BatchKGEnvironment **********************
if args.envir == 'p1':
    if args.env_meta_path == True:
        print('KGEnvironment = BatchKGEnvironment_metapath')
        KGEnvironment = BatchKGEnvironment_metapath
    else:
        print('KGEnvironment = BatchKGEnvironment')
        KGEnvironment = BatchKGEnvironment
elif args.envir == 'p2':
    if args.env_meta_path == True:
        print('KGEnvironment = BatchKGEnvironment_graph_meta')
        KGEnvironment = BatchKGEnvironment_graph_metapath
    else: 
        print('KGEnvironment = BatchKGEnvironment_graph')
        KGEnvironment = BatchKGEnvironment_graph
# ********************* BatchKGEnvironment ************************