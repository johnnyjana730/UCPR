
from parser import parse_args

from model.lstm_base.model_lstm_mf_emb import AC_lstm_mf_dummy
from model.UCPR import UCPR

from env.env import *

args = parse_args()

# ********************* model select *****************************
if args.model == 'lstm': 
    Memory_Model = AC_lstm_mf_dummy
elif args.model == 'UCPR':
    Memory_Model = UCPR
elif args.model == 'state_history':
    Memory_Model = ActorCritic_lstm_histat
elif args.model == 'state_history_no_emb':
    Memory_Model = ActorCritic_lstm_histat_no_emb

# ********************* model select *****************************

KGEnvironment = BatchKGEnvironment

# ********************* BatchKGEnvironment ************************