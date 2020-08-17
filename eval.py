from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import opts
from models import setup

import eval_utils as eval_utils
import misc.utils as utils
from utils.logger import *
from utils.load_save import *
from misc.rewards_graph import init_scorer, get_self_critical_reward
from dataloader import *
import os, json

opt = opts.parse_opt()
opt.use_att = utils.if_use_att(opt.caption_model)
opt.use_fc = utils.if_use_fc(opt.caption_model)

loader = DataLoader(opt)
opt.vocab_size = loader.vocab_size
opt.seq_length = loader.seq_length

infos = load_info(opt)
opt.resume_from = 'experiment-xe-mod'
opt.resume_from_best = True
opt.beam_size = 3

decoder = setup(opt).train().cuda()

logger = define_logger(opt)
models = {'decoder': decoder}
optimizers = None
load_checkpoint(models, optimizers, opt)
print ('opt', opt)

eval_kwargs = {'split': 'test',
               'dataset': opt.input_json,
               'expand_features': False}
eval_kwargs.update(vars(opt))
predictions, lang_stats = eval_utils.eval_split(decoder, loader, eval_kwargs)

if opt.dump_json == 1:
    # dump the json
    if not os.path.exists('vis'):
        os.makedirs('vis')
    json.dump(predictions, open('vis/vis.json', 'w'))
