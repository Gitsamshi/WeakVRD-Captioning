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





# # Input arguments and options
# parser = argparse.ArgumentParser()
# # Input paths
# parser.add_argument('--model', type=str, default='',
#                 help='path to model to evaluate')
# parser.add_argument('--cnn_model', type=str,  default='resnet101',
#                 help='resnet101, resnet152')
# parser.add_argument('--infos_path', type=str, default='',
#                 help='path to infos to evaluate')
# parser.add_argument('--dataset', type=str, default='coco',
#                     help='Cached token file for calculating cider score during self critical training.')
#
# opts.add_eval_options(parser)
#
# opt = parser.parse_args()
#
# with open(opt.infos_path, 'rb') as f:
#     infos = cPickle.load(f)
#
# # override and collect parameters
# replace = ['input_fc_dir', 'input_att_dir', 'input_box_dir', 'input_label_h5', 'input_json']
# ignore = ['start_from']
#
# for k in vars(infos['opt']).keys():
#     if k in replace:
#         setattr(opt, k, getattr(opt, k) or getattr(infos['opt'], k, ''))
#     elif k not in ignore:
#         if not k in vars(opt):
#             vars(opt).update({k: vars(infos['opt'])[k]}) # copy over options from model
#
# vocab = infos['vocab'] # ix -> word mapping
#
# # Setup the model
# decoder = models.setup(opt)
# params = torch.load(opt.model)
# models = {'decoder': decoder}
# for name, model in models.items():
#     model.load_state_dict(params[name])
# model.cuda()
# model.eval()
#
# # Create the Data Loader instance
# loader = DataLoader(opt)
#
# # When eval using provided pretrained model, the vocab may be different from what you have in your cocotalk.json
# # So make sure to use the vocab in infos file.
# loader.ix_to_word = infos['vocab']
#
#
# # Set sample options
# eval_kwargs = {'split': 'test',
#                'dataset': opt.input_json,
#                'expand_features': False}
# eval_kwargs.update(vars(opt))
# # loss, split_predictions, lang_stats = eval_utils.eval_split(model, crit, loader,
# #     vars(opt))
#
# predictions, lang_stats = eval_utils.eval_split(model, loader, eval_kwargs)
#
# if lang_stats:
#   print(lang_stats)
#
# if opt.dump_json == 1:
#     # dump the json
#     if not os.path.exists('vis'):
#         os.makedirs('vis')
#     json.dump(predictions, open('vis/vis.json', 'w'))