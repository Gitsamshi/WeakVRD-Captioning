"""
Preprocess a raw json dataset into hdf5/json files for use in data_loader.lua

Input: json file that has the form
[{ file_path: 'path/img.jpg', captions: ['a caption', ...] }, ...]
example element in this list would look like
{'captions': [u'A man with a red helmet on a small moped on a dirt road. ', u'Man riding a motor bike on a dirt road on the countryside.', u'A man riding on the back of a motorcycle.', u'A dirt path with a young person on a motor bike rests to the foreground of a verdant area with a bridge and a background of cloud-wreathed mountains. ', u'A man in a red shirt and a red hat is on a motorcycle on a hill side.'], 'file_path': u'val2014/COCO_val2014_000000391895.jpg', 'id': 391895}

This script reads this json, does some basic preprocessing on the captions
(e.g. lowercase, etc.), creates a special UNK token, and encodes everything to arrays

Output: a json file and an hdf5 file
The hdf5 file contains several fields:
/images is (N,3,256,256) uint8 array of raw image data in RGB format
/labels is (M,max_length) uint32 array of encoded labels, zero padded
/label_start_ix and /label_end_ix are (N,) uint32 arrays of pointers to the 
  first and last indices (in range 1..M) of labels for each image
/label_length stores the length of the sequence for each of the M sequences

The json file has a dict that contains:
- an 'ix_to_word' field storing the vocab in form {ix:'word'}, where ix is 1-indexed
- an 'images' field that is a list holding auxiliary information for each image, 
  such as in particular the 'split' it was assigned to.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
import argparse
from random import shuffle, seed
import string
# non-standard dependencies:
import h5py
import numpy as np
import torch
import torchvision.models as models
import skimage.io
from PIL import Image
import collections
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm
wnl = WordNetLemmatizer()

def build_vocab(imgs, params):
  count_thr = params['word_count_threshold']

  # count up the number of words
  counts = {}
  for img in imgs:
    for sent in img['sentences']:
      for w in sent['tokens']:
        counts[w] = counts.get(w, 0) + 1
  cw = sorted([(count,w) for w,count in counts.items()], reverse=True)
  print('top words and their counts:')
  print('\n'.join(map(str,cw[:20])))

  # print some stats
  total_words = sum(counts.values())
  print('total words:', total_words)
  bad_words = [w for w,n in counts.items() if n <= count_thr]
  vocab = [w for w,n in counts.items() if n > count_thr]
  bad_count = sum(counts[w] for w in bad_words)
  print('number of bad words: %d/%d = %.2f%%' % (len(bad_words), len(counts), len(bad_words)*100.0/len(counts)))
  print('number of words in vocab would be %d' % (len(vocab), ))
  print('number of UNKs: %d/%d = %.2f%%' % (bad_count, total_words, bad_count*100.0/total_words))

  # lets look at the distribution of lengths as well
  sent_lengths = {}
  for img in imgs:
    for sent in img['sentences']:
      txt = sent['tokens']
      nw = len(txt)
      sent_lengths[nw] = sent_lengths.get(nw, 0) + 1
  max_len = max(sent_lengths.keys())
  print('max length sentence in raw data: ', max_len)
  print('sentence length distribution (count, number of words):')
  sum_len = sum(sent_lengths.values())
  for i in range(max_len+1):
    print('%2d: %10d   %f%%' % (i, sent_lengths.get(i,0), sent_lengths.get(i,0)*100.0/sum_len))

  # lets now produce the final annotations
  if bad_count > 0:
    # additional special UNK token we will use below to map infrequent words to
    print('inserting the special UNK token')
    vocab.append('UNK')
  
  for img in imgs:
    img['final_captions'] = []
    for sent in img['sentences']:
      txt = sent['tokens']
      caption = [w if counts.get(w,0) > count_thr else 'UNK' for w in txt]
      img['final_captions'].append(caption)

  return vocab

def encode_captions(imgs, params, wtoi):
  """ 
  encode all captions into one large array, which will be 1-indexed.
  also produces label_start_ix and label_end_ix which store 1-indexed 
  and inclusive (Lua-style) pointers to the first and last caption for
  each image in the dataset.
  """

  max_length = params['max_length']
  N = len(imgs)
  M = sum(len(img['final_captions']) for img in imgs) # total number of captions

  label_arrays = []
  label_start_ix = np.zeros(N, dtype='uint32') # note: these will be one-indexed
  label_end_ix = np.zeros(N, dtype='uint32')
  label_length = np.zeros(M, dtype='uint32')
  caption_counter = 0
  counter = 1
  for i,img in enumerate(imgs):
    n = len(img['final_captions'])
    assert n > 0, 'error: some image has no captions'

    Li = np.zeros((n, max_length), dtype='uint32')
    for j,s in enumerate(img['final_captions']):
      label_length[caption_counter] = min(max_length, len(s)) # record the length of this sequence
      caption_counter += 1
      for k,w in enumerate(s):
        if k < max_length:
          Li[j,k] = wtoi[w]

    # note: word indices are 1-indexed, and captions are padded with zeros
    label_arrays.append(Li)
    label_start_ix[i] = counter
    label_end_ix[i] = counter + n - 1
    
    counter += n
  
  L = np.concatenate(label_arrays, axis=0) # put all the labels together
  assert L.shape[0] == M, 'lengths don\'t match? that\'s weird'
  assert np.all(label_length > 0), 'error: some caption had no words?'

  print('encoded captions to array of size ', L.shape)
  return L, label_start_ix, label_end_ix, label_length

def main(params):

  imgs = json.load(open(params['input_json'], 'r'))
  imgs = imgs['images']

  seed(123) # make reproducible
  
  # create the vocab
  vocab = build_vocab(imgs, params)
  itow = {i+1:w for i,w in enumerate(vocab)} # a 1-indexed vocab translation table
  wtoi = {w:i+1 for i,w in enumerate(vocab)} # inverse table
  
  # encode captions in large arrays, ready to ship to hdf5 file
  L, label_start_ix, label_end_ix, label_length = encode_captions(imgs, params, wtoi)

  # create output h5 file
  N = len(imgs)
  f_lb = h5py.File(params['output_h5']+'_label.h5', "w")
  f_lb.create_dataset("labels", dtype='uint32', data=L)
  f_lb.create_dataset("label_start_ix", dtype='uint32', data=label_start_ix)
  f_lb.create_dataset("label_end_ix", dtype='uint32', data=label_end_ix)
  f_lb.create_dataset("label_length", dtype='uint32', data=label_length)
  f_lb.close()

  # create output json file
  out = {}
  out['ix_to_word'] = itow # encode the (1-indexed) vocab
  out['images'] = []
  for i,img in enumerate(imgs):
    
    jimg = {}
    jimg['split'] = img['split']
    if 'filename' in img: jimg['file_path'] = os.path.join(img.get('filepath', ''), img['filename']) # copy it over, might need
    if 'cocoid' in img:
      jimg['id'] = img['cocoid'] # copy over & mantain an id, if present (e.g. coco ids, useful)
    elif 'imgid' in img:
      jimg['id'] = img['imgid']

    if params['images_root'] != '':
      with Image.open(os.path.join(params['images_root'], img['filepath'], img['filename'])) as _img:
        jimg['width'], jimg['height'] = _img.size

    out['images'].append(jimg)
  
  json.dump(out, open(params['output_json'], 'w'))
  print('wrote ', params['output_json'])


# def weak_vrd_append(params):
#   imgs = json.load(open(params['input_json'], 'r'))
#   imgs = imgs['images']
#   itow = json.load(open(params['output_json'], 'r'))['ix_to_word']
#   wtoi = {v: k for k, v in itow.items()}
#   wtoi_lemma = {wnl.lemmatize(k): v for k, v in wtoi.items()}
#   rela_weak = json.load(open(params['rela_weak_json'], 'r'))
#   rtoi = {key: str(i) for i, (key, _) in enumerate(collections.Counter(json.load(
#     open(params['rela_weak_dict'], 'r'))).most_common(params['num_of_relas']))}
#   # rtoi.update({'[PAD]': params['num_of_relas'] + 1})
#
#
#   itor = {v: k for k, v in rtoi.items()}
#
#   max_length = params['max_length']
#   N = len(imgs)
#
#   def get_idx(wtoi, wtoi_lemma, relas):
#     ret_lst = []
#     for rela in relas:
#       ret = 0
#       if rela in wtoi:
#         ret = wtoi[rela]
#       elif rela in wtoi_lemma:
#         ret = wtoi_lemma[rela]
#       ret_lst.append(ret)
#     return ret_lst
#
#   weak_relas_arrays = np.zeros((N, max_length), dtype='uint32')
#   weak_relas_length = np.zeros(N, dtype='uint32')
#   for i, img in enumerate(imgs):
#     relas_ix_list = []
#     for ele in rela_weak[str(img['cocoid'])].keys():
#       relas_ix_list.extend(get_idx(wtoi, wtoi_lemma, itor[ele].split()))
#     for ix, ele in enumerate(relas_ix_list):
#       if ix >= params['max_length']:
#         break
#       weak_relas_arrays[i, ix] = ele
#     weak_relas_length[i] = sum(x > 0 for x in relas_ix_list)
#   # add to output h5 file
#   f_lb = h5py.File(params['output_h5'] + '_label.h5', "a")
#   del f_lb["weak_relas"]
#   del f_lb["weak_relas_length"]
#   f_lb.create_dataset("weak_relas", dtype='uint32', data=weak_relas_arrays)
#   f_lb.create_dataset("weak_relas_length", dtype='uint32', data=weak_relas_length)
#   f_lb.close()

def update_vrg(params):
  # coco_dicts
  coco_dic = json.load(open(params['ro_dict']))
  r2i = coco_dic['predicate_to_idx'] # index from 1 for these two dicts
  i2r = {v - 1: k.split()[0] for k, v in r2i.items()} # start from zero and take the first verb of phrase
  # data / coco_pred_sg_rela.npy
  i2p = np.load(params['ori_rela_dict'])[()]['i2w']  # index from 0
  i2p = {k: v.split()[0] for k, v in i2p.items()}
  print ('Dict size of wrela, prela --> {} {}'.format(len(r2i), len(i2p)))
  # big_vocab
  cocotalk = json.load(open(params['output_json'], 'r')) # index from '1'
  i2w = cocotalk['ix_to_word'] # index from '1'
  w2i = {v: k for k, v in i2w.items()}
  w2i_lemma = {wnl.lemmatize(k): v for k, v in w2i.items()}

  key_to_append = list(i2r.values()) + list(i2p.values())
  for k in key_to_append:
    if k in w2i or k in w2i_lemma:
      continue

    idx = len(w2i) + 1
    assert idx > 9487, 'original idx changed'
    w2i.update({k: str(idx)})
    i2w.update({str(idx): k})
    print (k, str(idx))

  cocotalk['ix_to_word'] = i2w
  json.dump(cocotalk, open(params['output_json'].split('.')[0] + '_final.json', 'w'))

  print ('updated size {}'.format(len(i2w)))

  cmb_folder = params['cmb_folder']
  cmb_folder_final = params['cmb_folder'] + '_final'
  if not os.path.exists(cmb_folder_final):
    os.makedirs(cmb_folder_final)
  for root, dirs, files in os.walk(cmb_folder):
    for name in tqdm(files):
      filename = os.path.join(root, name)
      file_out = os.path.join(cmb_folder_final, name.split('.')[0])
      f = np.load(filename)
      wrela, prela, obj = f['wrela'], f['prela'], f['obj'] # obj, N * 2, rela, N * 3, wrela, N * 3
      for rela in prela:
        if i2p[rela[2]] in w2i:
          rela[2] = int(w2i[i2p[rela[2]]])
        else:
          rela[2] = int(w2i_lemma[i2p[rela[2]]])

      for rela in wrela:
        # print (rela, rela.shape)
        if i2r[rela[2]] in w2i:
          rela[2] = int(w2i[i2r[rela[2]]])
        else:
          rela[2] = int(w2i_lemma[i2r[rela[2]]])

      for ob in obj:
        if i2p[ob[1]] in w2i:
          ob[1] = int(w2i[i2p[ob[1]]])
        else:
          ob[1] = int(w2i_lemma[i2p[ob[1]]])

      np.savez(file_out, wrela=wrela, prela=prela, obj=obj)



if __name__ == "__main__":

  parser = argparse.ArgumentParser()

  # input json
  parser.add_argument('--input_json', default='data/dataset_coco.json', help='input json file to process into hdf5')
  parser.add_argument('--output_json', default='data/cocotalk.json', help='output json file')
  parser.add_argument('--output_h5', default='data/cocotalk', help='output h5 file')
  parser.add_argument('--images_root', default='', help='root location in which images are stored, to be prepended to file_path in input json')

  parser.add_argument('--ro_dict', default='data/coco_dicts.json', help='coco_dicts for objs and predicates')
  parser.add_argument('--ori_rela_dict', default='data/coco_pred_sg_rela.npy', help='original vrg dict for objs, predicates')
  parser.add_argument('--cmb_folder', default='data/coco_cmb_vrg', help='original combined vrgs folder')

  parser.add_argument('--rela_weak_json', default='data/aligned_triplets_final.json', help='pre-extracted weak supervised relationships')
  parser.add_argument('--rela_weak_dict', default='data/all_predicates_final.json', help='identify weak supervised words')

  # options
  parser.add_argument('--max_length', default=16, type=int, help='max length of a caption, in number of words. captions longer than this get clipped.')
  parser.add_argument('--word_count_threshold', default=5, type=int, help='only words that occur more than this number of times will be put in vocab')
  parser.add_argument('--num_of_relas', default=1000, type=int, help='top n predicates chosen.')

  args = parser.parse_args()
  params = vars(args) # convert to ordinary dict
  print('parsed input parameters:')
  print(json.dumps(params, indent = 2))
  # main(params)
  update_vrg(params)
  # weak_vrd_append(params) #!!! need to del weak_relas, weak_relas_length

