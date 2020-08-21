import numpy as np
import torch
import os
import pickle as pkl
import json
import argparse
import collections
import itertools
from tqdm import tqdm


def decode(sub, obj, pred, sent_dict):
    '''
    :param sub:  sub[0] is subject
    :param obj:  obj[0] is object
    :param pred:  predicate
    :param sent_dict: sentence sg graph dict (i2w) the w is usually like w1/w2
    :return:
    '''
    sbj = sent_dict[sub[0]].split('/')
    obj = sent_dict[obj[0]].split('/')
    pred = sent_dict[pred].split('/')
    return sbj, obj, pred[0]


def check(sbj, obj, noun_set, cat2super):
    '''
    :param sbj:  sbj words in sentence
    :param obj:  obj words in sentence
    :param noun_set:  detected categories in images
    :param cat2super: category to super-category
    :return:
    '''
    def exists(e, s_dict, n_set):
        if any([e in n_set, s_dict.get(e, None) in n_set]):
            return True

    sbj_flag = any([exists(ele, cat2super, noun_set) for ele in sbj])
    obj_flag = any([exists(ele, cat2super, noun_set) for ele in obj])
    ret = sbj_flag and obj_flag
    return ret


def get_index(sbj, obj, sg_img_obj, cat2super, maxnum=3):
    '''
    :param sbj: sentence subject word
    :param obj: sentence object word
    :param sg_img_obj: image detected regions
    :param cat2super: category to super-category
    :param maxnum: max subject or object candidates for each predicate
    :return:
    '''
    def index(tgt, objs, s_dict):
        ret = [i for i, x in enumerate(objs) if x == tgt]
        if len(ret) == 0:
            ret = [i for i, x in enumerate(objs) if x == s_dict.get(tgt, None)]

        return ret

    sbj_lst, obj_lst = [], []
    for e in sbj:
        sbj_lst.extend(index(e, sg_img_obj, cat2super))
    for e in obj:
        obj_lst.extend(index(e, sg_img_obj, cat2super))

    pairs = []
    if len(sbj_lst) > 0 and len(obj_lst) > 0:
        sbj_set, obj_set = set(), set()
        for e in sbj_lst:
            sbj_set.add(e)
            if len(sbj_set) >= maxnum:
                break
        for e in obj_lst:
            obj_set.add(e)
            if len(obj_set) >= maxnum:
                break
        pairs = [ele for ele in itertools.product(sbj_set, obj_set) if ele[0] != ele[1]]
    return pairs


def main(params):
    """
        :param params:
        :return:
        coco_pred_sg:
        {'rela_matrix': array([[0., 2., 425.]),
           'obj_attr': array([[0., 36., 313.])}

        coco_spice_sg:
        {'rela_info': array([[5.0000e+00, 0.0000e+00, 1.0414e+04]),
          'obj_info': [[179], [833], [1018], [1092], [3788], [5989, 6623, 4081], [7128], [7372, 5989]]}
    """
    imgs = json.load(open(params['input_json'], 'r'))['images']
    # category to supercategory
    cat2supercat = {}
    text = open(params['category_txt'], 'r').readlines()
    for line in text:
        line = line.strip().split(',')
        assert len(line) >= 1
        for i in range(len(line)):
            cat2supercat[line[i]] = line[0]

    # detection categories
    det_dict = np.load(params['img_sg_dict'], allow_pickle=True)[()]['i2w']
    sent_dict = np.load(params['sent_sg_dict'], allow_pickle=True)['spice_dict'][()]['ix_to_word']

    # sentence scene graph and pre-trained image scene graph
    sg_img_dir = 'data/coco_img_sg'
    sg_snt_dir = 'data/coco_spice_sg2'

    # load predicates vocab if not None
    if os.path.isfile(params['pred_category']):
        print ('--loading predicates--')
        predicates = {key: i for i, (key, _) in enumerate(collections.Counter(json.load(
                                    open(params['pred_category'], 'r'))).most_common(params['top_predicates']))}
    else:
        predicates = None
    aligned_triplets = {}
    pred_candidate = []

    if predicates is None:
        print('----------------collecting predicates---------------------')
    else:
        print('----------------collecting alignments---------------------')

    for img in tqdm(imgs):
        split = img['split']
        if split not in ['train', 'restval'] and predicates is None:
            continue
        name = str(img['id']) + '.npy'

        sg_img_file = os.path.join(sg_img_dir, name)
        sg_snt_file = os.path.join(sg_snt_dir, name)
        sg_img_use = np.load(sg_img_file, encoding='latin1', allow_pickle=True)[()]
        sg_snt_use = np.load(sg_snt_file, encoding='latin1', allow_pickle=True)[()]
        sg_snt_rela = sg_snt_use['rela_info'].astype(int)
        sg_snt_obj = sg_snt_use['obj_info']

        sg_img_obj_set = set(
            [det_dict[ele] for ele in set(sg_img_use['obj_attr'].astype(int)[:, 1].reshape(-1).tolist())])
        sg_img_obj = [det_dict[ele] for ele in sg_img_use['obj_attr'].astype(int)[:, 1].reshape(-1).tolist()]

        tmp = dict()
        for snt_rela in sg_snt_rela:
            sub_ix, obj_ix, pred_ix = snt_rela
            sub, obj = sg_snt_obj[sub_ix], sg_snt_obj[obj_ix]
            sub, obj, pred = decode(sub, obj, pred_ix, sent_dict)

            if predicates is not None and pred in predicates:
                all_pairs = get_index(sub, obj, sg_img_obj, cat2supercat)
                if len(all_pairs) != 0:
                    tmp[predicates[pred]] = all_pairs

            if predicates is None and check(sub, obj, sg_img_obj_set, cat2supercat):
                pred_candidate.append(pred)

        if predicates is not None:
            aligned_triplets[name.split('.')[0]] = tmp

    if predicates is None:
        cnt_pred = collections.Counter(pred_candidate)
        json.dump(dict(cnt_pred), open(params['pred_category'], 'w'))
    else:
        json.dump(aligned_triplets, open(params['aligned_triplets'], 'w'), indent=4, sort_keys=True, default=str)
        get_dict_file(params)

    return


def get_dict_file(params):
    '''
    :param params:
    :return:
    dump predicates and objects categories
    Warning: index are offset to start from 1 (make room for  __background__ class)
    '''
    predicates = {key: i + 1 for i, (key, _) in enumerate(collections.Counter(json.load(
        open(params['pred_category'], 'r'))).most_common(params['top_predicates']))}

    det_dict = np.load(params['img_sg_dict'], allow_pickle=True)[()]['i2w']
    objects = {v: k + 1 for k, v in det_dict.items()}
    json.dump({'label_to_idx': objects, 'predicate_to_idx': predicates}, open(params['info_file'], 'w'))
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_json', default='data/cocotalk.json', help='input json file')
    parser.add_argument('--category_txt', default='data/coco_class_names.txt', help='convert supercategory to category')
    parser.add_argument('--img_sg_dict', default='data/coco_pred_sg_rela.npy', help='img graph dict')
    parser.add_argument('--sent_sg_dict', default='data/spice_sg_dict2.npz', help='snt graph dict')
    parser.add_argument('--pred_category', default='data/all_predicates_final.json', help='get all predicates')
    parser.add_argument('--aligned_triplets', default='data/aligned_triplets_final.json', help='get aligned weak supervision')
    parser.add_argument('--top_predicates', default=1000, type=int, help='top predicates choosen')
    parser.add_argument('--info_file', default='data/coco_dicts.json', help='coco dict for obj and pred')

    args = parser.parse_args()
    params = vars(args)
    print('parsed input parameters:')
    print(json.dumps(params, indent=2))
    # do main twice, lol
    main(params)
    main(params)