## Paper "Improving image captioning with better use of captions"

```
@inproceedings{shi2020improving,
  title={Improving Image Captioning with Better Use of Caption},
  author={Shi, Zhan and Zhou, Xu and Qiu, Xipeng and Zhu, Xiaodan},
  booktitle={Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics},
  pages={7454--7464},
  year={2020}
}
```

### Requirements

python 2.7.15

torch 1.0.1

Specific conda env is shown in ezs.yml

BTW, you need to download [coco-captions](https://github.com/tylin/coco-caption) and [cider](https://github.com/vrama91/cider) folder in this directory for evaluation.

### Data Files and Models

Files: Add files in data directory in [google drive](https://drive.google.com/drive/folders/1VYeFocLMz2msICHu8DFRWBwTKcok7VAe?usp=sharing) or [baidu netdisk](链接：https://pan.baidu.com/s/1ddtfdlwD65cm4JmVu6GF3w 
提取码：39pa) to data directory here. See data/README for more details. 

Models: Add log directory in [google drive](https://drive.google.com/drive/folders/1VYeFocLMz2msICHu8DFRWBwTKcok7VAe?usp=sharing) or or [baidu netdisk](链接：https://pan.baidu.com/s/1ddtfdlwD65cm4JmVu6GF3w 
提取码：39pa) here.

### Scripts

MLE training:

`python train.py --gpus 0 --id experiment-mle`

RL training

`python train.py --gpus 0 --id experiment-rl --learning_rate 2e-5 --resume_from experiment-mle --resume_from_best True --self_critical_after 0 --max_epochs 60 --learning_rate_decay_start -1 --scheduled_sampling_start -1 --reduce_on_plateau`

Evaluate your own model or Load trained model:

`python eval.py --gpus 0 --resume_from experiment-mle`

and

`python eval.py --gpus 0 --resume_from experiment-rl`

### Acknowledgement

This code is based on Ruotian Luo's brilliant image captioning repo [ruotianluo/self-critical.pytorch](https://github.com/ruotianluo/self-critical.pytorch). We use the detected bounding boxes/categories/features provided by Bottom-Up [peteanderson80/bottom-up-attention](https://github.com/peteanderson80/bottom-up-attention),  [yangxuntu/SGAE](https://github.com/yangxuntu/SGAE). Many thanks for their work!









