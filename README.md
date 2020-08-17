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

### Data Files and Models

Uploading...

### Script

MLE training:

`python train.py --gpus 0 --id experiment-mle`

RL training

`python train.py --gpus 0 --id experiment-rl --learning_rate 2e-5 --resume_from experiment-mle resume_from_best False --self_critical_after 0 --max_epochs 50 --learning_rate_decay_start -1 --scheduled_sampling_start -1 --reduce_on_plateau`













