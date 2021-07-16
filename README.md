# BarlowTwins-Implementation
Pytorch/XLA mplementation for ["Barlow Twins: Self-Supervised Learning via Redundancy Reduction"](https://arxiv.org/pdf/2103.03230.pdf) 

**Current results**

Model        |  dataset    | linear_classifier test acc |  finetuning with 10%  | 
------------ | ------------|    ------------------      |   ------------------  |
resnet18     | Cifar10     |         70.0%              |          70 .0%       |
 
The table contains the first result I got (without any hyperparameters tuning), so the results can be improved. The resullts for resnet50 was not good (for the first try also), more experiments on resnet50 will be done 


```
!python BarlowTwins-Implementation/main.py --batch-size=64 \
                                           --checkpoint-dir=$model_path\
                                           --load-model=False\
                                           --epochs=700

```

```
!python BarlowTwins-Implementation/supervisedEvaluation.py --checkpoint-dir=$model_path\
                                                           --weights='freeze'


```

```
!python BarlowTwins-Implementation/supervisedEvaluation.py --checkpoint-dir=$model_path\
                                                           --weights='finetune'
```

### Linear Classification evaluation: 

<img width="600" alt="Screen Shot 2021-07-14 at 18 36 39" src="https://user-images.githubusercontent.com/37993690/125674825-18c1ff1a-8040-4367-a371-eab3e0cd196d.png">

<img width="600" alt="Screen Shot 2021-07-14 at 18 36 56" src="https://user-images.githubusercontent.com/37993690/125674855-589659ed-54fd-4cdc-bc39-1fd2a4009e5a.png">

