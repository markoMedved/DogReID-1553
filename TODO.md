EXPERIMENTS:
- resnet full fintune
- resnet frozen finetune
- DINOV2 full
- dinov2 frozen (bot already started, use resume flag)


maybe the whole fine-tune is actually better for the resnet than the partially frozen (but use the smaller learning rate)


- try a combination of different components observed in openanimals (No random erasing and stuff) - which ones work good,
 that can still go with the dinov2 (note that for dinov2 the crops are square by default)

 for the other methods use everything default (including crops ), do we need linear warmup for dino(look at what this linear warmup is good for, how many epochs to even take)?

 if overfit change to previous bacbkbone smaller lr 

 mogoce lahko probamo samo dropnt backbone lr po 5 epochih