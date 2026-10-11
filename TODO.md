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

 RIGHT NOW THE LR DROPS AT EPOCH 5 AUTOMATICALLY

 TODO - give credit to psta and opeanimal repositories as well

 maybe we need to use the space annotation for it to not overfit so bad


2. Reduce the backbone learning rate
Keep the new head's learning rate higher initially, while reducing the learning rate of the pretrained backbone. This helps preserve useful pretrained features.



3. Add or increase regularization
Try weight decay, stronger data augmentation, or dropout in the attention head. If your dataset is small, these may help the model generalize.



4. Freeze the backbone temporarily
Train only the attention head for a short period, then unfreeze the backbone and fine-tune it with a smaller learning rate. This is worth testing, although it is not always better than joint training.