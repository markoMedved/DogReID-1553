EXPERIMENTS:
- resnet full fintune
- resnet frozen finetune
- DINOV2 full
- dinov2 frozen (bot already started, use resume flag)


TODO:
READ ABOUT HOW THE MODELS WORK AND DECIDE HOW TO TRANSFORM THEM TO BE READY FOR DINOV2 ADAPTATION
RUN THE DINOV2 JOBS IN A SEPERATE CHECKPOINTS_DINOV2 FOLDER FIRST MAYBE FOR ABOUT 50 EPOCHS IS FINE



 TODO - give credit to tf-clip and opeanimal repositories as well

for TF-clip we use a square resolution since thatis what they say in openanimals


# Improvements

 maybe we need to use the space annotation for it to not overfit so bad
 make our own baseline slightly better than the other models

# Resnet Tuning 
1. Maybe we can try simply reducing the learning rate as a whole (like most of the methods simply start overfitting after 10 epochs so the warmup is good but then it fails)

2. Reduce the backbone learning rate
Keep the new head's learning rate higher initially, while reducing the learning rate of the pretrained backbone. This helps preserve useful pretrained features.



3. Add or increase regularization
Try weight decay, stronger data augmentation, or dropout in the attention head. If your dataset is small, these may help the model generalize.



4. Freeze the backbone temporarily
Train only the attention head for a short period, then unfreeze the backbone and fine-tune it with a smaller learning rate. This is worth testing, although it is not always better than joint training.