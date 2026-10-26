
SBS good because of heavy augmentation? our train dataset doesnt separate the places - yeah let's go with this, after a nice arbbase fit - just explore this more - we can confirm in the baseline ablations. Recommend using heavy augmentation or even somehow use our annotation for training. 
It could also be because of the freezing of the backbone - depending on the ablation results, you can make any of the arguments - actually just both - say that resnet overfits (dinov2 we mostly freeze so it is not that much of a problem)

gallery sensitivity - I think we have to somehow focus on raising the number of ID-s, the current analysis is somewhat meaningless