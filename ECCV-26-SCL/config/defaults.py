from yacs.config import CfgNode as CN


_C = CN()

# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.DEVICE = "cuda"
_C.MODEL.DEVICE_ID = ["0"]
_C.MODEL.NAME = "transformer"
_C.MODEL.PRETRAIN_CHOICE = "imagenet"
_C.MODEL.PRETRAIN_PATH = "pretrained/jx_vit_base_p16_224-80ecf9dd.pth"
_C.MODEL.TRANSFORMER_TYPE = "vit_base_patch16_224_TransReID"
_C.MODEL.STRIDE_SIZE = [16, 16]
_C.MODEL.DIST_TRAIN = True
_C.MODEL.DROP_PATH = 0.1
_C.MODEL.DROP_OUT = 0.0
_C.MODEL.ATT_DROP_RATE = 0.0
_C.MODEL.SIE_COE = 3.0
_C.MODEL.METRIC_LOSS_TYPE = "triplet"
_C.MODEL.ID_LOSS_WEIGHT = 1.0
_C.MODEL.TRIPLET_LOSS_WEIGHT = 1.0

# -----------------------------------------------------------------------------
# MetaN: SCFG + NFC
# -----------------------------------------------------------------------------
_C.CHANGE = CN()
_C.CHANGE.METHODS = CN()
_C.CHANGE.METHODS.NORM_TYPE = "batch_norm"

_C.CHANGE.METHODS.FREQ_NORM_ENABLED = True
_C.CHANGE.METHODS.FREQ_NORM_KIND = "scfg"
_C.CHANGE.METHODS.FREQ_NORM_BASE = "in"
_C.CHANGE.METHODS.FREQ_NORM_POSITIONS = [0, 4, 8]
_C.CHANGE.METHODS.FREQ_NORM_POS0_KIND = "scfg"
_C.CHANGE.METHODS.SCNORM_FG_RATIO = 0.4
_C.CHANGE.METHODS.SCNORM_FG_MASK_TEMP = 0.15
_C.CHANGE.METHODS.SCFG_CONSTRAINED = True
_C.CHANGE.METHODS.SCFG_SPATIAL_SPLIT = False
_C.CHANGE.METHODS.SCFG_MIN_FG_NORM = 0.05
_C.CHANGE.METHODS.SCFG_MIN_BG_GAP = 0.10
_C.CHANGE.METHODS.SCFG_MAX_NORM = 0.95
_C.CHANGE.METHODS.SCFG_WARMUP_EPOCHS = 0

_C.CHANGE.METHODS.SCNORM_BEFORE_BNNECK = True
_C.CHANGE.METHODS.SCNORM_BNNECK_BASE = "ln"

_C.CHANGE.METHODS.NFC_TRAINING = True
_C.CHANGE.METHODS.NFC_QUEUE_SIZE = 4096
_C.CHANGE.METHODS.NFC_K1 = 8
_C.CHANGE.METHODS.NFC_K2 = 8
_C.CHANGE.METHODS.NFC_TEMPERATURE = 0.07
_C.CHANGE.METHODS.NFC_EMA_MOMENTUM = 0.993
_C.CHANGE.METHODS.NFC_FEATURE_DIM = 768
_C.CHANGE.METHODS.NFC_WARMUP_EPOCHS = 3
_C.CHANGE.METHODS.NFC_LOSS_WEIGHT = 3.0
_C.CHANGE.METHODS.NFC_DUAL_COSINE_ALIGN = False
_C.CHANGE.METHODS.NFC_FREQ_ALIGN_WEIGHT = 0.0
_C.CHANGE.METHODS.NFC_CROSS_SPECIES = True
_C.CHANGE.METHODS.NFC_CROSS_SPECIES_K1 = 4
_C.CHANGE.METHODS.NFC_CROSS_SPECIES_K2 = 4
_C.CHANGE.METHODS.NFC_SAME_SPECIES_WEIGHT = 1.0
_C.CHANGE.METHODS.NFC_CROSS_SPECIES_WEIGHT = 3.0
_C.CHANGE.METHODS.NFC_CROSS_SPECIES_MARGIN = 1.0
_C.CHANGE.METHODS.NFC_CROSS_SPECIES_METRIC = "cosine"

# -----------------------------------------------------------------------------
# Input
# -----------------------------------------------------------------------------
_C.INPUT = CN()
_C.INPUT.SIZE_TRAIN = [256, 256]
_C.INPUT.SIZE_TEST = [256, 256]
_C.INPUT.PROB = 0.5
_C.INPUT.RE_PROB = 0.0
_C.INPUT.PADDING = 10
_C.INPUT.PIXEL_MEAN = [0.5, 0.5, 0.5]
_C.INPUT.PIXEL_STD = [0.5, 0.5, 0.5]

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
_C.DATASETS.NAMES = "combine"
_C.DATASETS.COMBINE_PID = True
_C.DATASETS.TRAIN_COMBINE_NAMES = ["wildlifereid10k_species"]
_C.DATASETS.TRAIN_ROOTS = ["data/WildlifeReID-10K"]
_C.DATASETS.TEST_COMBINE_NAMES = []
_C.DATASETS.TEST_ROOTS = []

_C.DATASETS.WILDLIFE71 = CN()
_C.DATASETS.WILDLIFE71.WILDLIFE_NAMES = None

_C.DATASETS.WILDLIFEREID10K = CN()
_C.DATASETS.WILDLIFEREID10K.TRAIN_SPLIT_MODE = "all"

# -----------------------------------------------------------------------------
# Dataloader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
_C.DATALOADER.SAMPLER = "softmax_triplet"
_C.DATALOADER.NUM_INSTANCE = 4
_C.DATALOADER.NUM_WORKERS = 8

# -----------------------------------------------------------------------------
# Solver
# -----------------------------------------------------------------------------
_C.SOLVER = CN()
_C.SOLVER.OPTIMIZER_NAME = "SGD"
_C.SOLVER.MAX_EPOCHS = 60
_C.SOLVER.BASE_LR = 0.004
_C.SOLVER.IMS_PER_BATCH = 128
_C.SOLVER.MOMENTUM = 0.9
_C.SOLVER.WEIGHT_DECAY = 1e-4
_C.SOLVER.WEIGHT_DECAY_BIAS = 1e-4
_C.SOLVER.BIAS_LR_FACTOR = 2
_C.SOLVER.LARGE_FC_LR = False
_C.SOLVER.MARGIN = 0.7
_C.SOLVER.WARMUP_EPOCHS = 5
_C.SOLVER.WARMUP_METHOD = "linear"
_C.SOLVER.CHECKPOINT_PERIOD = 5
_C.SOLVER.LOG_PERIOD = 400
_C.SOLVER.EVAL_PERIOD = 60
_C.SOLVER.SEED = 1

# -----------------------------------------------------------------------------
# Test
# -----------------------------------------------------------------------------
_C.TEST = CN()
_C.TEST.EVAL = False
_C.TEST.IMS_PER_BATCH = 64
_C.TEST.RE_RANKING = False
_C.TEST.WEIGHT = ""
_C.TEST.NECK_FEAT = "before"
_C.TEST.FEAT_NORM = "yes"

_C.OUTPUT_DIR = "outputs/metan_w10k29_to_full12"
_C.RESUME = ""
