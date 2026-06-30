import argparse


def add_task_args(parent_parser):
    parser = parent_parser.add_argument_group("Task")

    parser.add_argument("--task", type=str, default="mort_hosp", choices=['mort_hosp', 'mort_icu', 'los_3', 'los_7', 'ms_drg', 'apr_drg', 'bone_class'])

    parser.add_argument('--trim_cohort', '-T', action="store_const", const=True, default=False, help='remove cases w/ missing input modality')

    parser.add_argument("--root_dir", type=str, default="./")
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--inf_dir", type=str, default="inf")
    parser.add_argument("--ckpt_dir", type=str, default="ckpt")
    parser.add_argument("--modality", type=str, default="both")
    parser.add_argument("--note_encode_name", type=str, default="ClinicalBERT")
    parser.add_argument("--train_size_frac", type=float, default=1.)

    # New options for bone disease classification (image-text flow)
    parser.add_argument("--metadata_csv", type=str, default="data/bone_disease_mock/metadata.csv")
    parser.add_argument("--img_encode_path", type=str, default="data/bone_disease_mock/images_encoded.p")
    parser.add_argument("--txt_encode_path", type=str, default="data/bone_disease_mock/notes_encoded.p")
    parser.add_argument("--img_feat_dim", type=int, default=3584)
    parser.add_argument("--txt_feat_dim", type=int, default=3584)

    # Imbalance-handling toggles for bone_class. Default both OFF so you can
    # A/B test: sampler alone, loss-weight alone, both, or neither.
    parser.add_argument('--use_weighted_loss', action="store_const", const=True, default=False,
                         help='Apply inverse-frequency class weights to CrossEntropyLoss (bone_class task only)')
    parser.add_argument('--use_weighted_sampler', action="store_const", const=True, default=False,
                         help='Use a WeightedRandomSampler to oversample minority classes during training (bone_class task only)')
    parser.add_argument('--sampler_smoothing', type=float, default=0.4,
                         help='Exponent applied to inverse class counts for the sampler (1.0=full inverse-freq, 0=uniform)')

    # Missing-modality training (image optional, clinical notes always required).
    # Randomly masks out the image embedding during TRAINING ONLY so the model learns to
    # fall back on text-only inference via the existing attention-masking pathway in layers.py.
    parser.add_argument('--img_dropout_prob', type=float, default=0.0,
                         help='Probability of simulating a missing image at train time (0=off, e.g. 0.3 = drop image on 30%% of train samples each epoch)')
    parser.add_argument('--eval_missing_image', action="store_const", const=True, default=False,
                         help='If set, also run an extra eval pass on val/test with ALL images forced missing, to measure text-only robustness')


    return parent_parser


def add_hp_args(parent_parser):
    parser = parent_parser.add_argument_group("Hyperparameter")

    parser.add_argument("--ts_size", type=int, default=128)
    parser.add_argument("--txt_size", type=int, default=768)

    parser.add_argument("--num_layer_ts", type=int, default=1)
    parser.add_argument("--num_layer_txt", type=int, default=1)
    parser.add_argument("--num_layer_cross", type=int, default=1)

    parser.add_argument("--num_attention_heads", type=int, default=1)
    parser.add_argument("--intermediate_multiplier", type=int, default=1)

    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--dropout_grud", type=float, default=0.2)


    parser.add_argument('--add_contrast', '-C', action="store_const", const=True, default=False)

    parser.add_argument("--contrast_embed_dim", type=int, default=256)
    parser.add_argument("--queue_size", type=int, default=2000)
    parser.add_argument("--momentum", type=float, default=0.99)
    parser.add_argument("--temp", type=float, default=0.1)
    parser.add_argument("--alpha", type=float, default=0.2)

    return parent_parser



def add_train_args(parent_parser):
    parser = parent_parser.add_argument_group("Train")

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--wd", type=float, default=0)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--patience', default=5, type=int)
    parser.add_argument('--epochs', default=100, type=int)

    parser.add_argument('--device', type=int, default=0)

    parser.add_argument('--silent', action="store_const", const=True, default=False)
    parser.add_argument('--debug', '-D', action="store_const", const=True, default=False)

    parser.add_argument('--load_ckpt', "-L", default="", type=str)
    
    return parent_parser



parser = argparse.ArgumentParser()

parser = add_task_args(parser)
parser = add_hp_args(parser)
parser = add_train_args(parser)

args = parser.parse_args()
