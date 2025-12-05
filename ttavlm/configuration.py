# flake8: ignore E128
import argparse

from ttavlm.datasets import CORRUPTIONS, PACS_DOMAINS, OFFICEHOME_DOMAINS, VISDA_DOMAINS

SHIFTS = ["original"] + CORRUPTIONS + PACS_DOMAINS + OFFICEHOME_DOMAINS + VISDA_DOMAINS

ArgsType = argparse.Namespace


def argparser() -> ArgsType:
    parser = argparse.ArgumentParser()

    # Weights & Biases
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--exp_name", type=str, required=True)

    # Dev
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--display_progress", action="store_true")

    # Directories
    parser.add_argument("--root", type=str, default="/ADD/PROJECT/ROOT")
    parser.add_argument("--dataroot", type=str, default="/ADD/DATASETS/ROOT/")
    parser.add_argument("--save_root", type=str, default="work/", help="Path for base training weights")
    parser.add_argument("--save-iter", type=str, default="work/", help="Path for base training weights")
    parser.add_argument("--seeds", default=[42, 43, 44], type=int, help="List of random seeds", nargs="+")

    # Model
    parser.add_argument("--base_model_name", type=str, default="clip-ViT-B/16", choices=["clip-ViT-B/32", "clip-ViT-B/16", "resnet18", "resnet50"])

    # Dataset
    parser.add_argument(
        "--dataset",
        type=str,
        default=["cifar10"],
        nargs="+",
        choices=[
            "cifar10",
            "cifar10c",
            "cifar10new",
            "cifar100",
            "cifar100c",
            "imagenet",
            "imagenetc",
            "visda",
            "cars",
            "caltech",
            "dtd",
            "eurosat",
            "aircraft",
            "flowers",
            "food",
            "pets",
            "sun",
            "ucf",
            "dataset_suite",
            "imagenet-a",
            "imagenet-r",
            "imagenet-s",
            "imagenet-v2",
            "domains",
            "officehome",
            "pacs",
        ],
    )
    parser.add_argument("--shift_type", type=str, default=["gaussian_noise"], choices=SHIFTS + ["all"], nargs="+")
    parser.add_argument(
        "--domain",
        default="train",
        help="Domain for VisDA-C/PACS/OfficeHome",
        choices=[
            "train",
            "val",
            "art_painting",
            "cartoon",
            "photo",
            "sketch",
            "Art",
            "Clipart",
            "Product",
            "Real World",
        ],
    )
    parser.add_argument("--severity", default=[5], type=int, help="Corruption severity level (from 1 to 5)", nargs="+")
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--workers", type=int, default=8, help="Number of workers for dataloader")
    parser.add_argument("--template_type", type=str, default="default", choices=["default", "select", "all"])

    # OOD Dataset
    parser.add_argument(
        "--ood_dataset",
        default="svhnc",
        choices=[
            "svhn",
            "svhnc",
            "ninco",
            "nincoc",
            "places",
            "placesc",
            "textures",
            "texturesc",
            "lsunc",
            "tinyinc",
        ],
    )
    parser.add_argument("--ood_batch_size", default=128, type=int)

    # Test-Time Adaptation
    parser.add_argument(
        "--adaptation",
        default="tent",
        choices=[
            "source",
            "norm",
            "tent",
            "tent_oracle",
            "lame",
            "unient",
            "eta",
            "sar",
            "ostta",
            "clipartt",
            "cliptta",
            "cliptta_old",
            "ostta",
            "rotta",
            "sotta",
            "adacontrast",
            "stamp",
            "calip",
            "tda",
            "watt",
            "watt_otsu",
            "watt_unient",
            "zero",
        ],
    )
    parser.add_argument("--steps", default=1, type=int)
    parser.add_argument("--lr", default=1e-3, type=float, help="Standard learning rate")
    parser.add_argument("--optimizer_type", type=str, default="adam", choices=["sgd", "adam", "adamw"])
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay, should be 0.0 except for Imagenet (0.2)")
    parser.add_argument(
        "--score_type",
        type=str,
        default="max_prob",
        choices=(
            "logsumexp",
            "max_prob",
            "max_log_prob",
            "tcp",
            "max_logit",
            "sim",
            "entropy",
            "neglabel",
        ),
    )
    parser.add_argument(
        "--id_score_type",
        type=str,
        default="max_prob",
        choices=(
            "logsumexp",
            "max_prob",
            "max_log_prob",
            "tcp",
            "max_logit",
            "sim",
            "entropy",
            "neglabel",
        ),
    )
    parser.add_argument("--logit_scale", type=float, default=100.0)
    parser.add_argument("--ood_logit_scale", type=float, default=1.0)
    parser.add_argument("--use_sam", action="store_true")
    parser.add_argument("--skip_top_layers", action="store_true")
    parser.add_argument("--use_batch_stats_only", action="store_true")
    parser.add_argument("--use_tta", action="store_true")
    parser.add_argument("--n_augment", type=int, default=16, help="Number of augmentations for methods relying on TTA (test time augmentation)")
    parser.add_argument("--tta_reduction", type=str, default="probs", choices=["logits", "probs"])
    parser.add_argument("--update_all_params", action="store_true", help="Update all parameters or only layer norms")
    parser.add_argument("--update_text", action="store_true", help="Update text encoder's parameters as well (vision is always true).")
    parser.add_argument("--segments", type=int, default=0, help="Number of checkpointing segments to use in clip's textual encoder")
    parser.add_argument("--beta_tta", type=float, default=1.0)
    parser.add_argument("--beta_reg", type=float, default=0.1)
    parser.add_argument("--beta_ood", type=float, default=0.1)
    parser.add_argument("--beta_miss", type=float, default=0.0)

    # Plotting & measurements
    parser.add_argument("--tsne", action="store_true")
    parser.add_argument("--measure_collapse", action="store_true")
    parser.add_argument("--measure_improvement", action="store_true")

    # Memory arguments
    parser.add_argument("--num_shots", type=int, default=4, help="Number of shots per class in the memory")
    parser.add_argument("--sample_size", type=int, default=256, help="Number of example to sample from the memory at each adaptation step")

    # OOD detection
    parser.add_argument("--use_ood_loss", action="store_true")
    parser.add_argument("--detect_ood", action="store_true")

    # Oracle experiments
    parser.add_argument("--oracle_miss", action="store_true")
    parser.add_argument("--miss_weight", type=float, default=-0.1, help="Weight for missclassified examples")
    parser.add_argument("--oracle_ood", action="store_true")
    parser.add_argument("--u_split", action="store_true")

    # TTA protocole
    parser.add_argument("--episodic", action="store_true")
    parser.add_argument("--fully_continual", action="store_true")
    parser.add_argument("--closed_set", action="store_true", help="Disable open-set TTA")

    # Lame specific arguments
    parser.add_argument("--affinity", type=str, default="knn", choices=("knn", "rbf", "linear"), help="Type of affinity matrix")

    # ETA specific arguments
    parser.add_argument("--alpha_entropy", type=float, default=0.4)
    parser.add_argument("--d_margin", type=float, default=0.05)

    # SAR specific arguments
    parser.add_argument("--reset_constant_em", type=float, default=0.2)

    # CLIPTTA specific argumetns
    parser.add_argument("--use_memory", action="store_true")
    parser.add_argument("--use_scheduler", action="store_true")
    parser.add_argument("--max_iter", type=int, default=150, help="Maximum number of iterations for the lr scheduler")
    parser.add_argument("--use_softmax_entropy", action="store_true")
    parser.add_argument("--use_tent", action="store_true")

    # UniEnt specific arguments
    parser.add_argument("--use_cliptta_loss", action="store_true")

    # Shared by CLIPTTA and UniEnt for ablation experiment
    parser.add_argument("--use_clipartt_loss", action="store_true")

    # CLIPArTT specific arguments
    parser.add_argument("--K", type=int, default=3, help="Number of classes for CLIPArTT")
    parser.add_argument("--clipartt_temp", type=float, default=0.01, help="Softmax temperature for targets")

    # OTSU specific arguments
    parser.add_argument("--use_weights", action="store_true", help="Use the weighted Tent loss or not")
    parser.add_argument("--loss_ood_type", type=str, default="max_inter_var", choices=["max_inter_var", "min_intra_var", "avg_contrastive"])
    parser.add_argument("--loss_tta_type", type=str, default="entropy", choices=("entropy", "cliptta", "cliptta_star", "clipartt"))
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--detect_missclassified", action="store_true")
    parser.add_argument("--update_alpha", action="store_true")
    parser.add_argument("--update_alpha_miss", action="store_true")
    parser.add_argument("--lr_miss", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.005)
    parser.add_argument("--prior_pin", type=float, default=0.5, help="Value to bias the Otsu algorithm, default=0.5")
    parser.add_argument("--beta_schedule", type=str, default="none", choices=("sequential", "step", "none"))
    parser.add_argument("--milestone", type=int, default=2, help="Milestone for step scheduler")
    parser.add_argument("--ensemble_logits", action="store_true", help="Using an ensemble of text templates to get the logits")
    parser.add_argument("--logits_mode", type=str, default="max", choices=("max", "avg"))

    # STAMP specific arguments
    parser.add_argument("--use_consistency_filtering", action="store_true")
    parser.add_argument("--alpha_stamp", type=float, default=0.45, help="Coefficient to compute entropy threshold in STAMP (warning may vary per dataset)")
    parser.add_argument("--memory_length", type=int, default=128, help="Size of RBM memory in STAMP")

    # RoTTA specific arguments
    parser.add_argument("--capacity", type=int, default=64)
    parser.add_argument("--update_frequency", type=int, default=64)
    parser.add_argument("--nu", type=float, default=0.001)
    parser.add_argument("--lambda_t", type=float, default=1.0)
    parser.add_argument("--lambda_u", type=float, default=1.0)
    parser.add_argument("--alpha_rotta", type=float, default=0.05)

    # OSTTA specific arguments (not in official implementation)
    parser.add_argument("--margin_ostta", type=float, default=0.4)

    # SoTTA specific arguments
    parser.add_argument("--high_threshold", type=float, default=0.5)

    # AdaContrast specific arguments
    parser.add_argument("--queue_size", type=int, default=16384)
    parser.add_argument("--aug_type", type=str, default="moco-v2", choices=("moco-v1", "moco-v2"))
    parser.add_argument("--beta_ins", type=float, default=0.1)
    parser.add_argument("--n_neighbors", type=int, default=3)
    parser.add_argument("--m", type=float, default=0.999)
    parser.add_argument("--T_moco", type=float, default=0.07)

    # Watt specific arguments
    parser.add_argument("--meta_reps", type=int, default=2)
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--avg_type", type=str, default="parallel", choices=("parallel", "sequential"))

    # Zero specific arguments
    parser.add_argument("--zero_gamma", type=float, default=0.3)

    # CALIP specific arguments
    parser.add_argument("--beta_calip", nargs="+", default=[2.0, 0.1], type=float)

    # TDA specific arguments
    parser.add_argument("--pos_alpha_beta", nargs="+", default=[2.0, 5.0], type=float)
    parser.add_argument("--neg_alpha_beta", nargs="+", default=[0.117, 1.0], type=float)
    parser.add_argument("--entropy_threshold", nargs="+", default=[0.2, 0.5], type=float)
    parser.add_argument("--mask_threshold", nargs="+", default=[0.03, 1.0], type=float)
    parser.add_argument("--pos_shot_capacity", type=int, default=3)
    parser.add_argument("--neg_shot_capacity", type=int, default=2)

    # Distributed
    parser.add_argument("--distributed", action="store_true", help="Activate distributed training")
    parser.add_argument("--init-method", type=str, default="tcp://127.0.0.1:3456", help="url for distributed training")
    parser.add_argument("--dist-backend", default="gloo", type=str, help="distributed backend")
    parser.add_argument("--world-size", type=int, default=1, help="Number of nodes for training")

    return parser.parse_args()
