import argparse
# from comp.zoo import models
from custom_comp.zoo import models



def int2bool(i):
    i = int(i)
    assert i == 0 or i == 1
    return i == 1



def parse_args():
    parser = argparse.ArgumentParser(description="Example training script.")
    parser.add_argument(
        "-m",
        "--model",
        default="stf",
        choices=models.keys(),
        help="Model architecture (default: %(default)s)",
    )
    parser.add_argument(
        "-d", "--dataset", type=str, required=True, help="Training dataset"
    )
    parser.add_argument(
        "-e",
        "--epochs",
        default=100,
        type=int,
        help="Number of epochs (default: %(default)s)",
    )
    parser.add_argument(
        "-lr",
        "--learning-rate",
        default=1e-4,
        type=float,
        help="Learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "-n",
        "--num-workers",
        type=int,
        default=30,
        help="Dataloaders threads (default: %(default)s)",
    )
    parser.add_argument(
        "--lambda",
        dest="lmbda",
        type=float,
        default=1e-2,
        help="Bit-rate distortion parameter (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Batch size (default: %(default)s)"
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=64,
        help="Test batch size (default: %(default)s)",
    )
    parser.add_argument(
        "--aux-learning-rate",
        default=1e-3,
        type=float,
        help="Auxiliary loss learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        nargs=2,
        default=(256, 256),
        help="Size of the patches to be cropped (default: %(default)s)",
    )
    # parser.add_argument("--cuda", action="store_true", help="Use cuda")
    parser.add_argument("--cuda", type=int2bool, default=1) # 1 == True

    # parser.add_argument(
    #     "--save", action="store_true", default=True, help="Save model to disk"
    # )
    parser.add_argument("--save", type=int2bool, default=1) # 1 == True
    
    parser.add_argument(
        "--seed", type=int, help="Set random seed for reproducibility", default=42
    )
    parser.add_argument(
        "--clip_max_norm",
        default=1.0,
        type=float,
        help="gradient clipping max norm (default: %(default)s",
    )
    parser.add_argument("--checkpoint", type=str, help="Path to a checkpoint", default=None)



    # parser.add_argument("--resume-train", action="store_true", help="Resume the training procedure from a checkpoint")
    parser.add_argument("--resume-train", type=int2bool, default=0) # 1 == True

    parser.add_argument("--save-dir", type = str, help = "Save directory", default = "./exp")
    parser.add_argument("--test-dir", type = str, help = "Kodak Test directory", default = "/data/kodak")

    # parser.add_argument("--lora", action="store_true", default=False)
    parser.add_argument("--lora", type=int2bool, default=0) # 1 == True
    parser.add_argument("--vanilla-adapt", type=int2bool, default=0) # 1 == True


    parser.add_argument("--lora-config", type = str)

    parser.add_argument("--lora-opt", type = str, default='adam', choices=['adam','sgd'])
    parser.add_argument("--lora-sched", type = str, default='lr_plateau', choices=['lr_plateau','cosine'])
    parser.add_argument("--mask", action="store_true",help="Apply pruning mask to the model")
    
    parser.add_argument("--nameRun", type = str, default='Magv', help="name of the run")
    parser.add_argument("--adjustDistrib", action="store_true",help="modifyng the sampling distribution of mask")
    parser.add_argument("--fisher", action="store_true",help="compute neuron fisher information")

    parser.add_argument("--alpha", type=int,help="LoRA scale factor")
    parser.add_argument("--rank", type=int,help="LoRA rank")
    parser.add_argument("--maxPoint", type=int, default=6, help="Maximum point on the RD curve")
    parser.add_argument("--maxPrunning", type=float, default=0.6, help="Maximum prunning amount")
    parser.add_argument("--minPruning", type=float, default=0.0, help="Maximum pruning amount")
    parser.add_argument("--pruningType", type=str, default="unstructured", help="type of pruning [unstructured, structured, adapter]")
    parser.add_argument("--put_lambda_max", action="store_true", help="put last pruned mask to 0.0483")




    args = parser.parse_args()
    return args