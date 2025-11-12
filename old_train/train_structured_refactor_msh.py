
from opt import parse_args
from utils import seed_all
import os
import wandb

from experiment import Experiment

def main():
    log_wandb = True
    args = parse_args()
    project_run_path=os.path.dirname(__file__)

    if not args.mask:
        print("Mask is 0 or False — exiting program.")
        return
    
    if args.seed is not None:
        seed_all(args.seed)
        args.save_dir = f'{args.save_dir}_seed_{args.seed}'

    if log_wandb:
        wandb.init(
            project='training',
            entity='MagV',
            name=f'{args.nameRun}',
            config=vars(args)
        )

    exp = Experiment(args,project_run_path)

    # RD before training
    exp.make_baseline_plot()

    for epoch in range(exp.ctx.last_epoch, args.epochs):
        
        #train 
        exp.train(epoch)

        # test 
        exp.validate(epoch,exp.ctx.val_dataloader,'val')

        # kodak
        exp.validate(epoch,exp.ctx.kodak_dataloader,'kodak')

        if epoch%5==0:
            exp.make_plot(epoch)

    if log_wandb:
        wandb.run.finish()
        
if __name__ == "__main__":
    main()