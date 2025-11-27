from opt import parse_args
from utils import seed_all
import os
import wandb

from experiment import Experiment

def main():
    log_wandb = True
    args = parse_args()
    project_run_path=os.path.dirname(__file__)

    # security to not waste run
    # if not args.mask and args.pruningType!="adapter":
    #     print("Mask is 0 or False, not adapter run — exiting program.")
    #     return
    
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
    if args.mask :
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

        # save model and mask for the last epoch in order to use the simplify lib
        # if args.mask and epoch==args.epoch -1:


    if log_wandb:
        wandb.run.finish()
        
if __name__ == "__main__":
    main()