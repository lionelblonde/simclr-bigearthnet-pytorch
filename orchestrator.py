import time
import os
import os.path as osp
import signal

import wandb
import numpy as np

from helpers import logger
from helpers.console_util import timed_cm_wrapper, log_epoch_info
from helpers.dataloader_utils.bigearthnet_utils.dataloader import get_dataloader
from helpers.dataloader_utils.bigearthnet_utils.splitter import split_datasets


debug_lvl = os.environ.get('DEBUG_LVL', 0)
try:
    debug_lvl = np.clip(int(debug_lvl), a_min=0, a_max=3)
except ValueError:
    debug_lvl = 0
DEBUG = bool(debug_lvl >= 1)


def learn(
    args,
    algo_wrapper,
    experiment_name,
    num_transforms,
    with_labels,
):

    # Create context manager that records the time taken by encapsulated ops
    timed = timed_cm_wrapper(logger, use=DEBUG)

    with timed("splitting"):
        paths_dict = split_datasets(
            data_path=args.data_path,
            val_split=args.val_split,
            test_split=args.test_split,
            seed=args.seed,
            truncate_at=args.truncate_at,
        )
    print(paths_dict)

    with timed("dataloading"):
        dataloaders = {}
        tmpdict = {
            "data_path": args.data_path,
            "dataset_handle": args.dataset_handle,
            "batch_size": args.batch_size,
            "num_transforms": num_transforms,
            "with_labels": with_labels,
        }
        # Create the dataloaders
        dataloaders['train'] = get_dataloader(
            **tmpdict,
            split_path=paths_dict["train"],
            train_stage=True,
            shuffle=True
        )
        dataloaders['val'] = get_dataloader(
            **tmpdict,
            split_path=paths_dict["val"],
            val_stage=True,
            shuffle=False
        )
        dataloaders['test'] = get_dataloader(
            **tmpdict,
            split_path=paths_dict["test"],
            test_or_inference_stage=True,
            shuffle=False
        )

        for k, v in dataloaders.items():
            # Log stats about the dataloaders
            ds_len = v.dataset_length
            dl_len = len(v)
            logger.info(f"({k} dataloader) {ds_len = } | {dl_len = }")

        if args.linear_probe or args.fine_tuning:
            finetune_probe_dataloaders = {}
            tmpdict = {
                "data_path": args.data_path,
                "dataset_handle": args.dataset_handle,
                "batch_size": args.finetune_probe_batch_size,
                "num_transforms": 1,
                "with_labels": with_labels,
            }
            # Create the dataloaders
            finetune_probe_dataloaders['train'] = get_dataloader(
                **tmpdict,
                split_path=paths_dict["train"],
                train_stage=True,
            )
            finetune_probe_dataloaders['val'] = get_dataloader(
                **tmpdict,
                split_path=paths_dict["val"],
                val_stage=True,
            )
            finetune_probe_dataloaders['test'] = get_dataloader(
                **tmpdict,
                split_path=paths_dict["test"],
                test_or_inference_stage=True,
            )

            for k, v in finetune_probe_dataloaders.items():
                # Log stats about the dataloaders
                ds_len = v.dataset_length
                dl_len = len(v)
                logger.info(f"({k} dataloader) {ds_len = } | {dl_len = }")

    # Create an algorithm
    algo = algo_wrapper()

    if not hasattr(algo, "scheduler"):
        # In the case of compression, we use an exotic lr scheduler (OneCycleLR)
        # but we first needed more information about the dataset/dataloaders
        algo.set_scheduler(len(dataloaders['train']))

    tstart = time.time()

    # Set up model save directory
    ckpt_dir = osp.join(args.checkpoint_dir, experiment_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    # Save the model as a dry run, to avoid bad surprises at the end
    algo.save(ckpt_dir, f"{algo.epochs_so_far}_dryrun")
    logger.info(f"dry run. Saving model @: {ckpt_dir}")

    # Handle timeout signal gracefully
    def timeout(signum, frame):
        # Save the model
        algo.save(ckpt_dir, f"{algo.epochs_so_far}_timeout")
        # No need to log a message, orterun stopped the trace already
        # No need to end the run by hand, SIGKILL is sent by orterun fast enough after SIGTERM

    # Tie the timeout handler with the termination signal
    # Note, orterun relays SIGTERM and SIGINT to the workers as SIGTERM signals,
    # quickly followed by a SIGKILL signal (Open-MPI impl)
    signal.signal(signal.SIGTERM, timeout)

    # Group by everything except the seed, which is last, hence index -1
    # For 'gail', it groups by uuid + gitSHA + env_id + num_demos,
    # while for 'ppo', it groups by uuid + gitSHA + env_id
    group = '.'.join(experiment_name.split('.')[:-1])

    # Set up wandb
    while True:
        try:
            wandb.init(
                project=args.wandb_project,
                name=experiment_name,
                id=experiment_name,
                group=group,
                config=args.__dict__,
                dir=args.root,
            )
            break
        except Exception:
            pause = 10
            logger.info(f"wandb co error. Retrying in {pause} secs.")
            time.sleep(pause)
    logger.info("wandb co established!")

    while algo.epochs_so_far <= args.epochs:

        log_epoch_info(logger, algo.epochs_so_far, args.epochs, tstart)

        algo.train(dataloaders['train'], dataloaders['val'])

        if algo.epochs_so_far % args.save_freq == 0:
            algo.save(ckpt_dir, algo.epochs_so_far)

    logger.info("testing")
    algo.test(dataloaders['test'])

    if algo.epochs_so_far > 0:
        # Save once we are done, unless we have not done a single epoch of training
        algo.save(ckpt_dir, f"{algo.epochs_so_far}_done")
        logger.info(f"we're done. Saving model @: {ckpt_dir}")
        logger.info("bye.")

    if args.linear_probe or args.fine_tuning:

        logger.info("we are now either fine-tuning or linear probing.")

        tstart = time.time()

        algo.renew_head()  # also resets the epoch counter!

        while algo.epochs_so_far <= args.finetune_probe_epochs:

            log_epoch_info(logger, algo.epochs_so_far, args.finetune_probe_epochs, tstart)

            algo.finetune_or_train_probe(finetune_probe_dataloaders['train'], finetune_probe_dataloaders['val'])

        logger.info("testing")
        algo.test_finetuned_or_probed_model(finetune_probe_dataloaders['test'])

        # Save once we are done
        algo.save(ckpt_dir, f"{algo.epochs_so_far}_with_new_head_done")
        logger.info(f"we're done. Saving model @: {ckpt_dir}")
        logger.info("bye.")

        logger.info("now we are really done. bye.")
