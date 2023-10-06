from typing import Optional

import argparse


def boolean_flag(
    parser: argparse.ArgumentParser,
    name: str,
    default: Optional[bool] = False,
    help: Optional[str] = None,  # noqa
):
    """Add a boolean flag to argparse parser."""
    dest = name.replace('-', '_')
    parser.add_argument(
        "--" + name, action="store_true", default=default, dest=dest, help=help,
    )
    parser.add_argument(
        "--no-" + name, action="store_false", dest=dest,
    )


def agg_argparser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # meta
    parser.add_argument(
        "--seed", type=int, default=0,
        help="RNG seed for reproducibility",
    )
    parser.add_argument(
        "--uuid", type=str, default=None,
    )
    # resources
    boolean_flag(
        parser, "cuda", default=True,
        help="whether to use gpu(s) or cpu(s)",
    )
    boolean_flag(
        parser, "fp16", default=False,
        help="whether to use fp16 precision",
    )
    # logging
    parser.add_argument(
        "--wandb_project", default='DEFAULT',
        help="wandb project name",
    )
    # dataset
    parser.add_argument(
        "--dataset_handle", type=str, default=None,
    )
    parser.add_argument(
        "--data_path", type=str,
        help="path to folders with images to train on.",
    )
    parser.add_argument(
        "--truncate_at", type=float, default=100.,
        help="amount of data to keep in %",
    )
    parser.add_argument(
        "--num_workers", type=int, default=0,
        help="parallel workers in dataloader; 0 means no parallelism",
    )
    # training
    parser.add_argument(
        "--epochs", type=int, default=100,
        help="number of epochs to train model",
    )
    parser.add_argument(
        "--batch_size", type=int, default=128,
        help="batch size for SSL",
    )
    parser.add_argument(
        "--save_freq", type=int, default=1,
    )
    parser.add_argument(
        "--eval_every", type=int, default=100,
    )
    # opt
    parser.add_argument(
        "--lr", type=float, default=3e-4,
    )
    parser.add_argument(
        "--wd", type=float, default=1e-6,
    )
    parser.add_argument(
        "--clip_norm", type=float, default=60.,
    )
    parser.add_argument(
        "--acc_grad_steps", type=int, default=8,
    )
    boolean_flag(
        parser, "lars", default=False,
        help="whether to use layerwise lr adaption",
    )
    boolean_flag(
        parser, "sched", default=False,
        help="whether to use lr scheduler",
    )
    # algo
    parser.add_argument(
        "--algo_handle", type=str,
        choices=['classifier', 'simclr'],
        default=None,
    )

    # CLASSIFIER

    parser.add_argument(
        "--backbone", type=str, default=None,
    )
    boolean_flag(
        parser, "pretrained_w_imagenet", default=False,
    )
    parser.add_argument(
        "--fc_hid_dim", type=int, default=128,
    )
    parser.add_argument(
        "--num_classes", type=int, default=19,
    )

    # SIMCLR

    parser.add_argument(
        "--fc_out_dim", type=int, default=64,
    )
    parser.add_argument(
        "--ntx_temp", type=float, default=0.07,
    )
    parser.add_argument(
        "--load_checkpoint", type=str, default=None,
    )
    boolean_flag(
        parser, "linear_probe", default=False,
    )
    boolean_flag(
        parser, "fine_tuning", default=False,
    )
    parser.add_argument(
        "--ftop_epochs", type=int, default=10,
    )  # same as SimCLR
    parser.add_argument(
        "--ftop_batch_size", type=int, default=128,
    )

    return parser
