from argparse import ArgumentParser


def get_config_from_cli():
    parser = ArgumentParser()

    parser.add_argument(
        "--topology", "-t",
        type=str,
        help="which topology to use for DL",
        default="fc_equal_weights"
    )

    parser.add_argument(
        "--report_every_n", "-r",
        type=int,
        help="report test set result every n batch per node",
        default=300
    )
    parser.add_argument(
        "--batch_per_iter", "-b",
        type=int,
        help="how many batches to train before each communication round",
        default=5
    )

    return parser.parse_args()