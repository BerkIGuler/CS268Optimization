from argparse import ArgumentParser


def get_config_from_cli():
    parser = ArgumentParser()

    parser.add_argument(
        "--topology", "-t",
        type=str,
        help="which topology to use for DL",
        default="regular"
    )
    parser.add_argument(
        "--report_every_n", "-v",
        type=int,
        help="report test set result every n batch per node",
        default=50
    )
    parser.add_argument(
        "--batch_per_iter", "-b",
        type=int,
        help="how many batches to train before each communication round",
        default=3
    )
    parser.add_argument(
        "--degree", "-d",
        type=int,
        help="max degree per node",
        default=6
    )
    parser.add_argument(
        "--num_nodes", "-n",
        type=int,
        help="Number of nodes",
        default=50
    )
    parser.add_argument(
        "--rounds", "-r",
        type=int,
        help="Number of communication rounds",
        default=1000
    )


    return parser.parse_args()