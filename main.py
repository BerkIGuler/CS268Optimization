from node import Node, DecentralizedNetwork
from models import NeuralNetwork
from topologies import get_weight_matrix
from dataset import Data
import time
from config import get_config_from_cli


args = get_config_from_cli()
batch_size = 32
num_nodes = 10
weight_matrix = get_weight_matrix(num_nodes, args.topology)
num_rounds = 9000


data = Data(batch_size=batch_size, num_nodes=num_nodes)
# iid split of train data
train_dataloaders = data.partition_data()
total_train_dataloader, test_dataloader = data.total_data()

nodes = [Node(train_dataloaders[i], test_dataloader, i, model=NeuralNetwork()) for i in range(num_nodes)]

tic = time.time()
dl_network = DecentralizedNetwork(nodes=nodes, weight_matrix=weight_matrix)
for curr_round in range(num_rounds):
    dl_network.train_local_nodes(
        batch_per_iter=args.batch_per_iter,
        report_every_n=args.report_every_n)
    dl_network.communicate()
    if (curr_round + 1) % args.report_every_n == 0:
        stats = dl_network.evaluate(verbose=True)


print(f"Elapsed Time: {(time.time() - tic):.2f} seconds")
