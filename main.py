from node import Node, DecentralizedNetwork
from models import NeuralNetwork
from topologies import get_weight_matrix
from dataset import Data
from plots import save_results
import time
from config import get_config_from_cli


args = get_config_from_cli()
batch_size = 32
num_nodes = args.num_nodes
weight_matrix = get_weight_matrix(num_nodes, args.topology, args.degree)
num_rounds = args.rounds

data = Data(batch_size=batch_size, num_nodes=num_nodes)
# iid split of train data
train_dataloaders = data.partition_data()
total_train_dataloader, test_dataloader = data.total_data()

nodes = [Node(train_dataloaders[i], test_dataloader, i, model=NeuralNetwork()) for i in range(num_nodes)]

# init models with the same weight
original_model_sd = nodes[0].model.state_dict()
for node in nodes:
    node.model.load_state_dict(original_model_sd)

tic = time.time()
dl_network = DecentralizedNetwork(nodes=nodes, weight_matrix=weight_matrix)
results = []
for curr_round in range(num_rounds):
    dl_network.train_local_nodes(
        batch_per_iter=args.batch_per_iter,
        report_every_n=args.report_every_n)
    dl_network.communicate()
    if curr_round % args.report_every_n == 0:
        stats = dl_network.evaluate(verbose=True)
        stats["round"] = curr_round + 1
        results.append(stats)

save_results(results, args)
print(f"Elapsed Time: {(time.time() - tic):.2f} seconds")
