from node import Node, combine
from models import NeuralNetwork
from dataset import Data
import time


batch_size = 64
num_nodes = 3
epochs = 1
max_training_iters = 500
init_with_same_weights = False
model = NeuralNetwork()


data = Data(batch_size=batch_size, num_nodes=num_nodes)
# split train data equally between nodes
train_dataloaders = data.partition_data()
total_train_dataloader, test_dataloader = data.total_data()

nodes = [Node(train_dataloaders[i], test_dataloader, model=model) for i in range(num_nodes)]

if init_with_same_weights:
    initial_sd = nodes[0].model.state_dict()

# local training
tic = time.time()
for i, node in enumerate(nodes):
    if i != 0 and init_with_same_weights:
        node.model.load_state_dict(initial_sd)
    # node.train(epochs)
    while node.total_num_iter < max_training_iters:
        node.train_n_iter(5)
    node.test(print_test=True)
dist_train_time = time.time() - tic

print(f"Total time for local trainings: {dist_train_time:.2f}s")

# average weights across nodes
node = Node(total_train_dataloader, test_dataloader, combine([node.model for node in nodes], target_model=model))
node.test(print_test=True)
