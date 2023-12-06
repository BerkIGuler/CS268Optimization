from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from torchvision.transforms import ToTensor, Normalize, Compose


class Data:
    def __init__(self, batch_size=64, num_nodes=3):
        transform = Compose(
            [
                ToTensor(),
                Normalize(mean=(0.1307,), std=(0.3081,))
            ]
        )
        # Download training data from open datasets.
        self.training_data = datasets.MNIST(
            root="data",
            train=True,
            download=True,
            transform=transform,
        )

        # Download test data from open datasets.
        self.test_data = datasets.MNIST(
            root="data",
            train=False,
            download=True,
            transform=transform,
        )

        self.batch_size = batch_size
        self.num_nodes = num_nodes

    def partition_data(self):
        n, b_size = self.num_nodes, self.batch_size
        train_par = random_split(
            self.training_data,
            self._get_partition_sizes(self.training_data, self.num_nodes)
        )

        train = [DataLoader(data, batch_size=b_size) for data in train_par]
        return train

    @staticmethod
    def _get_partition_sizes(training_data, num_nodes):
        num_data = len(training_data)
        sizes = [int(num_data / num_nodes) for _ in range(num_nodes - 1)]
        sizes.append(num_data - sum(sizes))
        assert sum(sizes) == num_data, "partition error, size mismatch"
        return sizes

    def total_data(self):
        train = DataLoader(self.training_data, batch_size=self.batch_size)
        test = DataLoader(self.test_data, batch_size=self.batch_size)

        return train, test