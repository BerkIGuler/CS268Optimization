import matplotlib.pyplot as plt
from datetime import datetime
import json


def _get_file_name(args):
    file_name = ""
    for arg_name in vars(args):
        value = getattr(args, arg_name)
        file_name += (arg_name[:3] + str(value) + "_")
    return file_name[:-1]


def save_results(results, args):
    """saves accuracy and loss plots + accuracy and loss info across rounds in json"""
    save_file_suffix = _get_file_name(args)
    x = []
    mean_acc = []
    mean_loss = []
    for result in results:
        accs = []
        loss = []
        x.append(result["round"])
        for node in result:
            if node == "round":
                continue
            accs.append(result[node]["test accuracy"])
            loss.append(result[node]["average batch loss on test"])
        mean_acc.append(sum(accs) / len(accs))
        mean_loss.append(sum(loss) / len(loss))

    timestamp = str(datetime.now())
    plt.figure(figsize=(8, 8))
    plt.plot(x, mean_acc)
    plt.xlabel("Communication Round")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs. Communication Rounds")
    plt.grid()
    plt.savefig(f"results/accuracy_{save_file_suffix}_{timestamp}.png")

    plt.figure(figsize=(8, 8))
    plt.plot(x, mean_loss)
    plt.xlabel("Communication Round")
    plt.ylabel("Loss")
    plt.title("Loss vs. Communication Rounds")
    plt.grid()
    plt.savefig(f"results/loss_{save_file_suffix}_{timestamp}.png")

    with open(f"results/stats_{save_file_suffix}_{timestamp}.json", "w") as f_out:
        json.dump(results, f_out)
