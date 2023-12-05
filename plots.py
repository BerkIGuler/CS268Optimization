import matplotlib.pyplot as plt
from datetime import datetime


def plot_results(results):
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
    plt.savefig(f"results/accuracy_{timestamp}.png")

    plt.figure(figsize=(8, 8))
    plt.plot(x, mean_loss)
    plt.xlabel("Communication Round")
    plt.ylabel("Loss")
    plt.title("Loss vs. Communication Rounds")
    plt.grid()
    plt.savefig(f"results/loss_{timestamp}.png")


if __name__ == "__main__":
    sample_results = [{0: {'test accuracy': 0.3874, 'average batch loss on test': 2.2903778956721004},
                1: {'test accuracy': 0.3884, 'average batch loss on test': 2.290377706765367},
                2: {'test accuracy': 0.3885, 'average batch loss on test': 2.290378119617986},
                3: {'test accuracy': 0.388, 'average batch loss on test': 2.290378211786191},
                4: {'test accuracy': 0.388, 'average batch loss on test': 2.290377490436688},
                5: {'test accuracy': 0.3877, 'average batch loss on test': 2.290377848445417},
                6: {'test accuracy': 0.3873, 'average batch loss on test': 2.290377587175217},
                7: {'test accuracy': 0.3884, 'average batch loss on test': 2.2903779505159907},
                8: {'test accuracy': 0.3885, 'average batch loss on test': 2.2903788379206063},
                9: {'test accuracy': 0.3878, 'average batch loss on test': 2.290378412118735}, 'round': 300},
               {0: {'test accuracy': 0.4197, 'average batch loss on test': 2.2537697618380905},
                1: {'test accuracy': 0.4201, 'average batch loss on test': 2.253770278284725},
                2: {'test accuracy': 0.4199, 'average batch loss on test': 2.2537682513459423},
                3: {'test accuracy': 0.4206, 'average batch loss on test': 2.2537744928853582},
                4: {'test accuracy': 0.4212, 'average batch loss on test': 2.2537785216261406},
                5: {'test accuracy': 0.4212, 'average batch loss on test': 2.25377944635507},
                6: {'test accuracy': 0.4212, 'average batch loss on test': 2.2537855332651837},
                7: {'test accuracy': 0.4216, 'average batch loss on test': 2.2537848149625637},
                8: {'test accuracy': 0.4206, 'average batch loss on test': 2.2537749765780024},
                9: {'test accuracy': 0.4205, 'average batch loss on test': 2.2537764878318716}, 'round': 600},
               {0: {'test accuracy': 0.391, 'average batch loss on test': 2.0681791930152964},
                1: {'test accuracy': 0.3901, 'average batch loss on test': 2.0681634581507966},
                2: {'test accuracy': 0.39, 'average batch loss on test': 2.0681429233033057},
                3: {'test accuracy': 0.3897, 'average batch loss on test': 2.06813496256027},
                4: {'test accuracy': 0.3897, 'average batch loss on test': 2.0681125961553555},
                5: {'test accuracy': 0.3888, 'average batch loss on test': 2.0681007354023357},
                6: {'test accuracy': 0.3884, 'average batch loss on test': 2.068106619313883},
                7: {'test accuracy': 0.3891, 'average batch loss on test': 2.0681183079180245},
                8: {'test accuracy': 0.3889, 'average batch loss on test': 2.0681121829218756},
                9: {'test accuracy': 0.3892, 'average batch loss on test': 2.0681064262176854}, 'round': 900}]

    plot_results(sample_results)
