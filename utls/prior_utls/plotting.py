import matplotlib.pyplot as plt
import torch

def compare_statistics(stat_dicts, labels):
    first_dict = stat_dicts[0]
    for layer_key in first_dict.keys():
        for type_key in first_dict[layer_key].keys():
            for stat_key in first_dict[layer_key][type_key].keys():
                stat_name = ' '.join([layer_key, type_key, stat_key])
                plt.figure(figsize=(12,6))
                plt.title(stat_name)
                for stat_dict, label in zip(stat_dicts, labels):
                    stats = (stat_dict[layer_key][type_key][stat_key]).cpu().flatten()
                    plt.plot(stats, label=label)                    
                plt.legend()
                plt.show()

def plot_losses(losses):
    plt.figure()
    plt.plot(losses)
    plt.xlabel('Synthesis Iteration')
    plt.ylabel('Loss')
    plt.grid()
    plt.show()

    plt.figure()
    plt.plot(torch.log(losses))
    plt.xlabel('Synthesis Iteration')
    plt.ylabel('Log Loss')
    plt.grid()
    plt.show()