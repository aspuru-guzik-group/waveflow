import numpy as np
import matplotlib. pyplot as plt
import os

root_path = './results/pdf'

color_dict = {'Flow': 'r', 'IFlow': 'b', 'MFlow': 'g'}
style_dict = {'0': '-', '01': '--', '1': '-.'}
label_dict = {'0': '0.0', '01': '0.1', '1': '1.0'}

experiments = os.listdir(root_path)
burn_in = 10000
for experiment in experiments:
    experiment_path = '{}/{}'.format(root_path, experiment)

    losses_dict = {}
    kl_dict = {}
    hellinger_dict = {}
    models = os.listdir(experiment_path)
    for model in models:
        model_path = '{}/{}'.format(experiment_path, model)
        if os.path.isfile(model_path):
            continue
        losses = np.loadtxt('{}/losses.txt'.format(model_path))
        kl_divergences = np.loadtxt('{}/kl_divergences.txt'.format(model_path))
        hellinger_divergences = np.loadtxt('{}/hellinger_divergences.txt'.format(model_path))

        losses_dict[model] = losses
        kl_dict[model] = kl_divergences
        hellinger_dict[model] = hellinger_divergences

    fig, ax = plt.subplots()
    for key, val in losses_dict.items():
        keys = key.split('_')
        linestyle = style_dict[keys[1]] if len(keys) == 2 else '-'
        label = '{} {}'.format(keys[0], label_dict[keys[1]]) if len(keys) == 2 else 'Flow'
        ax.plot(np.clip(val[burn_in:], a_min=None, a_max=val[burn_in]), color=color_dict[keys[0]], linestyle=linestyle, label=label)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.grid(True)
    ax.legend(loc='best')
    plt.savefig('{}/losses.png'.format(experiment_path))
    # plt.show()


    fig, ax = plt.subplots()
    for key, val in kl_dict.items():
        keys = key.split('_')
        linestyle = style_dict[keys[1]] if len(keys) == 2 else '-'
        label = '{} {}'.format(keys[0], label_dict[keys[1]]) if len(keys) == 2 else 'Flow'
        ax.plot(val, color=color_dict[keys[0]], linestyle=linestyle, label=label)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('KL Divergence')
    ax.grid(True)
    ax.legend(loc='best')
    plt.savefig('{}/kl_divergences.png'.format(experiment_path))
    # plt.show()

    fig, ax = plt.subplots()
    for key, val in hellinger_dict.items():
        keys = key.split('_')
        linestyle = style_dict[keys[1]] if len(keys) == 2 else '-'
        label = '{} {}'.format(keys[0], label_dict[keys[1]]) if len(keys) == 2 else 'Flow'
        ax.plot(val, color=color_dict[keys[0]], linestyle=linestyle, label=label)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Squared Hellinger Divergence')
    ax.grid(True)
    ax.legend(loc='best')
    plt.savefig('{}/hellinger_divergences.png'.format(experiment_path))
    # plt.show()


