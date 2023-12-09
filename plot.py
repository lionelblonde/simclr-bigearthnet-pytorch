import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# hard-coded test values
test_results = {
    '0.1': {
        'classifier': {
            'recall': 0.153,
            'precision': 0.6496,
            'f1': 0.2476,
            'f2': 0.1806,
            'accuracy': 0.8589,
            'subset_accuracy': 0.1274,
            'balanced_accuracy': 0.5691,
        },
        'probe': {
            'recall': 0.47,
            'precision': 0.6379,
            'f1': 0.5413,
            'f2': 0.4961,
            'accuracy': 0.8791,
            'subset_accuracy': 0.1756,
            'balanced_accuracy': 0.7111,
        },
        'finetune': {
            'recall': 0.3757,
            'precision': 0.6947,
            'f1': 0.4877,
            'f2': 0.4137,
            'accuracy': 0.8802,
            'subset_accuracy': 0.1704,
            'balanced_accuracy': 0.6731,
        },
    },
    '1.0': {
        'classifier': {
            'recall': 0.5342,
            'precision': 0.7137,
            'f1': 0.6111,
            'f2': 0.5625,
            'accuracy': 0.8968,
            'subset_accuracy': 0.2282,
            'balanced_accuracy': 0.7479,
        },
        'probe': {
            'recall': 0.5309,
            'precision': 0.6935,
            'f1': 0.6014,
            'f2': 0.557,
            'accuracy': 0.8932,
            'subset_accuracy': 0.2145,
            'balanced_accuracy': 0.7445,
        },
        'finetune': {
            'recall': 0.5222,
            'precision': 0.7031,
            'f1': 0.5993,
            'f2': 0.5505,
            'accuracy': 0.894,
            'subset_accuracy': 0.2172,
            'balanced_accuracy': 0.7414,
        },
    },
    '10.0': {
        'classifier': {
            'recall': 0.6704,
            'precision': 0.7264,
            'f1': 0.6972,
            'f2': 0.6809,
            'accuracy': 0.9116,
            'subset_accuracy': 0.2698,
            'balanced_accuracy': 0.8126,
        },
        'probe': {
            'recall': 0.5616,
            'precision': 0.7488,
            'f1': 0.6418,
            'f2': 0.5911,
            'accuracy': 0.9048,
            'subset_accuracy': 0.2446,
            'balanced_accuracy': 0.7639,
        },
        'finetune': {
            'recall': 0.5558,
            'precision': 0.7538,
            'f1': 0.6398,
            'f2': 0.5866,
            'accuracy': 0.905,
            'subset_accuracy': 0.2449,
            'balanced_accuracy': 0.7616,
        },
    },
}

classifier100 = {
    'recall': 0.7407,
    'precision': 0.79,
    'f1': 0.7646,
    'f2': 0.7501,
    'accuracy': 0.9304,
    'subset_accuracy': 0.3256,
    'balanced_accuracy': 0.8526,
}

ms = ['recall', 'precision', 'f1',  # 'f2' can be ignored
      'accuracy', 'subset_accuracy', 'balanced_accuracy']
cs = ['classifier', 'probe', 'finetune']
colors = {
    'classifier': 'lightcoral',
    'probe': "springgreen",
    'finetune': "deepskyblue",
}
assert list(colors.keys()) == cs, "mismatching keys"
labels = [
    'Classifier',
    'SimCLR Frozen Trunk + Linear Trainable Head ("linear probe")',
    'SimCLR Trainable Trunk + Linear Trainable Head ("fine-tuning")',
]
n_rows, n_cols, unit_size = 2, 3, 5  # hps, careful
fig, axs = plt.subplots(
    n_rows, n_cols,
    figsize=(n_cols * unit_size,
             n_rows * unit_size),
)  # everything in one figure
percs = [float(k) for k in test_results.keys()]
lines = []  # will contain what's needed in the legend
hline_flag = True
for c in cs:
    i, j = 0, 0
    for k, m in enumerate(ms):
        axs[i, j].set_title('')
        axs[i, j].set_xscale("log")
        axs[i, j].xaxis.set_major_formatter(mtick.PercentFormatter())
        axs[i, j].set_ylim(ymin=0., ymax=1.)
        axs[i, j].set_xlabel("percentage of labels available")
        axs[i, j].set_ylabel(m.replace('_', ' '))
        axs[i, j].set_yticks(np.arange(0., 1., 0.1))

        line = axs[i, j].axhline(y=classifier100[m], color='grey')
        if k == 0 and hline_flag:
            lines.append(line)
            hline_flag = False

        line, = axs[i, j].plot(
            percs,
            [test_results[k][c][m]
             for k in test_results.keys()],
            marker='X',
            markersize=12,
            color=colors[c],
        )
        if k == 0:
            lines.append(line)  # for the legend
        j += 1
        if j >= n_cols:
            if i < (n_rows - 1):
                i += 1
                j = 0
            else:
                break
# handle the legend
fig.legend(lines, ['Classifier trained on every labelled pair', *labels])
fig.savefig("res.png")

# ok now we make one figure per percentage
n_rows, n_cols, unit_size = len(percs), 1, 5  # hps, careful
fig, axs = plt.subplots(
    n_rows, n_cols,
    figsize=(n_cols * unit_size * 3,  # eyeballed
             n_rows * unit_size),
)  # everything in one figure
index = np.arange(len(ms))
width = 0.25
lines = []
for k, p in enumerate(test_results.keys()):
    for u, (c, label) in enumerate(zip(cs, labels, strict=True)):
        axs[k].set_ylim(ymin=0., ymax=1.)
        line = axs[k].bar(
            index + (u * width),
            [test_results[p][c][m]
             for m in ms],
            width,
            label=label,
            color=colors[c],
        )
        if k == 0:
            lines.append(line)  # for the legend
    axs[k].set_title(f"only {p}% of labels are available", fontsize=20)
    axs[k].set_xticks(
        index + ((u * width) / 2),  # u contains its latest iterate
        [m.replace('_', ' ') for m in ms],
    )
    axs[k].set_yticks(np.arange(0., 1., 0.1))
    axs[k].grid(axis='y')  # easier to see the values
fig.legend(lines, labels, prop={'size': 12})
fig.savefig("res-bars.png")
