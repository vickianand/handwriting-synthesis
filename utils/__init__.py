import os
from collections import Counter

import numpy as np
from matplotlib import pyplot
import torch


def plot_stroke(stroke, save_name=None):
    # Plot a single example.
    f, ax = pyplot.subplots()

    x = np.cumsum(stroke[:, 1])
    y = np.cumsum(stroke[:, 2])

    size_x = x.max() - x.min() + 1.0
    size_y = y.max() - y.min() + 1.0

    f.set_size_inches(5.0 * size_x / size_y, 5.0)

    cuts = np.where(stroke[:, 0] == 1)[0]
    start = 0

    for cut_value in cuts:
        ax.plot(x[start:cut_value], y[start:cut_value], "k-", linewidth=3)
        start = cut_value + 1
    ax.axis("equal")
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)

    if save_name is None:
        pyplot.show()
    else:
        os.makedirs("/".join(save_name.split("/")[:-1]), exist_ok=True)
        try:
            pyplot.savefig(save_name, bbox_inches="tight", pad_inches=0.5)
        except Exception:
            print("Error building image!: " + save_name)

    # pyplot.close()
    return f


# ======================== normalize the strokes offset ========================


def normalize_data(strokes):
    """
    normalize to mean 0 and standard deviation 1
    """
    for i in range(strokes.shape[0]):
        strokes[i][:, 1] = strokes[i][:, 1] / strokes[i][:, 1].std()
        strokes[i][:, 2] = strokes[i][:, 2] / strokes[i][:, 2].std()

    return strokes


def count_parameters(model):
    """
    A simple function to count the number of parameters in the given model
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def filter_long_strokes(strokes, senences, stroke_len_threshold, max_index=None):
    """
    arguments:
        strokes: (np.ndarray)
        senences: (List[str]))
        stroke_len_threshold: (int)
        max_index: (Optional[int]) Index till which the data will be considered
    """
    assert strokes.size == len(senences)
    orig_count = strokes.size if max_index is None else max_index
    select_indices = np.array([s.shape[0] <= stroke_len_threshold for s in strokes])
    if max_index is not None:
        select_indices = np.array(
            [
                (s.shape[0] <= stroke_len_threshold and i < max_index)
                for i, s in enumerate(strokes)
            ]
        )
    strokes = strokes[select_indices]
    senences = list(np.array(senences)[select_indices])
    assert strokes.size == len(senences)
    final_count = strokes.size
    print(
        f"Filtered strokes longer than {stroke_len_threshold}: original count = "
        f"{orig_count}, final count = {final_count}"
    )
    max_sentence_len = max([len(s) for s in senences])
    return strokes, senences, max_sentence_len


class OneHotEncoder:
    def __init__(self, sentences, n_char=57):
        """ 
        Takes a list of setences and builds a dictionary (char_to_idx) for 
        char to integer-index mapping
        Arguments:
            sentences : list of strings (s)
            n_char : integer
        """
        self.n_char = n_char
        char_counts = Counter("".join(sentences))
        char_counts = sorted(char_counts.items(), key=lambda x: x[1], reverse=True)

        self.char_to_idx = {}
        self.idx_to_char = {}

        for i, kv in enumerate(char_counts):
            if i < n_char - 1:
                # for 56 most frequent characters, we give a unique id
                self.char_to_idx[kv[0]] = i
                self.idx_to_char[i] = kv[0]
            else:
                self.char_to_idx[kv[0]] = n_char - 1

        self.idx_to_char[n_char - 1] = "~"
        # print(self.char_to_idx, "\n\n", self.idx_to_char, "\n\n")

    def one_hot(self, sentences):
        """
        Takes a list of setences and returns a list of torch tensors of the one-hot  
        encoded (char level) forms of the sentences.
        Arguments:
            sentences : list of strings (s)
        returns:
            one_hot_ : list of torch tensors of shapes (len(s), n_char)
        """
        sentences_idx = [
            torch.tensor([[self.char_to_idx.get(c, self.n_char - 1)] for c in snt])
            for snt in sentences
        ]

        one_hot_ = [
            torch.zeros(idx_tnsr.shape[0], self.n_char).scatter_(
                dim=1, index=idx_tnsr, value=1.0
            )
            for idx_tnsr in sentences_idx
        ]
        return one_hot_


def plot_phi(phi_list):
    """
    phi_list: list of (len_seq number of) phi each of shape (B, n_char)
    """
    # print(f"len(phi_list): {len(phi_list)}, shape_item: {phi_list[0].shape}")
    fig_list = []
    for i in range(phi_list[0].shape[0]):
        single_list = [phi[i] for phi in phi_list]  # select phi one sequence from batch
        arr = torch.stack(single_list, dim=1).cpu().numpy()

        fig = pyplot.Figure()
        ax = fig.add_subplot(111)
        im = ax.imshow(
            arr, origin="lower", aspect="auto", interpolation="nearest", cmap="hot"
        )
        fig.colorbar(im)

        fig_list.append(fig)

    return fig_list


def plot_attn_scalar(sclr_list):
    """
    sclr_list: list of (len_seq number of) phi each of shape (B, K)
    """
    # print(f"len(sclr_list): {len(sclr_list)}, shape_item: {sclr_list[0].shape}")
    fig_list = []
    for i in range(sclr_list[0].shape[0]):
        single_list = [sclr[i] for sclr in sclr_list]
        arr = torch.stack(single_list, dim=1).cpu().numpy()

        fig = pyplot.Figure()
        ax = fig.add_subplot(111)
        for i in range(arr.shape[0]):
            ax.plot(arr[i], label="%d" % (i + 1))
        ax.legend()
        fig_list.append(fig)

    return fig_list
