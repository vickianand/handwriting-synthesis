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
    # ax.axes.get_xaxis().set_visible(False)
    # ax.axes.get_yaxis().set_visible(False)

    if save_name is None:
        pyplot.show()
    else:
        os.makedirs("/".join(save_name.split("/")[:-1]), exist_ok=True)
        try:
            pyplot.savefig(save_name, bbox_inches="tight", pad_inches=0.5)
        except Exception:
            print("Error building image!: " + save_name)

    pyplot.close()


def log_stroke(stroke, writer, epoch):
    """
    Logging the generated image into TensorBoardX SummaryWriter()
    """
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
    # ax.axes.get_xaxis().set_visible(False)
    # ax.axes.get_yaxis().set_visible(False)

    f.canvas.draw()

    # Now we can save it to a numpy array.
    data = np.fromstring(f.canvas.tostring_rgb(), dtype=np.uint8, sep="")
    data = data.reshape(f.canvas.get_width_height()[::-1] + (3,))
    writer.add_image("Sample Image", data, epoch)


# ======================== normalize the strokes offset ========================


def normalize_data1(strokes):
    """
    normalize to [-1, 1] separately for x1 and x2
    """
    strokes_cat = np.concatenate(list(strokes))

    (min1, max1) = strokes_cat[:, 1].min(), strokes_cat[:, 1].max()
    (min2, max2) = strokes_cat[:, 2].min(), strokes_cat[:, 2].max()
    range1 = max1 - min1
    range2 = max2 - min2
    for i in range(strokes.shape[0]):
        strokes[i][:, 1] = 2 * (strokes[i][:, 1] - (min1 + range1 / 2.0)) / range1
        strokes[i][:, 2] = 2 * (strokes[i][:, 2] - (min2 + range2 / 2.0)) / range2

    return strokes


def normalize_data2(strokes):
    """
    normalize to [-1, 1] range using combined min and max of x1 and x2
    """
    strokes_cat = np.concatenate(list(strokes))

    (min1, max1) = strokes_cat[:, 1:].min(), strokes_cat[:, 1:].max()
    # print("min1 = {}, max1 = {}".format(min1, max1))
    range1 = max1 - min1
    for i in range(strokes.shape[0]):
        strokes[i][:, 1:] = 2 * (strokes[i][:, 1:] - (min1 + range1 / 2.0)) / range1

    return strokes


def normalize_data3(strokes):
    """
    normalize to mean 0 and standard deviation 1
    """
    strokes_cat = np.concatenate(list(strokes))

    (mean, std) = strokes_cat[:, 1:].mean(), strokes_cat[:, 1:].std()

    # (mean1, std1) = strokes_cat[:, 1].mean(), strokes_cat[:, 1].std()
    # (mean2, std2) = strokes_cat[:, 2].mean(), strokes_cat[:, 2].std()

    for i in range(strokes.shape[0]):
        # strokes[i][:, 1:] = (strokes[i][:, 1:] - mean) / std
        strokes[i][:, 1:] = strokes[i][:, 1:] / std

        # strokes[i][:, 1] = (strokes[i][:, 1] - mean1) / std1
        # strokes[i][:, 2] = (strokes[i][:, 2] - mean2) / std2

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
    orig_count = strokes.size
    select_indices = np.array([s.shape[0] <= stroke_len_threshold for s in strokes])
    if max_index is not None:
        select_indices = np.array(
            [
                (s.shape[0] <= stroke_len_threshold and i < 100)
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

        for i in range(n_char - 1):
            # for 56 most frequent characters, we give a unique id
            self.char_to_idx[char_counts[i][0]] = i

        for kv in char_counts[n_char - 1 :]:
            self.char_to_idx[kv[0]] = n_char - 1

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
            torch.tensor([[self.char_to_idx[c]] for c in snt]) for snt in sentences
        ]

        one_hot_ = [
            torch.zeros(idx_tnsr.shape[0], self.n_char).scatter_(
                dim=1, index=idx_tnsr, value=1.0
            )
            for idx_tnsr in sentences_idx
        ]
        return one_hot_
