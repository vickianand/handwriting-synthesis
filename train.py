import sys
import os
import argparse
import math
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler

from utils import plot_stroke, normalize_data3, filter_long_strokes, OneHotEncoder
from model import HandWritingRNN, HandWritingSynthRNN

# ------------------------------------------------------------------------------


class HandWritingData(Dataset):
    """ Takes care of padding; So input is a list of tensors of different length
    """

    def __init__(self, sentences, strokes):
        assert len(sentences) == len(strokes)
        self.len = len(strokes)
        self.sentences = sentences
        self.pad_data(sentences, strokes)

    def pad_data(self, sentences, strokes):
        """
        input:
            strokes: list having N tensors of dimensions (*, d)
        output:
            padded_strokes: tensor of padded sequences of dimension (T, N, d) where
                T is the length of the longest tensor.
            masks: tensor of same dimension as strokes but having value 0 at
                positions where padding was done and value 1 at all other places
        """
        # first pad strokes and create masks corresponding to it
        self.padded_strokes = torch.nn.utils.rnn.pad_sequence(
            strokes, batch_first=True, padding_value=0.0
        )
        self.masks = self.padded_strokes.new_zeros(
            self.len, self.padded_strokes.shape[1]
        )
        for i, s in enumerate(strokes):
            self.masks[i, : s.shape[0]] = 1

        # now pad sentences
        self.padded_sentences = torch.nn.utils.rnn.pad_sequence(
            sentences, batch_first=True, padding_value=0.0
        )
        self.sentence_masks = self.padded_sentences.new_zeros(
            self.len, self.padded_sentences.shape[1]
        )
        for i, s in enumerate(sentences):
            self.sentence_masks[i, : s.shape[0]] = 1

        self.strokes_padded_len = self.padded_strokes.shape[1]
        self.sentences_padded_len = self.padded_sentences.shape[1]

        print(
            "Strokes are padded to length {}, and Sentences are padded to length {}".format(
                self.strokes_padded_len, self.sentences_padded_len
            )
        )

    def __getitem__(self, idx):
        return self.padded_sentences[idx], self.padded_strokes[idx], self.masks[idx]

    def __len__(self):
        return self.len


# ------------------------------------------------------------------------------


def mog_density_2d(x, pi, mu, sigma, rho):
    """
    Calculates The probability density of the next input x given the output vector
    as given in Eq. 23, 24 & 25 of the paper
    Expected dimensions of input:
        x : (n, 2)
        pi : (n , m)
        mu : (n , m, 2)
        sigma : (n , m, 2)
        rho : (n, m)
    """
    x_c = (x.unsqueeze(1) - mu) / sigma

    z = (x_c ** 2).sum(dim=2) - 2 * rho * x_c[:, :, 0] * x_c[:, :, 1]

    log_densities = (-z / 2 * (1 - rho ** 2)) - \
        (np.log(2 * math.pi) + sigma[:, :, 0].log() + sigma[:, :, 1].log() +
         0.5*(1 - rho ** 2).log()) + pi.log()
    # dimensions - log_densities : (n, m)

    # log_sum_exp trick for stability; return tensor of shape (n,)
    max_ld = log_densities.max(dim=1, keepdim=True)[0]
    log_densities = max_ld + (log_densities - max_ld).exp().sum().log()

    return log_densities


def criterion(x, e, pi, mu, sigma, rho, masks):
    """
    Calculates the sequence loss as given in Eq. 26 of the paper
    Expected dimensions of input:
        x: (n, b, 3)
        e: (n, b, 1)
        pi: (n, b, m)
        mu: (n, b, 2*m)
        sigma: (n, b, 2*m)
        rho: (n, b, m)
    Here n is the sequence length and m in number of components assumed for MoG
    """
    n, b, m = pi.shape  # n=sequence_length, b=batch_size, m=number_of_component_in_MoG

    x = x.contiguous().view(n * b, 3)
    e = e.view(n * b)
    e = e * x[:, 0] + (1 - e) * (1 - x[:, 0])  # e = (x0 == 1) ? e : (1 - e)

    x = x[:, 1:3]  # 2-dimensional offset values which is needed for MoG density

    pi = pi.view(n * b, m)  # change dimension to (n*b, m) from (n, b, m)
    mu = mu.view(n * b, m, 2)
    sigma = sigma.view(n * b, m, 2)
    rho = rho.view(n * b, m)

    """
        sigma2d = sigma.zeros(n, m, 4)
        sigma2d[:, :, 0, 0] = sigma[:, :, 0]**2
        sigma2d[:, :, 1, 1] = sigma[:, :, 1]**2
        sigma2d[:, :, 0, 1] = rho[:, :, ] * sigma[:, :, 0] * sigma[:, :, 1]
        sigma2d[:, :, 1, 0] = sigma2d[:, :, 0, 1]
    """
    # add small constant for numerical stability
    log_density = mog_density_2d(x, pi, mu, sigma, rho) + 1e-8

    masks = masks.view(n * b)
    ll = ((log_density + e.log()) * masks).mean()  # final log-likelihood
    # .mean() may be wrong because if we do .mean() the for different
    # length sequence denominator is always same
    return -ll


# ------------------------------------------------------------------------------


def train(device, batch_size, data_path="data/", uncond=False):
    """
    """
    random_seed = 42

    model_path = data_path + (
        "unconditional_models/" if uncond else "conditional_models/"
    )
    os.makedirs(model_path, exist_ok=True)

    strokes = np.load(data_path + "strokes.npy", encoding="latin1")
    sentences = ""
    with open(data_path + "sentences.txt") as f:
        sentences = f.readlines()
    sentences = [snt.replace("\n", "") for snt in sentences]
    # Instead of removing the newline symbols should it be used instead

    MAX_STROKE_LEN = 1000
    strokes, sentences, MAX_SENTENCE_LEN = filter_long_strokes(
        strokes, sentences, MAX_STROKE_LEN
    )
    # print("Max sentence len after filter is: {}".format(MAX_SENTENCE_LEN))
    N_CHAR = 57  # dimension of one-hot representation
    oh_encoder = OneHotEncoder(sentences, n_char=N_CHAR)
    pickle.dump(oh_encoder, open("data/one_hot_encoder.pkl", "wb"))
    sentences_oh = [s.to(device) for s in oh_encoder.one_hot(sentences)]

    # normalize strokes data and convert to pytorch tensors
    strokes = normalize_data3(strokes)
    # plot_stroke(strokes[sample_idx])
    tstrokes = [torch.from_numpy(stroke).to(device) for stroke in strokes]

    # pytorch dataset
    dataset = HandWritingData(sentences_oh, tstrokes)

    # validating the padding lengths
    assert dataset.strokes_padded_len == MAX_STROKE_LEN
    assert dataset.sentences_padded_len == MAX_SENTENCE_LEN

    # train - validation split
    train_split = 0.95
    train_size = int(train_split * len(dataset))
    validn_size = len(dataset) - train_size
    dataset_train, dataset_validn = torch.utils.data.random_split(
        dataset, [train_size, validn_size]
    )

    dataloader_train = DataLoader(
        dataset_train, batch_size=batch_size, shuffle=True, drop_last=False
    )  # last batch may be smaller than batch_size
    dataloader_validn = DataLoader(
        dataset_validn, batch_size=batch_size, shuffle=True, drop_last=False
    )

    model = (
        HandWritingRNN()
        if uncond
        else HandWritingSynthRNN(
            n_char=N_CHAR,
            max_stroke_len=MAX_STROKE_LEN,
            max_sentence_len=MAX_SENTENCE_LEN,
        )
    )

    if device != torch.device("cpu"):
        model = model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0)
    # optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-2,
    #                                   weight_decay=0, momentum=0)

    best_epoch_avg_loss = 100
    for epoch in range(100):

        train_losses = []
        validation_iters = []
        validation_losses = []
        for i, (c_seq, x, masks) in enumerate(dataloader_train):

            # make batch_first = false
            x = x.permute(1, 0, 2)
            # masks = masks.permute(1, 0, 2)

            # prepend a dummy point (zeros) and remove last point
            inp_x = torch.cat(
                [x.new_zeros(1, x.shape[1], x.shape[2]), x[:-1, :, :]], dim=0
            )

            inputs = (inp_x, c_seq)
            if uncond:
                inputs = (inp_x,)

            e, pi, mu, sigma, rho, *_ = model(*inputs)

            loss = criterion(x, e, pi, mu, sigma, rho, masks)
            train_losses.append(loss.detach().cpu().numpy())

            optimizer.zero_grad()

            loss.backward()

            # torch.nn.utils.clip_grad_norm_(model.parameters(), 100)

            optimizer.step()

            print("{},\t".format(loss))

        epoch_avg_loss = np.array(train_losses).mean()
        print("Average training-loss for epoch {} is: {}".format(epoch, epoch_avg_loss))

        # save model if loss is better than previous best
        if epoch_avg_loss < best_epoch_avg_loss:
            best_epoch_avg_loss = epoch_avg_loss
            model_file = model_path + "handwriting_{}cond_ep{}.pt".format(
                ("un" if uncond else ""), epoch
            )
            torch.save(model.state_dict(), model_file)

        # generate samples from model
        sample_count = 2
        sentences = ["Welcome to Lyrebird"] + \
            ["Hello World"] * (sample_count - 1)
        sentences = [s.to(device) for s in oh_encoder.one_hot(sentences)]
        generated_samples = (
            model.generate(length=600, batch=sample_count, device=device)
            if uncond
            else model.generate(sentences=sentences, device=device)
        )

        # save png files of the generated models
        for i in range(sample_count):
            plot_stroke(
                generated_samples[:, i, :].cpu().numpy(),
                save_name="data/training/{}cond_ep{}_{}.png".format(
                    ("un" if uncond else ""), epoch, i
                ),
            )


def main():

    parser = argparse.ArgumentParser(
        description="Train a handwriting generation model")
    parser.add_argument(
        "--uncond",
        help="If want to train the unconditional model",
        action="store_const",
        const=True,
        default=False,
    )
    parser.add_argument(
        "--batch_size", help="Batch size for training", type=int, default=64
    )

    args = parser.parse_args()

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    if use_cuda:
        torch.cuda.empty_cache()

    np.random.seed(101)
    torch.random.manual_seed(101)

    # training
    train(device=device, **vars(args))


if __name__ == "__main__":
    main()
