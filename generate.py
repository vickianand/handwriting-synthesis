import sys
import os
import argparse
import math
import pickle
import numpy as np
import torch

from utils import plot_stroke
from model import HandWritingRNN, HandWritingSynthRNN


def generate_from_model(
    model_name,
    model_path="data/unconditional_models/",
    sample_length=600,
    num_sample=2,
    device=torch.device("cpu"),
):
    """
    Generate num_sample (default 2) number of samples each of length 
    sample_length (default 300) using a pretrained model
    """
    model_file = model_path + model_name
    handWritingRNN = HandWritingRNN()
    handWritingRNN.load_state_dict(torch.load(model_file, map_location=device))
    handWritingRNN.to(device)
    generated_samples = handWritingRNN.generate(
        device=device, length=sample_length, batch=num_sample
    )

    for i in range(num_sample):
        plot_stroke(
            generated_samples[:, i, :].cpu().numpy(),
            save_name="data/samples/{}_{}.png".format(
                model_name.replace(".pt", ""), i),
        )


def generate_from_synth_model(
    model_name="None",
    model_path="data/conditional_models/",
    sentence_list=["Welcome to Lyrebird", "Hello World"],
    device=torch.device("cpu"),
):
    model = HandWritingSynthRNN()
    model_file = model_path + model_name
    if os.path.exists(model_file):
        model.load_state_dict(torch.load(model_file, map_location=device))
        # handWritingRNN.to(device)
    oh_encoder = pickle.load(open("data/one_hot_encoder.pkl", "rb"))
    sentences = oh_encoder.one_hot(sentence_list)
    generated_samples = model.generate(sentences=sentences, device=device)

    for i in range(len(sentences)):
        plot_stroke(
            generated_samples[:, i, :].cpu().numpy(),
            save_name="data/synth_samples/{}_{}.png".format(
                model_name.replace(".pt", ""), i
            ),
        )


def main():
    """ Generate samples from a list of unconditional models
    """
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--epoch_list",
        help="epoch numbers whose models to be used for generating samples",
        nargs="+",
        default=[93],
    )
    parser.add_argument("-sl", "--sample_length",
                        dest="sl", default=600, type=int)
    parser.add_argument("-ns", "--num_sample", dest="ns", default=2, type=int)

    args = vars(parser.parse_args())

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    if use_cuda:
        torch.cuda.empty_cache()

    np.random.seed(101)
    torch.random.manual_seed(101)

    # generate samples from some available trained models
    epoch_list = args["epoch_list"]
    for epoch in epoch_list:
        print("Sampling from epoch {} model.".format(epoch))
        generate_from_model(
            model_name="handwriting_uncond_ep{}.pt".format(epoch),
            device=device,
            sample_length=args["sl"],
            num_sample=args["ns"],
        )


if __name__ == "__main__":
    # main()
    generate_from_synth_model(model_name="handwriting_cond_ep0.pt")
