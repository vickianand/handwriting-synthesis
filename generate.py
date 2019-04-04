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
    model_path="data/model_files/handwriting_uncond_best.pt",
    sample_length=600,
    num_sample=2,
    bias=0.5,
    device=torch.device("cpu"),
):
    """
    Generate num_sample number of samples each of length sample_length using a 
    pretrained model
    """
    handWritingRNN = HandWritingRNN()
    handWritingRNN.load_state_dict(torch.load(model_path, map_location=device))
    generated_samples = handWritingRNN.generate(
        device=device, length=sample_length, batch=num_sample, bias=bias
    )

    model_name = model_path.split("/")[-1].replace(".pt", "")
    for i in range(num_sample):
        plot_stroke(
            generated_samples[:, i, :].cpu().numpy(),
            save_name="samples/{}_{}.png".format(model_name, i),
        )


def generate_from_synth_model(
    model_path="data/model_files/handwriting_cond_best.pt",
    sentence_list=[
        "hello world !!",
        "this text is generated using an RNN model",
        "Welcome to Lyrebird!",
    ],
    bias=3.0,
    device=torch.device("cpu"),
):
    model = HandWritingSynthRNN()
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    oh_encoder = pickle.load(open("data/one_hot_encoder.pkl", "rb"))
    sentences = [s.to(device) for s in oh_encoder.one_hot(sentence_list)]
    generated_samples, attn_vars = model.generate(
        sentences=sentences, bias=bias, device=device, use_stopping=True
    )

    model_name = model_path.split("/")[-1].replace(".pt", "")
    for i in range(len(sentence_list)):
        plot_stroke(
            generated_samples[:, i, :].cpu().numpy(),
            save_name="samples/{}_{}.png".format(model_name, i),
        )
        print(f"generated strokes for: {sentence_list[i]}")


def main():
    """ Generate samples from a list of unconditional models
    """
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--uncond",
        help="If want to generate using the unconditional model. Default is conditional",
        action="store_const",
        const=True,
        default=False,
    )
    parser.add_argument(
        "--model_path",
        required=True,
        help="path to the saved sate_dict file to be used for generating samples",
    )
    parser.add_argument(
        "--text",
        help="text for which handwriting to be synthesized (for conditional model)",
        nargs="+",
        default=["Hello world!"],
    )
    parser.add_argument(
        "--sample_length",
        default=600,
        type=int,
        help="sample length for unconditional model",
    )
    parser.add_argument(
        "--num_sample",
        default=5,
        type=int,
        help="number of samples to generate from unconditional model",
    )
    parser.add_argument("--seed", default=42, type=int)

    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(f"using seed = {args.seed}")
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    # generate samples from some available trained models
    print(f"Sampling from: {args.model_path}")
    if args.uncond:
        generate_from_model(
            model_path=args.model_path,
            device=device,
            sample_length=args.sample_length,
            num_sample=args.num_sample,
        )
    else:
        generate_from_synth_model(
            model_path=args.model_path, device=device, sentence_list=args.text
        )


if __name__ == "__main__":
    main()
    # generate_from_model()
    # generate_from_synth_model()
