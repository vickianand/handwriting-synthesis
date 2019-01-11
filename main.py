import sys
import os
import math
import numpy as np
from matplotlib import pyplot
import torch
from torch.utils.data import DataLoader, Dataset

from utils import plot_stroke, normalize_data3
from model import HandWritingRNN

# ------------------------------------------------------------------------------


def mog_density_2d(x, pi, mu, sigma, rho):
    '''
    Calculates The probability density of the next input x given the output vector
    as given in Eq. 23, 24 & 25 of the paper
    Expected dimensions of input:
        x : (n, 2)
        pi : (n , m)
        mu : (n , m, 2)
        sigma : (n , m, 2)
        rho : (n, m)
    '''
    m = pi.shape[1]
    x_c = mu.new_zeros(mu.shape)   # x (_centered) is of shape : (n, m, 2)
    for i in range(m):
        x_c[:, i, :] = x - mu[:, i, :]

    z = x_c[:, :, 0]**2 / sigma[:, :, 0]**2 \
        + x_c[:, :, 1]**2 / sigma[:, :, 1]**2 \
        - 2*rho*x_c[:, :, 0]*x_c[:, :, 1] / (sigma[:, :, 0] * sigma[:, :, 1])

    densities = torch.exp(-z / 2*(1-rho**2)) \
            / (2 * math.pi * sigma[:, :, 0] * sigma[:, :, 1] * torch.sqrt(1 - rho**2))
    
    # dimensions - densities : (n, m); pi : (n, m)
    # aggregate along dimension 1, by weighing with pi and return tensor of shape (n)
    return (pi * densities).sum(dim=1)


def criterion(x, e, pi, mu, sigma, rho, masks):
    '''
    Calculates the sequence loss as given in Eq. 26 of the paper
    Expected dimensions of input:
        x: (n, b, 3)
        e: (n, b, 1)
        pi: (n, b, m)
        mu: (n, b, 2*m)
        sigma: (n, b, 2*m)
        rho: (n, b, m)
    Here n is the sequence length and m in number of components assumed for MoG
    '''
    n, b, m = pi.shape

    x = x.contiguous().view(n*b, 3)
    e = e.view(n*b)
    e = e * x[:, 0] + (1-e)*(1 - x[:, 0])    # e = (x0 == 1) ? e : (1 - e)
    
    x = x[:, 1:3]   # 2-dimensional offset values which is needed for MoG density
    
    pi = pi.view(n*b, m) # change dimension to (n*b, m) from (n, b, m)
    mu = mu.view(n*b, m, 2)
    sigma = sigma.view(n*b, m, 2)
    rho = rho.view(n*b, m)
    
    '''
        sigma2d = sigma.zeros(n, m, 4)
        sigma2d[:, :, 0, 0] = sigma[:, :, 0]**2
        sigma2d[:, :, 1, 1] = sigma[:, :, 1]**2
        sigma2d[:, :, 0, 1] = rho[:, :, ] * sigma[:, :, 0] * sigma[:, :, 1]
        sigma2d[:, :, 1, 0] = sigma2d[:, :, 0, 1]
    '''
    # add small constant for numerical stability
    density = mog_density_2d(x, pi, mu, sigma, rho) + 1e-8

    masks = masks.view(n*b)
    ll = ((torch.log(density) + torch.log(e)) * masks).mean() # final log-likelihood
    return -ll

# ------------------------------------------------------------------------------


class HandWritingData(Dataset):
    ''' Takes care of padding; So input is a list of tensors of different length
    '''
    def __init__(self, strokes):
        self.len = len(strokes)
        self.pad_data(strokes)

    def pad_data(self, strokes):
        '''
        input:
            strokes: list having N tensors of dimensions (*, d)
        output:
            padded_strokes: tensor of padded sequences of dimension (T, N, d) where 
                T is the length of the longest tensor.
            masks: tensor of same dimension as strokes but having value 0 at 
                positions where padding was done and value 1 at all other places
        '''
        self.padded_strokes = torch.nn.utils.rnn.pad_sequence(strokes, 
                                batch_first=True, padding_value=0.)

        self.masks = self.padded_strokes.new_zeros(self.len, 
                        self.padded_strokes[0].shape[0])
        for i, s in enumerate(strokes):
            self.masks[i, :s.shape[0]] = 1

    def __getitem__(self, idx):
        return self.padded_strokes[idx], self.masks[idx]

    def __len__(self):
        return self.len

# ------------------------------------------------------------------------------


def train(device, data_path = "data/", batch_size = 128):
    '''
    '''
    model_path = data_path + "unconditional_models/"
    os.makedirs(model_path, exist_ok=True)

    strokes = np.load(data_path + "strokes.npy", encoding='latin1')
    sentences = ""
    with open(data_path + "sentences.txt") as f:
        sentences = f.readlines()

    # training, validation split
    train_size = int(0.9 * len(strokes))
    validn_size = len(strokes) - train_size

    strokes_validn = strokes[train_size:]
    sentences_validn = sentences[train_size:]
    strokes = strokes[:train_size]
    sentences = sentences[:train_size]

    sample_idx = 25
    # plot_stroke(strokes[sample_idx])
    # print(sentences[sample_idx])
    
    strokes = normalize_data3(strokes)
    # plot_stroke(strokes[sample_idx])
    
    handWritingRNN = HandWritingRNN()
    if device != torch.device('cpu'):
      handWritingRNN = handWritingRNN.cuda()
    
    # generated_samples = handWritingRNN.random_sample(count=2)

    optimizer = torch.optim.Adam(handWritingRNN.parameters(), lr=1e-3, 
                                    weight_decay=0)
    # optimizer = torch.optim.RMSprop(handWritingRNN.parameters(), lr=1e-2, 
    #                                   weight_decay=0, momentum=0)
    
    tstrokes = [torch.from_numpy(stroke).to(device) for stroke in strokes]
    handWritingData = HandWritingData(tstrokes)
    dataloader = DataLoader(handWritingData, batch_size=batch_size, shuffle=True,
                    drop_last=False) # last batch may be smaller than batch_size

    tstrokes_validn = [torch.from_numpy(stroke).to(device) for stroke in strokes_validn]
    handWritingData_validn = HandWritingData(tstrokes_validn)
    dataloader_validn = DataLoader(handWritingData_validn, 
                            batch_size=batch_size, shuffle=True, drop_last=False)

    best_epoch_avg_loss = 100
    for epoch in range(100):

        train_losses = []
        validation_iters = []
        validation_losses = []
        for i, (x, masks) in enumerate(dataloader):
            
            # make batch_first = false
            x = x.permute(1, 0, 2)
            # masks = masks.permute(1, 0, 2)

            # prepend a dummy point (zeros) and remove last point
            inp_x = torch.cat([x.new_zeros(1, x.shape[1], x.shape[2]), 
                                x[:-1, :, :]], dim=0)

            e, pi, mu, sigma, rho, _ = handWritingRNN(inp_x)

            loss = criterion(x, e, pi, mu, sigma, rho, masks)
            train_losses.append(loss.detach().cpu().numpy())

            optimizer.zero_grad()

            loss.backward()
            
            # torch.nn.utils.clip_grad_norm_(handWritingRNN.parameters(), 100)
            
            optimizer.step()
            
            print("{},\t".format(loss),)

        epoch_avg_loss = np.array(train_losses).mean()
        print("Average training-loss for epoch {} is: {}".format(epoch, 
                epoch_avg_loss))
        
        if(epoch_avg_loss < best_epoch_avg_loss):
          best_epoch_avg_loss = epoch_avg_loss
          model_file = model_path + "handwriting_uncond_ep{}.pt".format(epoch)
          torch.save(handWritingRNN.state_dict(), model_file)
        
        generated_samples = handWritingRNN.random_sample(length=600, count=2, 
                                device=device)

        plot_stroke(generated_samples[:, 0, :].cpu().numpy(), 
                save_name="data/training/uncond_ep{}_1.png".format(epoch))
        plot_stroke(generated_samples[:, 1, :].cpu().numpy(),
                save_name="data/training/uncond_ep{}_2.png".format(epoch))


def generate_from_model(model_name, model_path="data/unconditional_models/", 
        sample_length=600, num_sample=2, device=torch.device("cpu")):
    '''
    Generate num_sample (default 2) number of samples each of length 
    sample_length (default 300) using a pretrained model
    '''
    model_file = model_path + model_name
    handWritingRNN = HandWritingRNN()
    handWritingRNN.load_state_dict(torch.load(model_file
                                    , map_location=device))
    handWritingRNN.to(device)
    generated_samples = handWritingRNN.random_sample(device=device, 
                            length=sample_length, count=num_sample)

    for i in range(num_sample):
        plot_stroke(generated_samples[:, i, :].cpu().numpy(),
            save_name="data/samples/{}_{}.png".format(
                model_name.replace(".pt", ""), i))
# ------------------------------------------------------------------------------


def main():

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    if(use_cuda):
      torch.cuda.empty_cache()

    # np.random.seed(101)
    # torch.random.manual_seed(101)

    # training
    train(device=device, batch_size=3)

    # generate samples from some available trained models
    epoch_list = [93]
    for epoch in epoch_list:
        print("Sampling from epoch {} model.".format(epoch))
        generate_from_model(model_name='handwriting_uncond_ep{}.pt'.format(epoch),
                                device=device)



if __name__ == "__main__":
    main()
