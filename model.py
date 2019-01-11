import torch
from torch.distributions import MultivariateNormal, Bernoulli, Categorical

class HandWritingRNN(torch.nn.Module):

    def __init__(self, memory_cells=400, n_gaussians=20, num_layers=3):
        '''
        input_size is fixed to 3.
        hidden_size = memory_cells 
        Output dimension after the fully connected layer = (6 * n_gaussians + 1)
        '''
        super().__init__()
        self.n_gaussians = n_gaussians

        self.rnns = torch.nn.ModuleList()
        for i in range(num_layers):
            input_size = 3 if i == 0 else (3 + memory_cells)
            self.rnns.append(torch.nn.LSTM(input_size, memory_cells, 1))

        self.last_layer = torch.nn.Linear(in_features = memory_cells*num_layers, 
                                            out_features = (6*n_gaussians + 1))

        self.sigmoid = torch.nn.Sigmoid()
        self.softmax = torch.nn.Softmax(dim=2)
        self.tanh = torch.nn.Tanh()

    def forward(self, inp, lstm_in_states=None):
        '''
        first_layer of self.rnns gets inp as input
        subsequent layers of self.rnns get inp concatenated with output of
        previous layer as the input. 
        args : 
            inp : input sequence of dimensions (T, B, 3)
            lstm_in_states : input states for num_layers number of lstm layers;
                            it is a list of num_layers tupels (h_i, c_i), with 
                            both h_i and c_i tensor of dimensions (memory_cells,)
        '''
        rnn_out = []
        rnn_out.append(
            self.rnns[0](inp, lstm_in_states[0]) if lstm_in_states != None
            else self.rnns[0](inp)
            )
        
        for i, rnn in enumerate(self.rnns[1:]):
            rnn_inp = torch.cat((rnn_out[-1][0], inp), dim=2)
            rnn_out.append(
                rnn(rnn_inp, lstm_in_states[i+1]) if lstm_in_states != None
                else rnn(rnn_inp)
                )

        # rnn_out is a list of tuples (out, (h, c))
        lstm_out_states = [o[1] for o in rnn_out]
        rnn_out = torch.cat([o[0] for o in rnn_out], dim=2)
        y = self.last_layer(rnn_out)

        if(y.requires_grad):
            y.register_hook(lambda x: x.clamp(min=-10, max=.10))

        pi = self.softmax(y[:, :, :self.n_gaussians])
        mu = y[:, :, self.n_gaussians:3*self.n_gaussians]
        sigma = torch.exp(y[:, :, 3*self.n_gaussians:5*self.n_gaussians])
        # sigma = y[:, :, 3*self.n_gaussians:5*self.n_gaussians]
        rho = self.tanh(y[:, :, 5*self.n_gaussians:6*self.n_gaussians]) * 0.9
        e = self.sigmoid(y[:, :, 6*self.n_gaussians])

        return e, pi, mu, sigma, rho, lstm_out_states


    def random_sample(self, length=300, count=1, device=torch.device("cpu")):
        '''
        Get a random sample from the distribution (model)
        '''
        samples = torch.zeros(length+1, count, 3, device=device) # batch_first=false
        lstm_states = None

        for i in range(1, length+1):
            # get distribution parameters
            with torch.no_grad():
                e, pi, mu, sigma, rho, lstm_states = self.forward(samples[i-1:i]
                                                            , lstm_states)
            # sample from the distribution (returned parameters)
            # samples[i, :, 0] = e[-1, :] > 0.5 # can be sampled from bernoulli instead
            distrbn1 = Bernoulli(e[-1, :])
            samples[i, :, 0] = distrbn1.sample()

            # selected_mode = torch.argmax(pi[-1, :, :], dim=1) # shape = (count,)
            distrbn2 = Categorical(pi[-1, :, :])
            selected_mode = distrbn2.sample()

            index_1 = selected_mode.unsqueeze(1) # shape = (count, 1)
            index_2 = torch.stack([index_1, index_1], dim=2) # shape = (count, 1, 2)

            mu = mu[-1].view(count, self.n_gaussians, 2).gather(dim=1, index=index_2).squeeze()
            sigma = sigma[-1].view(count, self.n_gaussians, 2).gather(dim=1, index=index_2).squeeze()
            rho = rho[-1].gather(dim=1, index=index_1).squeeze()

            sigma2d = sigma.new_zeros(count, 2, 2)
            sigma2d[:, 0, 0] = sigma[:, 0]**2
            sigma2d[:, 1, 1] = sigma[:, 1]**2
            sigma2d[:, 0, 1] = rho[:] * sigma[:, 0] * sigma[:, 1]
            sigma2d[:, 1, 0] = sigma2d[:, 0, 1]

            distribn = MultivariateNormal(loc=mu, covariance_matrix=sigma2d)

            samples[i, :, 1:]  = distribn.sample()

        return samples[1:, :, :] # remove dummy first zeros
