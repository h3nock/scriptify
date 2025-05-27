import torch  
import numpy as np
import torch.nn as nn  
import torch.nn.functional as F

from src.data.dataloader import ProcessedHandwritingDataset  
from .attention import AttentionMechanism  
from .gmm import GMMLayer  
  
class HandwritingRNN(nn.Module):  
    def __init__(self,   
                 lstm_size=400,   
                 output_mixture_components=20,   
                 attention_mixture_components=10,  
                 alphabet_size = ProcessedHandwritingDataset.get_alphabet_size(), 
                 dropout_prob = 0.2
                 ):  
        super(HandwritingRNN, self).__init__()  
        self.lstm_size = lstm_size  
        self.output_mixture_components = output_mixture_components  
        self.attention_mixture_components = attention_mixture_components  
        self.alphabet_size = alphabet_size
        self.dropout_prob = dropout_prob 

        # LSTM layers  
        self.lstm1 = nn.LSTMCell(3 + self.alphabet_size, self.lstm_size)  # Input + window vector  
        self.lstm2 = nn.LSTMCell(3 + self.lstm_size + self.alphabet_size, self.lstm_size)  # Input + lstm1 output + window  
        self.lstm3 = nn.LSTMCell(3 + 2*self.lstm_size + self.alphabet_size, self.lstm_size)  # Input + lstm1 output + lstm2 output + window  

        if self.dropout_prob > 0:
            self.dropout = nn.Dropout(p = self.dropout_prob)
          
        # attention mechanism  
        self.attention = AttentionMechanism(  
            self.lstm_size,   
            self.attention_mixture_components, 
            self.alphabet_size
        )  
          
        self.gmm = GMMLayer(3 * self.lstm_size, self.output_mixture_components)  

    def one_hot_encode(self, char_seq):
        return F.one_hot(char_seq, num_classes=self.alphabet_size,).float() 
          
    def forward(self, inputs, char_seq, char_seq_lengths, hidden_state=None, bias=None):  
        assert inputs.dim() == 3 and inputs.size(2) == 3 
        assert char_seq.dim() == 2 
        assert char_seq_lengths.dim() == 1 

        batch_size = inputs.size(0)  
        seq_length = inputs.size(1)  
          
        # initialize hidden states if not provided  
        if hidden_state is None:  
            h1 = torch.zeros(batch_size, self.lstm_size, device=inputs.device)  
            c1 = torch.zeros(batch_size, self.lstm_size, device=inputs.device)  
            h2 = torch.zeros(batch_size, self.lstm_size, device=inputs.device)  
            c2 = torch.zeros(batch_size, self.lstm_size, device=inputs.device)  
            h3 = torch.zeros(batch_size, self.lstm_size, device=inputs.device)  
            c3 = torch.zeros(batch_size, self.lstm_size, device=inputs.device)  
            window = torch.zeros(batch_size, self.alphabet_size, device=inputs.device)  
            kappa = torch.zeros(batch_size, self.attention_mixture_components, device=inputs.device)   
        else:  
            h1, c1, h2, c2, h3, c3, window, kappa = hidden_state 
          
        # one-hot encode 
        char_one_hot = self.one_hot_encode(char_seq) 
        
        B, T_c, C1 = char_one_hot.shape 
        assert C1 == self.alphabet_size 

          
        # process sequence  
        outputs = []  
        for t in range(seq_length):  
            x_t = inputs[:, t, :]  
              
            # LSTM 1  
            lstm1_input = torch.cat([window, x_t], dim=1)  
            h1, c1 = self.lstm1(lstm1_input, (h1, c1))  

            if self.dropout_prob > 0 and self.training:
                h1_d = self.dropout(h1) 
            else:
                h1_d = h1 
              
            # Attention  
            window, kappa, phi = self.attention(  
                h1_d, window, kappa, x_t, char_one_hot, char_seq_lengths  
            )  
              
            # LSTM 2  
            lstm2_input = torch.cat([x_t, h1_d, window], dim=1)  
            h2, c2 = self.lstm2(lstm2_input, (h2, c2))  

            if self.dropout_prob > 0 and self.training:
                h2_d = self.dropout(h2)
            else:
                h2_d = h2
              
            # LSTM 3  
            lstm3_input = torch.cat([x_t,h1_d, h2_d, window], dim=1)  
            h3, c3 = self.lstm3(lstm3_input, (h3, c3))  
            
            if self.dropout_prob > 0 and self.training:
                h3_d = self.dropout(h3) 
            else:
                h3_d = h3 
                
            # GMM output 
            gmm_input = torch.cat([h1_d,h2_d,h3_d], dim=1) 
            gmm_params = self.gmm(gmm_input, bias)  
            outputs.append(gmm_params)  
          
        stacked_outputs = [torch.stack([out[i] for out in outputs], dim=1) for i in range(5)]  
          
        # hidden state for next sequence  
        hidden_state = (h1, c1, h2, c2, h3, c3, window, kappa)  
          
        return stacked_outputs, hidden_state  
      
    def sample(self, char_seq, char_seq_lengths, max_length=1000, bias=0.5, prime=None):  
        batch_size = char_seq.size(0)  
        device = char_seq.device  
        
        pen_up_ctr = 0 
        done_ctr = 0
        
        # embed character sequence  
        char_one_hot = self.one_hot_encode(char_seq) 
          
        # initialize hidden states  
        h1 = torch.zeros(batch_size, self.lstm_size, device=device)  
        c1 = torch.zeros(batch_size, self.lstm_size, device=device)  
        h2 = torch.zeros(batch_size, self.lstm_size, device=device)  
        c2 = torch.zeros(batch_size, self.lstm_size, device=device)  
        h3 = torch.zeros(batch_size, self.lstm_size, device=device)  
        c3 = torch.zeros(batch_size, self.lstm_size, device=device)  
        window = torch.zeros(batch_size, self.alphabet_size, device=device)
        kappa = torch.zeros(batch_size, self.attention_mixture_components, device=device)  
          
        # initial input  
        x = torch.zeros(batch_size, 3, device=device)  
        x[:, 2] = 1.0  # initial pen state  
          
        if prime is not None:  
            prime_seq_length = prime.size(1)  
            # process prime sequence to get initial hidden states  
            for t in range(prime_seq_length):  
                x_t = prime[:, t, :]  
                  
                # LSTM 1  
                lstm1_input = torch.cat([window, x_t], dim=1)  
                h1, c1 = self.lstm1(lstm1_input, (h1, c1))  
                  
                # Attention  
                window, kappa, phi = self.attention(  
                    h1, window, kappa, x_t, char_one_hot, char_seq_lengths  
                )  
                  
                # LSTM 2  
                lstm2_input = torch.cat([x_t, h1, window], dim=1)  
                h2, c2 = self.lstm2(lstm2_input, (h2, c2))  
                  
                # LSTM 3  
                lstm3_input = torch.cat([x_t,h1, h2, window], dim=1)  
                h3, c3 = self.lstm3(lstm3_input, (h3, c3))  
                  
                if t == prime_seq_length - 1:  
                    x = x_t  
          
        # generate sequence  
        strokes = [] 
        if isinstance(bias, (float, int)):
             bias_tensor = torch.tensor([bias] * batch_size, device=device, dtype=torch.float32)
        elif isinstance(bias, torch.Tensor):
             bias_tensor = bias.to(device)
        else:
             bias_tensor = torch.tensor([0.5] * batch_size, device=device, dtype=torch.float32)

        for t in range(max_length):  
            # LSTM 1  
            lstm1_input = torch.cat([window, x], dim=1)  
            h1, c1 = self.lstm1(lstm1_input, (h1, c1))  
              
            # Attention  
            window, kappa, phi = self.attention(  
                h1, window, kappa, x, char_one_hot, char_seq_lengths  
            )  
              
            # LSTM 2  
            lstm2_input = torch.cat([x, h1, window], dim=1)  
            h2, c2 = self.lstm2(lstm2_input, (h2, c2))  
              
            # LSTM 3  
            lstm3_input = torch.cat([x, h1, h2, window], dim=1)  
            h3, c3 = self.lstm3(lstm3_input, (h3, c3))  
              
            # GMM output  
            gmm_input = torch.cat([h1, h2, h3], dim=1)
            pis, sigmas, rhos, mus, es = self.gmm(gmm_input, bias_tensor)  
              
            # sample from GMM  
            stroke = self.gmm.sample(pis, sigmas, rhos, mus, es)  
            strokes.append(stroke)  
              
            # update input for next step  
            x = stroke  
            
            # early stopping criteria 
            pen_up   = (stroke[:, 2] > 0.7).all()
            tiny_mv  = (stroke[:, :2].abs().max() < 0.02).all()
            if pen_up and tiny_mv:
                pen_up_ctr += 1
            else:
                pen_up_ctr = 0

            if (phi[:, -1] > 0.98).all():
                done_ctr += 1
            else:
                done_ctr = 0

            if pen_up_ctr >= 10 or done_ctr >= 8:
                break

        strokes = torch.stack(strokes, dim=1)          
        return [seq for seq in strokes]
        