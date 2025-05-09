import torch  
import torch.nn as nn  
import torch.nn.functional as F  

class AttentionMechanism(nn.Module):  
    def __init__(self, hidden_size, num_mixture_components, alphabet_size):  
        super(AttentionMechanism, self).__init__()  
        self.hidden_size = hidden_size  
        self.num_mixture_components = num_mixture_components  
        self.alphabet_size = alphabet_size # alphabet size  

        attention_input_dim = self.alphabet_size + 3 + self.hidden_size
        attention_output_dim = 3 * self.num_mixture_components 
        # attention parameters layer  
        self.attention_params = nn.Linear(attention_input_dim, attention_output_dim)  
          
    def forward(self, hidden_state, prev_window, prev_kappa, inputs, char_seq, char_seq_lengths):  
        B = hidden_state.size(0) 

        attention_input = torch.cat([prev_window, inputs, hidden_state], dim=1)  
        # get attention parameters  
        attention_params = F.softplus(self.attention_params(attention_input))  
        alpha, beta, kappa_delta = torch.split(attention_params,   
                                              self.num_mixture_components,   
                                              dim=1)  

        # update kappa (position in the character sequence)  
        kappa = prev_kappa + kappa_delta / 25.0  
        beta = torch.clamp(beta, min=0.01)  
          
        # create attention weights  
        batch_size = hidden_state.size(0)  
        max_char_len = char_seq.size(1) 
        u = torch.arange(max_char_len, device=hidden_state.device).float()  
        u = u.unsqueeze(0).unsqueeze(1).expand(batch_size, self.num_mixture_components, -1) # shape: (batch, num_comp, max_char_len) 
          
        kappa_expanded = kappa.unsqueeze(2).expand(-1, -1,max_char_len)  # shape: (batch, num_comp, max_char_len)
        beta_expanded = beta.unsqueeze(2).expand(-1, -1, max_char_len)  
        alpha_expanded = alpha.unsqueeze(2).expand(-1, -1, max_char_len)  
        
        phi = torch.sum(
        alpha_expanded *
        torch.exp(-(kappa_expanded - u) ** 2 / ( 2.0 * beta_expanded ** 2)),
        dim=1) # shape: (batch, max_char_len) 

        # apply sequence mask  
        mask = torch.arange(max_char_len, device=hidden_state.device).unsqueeze(0)  
        mask = mask.expand(batch_size, -1) < char_seq_lengths.unsqueeze(1)  
        phi = phi * mask.float()  
          
        # get context vector  
        context = torch.bmm(phi.unsqueeze(1), char_seq).squeeze(1)  
          
        return context, kappa, phi