import torch  
import torch.nn as nn  
import torch.nn.functional as F  

class AttentionMechanism(nn.Module):  
    def __init__(self, hidden_size, num_mixture_components, char_vector_size):  
        super(AttentionMechanism, self).__init__()  
        self.hidden_size = hidden_size  
        self.num_mixture_components = num_mixture_components  
        self.char_vector_size = char_vector_size  
          
        # attention parameters layer  
        self.attention_params = nn.Linear(hidden_size * 2 + 3, 3 * num_mixture_components)  
          
    def forward(self, hidden_state, prev_window, prev_kappa, inputs, char_seq, char_seq_lengths):  
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
        u = torch.arange(self.char_vector_size, device=hidden_state.device).float()  
        u = u.unsqueeze(0).unsqueeze(1).expand(batch_size, self.num_mixture_components, -1)  
          
        kappa_expanded = kappa.unsqueeze(2).expand(-1, -1, self.char_vector_size)  
        beta_expanded = beta.unsqueeze(2).expand(-1, -1, self.char_vector_size)  
        alpha_expanded = alpha.unsqueeze(2).expand(-1, -1, self.char_vector_size)  
          
        phi = torch.sum(alpha_expanded * torch.exp(-torch.square(kappa_expanded - u) / beta_expanded), dim=1)  
          
        # apply sequence mask  
        mask = torch.arange(self.char_vector_size, device=hidden_state.device).unsqueeze(0)  
        mask = mask.expand(batch_size, -1) < char_seq_lengths.unsqueeze(1)  
        phi = phi * mask.float()  
          
        # get context vector  
        context = torch.bmm(phi.unsqueeze(1), char_seq).squeeze(1)  
          
        return context, kappa, phi