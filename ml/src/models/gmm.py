from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

class GMMLayer(nn.Module):
    """
    GMM layer. 
    """
    def __init__(self,input_size: int, num_mixtures: int,bias: float = 1, eps: float = 1e-8, sigma_eps: float = 1e-4):
        super(GMMLayer, self).__init__()
        self.input_size = input_size 
        self.num_mixtures = num_mixtures
        self.output_size = num_mixtures * 6 + 1  
        self.output_layer = nn.Linear(input_size, self.output_size)
        self.eps = eps 
        self.sigma_eps = sigma_eps 

    def forward(self, x: torch.Tensor,
                bias: torch.Tensor) ->  tuple [
                    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
                    torch.Tensor]:
        """
        """
        z = self.output_layer(x)  

        pis_logits, sigmas_logits, rhos, mus, es = torch.split(z, [
            1 * self.num_mixtures, 
            2 * self.num_mixtures, 
            1 * self.num_mixtures, 
            2 * self.num_mixtures, 
            1
        ], dim = 1)         

        if not isinstance(bias, torch.Tensor):  
            bias_tensor = torch.tensor(bias, device=x.device, dtype=x.dtype)  
        else:  
            bias_tensor = bias  
        
        if bias_tensor.numel() == 1: # scalar bias
            pis_logits = pis_logits * (1 + bias_tensor)
            sigmas_logits = sigmas_logits - bias_tensor
        else: # per-batch item bias
            pis_logits = pis_logits * (1 + bias_tensor.unsqueeze(-1))  
            sigmas_logits = sigmas_logits - bias_tensor.unsqueeze(-1) 


        pis = F.softmax(pis_logits, dim=-1)
        sigmas = torch.exp(sigmas_logits).clamp(min = self.sigma_eps) 
        rhos = torch.tanh(rhos).clamp(min = self.eps - 1.0, max=1.0 - self.eps) 
        es = torch.sigmoid(es).clamp(min = self.eps, max = 1.0 - self.eps)
 
        return pis, sigmas, rhos, mus, es 
    
    def sample(self, pis: torch.Tensor, sigmas: torch.Tensor,
               rhos: torch.Tensor, mus: torch.Tensor, es: torch.Tensor):

        batch_size = pis.size(0) 

        idx = torch.multinomial(pis, 1).squeeze(1)
        
        mu_x, mu_y = torch.split(mus, self.num_mixtures, dim = 1) 
        sigma_x, sigma_y = torch.split(sigmas, self.num_mixtures, dim = 1) 

        idx_expanded = idx.unsqueeze(1) 

        mu_x_selected = torch.gather(mu_x, 1, idx_expanded).squeeze(1) 
        mu_y_selected = torch.gather(mu_y, 1, idx_expanded).squeeze(1) 
        sigma_x_selected = torch.gather(sigma_x, 1, idx_expanded).squeeze(1) 
        sigma_y_selected = torch.gather(sigma_y, 1, idx_expanded).squeeze(1) 

        rho_selected = torch.gather(rhos, 1, idx_expanded).squeeze(1) 

        # sample from bivariate normal samples 
        z0 = torch.randn_like(mu_x_selected)
        z1 = torch.randn_like(mu_y_selected)

        # transform to correlated bivariate normal
        # x = mu_x + sigma_x * z0
        x_sample = mu_x_selected + sigma_x_selected * z0
        
        # y = mu_y + sigma_y * (rho * z0 + sqrt(1 - rho^2) * z1)
        sqrt_term = torch.sqrt(torch.clamp(1.0 - rho_selected.pow(2), min=self.eps))
        y_sample = mu_y_selected + sigma_y_selected * (rho_selected * z0 + sqrt_term * z1)
        
        # combine x and y samples
        xy = torch.stack([x_sample, y_sample], dim=1)   
          
        # sample end-of-stroke  
        # es should be (batch_size, 1)
        e_probs = es.squeeze(-1) if es.dim() > 1 else es
        e = torch.bernoulli(e_probs).unsqueeze(-1) 

        # combine into stroke  
        stroke = torch.cat([xy, e], dim=1)  
        return stroke