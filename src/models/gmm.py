import torch
import torch.nn as nn
import torch.nn.functional as F

class GMMLayer(nn.Module):
    """
    GMM layer. 
    """
    def __init__(self,input_size, num_mixtures,bias =0.75, eps=1e-8, sigma_eps=1e-4):
        super(GMMLayer, self).__init__()
        self.input_size = input_size 
        self.num_mixtures = num_mixtures
        self.output_size = num_mixtures * 6 + 1  
        self.output_layer = nn.Linear(input_size, self.output_size)
        self.eps = eps 
        self.sigma_eps = sigma_eps 

    def forward(self, x, bias= None):
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

        if bias is not None:  
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
        mus = torch.tanh(mus)  
        es = torch.sigmoid(es).clamp(min = self.eps, max = 1.0 - self.eps)
 
        return pis, sigmas, rhos, mus, es 
    
    def sample(self, pis, sigmas, rhos, mus, es):

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

        # sample from bivariate normal 
        mean = torch.stack([mu_x_selected, mu_y_selected], dim=1) 

        # create covariance matrices 
        cov = torch.zeros(batch_size, 2, 2, device=pis.device)  

        jitter = 1e-6 

        cov[:, 0, 0] = sigma_x_selected.pow(2)  + jitter 
        cov[:, 1, 1] = sigma_y_selected.pow(2) + jitter  
        cov[:, 0, 1] = rho_selected * sigma_x_selected * sigma_y_selected  
        cov[:, 1, 0] = cov[:, 0, 1]  
          
        # sample from multivariate normal  
        m = torch.distributions.MultivariateNormal(mean, cov)  
        xy = m.sample()  
          
        # sample end-of-stroke  
        e = torch.bernoulli(es)  
          
        # combine into stroke  
        stroke = torch.cat([xy, e], dim=1)  
        return stroke