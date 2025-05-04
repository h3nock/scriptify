import torch
import torch.nn as nn
import torch.nn.functional as F

class GMMLayer(nn.Module):
    """
    GMM layer. 
    """
    def __init__(self,input_size, num_mixtures):
        super(GMMLayer, self).__init__()
        self.input_size = input_size 
        self.num_mixtures = num_mixtures
        self.output_size = num_mixtures * 6 + 1  
        self.output_layer = nn.Linear(input_size, self.out_dim)
    
    def forward(self, x):
        """
        """
        z = self.output_layer(x)  

        pis, sigmas, rhos, mus, es = torch.split(z, [
            1 * self.num_mixtures, 
            2 * self.num_mixtures, 
            1 * self.num_mixtures, 
            2 * self.num_mixtures, 
            1
        ])         

        pis = F.softmax(pis, dim=-1)
        sigmas = torch.exp(sigmas).clamp(min = 1e-4) 
        rhos = torch.tanh(rhos).clamp(min = -0.99, max=0.99) 
        es = torch.sigmoid(es).clamp(min = 1e-8, max = 1 - 1e-8)
 
        mdn_params = {
            'pis': pis,       
            'mus': mus,       
            'sigmas': sigmas, 
            'rhos': rhos, 
            'es': es     
        }
        return mdn_params 
    
    def sample(self, pis, mus, sigmas, rhos, es):

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
        # create covariance matrices  
        cov = torch.zeros(batch_size, 2, 2, device=pis.device)  
        cov[:, 0, 0] = sigma_x_selected.pow(2)  
        cov[:, 1, 1] = sigma_y_selected.pow(2)  
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



        
        