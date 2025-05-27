import torch  
import numpy as np  
  
def gaussian_mixture_loss(y, y_lengths, pis, sigmas, rhos,mus, es, eps=1e-8, sigma_eps=1e-4):  
    """  
    Calculate negative log-likelihood loss for the GMM  
    """  
    batch_size = y.size(0)  
    seq_length = y.size(1)  
      
    y_1, y_2, y_3 = torch.split(y, 1, dim=2)  
      
    num_mixtures = pis.size(2)  
      
    sigma_1, sigma_2 = torch.split(sigmas, num_mixtures, dim=2)  
    mu_1, mu_2 = torch.split(mus, num_mixtures, dim=2)  
      
    # calculate bivariate Gaussian probability 
    norm = 1.0 / (2 * np.pi * sigma_1.clamp(min=sigma_eps) * sigma_2.clamp(min=sigma_eps) * 
               torch.sqrt((1 - torch.square(rhos)).clamp(min=sigma_eps)))
    Z = (torch.square((y_1 - mu_1) / sigma_1) +   
         torch.square((y_2 - mu_2) / sigma_2) -   
         2 * rhos * (y_1 - mu_1) * (y_2 - mu_2) / (sigma_1 * sigma_2))  
      
    exp = -1.0 * Z / (2 * (1 - torch.square(rhos)))  
    gaussian_likelihoods = torch.exp(exp) * norm  
      
    # mixture probabilities  
    gmm_likelihood = torch.sum(pis * gaussian_likelihoods, dim=2).clamp(min=eps)
      
    # bernoulli likelihood for pen state  
    bernoulli_likelihood = torch.where(  
        y_3 > 0.5,   
        es,   
        1 - es  
    ).squeeze(2)  
      
    nll = -(torch.log(gmm_likelihood) + torch.log(bernoulli_likelihood)) 
    # mask padded values  
    mask = (torch.arange(seq_length, device=y.device)[None] < y_lengths[:,None])
    nll  = torch.where(mask, nll, torch.zeros_like(nll))
      
    # calculate losses  
    sequence_loss = torch.sum(nll, dim=1) / y_lengths.float()  
    element_loss = torch.sum(nll) / torch.sum(y_lengths.float())  
      
    return sequence_loss, element_loss