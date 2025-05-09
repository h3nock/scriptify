import torch.optim as optim  
  
def get_optimizer(model, optimizer_type='adam', learning_rate=0.001, beta1=0.9, weight_decay=0.0):  
    """  
    Configure optimizer for the model  
    """  
    if optimizer_type.lower() == 'adam':  
        return optim.Adam(  
            model.parameters(),  
            lr=learning_rate,  
            betas=(beta1, 0.999),  
            weight_decay=weight_decay  
        )  
    elif optimizer_type.lower() == 'rmsprop':  
        return optim.RMSprop(  
            model.parameters(),  
            lr=learning_rate,  
            alpha=0.9,  
            weight_decay=weight_decay  
        )  
    elif optimizer_type.lower() == 'sgd':  
        return optim.SGD(  
            model.parameters(),  
            lr=learning_rate,  
            momentum=0.9,  
            weight_decay=weight_decay  
        )  
    else:  
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")  
  
def get_lr_scheduler(optimizer, scheduler_type='plateau', patience=5, factor=0.5):  
    """  
    Configure learning rate scheduler  
    """  
    if scheduler_type.lower() == 'plateau':  
        return optim.lr_scheduler.ReduceLROnPlateau(  
            optimizer,  
            mode='min',  
            factor=factor,  
            patience=patience,  
        )  
    elif scheduler_type.lower() == 'step':  
        return optim.lr_scheduler.StepLR(  
            optimizer,  
            step_size=patience,  
            gamma=factor  
        )  
    elif scheduler_type.lower() == 'exponential':  
        return optim.lr_scheduler.ExponentialLR(  
            optimizer,  
            gamma=factor  
        )  
    else:  
        raise ValueError(f"Unsupported scheduler type: {scheduler_type}")