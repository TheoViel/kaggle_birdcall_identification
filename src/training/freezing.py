

def freeze_bn(m):
    """
    Freezes batchnorm in a model

    Arguments:
        m {torch module} -- Model
    """
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()
        m.weight.requires_grad = False
        m.bias.requires_grad = False


def unfreeze_bn(m):
    """
    Unfreezes batchnorm in a model

    Arguments:
        m {torch module} -- Model
    """
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.train()
        m.weight.requires_grad = True
        m.bias.requires_grad = True


def freeze(m):
    """
    Freezes a model
    
    Arguments:
        m {torch module} -- Model to freeze
    """
    for param in m.parameters():
        param.requires_grad = False
    unfreeze_bn(m)


def unfreeze(m):
    """
    Unfreezes a model
    
    Arguments:
        m {torch module} -- Model to unfreeze
    """
    for param in m.parameters():
        param.requires_grad = True


def freeze_layer(model, name):
    """
    Freezes layer(s) of a model
    
    Arguments:
        model {[torch module]} -- Model to freeze layers from
        name {[str]} -- Layers containing "name" in their name will be frozen
    """
    for n, p in list(model.named_parameters()):
        if name in n:
            p.requires_grad = False


def unfreeze_layer(model, name):
    """
    Unfreezes layer(s) of a model
    
    Arguments:
        model {[torch module]} -- Model to unfreeze layers from
        name {[str]} -- Layers containing "name" in their name will be unfrozen
    """
    for n, p in list(model.named_parameters()):
        if name in n:
            p.requires_grad = True