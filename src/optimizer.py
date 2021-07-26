import torch


class Optimizer(object):
    def __init__(
        self,
        params,
        optimizer_type,
        lr=0.05,
        momentum=0,
        weight_decay=0,
        eps=1e-6
    ):
        """
        Args:
            params: torch.nn.Parameter. The NN parameters to optimize
            optimizer_type: type of the optimizer to use
            lr: learning rate
            momentum: momentum, if needed
            weight_decay: weight decay, if needed. Equivalent to L2 regulariztion.
            eps: eps parameter, if needed.
        """
        if optimizer_type == "RMSProp":
            self.optimizer = torch.optim.RMSprop(params, lr=lr, eps=eps, weight_decay=weight_decay, momentum=momentum)
        elif optimizer_type == "Adagrad":
            self.optimizer = torch.optim.Adagrad(params, lr=lr, weight_decay=weight_decay)
        elif optimizer_type == "Adadelta":
            self.optimizer = torch.optim.Adadelta(params, lr=lr, eps=eps, weight_decay=weight_decay)
        elif optimizer_type == "Adam":
            self.optimizer = torch.optim.Adam(params, lr=lr, eps=eps, weight_decay=weight_decay)
        elif optimizer_type == "SGD":
            self.optimizer = torch.optim.SGD(params, lr=lr, weight_decay=weight_decay, momentum=momentum)
        else:
            raise NotImplementedError

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        self.optimizer.step()
