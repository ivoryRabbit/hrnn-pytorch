import torch


class Optimizer(object):
    def __init__(self, args, params, optimizer_type):
        """
        Args:
            params: torch.nn.Parameter. The NN parameters to optimize
            optimizer_type: type of the optimizer to use
            lr: learning rate
            momentum: momentum, if needed
            weight_decay: weight decay, if needed. Equivalent to L2 regulariztion
            eps: eps parameter, if needed
        """
        if optimizer_type == "RMSProp":
            self.optimizer = torch.optim.RMSprop(
                params,
                lr=args.lr, eps=args.eps, weight_decay=args.weight_decay, momentum=args.momentum
            )
        elif optimizer_type == "Adagrad":
            self.optimizer = torch.optim.Adagrad(
                params,
                lr=args.lr, weight_decay=args.weight_decay
            )
        elif optimizer_type == "Adadelta":
            self.optimizer = torch.optim.Adadelta(
                params,
                lr=args.lr, eps=args.eps, weight_decay=args.weight_decay
            )
        elif optimizer_type == "Adam":
            self.optimizer = torch.optim.Adam(
                params,
                lr=args.lr, eps=args.eps, weight_decay=args.weight_decay
            )
        elif optimizer_type == "SGD":
            self.optimizer = torch.optim.SGD(
                params,
                lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum
            )
        else:
            raise NotImplementedError

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        self.optimizer.step()
