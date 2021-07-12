import torch


class DeepNNO:
    def __init__(self, tau, device, factor=2., bm=True, online=True):
        self.tau = tau
        self.factor = factor
        self.device = device
        self.compute_tau_fn = self.compute_tau_bm # if bm else self.compute_tau
        self.online = online
        self.counter3 = torch.zeros(1, device=self.device)

    def reset(self):
        self.counter3 = torch.zeros(1, device=self.device)

    def update_taus(self, x, y, n_classes, gamma=0.9, weight=None):
        tau_tot = 0.
        cl = 0

        for i in range(n_classes):
            # For each label compute tau with weights for positive and negative
            N, tau = self.compute_tau_fn(x, y, i, alpha=self.factor, w=weight)  # Compute mean
            # If labels already in the set, just update holder, otherwise add it to the model
            if N == 0:
                continue
            else:
                cl += 1.
                tau_tot += tau
                assert tau <= 1.0, 'Tau is larger than 1, that is not good'
                self.counter3 += 1  # counter tempo (è il t del paper)

        # update tau for mini-batch t+1 considering tau at mini-batch t
        if self.online:
            self.tau.data = gamma * self.tau.data + (1. - gamma) * tau_tot / cl
        else:
            self.tau.data = (self.tau.data * self.counter3 + tau_tot / cl) / (1 + self.counter3)

    def compute_tau(self, x, y, i, alpha=None, w=None):
        mask = (i == y.data).view(-1).float()
        mask = mask.cuda()
        N = mask.sum()
        if N == 0:
            return 0, 0
        else:
            return N, ((x   .data[:, i]) * mask).sum() / N

    def compute_tau_bm(self, x, y, i, alpha=1., w=None):
        mask = (y.data == i).view(-1, 1).float()
        N = mask.sum()

        if w is not None:
            mask = mask * w
            mask = mask / mask.sum() * N
            mask = mask.view(-1)
        masked = ((x.data[:, i] - self.tau).unsqueeze(1)) * mask

        samples = ((x.data[:, i]).unsqueeze(1)) * mask
        wrongly = (masked < 0.).view(-1, 1).float()
        NW = wrongly.sum()
        if NW == 0:
            NW = 1
        if N == 0:
            return 0, 0
        else:
            return N, (samples + samples * wrongly * alpha).sum() / (N + alpha * NW)
