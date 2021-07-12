import numpy as np
import torch

MEMORY = 2000.

class Exemplars:
    def __init__(self, network, device):
        self.network = network
        self.device = device
        self.exemplar_sets = []
        self.exemplar_valid_sets = []

    def get_exemplar_samples(self, train=True):
        if not train:
            ex_set = self.exemplar_valid_sets
        else:
            ex_set = self.exemplar_sets

        return [exemplar for exemplar_list in ex_set for exemplar in exemplar_list]

    # STORE IN MEMORY THE ACTUAL IMAGES FOR EXEMPLARS
    def construct_exemplar_set(self, loader, m, converted, type='train'):
        """Construct an exemplar set for image set
        Args:
        images: np.array containing images of a class
        """
        self.network.eval()
        vals = []
        exemplar = []
        added = 0
        for idx, (inputs, targets) in enumerate(loader):
            # compute distances for each image in order to find the closest to the mean
            inputs = inputs.to(self.device)
            outputs, _ = self.network(inputs)
            _, _, distances = self.network.predict(outputs)
            val = distances.detach().to('cpu').numpy()[:, converted]
            vals = vals + val.tolist()

        # select the topk images that will become exemplars
        minimals = torch.from_numpy(np.array(vals))
        _, idxs = torch.topk(minimals, k=int(m), largest=False)
        for i in idxs:
            exemplar.append(loader.dataset.samples[i])

        if type == 'train':
            self.exemplar_sets.append(exemplar)
        else:
            self.exemplar_valid_sets.append(exemplar)

    def reduce_exemplar_sets(self, exemplar_m, valid_m=None):
        for i in range(len(self.exemplar_sets)):
            self.exemplar_sets[i] = self.exemplar_sets[i][:int(exemplar_m)]
        if valid_m is not None:
            for i in range(len(self.exemplar_valid_sets)):
                self.exemplar_valid_sets[i] = self.exemplar_valid_sets[i][:int(valid_m)]
