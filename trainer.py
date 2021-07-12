import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Parameter
from loss import SNNLoss, MarginLoss
from deep_nno import DeepNNO
import copy
import numpy as np
from torch.autograd import Variable
import torchvision.transforms as T
import random
import numpy.random as npr

def save_image(tensor, index):
    t = T.ToPILImage()
    pil_image = t(tensor)
    pil_image.save(f'{index}.png')

class Trainer:
    def __init__(self, opts, network, discriminator, device, logger, augment_ops=None):
        self.ce = nn.CrossEntropyLoss(reduction='mean')
        self.bce = nn.BCELoss(reduction='mean')
        self.distill_loss = nn.MSELoss(reduction='mean')
        self.snnl = SNNLoss()
        self.margin_loss = MarginLoss()
        self.augment_ops = augment_ops

        # DA method
        self.rsda = opts.rsda
        self.self_challenging = opts.self_challenging

        # store network
        self.network = network
        self.discriminator = discriminator
        self.network2 = None

        # method section
        self.features_dw = opts.features_dw
        self.snnl_weight = opts.snnlw
        self.ce_weight = opts.ce
        self.bce_weight = opts.bce
        self.ss_weight = opts.ssw
        self.nno = opts.nno

        # tau
        self.deep_nno = opts.deep_nno  # deep nno
        self.tau_val = not opts.no_tau_val
        self.multiple_taus = opts.multiple_taus
        if opts.multiple_taus:
            self.tau = Parameter(torch.ones(opts.initial_classes, device=device)*0.5, requires_grad=True)
        else:
            self.tau = Parameter(torch.tensor([0.5], device=device), requires_grad=True)

        if self.deep_nno or self.nno:
            self.deep_nno_handler = DeepNNO(self.tau, device, factor=opts.tau_factor, bm=self.deep_nno)

        # others
        self.device = device
        self.logger = logger
        self.num_classes = opts.initial_classes
        self.epochs = opts.epochs
        self.dataset = opts.dataset

    def next_iteration(self, new_classes):
        self.network2 = copy.deepcopy(self.network)
        self.network2.eval()
        # Duplicate current network to distillate info
        self.network.linear.reset()  # reset the counters for NCM
        # Prepare internal structure (allocate mean array, etc) for the new classes
        self.network.add_classes(new_classes)
        # Store the new number of classes
        self.num_classes += new_classes

        if self.tau_val and self.multiple_taus:
            self.tau = Parameter(torch.cat((self.tau, torch.ones(new_classes, device=self.device)*0.5), 0))
        if self.tau_val and not self.multiple_taus:
            self.tau = Parameter(torch.tensor([0.5], device=self.device), requires_grad=True)

    def reject(self, x, dist, tau=None):
        out = torch.zeros(x.shape[0], x.shape[1] + 1).to(x.device)

        if self.deep_nno or self.nno:
            out[:, :x.shape[1]] = (x > tau).float() * 1. * x
        else:
            out[:, :x.shape[1]] = (dist <= tau).float() * 1. * x

        # last column contains probabilities (distances) for unknown.
        out[:, -1] = 1. - ((out[:, :x.shape[1]]).sum(1) > 0).float()
        return out

    def train(self, epoch, train_loader, subset_trainloader, optimizer, class_dict, iteration):
        # Training, single epoch
        print(f'Epoch: {epoch}')

        if iteration == 0 or not self.nno:
            self.network.train()
        else:
            self.network.eval()

        train_loss = 0
        ss_loss=0
        correct = 0
        ss_correct = 0
        total = 0
        count = 0
        print(f"Tau: {self.tau}")

        # very first for the first batch
        if self.augment_ops is not None and epoch == 0:
            transform, level = self.augment_ops.get_augment()
            transform = self.augment_ops.compose(transform, level)
            train_loader.dataset.set_transform(transform)

        for idx, (inputs, targets_prep) in enumerate(train_loader):
            count += 1

            # one_hot_encoding targets
            targets_prep = targets_prep.to(self.device)
            targets = torch.zeros(inputs.shape[0], len(class_dict.keys())).to(self.device)
            targets.scatter_(1, targets_prep.view(-1, 1), 1).view(inputs.shape[0], -1)

            inputs = inputs.to(self.device)
            optimizer.zero_grad()
            outputs, feat = self.network(inputs)

            if self.self_challenging:
                x_new = feat.clone().detach()
                x_new = Variable(x_new.data, requires_grad=True)
                x_new_view = x_new.mean(-1).mean(-1)
                x_new_view = x_new_view.view(x_new_view.size(0), -1)
                output, exponential, _ = self.network.predict(x_new_view)
                class_num = output.shape[1]
                index = targets_prep
                batch_size = x_new.shape[0]
                num_channel = x_new.shape[1]
                H = x_new.shape[2]
                HW = x_new.shape[2] * x_new.shape[3]
                sp_i = torch.ones([2, batch_size]).long()
                sp_i[0, :] = torch.arange(batch_size)
                sp_i[1, :] = index
                sp_v = torch.ones([batch_size])
                one_hot_sparse = torch.sparse.FloatTensor(sp_i, sp_v,
                                                          torch.Size([batch_size, class_num])).to_dense().cuda()
                one_hot_sparse = Variable(one_hot_sparse, requires_grad=False)
                one_hot = torch.sum(output * one_hot_sparse)
                optimizer.zero_grad()
                one_hot.backward()
                grads_val = x_new.grad.clone().detach()
                grad_channel_mean = torch.mean(grads_val.view(batch_size, num_channel, -1), dim=2)
                channel_mean = grad_channel_mean
                spatial_mean = torch.mean(grads_val, dim=1)
                spatial_mean = spatial_mean.view(batch_size, H, H).view(batch_size, HW)
                optimizer.zero_grad()

                choose_one = random.randint(0, 9)
                if choose_one <= 4:
                    # ---------------------------- spatial -----------------------
                    spatial_drop_num = int(HW * 1 / 3)
                    th_mask_value = torch.sort(spatial_mean, dim=1, descending=True)[0][:, spatial_drop_num]
                    th_mask_value = th_mask_value.view(batch_size, 1).expand(batch_size, HW)
                    mask_all_cuda = torch.where(spatial_mean >= th_mask_value, torch.zeros(spatial_mean.shape).cuda(),
                                                torch.ones(spatial_mean.shape).cuda())
                    mask_all = mask_all_cuda.detach().cpu().numpy()
                    for q in range(batch_size):
                        mask_all_temp = np.ones((HW), dtype=np.float32)
                        zero_index = np.where(mask_all[q, :] == 0)[0]
                        num_zero_index = zero_index.size
                        if num_zero_index >= spatial_drop_num:
                            dumy_index = npr.choice(zero_index, size=spatial_drop_num, replace=False)
                        else:
                            zero_index = np.arange(HW)
                            dumy_index = npr.choice(zero_index, size=spatial_drop_num, replace=False)
                        mask_all_temp[dumy_index] = 0
                        mask_all[q, :] = mask_all_temp
                    mask_all = torch.from_numpy(mask_all.reshape(batch_size, 16, 16)).cuda()
                    mask_all = mask_all.view(batch_size, 1, 16, 16)
                else:
                # -------------------------- channel ----------------------------
                    mask_all = torch.zeros((batch_size, num_channel, 1, 1)).cuda()
                    vector_thresh_percent = int(num_channel * 1 / 3.1)
                    vector_thresh_value = torch.sort(channel_mean, dim=1, descending=True)[0][:, vector_thresh_percent]
                    vector_thresh_value = vector_thresh_value.view(batch_size, 1).expand(batch_size, num_channel)
                    vector = torch.where(channel_mean > vector_thresh_value,
                                         torch.zeros(channel_mean.shape).cuda(),
                                         torch.ones(channel_mean.shape).cuda())
                    vector_all = vector.detach().cpu().numpy()
                    channel_drop_num = int(num_channel * 1 / 3.2)
                    vector_all_new = np.ones((batch_size, num_channel), dtype=np.float32)
                    for q in range(batch_size):
                        vector_all_temp = np.ones((num_channel), dtype=np.float32)
                        zero_index = np.where(vector_all[q, :] == 0)[0]
                        num_zero_index = zero_index.size
                        if num_zero_index >= channel_drop_num:
                            dumy_index = npr.choice(zero_index, size=channel_drop_num, replace=False)
                        else:
                            zero_index = np.arange(num_channel)
                            dumy_index = npr.choice(zero_index, size=channel_drop_num, replace=False)
                        vector_all_temp[dumy_index] = 0
                        vector_all_new[q, :] = vector_all_temp
                    vector = torch.from_numpy(vector_all_new).cuda()
                    for m in range(batch_size):
                        index_channel = vector[m, :].nonzero()[:, 0].long()
                        index_channel = index_channel.detach().cpu().numpy().tolist()
                        mask_all[m, index_channel, :, :] = 1

                    # ----------------------------------- batch ----------------------------------------
                    cls_prob_before = F.softmax(output, dim=1)
                    x_new_view_after = x_new * mask_all

                    x_new_view_after = x_new_view_after.mean(-1).mean(-1)
                    x_new_view_after = x_new_view_after.view(x_new_view_after.size(0), -1)
                    x_new_view_after, _, _ = self.network.predict(x_new_view_after)

                    cls_prob_after = F.softmax(x_new_view_after, dim=1)

                    sp_i = torch.ones([2, batch_size]).long()
                    sp_i[0, :] = torch.arange(batch_size)
                    sp_i[1, :] = index
                    sp_v = torch.ones([batch_size])
                    one_hot_sparse = torch.sparse.FloatTensor(sp_i, sp_v,
                                                              torch.Size([batch_size, class_num])).to_dense().cuda()
                    before_vector = torch.sum(one_hot_sparse * cls_prob_before, dim=1)
                    after_vector = torch.sum(one_hot_sparse * cls_prob_after, dim=1)
                    change_vector = before_vector - after_vector - 0.0001
                    change_vector = torch.where(change_vector > 0, change_vector,
                                                torch.zeros(change_vector.shape).cuda())
                    th_fg_value = torch.sort(change_vector, dim=0, descending=True)[0][
                        int(round(float(batch_size) * 1 / 3))]
                    drop_index_fg = change_vector.gt(th_fg_value)
                    ignore_index_fg = 1 - drop_index_fg
                    not_01_ignore_index_fg = ignore_index_fg.nonzero()[:, 0]
                    mask_all[not_01_ignore_index_fg.long(), :] = 1

                    self.network.train()
                    mask_all = Variable(mask_all, requires_grad=True)
                    feat = feat * mask_all

                    feat = feat.mean(-1).mean(-1)
                    outputs = feat.view(feat.size(0), -1)

            prediction, exp, distances = self.network.predict(outputs)  # prediction from NCM

            loss_bx = self.ce_weight * self.ce(exp, targets_prep) + \
                      self.snnl_weight * self.snnl(outputs, targets_prep) + \
                      self.bce_weight * self.bce(prediction, targets)

            # add distillation to losses
            if iteration > 0 and self.features_dw > 0:
                outputs_old,_ = self.network2(inputs)
                outputs_old = outputs_old.detach()
                loss_bx = loss_bx + self.features_dw * (self.distill_loss(outputs, outputs_old))

            # if self supervision
            if self.ss_weight:
                inputs_ss = []
                targets_ss = []
                for index in range(len(inputs)):
                    ss_transformation = np.random.randint(self.discriminator.n_classes)
                    sample = inputs[index]
                    label = 0
                    if ss_transformation == 0:
                        pass
                    elif ss_transformation == 1:
                        sample = torch.rot90(sample, k=1, dims=[1,2])
                        label = 1
                    elif ss_transformation == 2:
                        sample = torch.rot90(sample, k=2, dims=[1,2])
                        label = 2
                    elif ss_transformation == 3:
                        sample = torch.rot90(sample, k=3, dims=[1,2])
                        label = 3
                    else:
                        raise NotImplementedError("Rotation invalid")
                    inputs_ss.append(sample)
                    targets_ss.append(label)

                inputs_ss = torch.stack(inputs_ss).to(self.device)
                _, feat_ss = self.network(inputs_ss)
                double_output = torch.cat((feat, feat_ss), 1)
                double_output = double_output.view(feat.size(0), -1)

                targets_prep_ss = torch.tensor(np.asarray(targets_ss)).to(self.device)

                p0 = self.discriminator(double_output)
                loss_rotation = self.ss_weight * self.ce(p0, targets_prep_ss)
                loss_bx = loss_bx + loss_rotation
                ss_loss += loss_rotation.item()

            if iteration == 0 or not self.nno and not (epoch == 0 and idx == 0):
                loss_bx.backward()
                optimizer.step()
                train_loss += loss_bx.item()

            # ## UPDATE MEANS ###
            # For the just learned classes it computes the online (moving) mean
            self.network.update_means(outputs, targets_prep.to('cpu'))

            if not self.deep_nno and not self.nno:
                self.tau.data = self.network.linear.get_average_dist(dim=-1)

            if self.deep_nno or (iteration == 0 and self.nno):
                # Update tau parameters
                self.deep_nno_handler.update_taus(prediction, targets_prep, self.num_classes)

            if self.augment_ops is not None:

                go = False
                if self.dataset == 'rgbd-dataset' or self.dataset == 'arid':
                    if iteration == 0 and (((idx * 128 + 1) % 2048)) and epoch <= 2:
                        if idx == 0 and (epoch == 0 or (epoch % (self.epochs // 2 + 1))):
                            go = True

                else:
                    if iteration == 0 and (((idx * 128 + 1) % 2048)) and epoch <= 2:
                        if idx == 0 and (epoch == 0 or (epoch % (self.epochs // 2 + 1))):
                            go = True
                if go:
                    transformations, levels = self.augment_ops.random_search(string_length=5,
                                                                             compute_fitness_f=self.test_closed_world,
                                                                             trainloader=subset_trainloader)
                    self.augment_ops.add_augment(transformations, levels)

                # for the batches != from first one
                transform, level = self.augment_ops.get_augment()
                transform = self.augment_ops.compose(transform, level)
                train_loader.dataset.set_transform(transform)

            # Display loss and accuracy
            _, predicted = prediction.max(1)
            total += targets_prep.size(0)
            correct += predicted.eq(targets_prep).sum().item()
            if self.ss_weight:
                _, ss_predicted = p0.max(1)
                ss_correct += ss_predicted.eq(targets_prep_ss).sum().item()

            if count % 20 == 0:
                print(f'[{int((100. * idx) / len(train_loader)):03d}%] == Loss: {train_loss / count:.3f}, '
                      f'Acc: {100. * correct / total:.3f} [{correct}/{total}]')

        self.logger.log_training(epoch, train_loss / len(train_loader), 100. * correct / total, iteration)

        if self.ss_weight:
            self.logger.log_ss(epoch, ss_loss / len(train_loader), 100. * ss_correct / total, iteration)

    def test_closed_world(self, test_loader, display=True):
        self.network.eval()
        correct = 0
        total = 0
        count = 0

        with torch.no_grad():
            for idx, (inputs, targets_prep) in enumerate(test_loader):
                count += 1
                inputs = inputs.to(self.device)
                outputs, _= self.network(inputs)
                outputs, _, distances = self.network.predict(outputs)
                targets_prep = targets_prep.to(outputs.device)
                _, predicted = outputs.max(1)
                total += inputs.size(0)
                correct += predicted.eq(targets_prep).sum().item()

        acc = 100. * correct / total
        if display:
            print(f"TEST ACCURACY: {acc}")
        return acc

    def test_open_set(self, test_loader, last_class):
        print(f"Final Tau: {self.tau.data.cpu().numpy()}")

        self.network.eval()
        correct = 0
        total = 0
        count = 0
        unk = 0.
        tp = 0.
        rejected = 0.
        total_rejected = 0.
        stat_distances = [0., 0., 0.]
        stat_pred = [0., 0., 0.]

        with torch.no_grad():
            for idx, (inputs, targets_prep) in enumerate(test_loader):
                count += 1

                inputs = inputs.to(self.device)
                outputs,_ = self.network(inputs)
                outputs, _, distances = self.network.predict(outputs)
                targets_prep = targets_prep.to(outputs.device)
                # changes the probabilities (distances) for known and unknown
                new_outputs = self.reject(outputs, distances, self.tau)
                _, predicted = new_outputs.max(1)

                # last_class is unknown class
                unk += (targets_prep == last_class).sum().item()
                # true positive
                tp += ((targets_prep == last_class) * predicted.eq(targets_prep)).sum().item()
                rejected += (predicted == last_class).sum().item()

                total += inputs.size(0)
                correct += predicted.eq(targets_prep).sum().item()
                stat_distances[0] += distances.min(dim=1)[0].sum()
                stat_distances[1] += distances.max(dim=1)[0].sum()
                stat_distances[2] += distances.mean(dim=1).sum()
                stat_pred[0] += outputs.min(dim=1)[0].sum()
                stat_pred[1] += outputs.max(dim=1)[0].sum()
                stat_pred[2] += outputs.mean(dim=1).sum()
                total_rejected += new_outputs[predicted == last_class].sum()

        acc = 100. * correct / total

        if rejected == 0:
            precision = 0
        else:
            precision = 100. * tp / rejected
        if unk == 0:
            recall = 0.
        else:
            recall = 100. * tp / unk
        if (precision + recall) == 0:
            f1score = 0
        else:
            f1score = 2. * (precision * recall) / (precision + recall)

        print(f"Accuracy: {acc:.2f}; Rej Rate {rejected / total:.2f}, Avg Pred Rej {total_rejected / total:.2f}"
              f"\n\t Min Pred {stat_pred[0] / total:.2f}, Max Pred {stat_pred[1] / total:.2f}, Mean_Pred {stat_pred[2] / total:.2f}"
              f"\n\t Min Dist {stat_distances[0] / total:.2f}, Max Dist {stat_distances[1] / total:.2f}, Mean Dist {stat_distances[2] / total:.2f}")

        return acc, precision, recall, f1score, rejected, unk

    def valid(self, epoch, valid_loader, optimizer, iteration, class_dict):
        self.network.eval()

        valid_loss = 0

        print(f"Tau: {self.tau.mean().cpu().detach().numpy()}")

        for idx, (inputs, targets_prep) in enumerate(valid_loader):

            optimizer.zero_grad()

            inputs = inputs.to(self.device)
            outputs,_ = self.network(inputs)
            predictions, _, distances = self.network.predict(outputs)  # prediction from NCM, not FC
            _, predicted = predictions.max(1)
            targets_prep = targets_prep.to(outputs.device)
            # one_hot_encoding targets
            targets = torch.zeros(inputs.shape[0], len(class_dict.keys())).to(self.device)
            targets.scatter_(1, targets_prep.view(-1, 1), 1).view(inputs.shape[0], -1)

            loss_bx = self.margin_loss(distances[predicted == targets_prep],
                                       targets[predicted == targets_prep], self.tau)

            if iteration == 0 or not self.nno:
                loss_bx.backward()
                optimizer.step()
                valid_loss += loss_bx.item()

        self.logger.log_valid(epoch, valid_loss / len(valid_loader), self.tau.mean().item(), iteration)

    def state_dict(self):
        state = {"tau": self.tau}
        return state

    def load_state_dict(self, state):
        if state["tau"] is not None:
            self.tau = Parameter(torch.tensor(state["tau"], device=self.device), requires_grad=True)
