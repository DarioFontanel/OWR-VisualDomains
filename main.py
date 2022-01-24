import numpy as np
import torch
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler

import argparse
import networks
from exemplars import Exemplars
from augment import AugmentOps
import os
import random
from data_loader import RODFolder, LightFilteredDatasetFolder
from logger import TensorboardXLogger
from trainer import Trainer


def save_ckpt(path, model, trainer):
    """ save current model """
    state = {
        "model_state": model.state_dict(),
        "trainer_state": trainer.state_dict()
    }
    torch.save(state, path)


def create_log_folder(log):
    os.makedirs(log, exist_ok=True)


def make_dataset(opts):
    full_order = np.load(opts.dataset_path + '/fixed_order.npy')

    if opts.dataset == 'rgbd-dataset':
        opts.data_class = RODFolder
        opts.batch_size = 128 if opts.batch_size == -1 else opts.batch_size
        opts.valid_batchsize = 64 if opts.valid_batchsize == -1 else opts.valid_batchsize
        transform_train = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.RandomCrop(64, padding=8, padding_mode='edge'),

        ])
        if not opts.search and opts.ss_weight:
            transform_train = transforms.Compose([transform_train,
                                                      transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5), ])
                
        transform_train = transforms.Compose([transform_train,
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (1., 1., 1.)),
        ])

        transform_basic = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.RandomCrop(64, padding=8, padding_mode='edge'),
            transforms.RandomHorizontalFlip(),
        ])

        transform_val = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.RandomCrop(64, padding=8, padding_mode='edge'), ])
        if not opts.no_tau_val:
            transform_val = transforms.Compose([transform_val,
                                                transforms.ColorJitter(brightness=0.1, saturation=0.1, hue=0.1), ])
        transform_val = transforms.Compose([transform_val,
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.5, 0.5, 0.5), (1., 1., 1.)),
                                            ])

        transform_test = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (1., 1., 1.)),
        ])
        dataset_path = opts.dataset_path + '/rgbd-dataset/rgbd-dataset_reorganized/'
        test_dataset_paths = [dataset_path]
        if not opts.search:
            test_dataset_paths = [f'{opts.dataset_path}/{test}/{test}_reorganized/' for test in opts.test]

    elif opts.dataset == 'synARID_crops_square':
        opts.data_class = RODFolder
        opts.batch_size = 128 if opts.batch_size == -1 else opts.batch_size
        opts.valid_batchsize = 64 if opts.valid_batchsize == -1 else opts.valid_batchsize
        transform_train = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.RandomCrop(64, padding=8, padding_mode='edge'),

        ])
        if not opts.search and opts.ss_weight:
            transform_train = transforms.Compose([transform_train,
                                                      transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5), ])
        transform_train = transforms.Compose([transform_train,
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              transforms.Normalize((0.5, 0.5, 0.5), (1., 1., 1.)),
                                              ])

        transform_basic = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.RandomCrop(64, padding=8, padding_mode='edge'),
            transforms.RandomHorizontalFlip(),
        ])

        transform_val = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.RandomCrop(64, padding=8, padding_mode='edge'), ])
        if not opts.no_tau_val:
            transform_val = transforms.Compose([transform_val,
                                                transforms.ColorJitter(brightness=0.1, saturation=0.1, hue=0.1), ])
        transform_val = transforms.Compose([transform_val,
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.5, 0.5, 0.5), (1., 1., 1.)),
                                            ])

        transform_test = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (1., 1., 1.)),
        ])
        dataset_path = opts.dataset_path + '/synARID_crops_square/synARID_crops_square_reorganized/'
        test_dataset_paths = [dataset_path]
        if not opts.search:
            test_dataset_paths = [f'{opts.dataset_path}/{test}/{test}_reorganized/' for test in opts.test]

    elif opts.dataset == 'arid_40k_dataset_crops':
        opts.data_class = RODFolder
        opts.batch_size = 128 if opts.batch_size == -1 else opts.batch_size
        opts.valid_batchsize = 64 if opts.valid_batchsize == -1 else opts.valid_batchsize
        transform_train = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.RandomCrop(64, padding=8, padding_mode='edge'),

        ])
        
        if not opts.search and opts.ss_weight:
            transform_train = transforms.Compose([transform_train,
                                                      transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5), ])
        
        transform_train = transforms.Compose([transform_train,
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              transforms.Normalize((0.5, 0.5, 0.5), (1., 1., 1.)),
                                              ])

        transform_basic = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.RandomCrop(64, padding=8, padding_mode='edge'),
            transforms.RandomHorizontalFlip(),
        ])

        transform_val = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.RandomCrop(64, padding=8, padding_mode='edge'), ])
        if not opts.no_tau_val:
            transform_val = transforms.Compose([transform_val,
                                                transforms.ColorJitter(brightness=0.1, saturation=0.1, hue=0.1), ])
        transform_val = transforms.Compose([transform_val,
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.5, 0.5, 0.5), (1., 1., 1.)),
                                            ])

        transform_test = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (1., 1., 1.)),
        ])
        dataset_path = opts.dataset_path + '/arid_40k_dataset_crops/arid_40k_dataset_crops_reorganized/'
        test_dataset_paths = [dataset_path]
        if not opts.search:
            test_dataset_paths = [f'{opts.dataset_path}/{test}/{test}_reorganized/' for test in opts.test]

    else:
        raise NotImplementedError('Dataset not implemented')

    start = 0
    end = 1000

    return opts, transform_train, transform_basic, transform_val, transform_test, full_order, dataset_path, test_dataset_paths, start, end


def get_search_params(opts, param_name, param_value, best):
    if param_name == 'lr':
        opts.lr = param_value
        opts.decay = 0.
        opts.ce = 1. if opts.method == 'bdoc' else 0.
        opts.bce = 0. if opts.method == 'bdoc' else 1.
        opts.snnlw = 0.
        opts.features_dw = 0.
        opts.tau_factor = 0.
        opts.tau_lr = 0.
    elif param_name == 'decay':
        opts.lr = best['lr']
        opts.decay = param_value
        opts.ce = 1. if opts.method == 'bdoc' else 0.
        opts.bce = 0. if opts.method == 'bdoc' else 1.
        opts.snnlw = 0.
        opts.features_dw = 0.
        opts.tau_factor = 0.
        opts.tau_lr = 0.
    elif param_name == 'ce' and opts.bdoc:
        opts.lr = best['lr']
        opts.decay = best['decay']
        opts.ce = param_value
        opts.bce = 0.
        opts.snnlw = 0.
        opts.features_dw = 0.
        opts.tau_factor = 0.
        opts.tau_lr = 0.
    elif param_name == 'bce' and not opts.bdoc:
        opts.lr = best['lr']
        opts.decay = best['decay']
        opts.ce = 0.
        opts.bce = param_value
        opts.snnlw = 0.
        opts.features_dw = 0.
        opts.tau_factor = 0.
        opts.tau_lr = 0.
    elif param_name == 'features_dw':
        opts.lr = best['lr']
        opts.decay = best['decay']
        opts.ce = best['ce'] if opts.method == 'bdoc' else 0.
        opts.bce = best['bce'] if opts.method != 'bdoc' else 0.
        opts.snnlw = 0.
        opts.features_dw = param_value
        opts.tau_factor = 0.
        opts.tau_lr = 0.
    elif param_name == 'snnlw' and opts.bdoc:
        opts.lr = best['lr']
        opts.decay = best['decay']
        opts.ce = best['ce'] if opts.method == 'bdoc' else 0.
        opts.bce = best['bce'] if opts.method != 'bdoc' else 0.
        opts.snnlw = param_value
        opts.features_dw = best['features_dw']
        opts.tau_factor = 0.
        opts.tau_lr = 0.
    elif param_name == 'tau_factor' and opts.deep_nno:
        opts.lr = best['lr']
        opts.decay = best['decay']
        opts.ce = 0.
        opts.bce = best['bce']
        opts.snnlw = 0.
        opts.features_dw = best['features_dw']
        opts.tau_factor = param_value
    elif param_name == 'tau_lr' and opts.bdoc:
        opts.lr = best['lr']
        opts.decay = best['decay']
        opts.ce = best['ce']
        opts.bce = 0.
        opts.snnlw = best['snnlw']
        opts.features_dw = best['features_dw']
        opts.tau_lr = param_value
    return opts

def get_params(opts):
    if opts.search:
        opts.epochs_init = 12
        opts.epochs = int(opts.epochs_init * opts.incremental_classes / opts.initial_classes)
        opts.orders = 1
        opts.validation_size = 80
        opts.unk = 2
        opts.unk_step = 1
        opts.initial_classes = 5
        opts.CLASSES = 11
    else:
        # standard OWR
        opts.initial_classes = 11
        opts.incremental_classes = 5
        opts.CLASSES = 51
        opts.unk = 25
        opts.unk_step = 5
        opts.validation_size = 50
        opts.orders = 5
        opts.epochs_init = 12 if opts.epochs_init == -1 else opts.epochs_init
        opts.epochs = int(
            opts.epochs_init * opts.incremental_classes / opts.initial_classes) if opts.epochs == -1 else opts.epochs

        if opts.rsda:
            opts.epochs_init = 25
            opts.epochs = 12

    if opts.dataset == 'synARID_crops_square':
        if opts.rsda or opts.ssw:
            opts.epochs_init = 120
        else:
            opts.epochs_init = 70

        opts.epochs = 35

    return opts

def load(opts):
    print("\nLoading best configuration...\n")

    # load the .npy
    config_path = os.path.realpath(".") + f'/{opts.dataset_path}/{opts.dataset}/additionals/{opts.config}'
    assert os.path.exists(config_path), f"Error, {config_path} does not exist. "
    best = np.load(config_path, allow_pickle=True).item()

    for idx, (param_name, best_value) in enumerate(best.items()):
        if param_name == 'lr':
            opts.lr = best_value
        elif param_name == 'decay':
            opts.decay = best_value
        elif param_name == 'ce':
            opts.ce = best_value
        elif param_name == 'bce':
            opts.bce = best_value
        elif param_name == 'features_dw':
            opts.features_dw = best_value
        elif param_name == 'snnlw':
            opts.snnlw = best_value
        elif param_name == 'tau_factor':
            opts.tau_factor = best_value
        elif param_name == 'tau_lr':
            opts.tau_lr = best_value
        print(f'{param_name} = {best_value} loaded')

    print(f"\nConfiguration /{opts.config}/ loaded\n")

    return opts

def perform_search(opts):
    print("\nStarting search...\n")

    opts = get_params(opts)

    all_incremental_classes = [1, 2, 4]

    network_exp = {'lr': [0.01, 0.1, 1.], 'decay': [1e-5, 1e-4, 1e-3]}
    if opts.bdoc:
        network_exp.update({'ce': [0.001, 0.01, 0.1, 1.]})
    else:
        network_exp.update({'bce': [0.01, 0.1, 1.]})
    network_exp.update({'features_dw': [0.01, 0.1, 1.]})
    if opts.bdoc:
        network_exp.update({'snnlw': [0.01, 0.1, 1.]})
        network_exp.update({'tau_lr': [0.01, 0.1, 0.2, 0.3, 1.]})
    if opts.deep_nno:
        network_exp.update({'tau_factor': [1., 2., 3.]})

    best_params = {}
    # for all network parameters
    for index, (param_name, values) in enumerate(network_exp.items()):
        results = {}
        # for all incremental classes
        for incremental_classes in all_incremental_classes:
            opts.incremental_classes = incremental_classes
            results[opts.incremental_classes] = []
            # for all value of each parameter
            for param_value in values:
                opts = get_search_params(opts, param_name, param_value, best_params)
                metric = main(opts, (param_name, param_value))
                results[opts.incremental_classes].append(metric)

        # RANKING MEAN
        rank = np.zeros((len(all_incremental_classes), len(values)))
        j = 0
        for incremental_classes in all_incremental_classes:
            r = []
            for i in results[incremental_classes]:
                r.append(sorted(results[incremental_classes])[::-1].index(i) + 1)
            rank[j] = np.asarray(r)
            j = j + 1

        mean = np.mean(rank, axis=0)
        min_rank, idx = torch.tensor(mean).min(0)
        best_params[param_name] = values[idx.item()]

    # Save
    np.save(f'{opts.dataset_path}/{opts.dataset}/additionals/{opts.name}_{opts.dataset}_best_config', best_params)


def main(opts, search_params=None):
    # FIX SEEDS.
    torch.manual_seed(opts.seed)
    torch.cuda.manual_seed(opts.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(opts.seed)
    random.seed(opts.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # get Dataset
    opts, transform_train, transform_basic, transform_val, transform_test, full_order, dataset_path, test_datasets, \
    start, end = make_dataset(opts)

    # Stable parameters
    batch_size = opts.batch_size  # Batch size
    nb_initial = opts.initial_classes  # Classes in initial group
    nb_cl = opts.incremental_classes  # Classes per group
    nb_orders = opts.orders # Orders to perform
    epochs_strat = [80]  # Epochs where learning rate gets decreased
    lr_factor = opts.lr_factor  # Learning rate decrease factor
    wght_decay = opts.decay  # Weight Decay
    print(opts)

    # set up metrics
    icl_steps = int((opts.CLASSES - nb_initial - opts.unk) / nb_cl + 1) if nb_cl != 0 else 1
    measures = int(2 + opts.unk / opts.unk_step) if opts.unk_step != 0 else 2
    top1_acc_list = {}
    for test in test_datasets:
        top1_acc_list.update({test: np.zeros((icl_steps, measures, nb_orders))})

    # logger
    create_log_folder(f"{opts.logs}/{opts.dataset}/")

    if opts.search and search_params is not None:
        create_log_folder(f'{args.logs}/{opts.dataset}/search/')
        results_file = open(f"{opts.logs}/{opts.dataset}/search/{opts.name}_search.txt", "a")
        param_name = search_params[0]
        param_value = search_params[1]
        threshold_validation = (param_name == 'tau_factor' or param_name == 'tau_lr')
    else:
        results_file = open(f"{opts.logs}/{opts.dataset}/{opts.name}.txt", "a")

    # Launch the different orders
    for iteration_total in range(nb_orders):
        log_path = f'{opts.logs}/{opts.dataset}/search/{opts.name}_{nb_initial}then{nb_cl}_exp_{param_name}{param_value}' \
            if opts.search else f'{opts.logs}/{opts.dataset}/{opts.name}_ordine{iteration_total}'

        logger = TensorboardXLogger(log_path)
        logger.save_opts(opts)

        class_dict = {}  # dictionary of class - inverted order
        order = full_order[nb_cl][:(opts.CLASSES)] if opts.search else full_order[iteration_total]
        print(order)

        print(f'Order {iteration_total} starting ...')

        # set up network
        network = networks.ResNet18(classes=opts.initial_classes, pretrained=None, relu=not opts.no_relu,
                                    deep_nno=(opts.deep_nno or opts.nno)).to(device)
        # set up discriminator
        discriminator = networks.Discriminator(batch_size=batch_size, n_feat=256, n_classes=4).to(device)

        # setup trainer
        augment_ops = AugmentOps(transform_basic) if opts.rsda else None
        trainer = Trainer(opts, network, discriminator, device, logger, augment_ops)
        exemplar_handler = Exemplars(network, device)

        # ## INCREMENTAL TRAINING ##
        # Incremental step: in each step the model learns new classes
        for iteration in range(icl_steps):
            # Prepare the training data for the current batch of classes
            start_class = (iteration > 0) * (nb_initial + (iteration - 1) * nb_cl)
            last_class = nb_initial + iteration * nb_cl
            actual_cl = order[range(start_class, last_class)]

            for i, el in enumerate(actual_cl):
                class_dict[el] = i + start_class  # dict: class : index. (Es. 4:0, 7:1, ...)

            # make the new dataset
            train_set = opts.data_class(root=dataset_path, split='train', classes=actual_cl,
                                        target_transform=class_dict, transform=transform_train)
            val_set = opts.data_class(root=dataset_path, split='val', classes=actual_cl,
                                      target_transform=class_dict, transform=transform_test)
            if not opts.search:
                train_set.samples.extend(val_set.samples)

            # if bdoc
            if not opts.no_tau_val:
                set_tau = train_set.reduce(opts.validation_size)
                val_set_tau = LightFilteredDatasetFolder(samples=set_tau, target_transform=class_dict,
                                                         transform=transform_val)

            sampler = None
            shuffle = True
            # If iteration is > 0 then the batch should be weighted between old and new
            if iteration > 0 and not opts.nno:  # make exemplars weighted
                len_no_ex = len(train_set.samples)
                ex_samples = exemplar_handler.get_exemplar_samples()
                train_set.samples.extend(ex_samples)
                len_with_ex = len(train_set.samples)
                weights = np.ones(len_with_ex)
                weights[len_no_ex:] *= opts.ratio / ((len_with_ex - len_no_ex + 0.0) / len_with_ex)

                shuffle = False
                sampler = WeightedRandomSampler(weights=weights, num_samples=len_with_ex)

            trainloader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle,
                                     num_workers=opts.workers, sampler=sampler)

            # Setup Network
            if iteration > 0:
                trainer.next_iteration(len(actual_cl))
                network.to(device)

            # Start iteration
            print(f'Batch of classes number {iteration + 1} arrives ...')
            # Setup epoch and optimizer
            epochs = opts.epochs_init if iteration == 0 else opts.epochs

            learning_rate = opts.lr
            if opts.nno:
                wght_decay = 0.0

            optimizer = optim.SGD(filter(lambda p: p.requires_grad, network.parameters()), lr=learning_rate,
                                  momentum=0.9, weight_decay=wght_decay, nesterov=False)

            lr_strat = [epstrat * epochs // 100 for epstrat in epochs_strat]
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_strat, gamma=lr_factor)

            # restore checkpoint after training
            if opts.restore_ckpt and iteration == 0:
                if opts.ckpt is not None:
                    ckpt_path = opts.ckpt
                else:
                    ckpt_path = f"checkpoints/{opts.dataset}/{opts.name}_ordine{iteration_total}_{iteration}"
                assert os.path.isfile(ckpt_path), f"Error, checkpoint {ckpt_path} does not exist. "

                state = torch.load(ckpt_path)
                state['model_state']['linear.variance'].unsqueeze_(0)
                network.load_state_dict(state['model_state'])
                epochs = 0

            subset = list(train_set.samples)
            random.shuffle(subset)
            subset_train_set = LightFilteredDatasetFolder(samples=subset[start:end],
                                                          target_transform=class_dict, transform=None)

            subset_trainloader = DataLoader(subset_train_set, batch_size=batch_size, shuffle=True,
                                     num_workers=opts.workers, sampler=None)
            for epoch in range(epochs):
                trainer.train(epoch, trainloader, subset_trainloader, optimizer, class_dict, iteration)
                scheduler.step()

            # save checkpoint after training
            if opts.save_ckpt and (iteration == 0 or iteration == icl_steps - 1):
                create_log_folder(f"checkpoints/{opts.dataset}/")
                ckpt_path = f"checkpoints/{opts.dataset}/{opts.name}_ordine{iteration_total}_{iteration}"
                save_ckpt(ckpt_path, network, trainer)

            # ## VALIDATION PHASE ###
            if not opts.no_tau_val:
                # add exemplars to val dataset
                if iteration > 0:
                    ex_samples_val = exemplar_handler.get_exemplar_samples(train=False)
                    val_set_tau.samples.extend(ex_samples_val)
                print("Starting threshold(s) validation...")
                # dataset
                validloader = torch.utils.data.DataLoader(val_set_tau, batch_size=args.valid_batchsize, shuffle=False,
                                                          num_workers=args.workers)
                # optimizer
                valid_optimizer = optim.SGD([trainer.tau], lr=args.tau_lr, momentum=0.9, weight_decay=wght_decay,
                                            nesterov=False)

                for epoch in range(opts.epochs_val):
                    trainer.valid(epoch, validloader, valid_optimizer, iteration, class_dict)

            # ## EXEMPLARS MANAGEMENT ###
            known_cl = last_class  # number of total known classes
            m = opts.memory / known_cl
            exemplars_m = m * opts.exemplars_m
            valid_m = m * opts.valid_m

            # Reduce exemplar sets for known classes
            if iteration > 0 and not opts.nno:
                exemplar_handler.reduce_exemplar_sets(exemplars_m, valid_m)
            # Load exemplars for the classes known until now
            if not opts.nno:
                print(f"Constructing exemplar set for class:", end=" ")
                for y in range(start_class, last_class):
                    print(f"{y}", end=" ")

                    # Training exemplar
                    class_set = LightFilteredDatasetFolder(samples=train_set.get_samples_class(order[y]),
                                                           target_transform=None, transform=transform_test)
                    loader = torch.utils.data.DataLoader(class_set, batch_size=batch_size, shuffle=False,
                                                         num_workers=args.workers)
                    exemplar_handler.construct_exemplar_set(loader, exemplars_m, y, type='train')

                    if not opts.no_tau_val:
                        # Aggiunge agli exemplars-validation
                        class_set = LightFilteredDatasetFolder(samples=val_set_tau.get_samples_class(order[y]),
                                                               target_transform=None, transform=transform_test)
                        loader = torch.utils.data.DataLoader(class_set, batch_size=batch_size, shuffle=False,
                                                             num_workers=args.workers)
                        exemplar_handler.construct_exemplar_set(loader, valid_m, y, type='val')
                print(f"Done")

            # ## TEST ###
            for test_dataset in test_datasets:
                # Testing from class 0 to the latest learned one
                test_set = opts.data_class(root=test_dataset, split='val' if opts.search else 'test',
                                           classes=order[range(0, last_class)], target_transform=class_dict,
                                           transform=transform_test)

                testloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False,
                                                         num_workers=opts.workers)

                acc = trainer.test_closed_world(testloader)
                top1_acc_list[test_dataset][iteration, 0, iteration_total] = acc

                for i in range(int(opts.unk / opts.unk_step + 1)):
                    idx_classes = range(0, last_class)  # classi known
                    if i > 0:
                        idx_classes = list(range(opts.CLASSES - opts.unk, opts.CLASSES - opts.unk + i * opts.unk_step))

                    test_set = opts.data_class(root=test_dataset, split='val' if opts.search else 'test',
                                               classes=order[idx_classes],
                                               target_transform=class_dict, transform=transform_test)

                    testloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False,
                                                             num_workers=opts.workers)

                    acc, precision, recall, f1score, rejected, unk = trainer.test_open_set(testloader, last_class)

                    top1_acc_list[test_dataset][iteration, 1 + i, iteration_total] = acc

                logger.log_test(top1_acc_list[test_dataset][iteration, :, iteration_total], iteration, test_dataset)

        # SAVE RESULTS ###
        # File results
        if opts.search:
            results_file.write(f'\n{nb_initial}_then_{nb_cl} \n\n' f'lr: {opts.lr} \n' f'decay: {opts.decay} \n'
                               f'ce: {opts.ce} \n' f'bce: {opts.bce} \n' f'snnl: {opts.snnlw} \n'
                               f'features_dw: {opts.features_dw}\n'  f'tau_factor: {opts.tau_factor}\n'
                               f'tau_lr: {opts.tau_lr}\n')
        else:
            results_file.write(f'\n\n\nOrder {iteration_total}:')

        for idx, column in enumerate(top1_acc_list[test_dataset][:, :, iteration_total].T):
            if idx == 0:
                results_file.write(f'\n\nClosed world without rejection\n')
            elif idx == 1:
                results_file.write(f'\n\nClosed world with rejection\n')
            elif idx == 2:
                continue
            elif idx == 3:
                results_file.write(f'\n\nOpen set with 5 unknown classes\n')

            for row in column:
                results_file.write(f'{row:.3f} | ')
        results_file.write('\n\n')
        logger.writer.close()

        # COMPUTE HARMONIC MEAN
        if opts.search:
            if threshold_validation:
                avg_with_rej = top1_acc_list[test_dataset][:, 1, 0].mean()
                avg_open_set = top1_acc_list[test_dataset][:, -1, 0].mean()
                measure = (2 * avg_with_rej * avg_open_set) / (avg_with_rej + avg_open_set)
            else:
                measure = top1_acc_list[test_dataset][:, 0, 0].mean()
            return measure

    # ## SAVE RESULTS ###
    for test_dataset in test_datasets:
        log_path = f'{opts.logs}/{opts.dataset}->{test_dataset.split("/")[1]}/{opts.name}-table'
        logger = TensorboardXLogger(log_path)
        logger.add_results(top1_acc_list[test_dataset])
        logger.save_opts(opts)
        logger.writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="On the Challenges of OWR under Shifting Visual Domains")
    parser.add_argument("--batch_size", type=int, default=-1,  # 128
                        help="Batch size (both for training and test)")
    parser.add_argument("--valid_batchsize", help="Batch size for validation", type=int, default=-1)
    parser.add_argument("--memory", help="Memory per class at the end", type=int, default=2000)

    parser.add_argument("--seed", help="Seed to use for the run", type=int, default=1993)
    parser.add_argument("--no_relu", help="Whether to use relu in the last block of resnet", action='store_true',
                        default=False)

    parser.add_argument("--logs", help="Name of log folder", type=str, default='logs')
    parser.add_argument("--orders", help="Number of orders", type=int, default=5)
    parser.add_argument("--name", help="Name of the experiments", type=str, default='exp')
    parser.add_argument("--dataset", help="Name of the dataset used for training", type=str,
                        choices=['rgbd-dataset', 'arid_40k_dataset_crops', 'synARID_crops_square'])
    parser.add_argument("--test", help="Name of the dataset(s) used for testing", nargs='+', type=str, default='all')
    parser.add_argument("--dataset_path", help="Where data are located", type=str, default='data')
    parser.add_argument("--workers", help="Number of workers for data loader", type=int, default=2)

    parser.add_argument("--epochs_init", help="Initial epochs", type=int, default=-1)
    parser.add_argument("--epochs", help="Epochs after first iteration", type=int, default=-1)
    parser.add_argument("--epochs_val", help="Epochs for the validation", type=int, default=20)
    parser.add_argument("--initial_classes", help="initial classes for validation", type=int, default=11)
    parser.add_argument("--incremental_classes", help="incremental classes validation", type=int, default=5)
    parser.add_argument("--CLASSES", help="Total number of classes for validation", type=int, default=51)
    parser.add_argument("--lr", help="LR after first iteration", type=float, default=1.)
    parser.add_argument("--decay", help="Weight Decay", type=float, default=0.00001)
    parser.add_argument("--lr_factor", help="LR scale factor", type=float, default=0.1)

    parser.add_argument("--unk", help="How many unknown classes must be considered", type=int, default=25)
    parser.add_argument("--unk_step", help="Incremental step for unknown classes", type=int, default=5)
    parser.add_argument("--tau_factor", help="Negative weight for computing tau (DeepNNO)", type=float, default=2.)
    parser.add_argument("--tau_lr", help="LR for threshold on held out set (B-DOC)", type=float, default=0.1)

    parser.add_argument("--exemplars_m", help="Percentage of exemplars for training", type=float, default=0.8)
    parser.add_argument("--valid_m", help="Percentage of exemplars for validation", type=float, default=0.2)
    parser.add_argument("--ratio", help="Exemplar ratio in batch", type=float, default=0.4)
    parser.add_argument("--validation_size", help="Number of samples for held out dataset", type=int, default=50)

    # PIPELINE
    parser.add_argument("--search", help="Whether to perform initial grid search or not", default=False, action='store_true')
    parser.add_argument("--config", help="Name of configuration to load", type=str, default=None)

    # TAU METHODS
    parser.add_argument("--deep_nno", help="Whether to update taus after iterations (use standard DeepNNO)",
                        default=False, action='store_true')
    parser.add_argument("--no_tau_val", help="Whether to not learn tau in validation",
                        default=False, action='store_true')
    parser.add_argument("--nno", help="Use shallow NNO method? Default no",
                        default=False, action='store_true')
    parser.add_argument("--multiple_taus", help="Whether to use only one global tau or one tau per class",
                        action='store_false', default=True)
    parser.add_argument("--bdoc", help="Use bdoc by clustering method", action='store_true', default=False)

    # DOMAIN ADAPTATION METHODS
    parser.add_argument("--rsda", help="Use RSDA by Volpi et all", action='store_true', default=False)
    parser.add_argument("--ssw", help="Self supervised weight for Rotation Task", type=float, default=0.)
    parser.add_argument("--self_challenging", help="Use self challenging (SC) for DA", action='store_true', default=False)

    # checkpoints
    parser.add_argument("--save_ckpt", help="Whether to save the model after the training",
                        default=False, action='store_true')
    parser.add_argument("--restore_ckpt", help="Whether to restore the model before training (set epoch = 0)",
                        default=False, action='store_true')
    parser.add_argument("--ckpt", help="Path to the ckpt (if not specified, look for default",
                        default=None)

    # LOSS Methods
    parser.add_argument("--features_dw", help="Features distillation weight", type=float, default=1.)
    parser.add_argument("--snnlw", help="SNNL weight", type=float, default=0.)
    parser.add_argument("--ce", help="CE weight", type=float, default=0.)
    parser.add_argument("--bce", help="BCE_nologits weight", type=float, default=0.)

    args = parser.parse_args()

    # TEST DATASETS
    if args.test == 'all':
        args.test = ['rgbd-dataset', 'arid_40k_dataset_crops', 'synARID_crops_square']

    # METHODS
    if args.nno or args.deep_nno:  # dnno
        args.no_tau_val = True
        args.multiple_taus = False
        args.ce = 0.
        args.snnlw = 0.
        args.bce = 1.
        args.features_dw = 1.

        args.method = 'nno' if args.nno else 'deepnno'

    if args.bdoc:
        args.no_tau_val = False
        args.multiple_taus = True
        args.deep_nno = False
        args.nno = False

        args.method = 'bdoc'

    if args.search:
        # Grid search section
        perform_search(args)
    else:
        # Load search params
        if args.config is None:
            raise NotImplementedError("A configuration must be provided to start the training phase.")
        else:
            if args.config == 'default':
                args.config = f'{args.method}_{args.dataset}_best_config.npy'
            args = load(args)

        args = get_params(args)
        main(args)

    print("\nDone\n")
