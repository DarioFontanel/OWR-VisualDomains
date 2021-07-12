import numpy as np
import torchvision.transforms as T

def identity(image, level=None):
    return image

class hue:
    def __init__(self, level):
        self.level = level
    def __call__(self, image):
        return T.ColorJitter(hue=0.5)(image)

class brightness:
    def __init__(self, level):
        self.level = level
    def __call__(self, image):
        return T.ColorJitter(brightness=0.5)(image)

class contrast:
    def __init__(self, level):
        self.level = level
    def __call__(self, image):
        return T.ColorJitter(contrast=0.5)(image)

class saturation:
    def __init__(self, level):
        self.level = level
    def __call__(self, image):
        return T.ColorJitter(saturation=0.5)(image)

class AugmentOps:
    '''
    Class to handle all the search procedures.
    Currently implemented: random search and evolution search
    '''

    def __init__(self, basic_transformations):
        self.transformations = [['identity']]
        self.levels = [[None]]
        self.basic_transformations = basic_transformations
        self.pil_to_tensor = T.ToTensor()
        self.tensor_to_pil = T.ToPILImage()
        self.normalize = T.Normalize((0.5, 0.5, 0.5), (1., 1., 1.))
        self.transformation_list = ['brightness', 'contrast', 'saturation', 'hue']

        # it initilizes the possible levels of each transformation
        self.define_code_correspondances()

    def get_augment(self):
        rnd_transf_idx = np.random.randint(len(self.transformations))
        transf = self.transformations[rnd_transf_idx]
        level = self.levels[rnd_transf_idx]

        if transf == ['identity']:  # do nothing for 'identity'
            return None, None
        else:
            return transf, level

    def add_augment(self, transformations, levels):
        if transformations is not None:
            self.transformations.append(transformations)
        if levels is not None:
            self.levels.append(levels)

        print(f"\nTransformations until now: {self.transformations}")
        print(f"Levels until now: {self.levels}\n")

    def random_search(self, string_length, compute_fitness_f, trainloader, no_iters=None):

        '''
        Sampling random image transformations and testing them on a provided model.
        Referring to the paper, this is Algorithm 1.

            no_iters: number of iterations.
            string_length: number of transformations to be concatenated.
            save_file_name: file name used to save .png and .pkl outputs.
            compute_fitness_f: test function associated with the desired model.
            original_images: images to give in input to compute_fitness_f.
            args: other input eventually required by compute_fitness_f (e.g., ground truth labels, sess, etc.)

        '''
        print("\nStarting Random Search Data Augmentation...")

        all_accuracies = []
        all_best_accuracies = []
        all_transformations = []
        all_levels = []
        current_minimum = 100.

        number_fitness_evals = 0
        no_iters = int(len(trainloader.dataset) / 10.)

        for t in range(no_iters):
            transformations, levels = self.decode_string(transf_string='random_' + str(string_length))
            transforms = self.compose(transformations, levels)
            trainloader.dataset.set_transform(transforms)

            number_fitness_evals += 1

            accuracy = compute_fitness_f(trainloader, display=False)

            all_accuracies.append(accuracy)

            if accuracy < current_minimum:
                print('%d Current minimum: [%.4f], # fitness evals: [%d]' % (t, accuracy, number_fitness_evals))
                current_minimum = accuracy

                all_best_accuracies.append(accuracy)
                all_transformations.append(transformations)
                all_levels.append(levels)

        # le ultime sono le best tra tutte (fitness più bassa)
        print('Overall minimum accuracy: [%.4f]' % (all_best_accuracies[-1]))
        print('_'.join(all_transformations[-1].tolist()))
        print("End Random Search Data Augmentation")

        return all_transformations[-1].tolist(), all_levels[-1]

    def genetic_algorithm(self, no_iters, pop_size, string_length, trainloader, mutation_rate, compute_fitness_f,
                          *args):
        '''
        Sampling random image transformations and testing them on a provided model.
        Referring to the paper, this is Algorithm 2.

            no_iters: number of iterations.
            string_length: number of transformations to be concatenated.
            mutation_rate: a value in [0.0,1.0]
            save_file_name: file name used to save .png and .pkl outputs.
            compute_fitness_f: test function associated with the desired model.
            original_images: images to give in input to compute_fitness_f.
            args: other input eventually required by compute_fitness_f (e.g., ground truth labels, sess, etc.)
        '''

        min_accuracy = 100.  # initialized with the maximum value
        current_minimum = 100.  # initialized with the maximum value

        number_fitness_evals = 0
        number_fitness_needed = pop_size

        pop_accuracies = []
        pop_probabilities = []
        pop_transformations = []
        pop_levels = []

        min_accs = []
        min_transfs = []
        min_levels = []

        all_fitnesses = []

        print('Initializing population')

        for p in range(pop_size):  # number of items in the population

            transformations, levels = self.decode_string(transf_string='random_' + str(string_length))
            transforms = self.compose(transformations, levels)
            trainloader.dataset.set_transform(transforms)

            number_fitness_evals += 1

            target_accuracy = compute_fitness_f(trainloader, display=False)

            pop_accuracies.append(target_accuracy)
            pop_transformations.append(transformations)
            pop_levels.append(levels)

        pop_probabilities = (1. - np.array(pop_accuracies)) / np.sum(1. - np.array(pop_accuracies))

        current_minimum = np.min(pop_accuracies)
        print('Current minimum:', str(current_minimum), '# fitness evals', str(number_fitness_evals))

        min_accs.append(current_minimum)

        all_fitnesses.append(current_minimum)

        pop_transformations = [arr.tolist() for arr in pop_transformations]

        min_transfs.append(pop_transformations[np.argmin(pop_accuracies)])
        min_levels.append(pop_levels[np.argmin(pop_accuracies)])

        print('Running evolution search')

        for step in range(no_iters):  # number of iters for the evolution search

            if current_minimum == 0.0:
                break

            new_pop_accuracies = []
            new_pop_transformations = [None for i in range(pop_size)]
            new_pop_levels = [None for i in range(pop_size)]

            for p in range(int(pop_size / 2)):
                # randomly choose two parents to be mated <3
                idx_1 = np.random.choice(pop_size, p=pop_probabilities)
                idx_2 = np.random.choice(pop_size, p=pop_probabilities)

                transformations_1 = pop_transformations[idx_1]
                transformations_2 = pop_transformations[idx_2]
                levels_1 = pop_levels[idx_1]
                levels_2 = pop_levels[idx_2]

                # cutting transformations/levels on a random point and
                crossover_point = np.random.randint(string_length)

                new_transformations_1 = transformations_1[:crossover_point] + transformations_2[crossover_point:]
                new_levels_1 = levels_1[:crossover_point] + levels_2[crossover_point:]
                new_transformations_2 = transformations_2[:crossover_point] + transformations_1[crossover_point:]
                new_levels_2 = levels_2[:crossover_point] + levels_1[crossover_point:]

                # adding the new offspring to the new population
                new_pop_transformations[p] = new_transformations_1
                new_pop_levels[p] = new_levels_1
                new_pop_transformations[int(p + pop_size / 2)] = new_transformations_2
                new_pop_levels[int(p + pop_size / 2)] = new_levels_2

            # mutating some genes
            for i, transformations in enumerate(new_pop_transformations):
                for j, transf in enumerate(transformations):
                    if np.random.rand() < mutation_rate:
                        new_pop_transformations[i][j] = np.random.choice(self.transformation_list, 1)[0]
                        new_pop_levels[i][j] = np.random.choice(list(self.code_to_level_dict[new_pop_transformations[i][j]].values()), 1)[0]

            for transformations, levels in zip(new_pop_transformations, new_pop_levels):
                transforms = self.compose(transformations, levels)
                trainloader.dataset.set_transform(transforms)

                number_fitness_evals += 1

                target_accuracy = compute_fitness_f(trainloader, display=False)

                new_pop_accuracies.append(target_accuracy)

            pop_transformations = new_pop_transformations
            pop_levels = new_pop_levels
            pop_accuracies = new_pop_accuracies

            pop_probabilities = (1. - np.array(pop_accuracies)) / np.sum(1. - np.array(pop_accuracies))

            if np.min(pop_accuracies) < current_minimum:
                current_minimum = np.min(pop_accuracies)
                print(str(step), '- Current minimum:', str(current_minimum), '#number fitness evals', str(number_fitness_evals))
                print(pop_transformations[np.argmin(pop_accuracies)])
                print(pop_levels[np.argmin(pop_accuracies)])

                number_fitness_needed = number_fitness_evals

                min_accs.append(current_minimum)
                min_transfs.append(pop_transformations[np.argmin(pop_accuracies)])
                min_levels.append(pop_levels[np.argmin(pop_accuracies)])

            all_fitnesses.append(current_minimum)

        return min_transfs[np.argmin(min_accs)], min_levels[np.argmin(min_accs)]

    def compose (self, transforms=None, levels=None):
        transf = []
        transf.append(self.basic_transformations)

        # if not identity
        if transforms is not None and levels is not None:
            for transform, level in zip(transforms, levels):
                t = globals()[transform]
                transform_class = t(level)
                transf.append(transform_class)

        transf.append(self.pil_to_tensor)
        transf.append(self.normalize)

        return T.Compose(transf)

    def decode_string(self, transf_string):

        '''
        Code to decode the string used by the genetic algorithm
        String example: 't1,l1_3,t4,l4_0,t0,l0_1'. First transformation is the one
        associated with index '1', with level set to '3', and so on.
        'random_N' with N integer gives N rnd transformations with rnd levels.
        '''

        if 'random' in transf_string:
            transformations = np.random.choice(self.transformation_list,
                                         int(transf_string.split('_')[-1]))  # the string is 'random_N'
            levels = [np.random.choice(list(self.code_to_level_dict[t].values()), 1)[0] for t in
                      transformations]  # list() to make it compatible with Python3
        else:
           raise NotImplementedError(f"{transf_string} not implemented")

        return transformations, levels

    def code_to_transf(self, code):

        '''
        Takes in input a code (e.g., 't0', 't1', ...) and gives in output
        the related transformation.
        '''

        return self.code_to_transf_dict[code]

    def code_to_level(self, transformation, code):

        '''
        Takes in input a transfotmation (e.g., 'invert', 'colorize', ...) and
        a level code (e.g., 'l0_1', 'l1_3', ...) and gives in output the related level.
        '''

        return self.code_to_level_dict[transformation][code]

    def define_code_correspondances(self):

        '''
        Define the correpondances between transformation/level codes
        and the actual types and values.
        '''

        self.code_to_transf_dict = dict()

        self.code_to_transf_dict['t1'] = 'brightness'
        self.code_to_transf_dict['t2'] = 'contrast'
        self.code_to_transf_dict['t3'] = 'saturation'
        self.code_to_transf_dict['t4'] = 'hue'

        self.code_to_level_dict = dict()

        for k in self.transformation_list:
            self.code_to_level_dict[k] = dict()

        # percentages
        self.code_to_level_dict['brightness'] = dict()
        for n, l in enumerate(np.linspace(0., 1.5, 20)):
            self.code_to_level_dict['brightness']['l1_' + str(n)] = l

        # factors
        self.code_to_level_dict['contrast'] = dict()
        for n, l in enumerate(np.linspace(0., 1.5, 20)):
            self.code_to_level_dict['contrast']['l2_' + str(n)] = l

        # factors
        self.code_to_level_dict['saturation'] = dict()
        for n, l in enumerate(np.linspace(0., 1.5, 20)):
            self.code_to_level_dict['saturation']['l3_' + str(n)] = l

        # factors
        self.code_to_level_dict['hue'] = dict()
        for n, l in enumerate(np.linspace(0.0, 0.5, 20)):
            self.code_to_level_dict['hue']['l4_' + str(n)] = l


if __name__ == '__main__':
    print('...')

