import numpy as np
import logging
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
from sklearn import manifold


class TensorboardXLogger:
    def __init__(self, path):
        self.writer = SummaryWriter(path)

    def per_batch_results(self, acc_dict):
        file = open(self.path + "/results.txt", "a")
        for mname in acc_dict:
            results = ",".join([f"{acc_dict[mname]:.3f}"])
            file.write(f"{self.name},{mname},{results}\n")
        file.close()

    def write_results(self, name, cumulative_accuracies, iteration, type_results):
        file = open(self.path+f"/{name}.txt", "a")
        file.write(f'\n\nType results: {type_results} | Iteration: {iteration}')

        for type, res in cumulative_accuracies.items():
            #type could be acc_cum, new, base
            file.write(f'\n\nType evaluation: {type}\n')
            # ADD VALUES
            for metrics, values in res.items():
                file.write(f"\n{metrics} " + " ".join([str(f' {x:.3f} ') for x in values]))

        file.close()

    def save_opts(self, opts):
        text = ''
        # CREATE TABLE
        text += f'<table width=\"100%\">'
        # ADD TWO COLUMN
        text += f'<td>PARAMS</td><td>VALUE</td>'

        for arg in vars(opts):
            # ADD VALUES
            text += f"<tr><td>{arg}</td>" + f"<td>{getattr(opts, arg)}</td></tr>"

        text += "</table>"
        self.writer.add_text('Parameters', text, 0)

    def save_results(self, name, acc_base, acc_new, acc_cum, iteration, type_results):
        # type_results+'/results/' + name + '_base' per avere i 3 plot diversi per ogni metodo
        self.writer.add_scalar(type_results + '/results/base', acc_base['nme'], iteration)
        self.writer.add_scalar(type_results + '/results/new', acc_new['nme'], iteration)
        self.writer.add_scalar(type_results + '/results/cum', acc_cum['nme'], iteration)

    def save_cumulative_results(self, results):
        # CUMULATIVE RESULTS DICTIONARY
        text = ''
        for type, res in results.items():
            # CREATE TABLE
            text += f'<table width=\"100%\"><td>{type}</td>'

            # ADD THE CORRECT NUMBER OF COLUMN ACCORDING TO THE NUMBER OF BATCHES
            for index in range(len(res['nme'])):
                text += f'<td>{index}</td>'

            # ADD VALUES
            for metrics, values in res.items():
                text += f"<tr><td>{metrics}</td>" + " ".join([str(f'<td>{x:.3f}</td>') for x in values]) + "</tr>"
            text += "</table>"
        self.writer.add_text('results', text, 0)

    def track_means(self, class_means, sqd, all_features_extracted, iteration):
        # self.writer.add_histogram(f'class_means/{self.name}', class_means, global_step=self.iteration)
        np.save(self.path + f"/class_means@iteration-{iteration}", class_means)
        np.save(self.path + f"/sqd@iteration-{iteration}", sqd)
        np.save(self.path + f"/extracted_feature@iteration-{iteration}", all_features_extracted)

    def print_accuracy(self, method, acc_base, acc_new, acc_cum):
        logging.info("Cumulative results")
        logging.info(f" Accuracy NME {method:<15}:\t{acc_cum['nme']:.2f}")
        logging.info(f" Accuracy CNN {method:<15}:\t{acc_cum['cnn']:.2f}")

        logging.info("New batch results")
        logging.info(f" Accuracy NME {method:<15}:\t{acc_new['nme']:.2f}")
        logging.info(f" Accuracy CNN {method:<15}:\t{acc_new['cnn']:.2f}")

        logging.info("First results")
        logging.info(f" Accuracy NME {method:<15}:\t{acc_base['nme']:.2f}")
        logging.info(f" Accuracy CNN {method:<15}:\t{acc_base['cnn']:.2f}")
        logging.info("")

    def log_training(self, epoch, train_loss, train_acc, iteration, type='train'):
        self.writer.add_scalar(f'loss/{type}-{iteration}', train_loss, epoch)
        self.writer.add_scalar(f'train/acc/{type}-{iteration}', train_acc, epoch)

    def log_ss(self, epoch, ss_loss, ss_acc, iteration):
        self.writer.add_scalar(f'ss/loss/ss-{iteration}', ss_loss, epoch)
        self.writer.add_scalar(f'ss/acc/ss-{iteration}', ss_acc, epoch)

    def log_valid(self, epoch, loss, tau, iteration):
        self.writer.add_scalar(f'tau/tau-{iteration}', tau, epoch)
        self.writer.add_scalar(f'tau-loss/loss-{iteration}', loss, epoch)

    def log_search(self, top1_acc_list, iteration):
        self.writer.add_scalar(f'search/Closed world without rejection', top1_acc_list[0], iteration)
        self.writer.add_scalar(f'search/Closed world with rejection', top1_acc_list[1], iteration)
        self.writer.add_scalar(f'search/Open Set', top1_acc_list[-1], iteration)

    def log_test(self, top1_acc_list, iteration, dataset):
        self.writer.add_scalar(f'test-{dataset.split("/")[1]}/Closed world without rejection', top1_acc_list[0], iteration)
        self.writer.add_scalar(f'test-{dataset.split("/")[1]}/Closed world with rejection', top1_acc_list[1], iteration)
        self.writer.add_scalar(f'test-{dataset.split("/")[1]}/Open Set', top1_acc_list[-1], iteration)

    def log_taus(self, closed_with_rejection_list, open_set_list, iterations):
        for idx, iteration in enumerate(iterations):
            self.writer.add_scalar(f'test/Closed world without rejection', closed_with_rejection_list[idx], iteration)
            self.writer.add_scalar(f'test/Closed world with rejection', open_set_list[idx], iteration)

    def log_grad(self, idx, grad, epoch, iteration):
        self.writer.add_scalar(f'gradient/iteration-{iteration}-epoch-{epoch}', grad, idx)

    @staticmethod
    def conf_matrix_figure(cm):
        fig, ax = plt.subplots()
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)

        ax.set(title=f'Confusion Matrix',
               ylabel='True label',
               xlabel='Predicted label')

        fig.tight_layout()
        return fig

    def confusion_matrix(self, y, y_hat, n_classes, results_type):
        conf = np.zeros((n_classes, n_classes))

        for i in range(len(y)):
            conf[y[i], y_hat[i]] += 1

        cm = conf.astype('float') / (conf.sum(axis=1) + 0.000001)[:, np.newaxis]

        fig = self.conf_matrix_figure(cm)
        self.writer.add_figure('conf_matrix/'+results_type, fig, global_step=self.iteration, close=True)

        avg_acc = np.diag(cm).mean() * 100.
        print(f"Per class accuracy ({results_type}): {avg_acc}")
        return conf

    @staticmethod
    def tsne_figure(x, y):
        colors = np.load('colors.npy')
        unk_color = colors[-1]
        colors = colors[0:-1]

        fig, ax = plt.subplots()
        ax.set(title=f't-SNE',
               ylabel='Dimension 2',
               xlabel='Dimension 1')

        for g in np.unique(y):
            ix = np.where(y == g)
            if g == 70:
                ax.scatter(x[ix, 0], x[ix, 1], s=4, c=unk_color, label=g, alpha=0.5)
            else:
                ax.scatter(x[ix, 0], x[ix, 1], s=4, c=colors[g], label=g, alpha=0.5)

        # ax.legend()

        return fig


    def tsne(self, x, y, results_type):
        res = manifold.TSNE(n_components=2, random_state=0).fit_transform(x)
        fig = self.tsne_figure(res, y)
        self.writer.add_figure(f't-SNE_{results_type}', fig,close=True)


    @staticmethod
    def means_matrix_figure(matrix):
        fig, ax = plt.subplots()
        im = ax.imshow(matrix, interpolation='nearest', aspect='auto', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)

        ax.set(title='Means Matrix',
               ylabel='Features extracted',
               xlabel='Classes')

        fig.tight_layout()
        return fig

    def means_matrix(self, class_means):
        fig = self.means_matrix_figure(class_means)
        self.writer.add_figure('means_matrix/', fig, global_step=self.iteration)

        return class_means

    def add_results(self, results):
        # CUMULATIVE RESULTS DICTIONARY
        metric_dict = {0: "WithoutRejection", 1: "WithRejection", results.shape[1]-1: "OpenSet", }

        for metric in metric_dict.keys():
            text = f'<table width=\"100%\"><tr><th>Iter</th>'

            # ADD THE CORRECT NUMBER OF COLUMN ACCORDING TO THE NUMBER OF BATCHES
            for order in range(results.shape[2]):
                text += f'<th>{order}</th>'
            text += '</tr>'

            for step in range(results.shape[0]):
                text += f"<tr><td>{step}</td>"+" ".join([f'<td>{x:.3f}</td>' for x in results[step, metric, :]])+"</tr>"

            # END TABLE
            text += "</table>"

            self.writer.add_text(metric_dict[metric], text, 0)
