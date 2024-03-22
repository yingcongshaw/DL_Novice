import numpy as np
from dataloader import build_dataloader
from optimizer import SGD
from loss import SoftmaxCrossEntropyLoss
from visualize import plot_loss_and_acc

class Solver(object):
    def __init__(self, cfg):
        self.cfg = cfg

        # build dataloader
        train_loader, val_loader, test_loader = self.build_loader(cfg)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader


        # build evaluation model
        self.model = SoftmaxCrossEntropyLoss(784, 10)

        # build optimizer
        self.optimizer = self.build_optimizer(self.model, cfg)

    @staticmethod
    def build_loader(cfg):
        train_loader = build_dataloader(
            cfg['data_root'], cfg['max_epoch'], cfg['batch_size'], shuffle=True, mode='train')

        val_loader = build_dataloader(
            cfg['data_root'], 1, cfg['batch_size'], shuffle=False, mode='val')

        test_loader = build_dataloader(
            cfg['data_root'], 1, cfg['batch_size'], shuffle=False, mode='test')

        return train_loader, val_loader, test_loader

    @staticmethod
    def build_optimizer(model, cfg):
        return SGD(model, cfg['learning_rate'], cfg['momentum'])

    def train(self):
        max_epoch = self.cfg['max_epoch']

        epoch_train_loss, epoch_train_acc = [], []
        for epoch in range(max_epoch):

            iteration_train_loss, iteration_train_acc = [], []
            for iteration, (images, labels) in enumerate(self.train_loader):
                # forward pass
                loss, acc = self.model.forward(images, labels)

                self.model.gradient_computing()

                # updata the model weights
                self.optimizer.step()

                # restore loss and accuracy
                iteration_train_loss.append(loss)
                iteration_train_acc.append(acc)

                # display iteration training info
                if iteration % self.cfg['display_freq'] == 0:
                    print("Epoch [{}][{}]\t Batch [{}][{}]\t Training Loss {:.4f}\t Accuracy {:.4f}".format(
                        epoch, max_epoch, iteration, len(self.train_loader), loss, acc))

            avg_train_loss, avg_train_acc = np.mean(iteration_train_loss), np.mean(iteration_train_acc)
            epoch_train_loss.append(avg_train_loss)
            epoch_train_acc.append(avg_train_acc)

            # validate
            avg_val_loss, avg_val_acc = self.validate()

            # display epoch training info
            print('\nEpoch [{}]\t Average training loss {:.4f}\t Average training accuracy {:.4f}'.format(
                epoch, avg_train_loss, avg_train_acc))

            # display epoch valiation info
            print('Epoch [{}]\t Average validation loss {:.4f}\t Average validation accuracy {:.4f}\n'.format(
                epoch, avg_val_loss, avg_val_acc))

        return epoch_train_loss, epoch_train_acc

    def validate(self):
        loss_set, acc_set = [], []
        for images, labels in self.val_loader:
            loss, acc = self.model.forward(images, labels)
            loss_set.append(loss)
            acc_set.append(acc)

        loss = np.mean(loss_set)
        acc = np.mean(acc_set)
        return loss, acc

    def test(self):
        loss_set, acc_set = [], []
        for images, labels in self.test_loader:
            loss, acc = self.model.forward(images, labels)
            # loss_set.append(logits)   //shaw
            loss_set.append(loss)
            acc_set.append(acc)

        loss = np.mean(loss_set)
        acc = np.mean(acc_set)
        return loss, acc

