import matplotlib.pyplot as plt

def plot_loss_and_acc(loss_and_acc_dict):
    # visualize loss curve
    plt.figure()

    min_loss, max_loss = 100.0, 0.0
    for key, (loss_list, acc_list) in loss_and_acc_dict.items():
        min_loss = min(loss_list) if min(loss_list) < min_loss else min_loss
        max_loss = max(loss_list) if max(loss_list) > max_loss else max_loss

        num_epoch = len(loss_list)
        plt.plot(range(1, 1 + num_epoch), loss_list, '-s', label=key)

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.xticks(range(0, num_epoch + 1, 2))
    plt.axis([0, num_epoch + 1, min_loss - 0.1, max_loss + 0.1])
    plt.show()

    # visualize acc curve
    plt.figure()

    min_acc, max_acc = 1.0, 0.0
    for key, (loss_list, acc_list) in loss_and_acc_dict.items():
        min_acc = min(acc_list) if min(acc_list) < min_acc else min_acc
        max_acc = max(acc_list) if max(acc_list) > max_acc else max_acc

        num_epoch = len(acc_list)
        plt.plot(range(1, 1 + num_epoch), acc_list, '-s', label=key)

    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.xticks(range(0, num_epoch + 1, 2))
    plt.axis([0, num_epoch + 1, min_acc, 1.0])
    plt.show()
