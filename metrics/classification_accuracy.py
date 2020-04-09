from sklearn import metrics


def accuracy(num_classes=2):
    return Accuracy(num_classes)


class Accuracy(object):
    def __init__(self, num_classes):
        self.num_class = num_classes
        # self.confusion_matrix = np.zeros((self.num_class, self.num_class))

    def __call__(self, epoch_labels, epoch_preds):
        accuracy = metrics.accuracy_score(epoch_labels, epoch_preds)
        return accuracy
