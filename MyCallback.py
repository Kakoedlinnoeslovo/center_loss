import matplotlib.pyplot as plt


def visualize(feat, labels, epoch, centers, lambda_cl, is_train = False):
    c = ['#ff0000', '#ffff00', '#00ff00', '#00ffff', '#0000ff',
         '#ff00ff', '#990000', '#999900', '#009900', '#009999']
    plt.figure()
    plt.clf()
    for i in range(10):
        plt.plot(feat[labels == i, 0], feat[labels == i, 1], '.', c=c[i])
    plt.plot(centers[:, 0], centers[:, 1], 'kx', mew=2, ms=4)
    plt.title('Validation data. Lambda_centerloss = {}, Epoch = {}'.format(lambda_cl, epoch))
    plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], loc='upper right')
    if is_train:
        plt.savefig('./results/epoch-{}-lambda-{}-train.png'.format(epoch, lambda_cl))
    else:
        plt.savefig('./results/epoch-{}-lambda-{}-val.png'.format(epoch, lambda_cl))
    plt.close()
