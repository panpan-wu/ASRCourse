# Author: Sining Sun, Zhanheng Yang, Binbin Zhang
"""
实现了 ReLU，Tanh，Sigmoid，Dropout，L2 Regularization 和 Adam
相关代码：
    - class ReLU
    - class Tanh
    - class Sigmoid
    - class Dropout
    - class FullyConnect
        - l2 方法实现了 L2 Regularization
        - adam 方法实现了 Adam
    - class DNNExtented
        - predict 方法用来预测，预测时跳过 Dropout Layer
        - backward 添加了调用 l2 和 adam 的代码
        - compute_loss 用来计算损失

    下列 3 个函数用来对 DNNExtented 进行测试
    - test_dnn_extented
    - train_dnn_extented
    - predict
正确率：
    main() 95.45%
    test_dnn_extented() 98.18%
"""

import numpy as np
import kaldi_io
from utils import *

targets_list = ['Z', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']
targets_mapping = {}
for i, x in enumerate(targets_list):
    targets_mapping[x] = i


class Layer:
    def forward(self, input):
        ''' Forward function by input
        Args:
            input: input, B * N matrix, B for batch size
        Returns:
            output when applied this layer
        '''
        raise NotImplementedError('Not implement error')

    def backward(self, input, output, d_output):
        ''' Compute gradient of this layer's input by (input, output, d_output)
            as well as compute the gradient of the parameter of this layer
        Args:
            input: input of this layer
            output: output of this layer
            d_output: accumulated gradient from final output to this
                      layer's output
        Returns:
            accumulated gradient from final output to this layer's input
        '''
        raise NotImplementedError('Not implement error')

    def set_learning_rate(self, lr):
        ''' Set learning rate of this layer'''
        self.learning_rate = lr

    def update(self):
        ''' Update this layers parameter if it has or do nothing
        '''


class ReLU(Layer):
    def forward(self, input):
        # BEGIN_LAB
        return np.maximum(0, input)
        # END_LAB

    def backward(self, input, output, d_output):
        # BEGIN_LAB
        return d_output * (input > 0)
        # END_LAB


class Tanh(Layer):
    def forward(self, input):
        return np.tanh(input)

    def backward(self, input, output, d_output):
        # tanh'(z) = 1 - tanh(z) ** 2
        # dz = da * (1 - a ** 2)
        return d_output * (1.0 - output ** 2)


class Sigmoid(Layer):
    def forward(self, input):
        return 1 / (1 + np.exp(-input))

    def backward(self, input, output, d_output):
        # sigmoid'(z) = sigmoid(z) * (1 - sigmoid(z))
        # dz = da * (a * (1 - a))
        return d_output * (output * (1 - output))


class FullyConnect(Layer):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.w = np.random.randn(out_dim, in_dim) * np.sqrt(2.0 / in_dim)
        self.b = np.zeros((out_dim, 1))
        self.dw = np.zeros((out_dim, in_dim))
        self.db = np.zeros((out_dim, 1))

        # Adam
        self.Vdw = np.zeros((out_dim, in_dim))
        self.Sdw = np.zeros((out_dim, in_dim))
        self.Vdb = np.zeros((out_dim, 1))
        self.Sdb = np.zeros((out_dim, 1))
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 10e-8
        self.t = 1

    def l2(self, lambd, batch_size):
        self.dw += self.w * lambd / batch_size

    def adam(self):
        self.Vdw = self.beta1 * self.Vdw + (1 - self.beta1) * self.dw
        self.Vdb = self.beta1 * self.Vdb + (1 - self.beta1) * self.db
        self.Sdw = self.beta2 * self.Sdw + (1 - self.beta2) * (self.dw ** 2)
        self.Sdb = self.beta2 * self.Sdb + (1 - self.beta2) * (self.db ** 2)

        t = self.t
        self.dw = self.Vdw / (1 - self.beta1 ** t) / (np.sqrt(self.Sdw / (1 - self.beta2 ** t)) + self.epsilon)
        self.db = self.Vdb / (1 - self.beta1 ** t)/ (np.sqrt(self.Sdb / (1 - self.beta2 ** t)) + self.epsilon)
        self.t += 1

    def forward(self, input):
        # BEGIN_LAB
        return (np.dot(self.w, input.T) + self.b).T
        # END_LAB

    def backward(self, input, output, d_output):
        batch_size = input.shape[0]
        in_diff = None
        # BEGIN_LAB, compute in_diff/dw/db here
        self.dw = np.dot(d_output.T, input)
        self.db = np.sum(d_output, axis=0).reshape((d_output.shape[1], 1))
        in_diff = np.dot(d_output, self.w)
        # END_LAB
        # Normalize dw/db by batch size
        self.dw = self.dw / batch_size
        self.db = self.db / batch_size
        return in_diff

    def update(self):
        self.w = self.w - self.learning_rate * self.dw
        self.b = self.b - self.learning_rate * self.db


class Softmax(Layer):
    def forward(self, input):
        row_max = input.max(axis=1).reshape(input.shape[0], 1)
        x = input - row_max
        return np.exp(x) / np.sum(np.exp(x), axis=1).reshape(x.shape[0], 1)

    def backward(self, input, output, d_output):
        ''' Directly return the d_output as we show below, the grad is to
            the activation(input) of softmax
        '''
        return d_output


class Dropout(Layer):
    def __init__(self, keep_prob):
        super().__init__()
        self.keep_prob = keep_prob
        self.dropout_mask = None

    def forward(self, input):
        if 0 < self.keep_prob < 1:
            self.dropout_mask = np.random.rand(*input.shape) < self.keep_prob
            input = input * self.dropout_mask / self.keep_prob
        return input

    def backward(self, input, output, d_output):
        if 0 < self.keep_prob < 1:
            d_output = d_output * self.dropout_mask / self.keep_prob
        return d_output


class DNN:
    def __init__(self, in_dim, out_dim, hidden_dim, num_hidden, activation=ReLU):
        self.layers = []
        self.layers.append(FullyConnect(in_dim, hidden_dim))
        self.layers.append(activation())
        for i in range(num_hidden):
            self.layers.append(FullyConnect(hidden_dim, hidden_dim))
            self.layers.append(activation())
        self.layers.append(FullyConnect(hidden_dim, out_dim))
        self.layers.append(Softmax())

    def set_learning_rate(self, lr):
        for layer in self.layers:
            layer.set_learning_rate(lr)

    def forward(self, input):
        self.forward_buf = []
        out = input
        self.forward_buf.append(out)
        for i in range(len(self.layers)):
            out = self.layers[i].forward(out)
            self.forward_buf.append(out)
        assert (len(self.forward_buf) == len(self.layers) + 1)
        return out

    def backward(self, grad):
        '''
        Args:
            grad: the grad is to the activation before softmax
        '''
        self.backward_buf = [None] * len(self.layers)
        self.backward_buf[len(self.layers) - 1] = grad
        for i in range(len(self.layers) - 2, -1, -1):
            grad = self.layers[i].backward(self.forward_buf[i],
                                           self.forward_buf[i + 1],
                                           self.backward_buf[i + 1])
            self.backward_buf[i] = grad

    def update(self):
        for layer in self.layers:
            layer.update()


class DNNExtented(DNN):
    """
    额外实现 L2 Regularization, Dropout 和 Adam
    """
    def __init__(self, in_dim, out_dim, hidden_dim, num_hidden, activation=ReLU, lambd=0, keep_prob=1):
        self.lambd = lambd
        self.keep_prob = keep_prob

        self.layers = []
        self.layers.append(FullyConnect(in_dim, hidden_dim))
        self.layers.append(activation())
        self.layers.append(Dropout(self.keep_prob))
        for i in range(num_hidden):
            self.layers.append(FullyConnect(hidden_dim, hidden_dim))
            self.layers.append(activation())
            self.layers.append(Dropout(self.keep_prob))
        self.layers.append(FullyConnect(hidden_dim, out_dim))
        self.layers.append(Softmax())

    def predict(self, input):
        """
        预测时跳过 Dropout 层。
        """
        out = input
        for i in range(len(self.layers)):
            if isinstance(self.layers[i], Dropout):
                continue
            out = self.layers[i].forward(out)
        return out

    def backward(self, grad):
        '''
        Args:
            grad: the grad is to the activation before softmax
        '''
        self.backward_buf = [None] * len(self.layers)
        self.backward_buf[len(self.layers) - 1] = grad
        dropout_mask = None
        for i in range(len(self.layers) - 2, -1, -1):
            grad = self.layers[i].backward(self.forward_buf[i],
                                           self.forward_buf[i + 1],
                                           self.backward_buf[i + 1])

            layer = self.layers[i]
            batch_size = self.forward_buf[i].shape[0]
            if isinstance(layer, FullyConnect):
                layer.l2(self.lambd, batch_size)
                layer.adam()

            self.backward_buf[i] = grad

    def compute_loss(self, input, one_hot_label):
        out = self.predict(input)
        loss = -np.sum(np.log(out + 1e-20) * one_hot_label) / out.shape[0]
        if self.lambd > 0:
            loss_of_l2 = 0.0
            for layer in self.layers:
                if isinstance(layer, FullyConnect):
                    loss_of_l2 += np.sum(layer.w ** 2)
            loss += loss_of_l2 * self.lambd / (2 * out.shape[0])
        return loss


def one_hot(labels, total_label):
    output = np.zeros((labels.shape[0], total_label))
    for i in range(labels.shape[0]):
        output[i][labels[i]] = 1.0
    return output


def train(dnn):
    utt2feat, utt2target = read_feats_and_targets('train/feats.scp',
                                                  'train/text')
    inputs, labels = build_input(targets_mapping, utt2feat, utt2target)
    num_samples = inputs.shape[0]
    # Shuffle data
    permute = np.random.permutation(num_samples)
    inputs = inputs[permute]
    labels = labels[permute]
    num_epochs = 20
    batch_size = 100
    for i in range(num_epochs):
        cur = 0
        while cur < num_samples:
            end = min(cur + batch_size, num_samples)
            input = inputs[cur:end]
            label = labels[cur:end]
            # Step1: forward
            out = dnn.forward(input)
            one_hot_label = one_hot(label, out.shape[1])
            # Step2: Compute cross entropy loss and backward
            loss = -np.sum(np.log(out + 1e-20) * one_hot_label) / out.shape[0]
            # The grad is to activation before softmax
            grad = out - one_hot_label
            dnn.backward(grad)
            # Step3: update parameters
            dnn.update()
            print('Epoch {} num_samples {} loss {}'.format(i, cur, loss))
            cur += batch_size


def test(dnn):
    utt2feat, utt2target = read_feats_and_targets('test/feats.scp',
                                                  'test/text')
    total = len(utt2feat)
    correct = 0
    for utt in utt2feat:
        t = utt2target[utt]
        ark = utt2feat[utt]
        mat = kaldi_io.read_mat(ark)
        mat = splice(mat, 5, 5)
        posterior = dnn.forward(mat)
        posterior = np.sum(posterior, axis=0) / float(mat.shape[0])
        predict = targets_list[np.argmax(posterior)]
        if t == predict: correct += 1
        print('label: {} predict: {}'.format(t, predict))
    print('Acc: {}'.format(float(correct) / total))


def train_dnn_extented(dnn):
    utt2feat, utt2target = read_feats_and_targets('train/feats.scp',
                                                  'train/text')
    inputs, labels = build_input(targets_mapping, utt2feat, utt2target)
    num_samples = inputs.shape[0]
    # Shuffle data
    permute = np.random.permutation(num_samples)
    inputs = inputs[permute]
    labels = labels[permute]
    num_epochs = 20
    batch_size = 100
    losses = []
    for i in range(num_epochs):
        cur = 0
        while cur < num_samples:
            end = min(cur + batch_size, num_samples)
            input = inputs[cur:end]
            label = labels[cur:end]
            # Step1: forward
            out = dnn.forward(input)
            one_hot_label = one_hot(label, out.shape[1])
            # Step2: Compute cross entropy loss and backward
            loss = dnn.compute_loss(input, one_hot_label)
            losses.append(loss)
            # The grad is to activation before softmax
            grad = out - one_hot_label
            dnn.backward(grad)
            # Step3: update parameters
            dnn.update()
            print('Epoch {} num_samples {} loss {}'.format(i, cur, loss))
            cur += batch_size
    return losses


def predict(dnn):
    utt2feat, utt2target = read_feats_and_targets('test/feats.scp',
                                                  'test/text')
    total = len(utt2feat)
    correct = 0
    for utt in utt2feat:
        t = utt2target[utt]
        ark = utt2feat[utt]
        mat = kaldi_io.read_mat(ark)
        mat = splice(mat, 5, 5)
        posterior = dnn.predict(mat)
        posterior = np.sum(posterior, axis=0) / float(mat.shape[0])
        predict = targets_list[np.argmax(posterior)]
        if t == predict: correct += 1
        print('label: {} predict: {}'.format(t, predict))
    print('Acc: {}'.format(float(correct) / total))


def test_dnn_extented():
    np.random.seed(777)

    in_dim = 429
    out_dim = 11
    # hidden_dim = 128
    hidden_dim = 128
    # num_hidden = 1
    num_hidden = 1
    # learning_rate = 1e-2
    learning_rate = 0.001
    lambd = 0.1
    keep_prob = 0.8
    dnn = DNNExtented(
            in_dim, out_dim, hidden_dim, num_hidden,
            activation=ReLU,
            lambd=lambd,
            keep_prob=keep_prob)
    dnn.set_learning_rate(learning_rate)

    losses = train_dnn_extented(dnn)
    predict(dnn)

    # import matplotlib.pyplot as plt
    # plt.plot(losses)
    # plt.ylabel('losses')
    # plt.xlabel('iterations')
    # plt.show()


def main():
    np.random.seed(777)
    # We splice the raw feat with left 5 frames and right 5 frames
    # So the input here is 39 * (5 + 1 + 5) = 429
    dnn = DNN(429, 11, 128, 1)
    dnn.set_learning_rate(1e-2)
    train(dnn)
    test(dnn)


if __name__ == '__main__':
    # main()
    test_dnn_extented()
