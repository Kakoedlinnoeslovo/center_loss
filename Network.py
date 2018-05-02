from keras.datasets import mnist
from keras.layers import Input, Conv2D, PReLU, MaxPool2D, Dense, Flatten, Embedding, BatchNormalization
from keras.engine.topology import Layer
from keras.layers.merge import concatenate
from keras.models import Model
from keras.optimizers import SGD
from keras import backend as K
from keras.regularizers import l2
from keras import losses
import numpy as np
from keras.utils import to_categorical
from tqdm import tqdm
from keras import initializers

from MyCallback import visualize


def prelu(x, name='default'):
	if name == 'default':
		return PReLU(alpha_initializer=initializers.Constant(value=0.25))(x)
	else:
		return PReLU(alpha_initializer=initializers.Constant(value=0.25), name=name)(x)


class Network():
	def __init__(self, alpha_center=0.5,
	             lambda_centerloss=0.1):
		self.alpha_center = alpha_center
		self.lambda_centerloss = lambda_centerloss
		self.pool_name = "pool_"
		self.conv_name = "conv_"
		self.model = None

	def _conv_block(self, input, out_dim, kernel, counter, weight_decay):
		x = Conv2D(out_dim, (kernel, kernel), name=self.conv_name + str(counter),
		           kernel_regularizer=l2(weight_decay), padding='same')(input)
		out = prelu(x)
		return out

	class CenterLayer(Layer):
		def __init__(self, num_classes, feature_dim, alpha_center, **kwargs):
			super().__init__(**kwargs)
			self.alpha_center = alpha_center
			self.num_classes = num_classes
			self.feature_dim = feature_dim

		def build(self, input_shape):
			# Create a trainable weight variable for this layer
			self.centers = self.add_weight(name='centers',
			                               shape=(self.num_classes, self.feature_dim),
			                               initializer='uniform',
			                               trainable=False)
			super().build(input_shape)  # Be sure to call this somewhere!

		def call(self, x, mask=None):
			## x[0] is Nx2, x[1] is Nx10 onehot, self.centers is 10x2
			delta_centers = K.dot(K.transpose(x[1]), (K.dot(x[1], self.centers) - x[0]))  # 10x2
			denominator = K.sum(K.transpose(x[1]), axis=1, keepdims=True) + 1
			delta_centers /= denominator
			new_centers = self.centers - self.alpha_center * delta_centers
			self.add_update((self.centers, new_centers), x)
			self.result = (K.dot(x[1], self.centers) - x[0])
			self.result = K.sum(self.result ** 2, axis=1, keepdims=True)
			return self.result

	def prepare_data(self):
		(x_train, y_train), (x_test, y_test) = mnist.load_data()
		print('x_train shape:', x_train.shape)
		print('x_test shape:', x_test.shape)
		x_train = x_train.reshape((-1, 28, 28, 1))
		x_test = x_test.reshape((-1, 28, 28, 1))
		y_train_onehot = to_categorical(y_train, 10)
		y_test_onehot = to_categorical(y_test, 10)

		return x_train, y_train_onehot, x_test, y_test_onehot, y_train, y_test

	def _build_model(self, im_size=28, hidden_dim=128, kernel = 3, weight_decay=0.005,
	                 num_classes=10, feature_dim=2, is_concated = True):
		input = Input((im_size, im_size, 1))
		labels = Input((num_classes,))

		x = BatchNormalization()(input)
		x = self._conv_block(input=x, out_dim=hidden_dim, kernel=3, counter=0, weight_decay=weight_decay)
		x = self._conv_block(input=x, out_dim=hidden_dim, kernel=3, counter=1, weight_decay=weight_decay)
		x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(x)
		x = self._conv_block(input=x, out_dim=hidden_dim, kernel=3, counter=2, weight_decay=weight_decay)
		x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(x)
		x = self._conv_block(input=x, out_dim=hidden_dim * 2, kernel=4, counter=3, weight_decay=weight_decay)
		x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(x)
		x = self._conv_block(input=x, out_dim=hidden_dim * 2, kernel=2, counter=4, weight_decay=weight_decay)
		x1 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(x)
		x2 = self._conv_block(input=x1, out_dim=hidden_dim * 2, kernel=1, counter=5, weight_decay=weight_decay)
		if is_concated:
			x_concated = concatenate([x1, x2], axis=3)
			x_flatten = Flatten()(x_concated)
		else:
			x_flatten = Flatten()(x2)
		x = Dense(units=feature_dim, kernel_regularizer=l2(weight_decay))(x_flatten)
		x = PReLU(name="deep_deatures")(x)

		y_out = Dense(num_classes, activation="softmax", kernel_regularizer=l2(weight_decay))(x)
		y_side = self.CenterLayer(num_classes=num_classes,
		                          alpha_center=self.alpha_center,
		                          feature_dim=feature_dim, name="centerlosslayer")([x, labels])

		model = Model(inputs=[input, labels], outputs=[y_out, y_side])
		model.summary()
		self.model = model



	def my_model(self, im_size = 28, num_classes = 10, weight_decay = 0.005):
		input = Input((im_size, im_size, 1))
		labels = Input((num_classes,))

		x = BatchNormalization()(input)
		#
		x = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same',
		           kernel_regularizer=l2(weight_decay))(x)
		x = prelu(x)
		x = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same',
		           kernel_regularizer=l2(weight_decay))(x)
		x = prelu(x)
		x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(x)
		#
		x = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same',
		           kernel_regularizer=l2(weight_decay))(x)
		x = prelu(x)
		x = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same',
		           kernel_regularizer=l2(weight_decay))(x)
		x = prelu(x)
		x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(x)
		#
		x = Conv2D(filters=128, kernel_size=(5, 5), strides=(1, 1), padding='same',
		           kernel_regularizer=l2(weight_decay))(x)
		x = prelu(x)
		x = Conv2D(filters=128, kernel_size=(5, 5), strides=(1, 1), padding='same',
		           kernel_regularizer=l2(weight_decay))(x)
		x = prelu(x)
		x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(x)
		#
		x = Flatten()(x)
		x = Dense(2, kernel_regularizer=l2(weight_decay))(x)
		x = prelu(x, name='deep_deatures')
		#
		main = Dense(10, activation='softmax', name='main_out', kernel_regularizer=l2(weight_decay))(x)
		side = self.CenterLayer(num_classes=num_classes, feature_dim=2, alpha_center=0.5,
		                        name='centerlosslayer')([x, labels])
		model = Model(inputs=[input, labels], outputs=[main, side])
		model.summary()
		self.model = model

	def center_loss(self, y_true, y_pred):
		return 0.5 * K.sum(y_pred, axis=0)

	def _fit(self, x_train, y_train, x_test, y_test,
	         learning_rate=0.01, momentum=0.9, epochs=50,
	         batch_size=64, train_percentage=0.1):
		if self.model is None:
			print("first you need to init model")
			self._build_model()

		dummy1 = np.zeros((x_train.shape[0], 1))
		dummy2 = np.zeros((x_test.shape[0], 1))

		optimizer = SGD(lr=learning_rate, momentum=momentum)
		self.model.compile(optimizer=optimizer,
		                   loss=[losses.categorical_crossentropy, self.center_loss],
		                   loss_weights=[1, self.lambda_centerloss])

		N = x_train.shape[0]
		n = int(train_percentage * N)

		self.model.fit([x_train[:n], y_train[:n]], [y_train[:n], dummy1[:n]],
		               batch_size=batch_size,
		               epochs=epochs,
		               verbose=2,
		               validation_data=([x_test[:n], y_test[:n]], [y_test[:n], dummy2[:n]]))

	def _plot_results(self, x_train, y_train, x_test, y_test, epochs=50, train_percentage=0.1):
		N = x_train.shape[0]
		n = int(train_percentage * N)

		reduced_model = Model(inputs=self.model.input[0], outputs=self.model.get_layer('deep_deatures').output)
		feats = reduced_model.predict(x_train[:n])
		visualize(feats[:n], y_train[:n], epoch=epochs - 1,
		          centers=self.model.get_layer('centerlosslayer').get_weights()[0],
		          lambda_cl=self.lambda_centerloss, is_train=True)

		feats = reduced_model.predict(x_test[:n])
		visualize(feats[:n], y_test[:n], epoch=epochs - 1,
		          centers=self.model.get_layer('centerlosslayer').get_weights()[0],
		          lambda_cl=self.lambda_centerloss, is_train=False)


def test():
	network = Network(alpha_center=0.5, lambda_centerloss=0.1)
	network._build_model(is_concated=True)
	#network.my_model()
	x_train, y_train_onehot, x_test, y_test_onehot, y_train, y_test = network.prepare_data()
	for i in tqdm(range(1, 11)):
		network._fit(x_train, y_train_onehot, x_test, y_test_onehot, train_percentage=0.3,
		             epochs=10, learning_rate=0.001, momentum=0.9, batch_size=64)
		network._plot_results(x_train, y_train, x_test, y_test, train_percentage=0.3, epochs=10 * i + 1)


if __name__ == "__main__":
	test()
