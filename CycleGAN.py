import tensorflow as tf
import numpy as np
import random
import time
import os
import cv2


def save_ndarray(path, ndarray, binary=True):
    if binary:
        np.save(path, ndarray)
    else:
        strShape = str(ndarray.shape)
        ndarray = ndarray.reshape((-1,))
        np.savetxt(path, ndarray, header=strShape)


def load_ndarray(file, binary=True):
    if binary:
        ndarray = np.load(file + ".npy")
    else:
        with open(file) as f:
            shape = tuple(map(int, (f.readline()[3:-2]).split(",")))
        ndarray = np.loadtxt(file)
        ndarray = ndarray.reshape(shape)
    return ndarray


def secs_to_string(secs):
    hours = int(secs / 60 / 60)
    secs %= 60 * 60
    minutes = int(secs / 60)
    secs %= 60
    res = []
    if hours != 0:
        res.append(str(hours) + " hours")
    if minutes != 0:
        res.append(str(minutes) + " minutes")
    if secs != 0:
        res.append(str(int(secs+0.5)) + " seconds")
    return ", ".join(res)

def random_element(arr):
    return arr[np.random.randint(0, len(arr))]

class DiscriminatorFilters:
    def __init__(self, input_channels=0, num_filters=0, num_layers=0, filter_size=0, stdv=0, path=None, name="dsc_filters"):
        if path is None or not os.path.exists(path):
            with tf.variable_scope(name):
                self.filter_in = tf.Variable(np.random.normal(0, stdv, (filter_size, filter_size, input_channels, num_filters)).astype(np.float32), name="Layer_1")
                self.filters = [tf.Variable(np.random.normal(0, stdv, (filter_size, filter_size, num_filters * (2 ** i), num_filters * (2 ** (i + 1)))).astype(np.float32), name="Layer_" + str(i + 2)) for i in range(num_layers - 2)]
                self.filter_out = tf.Variable(np.random.normal(0, stdv, (filter_size, filter_size, num_filters * (2 ** (num_layers - 2)), 1)).astype(np.float32), name="Layer_" + str(num_layers))
                self.flatten = tf.Variable(np.random.normal(0, stdv, (1, 5, 1)).astype(np.float32), name="Flatten")
                self.decision = tf.Variable(np.random.normal(0, stdv, (1, 1, 5)).astype(np.float32), name="Decision")
        else:
            with tf.variable_scope(name):
                self.filter_in = tf.Variable(load_ndarray(os.path.join(path, "filter_in")).astype(np.float32), name="Layer_1")
                self.filters = []
                layerNum = 0
                files = set(os.listdir(path))
                while "layer_" + str(layerNum) + ".npy" in files:
                    self.filters.append(tf.Variable(load_ndarray(os.path.join(path, "layer_" + str(layerNum) + "")).astype(np.float32), name="Layer_" + str(layerNum + 2)))
                    layerNum += 1
                self.filter_out = tf.Variable(load_ndarray(os.path.join(path, "filter_out")).astype(np.float32), name="Layer_" + str(layerNum + 2))
                self.flatten = tf.Variable(load_ndarray(os.path.join(path, "flatten")).astype(np.float32), name="Flatten")
                self.decision = tf.Variable(load_ndarray(os.path.join(path, "decision")).astype(np.float32), name="Decision")
        self.vars = [self.filter_in, self.flatten, self.decision, self.filter_out] + self.filters

    def save(self, session, folder):
        save_ndarray(os.path.join(folder, "filter_in"), session.run(self.filter_in))
        for filter_num in range(len(self.filters)):
            save_ndarray(os.path.join(folder, "layer_" + str(filter_num) + ""), session.run(self.filters[filter_num]))
        save_ndarray(os.path.join(folder, "filter_out"), session.run(self.filter_out))
        save_ndarray(os.path.join(folder, "flatten"), session.run(self.flatten))
        save_ndarray(os.path.join(folder, "decision"), session.run(self.decision))

    def load(self, session, folder):
        session.run(tf.assign(self.filter_in, load_ndarray(os.path.join(folder, "filter_in"))))
        session.run(tf.assign(self.filter_out, load_ndarray(os.path.join(folder, "filter_in"))))
        session.run(tf.assign(self.flatten, load_ndarray(os.path.join(folder, "flatten"))))
        session.run(tf.assign(self.decision, load_ndarray(os.path.join(folder, "decision"))))

        filterNum = 0
        files = set(os.listdir(folder))
        while "layer_" + str(filterNum) + ".npy" in files:
            session.run(tf.assign(self.filters[filterNum], load_ndarray(os.path.join(folder, "decision"))))
            filterNum += 1


class GeneratorFilters:
    def __init__(self, input_channels=0, num_filters=0, num_encode_decode_layers=0, num_transform_blocks=0, encode_decode_io_filter_size=0, encode_decode_filter_size=0, transform_filter_size=0, stdv=0, path=None, name="gen_filters"):
        if path is None or not os.path.exists(path):
            with tf.variable_scope(name):
                with tf.variable_scope("Encode"):
                    self.encode_in = tf.Variable(np.random.normal(0, stdv, (encode_decode_io_filter_size, encode_decode_io_filter_size, input_channels, num_filters)).astype(np.float32), name="Layer_1")
                    self.encode = [tf.Variable(np.random.normal(0, stdv, (encode_decode_filter_size, encode_decode_filter_size, num_filters * (2 ** i), num_filters * (2 ** (i + 1)))).astype(np.float32), name="Layer_" + str(i + 2)) for i in range(num_encode_decode_layers - 1)]
                with tf.variable_scope("Transform"):
                    self.transform = []
                    for i in range(num_transform_blocks):
                        with tf.variable_scope("Block_" + str(i + 1)):
                            self.transform.append(
                                (
                                    tf.Variable(np.random.normal(0, stdv, (transform_filter_size, transform_filter_size, num_filters * (2 ** (num_encode_decode_layers - 1)), num_filters * (2 ** (num_encode_decode_layers - 1)))).astype(np.float32), name="Layer_1"),
                                    tf.Variable(np.random.normal(0, stdv, (transform_filter_size, transform_filter_size, num_filters * (2 ** (num_encode_decode_layers - 1)), num_filters * (2 ** (num_encode_decode_layers - 1)))).astype(np.float32), name="Layer_2")
                                )
                            )
                with tf.variable_scope("Decode"):
                    self.decode = [tf.Variable(np.random.normal(0, stdv, (encode_decode_filter_size, encode_decode_filter_size, num_filters * (2 ** (i - 1)), num_filters * (2 ** i))).astype(np.float32), name="Layer_" + str(num_encode_decode_layers - i)) for i in
                                   range(num_encode_decode_layers - 1, 0, -1)]
                    self.decode_out = tf.Variable(np.random.normal(0, stdv, (encode_decode_io_filter_size, encode_decode_io_filter_size, num_filters, input_channels)).astype(np.float32), name="Layer_" + str(num_encode_decode_layers))
        else:
            with tf.variable_scope(name):
                files = set(os.listdir(path))
                with tf.variable_scope("Encode"):
                    self.encode_in = tf.Variable(load_ndarray(os.path.join(path, "encode_in")).astype(np.float32), name="Layer_1")
                    self.encode = []
                    layerNum = 0
                    while "encode_layer_" + str(layerNum) + ".npy" in files:
                        self.encode.append(tf.Variable(load_ndarray(os.path.join(path, "encode_layer_" + str(layerNum) + "")).astype(np.float32), name="Layer_" + str(layerNum + 2)))
                        layerNum += 1
                with tf.variable_scope("Transform"):
                    self.transform = []
                    blockNum = 0
                    while "transform_block_" + str(blockNum) + "_layer_1.npy" in files and "transform_block_" + str(blockNum) + "_layer_2.npy" in files:
                        with tf.variable_scope("Block_" + str(blockNum + 1)):
                            self.transform.append(
                                (
                                    tf.Variable(load_ndarray(os.path.join(path, "transform_block_" + str(blockNum) + "_layer_1")).astype(np.float32), name="Layer_1"),
                                    tf.Variable(load_ndarray(os.path.join(path, "transform_block_" + str(blockNum) + "_layer_2")).astype(np.float32), name="Layer_2")
                                )
                            )
                        blockNum += 1
                with tf.variable_scope("Decode"):
                    self.decode = []
                    layerNum = 0
                    while "decode_layer_" + str(layerNum) + ".npy" in files:
                        self.decode.append(tf.Variable(load_ndarray(os.path.join(path, "decode_layer_" + str(layerNum) + "")).astype(np.float32), name="Layer" + str(layerNum + 1)))
                        layerNum += 1
                    self.decode_out = tf.Variable(load_ndarray(os.path.join(path, "decode_out")).astype(np.float32), name="Layer" + str(layerNum + 1))
        self.vars = [self.encode_in, self.decode_out] + self.encode + [block[0] for block in self.transform] + [block[1] for block in self.transform] + self.decode

    def save(self, session, folder):
        save_ndarray(os.path.join(folder, "encode_in"), session.run(self.encode_in))
        for filter_num in range(len(self.encode)):
            save_ndarray(os.path.join(folder, "encode_layer_" + str(filter_num) + ""), session.run(self.encode[filter_num]))
        for filter_num in range(len(self.transform)):
            save_ndarray(os.path.join(folder, "transform_block_" + str(filter_num) + "_layer_1"), session.run(self.transform[filter_num][0]))
            save_ndarray(os.path.join(folder, "transform_block_" + str(filter_num) + "_layer_2"), session.run(self.transform[filter_num][1]))
        for filter_num in range(len(self.decode)):
            save_ndarray(os.path.join(folder, "decode_layer_" + str(filter_num) + ""), session.run(self.decode[filter_num]))
        save_ndarray(os.path.join(folder, "decode_out"), session.run(self.decode_out))


class Generator:
    def __init__(self, input, filters, name="gen"):
        self.output = input
        self.filters = filters
        with tf.variable_scope(name):
            with tf.variable_scope("Encode"):
                self.output = tf.nn.conv2d(self.output, filters.encode_in, (1, 1, 1, 1), "SAME", name="Layer_1")
                for i in range(len(filters.encode)):
                    self.output = tf.nn.conv2d(self.output, filters.encode[i], (1, 2, 2, 1), "SAME", name="Layer_" + str(i + 2))
            with tf.variable_scope("Transform"):
                for i in range(len(filters.transform)):
                    with tf.variable_scope("Block_" + str(i + 1)):
                        input_res = self.output
                        self.output = tf.nn.conv2d(self.output, filters.transform[i][0], (1, 1, 1, 1), "SAME", name="Layer_1")
                        self.output = tf.nn.conv2d(self.output, filters.transform[i][1], (1, 1, 1, 1), "SAME", name="Layer_2")
                        self.output += input_res
            with tf.variable_scope("Decode"):
                for i in range(len(filters.decode)):
                    self.output = tf.nn.conv2d_transpose(self.output, filters.decode[i], [self.output.shape[0].value, self.output.shape[1].value * 2, self.output.shape[2].value * 2, filters.decode[i].get_shape()[2].value], [1, 2, 2, 1], name="Deconv_Layer_" + str(i + 1))
                self.output = tf.nn.conv2d(self.output, filters.decode_out, (1, 1, 1, 1), "SAME", name="Conv_Layer")
            with tf.variable_scope("Result"):
                self.result = self.output * 1


class Discriminator:
    def __init__(self, input, filters, name="dsc"):
        self.output = input
        self.filters = filters
        with tf.variable_scope(name):
            self.output = tf.random_crop(self.output, [1, 70, 70, input.shape[-1].value])
            self.output = tf.nn.conv2d(self.output, filters.filter_in, (1, 2, 2, 1), "SAME", name="Layer_1")
            for i in range(len(filters.filters)):
                self.output = tf.nn.conv2d(self.output, filters.filters[i], (1, 2, 2, 1), "SAME", name="Layer_" + str(i + 2))
            self.output = tf.nn.conv2d(self.output, filters.filter_out, (1, 1, 1, 1), "SAME", name="Layer_" + str(len(filters.filters) + 2))
            self.output = tf.reshape(self.output, [d.value for d in self.output.shape[:-1]])
            # print(self.output.shape, filters.flatten.get_shape())
            self.output = tf.matmul(self.output, filters.flatten, name="Flatten")
            self.output = tf.matmul(filters.decision, self.output, name="Decision")
            with tf.variable_scope("Result"):
                self.result = self.output * 1


class CycleGan:
    def __init__(self, session, width, height, channels, batch_size=1, num_dsc_features=64, num_gen_features=64, num_dsc_layers=5, num_gen_layers=3, num_transform_layers=9, dsc_filter_size=4, encode_decode_io_filter_size=7, encode_decode_filter_size=3, transform_filter_size=3, stdv=0.02,
                 name="cyclegan", train=True, folder=None):
        self.sess = session
        self.image_width = width
        self.image_height = height
        self.image_channels = channels
        self.batch_size = batch_size
        self.trainable = train
        with tf.variable_scope(name):
            with tf.variable_scope("Filters"):
                with tf.variable_scope("Generators"):
                    self.generator_x_filters = GeneratorFilters(channels, num_gen_features, num_gen_layers, num_transform_layers, encode_decode_io_filter_size, encode_decode_filter_size, transform_filter_size, stdv, os.path.join(folder, "generator_x") if folder is not None else folder, name="X")
                    self.generator_y_filters = GeneratorFilters(channels, num_gen_features, num_gen_layers, num_transform_layers, encode_decode_filter_size, encode_decode_filter_size, transform_filter_size, stdv, os.path.join(folder, "generator_y") if folder is not None else folder, name="Y")
                with tf.variable_scope("Discriminators"):
                    self.discriminator_x_filters = DiscriminatorFilters(channels, num_dsc_features, num_dsc_layers, dsc_filter_size, stdv, os.path.join(folder, "discriminator_x") if folder is not None else folder, name="X")
                    self.discriminator_y_filters = DiscriminatorFilters(channels, num_dsc_features, num_dsc_layers, dsc_filter_size, stdv, os.path.join(folder, "discriminator_y") if folder is not None else folder, name="Y")
            with tf.variable_scope("Test"):
                with tf.variable_scope("Inputs"):
                    self.input_x = tf.placeholder(tf.float32, [1, width, height, channels], name="X")
                    self.input_y = tf.placeholder(tf.float32, [1, width, height, channels], name="Y")

                with tf.variable_scope("Generators"):
                    self.generator_x = Generator(self.input_y, self.generator_x_filters, name="Y_to_X")
                    self.generator_y = Generator(self.input_x, self.generator_y_filters, name="X_to_Y")

                with tf.variable_scope("Discriminators"):
                    self.discriminator_x = Discriminator(self.input_x, self.discriminator_x_filters, name="X")
                    self.discriminator_y = Discriminator(self.input_y, self.discriminator_y_filters, name="Y")
            if self.trainable:
                with tf.variable_scope("Train"):
                    with tf.variable_scope("Inputs"):
                        self.train_input_x = tf.placeholder(tf.float32, [batch_size, width, height, channels], name="X")
                        self.train_input_y = tf.placeholder(tf.float32, [batch_size, width, height, channels], name="Y")

                    with tf.variable_scope("Generators"):
                        self.train_generator_x = Generator(self.train_input_y, self.generator_x_filters, name="Y_to_X")
                        self.train_generator_y = Generator(self.train_input_x, self.generator_y_filters, name="X_to_Y")
                        with tf.variable_scope("Cyclic"):
                            self.train_cycle_x = Generator(self.train_generator_y.output, self.generator_x_filters, "Y_to_X")
                            self.train_cycle_y = Generator(self.train_generator_x.output, self.generator_y_filters, "X_to_Y")

                    with tf.variable_scope("Discriminators"):
                        self.train_discriminator_x = Discriminator(self.train_input_x, self.discriminator_x_filters, name="X")
                        self.train_discriminator_y = Discriminator(self.train_input_y, self.discriminator_y_filters, name="Y")
                        with tf.variable_scope("Generated"):
                            self.train_discriminator_generated_x = Discriminator(self.train_generator_x.output, self.discriminator_x_filters, name="X")
                            self.train_discriminator_generated_y = Discriminator(self.train_generator_y.output, self.discriminator_y_filters, name="Y")

                    with tf.variable_scope("Losses"):
                        with tf.variable_scope("Discriminator"):
                            with tf.variable_scope("X"):
                                with tf.variable_scope("Given"):
                                    self.discriminate_given_x_loss = tf.reduce_mean(tf.squared_difference(self.train_discriminator_x.output, 1), name="Given")
                                with tf.variable_scope("Generated"):
                                    self.discriminate_generated_x_loss = tf.reduce_mean(tf.square(self.train_discriminator_generated_x.output), name="Generated")
                                with tf.variable_scope("Total"):
                                    self.discriminate_x_loss = (self.discriminate_given_x_loss + self.discriminate_generated_x_loss) / 2
                            with tf.variable_scope("Y"):
                                with tf.variable_scope("Given"):
                                    self.discriminate_given_y_loss = tf.reduce_mean(tf.squared_difference(self.train_discriminator_y.output, 1), name="Given")
                                with tf.variable_scope("Generated"):
                                    self.discriminate_generated_y_loss = tf.reduce_mean(tf.square(self.train_discriminator_generated_y.output), name="Generated")
                                with tf.variable_scope("Total"):
                                    self.discriminate_y_loss = (self.discriminate_given_y_loss + self.discriminate_generated_y_loss) / 2
                        with tf.variable_scope("Generator"):
                            with tf.variable_scope("Discrimination"):
                                self.anti_discriminate_x_loss = tf.reduce_mean(tf.squared_difference(self.train_discriminator_generated_x.output, 1), name="Y")
                                self.anti_discriminate_y_loss = tf.reduce_mean(tf.squared_difference(self.train_discriminator_generated_y.output, 1), name="Y")
                            with tf.variable_scope("Cycle"):
                                self.cyclic_loss = tf.reduce_mean(tf.abs(self.train_input_x - self.train_cycle_x.output), name="X") + tf.reduce_mean(tf.abs(self.train_input_y - self.train_cycle_y.output), name="Y")
                            with tf.variable_scope("X"):
                                self.generate_x_loss = self.anti_discriminate_x_loss + 5 * self.cyclic_loss
                            with tf.variable_scope("Y"):
                                self.generate_y_loss = self.anti_discriminate_y_loss + 5 * self.cyclic_loss
                        with tf.variable_scope("Summaries"):
                            self.generate_x_loss_summary = tf.summary.scalar("generate_x_loss", self.generate_x_loss)
                            self.generate_y_loss_summary = tf.summary.scalar("generate_y_loss", self.generate_y_loss)
                            self.discriminate_x_loss_summary = tf.summary.scalar("discriminate_x_loss", self.discriminate_x_loss)
                            self.discriminate_y_loss_summary = tf.summary.scalar("discriminate_y_loss", self.discriminate_y_loss)
                with tf.variable_scope("Training"):
                    self.learning_rate = tf.Variable(0.0, name='Learning_Rate')
                    self.optimizer = tf.train.AdamOptimizer(self.learning_rate, 0.5, name="Optimizer")

                    self.discriminator_x_trainer = self.optimizer.minimize(self.discriminate_x_loss, var_list=self.discriminator_x_filters.vars, name="Discriminator_X_Trainer")
                    self.discriminator_y_trainer = self.optimizer.minimize(self.discriminate_y_loss, var_list=self.discriminator_y_filters.vars, name="Discriminator_Y_Trainer")
                    self.generator_x_trainer = self.optimizer.minimize(self.generate_x_loss, var_list=self.generator_x_filters.vars, name="Generator_X_Trainer")
                    self.generator_y_trainer = self.optimizer.minimize(self.generate_y_loss, var_list=self.generator_y_filters.vars, name="Generator_Y_Trainer")
        self.sess.run(tf.global_variables_initializer())

    def train(self, learning_rate, num_epochs, folder_X, folder_Y, model_folder, save_every=-1):
        if not self.trainable:
            return
        self.sess.run(tf.assign(self.learning_rate, learning_rate))
        self.save(model_folder)
        print("Reading Images")
        images_X = [cv2.imread(os.path.join(folder_X, file), cv2.IMREAD_COLOR) for file in os.listdir(folder_X) if os.path.isfile(os.path.join(folder_X, file))]
        images_Y = [cv2.imread(os.path.join(folder_Y, file), cv2.IMREAD_COLOR) for file in os.listdir(folder_Y) if os.path.isfile(os.path.join(folder_Y, file))]
        batches_X = [[]]
        batches_Y = [[]]
        print("Building Batches")
        for i in range(len(images_X)):
            if len(batches_X[-1]) == self.batch_size:
                batches_X.append([])
            batches_X[-1].append(images_X[i])
        if len(batches_X[-1]) != self.batch_size:
            batches_X = batches_X[:-1]

        for i in range(len(images_Y)):
            if len(batches_Y[-1]) == self.batch_size:
                batches_Y.append([])
            batches_Y[-1].append(images_Y[i])
        if len(batches_Y[-1]) != self.batch_size:
            batches_Y = batches_Y[:-1]
        print("Converting Batches")
        batches_X = [np.stack(batch) for batch in batches_X]
        batches_Y = [np.stack(batch) for batch in batches_Y]

        pool_size = max(len(batches_X), len(batches_Y))
        infinity = float("inf")
        cv2.namedWindow("Generated", cv2.WINDOW_KEEPRATIO)
        epochTimes = []
        for epoch in range(num_epochs):
            print("Epoch " + str(epoch) + " / " + str(num_epochs) + ":")
            random_x = random_element(images_X)#images_X[np.random.randint(0, len(images_X))]
            random_y = random_element(images_Y)#images_Y[np.random.randint(0, len(images_Y))]
            generated_x = self.generate_x(random_y)
            generated_y = self.generate_y(random_x)
            cycle_x = self.generate_x(generated_y)
            cycle_y = self.generate_y(generated_x)
            array = np.vstack([np.hstack([random_x, random_y]), np.hstack([generated_y, generated_x]), np.hstack([cycle_x, cycle_y])])
            cv2.imshow("Generated", array.astype(np.uint8))
            epochStart = time.clock()
            batchTimes = []
            generator_x_losses = []
            generator_y_losses = []
            discriminator_x_losses = []
            discriminator_y_losses = []
            for i in range(pool_size):
                if cv2.waitKey(1) == ord('q'):
                    self.save(model_folder)
                    return
                batch_start = time.clock()
                batch_x = np.stack([random_element(images_X) for i in range(self.batch_size)])#batches_X[np.random.randint(0, len(batches_X))]
                batch_y = np.stack([random_element(images_Y) for i in range(self.batch_size)])#batches_Y[np.random.randint(0, len(batches_X))]
                generator_x_loss, generator_y_loss, discriminator_x_loss, discriminator_y_loss, generator_x_train_status, generator_y_train_status, discriminator_x_train_status, discriminator_y_train_status = self.sess.run(
                    fetches=[
                        self.generate_x_loss, self.generate_y_loss, self.discriminate_x_loss, self.discriminate_y_loss,
                        self.generator_x_trainer, self.generator_y_trainer, self.discriminator_x_trainer, self.discriminator_y_trainer
                    ],
                    feed_dict={
                        self.train_input_x: batch_x,
                        self.train_input_y: batch_y
                    }
                )
                batchTimes.append(time.clock() - batch_start)
                print("Batch " + str(i) + "/" + str(pool_size) + " took " + str(batchTimes[-1]) + " seconds. " + secs_to_string((pool_size - i) * sum(batchTimes) / len(batchTimes)) + " left in epoch.")
                generator_x_losses.append(generator_x_loss)
                generator_y_losses.append(generator_y_loss)
                discriminator_x_losses.append(discriminator_x_loss)
                discriminator_y_losses.append(discriminator_y_loss)
                if generator_x_loss > 1000 or generator_y_loss > 1000:
                    random_x = random_element(batch_x)#batch_x[np.random.randint(0, len(batch_x))]
                    random_y = random_element(batch_y)#batch_y[np.random.randint(0, len(batch_y))]
                    generated_x = self.generate_x(random_y)
                    generated_y = self.generate_y(random_x)
                    cycle_x = self.generate_x(generated_y)
                    cycle_y = self.generate_y(generated_x)
                    array = np.vstack([np.hstack([random_x, random_y]), np.hstack([generated_y, generated_x]), np.hstack([cycle_x, cycle_y])])
                    cv2.imshow("Failed", array.astype(np.uint8))
                    cv2.waitKey(1)
                print("X -> Y Loss:", generator_y_loss)
                print("Y -> X Loss:", generator_x_loss)
                print("D_X Loss:", discriminator_x_loss)
                print("D_Y Loss:", discriminator_y_loss)
            epochTimes.append(time.clock() - epochStart)
            if(epoch == num_epochs - 1):
                print("Epoch took " + secs_to_string(epochTimes[-1]) + ".")
            else:
                print("Epoch took " + secs_to_string(epochTimes[-1]) + ". " + secs_to_string((num_epochs - 1 - epoch) * sum(epochTimes) / len(epochTimes)) + " left.")
            print("Average X->Y Loss:", sum(generator_y_losses) / len(generator_y_losses))
            print("Average Y->X Loss:", sum(generator_x_losses) / len(generator_x_losses))
            print("Average D_X Loss:", sum(discriminator_x_losses) / len(discriminator_x_losses))
            print("Average D_Y Loss:", sum(discriminator_y_losses) / len(discriminator_y_losses))
            if save_every >= 0 and epoch % save_every == 0:
                self.save(model_folder)
        self.save(model_folder)
        cv2.waitKey()

    def save(self, folder):
        discriminator_x_folder = os.path.join(folder, "discriminator_x")
        discriminator_y_folder = os.path.join(folder, "discriminator_y")
        generator_x_folder = os.path.join(folder, "generator_x")
        generator_y_folder = os.path.join(folder, "generator_y")
        for path in (discriminator_x_folder, discriminator_y_folder, generator_x_folder, generator_y_folder):
            if not os.path.exists(path):
                os.mkdir(path)
        self.discriminator_x_filters.save(self.sess, discriminator_x_folder)
        self.discriminator_y_filters.save(self.sess, discriminator_y_folder)
        self.generator_x_filters.save(self.sess, generator_x_folder)
        self.generator_y_filters.save(self.sess, generator_y_folder)

    def generate_x(self, image):
        return np.squeeze(self.sess.run(self.generator_x.output, {self.input_y: np.expand_dims(image, 0)}))

    def generate_y(self, image):
        return np.squeeze(self.sess.run(self.generator_y.output, {self.input_x: np.expand_dims(image, 0)}))

    def discriminate_x(self, image):
        return np.squeeze(self.sess.run(self.discriminator_x.output, {self.input_x: np.expand_dims(image, 0)}))

    def discriminate_y(self, image):
        return np.squeeze(self.sess.run(self.discriminator_y.output, {self.input_y: np.expand_dims(image, 0)}))

with tf.device("/gpu:0"):
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        model_folder = "E:\Programs\Python\TensorFlowCycleGan\model\horse2zebra"
        data_folder_X = "E:/Programs/Python/TensorFlowCycleGan/data/horse2zebra/trainA"
        data_folder_Y = "E:/Programs/Python/TensorFlowCycleGan/data/horse2zebra/trainB", "E:\Programs\Python\TensorFlowCycleGan\model\horse2zebra"
        cyclegan = CycleGan(sess, 256, 256, 3, 1, folder=model_folder)
        cyclegan.train(0.000001, 4, data_folder_X, data_folder_Y, model_folder)
        writer = tf.summary.FileWriter("logs", sess.graph)
