#import numpy as np
import tensorflow as tf
#import pickle
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")
#matplotlib inline

print("TensorFlow Version: {}".format(tf.__version__))



def wavelet(x):
    return tf.multiply(x,3*tf.multiply(1-tf.square(x)/0.2,tf.exp(-(tf.square(x)/0.2))))

#def wavelet(x):
#    return 3*tf.multiply(1-tf.square(x)/0.2,tf.exp(-(tf.square(x)/0.2)))
    
#def dwt_c(imag):
#    """
#    @Haar小波分解  
#    """
#    result=np.zeros((32,32,3))
#    for row in range(0,32):
#        y=0
#        for col in range(0,16):
#            result[row,col]=(imag[row,y]+imag[row,y+1])/2
#            result[row,col+16]=(imag[row,y]-imag[row,y+1])/2
#            y+=2
#            
#            
#    imags=np.zeros((32,32,3))
#    
#    for col in range(0,32):
#        x=0
#        for row in range(0,16):
#            imags[row,col]=(result[x,col]+result[x+1,col])/2
#            imags[row+16,col]=(result[x,col]-result[x+1,col])/2
#            x+=2
#    return imags
#
#
#def dwt_r(imag):
#    """
#    #Haar小波還原
#    """
#    result=np.zeros((32,32,3))
#    for col in range(0,32):
#        x=0
#        for row in range(0,16):
#            result[x,col]=imag[row,col]+imag[row+16,col]
#            result[x+1,col]=imag[row,col]-imag[row+16,col]
#            x+=2  
#            
#    imags=np.zeros((32,32,3))
#    
#    for row in range(0,32):
#        y=0
#        for col in range(0,16):
#            imags[row,y]=result[row,col]+result[row,col+16]
#            imags[row,y+1]=result[row,col]-result[row,col+16]
#            y+=2
#           
#    
#    return imags        
        
def get_inputs(noise_dim, image_height, image_width, image_depth):
    """
    --------------------
    :param noise_dim: 噪聲圖片的size
    :param image_height: 真實圖片的height
    :param image_width: 真實圖片的width
    :param image_depth: 真實圖片的depth
    """ 
    inputs_real = tf.placeholder(tf.float32, [None, image_height, image_width, image_depth], name='inputs_real')
#    inputs_noise = tf.placeholder(tf.float32, [None, noise_dim], name='inputs_noise')
    inputs_noise = tf.placeholder(tf.float32, [None, image_height, image_width, image_depth], name='inputs_noise')
    tf.add_to_collection('inputs_noise', inputs_noise)
    return inputs_real, inputs_noise

#def selu(x):
#    with ops.name_scope('elu') as scope:
#        alpha = 1.6732632423543772848170429916717
#        scale = 1.0507009873554804934193349852946
#        return scale*tf.where(x>=0.0, x, alpha*tf.nn.elu(x))

def get_generator(noise_img, output_dim, is_train=True, alpha=0.2):
    """
    --------------------
    :param noise_img: 噪聲圖片，tensor類型
    :param output_dim: 生成圖片的depth
    :param is_train: 是否為訓練狀態，該參數主要用於作為batch_normalization方法中的參數使用
    :param alpha: Leaky ReLU係數
    """

    with tf.variable_scope("generator", reuse=(not is_train)):
        
        # 32 x 32 x 3 to 16 x 16 x 128
        layer1 = tf.layers.conv2d(noise_img, 128, 7, strides=2,padding='same')
#        layer1 = tf.layers.batch_normalization(layer1, training=is_train)
#        layer1 = tf.maximum(alpha * layer1, layer1)
        layer1 = wavelet(layer1)
#        layer1 = tf.nn.elu(layer1)
#        layer1 = tf.nn.selu(layer1)
#        layer1 = tf.nn.dropout(layer1, keep_prob=0.8)
        tf.add_to_collection('layer1', layer1)
        
        # 16 x 16 x 128 to 8 x 8 x 256
        layer2 = tf.layers.conv2d(layer1, 256, 5, strides=2,padding='same')
#        layer2 = tf.layers.batch_normalization(layer2, training=is_train)
#        layer2 = tf.maximum(alpha * layer2, layer2)
        layer2 = wavelet(layer2)
#        layer2 = tf.nn.elu(layer2)
#        layer2 = tf.nn.selu(layer2)
#        layer2 = tf.nn.dropout(layer2, keep_prob=0.8)
        tf.add_to_collection('layer2', layer2)
        
        # 8 x 8 x 256 to 4 x 4 x 512
        layer3 = tf.layers.conv2d(layer2, 512, 5, strides=2, padding='same')
#        layer3 = tf.layers.batch_normalization(layer3, training=is_train)
#        layer3 = tf.maximum(alpha * layer3, layer3)
        layer3 = wavelet(layer3)
#        layer3 = tf.nn.elu(layer3)
#        layer3 = tf.nn.selu(layer3)
#        layer3 = tf.nn.dropout(layer3, keep_prob=0.8)
        tf.add_to_collection('layer3', layer3)
        
        layer4 = tf.layers.conv2d(layer3, 512, 3, strides=1,padding='same')
#        layer4 = tf.layers.batch_normalization(layer4, training=is_train)
        layer4 = tf.maximum(alpha * layer4, layer4)
#        layer4 = tf.nn.elu(layer4)
#        layer4 = wavelet(layer4)
#        layer4 = tf.nn.selu(layer4)
#        layer4 = tf.nn.dropout(layer4, keep_prob=0.8)
        
        layer4 = tf.layers.conv2d(layer4, 512, 3, strides=1,padding='same')
#        layer4 = tf.layers.batch_normalization(layer4, training=is_train)
        layer4 = tf.maximum(alpha * layer4, layer4)
#        layer4 = tf.nn.elu(layer4)
#        layer4 = wavelet(layer4)
#        layer4 = tf.nn.selu(layer4)
        
                    
        # 4 x 4 x 512 to 8 x 8 x 256
        layer5 = tf.layers.conv2d_transpose(tf.add(layer4,layer3), 256, 3, strides=2,padding='same',name='layer2_c')
#        layer5 = tf.layers.batch_normalization(layer5, training=is_train,name='layer2_b')
        layer5 = tf.maximum(alpha * layer5, layer5)
#        layer5 = tf.nn.elu(layer5)
#        layer5 = wavelet(layer5)
#        layer5 = tf.nn.selu(layer5)
#        layer5 = tf.nn.dropout(layer5, keep_prob=0.8)
        
        
        # 8 x 8 x 256 to 16 x 16 x 128
        layer6 = tf.layers.conv2d_transpose(tf.add(layer5,layer2), 128, 3, strides=2, padding='same',name='layer3_c')
#        layer6 = tf.layers.batch_normalization(layer6, training=is_train ,name='layer3_b')
        layer6 = tf.maximum(alpha * layer6, layer6)
#        layer6 = tf.nn.elu(layer6)
#        layer6 = wavelet(layer6)
#        layer6 = tf.nn.selu(layer6)
#        layer6 = tf.nn.dropout(layer6, keep_prob=0.8)
        tf.add_to_collection('layer6', layer6)
        
        
        # 16 x 16 x 128 to 32 x 32 x 3
        logits = tf.layers.conv2d_transpose(tf.add(layer6,layer1), output_dim, 3, strides=2,padding='same',name='logits')
#        logits = tf.layers.conv2d_transpose(layer6, output_dim, 3, strides=2,padding='same',name='logits')
#        outputs = tf.tanh(logits,name='outputs')
        outputs = tf.sigmoid(logits,name='outputs')
        tf.add_to_collection('outputs', outputs)
        return outputs

def get_discriminator(inputs_img, reuse=False, alpha=0.2):
    """
    --------------------
    @param inputs_img: 輸入圖片，tensor類型
    @param alpha: Leaky ReLU係數
    """
    
    with tf.variable_scope("discriminator", reuse=reuse):
        
        # 32 x 32 x 3 to 16 x 16 x 128
        # 第一層不加入BN
        layer1 = tf.layers.conv2d(inputs_img, 128, 7, strides=2,padding='same')
#        layer1 = tf.maximum(alpha * layer1, layer1)
        layer1 = wavelet(layer1)
#        layer1 = tf.nn.elu(layer1)
#        layer1 = tf.nn.selu(layer1)
#        layer1 = tf.nn.dropout(layer1, keep_prob=0.8)
        
#        layer1 = tf.layers.conv2d(layer1, 128, 5, strides=1, use_bias=False,padding='same')
#        layer1 = tf.layers.batch_normalization(layer1, training=True)
#        layer1 = tf.maximum(alpha * layer1, layer1)
#        layer1 = tf.nn.selu(layer1)
#        layer1 = tf.nn.dropout(layer1, keep_prob=0.8)
        
        # 16 x 16 x 128 to 8 x 8 x 256
        layer2 = tf.layers.conv2d(layer1, 256, 5, strides=2,padding='same')
        layer2 = tf.layers.batch_normalization(layer2, training=True)
        layer2 = tf.maximum(alpha * layer2, layer2)
#        layer2 = tf.nn.elu(layer2)
#        layer2 = tf.nn.selu(layer2)
#        layer2 = tf.nn.dropout(layer2, keep_prob=0.8)
        
#        layer2 = tf.layers.conv2d(layer2, 256, 3, strides=1,use_bias=False,padding='same')
#        layer2 = tf.layers.batch_normalization(layer2, training=True)
#        layer2 = tf.maximum(alpha * layer2, layer2)
#        layer2 = tf.nn.selu(layer2)
#        layer2 = tf.nn.dropout(layer2, keep_prob=0.8)
        
        # 8 x 8 x 256 to 4 x 4 x 512
        layer3 = tf.layers.conv2d(layer2, 512, 3, strides=2, padding='same')
        layer3 = tf.layers.batch_normalization(layer3, training=True)
        layer3 = tf.maximum(alpha * layer3, layer3)
#        layer3 = tf.nn.elu(layer3)
#        layer3 = tf.nn.selu(layer3)
#        layer3 = tf.nn.dropout(layer3, keep_prob=0.8)
        
#        layer3 = tf.layers.conv2d(layer3, 512, 3, strides=1,use_bias=False, padding='same')
#        layer3 = tf.layers.batch_normalization(layer3, training=True)
#        layer3 = tf.maximum(alpha * layer3, layer3)
#        layer3 = tf.nn.selu(layer3)
#        layer3 = tf.nn.dropout(layer3, keep_prob=0.8)
        
        # 4 x 4 x 512 to 4*4*512 x 1
        flatten = tf.reshape(layer3, (-1, 4*4*512))
        logits = tf.layers.dense(flatten, 1)
        outputs = tf.sigmoid(logits)
        tf.add_to_collection('doutputs', outputs)
        
        return logits, outputs

def get_loss(inputs_real, inputs_noise, image_depth, smooth=0.1, alpha=0.5):
    """
    --------------------
    @param inputs_real: 真實圖片，tensor類型
    @param inputs_noise: 噪聲圖片，tensor類型
    @param image_depth: 圖片的depth（或者叫channel）
    @param smooth: label smoothing的參數
    """
#    train_vars = tf.trainable_variables()
    
#    g_vars = [var for var in train_vars if var.name.startswith("generator")]
#    d_vars = [var for var in train_vars if var.name.startswith("discriminator")]
    
    g_outputs = get_generator(inputs_noise, image_depth, is_train=True)
    d_logits_real, d_outputs_real = get_discriminator(inputs_real)
    d_logits_fake, d_outputs_fake = get_discriminator(g_outputs, reuse=True)
    
    # 計算Loss
#    g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake, 
#                                                                    labels=tf.ones_like(d_outputs_fake)*(1-smooth)))
    g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake, 
                                                                    labels=tf.ones_like(d_outputs_fake)))
    
    g_L1 = tf.reduce_sum(tf.losses.absolute_difference(labels=inputs_real, predictions=g_outputs))
    
    
#   g_L2 = tf.losses.mean_squared_error(labels=inputs_real, predictions=g_outputs) 
    
    g_loss = tf.add((1-smooth)*g_L1,smooth*g_loss)
#    g_loss = tf.add(g_loss,g_L2)
    
#    d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_real,
#                                                                         labels=tf.ones_like(d_outputs_real)*(1-smooth)))
    d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_real,
                                                                         labels=tf.ones_like(d_outputs_real)))
    
    d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake,
                                                                         labels=tf.zeros_like(d_outputs_fake)))
    d_loss = tf.add(alpha*d_loss_real, alpha*d_loss_fake)
    
    return g_loss, d_loss

def get_optimizer(g_loss, d_loss, beta1=0.5, learning_rate=0.0002):
    """
    --------------------
    @param g_loss: Generator的Loss
    @param d_loss: Discriminator的Loss
    @learning_rate: 學習率
    """
    
    train_vars = tf.trainable_variables()
    
    g_vars = [var for var in train_vars if var.name.startswith("generator")]
    d_vars = [var for var in train_vars if var.name.startswith("discriminator")]
    
    # Optimizer
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        g_opt = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(g_loss, var_list=g_vars)
        d_opt = tf.train.AdamOptimizer(learning_rate=0.0005, beta1=beta1).minimize(d_loss, var_list=d_vars)
#        g_opt = tf.train.GradientDescentOptimizer(learning_rate).minimize(g_loss, var_list=g_vars)
#        d_opt = tf.train.GradientDescentOptimizer(learning_rate).minimize(d_loss, var_list=d_vars)
    
    return g_opt, d_opt

def plot_images(samples):
#    samples = (samples + 1) / 2
    fig, axes = plt.subplots(nrows=1, ncols=15, sharex=True, sharey=True, figsize=(30,2))
    for img, ax in zip(samples, axes):
        img=img.reshape((32, 32, 3))
#        img=dwt_r(img)
#        for x in range(0,32):
#            for y in range(0,32):
#                for z in range(0,3):
#                    if img[x,y,z]<0:
#                       img[x,y,z]=0
        ax.imshow(img, cmap='Greys_r')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False) 
    fig.tight_layout(pad=0)

def show_generator_output(sess, n_images, inputs_noise, output_dim):
    """
    --------------------
    @param sess: TensorFlow session
    @param n_images: 展示圖片的數量
    @param inputs_noise: 噪聲圖片
    @param output_dim: 圖片的depth（或者叫channel）
    """
    cmap = 'Greys_r'
    noise_shape = inputs_noise.get_shape().as_list()[-1]
    # 生成噪聲圖片
#    examples_noise = np.random.uniform(-1, 1, size=[n_images, noise_shape])
    #print(examples_noise)
#    Img=testImg
    examples_noise=testImg.reshape(-1,32,32,3)
#    Img=dwt_c(testImg)
#    examples_noise=Img.reshape(-1,32*32*3)
    samples = sess.run(get_generator(inputs_noise, output_dim, False),
                       feed_dict={inputs_noise: examples_noise})
#    samples=(samples + 1) / 2
#    samples=samples.reshape((32, 32, 3))
#    samples=dwt_r(samples)
#    for x in range(0,32):
#            for y in range(0,32):
#                for z in range(0,3):
#                    if samples[x,y,z]<0:
#                       samples[x,y,z]=0
#    plt.imshow(samples)
    return samples



# 超參數
batch_size = 64
noise_size = 32*32*3
epochs = 200
n_samples = 1
learning_rate = 0.0002
beta1 = 0.5
losses = []


def train(noise_size, data_shape, batch_size, n_samples):
    """
    --------------------
    @param noise_size: 噪聲size
    @param data_shape: 真實圖片shape
    @batch_size: 批量大小
    @n_samples: 顯示範例圖片的數量
    """


    
    inputs_real, inputs_noise = get_inputs(noise_size, data_shape[1], data_shape[2], data_shape[3])
    g_loss, d_loss = get_loss(inputs_real, inputs_noise, data_shape[-1])
    g_train_opt, d_train_opt = get_optimizer(g_loss, d_loss, beta1, learning_rate)
    
    saver = tf.train.Saver()
    
    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        sess.run(tf.global_variables_initializer())
        # 迭代epoch
        for e in range(epochs):
            for batch_i in range(images.shape[0]//batch_size-1):
#                steps += 1
                batch_images = images[batch_i * batch_size: (batch_i+1)*batch_size]

                # scale to -1, 1
#                batch_images = batch_images * 2 - 1
#                batch_images = [dwt_c(im)*2-1  for im in batch_images]
                # noise
#                batch_noise = np.random.uniform(-1, 1, size=(batch_size, noise_size))
#                batch_noise = batch_noise.reshape(batch_noise.shape[0],32,32,3)
                batch_noise = noise_img[batch_i * batch_size: (batch_i+1)*batch_size]
#                batch_noise = batch_noise*2-1
                # run optimizer
                _ = sess.run(g_train_opt, feed_dict={inputs_real: batch_images,
                                                     inputs_noise: batch_noise})
                _ = sess.run(d_train_opt, feed_dict={inputs_real: batch_images,
                                                     inputs_noise: batch_noise})
                
#                if steps % 100 == 0:
            train_loss_d = d_loss.eval({inputs_real: batch_images,
                                                inputs_noise: batch_noise})
            train_loss_g = g_loss.eval({inputs_real: batch_images,
                                                inputs_noise: batch_noise})
            losses.append((train_loss_d, train_loss_g))
                    # 顯示圖片
            samples = show_generator_output(sess, n_samples, inputs_noise, data_shape[-1])
            plot_images(samples)
#                    saver.save(sess, './checkpoint_dir4/MyModel')#儲存模型
            print("Epoch {}/{}....".format(e+1, epochs), 
                          "Discriminator Loss: {:.4f}....".format(train_loss_d),
                          "Generator Loss: {:.4f}....". format(train_loss_g))
            saver.save(sess, './checkpoint_dir/MyModel')#儲存模型
            
            
with tf.Graph().as_default():
    train(noise_size, [-1, 32, 32, 3], batch_size, n_samples)
plt.show()    
