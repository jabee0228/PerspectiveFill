import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np

def gen_conv(x, cnum, ksize, stride=1, rate=1, name='conv',
             padding='SAME', activation=tf.nn.elu, training=True, dtype=tf.float32):
    x = conv2d(x, cnum, ksize, stride, dilation_rate=rate,
        activation=activation, padding=padding, name=name, dtype=dtype)
    return x

def gen_deconv(x, cnum, name='upsample', padding='SAME', training=True, dtype=tf.float32):
    x = tf.image.resize(x, size=[x.shape[1] * 2, x.shape[2] * 2], method='bilinear')
    x = gen_conv(x, cnum, 3, 1, name=name+'_conv', padding=padding,
            training=training, dtype=dtype)
    return x
def conv2d(x, output_dim, ksize, stride, dilation_rate=1, activation=None, padding='same', name='conv',
           dtype=tf.float32):
    # 使用 Keras 的 Conv2D 層來替代自定義的卷積操作
    conv_layer = tf.keras.layers.Conv2D(
        filters=output_dim,
        kernel_size=ksize,
        strides=stride,
        dilation_rate=dilation_rate,
        activation=activation,  # 激活函數
        padding=padding,
        #name=name,
        dtype=dtype,
        kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.05),  # 權重初始化
        bias_initializer=tf.keras.initializers.Constant(0.0)  # 偏置初始化
    )

    # 將輸入 x 通過 Conv2D 層進行卷積操作
    conv = conv_layer(x)
    return conv

def conv2d_layer(output_dim, ksize, stride, dilation_rate=1, activation=None, padding='same', name='conv',
           dtype=tf.float32):
    conv_layer = tf.keras.layers.Conv2D(
        filters=output_dim,
        kernel_size=ksize,
        strides=stride,
        dilation_rate=dilation_rate,
        activation=activation,  # 激活函數
        padding=padding,
        #name=name,
        dtype=dtype,
        kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.05),  # 權重初始化
        bias_initializer=tf.keras.initializers.Constant(0.0)  # 偏置初始化
    )
    return conv_layer

def gen_conv_gated(x, cnum, ksize, stride=1, rate=1, name='conv',
             padding='SAME', slim=True, activation=None, training=True, dtype=tf.float32):
    x1 = conv2d(x, cnum, ksize, stride, dilation_rate=rate,
        activation=None, padding=padding, name=name+'_feat', dtype=dtype)
    x2 = conv2d(x, cnum, ksize, stride, dilation_rate=rate,
        activation=None, padding=padding, name=name+'_gate', dtype=dtype)
    #x1, x2 = tf.split(x, 2, axis=3)
    x = tf.sigmoid(x2) * tf.nn.elu(x1)
    return x

def conv2d_ds(x, output_dim, ksize, stride, dilation_rate=1, activation=None, \
                        padding='SAME', name='conv', dtype=tf.float32):
    # 使用 Keras 的 SeparableConv2D 進行深度可分離卷積
    conv_layer = layers.SeparableConv2D(
        filters=output_dim,
        kernel_size=ksize,
        strides=stride,
        dilation_rate=dilation_rate,
        padding=padding,
        activation=activation,
        depthwise_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.05),
        pointwise_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.05),
        bias_initializer=tf.keras.initializers.Constant(0.0),
        name=name,
        dtype=dtype
    )

    # 將輸入 x 通過 SeparableConv2D 層進行卷積操作
    y = conv_layer(x)
    return y

def gen_deconv_gated_ds(x, cnum, name='upsample', padding='SAME', training=True, dtype=tf.float32):
    x = tf.image.resize(x, size=[x.shape[1] * 2, x.shape[2] * 2], method='bilinear')
    #x = resize(x, func=tf.compat.v1.image.resize_bilinear)
    x = gen_conv_gated_ds( x, cnum, 3, 1, name=name+'_conv', padding=padding,
        training=training, dtype=dtype)
    return x

def gen_conv_gated_ds(x, cnum, ksize, stride=1, rate=1, name='conv',
             padding='SAME', slim=True, activation=None, training=True, dtype=tf.float32):
    x1 = conv2d(x, cnum, ksize, stride, dilation_rate=rate,
        activation=None, padding=padding, name=name, dtype=dtype)
    x2 = conv2d_ds(x, cnum, 3, stride, dilation_rate=1,
        activation=None, padding=padding, name=name, dtype=dtype)
    x = tf.sigmoid(x2) * tf.nn.elu(x1)
    return x

def gen_conv_gated_slice(x, cnum, ksize, stride=1, rate=1, name='conv',
             padding='SAME', slim=True, activation=None, training=True, dtype=tf.float32):
    x1 = conv2d(x, cnum, ksize, stride, dilation_rate=rate,
        activation=None, padding=padding, name=name+'_feat', dtype=dtype)
    x2 = conv2d(x, 1, ksize, stride, dilation_rate=rate,
        activation=None, padding=padding, name=name+'_gate', dtype=dtype)
    #x1, x2 = tf.split(x, [cnum,1], axis=3)
    x = tf.sigmoid(x2) * tf.nn.elu(x1)
    return x

def gen_deconv_gated_slice(x, cnum, name='upsample', padding='SAME', training=True, dtype=tf.float32):
    #with tf.compat.v1.variable_scope(name):
        #x = resize(x, func=tf.compat.v1.image.resize_bilinear)
    x = tf.image.resize(x, size=[x.shape[1] * 2, x.shape[2] * 2], method='bilinear')
    x = gen_conv_gated_slice(  x, cnum, 3, 1, name=name+'_conv', padding=padding,
        training=training, dtype=dtype)
    return x

def gen_deconv_gated(x, cnum, name='upsample', padding='SAME', training=True, dtype=tf.float32):
    x = tf.image.resize(x, size=[x.shape[1] * 2, x.shape[2] * 2], method='bilinear')
    #with tf.variable_scope(name):
    #   x = resize(x, func=tf.image.resize_bilinear)
    x = gen_conv_gated( x, cnum, 3, 1, name=name+'_conv', padding=padding,
        training=training, activation=None, dtype=dtype)
    return x

def dilate_block2(x, name, conv_func):
    sz = x.get_shape().as_list()[1]
    nc = x.get_shape().as_list()[3]
    #conv_func = gen_conv_gated
    x = conv_func(x, nc, 3, 1, name= name + '_d1')
    x = conv_func(x, nc, 3, rate=2, name= name + '_d2')
    x = conv_func(x, nc, 3, rate=4, name= name+ '_d4')
    x = conv_func(x, nc, 3, rate=8, name= name + '_d8')
    x = conv_func(x, nc, 3, rate=16, name= name + '_d16')
    return x
def apply_contextual_attention(x, mask_s, method = 'SOFT', name='attention', dtype=tf.float32, conv_func = None):
    # mask_s shape = [4, 512, 512, 1]
    x_hallu = x
    sz = x.get_shape().as_list()[1]
    nc = x.get_shape().as_list()[3]
    x, corres, flow = contextual_attention(x, x, mask_s, method = method, ksize=3, rate=2, fuse=True, dtype=dtype)
    x = conv_func(x, nc, 3, 1, name= name + '_att1')
    #x = conv_func(x, nc, 3, 1, name= name + '_att2')
    x = tf.concat([x_hallu, x], axis=3)
    x = conv_func(x, nc, 3, 1, name= name + '_att3')
    #x = conv_func(x, nc, 3, 1, name= name + '_att4')
    return x, corres, flow

def contextual_attention(src, ref,mask=None,  method='SOFT', ksize=3, rate=1,
                         fuse_k=3, softmax_scale=10., fuse=True, dtype=tf.float32):
    # original: mask shape: [1, 512, 512, 1]
    # mask shape: [4, 512, 512, 1]

    # get shapes
    shape_src = src.get_shape().as_list()
    shape_ref = ref.get_shape().as_list()
    assert shape_src[0] == shape_ref[0] and shape_src[3] == shape_ref[3], 'error'
    batch_size = shape_src[0]
    nc = shape_src[3]

    # raw features
    kernel = rate * 2 - 1
    raw_feats = tf.compat.v1.extract_image_patches(ref, [1,kernel,kernel,1], [1,rate,rate,1], [1,1,1,1], padding='SAME')
    # raw_feats.shape = [batch, 32, 32, 1152]
    raw_feats = tf.reshape(raw_feats, [batch_size, -1, kernel, kernel, nc])
    raw_feats = tf.transpose(raw_feats, [0, 2, 3, 4, 1])  # transpose to b*k*k*c*hw
    raw_feats_lst = tf.split(raw_feats, batch_size, axis=0)

    # resize
    src = downsample(src, rate) # ??
    ref = downsample(ref, rate)

    #ss = tf.shape(src) # orginal
    ss = src.get_shape().as_list()
    #rs = tf.shape(ref) # orginal
    rs = ref.get_shape().as_list()
    # ss = [4, 32, 32, 512]

    shape_s = src.get_shape().as_list()
    shape_r = ref.get_shape().as_list()
    src_lst = tf.split(src, batch_size, axis=0)


    feats = tf.compat.v1.extract_image_patches(ref, [1,ksize,ksize,1], [1,1,1,1], [1,1,1,1], padding='SAME')
    feats = tf.reshape(feats, [batch_size, -1, ksize, ksize, nc])
    feats = tf.transpose(feats, [0, 2, 3, 4, 1])  # transpose to b*k*k*c*hw
    feats_lst = tf.split(feats, batch_size, axis=0)

    # process mask
    """
    if mask is None:
        mask = tf.zeros([1]+ shape_ref[1:3] + [1], dtype=dtype)
    mask = resize(mask, to_shape=[32,32], func=tf.image.resize_nearest_neighbor)
    mask = tf.extract_image_patches(mask, [1,ksize,ksize,1], [1,1,1,1], [1,1,1,1], padding='SAME')
    mask = tf.reshape(mask, [1, -1, ksize, ksize, 1])
    mask = tf.transpose(mask, [0, 2, 3, 4, 1])[0]  # bs k k c hw
    mask = tf.cast(tf.equal(tf.reduce_mean(mask, axis=[0,1,2], keepdims=True), 0.), dtype)

    """
    mask_lst = tf.split(mask, batch_size, axis=0)
    #mask = resize(mask, to_shape=[32,32], func=tf.image.resize_nearest_neighbor)
    new_mask_lst = []
    for mask in mask_lst:    
        mask = tf.nn.max_pool(mask, [1,16,16,1], [1,16,16,1],'SAME')
        mask = tf.nn.max_pool(mask, [1,3,3,1], [1,1,1,1],'SAME')
        mask = 1 - mask
        new_mask = tf.reshape(mask, [1, 1, 1, -1])
        new_mask_lst.append(new_mask)


    y_lst, y_up_lst = [], []
    offsets = []
    fuse_weight = tf.reshape(tf.eye(fuse_k, dtype=dtype), [fuse_k, fuse_k, 1, 1])
    for x, r, raw_r, mask in zip(src_lst, feats_lst, raw_feats_lst, new_mask_lst):
        r = r[0]
        r = r / tf.maximum(tf.sqrt(tf.reduce_sum(tf.square(r), axis=[0,1,2])), 1e-8)
        y = tf.nn.conv2d(x, r, strides=[1,1,1,1], padding="SAME")
        # y shape: (1, 32, 32, 4096)
        if fuse:
            # (1, 1024, 1024, 1)
            yi = tf.reshape(y, [1, ss[1]*ss[2], rs[1]*rs[2], 1]) # error occured
            yi = tf.nn.conv2d(yi, fuse_weight, strides=[1,1,1,1], padding='SAME')
            yi = tf.reshape(yi, [1, ss[1], ss[2], rs[1], rs[2]])
            yi = tf.transpose(yi, [0, 2, 1, 4, 3])
            yi = tf.reshape(yi, [1, ss[1]*ss[2], rs[1]*rs[2], 1])
            yi = tf.nn.conv2d(yi, fuse_weight, strides=[1,1,1,1], padding='SAME')
            yi = tf.reshape(yi, [1, ss[2], ss[1], rs[2], rs[1]])
            yi = tf.transpose(yi, [0, 2, 1, 4, 3])
            y = yi
        y = tf.reshape(y, [1, ss[1], ss[2], rs[1]*rs[2]])
        if method == 'HARD':
            ym = tf.reduce_max(y, keepdims=True, axis=3)
            y = y * mask
            coef = tf.cast( tf.greater_equal(y , tf.reduce_max(y, keepdims=True, axis=3)), dtype)
            y =  tf.pow( coef * tf.divide(y, ym + 1e-04 ), 2)
        elif method == 'SOFT':
            # 1024, 4096
            y = tf.nn.softmax(y * mask * softmax_scale, 3) * mask
        y.set_shape([1, shape_s[1], shape_s[2], shape_r[1]*shape_r[2]])

        if dtype == tf.float32:
            offset = tf.argmax(y, axis=3, output_type=tf.int32)
            offsets.append(offset)
        feats = raw_r[0]
        y_up = tf.nn.conv2d_transpose(y, feats, [1] + shape_src[1:], strides=[1,rate,rate,1])
        y_lst.append(y)
        y_up_lst.append(y_up)

    out, correspondence = tf.concat(y_up_lst, axis=0), tf.concat(y_lst, axis=0)
    out.set_shape(shape_src)

    #print(correspondence.get_shape().as_list())
    #correspondence.reshape([ss[0], ss[1], ss[2], -1])
    if dtype == tf.float32:
        offsets = tf.concat(offsets, axis=0)
        offsets = tf.stack([offsets // ss[2], offsets % ss[2]], axis=-1)
        offsets.set_shape(shape_s[:3] + [2])
        h_add = tf.tile(tf.reshape(tf.range(ss[1]), [1, ss[1], 1, 1]), [ss[0], 1, ss[2], 1])
        w_add = tf.tile(tf.reshape(tf.range(ss[2]), [1, 1, ss[2], 1]), [ss[0], ss[1], 1, 1])
        offsets = offsets - tf.concat([h_add, w_add], axis=3)
        flow = flow_to_image_tf(offsets)
        flow = ResizeLayer(scale=rate)(flow)
        # flow = resize(flow, scale=rate, func=tf.image.resize())
    else:
        flow = None
    return out, correspondence, flow

class ResizeLayer(tf.keras.layers.Layer):
    def __init__(self, to_shape=None, scale=None):
        super(ResizeLayer, self).__init__()
        self.scale = scale
        self.to_shape = to_shape

    def call(self, img):
        scale = self.scale
        to_shape = self.to_shape
        if to_shape is None:
            if scale is None:
                to_shape = img.get_shape().as_list()[1:3]
                to_shape[0], to_shape[1] = to_shape[0] * 2, to_shape[1] * 2
            else:
                to_shape = img.get_shape().as_list()[1:3]
                to_shape[0], to_shape[1] = int(to_shape[0] * scale), int(to_shape[1] * scale)
        return tf.compat.v1.image.resize_nearest_neighbor(img, to_shape)

def resize(img, to_shape = None, scale =None, func = None):
    if to_shape is None:
      if scale is None:
        to_shape = img.get_shape().as_list()[1:3]
        to_shape[0], to_shape[1] = to_shape[0] * 2, to_shape[1] * 2
      else:
        to_shape = img.get_shape().as_list()[1:3]
        to_shape[0], to_shape[1] = int(to_shape[0] * scale), int(to_shape[1] * scale)
    return func(img, to_shape)

def make_color_wheel():
    RY, YG, GC, CB, BM, MR = (15, 6, 4, 11, 13, 6)
    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros([ncols, 3])
    col = 0
    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.transpose(np.floor(255*np.arange(0, RY) / RY))
    col += RY
    # YG
    colorwheel[col:col+YG, 0] = 255 - np.transpose(np.floor(255*np.arange(0, YG) / YG))
    colorwheel[col:col+YG, 1] = 255
    col += YG
    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.transpose(np.floor(255*np.arange(0, GC) / GC))
    col += GC
    # CB
    colorwheel[col:col+CB, 1] = 255 - np.transpose(np.floor(255*np.arange(0, CB) / CB))
    colorwheel[col:col+CB, 2] = 255
    col += CB
    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.transpose(np.floor(255*np.arange(0, BM) / BM))
    col += + BM
    # MR
    colorwheel[col:col+MR, 2] = 255 - np.transpose(np.floor(255 * np.arange(0, MR) / MR))
    colorwheel[col:col+MR, 0] = 255
    return colorwheel

def compute_color(u,v):
    h, w = u.shape
    img = np.zeros([h, w, 3])
    nanIdx = np.isnan(u) | np.isnan(v)
    u[nanIdx] = 0
    v[nanIdx] = 0
    # colorwheel = COLORWHEEL
    colorwheel = make_color_wheel()
    ncols = np.size(colorwheel, 0)
    rad = np.sqrt(u**2+v**2)
    a = np.arctan2(-v, -u) / np.pi
    fk = (a+1) / 2 * (ncols - 1) + 1
    k0 = np.floor(fk).astype(int)
    k1 = k0 + 1
    k1[k1 == ncols+1] = 1
    f = fk - k0
    for i in range(np.size(colorwheel,1)):
        tmp = colorwheel[:, i]
        col0 = tmp[k0-1] / 255
        col1 = tmp[k1-1] / 255
        col = (1-f) * col0 + f * col1
        idx = rad <= 1
        col[idx] = 1-rad[idx]*(1-col[idx])
        notidx = np.logical_not(idx)
        col[notidx] *= 0.75
        img[:, :, i] = np.uint8(np.floor(255 * col*(1-nanIdx)))
    return img

def flow_to_image(flow):
    out = []
    maxu = -999.
    maxv = -999.
    minu = 999.
    minv = 999.
    maxrad = -1
    for i in range(flow.shape[0]):
        u = flow[i, :, :, 0]
        v = flow[i, :, :, 1]
        idxunknow = (abs(u) > 1e7) | (abs(v) > 1e7)
        u[idxunknow] = 0
        v[idxunknow] = 0
        maxu = max(maxu, np.max(u))
        minu = min(minu, np.min(u))
        maxv = max(maxv, np.max(v))
        minv = min(minv, np.min(v))
        rad = np.sqrt(u ** 2 + v ** 2)
        maxrad = max(maxrad, np.max(rad))
        u = u/(maxrad + np.finfo(float).eps)
        v = v/(maxrad + np.finfo(float).eps)
        img = compute_color(u, v)
        out.append(img)
    return np.float32(np.uint8(out))

def flow_to_image_tf(flow, name='flow_to_image'):
    # 使用 tf.name_scope 替代 variable_scope
    with tf.name_scope(name):
        # 使用 tf.numpy_function 來包裝 numpy 函數
        img = tf.numpy_function(flow_to_image, [flow], tf.float32)

        # 設置圖像的形狀
        img.set_shape(flow.shape.as_list()[0:-1] + [3])

        # 調整圖像的值範圍到 [-1, 1]
        img = img / 127.5 - 1.0

        return img

def downsample(x, rate):
    shp = x.get_shape().as_list()
    assert shp[1] % rate == 0 and shp[2] % rate == 0, 'height and width should be multiples of rate'
    shp[1], shp[2] = shp[1]//rate, shp[2]//rate
    x = tf.compat.v1.extract_image_patches(x, [1,1,1,1], [1,rate,rate,1], [1,1,1,1], padding='SAME')
    return tf.reshape(x, shp)

'''
def downsample(x, rate):
    # 獲取輸入張量的形狀
    shp = x.shape.as_list()
    # 確保高度和寬度是 rate 的整數倍
    height_mod = tf.math.mod(shp[1], rate)
    width_mod = tf.math.mod(shp[2], rate)
    if height_mod != 0 or width_mod!= 0: # error
        raise ValueError('height and width should be multiples of rate')

    # 計算新的高度和寬度
    new_height = shp[1] // rate
    new_width = shp[2] // rate
    
    # 使用 extract_patches 函數來提取圖像的補丁
    x = tf.image.extract_patches(
        images=x,
        sizes=[1, rate, rate, 1],
        strides=[1, rate, rate, 1],
        rates=[1, 1, 1, 1],
        padding='SAME'
    )

    # 調整形狀以達到下採樣的效果
    return tf.reshape(x, shape=(shp[0], new_height, new_width, -1))
'''
    
def apply_attention(x, correspondence, conv_func, name):
    shp = x.get_shape().as_list()
    shp_att = correspondence.get_shape().as_list()
    #print(shp, shp_att)
    rate = shp[1]// shp_att[1]
    kernel = rate * 2
    batch_size = shp[0]
    sz = shp[1]
    nc = shp[3]
    raw_feats = tf.compat.v1.extract_image_patches(x, [1,kernel,kernel,1], [1,rate,rate,1], [1,1,1,1], padding='SAME')
    raw_feats = tf.reshape(raw_feats, [batch_size, -1, kernel, kernel, nc])
    raw_feats = tf.transpose(raw_feats, [0, 2, 3, 4, 1])  # transpose to b*k*k*c*hw
    raw_feats_lst = tf.split(raw_feats, batch_size, axis=0)
    
    ys = []
    att_lst = tf.split(correspondence, batch_size, axis=0)
    for feats, att in zip(raw_feats_lst, att_lst):
        #print(att.get_shape().as_list(), feats.get_shape().as_list())
        y = tf.nn.conv2d_transpose(att, feats[0], [1] + shp[1:], strides=[1,rate,rate,1])
        ys.append(y)
    out = tf.concat(ys, axis=0)
    if conv_func is not None:
      out = conv_func(out, nc, 3, 1, rate=1, name = name + '_1')
      out = conv_func(out, nc, 3, 1, rate=2, name = name + '_2')
    return out

'''
def apply_attention(x, correspondence, conv_func, name):
    shp = tf.shape(x)  # 使用 tf.shape 獲取動態形狀
    shp_att = tf.shape(correspondence)
    rate = shp[1] // shp_att[1]
    kernel = rate * 2
    batch_size = shp[0]
    sz = shp[1]
    nc = shp[3]

    # 使用 TensorFlow v2 的 extract_patches
    raw_feats = tf.image.extract_patches(
        images=x,
        sizes=[1, kernel, kernel, 1],
        strides=[1, rate, rate, 1],
        rates=[1, 1, 1, 1],
        padding='SAME'
    )
    raw_feats = tf.reshape(raw_feats, [batch_size, -1, kernel, kernel, nc])
    raw_feats = tf.transpose(raw_feats, [0, 2, 3, 4, 1])  # transpose to b*k*k*c*hw
    raw_feats_lst = tf.split(raw_feats, batch_size, axis=0)

    ys = []
    att_lst = tf.split(correspondence, batch_size, axis=0)
    for feats, att in zip(raw_feats_lst, att_lst):
        # 使用 tf.nn.conv2d_transpose 進行反向卷積
        y = tf.nn.conv2d_transpose(
            input=att,
            filters=feats[0],
            output_shape=[1, sz, sz, nc],
            strides=[1, rate, rate, 1],
            padding='SAME'
        )
        ys.append(y)
    out = tf.concat(ys, axis=0)

    if conv_func is not None:
        out = conv_func(out, nc, 3, 1, rate=1, name=name + '_1')
        out = conv_func(out, nc, 3, 1, rate=2, name=name + '_2')

    return out
'''

def dis_conv(x, cnum, ksize=5, stride=2, name='conv', dtype=tf.float32):
    x = conv2d(x, cnum, ksize, stride, padding='SAME', name=name, dtype=dtype,activation='leaky_relu')
    # x = tf.nn.leaky_relu(x)
    return x

def flatten(x, name=""):
    return tf.reshape(x, [x.get_shape().as_list()[0], -1], name=name)

class Discriminator_block(keras.layers.Layer):
    def __init__(self, name, training=True, nc=64, ksize=5, stride=2, dtype=tf.float32):
        super(Discriminator_block, self).__init__(name=name)
        self.training = training
        self.nc = nc
        self.ksize = ksize
        self.stride = stride
        self.conv2d_1 = conv2d_layer(nc, ksize, stride, padding='SAME', name='conv1', dtype=dtype)
        self.conv2d_2 = conv2d_layer(nc*2, ksize, stride, padding='SAME', name='conv2', dtype=dtype)
        self.conv2d_3 = conv2d_layer(nc*4, ksize, stride, padding='SAME', name='conv3', dtype=dtype)
        self.conv2d_4 = conv2d_layer(nc*4, ksize, stride, padding='SAME', name='conv4', dtype=dtype)
        self.conv2d_5 = conv2d_layer(nc*4, ksize, stride, padding='SAME', name='conv5', dtype=dtype)
        self.conv2d_6 = conv2d_layer(nc*4, ksize, stride, padding='SAME', name='conv6', dtype=dtype)
        
    def call(self, x):
        x = self.conv2d_1(x)
        x = keras.layers.LeakyReLU()(x)
        x = self.conv2d_2(x)
        x = keras.layers.LeakyReLU()(x)
        x = self.conv2d_3(x)
        x = keras.layers.LeakyReLU()(x)
        x = self.conv2d_4(x)
        x = keras.layers.LeakyReLU()(x)
        x = self.conv2d_5(x)
        x = keras.layers.LeakyReLU()(x)
        x = self.conv2d_6(x)
        x = flatten(x)
        return x
    