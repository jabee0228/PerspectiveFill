

import tensorflow as tf
from .layers import Discriminator_block, gen_conv, gen_deconv, gen_deconv_gated, gen_conv_gated, gen_conv_gated_ds, \
    gen_deconv_gated_ds, gen_conv_gated_slice, gen_deconv_gated_slice, dilate_block2, apply_contextual_attention, \
    apply_attention

#from layers import residual_block
# def get_conv_op(conv_type):
#     #gen_conv = ops.GenConvGated()
#     #print(conv_type, 'ds')
#     if conv_type == 'none':
#         conv = gen_conv
#         deconv = gen_deconv
#     elif conv_type == 'regular':
#         conv = gen_conv_gated
#         deconv = gen_deconv_gated
#     elif conv_type == 'ds':
#         conv = gen_conv_gated_ds
#         deconv = gen_deconv_gated_ds
#     elif conv_type == 'slice':
#         conv = gen_conv_gated_slice
#         deconv = gen_deconv_gated_slice
#     else:
#         raise('wrong conv type ' + conv_type)
#     return conv, deconv

# Our model class
class MyModel:
    def __init__(self, name, config=None):
        self.name = name
        self.generator = self.build_generator(config)
        self.discriminator = self.build_discriminator()

    def build_generator(self, config=None, dtype=tf.float32, training=True):
        """
        Build the generator model
        """
        # conv and deconv for refine network
        conv2, deconv2 = gen_conv_gated_ds, gen_deconv_gated_ds
        sz = config.IMG_SHAPE[1]
        nc = config.GEN_NC

        img_input = tf.keras.layers.Input(shape=(512, 512, 3), batch_size=config.BATCH_SIZE)
        mask_input = tf.keras.layers.Input(shape=(512, 512, 1), batch_size=config.BATCH_SIZE)

        # add masked image
        xnow = tf.concat([img_input, mask_input], axis=3) # (512, 512, 4)
        activations = [img_input]
        # encoder
        sz_t = sz
        x = xnow
        nc = max(4, nc // (sz // 512)) // 2 # channel number = 16

        # convolution -> 64x64
        while sz_t > config.BOTTLENECK_SIZE:
            nc *= 2
            sz_t //= 2
            kkernal = 5 if sz_t == sz else 3
            x = conv2(x, nc, 3, 2, name='re_en_down_' + str(sz_t))
            short_cut = x
            x = conv2(x, nc, 3, 1, rate=1, name='re_en_conv_' + str(sz_t))
            x = tf.keras.layers.add([short_cut, x])
            x = tf.keras.layers.ReLU()(x)
            # residual block
            activations.append(x)
        
        # nc = 128
        # dilated conv
        x = dilate_block2(x, name='re_dil', conv_func=conv2)

        # attention
        mask_s = mask_input  # resize_like(mask, x)
        x, match, offset_flow = apply_contextual_attention(x, mask_s, method=config.ATTENTION_TYPE, \
                                                           name='re_att_' + str(sz_t), dtype=dtype, conv_func=conv2)
        # decoder
        activations.pop(-1)
        while sz_t < sz//2:
            nc = nc//2
            sz_t *= 2
            x = deconv2(x, nc, name='re_de_up__'+str(sz_t))
            x = conv2(x, nc, 3, 1, rate=1, name='re_de_conv_'+str(sz_t))
            x_att = apply_attention(activations.pop(-1), match, conv_func = conv2, name='re_de_att_' + str(sz_t))
            x = tf.concat([x_att, x], axis=3)
        
        x = deconv2(x, 3, name='re_de_toRGB__'+str(sz_t))
        x2 = tf.clip_by_value(x, -1., 1.)

        return tf.keras.Model(inputs=[img_input, mask_input], outputs=x2)

    def build_discriminator(self, training=True, nc=64):
        model = tf.keras.Sequential()
        model.add(Discriminator_block('discriminator_block',nc=nc, training=training))
        return model

        



        
