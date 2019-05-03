import tensorflow as tf
import numpy as np
from layers import EncoderLayer
import matplotlib.pyplot as plt
from positional_encoding import positional_encoding


class Encoder(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, pe_max_len,name,
                 dp=0.1):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers
        print('self.num_layers(encoder) ',self.num_layers)
        self.rate =dp

        self.pos_encoding = positional_encoding(pe_max_len, self.d_model) # 采用类似缓存的思想，申请超长的pe，后面只会用到一小部分

        self.input_proj = tf.keras.models.Sequential(name='en_proj')
        self.input_proj.add(tf.keras.layers.Dense(units=self.d_model,kernel_initializer='glorot_normal'))
        # self.input_proj.add(tf.keras.layers.Dropout(rate=dp))
        self.input_proj.add(tf.keras.layers.experimental.LayerNormalization(epsilon=1e-6))

        self.dropout = tf.keras.layers.Dropout(rate=0.1, name='en_proj_dp')

        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, 'EN'+str(_),dp)
                           for _ in range(num_layers)]

    def call(self, inputs, training):
        x = inputs[0] # B*T*D*n
        mask = inputs[1]
        seq_len = tf.shape(x)[1]

        x = tf.reshape(x,[x.shape[0],x.shape[1],-1]) # B*T*(D*n)  flatten on channels

        # doing projection and adding position encoding.
        x = self.input_proj(x)  # (batch_size, input_seq_len, d_model)
        # x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += tf.cast(self.pos_encoding[:, :seq_len, :], x.dtype)

        # print('dropout.rate: ',str(self.dropout.rate))
        # self.dropout.rate = self.rate
        # print('dropout.rate: ', str(self.dropout.rate))
        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)

        return x  # (batch_size, input_seq_len, d_model)


if __name__=='__main__':
    pos_encoding = positional_encoding(50, 512)
    print(pos_encoding.shape)

    plt.pcolormesh(pos_encoding[0], cmap='RdBu')
    plt.xlabel('Depth')
    plt.xlim((0, 512))
    plt.ylabel('Position')
    plt.colorbar()
    plt.show()

    sample_encoder = Encoder(num_layers=2, d_model=512, num_heads=8,dff=2048, pe_max_len=8500,name='Encoder',dp =0.1)

    sample_encoder_output = sample_encoder((tf.random.normal((2,64, 62)),None)
                                           ,training=True)

    # sample_encoder.summary()
    print(sample_encoder_output.shape)  # (batch_size, input_seq_len, d_model)