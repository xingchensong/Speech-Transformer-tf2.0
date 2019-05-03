import tensorflow as tf
import numpy as np
from layers import DecoderLayer
from positional_encoding import positional_encoding
from encoder import Encoder

class Decoder(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size,name,
                 pe_max_len=8000,rate=0.1):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers
        print('self.num_layers(decoder): ', self.num_layers)

        self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model,name='de_emb',embeddings_initializer='normal')
        self.pos_encoding = positional_encoding(pe_max_len, self.d_model)

        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, 'DE'+str(_),rate)
                           for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate,name='de_emb_dp')

        # FIXME: 是否需要Share the weight matrix between target word embedding & the final logit dense layer
        # https://github.com/kaituoxu/Speech-Transformer/blob/master/src/transformer/decoder.py#L48
        self.final_layer = tf.keras.layers.Dense(target_vocab_size)
        self.final_layer.set_weights(self.embedding.get_weights())

    def call(self, inputs, training):
        x =inputs[0]
        enc_output = inputs[1]
        look_ahead_mask = inputs[2]
        padding_mask = inputs[3]

        seq_len = tf.shape(x)[1]
        attention_weights = {}

        x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
        x *= tf.math.rsqrt(tf.cast(self.d_model, tf.float32))
        x += tf.cast(self.pos_encoding[:, :seq_len, :],x.dtype)

        # x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](x, enc_output, training,
                                                   look_ahead_mask, padding_mask)

            attention_weights['decoder_layer{}_block1'.format(i + 1)] = block1
            attention_weights['decoder_layer{}_block2'.format(i + 1)] = block2

        # before softmax
        x = self.final_layer(x)

        # x.shape == (batch_size, target_seq_len, target_vocab_size if proj else d_model)
        return x, attention_weights

if __name__=='__main__':
    sample_encoder = Encoder(num_layers=2, d_model=512, num_heads=8,
                             dff=2048, pe_max_len=8500, name='encoder')

    sample_encoder_output = sample_encoder((tf.random.uniform((2, 64, 62)),None)
                                           ,training=True)

    print(sample_encoder_output.shape)  # (batch_size, input_seq_len, d_model)

    sample_decoder = Decoder(num_layers=2, d_model=512, num_heads=8,
                             dff=2048, target_vocab_size=32,name='decoder',pe_max_len=8000)

    output, attn = sample_decoder((tf.random.uniform((2, 26)),sample_encoder_output,
                                   None,None),training=True,)

    # sample_encoder.summary()
    # sample_decoder.summary()
    print(output.shape, attn['decoder_layer2_block2'].shape)
    # (batchsize, target_seq_len, d_model) (batchsize, numheads, target_seq_len, input_seq_len)