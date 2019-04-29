import tensorflow as tf
import numpy as np
from attention import MultiHeadAttention

def point_wise_feed_forward_network(d_model, dff):
  return tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
      tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
  ])

class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, name, rate=0.1):
        super(EncoderLayer, self).__init__(name=name)

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.experimental.LayerNormalization(epsilon=1e-6,name=name+'_LN1')
        self.layernorm2 = tf.keras.layers.experimental.LayerNormalization(epsilon=1e-6,name=name+'_LN2')

        self.dropout1 = tf.keras.layers.Dropout(rate,name=name+'_dp1')
        self.dropout2 = tf.keras.layers.Dropout(rate,name=name+'_dp2')

    def call(self, inputs, training, mask):
        attn_output, slf_attn_weight = self.mha(inputs, inputs, inputs, mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

        return out2

class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, name, rate=0.1):
        super(DecoderLayer, self).__init__(name=name)

        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)

        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.experimental.LayerNormalization(epsilon=1e-6,name=name+'_LN1')
        self.layernorm2 = tf.keras.layers.experimental.LayerNormalization(epsilon=1e-6,name=name+'_LN2')
        self.layernorm3 = tf.keras.layers.experimental.LayerNormalization(epsilon=1e-6,name=name+'_LN3')

        self.dropout1 = tf.keras.layers.Dropout(rate,name=name+'_dp1')
        self.dropout2 = tf.keras.layers.Dropout(rate,name=name+'_dp2')
        self.dropout3 = tf.keras.layers.Dropout(rate,name=name+'_dp3')

    def call(self, inputs, enc_output, training,
             look_ahead_mask, padding_mask):
        # enc_output.shape == (batch_size, input_seq_len, d_model)

        attn1, attn_weights_block1 = self.mha1(inputs, inputs, inputs, look_ahead_mask)  # (batch_size, target_seq_len, d_model)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + inputs)

        attn2, attn_weights_block2 = self.mha2(
            enc_output, enc_output, out1, padding_mask)  # (batch_size, target_seq_len, d_model)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)  # (batch_size, target_seq_len, d_model)

        ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, d_model)

        return out3, attn_weights_block1, attn_weights_block2

if __name__=='__main__':
    sample_encoder_layer = EncoderLayer(512, 8, 2048,'encoderlayer')

    sample_encoder_layer_output = sample_encoder_layer(tf.random.uniform((64, 43, 512)), False, None)

    print(sample_encoder_layer_output.shape)  # (batch_size, input_seq_len, d_model)

    sample_decoder_layer = DecoderLayer(512, 8, 2048,'decoderlayer')

    sample_decoder_layer_output, _, _ = sample_decoder_layer(
        tf.random.uniform((64, 50, 512)), sample_encoder_layer_output,
        False, None, None)

    print(sample_decoder_layer_output.shape)  # (batch_size, target_seq_len, d_model)