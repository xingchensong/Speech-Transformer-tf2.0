import tensorflow as tf
import numpy as np

def scaled_dot_product_attention(q, k, v, mask):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    The mask has different shapes depending on its type(padding or look ahead)
    but it must be broadcastable for addition.

    Args:
      q: query shape == (..., seq_len_q, depth) or (N, num_heads, seq_len_q, depth)
      k: key shape == (..., seq_len_k, depth) or (N, num_heads, seq_len_k, depth)
      v: value shape == (..., seq_len_v, depth) or (N, num_heads, seq_len_v, depth)
      mask: Float tensor with shape broadcastable
            to (..., seq_len_q, seq_len_k). Defaults to None.
            shape == (N, 1, 1, seq_len_k) [用在encoder和docoder block2，后者seq_len_q和seq_len_k不同] or (N, 1, seq_len_q, seq_len_k) [用在decoder block1，三个len相同]
    Returns:
      output, attention_weights
    """

    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (N, heads, seq_q, d_k)*(N, heads, d_k, seq_k)=(N, heads, seq_q, seq_k)

    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.cast(tf.math.sqrt(dk),matmul_qk.dtype)
    # print('shape'+str(scaled_attention_logits.shape.as_list()))

    # add the mask to the scaled tensor.
    # -inf经过softmax后会接近于0，一方面保证句子里pad的部分几乎不会分到注意力，另一方面也保证解码的时候当前时间步后面的部分句子不会分配到注意力
    if mask is not None:
        scaled_attention_logits += tf.cast((mask * -1e9),scaled_attention_logits.dtype) # mask中padding部分是1,使得logits变成-inf

    # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

    # FIXME: 可能不需要dropout https://github.com/kaituoxu/Speech-Transformer/blob/master/src/transformer/attention.py#L83
    # attention_weights = tf.keras.layers.Dropout(rate=0.1)(attention_weights)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_v, depth) 实际上是(..., seq_len_q, depth)，只是三种len都一样

    return output, attention_weights

def print_out(q, k, v):
  temp_out, temp_attn = scaled_dot_product_attention(
      q, k, v, None)
  print ('Attention weights are:')
  print (temp_attn)
  print ('Output is:')
  print (temp_out)


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        # https://github.com/kaituoxu/Speech-Transformer/blob/master/src/transformer/attention.py#L19
        init  = tf.keras.initializers.RandomNormal(mean=0,stddev=np.sqrt(2.0 / (d_model + self.depth)))
        # init = tf.keras.initializers.glorot_normal()
        self.wq = tf.keras.layers.Dense(d_model,kernel_initializer=init) # (feature_in_dim, d_model) 第一维不是batchsize因为可以broadcast
        self.wk = tf.keras.layers.Dense(d_model,kernel_initializer=init) # (feature_in_dim, d_model)
        self.wv = tf.keras.layers.Dense(d_model,kernel_initializer=init) # (feature_in_dim, d_model)

        self.dense = tf.keras.layers.Dense(d_model,kernel_initializer='glorot_normal')# (feature_in_dim, d_model)

        # self.activation = tf.keras.layers.Activation(activation='relu')

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth)) # -1 for seq_len,
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        '''

        :param v: input data , shape(batch_size, seq_len, feature_in_dim)
        :param k: input data , shape(batch_size, seq_len, feature_in_dim)
        :param q: input data , shape(batch_size, seq_len, feature_in_dim)
        :param mask: padding mask, shape(batchsize, 1, 1, seq_len) # 1 for broadcast
        :return:
        '''
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_v, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention,
                                        perm=[0, 2, 1, 3])  # (batch_size, seq_len_v, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))  # (batch_size, seq_len_v, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_v, d_model)

        # output = self.activation(output)

        return output, attention_weights


class TwoD_Attention_layer(tf.keras.layers.Layer):
    def __init__(self,n = 64,c =64):
        super(TwoD_Attention_layer, self).__init__()
        self.n = n
        self.c = c
        self.convq = tf.keras.layers.Conv2D(c, 3, 1, 'same', kernel_initializer='glorot_normal')
        self.convk = tf.keras.layers.Conv2D(c, 3, 1, 'same', kernel_initializer='glorot_normal')
        self.convv = tf.keras.layers.Conv2D(c, 3, 1, 'same', kernel_initializer='glorot_normal')
        self.conv = tf.keras.layers.Conv2D(n, 3, 1, 'same', kernel_initializer='glorot_normal')
        self.bnq = tf.keras.layers.BatchNormalization()
        self.bnk = tf.keras.layers.BatchNormalization()
        self.bnv = tf.keras.layers.BatchNormalization()
        self.ln = tf.keras.layers.experimental.LayerNormalization()

        self.final_conv1 = tf.keras.layers.Conv2D(n, 3, 1, 'same', activation='relu',
                                                  kernel_initializer='glorot_normal')
        self.final_conv2 = tf.keras.layers.Conv2D(n, 3, 1, 'same', kernel_initializer='glorot_normal')
        self.bnf1 = tf.keras.layers.BatchNormalization()
        self.bnf2 = tf.keras.layers.BatchNormalization()
        self.act = tf.keras.layers.Activation('relu')

    def call(self,inputs,training):
        '''

        :param inputs: B*T*D*n
        :return: B*T*D*n
        '''
        residual = inputs
        batch_size = tf.shape(inputs)[0]
        q = self.bnq(self.convq(inputs),training=training)
        k = self.bnk(self.convk(inputs),training=training)
        v = self.bnv(self.convv(inputs),training=training)

        q_time = tf.transpose(q,[0,3,1,2])
        k_time = tf.transpose(k, [0, 3, 1, 2])
        v_time = tf.transpose(v, [0, 3, 1, 2])

        q_fre = tf.transpose(q,[0,3,2,1])
        k_fre = tf.transpose(k, [0, 3, 2, 1])
        v_fre = tf.transpose(v, [0, 3, 2, 1])

        scaled_attention_time, attention_weights_time = scaled_dot_product_attention(
            q_time, k_time, v_time, None)  # B*c*T*D
        scaled_attention_fre, attention_weights_fre = scaled_dot_product_attention(
            q_fre, k_fre, v_fre, None)     # B*c*D*T

        scaled_attention_time = tf.transpose(scaled_attention_time,[0,2,3,1])
        scaled_attention_fre = tf.transpose(scaled_attention_fre,[0,3,2,1])

        out = tf.concat([scaled_attention_time,scaled_attention_fre],-1) # B*T*D*2c

        out = self.ln(self.conv(out) + residual) # B*T*D*n

        final_out = self.bnf1(self.final_conv1(out),training=training)
        final_out = self.bnf2(self.final_conv2(final_out),training=training)

        final_out = self.act(final_out + out)

        return final_out


class Pre_Net(tf.keras.layers.Layer):
    def __init__(self,num_M=2,n=64,c=64):
        super(Pre_Net, self).__init__()
        self.num_M = num_M

        self.downsample = tf.keras.layers.Conv2D(n, 3, 2, 'same', activation='tanh',
                                                 kernel_initializer='glorot_normal')
        self.bn = tf.keras.layers.BatchNormalization()
        self.downsample2 = tf.keras.layers.Conv2D(n, 3, 2, 'same', activation='tanh',
                                                  kernel_initializer='glorot_normal')
        self.bn2 = tf.keras.layers.BatchNormalization()

        self.TwoD_layers = [TwoD_Attention_layer(n, c) for _ in range(num_M)]

    def call(self,inputs,training):
        '''

        :param inputs: B*T*D*n
        :return: B*T*D*c
        '''
        inputs = tf.cast(inputs,tf.float32)
        out = self.bn(self.downsample(inputs),training=training)
        out = self.bn2(self.downsample2(out),training=training)
        print('downsample.shape:',out.shape)

        for i in range(self.num_M):
            out = self.TwoD_layers[i](out,training)

        return out


if __name__=='__main__':
    np.set_printoptions(suppress=True)

    temp_k = tf.constant([[10, 0, 0],
                          [0, 10, 0],
                          [0, 0, 10],
                          [0, 0, 10]], dtype=tf.float32)  # (4, 3)

    temp_v = tf.constant([[1, 0],
                          [10, 0],
                          [100, 5],
                          [1000, 6]], dtype=tf.float32)  # (4, 3)

    # This `query` aligns with the second `key`,
    # so the second `value` is returned.
    temp_q = tf.constant([[0, 10, 0]], dtype=tf.float32)  # (1, 3)
    print_out(temp_q, temp_k, temp_v)

    temp_mha = MultiHeadAttention(d_model=512, num_heads=8)
    y = tf.random.uniform((1, 60, 512))  # (batch_size, encoder_sequence, d_model)
    out, attn = temp_mha(y, k=y, q=y, mask=None)
    print(out.shape, attn.shape)