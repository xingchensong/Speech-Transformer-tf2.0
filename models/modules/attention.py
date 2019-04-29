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

        self.wq = tf.keras.layers.Dense(d_model) # (feature_in_dim, d_model) 第一维不是batchsize因为可以broadcast
        self.wk = tf.keras.layers.Dense(d_model) # (feature_in_dim, d_model)
        self.wv = tf.keras.layers.Dense(d_model) # (feature_in_dim, d_model)

        self.dense = tf.keras.layers.Dense(d_model)# (feature_in_dim, d_model)

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

        return output, attention_weights


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