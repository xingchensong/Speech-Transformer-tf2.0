import tensorflow as tf
import numpy as np


PAD = -1
#####################################################

#####################################################
def create_padding_mask(seq):
    '''

    :param seq: [batch_size * seq_len_k] # k means key in MultiheadAttention
    :return: [batch_size, 1, 1, seq_len_k]
    '''
    if seq.dtype != np.int32:
        print("float")
        seq = tf.cast(tf.math.equal(seq, 0.), tf.float32)
    else:
        print("int")
        seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

    # add extra dimensions so that we can add the padding
    # to the attention logits.
    return seq[:, tf.newaxis,tf.newaxis, :]  # (batch_size, 1,1, seq_len)

#####################################################

#####################################################
def create_look_ahead_mask(size):
    '''

    :param size: == seq_len_k
    :return: (seq_len_q, seq_len_k) 只用在decoderblock1，此时qkv的len全相同，因为block1是对taget的encode
    '''
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)

#####################################################

#####################################################
def create_masks(inp, tar):
    '''

    :param inp: [batch_size * seq_len_k_of_encoder ]
    :param tar: [batch_size * seq_len_q_of_decoder_block1 ]
    :return:
    '''
    # Encoder padding mask
    enc_padding_mask = create_padding_mask(inp)

    # Used in the 2nd attention block in the decoder.
    # This padding mask is used to mask the encoder outputs.
    # encoder outputs [batch_size * seq_len * d_model] 中间那一维相比原始encoder的input不变，所以就按照inp计算了
    dec_padding_mask = create_padding_mask(inp)

    # Used in the 1st attention block in the decoder.
    # It is used to pad and mask future tokens in the input received by the decoder.
    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask = create_padding_mask(tar)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

    # print('enc_padding_mask',enc_padding_mask)
    # print('combined_mask', combined_mask)
    # print('dec_padding_mask', dec_padding_mask)

    return enc_padding_mask, combined_mask, dec_padding_mask

def create_DecBlock1_pad_mask(tar):
    tar = tf.cast(tf.math.equal(tar, PAD), tf.float32)
    return tar[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1,1, seq_len)

def create_combined_mask(tar):
    # Used in the 1st attention block in the decoder.
    # It is used to pad and mask future tokens in the input received by the decoder.
    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask = create_DecBlock1_pad_mask(tar)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

    return combined_mask

if __name__=='__main__':
    # x = np.array([[[7, 6, 1, 1, 1], [1, 2, 3, 1, 1], [1, 1, 1, 1, 1]],
    #                  [[7., 6, 6, 1, 1], [0.0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]])
    #
    # length = [3,1]
    # length2 = [3,3]
    # print(create_padding_mask(x[:,:,0]))
    #
    # x2 = np.random.randn(2,3,5)
    # print(create_padding_mask(x2[:,:,0]))
    # temp = create_look_ahead_mask(x2.shape[1])
    # print(temp)
    # temp = create_masks(x2[:,:,0],x[:,:,0])
    # print(temp)

    x = tf.cast(tf.constant([[[7, 6, 1, 1, 1], [1, 2, 3, 1, 1], [1, 1, 1, 1, 1]],
                     [[7., 6, 6, 1, 1], [0.0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]]),tf.float32)
    y = tf.cast(tf.constant([[2,3,4],[2,0,0]]),tf.float32)
    print('\n',create_masks(x[:,:,0],y))