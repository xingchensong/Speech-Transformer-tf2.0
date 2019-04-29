import tensorflow as tf
import numpy as np



#####################################################
# NOTE:
# 这个mask是为了遮住att输出(N,seq_q,seq_k)中
# 被padding的部分(seq_k对应的那一轴,k是key,也就是被查询的句子)
#####################################################
def create_padding_mask(seq):
    '''

    :param seq: [batch_size * seq_len_k] # k means key in MultiheadAttention
    :return: [batch_size, 1, seq_len_k]
    '''
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

    # add extra dimensions so that we can add the padding
    # to the attention logits.
    return seq[:, tf.newaxis, :]  # (batch_size, 1, seq_len)

def create_padding_mask2(seq,seq_lengths):
    '''
    padding position is set to 1.
    padding_mask can be broadcast on attention_logits (batch_size * seq_q* seq_k)
    :param seq: [batch_size * seq_len * feature_dim]
    :param seq_lengths: [batch_size]   (== seq_k in MultiheadAttention)
    :return: padding_mask: batch_size * 1 * seq_len
    '''
    # seq = tf.math.equal(seq[:,:,0],0)
    # seq = tf.math.equal(seq,False)
    # seq = tf.cast(seq, tf.float32)

    seq_lengths = tf.squeeze(seq_lengths).numpy()
    # print('seq_lengths shape: ' + str(seq_lengths.shape.as_list()))
    seq_shape =  seq.shape.as_list()
    padding_mask = np.zeros(seq_shape[:-1],dtype=seq.dtype.as_numpy_dtype) # batch_size * seq_len

    for i in range(seq_shape[0]):
        padding_mask[i,int(seq_lengths[i]):] = 1 # eager mode doesnt support item assignment,use numpy instead

    # add extra dimensions so that we can add the padding to the attention logits.
    return tf.convert_to_tensor(padding_mask[:,np.newaxis,:])





#####################################################
# NOTE:
# 这个mask是为了遮住att输出(N,seq_q,seq_k)中
# 当前时间步i后面的部分(seq_k对应的那一轴,k是key,也就是被查询的句子)
#####################################################
def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len) AKA (seq_q, seq_k)





#####################################################
# NOTE:
#     encoder与decoder第二个block的self-att只需要考虑遮盖被pad的部分
#     decoder第一个block的self-att需要考虑遮盖被pad的部分以及遮盖未来时间步的信息
#####################################################
def create_masks(inp, tar):
    '''

    :param inp: [batch_size * seq_len_k_of_encoder ]
    :param tar: [batch_size * seq_len_k_of_decoder_block2 ]
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

    return enc_padding_mask, combined_mask, dec_padding_mask


def create_masks2(inp, tar, inp_len, tar_len):
    '''

    :param inp: [batch_size * seq_len * feature_dim]
    :param tar: [batch_size * seq_len * feature_dim]
    :param inp_len: [batch_size]
    :param tar_len: [batch_size]
    :return:
    '''
    # Encoder padding mask
    enc_padding_mask = create_padding_mask2(inp,inp_len)

    # Used in the 2nd attention block in the decoder.
    # This padding mask is used to mask the encoder outputs.
    # encoder outputs [batch_size * seq_len * d_model] 中间那一维相比原始encoder的input不变，所以就按照inp计算了
    dec_padding_mask = create_padding_mask2(inp,inp_len)

    # Used in the 1st attention block in the decoder.
    # It is used to pad and mask future tokens in the input received by the decoder.
    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask = create_padding_mask2(tar,tar_len)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

    return enc_padding_mask, combined_mask, dec_padding_mask

if __name__=='__main__':
    x = np.array([[[7, 6, 1, 1, 1], [1, 2, 3, 1, 1], [1, 1, 1, 1, 1]],
                     [[7., 6, 6, 1, 1], [0.0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]])

    length = [3,1]
    length2 = [3,3]
    print(create_padding_mask(x[:,:,0]))

    x2 = np.random.randn(2,3,5)
    print(create_padding_mask(x2[:,:,0]))
    temp = create_look_ahead_mask(x2.shape[1])
    print(temp)
    temp = create_masks(x2[:,:,0],x[:,:,0])
    print(temp)

    x = tf.constant([[[7, 6, 1, 1, 1], [1, 2, 3, 1, 1], [1, 1, 1, 1, 1]],
                  [[7., 6, 6, 1, 1], [0.0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]])
    length = tf.constant([3,1])
    length2 = tf.constant([3,3])
    print(create_padding_mask2(x,length))
    x2 = tf.random.uniform((2,3, 5))
    print(create_padding_mask2(x2,length2))
    temp = create_look_ahead_mask(x2.shape[1])
    print(temp)
    temp = create_masks2(x2, x, length2,length)
    print(temp)