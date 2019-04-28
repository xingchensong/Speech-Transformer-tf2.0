import tensorflow as tf
import numpy as np

#####################################################
# NOTE:
# 这个mask是为了遮住att输出(N,seq_q,seq_k)中
# 被padding的部分(seq_k对应的那一轴,k是key,也就是被查询的句子)
#####################################################
def create_padding_mask(seq,seq_lengths):
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

    seq_shape =  seq.shape.as_list()
    padding_mask = np.zeros(seq_shape[:-1],dtype=seq.dtype.as_numpy_dtype) # batch_size * seq_len

    for i in range(seq_shape[0]):
        padding_mask[i,seq_lengths[i]:] = 1 # eager mode doesnt support item assignment,use numpy instead

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
def create_masks(inp, tar, inp_len, tar_len):
    '''

    :param inp: [batch_size * seq_len * feature_dim]
    :param tar: [batch_size * seq_len * feature_dim]
    :param inp_len: [batch_size]
    :param tar_len: [batch_size]
    :return:
    '''
    # Encoder padding mask
    enc_padding_mask = create_padding_mask(inp,inp_len)

    # Used in the 2nd attention block in the decoder.
    # This padding mask is used to mask the encoder outputs.
    # encoder outputs [batch_size * seq_len * d_model] 中间那一维相比原始encoder的input不变，所以就按照inp计算了
    dec_padding_mask = create_padding_mask(inp,inp_len)

    # Used in the 1st attention block in the decoder.
    # It is used to pad and mask future tokens in the input received by the decoder.
    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask = create_padding_mask(tar,tar_len)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

    return enc_padding_mask, combined_mask, dec_padding_mask

if __name__=='__main__':
    x = tf.constant([[[7, 6, 1, 1, 1], [1, 2, 3, 1, 1], [1, 1, 1, 1, 1]],
                     [[7., 6, 6, 1, 1], [0.0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]])

    print(create_padding_mask(x,[3,1]))
    x2 = tf.random.uniform((2,3, 5))
    print(create_padding_mask(x2,[3,3]))
    temp = create_look_ahead_mask(x2.shape[1])
    print(temp)
    temp = create_masks(x2,x,[3,3],[3,1])
    print(temp)