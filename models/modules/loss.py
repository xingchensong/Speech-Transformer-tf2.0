import tensorflow as tf

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
LableSmoothing_loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

def label_smoothing(inputs, epsilon=0.1):
    '''Applies label smoothing. See https://arxiv.org/abs/1512.00567.

    Args:
      inputs: A 3d tensor with shape of [N, T, V], where V is the number of vocabulary.
      epsilon: Smoothing rate.

    For example,

    ```
    import tensorflow as tf
    inputs = tf.convert_to_tensor([[[0, 0, 1],
       [0, 1, 0],
       [1, 0, 0]],
      [[1, 0, 0],
       [1, 0, 0],
       [0, 1, 0]]], tf.float32)

    outputs = label_smoothing(inputs)

    with tf.Session() as sess:
        print(sess.run([outputs]))

    >>
    [array([[[ 0.03333334,  0.03333334,  0.93333334],
        [ 0.03333334,  0.93333334,  0.03333334],
        [ 0.93333334,  0.03333334,  0.03333334]],
       [[ 0.93333334,  0.03333334,  0.03333334],
        [ 0.93333334,  0.03333334,  0.03333334],
        [ 0.03333334,  0.93333334,  0.03333334]]], dtype=float32)]
    ```
    '''
    K = inputs.get_shape().as_list()[-1]  # number of channels
    return ((1 - epsilon) * inputs) + (epsilon / K)

def LableSmoothingLoss(real,pred,vocab_size,epsilon):
    """
    pred (FloatTensor): batch_size x vocab_size
    real (LongTensor): batch_size
    """
    real = tf.cast(real,tf.int32)
    # print(real)
    real_smoothed = label_smoothing(tf.one_hot(real,depth=vocab_size),epsilon)
    # print(real_smoothed)
    # mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = LableSmoothing_loss_object(real_smoothed, pred)

    mask = tf.math.logical_not(tf.math.equal(real, 0))
    mask = tf.cast(mask, dtype=loss_.dtype)# 转换为与loss相同的类型

    # Since the target sequences are padded, it is important to apply a padding mask when calculating the loss.
    loss_ *= mask

    return tf.reduce_mean(loss_)

def Loss(real,pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_mean(loss_)

if __name__=='__main__':
    # vocab_size = 3(包括padding) && batch_size = 3
    real = tf.convert_to_tensor([2,1,0], tf.int32)
    pred = tf.convert_to_tensor([[0.1,0.1,0.8],[0.3,0.6,0.1],[0.98,0.01,0.01]], tf.float32)
    print(LableSmoothingLoss(real,pred,3,0.1))
    print(Loss(real,pred))
