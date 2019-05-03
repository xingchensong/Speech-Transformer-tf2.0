import tensorflow as tf
import numpy as np
from modules.encoder import Encoder
from modules.decoder import Decoder

class Transformer(tf.keras.Model):
    def __init__(self, num_layers=4, d_model=512, num_heads=8, dff=2048, pe_max_len=8000,
                 target_vocab_size=8000, rate=0.1,config=None,logger=None):
        super(Transformer, self).__init__()

        if config is not None:
            num_enc_layers = config.model.N_encoder
            if logger is not None:
                logger.info('config.model.N_encoder: '+str(num_enc_layers))
            num_dec_layers = config.model.N_decoder
            if logger is not None:
                logger.info('config.model.N_decoder: '+str(num_dec_layers))
            d_model = config.model.d_model
            if logger is not None:
                logger.info('config.model.d_model:   '+str(d_model))
            num_heads = config.model.n_heads
            if logger is not None:
                logger.info('config.model.n_heads:   '+str(num_heads))
            dff = config.model.d_ff
            if logger is not None:
                logger.info('config.model.d_ff:      '+str(dff))
            pe_max_len = config.model.pe_max_len
            if logger is not None:
                logger.info('config.model.pe_max_len:'+str(pe_max_len))
            target_vocab_size = config.model.vocab_size
            if logger is not None:
                logger.info('config.model.vocab_size:'+str(target_vocab_size))
            rate = config.model.dropout
            if logger is not None:
                logger.info('config.model.dropout:   '+str(rate))
        else:
            print('use default params')
            num_enc_layers = num_layers
            num_dec_layers = num_layers

        self.encoder = Encoder(num_enc_layers, d_model, num_heads, dff,
                                   pe_max_len,'encoder', rate)

        self.decoder = Decoder(num_dec_layers, d_model, num_heads, dff,
                               target_vocab_size, 'decoder',pe_max_len,rate)

        # self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    def call(self, inputs , training, enc_padding_mask,
             look_ahead_mask, dec_padding_mask):

        inp = tf.cast(inputs[0],tf.float32)
        tar = tf.cast(inputs[1],tf.int32)

        enc_output = self.encoder((inp, enc_padding_mask),training)  # (batch_size, inp_seq_len, d_model)

        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        dec_output, attention_weights = self.decoder(
            (tar, enc_output,  look_ahead_mask, dec_padding_mask),training)

        # final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)
        final_output = dec_output

        return final_output, attention_weights



if __name__=='__main__':
    sample_transformer = Transformer(
        num_layers=2, d_model=512, num_heads=4, dff=2048,
        pe_max_len=8500, target_vocab_size=32)

    temp_input = tf.random.uniform((64, 62))
    temp_target = tf.random.uniform((64, 26))
    # temp_input = tf.keras.layers.Input((64,62),dtype=tf.float32)
    # temp_target = tf.keras.layers.Input((16,),dtype=tf.float32)
    # 如果想对inputs浓缩，那么几个mask也要建立Input？
    fn_out, _ = sample_transformer(inputs=(temp_input, temp_target), training=False,
                                   enc_padding_mask=None,
                                   look_ahead_mask=None,
                                   dec_padding_mask=None)

    sample_transformer.summary()# 如果在调用call之前进行summary会提示model not build

    '''
    Model: "transformer"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    encoder (Encoder)            multiple                  10656768  
    _________________________________________________________________
    decoder (Decoder)            multiple                  12504064  
    _________________________________________________________________
    dense_32 (Dense)             multiple                  4104000   
    =================================================================
    Total params: 27,264,832
    Trainable params: 27,264,832
    Non-trainable params: 0
    _________________________________________________________________

    '''
    # tf.keras.utils.plot_model(sample_transformer)
    print(sample_transformer.get_layer('encoder'))
    tp = sample_transformer.trainable_variables
    for i in range(20):
        print(tp[i].name)
    '''
    <modules.encoder.Encoder object at 0x00000151AD449390>
    transformer/encoder/enc_embedding/embeddings:0
    transformer/encoder/encoder_layer/multi_head_attention/dense/kernel:0
    transformer/encoder/encoder_layer/multi_head_attention/dense/bias:0
    transformer/encoder/encoder_layer/multi_head_attention/dense_1/kernel:0
    transformer/encoder/encoder_layer/multi_head_attention/dense_1/bias:0
    transformer/encoder/encoder_layer/multi_head_attention/dense_2/kernel:0
    transformer/encoder/encoder_layer/multi_head_attention/dense_2/bias:0
    transformer/encoder/encoder_layer/multi_head_attention/dense_3/kernel:0
    transformer/encoder/encoder_layer/multi_head_attention/dense_3/bias:0
    transformer/encoder/encoder_layer/sequential/dense_4/kernel:0
    transformer/encoder/encoder_layer/sequential/dense_4/bias:0
    transformer/encoder/encoder_layer/sequential/dense_5/kernel:0
    transformer/encoder/encoder_layer/sequential/dense_5/bias:0
    transformer/encoder/encoder_layer/layer_normalization/gamma:0
    transformer/encoder/encoder_layer/layer_normalization/beta:0
    transformer/encoder/encoder_layer/layer_normalization_1/gamma:0
    transformer/encoder/encoder_layer/layer_normalization_1/beta:0
    transformer/encoder/encoder_layer_1/multi_head_attention_1/dense_6/kernel:0
    transformer/encoder/encoder_layer_1/multi_head_attention_1/dense_6/bias:0
    transformer/encoder/encoder_layer_1/multi_head_attention_1/dense_7/kernel:0
    '''

    # model = tf.keras.models.Model(inputs=[temp_input,temp_target],outputs=[fn_out])
    # model.summary()
    print(fn_out.shape)  # (batch_size, tar_seq_len, target_vocab_size)

    # summary_writer = tf.keras.callbacks.TensorBoard(log_dir='modules')
    # summary_writer.set_model(model)