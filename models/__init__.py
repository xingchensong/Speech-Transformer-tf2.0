from modules import *
from transformer import Transformer
from model import Speech_transformer

if __name__=='__main__':
    sample_encoder = Encoder(num_layers=2, d_model=512, num_heads=8, dff=2048, input_vocab_size=8500)
    sample_encoder_output = sample_encoder(tf.random.uniform((64, 62)), training=False, mask=None)
    print(sample_encoder_output.shape)  # (batch_size, input_seq_len, d_model)