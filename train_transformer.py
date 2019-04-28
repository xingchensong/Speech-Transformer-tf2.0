import tensorflow as tf
from models import Transformer,create_masks,LableSmoothingLoss,CustomSchedule
from utils import AttrDict,init_logger,ValueWindow
import yaml,argparse,os,time
from tensorflow.python.ops import summary_ops_v2

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', type=str, default='config/hparams.yaml')
    parser.add_argument('-load_model', type=str, default=None)
    # parser.add_argument('-fp16_allreduce', action='store_true', default=False,
    #                     help='use fp16 compression during allreduce')
    # parser.add_argument('-batches_per_allreduce', type=int, default=1,
    #                     help='number of batches processed locally before '
    #                          'executing allreduce across workers; it multiplies '
    #                          'total batch size.')
    parser.add_argument('-num_wokers', type=int, default=0,
                        help='how many subprocesses to use for data loading. '
                             '0 means that the data will be loaded in the main process')
    parser.add_argument('-log', type=str, default='train.log')
    opt = parser.parse_args()

    configfile = open(opt.config)
    config = AttrDict(yaml.load(configfile,Loader=yaml.FullLoader))

    log_name = config.model.name
    log_folder = os.path.join(os.getcwd(),'logging',log_name)
    if not os.path.isdir(log_folder):
        os.mkdir(log_folder)
    logger = init_logger(log_folder+'/'+opt.log)

    # TODO: build dataloader
    train_dataset = None

    # TODO: build model or load pre-trained model
    global global_step
    global_step = 0
    learning_rate = CustomSchedule(config.model.d_model)
    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=config.optimizer.beta1, beta_2=config.optimizer.beta2,
                                         epsilon=config.optimizer.epsilon)
    model = Transformer(config=config)

    #Create the checkpoint path and the checkpoint manager. This will be used to save checkpoints every n epochs.
    checkpoint_path = log_folder
    ckpt = tf.train.Checkpoint(transformer=model, optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

    # if a checkpoint exists, restore the latest checkpoint.
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        logger.info('Latest checkpoint restored!!')
    else:
        logger.info('Start new run')

    model.summary()

    # define metrics and summary writer
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    # summary_writer = tf.keras.callbacks.TensorBoard(log_dir=log_folder)
    summary_writer = summary_ops_v2.create_file_writer_v2(log_folder+'/train')


    @tf.function
    def train_step(inp, tar):
        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)
        with tf.GradientTape() as tape:
            predictions, _ = model((inp, tar_inp), True, enc_padding_mask,
                                         combined_mask, dec_padding_mask)
            loss = LableSmoothingLoss(tar_real, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        train_loss(loss)
        train_accuracy(tar_real, predictions)

    time_window = ValueWindow(100)
    loss_window = ValueWindow(100)
    acc_window = ValueWindow(100)
    for epoch in config.training.epochs:

        start_time = time.time()
        train_loss.reset_states()
        train_accuracy.reset_states()

        for (batch, (inp, tar)) in enumerate(train_dataset):
            step_time = time.time()
            train_step(inp, tar)
            time_window.append(time.time()-step_time)
            loss_window.append(train_loss.result())
            acc_window.append(train_accuracy.result())
            message = 'Step %-7d [%.03f sec/step, loss=%.05f, avg_loss=%.05f, acc=%.05f, avg_acc=%.05f]' % (
                    global_step, time_window.average, train_loss.result(), loss_window.average, train_accuracy.result(),acc_window.average)
            logger.infor(message)

            if global_step % 10 == 0:
                with summary_ops_v2.always_record_summaries():
                    with summary_writer.as_default():
                        summary_ops_v2.scalar('train_loss', train_loss.result(), step=global_step)
                        summary_ops_v2.scalar('train_acc', train_accuracy.result(), step=global_step)

            global_step += 1

        ckpt_save_path = ckpt_manager.save()
        logger.info('Saving checkpoint for epoch {} at {}'.format(epoch+1, ckpt_save_path))
        logger.info('Time taken for 1 epoch: {} secs\n'.format(time.time() - start_time))
        # TODO: eval

if __name__=='__main__':
    main()