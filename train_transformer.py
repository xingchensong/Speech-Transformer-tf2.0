import tensorflow as tf
from models import Speech_transformer,Transformer,create_masks,LableSmoothingLoss,CustomSchedule,create_combined_mask
from utils import AttrDict,init_logger,ValueWindow
import yaml,argparse,os,time
from tensorflow.python.ops import summary_ops_v2
from datasets.datafeeder import DataFeeder

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', type=str, default='config/hparams.yaml')
    parser.add_argument('-load_model', type=str, default=None)
    parser.add_argument('-model_name', type=str, default='P_S_Transformer_debug',
                        help='model name')
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

    log_name = opt.model_name or config.model.name
    log_folder = os.path.join(os.getcwd(),'logdir/logging',log_name)
    if not os.path.isdir(log_folder):
        os.mkdir(log_folder)
    logger = init_logger(log_folder+'/'+opt.log)

    # TODO: build dataloader
    train_datafeeder = DataFeeder(config,'debug')

    # TODO: build model or load pre-trained model
    global global_step
    global_step = 0
    learning_rate = CustomSchedule(config.model.d_model)
    # learning_rate = 0.00002
    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=config.optimizer.beta1, beta_2=config.optimizer.beta2,
                                         epsilon=config.optimizer.epsilon)
    logger.info('config.optimizer.beta1:' + str(config.optimizer.beta1))
    logger.info('config.optimizer.beta2:' + str(config.optimizer.beta2))
    logger.info('config.optimizer.epsilon:' + str(config.optimizer.epsilon))
    # print(str(config))
    model = Speech_transformer(config=config,logger=logger)

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


    # define metrics and summary writer
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    # summary_writer = tf.keras.callbacks.TensorBoard(log_dir=log_folder)
    summary_writer = summary_ops_v2.create_file_writer_v2(log_folder+'/train')


    # @tf.function
    def train_step(batch_data):
        inp = batch_data['the_inputs'] # batch*time*feature
        tar = batch_data['the_labels'] # batch*time
        # inp_len = batch_data['input_length']
        # tar_len = batch_data['label_length']
        gtruth = batch_data['ground_truth']
        tar_inp = tar
        tar_real = gtruth
        # enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp[:,:,0], tar_inp)
        combined_mask = create_combined_mask(tar=tar_inp)
        with tf.GradientTape() as tape:
            predictions, _ = model(inp, tar_inp, True, None,
                                   combined_mask, None)
            # logger.info('config.train.label_smoothing_epsilon:' + str(config.train.label_smoothing_epsilon))
            loss = LableSmoothingLoss(tar_real, predictions,config.model.vocab_size,config.train.label_smoothing_epsilon)
        gradients = tape.gradient(loss, model.trainable_variables)
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        train_loss(loss)
        train_accuracy(tar_real, predictions)

    time_window = ValueWindow(100)
    loss_window = ValueWindow(100)
    acc_window = ValueWindow(100)
    logger.info('config.train.epoches:' + str(config.train.epoches))
    first_time = True
    for epoch in range(config.train.epoches):
        logger.info('start epoch '+ str(epoch))
        logger.info('total wavs: '+ str(len(train_datafeeder)))
        logger.info('batch size: ' + str(train_datafeeder.batch_size))
        logger.info('batch per epoch: ' + str(len(train_datafeeder)//train_datafeeder.batch_size))
        train_data = train_datafeeder.get_batch()
        start_time = time.time()
        train_loss.reset_states()
        train_accuracy.reset_states()

        for step in range(len(train_datafeeder)//train_datafeeder.batch_size):
            batch_data = next(train_data)
            step_time = time.time()
            train_step(batch_data)
            if first_time:
                model.summary()
                first_time=False
            time_window.append(time.time()-step_time)
            loss_window.append(train_loss.result())
            acc_window.append(train_accuracy.result())
            message = 'Step %-7d [%.03f sec/step, loss=%.05f, avg_loss=%.05f, acc=%.05f, avg_acc=%.05f]' % (
                    global_step, time_window.average, train_loss.result(), loss_window.average, train_accuracy.result(),acc_window.average)
            logger.info(message)

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