from pydub import AudioSegment
from os.path import join, basename
import glob
from queue import Queue
import logging
import os
from threading import Thread
import audiotools
from audiotools.wav import InvalidWave

"""
Flac 2 Wav converter script
using audiotools
From http://magento4newbies.blogspot.com/2014/11/converting-wav-files-to-flac-with.html
"""
class F2W:

    logger = ''

    def __init__(self):
        global logger
        # create logger
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)

        # create a file handler
        handler = logging.FileHandler('converter.log')
        handler.setLevel(logging.INFO)

        # create a logging format
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)

        # add the handlers to the logger
        logger.addHandler(handler)

    def convert(self):
        global logger
        file_queue = Queue()
        num_converter_threads = 5

        # collect files to be converted
        data_path = 'E:\corpus_en\LibriSpeech_small'
        parts = ['dev-clean']

        for part in parts:
            # part/speaker/book/*.wav
            flac_paths = [p for p in glob(join(data_path, part, '*/*/*.flac'))]
            for flac_path in flac_paths:
                file_queue.put(flac_path)

        # for root, dirs, files in os.walk("/Volumes/music"):
        #
        #     for file in files:
        #         if file.endswith(".wav"):
        #             file_wav = os.path.join(root, file)
        #             file_flac = file_wav.replace(".wav", ".flac")
        #
        #             if (os.path.exists(file_flac)):
        #                 logger.debug(''.join(["File ",file_flac, " already exists."]))
        #             else:
        #                 file_queue.put(file_wav)

        logger.info("Start converting:  %s files", str(file_queue.qsize()))

        # Set up some threads to convert files
        for i in range(num_converter_threads):
            worker = Thread(target=self.process, args=(file_queue,))
            worker.setDaemon(True)
            worker.start()

        file_queue.join()

    def process(self, q):
        """This is the worker thread function.
        It processes files in the queue one after
        another.  These daemon threads go into an
        infinite loop, and only exit when
        the main thread ends.
        """
        while True:
            global logger
            compression_quality = '0' #min compression
            file_flac = q.get()
            file_wav = file_flac.replace(".flac", ".wav")

            try:
                audiotools.open(file_flac).convert(file_wav,audiotools.WavAudio, compression_quality)
                logger.info(''.join(["Converted ", file_flac, " to: ", file_wav]))
                os.remove(file_flac)
                q.task_done()
            except InvalidWave:
                logger.error(''.join(["Failed to open file ", file_flac, " to: ", file_wav," failed."]), exc_info=True)
            except Exception as e:
                logger.error('ExFailed to open file', exc_info=True)