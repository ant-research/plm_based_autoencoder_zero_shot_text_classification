import logging


class Logger(object):
    """docstring for AuthorMerger"""
    def __init__(self, log_path='./', log_name='log'):
        super(Logger, self).__init__()
        self.log_path = log_path
        self.log_name = log_name
        self.prepare_stream_logger()
        self.prepare_file_logger()

    def prepare_stream_logger(self):
        self.logger = logging.getLogger(self.log_path + 'stream')
        #设置为INFO级别
        self.logger.setLevel(logging.INFO)
        #formater
        self.log_formater = logging.Formatter(fmt = '%(asctime)s - %(levelname)s - %(message)s',
                                              datefmt = '%Y-%m-%d %H:%M:%S')
        #输出到控制台的handle
        ch = logging.StreamHandler()
        ch.setFormatter(self.log_formater)
        ch.setLevel(logging.INFO)
        self.logger.addHandler(ch)

    def prepare_file_logger(self):
        #写入日志文件的handle
        fh = logging.FileHandler(self.log_path + self.log_name + '.log', encoding='utf-8')
        fh.setLevel(logging.INFO)
        fh.setFormatter(self.log_formater)
        self.logger.addHandler(fh)
        self.logger.info('start log')