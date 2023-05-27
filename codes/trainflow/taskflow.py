# -*- coding: utf-8 -*-
'''
@Author: guokaihao.gkh
@Describe: 任务类
'''


class TaskFlow(object):
    def __init__(self):
        pass

    def requirement_install(self):
        raise NotImplementedError

    def preprocess_data(self):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError

    def test(self):
        pass

    def run(self, train=True, test=True, download=True):
        if download:
            self.requirement_install()
        self.preprocess_data()
        if train:
            self.train()
        else:
            pass
        if test:
            self.test()
        else:
            pass
