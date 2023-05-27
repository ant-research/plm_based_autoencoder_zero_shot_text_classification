import os


class Config:
    def __init__(self):
        self.parent_path = os.path.abspath(os.path.dirname(os.getcwd()))
        self.path = os.getcwd()
        
config = Config()
