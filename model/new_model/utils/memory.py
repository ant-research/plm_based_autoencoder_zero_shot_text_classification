import random
from model.new_model.utils.dataloader import Fake_Dataset
import torch.utils.data.distributed as data_dist
from torch.utils.data import DataLoader


class DisperseMemory():
    """
    Disperse Memory, for each class, save the occured disperse number
    """
    def __init__(self, class_num, max_len=100, disperse_max_len=32):
        self.memory = [[] for i in range(class_num)]
        self.max_len = max_len
        self.class_num = class_num
        self.disperse_max_len = disperse_max_len
        self.init_memory()
    
    def init_memory(self):
        self.memory_count = {}
        for i in range(self.class_num):
            self.memory_count[i] = {}
            for j in range(self.disperse_max_len):
                self.memory_count[i][j] = 0
    
    def update(self, indices, labels):
        disperse_indices = indices.cpu().numpy().tolist()
        print(disperse_indices)
        for i in range(len(disperse_indices)):
            self.memory[labels[i]].append(disperse_indices[i][0])
            self.memory_count[labels[i]][disperse_indices[i][0]] += 1
        
        for i in range(len(self.memory)):
            if len(self.memory[i]) > self.max_len:
                self.memory[i] = self.memory[i][-self.max_len:]
    
    def save_memory(self, epoch):
        with open('./memory2.txt', 'a+') as f:
            for i in self.memory_count.keys():
                for j in self.memory_count[i].keys():
                    f.write(f'epoch {epoch}, {i}, {j}, {self.memory_count[i][j]}')
                    f.write('\n\r')    
    
    def sample(self, index, sample_number):
        if len(self.memory[index]) == 0:
            result = random.choices([i for i in range(self.disperse_max_len)], k=sample_number)
        else:
            result = random.choices(self.memory[index], k=sample_number)
        return result
    
    def sample_negative(self, index, sample_number=None):
        if sample_number is None:
            sample_number = self.disperse_max_len
        if len(self.memory[index]) == 0:
            result = random.choices([i for i in range(self.disperse_max_len)], k=sample_number)
            final_result = []
            for j in result:
                p = random.choices([i for i in range(self.disperse_max_len) if i!=j], k=1)[0]
                final_result.append([j, p])
        else:
            result = random.choices(self.memory[index], k=sample_number)
            final_result = []
            for j in result:
                sample_list = [i for i in range(self.disperse_max_len) if i not in self.memory[index]]
                if len(sample_list) == 0:
                    sample_list = [i for i in range(self.disperse_max_len)]
                p = random.choices(sample_list, k=1)[0]
                final_result.append([j, p])
        return final_result

    def sample_positive(self, index, sample_number=None):
        if sample_number is None:
            sample_number = self.disperse_max_len
        if len(self.memory[index]) == 0:
            result = random.choices([i for i in range(self.disperse_max_len)], k=sample_number)
            final_result = []
            for j in result:
                p = random.choices([i for i in range(self.disperse_max_len) if i!=j], k=1)[0]
                final_result.append([j, p])
        else:
            result = random.choices(self.memory[index], k=sample_number)
            final_result = []
            for j in result:
                p = random.choices([i for i in range(self.disperse_max_len) if i in self.memory[index]], k=1)[0]
                final_result.append([j, p])
        return final_result
    

class DatasetMemory(object):
    def __init__(self, y_text_list, model_config, emb_func, batch_size, parallel=True):
        self.x_idx = []
        self.x_text = []
        self.y_data = []
        self.y_text_list = y_text_list
        self.model_config = model_config
        self.emb_func = emb_func
        self.parallel = parallel
        self.batch_size = batch_size
        
        self.update_dataset()
    
    def update_data(self, x_text, x_idx, y_data):
        self.x_idx = self.x_idx + x_idx
        self.x_text = self.x_text + x_text
        self.y_data = self.y_data + y_data
        
    def update_one_sentence(self, text, y_data):
        sent = text.split(self.model_config.split_str)
        new_sent_list = ['[SEP]'.join([i, sent[1]]) for i in self.y_text_list]
        idx_list, _ = self.emb_func(new_sent_list)
        
        # print(new_sent_list, idx_list, y_data)

        self.x_idx.append(idx_list)
        self.x_text.append(new_sent_list)
        self.y_data.append(y_data)
        
    def clear_data(self):
        self.x_idx = []
        self.x_text = []
        self.y_data = []
        self.update_dataset()
        
    def update_dataset(self):
        self.dataset = Fake_Dataset(self.x_text,
                                    self.x_idx,
                                    self.y_data)
        print('build fake dataset')

            
    def get_dataloader(self, epoch=0):
        if len(self.dataset) == 0:
            return None
        if self.parallel is True:
            # get sampler
            sampler = data_dist.DistributedSampler(self.dataset)
            
            self.dataloader = DataLoader(dataset=self.dataset,
                                        batch_size=self.batch_size,
                                        sampler=sampler)
            
            self.dataloader.sampler.set_epoch(epoch)
        else:
            self.dataloader = DataLoader(dataset=self.dataset,
                                batch_size=self.batch_size,
                                shuffle=True)
            
        return self.dataloader
