import torch
import os
from tqdm import tqdm
import torch.nn.functional as F
from model.zerogen.model.models import EntailmentModel, GenerateModel
from torch.utils.data import DataLoader
from model.train_base import Trainbasic
from codes.utils import save_data, MemTracker, download_file
from multiprocessing import cpu_count
import random
from typing import List


cpu_num = cpu_count() - 1 # 自动获取最大核心数目
os.environ['OMP_NUM_THREADS'] = str(cpu_num)
os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
os.environ['MKL_NUM_THREADS'] = str(cpu_num)
os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
torch.set_num_threads(cpu_num)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device is', device)


class OneModelTrainer_ZeroGen(Trainbasic):
    '''
    训练模型
    '''
    def __init__(self, model_config, logger, dataloader_list, label_mat, generated_dataset, emb_layer=None, local=False, **kwargs):
        super().__init__(model_config, logger, emb_layer=emb_layer, local=False, **kwargs)
        # learning params
        self.epochs = model_config.epochs
        self.batch_size = model_config.batch_size
        self.generate_size = model_config.generate_size
        self.lr = model_config.lr
        self.emb_layer = emb_layer
        self.max_epochs = model_config.epochs

        # model config
        self.class_num = model_config.class_num
        self.model_path = model_config.model_name
        self.save_local_path = os.path.join('./data', self.model_path)
        self.oss_path = model_config.save_model_oss_path
        # help tracker
        self.gpu_tracker = MemTracker()

        
        # build a new mode
        self.build_model()
        self.load_dataset(dataloader_list=dataloader_list, generated_dataset=generated_dataset)
        self.get_seen_class_id()
        # build optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=self.lr,
                                          betas=(0.9, 0.999),
                                          eps=1e-8,
                                          weight_decay=0.01)

    def build_model(self):
        '''
        initialize model
        '''
        self.model = EntailmentModel(config=self.model_config,
                                     device=device,
                                     emb_layer=self.emb_layer)
        self.model.to(device)
        self.generator = GenerateModel(config=self.model_config,
                                       device=device,
                                       emb_layer=self.emb_layer)
        self.generator.to(device)
        self.generator.eval()


    def load_dataset(self, dataloader_list, generated_dataset, **kwargs):
        self.train_dataset, self.test_dataset, self.eval_dataset, self.unlabel_dataset, self.label_dataset = dataloader_list
        self.train_loader = DataLoader(dataset=self.train_dataset,
                                       batch_size=self.batch_size,
                                       shuffle=True)
        self.test_loader = DataLoader(dataset=self.test_dataset,
                                      batch_size=self.batch_size,
                                      shuffle=True)
        self.eval_loader = DataLoader(dataset=self.eval_dataset,
                                      batch_size=self.batch_size,
                                      shuffle=True)
        self.unlabel_loader = DataLoader(dataset=self.unlabel_dataset,
                                         batch_size=self.batch_size,
                                         shuffle=True)
        self.label_loader = DataLoader(dataset=self.label_dataset,
                                         batch_size=self.batch_size,
                                         shuffle=True)
        
        self.generated_loader = DataLoader(dataset=generated_dataset,
                                           batch_size=self.generate_size,
                                           shuffle=True)

    def sample_class_by_weight(self, y:int, **kwargs):
        """_summary_

        Args:
            y (_type_): positive sample index
        """
        sampled_weight_list = [i for i in range(self.class_num) if i != y]
        sampled_y = random.sample(sampled_weight_list, 1)[0]
        return sampled_y

    def pseudo_data_generation(self):
        """
        Pseudo Data Generation Step
        """
        self.model.train()
        self.generator.eval()
        length = len(self.generated_loader)
        data_loader = self.generated_loader
        for step, batch in tqdm(enumerate(data_loader), total=length, leave=True):
            # start common batch
            y_idx, y, y_text, label_name = batch
            y_idx = y_idx.to(device)
            y = y.to(device)

            for i in range(y.shape[0]):
                one_y_idx = y_idx[i].unsqueeze(0)
                label_start_text = y_text[i] + self.model_config.add_str
                if isinstance(self.generator, torch.nn.parallel.DistributedDataParallel):
                    result_dict = self.generator.module.sample_all(label_start_text, generate_number=self.batch_size)  # type: ignore
                else:
                    result_dict = self.generator.sample_all(label_start_text, generate_number=self.batch_size)
                sent_list = result_dict['sentences']
                
                # if self.output_generate and self.first_batch:
                if one_y_idx.item() in self.seen_train:
                    seen_type = 'train'
                elif one_y_idx.item() in self.seen_eval:
                    seen_type = 'eval'
                else:
                    seen_type = 'unseen'

                # train the model
                # 从句子中重新生成batch
                negative_sent_list = []
                for sent in sent_list:
                    sent = sent.split(self.model_config.split_str)
                    # if dist.get_rank() == 0:
                    #     print('sent 0 is', sent[0], 'y_text is,', y_text)
                    tmp_y_text = [text for text in y_text if text != sent[0]]
                    sent[0] = random.sample(tmp_y_text, 1)[0]
                    sent = '[SEP]'.join(sent)
                    negative_sent_list.append(sent)
                # 获取positive sample
                sent_list = [sent.replace(self.model_config.split_str, '[SEP]') for sent in sent_list]
                    
                # neg_sent_list = []
                # if dist.get_rank() == 0:
                #     print('sent list', sent_list, 'negative sent list', negative_sent_list, 'y is', one_y_idx)
                
                generate_positive_x, _ = self.emb_data(sent_list)
                generate_negative_x, _ = self.emb_data(negative_sent_list)
                generate_positive_x = torch.tensor(generate_positive_x)
                generate_negative_x = torch.tensor(generate_negative_x)
                generate_y_pos = torch.ones(generate_positive_x.shape[0], 1)
                generate_y_neg = torch.zeros(generate_negative_x.shape[0], 1)
                generate_total_x = torch.cat([generate_positive_x, generate_negative_x], dim=0)
                generate_y = torch.cat([generate_y_pos, generate_y_neg], dim=0).to(device)
                # print('total_x', generate_total_x.shape, 'generate y', generate_y.shape)
                            
                # # get result
                encoder_result_dict = self.model(generate_total_x.detach(),
                                                y=generate_y.detach()
                                                )
                total_loss_1 = encoder_result_dict['loss']
                total_loss_1.backward()

                self.optimizer.step()
                self.optimizer.zero_grad()
                self.first_batch = False
             
    def labeled_train(self):
        """
        Train Bert classifier and Generator with labeled data
        """ 
        self.model.train()
        self.generator.train()

        
        final_targets = []
        final_probabs = []

        # 组合训练数据
        length = len(self.train_loader)
        data_loader = self.train_loader
        self.first_batch = True

        # start epoch
        for step, batch in tqdm(enumerate(data_loader), total=length, leave=True):
            # start common batch
            print(step)
            if step >= 10:
                break
            final_targets, final_probabs, total_loss = self.batch_labeled_train(batch=batch,
                                                                                final_targets=final_targets,
                                                                                final_probabs=final_probabs)
    
            
            if self.model_config.detect:
                with torch.autograd.detect_anomaly():
                    # backward
                    print('total loss is', total_loss)
                    total_loss.backward()
            else:
                total_loss.backward()

            if self.first_batch:
                for name, param in self.model.named_parameters():
                    if param.grad is None:
                        print('classifier', name)

            self.optimizer.step()
            self.optimizer.zero_grad()
            self.first_batch = False

        return final_targets, final_probabs

    def batch_labeled_train(self, batch, final_targets, final_probabs, is_train=True):
        '''
        common training
        '''
        # deal batch data
        result_dict = batch
        text_list = result_dict['origin']['text']
        batch_x = result_dict['origin']['idx']
        y = result_dict['y_data']
        
        # choose positive and negative sample to train classifier
        sampled_x = []
        sampled_y = []
        # store postivie samples for generator train
        positive_sampled_x = []
        positive_sentence_x = []

        assert len(batch_x) == y.shape[0]
        assert batch_x[0].shape[0] == self.class_num

        for i in range(y.shape[0]):
            label_id = y[i].item()
            neg_label_id = self.sample_class_by_weight(label_id)
            positive_sample = batch_x[i][label_id, :]
            negative_sample = batch_x[i][neg_label_id, :]
            positive_sampled_x.append(positive_sample.unsqueeze(0))
            positive_sentence_x.append(text_list[label_id][i].replace('[SEP]', self.model_config.split_str))
            sampled_x.append(positive_sample.unsqueeze(0))
            sampled_x.append(negative_sample.unsqueeze(0))
            sampled_y.append(1.0)
            sampled_y.append(0.0)
        sampled_x = torch.cat(sampled_x, dim=0).to(device)
        sampled_y = torch.tensor(sampled_y).to(device).unsqueeze(1)
        positive_sampled_x = torch.cat(positive_sampled_x, dim=0).to(device)
        # get classifier result
        result_dict = self.model(sampled_x,
                                 y=sampled_y,
                                 )
        total_loss = result_dict['loss']

        return final_targets, final_probabs, total_loss

    def train(self):
        """
        训练Flow
        """
        self.logger.info('Start model train with name: %s' % (self.model_path))
        best_loss = 0
        self.best_epoch = 0
        # output dataset length
        print('output length', len(self.train_loader), len(self.unlabel_loader))

        # start epoch
        for epoch in range(self.max_epochs):
            self.epoch = epoch
            self.logger.info('Start epoch %d' % self.epoch)

            # generator train
            self.pseudo_data_generation()
            
            # if trian in label fully unseen, zhushi this
            # self.labeled_train()
            
            ap_1 = self.eval(name='dev')
            self.seen_and_unseen_test(dataloader={'test':self.test_loader, 'eval':self.eval_loader}, data_type="unseen")
            ap_1 = torch.tensor(ap_1).to(device)

            self.logger.info("evaluation precision reduce result %.4f" % ap_1.item())
            ap_1 = ap_1.item()
            if best_loss < ap_1:
                best_loss = ap_1
                self.best_epoch = epoch
                state_dict = self.model.state_dict()  # type: ignore

                torch.save({'state_dict': state_dict}, self.save_local_path)
                self.logger.info('Save best model')
                if self.local is False:
                    save_data(os.path.join(self.oss_path, self.model_path), self.save_local_path)
        
        self.seen_and_unseen_test(dataloader=self.test_loader, data_type="test")


    def seen_and_unseen_test(self, dataloader, data_type="test"):
        self.logger.info('Start model test with name: %s, epoch %d' % (self.model_path, self.best_epoch))
        if data_type == "test":
            try:
                self.model.load_state_dict(torch.load(self.save_local_path)['state_dict'])
            except RuntimeError:
                try:
                    self.model.module.load_state_dict(torch.load(self.save_local_path)['state_dict'])
                except:
                    self.logger.info('Use origin model because not found any save model state')
        else:
            pass
        
        seen_prob = []
        seen_targets = []
        unseen_prob = []
        unseen_targets = []
        self.logger.info('start test')
        
        if type(dataloader) == dict:
            probabs = []
            targets = []
            for one_data_type in dataloader.keys():
                one_loader = dataloader[one_data_type]
                one_probabs, one_targets, _ = self.evaluate(loader=one_loader, data_type=one_data_type, get_loss=False)
                probabs = probabs + one_probabs
                targets = targets + one_targets
                for i in range(len(targets)):
                    y = targets[i].index(1)
                    if y in self.seen_train:
                        seen_prob.append(probabs[i])
                        seen_targets.append(targets[i])
                    else:
                        unseen_prob.append(probabs[i])
                        unseen_targets.append(targets[i])
        else:
            probabs, targets, _ = self.evaluate(loader=dataloader, get_loss=False, data_type=data_type)
            for i in range(len(targets)):
                y = targets[i].index(1)
                if y in self.seen_train:
                    seen_prob.append(probabs[i])
                    seen_targets.append(targets[i])
                else:
                    unseen_prob.append(probabs[i])
                    unseen_targets.append(targets[i])

        print('seen class number:', len(seen_targets), 'unseen_class number:', len(unseen_targets))
        self.get_metric_result(probabs, targets, name=f'{data_type} epoch {self.epoch} test total')
        self.get_metric_result(seen_prob, seen_targets, name=f'{data_type} epoch {self.epoch} test seen')
        self.get_metric_result(unseen_prob, unseen_targets, name=f'{data_type} epoch {self.epoch} test unseen')
        return probabs, targets

    def calculate_batch_prob(self, batch_x: List):
        if self.class_num * len(batch_x) > 600:
            prob_result = []
            for i in range(len(batch_x)):
                result_dict = self.model(batch_x[i].to(device))
                prob_result.append(result_dict['prob'].squeeze(1).unsqueeze(0).detach().reshape(-1, self.class_num))
            prob = torch.cat(prob_result, dim=0)
            print(prob.shape)
        else:
            total_batch_x = []
            for i in range(len(batch_x)):
                total_batch_x.append(batch_x[i])
            # print('check x and true label shape', x.shape, true_label.shape)
            total_batch_x = torch.cat(total_batch_x, dim=0)
            result_dict = self.model(total_batch_x.to(device))
            prob = result_dict['prob'].squeeze(1).unsqueeze(0).detach()
            prob = prob.reshape(-1, self.class_num)
        return prob

    def prob_mask(self, prob: torch.Tensor, y: torch.Tensor):
        """Deal with prob in semi-supervised training setting

        Args:
            prob (torch.Tensor): _description_
            y (torch.Tensor): _description_
        """
        for i in range(y.shape[0]):
            if y[i].item() in self.seen_train:
                for j in range(prob.shape[1]):
                    if j not in self.seen_train:
                        prob[i, j] = -1e9
            else:
                for j in range(prob.shape[1]):
                    if j in self.seen_train:
                        prob[i, j] = -1e9
        return prob

    def evaluate(self, loader, data_type:str ='eval', get_loss=False):
        final_targets = []
        final_probabs = []
        self.model.eval()
        self.generator.eval()
        eval_loss = 0
        
        with torch.no_grad():
            for step, batch in tqdm(enumerate(loader), total=len(loader), leave=True):
                result_dict = batch
                batch_x = result_dict['origin']['idx']
                batch_data_idx_list = result_dict['index']
                batch_data_idx_list = [i.item() for i in batch_data_idx_list]
                y = result_dict['y_data']
                
                assert len(batch_x) == y.shape[0]
                assert batch_x[0].shape[0] == self.class_num
                
                total_true_label = []
                total_batch_x = []
                for i in range(y.shape[0]):
                    true_label = [0.0 for _ in range(self.class_num)]
                    true_label[y[i].item()] = 1.0
                    total_true_label = total_true_label + true_label
                    total_batch_x.append(batch_x[i])
                # print('check x and true label shape', x.shape, true_label.shape)
                # batch_x = torch.cat(total_batch_x, dim=0)
                prob = self.calculate_batch_prob(total_batch_x)

                prob = self.prob_mask(prob=prob, y=y)
                prob = F.softmax(prob, dim=1)

                # print('output prob', prob, prob.shape, torch.tensor([y[i].item()]), torch.tensor([y[i].item()]).shape)
                origin_final_length = len(final_targets)
                final_targets, final_probabs = self.get_prob_and_target(y, prob, final_targets, final_probabs)

        return final_probabs, final_targets, eval_loss



class OneModelTrainerParallel_ZeroGen(OneModelTrainer_ZeroGen):
    def __init__(self, model_config, logger, dataloader_list, label_mat, generated_dataset, emb_layer=None, local=False):
        super().__init__(model_config, logger, dataloader_list, label_mat, generated_dataset, emb_layer=emb_layer, local=local)

    def build_model(self):
        '''
        initialize model
        '''
        super().build_model()
        initial_usage = torch.cuda.memory_allocated()
        print('initial usage', initial_usage)
        self.model = torch.nn.DataParallel(self.model.to(device))
        self.generator = torch.nn.DataParallel(self.generator.to(device))
        after_init_linear1 = torch.cuda.memory_allocated()
        print('model usage', after_init_linear1)
