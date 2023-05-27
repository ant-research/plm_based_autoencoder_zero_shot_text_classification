import torch
import os
from tqdm import tqdm
import transformers
import torch.nn.functional as F
from model.gpt2model.model.model import GPTClassifierModel
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.utils.data import DataLoader
from model.train_base import Trainbasic
import torch.distributed as dist
import torch.utils.data.distributed as data_dist
from codes.utils import save_data, MemTracker, download_file
from multiprocessing import cpu_count

cpu_num = cpu_count() - 1 # 自动获取最大核心数目
os.environ['OMP_NUM_THREADS'] = str(cpu_num)
os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
os.environ['MKL_NUM_THREADS'] = str(cpu_num)
os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
torch.set_num_threads(cpu_num)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device is', device)


class OneModelTrainer_GPT2(Trainbasic):
    '''
    训练模型
    '''
    def __init__(self, model_config, logger, dataloader_list, label_mat, generated_dataset, emb_layer=None, local=False, **kwargs):
        super().__init__(model_config, logger, emb_layer=emb_layer, local=False, **kwargs)
        # learning params
        self.model_config = model_config
        self.epochs = model_config.epochs
        self.batch_size = model_config.batch_size
        self.lr = model_config.lr
        # model config
        self.class_num = model_config.class_num
        self.model_path = model_config.model_name
        self.save_local_path = os.path.join('./data', self.model_path)
        self.oss_path = model_config.save_model_oss_path
        

        # deal with label matrix
        if label_mat.shape[1] == 1:
            label_mat = label_mat.squeeze(1)
        self.label_mat = label_mat

        # model config
        self.class_num = model_config.class_num
        self.model_path = model_config.model_name
        # help tracker
        self.gpu_tracker = MemTracker()
        # load dataset
        self.load_dataset(dataloader_list, generated_dataset)
        # 分seen和unseen clas
        self.get_seen_class_id()
        
        # build a new model
        self.build_model()
        # build optimizer
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),
                                          lr=self.lr)
        # schduler
        if self.model_config.use_scheduler:
            steps = 10
            # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, steps)
            self.scheduler = transformers.get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=200, 
                                                                          num_training_steps=200000)

    def build_model(self):
        '''
        initialize model
        '''
        self.gpu_tracker.track()
        self.model = GPTClassifierModel(model_config=self.model_config, device=device, label_matrix=self.label_mat)
        self.model.to(device)
        print('after model')
        self.gpu_tracker.track()

    def load_dataset(self, dataloader_list, generated_dataset, **kwargs):
        self.train_loader, self.test_loader, self.eval_loader, self.unlabel_dataset, self.label_dataset = dataloader_list
        self.train_loader = DataLoader(dataset=self.train_loader,
                                       batch_size=self.batch_size,
                                       shuffle=True)
        self.test_loader = DataLoader(dataset=self.test_loader,
                                      batch_size=self.batch_size,
                                      shuffle=True)
        self.eval_loader = DataLoader(dataset=self.eval_loader,
                                      batch_size=self.batch_size,
                                      shuffle=True)
        self.unlabel_loader = DataLoader(dataset=self.unlabel_dataset,
                                         batch_size=self.batch_size,
                                         shuffle=True)
        self.label_loader = DataLoader(dataset=self.label_dataset,
                                         batch_size=self.batch_size,
                                         shuffle=True)

    def train(self):
        """
        训练传入的模型
        """
        self.logger.info('Start model train with name: %s' % (self.model_path))
        best_loss = 0
        # output dataset length
        print('output length', len(self.train_loader), len(self.unlabel_loader))

        # start epoch
        for epoch in range(self.epochs):
            self.epoch = epoch
            self.logger.info('Start epoch %d' % self.epoch)

            # generator train
            gen_loss_dict, final_targets, final_probabs = self.generate_train()
            
            ap_1 = self.eval(name='dev')
            ap_1 = torch.tensor(ap_1).to(device)
            dist.all_reduce(ap_1.div_(torch.cuda.device_count()))

            self.logger.info("evaluation precision reduce result %.4f" % ap_1.item())
            ap_1 = ap_1.item()
            if dist.get_rank() == 0:
                if best_loss < ap_1:
                    best_loss = ap_1
                    self.best_epoch = epoch
                    state_dict = self.model.module.state_dict()

                    torch.save({'state_dict': state_dict}, self.save_local_path)
                    self.logger.info('Save best model')
                    if self.local is False:
                        save_data(os.path.join(self.oss_path, self.model_path), self.save_local_path)
        
        self.seen_and_unseen_test(dataloader=self.test_loader, type="test")

    def generate_train(self):
        """
        train model and freeze discriminator
        labeled Train: use labeled data to train total model
        unlabeled train: use unlabeled data to train total model without classifier
        generated train: generate data from label matrix and random label and train the classifier

        Args:
            generate (_type_): _description_
            unlabel (_type_): _description_

        Returns:
            _type_: _description_
        """        
        # init loss
        loss_dict = {
            'contrastive_t_loss': 0.0,
            'contrastive_p_loss': 0.0,
            'recon loss': 0.0,
            'vq_loss': 0.0,
            'classifier_loss': 0.0,
            'discriminator_loss': 0.0
        }
        step = 0

        # start train, freeze discriminator and train generator
        self.model.train()
        
        final_targets = []
        final_probabs = []
        

        # 组合训练数据
        length = len(self.train_loader)
        data_loader = self.train_loader

        # start epoch
        for step, batch in tqdm(enumerate(data_loader), total=length, leave=True):
            # start common batch
            train_batch = batch
            final_targets, final_probabs, total_loss = self.batch_gen_train(batch=train_batch,
                                                                            final_targets=final_targets,
                                                                            final_probabs=final_probabs)
            
            if self.model_config.detect:
                with torch.autograd.detect_anomaly():
                    # backward
                    print('total loss is', total_loss)
                    total_loss.backward()
            else:
                total_loss.backward()
                
            for name, param in self.model.named_parameters():
                if param.grad is None:
                    print('generator', name)

            if self.model_config.grad_norm:
                clip_grad_norm_(self.model.parameters(),
                                self.model_config.grad_norm)

            self.optimizer.step()
            self.optimizer.zero_grad()
            self.first_batch = False
        if self.model_config.use_scheduler:
            self.scheduler.step()

        # output classifier result
        _ = self.get_metric_result(final_probabs, final_targets, name='train')
        return loss_dict, final_targets, final_probabs

    def batch_gen_train(self, batch, final_targets, final_probabs, is_train=True):
        '''
        common training
        '''
        # deal batch data
        result_dict = batch
        batch_x_text = result_dict['origin']['text']
        
        y = result_dict['y_data']

        # get result
        result_dict = self.model(batch_x_text,
                                 y=y.to(device),
                                 )


        # calculate probability
        class_prob = result_dict['prob']
        total_loss = result_dict['loss']
        final_targets, final_probabs = self.get_prob_and_target(y, class_prob, final_targets, final_probabs)

        return final_targets, final_probabs, total_loss

    def evaluate(self, loader, get_loss=False):
        final_targets = []
        final_probabs = []
        self.model.eval()
        eval_loss = 0
        with torch.no_grad():
            if get_loss:
                for batch in tqdm(self.eval_loader):
                    result_dict = batch
                    batch_x_text = result_dict['origin']['text']
                    
                    y = result_dict['y_data']

                    # get result
                    result_dict = self.model(batch_x_text,
                                            y=y.to(device),
                                            )

                    # calculate loss
                    loss = result_dict['loss']
                    prob = result_dict['prob']
                    eval_loss += torch.mean(loss).item()
                    final_targets, final_probabs = self.get_prob_and_target(y, prob, final_targets, final_probabs)
            else:
                # Set the model to evaluation mode
                for batch in loader:
                    result_dict = batch
                    batch_x_text = result_dict['origin']['text']
                    
                    y = result_dict['y_data']

                    # get result
                    result_dict = self.model(batch_x_text,
                                            y=y.to(device),
                                            )

                    # calculate loss
                    loss = result_dict['loss']
                    prob = result_dict['prob']
                    final_targets, final_probabs = self.get_prob_and_target(y, prob, final_targets, final_probabs)
        print('max is', max(final_probabs))
        print('min is', min(final_probabs))
        # print(max(final_probabs), min(final_probabs))
        return final_probabs, final_targets, eval_loss

    def seen_and_unseen_test(self, dataloader, type="test"):
        self.logger.info('Start model test with name: %s, epoch %d' % (self.model_path, self.best_epoch))
        if type == "test":
            try:
                self.model.load_state_dict(torch.load(self.save_local_path)['state_dict'])
            except RuntimeError:
                self.model.module.load_state_dict(torch.load(self.save_local_path)['state_dict'])
        else:
            pass

        self.logger.info('start test')
        probabs, targets, _ = self.evaluate(loader=dataloader, get_loss=False)
        seen_idx_list = []
        unseen_idx_list = []
        index = 0
        for batch in dataloader:
            result_dict = batch
            y = result_dict['y_data']
            for i in range(y.shape[0]):
                if y[i].item() in self.seen_train:
                    seen_idx_list.append(index)
                else:
                    unseen_idx_list.append(index)
                index += 1
        print('seen class number:', len(seen_idx_list), 'unseen_class number:', len(unseen_idx_list))
        self.get_metric_result(probabs, targets, name='test total')
        seen_prob = [probabs[i] for i in seen_idx_list]
        seen_targets = [targets[i] for i in seen_idx_list]
        self.get_metric_result(seen_prob, seen_targets, name='test seen')
        unseen_prob = [probabs[i] for i in unseen_idx_list]
        unseen_targets = [targets[i] for i in unseen_idx_list]
        self.get_metric_result(unseen_prob, unseen_targets, name='test unseen')
        return probabs, targets


class OneModelTrainerParallel_GPT2(OneModelTrainer_GPT2):
    def __init__(self, model_config, logger, dataloader_list, label_mat, generated_dataset, emb_layer=None, local=False, **kwargs):
        dist.init_process_group('nccl')
        super().__init__(model_config, logger, dataloader_list, label_mat, generated_dataset, emb_layer=emb_layer, local=local, **kwargs)

    def build_model(self):
        '''
        initialize model
        '''
        super().build_model()
        rank = dist.get_rank()
        self.model = torch.nn.parallel.DistributedDataParallel(
            self.model, device_ids=[rank], find_unused_parameters=False)
        after_init_linear1 = torch.cuda.memory_allocated()
        print('model usage', after_init_linear1)
        
    def load_dataset(self, dataloader_list, generated_dataset, **kwargs):
        world_size = dist.get_world_size()
        self.div_batch_size = self.batch_size
        self.batch_size = self.batch_size * world_size
        
        self.train_loader, self.test_loader, self.eval_loader, self.unlabel_dataset, self.label_dataset = dataloader_list
        # get sampler
        train_sampler = data_dist.DistributedSampler(self.train_loader)
        test_sampler = data_dist.DistributedSampler(self.test_loader)
        dev_sampler = data_dist.DistributedSampler(self.eval_loader)
        label_sampler = data_dist.DistributedSampler(self.label_dataset)
        unlabel_sampler = data_dist.DistributedSampler(self.unlabel_dataset)
        
        self.train_loader = DataLoader(dataset=self.train_loader,
                                       batch_size=self.div_batch_size,
                                       sampler=train_sampler)
        self.test_loader = DataLoader(dataset=self.test_loader,
                                      batch_size=self.div_batch_size,
                                      sampler=test_sampler)
        self.eval_loader = DataLoader(dataset=self.eval_loader,
                                      batch_size=self.div_batch_size,
                                      sampler=dev_sampler)
        self.unlabel_loader = DataLoader(dataset=self.unlabel_dataset,
                                         batch_size=self.div_batch_size,
                                         sampler=unlabel_sampler)
        self.label_loader = DataLoader(dataset=self.label_dataset,
                                         batch_size=self.div_batch_size,
                                         sampler=label_sampler)

    def train(self):
        """
        训练传入的模型
        """
        rank = dist.get_rank()
        print(f"Start running basic DDP example on rank {rank}.")
        # self.logger.info('Start model train with name: %s' % (self.model_path))

        # output dataset length
        print('output length', len(self.train_loader), len(self.unlabel_loader))
        best_loss = 0
        self.best_epoch = 0
        # start epoch
        if self.model_config.detect is True:
            torch.autograd.set_detect_anomaly(True)
        for epoch in range(self.epochs):
            self.train_loader.sampler.set_epoch(epoch)  # type: ignore
            self.unlabel_loader.sampler.set_epoch(epoch)  # type: ignore
            self.label_loader.sampler.set_epoch(epoch)  # type: ignore

            self.epoch = epoch
            self.logger.info('Start epoch %d' % self.epoch)

            # generator train
            gen_loss_dict, final_targets, final_probabs = self.generate_train()
            
            ap_1 = self.eval(name='dev')
            ap_1 = torch.tensor(ap_1).to(device)
            dist.all_reduce(ap_1.div_(torch.cuda.device_count()))

            self.logger.info("evaluation precision reduce result %.4f" % ap_1.item())
            ap_1 = ap_1.item()
            if dist.get_rank() == 0:
                if best_loss < ap_1:
                    best_loss = ap_1
                    self.best_epoch = epoch
                    state_dict = self.model.module.state_dict()
                    

                    torch.save({'state_dict': state_dict}, self.save_local_path)
                    self.logger.info('Save best model')
                    if self.local is False:
                        save_data(os.path.join(self.oss_path, self.model_path), self.save_local_path)
                # early stop
                if epoch - self.best_epoch >= 50:
                    break
        
        self.seen_and_unseen_test(dataloader=self.test_loader, type="test")