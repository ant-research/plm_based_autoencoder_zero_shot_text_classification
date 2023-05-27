import torch
import os
from itertools import cycle
from tqdm import tqdm
from torch.utils.data import DataLoader
from codes.utils import save_data, MemTracker
from model.new_model.model.model import Model
from model.new_model.utils.loss import ContrastiveLoss, ContrastiveLoss_Neg
from model.train_base import Trainbasic
from multiprocessing import cpu_count

cpu_num = cpu_count() - 1 # 自动获取最大核心数目
os.environ['OMP_NUM_THREADS'] = str(cpu_num)
os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
os.environ['MKL_NUM_THREADS'] = str(cpu_num)
os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
torch.set_num_threads(cpu_num)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


class OneModelTrainer_Fix(Trainbasic):
    def __init__(self, model_config, logger, dataloader_list, label_mat, generated_dataset,
                 emb_layer=None, local=False, **kwargs):
        super().__init__(model_config, logger, emb_layer=emb_layer, local=False, **kwargs)
        # 训练组件设置
        self.output_generate = model_config.output_generate
        self.unlabel = model_config.is_unlabel_train  # 是否使用unlabel dataset训练
        # 学习参数
        self.epochs = model_config.epochs
        self.batch_size = model_config.batch_size
        self.generate_size = model_config.generate_size
        self.lr = model_config.lr
        # eos向量，用于截断输出句子
        self.eos = self.emb_layer.get_eos()
        # model config
        self.class_num = model_config.class_num
        self.model_path = model_config.model_name
        self.save_local_path = os.path.join('./data', self.model_path)
        self.oss_path = model_config.save_model_oss_path
        # help tracker
        self.gpu_tracker = MemTracker()
        # load dataset
        self.load_dataset(dataloader_list, generated_dataset)

        # deal with label matrix
        if label_mat.shape[1] == 1:
            label_mat = label_mat.squeeze(1)
        self.label_mat = label_mat.to(device)

        # build a new model
        self.build_model()
        # build optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=self.lr,
                                          betas=(0.9, 0.999),
                                          eps=1e-8,
                                          weight_decay=0.01)
        # 无用
        self.start_hard_epoch = 1000000
        # 分seen和unseen clas
        self.get_seen_class_id()

    def emb_data(self, x):
        '''
        emb_data
        '''
        if self.emb_layer:
            result = []
            padding_result = []
            for text in x:
                vector, padding_mask = self.emb_layer.emb_one_text(text)
                result.append(vector.unsqueeze(0).to(device))
                padding_result.append(padding_mask.unsqueeze(0).to(device))
            result = torch.cat(result, dim=0).to(device)
            padding_result = torch.cat(padding_result, dim=0).to(device)
            return result, padding_result
        else:
            return x.to(device), None

    def load_dataset(self, dataloader_list, generated_dataset):
        self.train_loader, self.test_loader, self.eval_loader, self.unlabel_dataset = dataloader_list
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
        self.generated_dataset = generated_dataset
        self.generated_loader = DataLoader(dataset=self.generated_dataset,
                                           batch_size=self.generate_size,
                                           shuffle=True)

    def build_model(self):
        '''
        initialize model
        '''
        self.model = Model(label_matrix=self.label_mat,
                            config=self.model_config,
                            device=device,
                            gpu_tracker=self.gpu_tracker,
                            emb_layer=self.emb_layer,
                            bos=self.emb_layer.get_bos().to(device).unsqueeze(0))
        self.model = self.model.to(device)

    def train(self):
        """
        训练传入的模型
        """
        self.logger.info('Start model train with name: %s' % (self.model_path))
        num_gpu = torch.cuda.device_count()  # parallel train or not
        # early stop setting
        best_loss = 0
        self.best_epoch = 0
        # output dataset length
        print('output length', len(self.train_loader), len(self.generated_loader), len(self.unlabel_loader))
        # start epoch
        for epoch in range(self.epochs):
            self.epoch = epoch
            # set data loader and get train setting
            if self.model_config.is_generate_train and epoch >= self.model_config.generate_epoch:
                generate = True
            else:
                generate = False
            if self.unlabel and epoch >= self.model_config.unlabel_epoch:
                unlabel = True
            else:
                unlabel = False

            loss_dict, final_targets, final_probabs = self.one_epoch_train(generate=generate,
                                                                           unlabel=unlabel,
                                                                           epoch=epoch)
            vq_loss = loss_dict['vq_loss']
            contrastive_loss = loss_dict['contrastive_loss']
            discriminator_loss = loss_dict['discriminator_loss']

            # output vq result
            self.model.vq_vae.emb_p.output_similarity()

            # start eval, use ap@k
            ap_1 = self.eval(name='dev')
            self.logger.info('Epoch: %d, use generate %d vq loss: %.4f, contrastive loss, %.4f, discriminator loss, %.4f eval loss %.4f' %
                             (epoch, generate, vq_loss, contrastive_loss, discriminator_loss, ap_1))

            # save best model
            if epoch > self.model_config.generate_epoch:
                if best_loss < ap_1:
                    best_loss = ap_1
                    self.best_epoch = epoch
                    state_dict = self.model.module.state_dict() if num_gpu > 1 else self.model.state_dict()

                    torch.save({'state_dict': state_dict}, self.save_local_path)
                    self.logger.info('Save best model')
                    if self.local is False:
                        save_data(os.path.join(self.oss_path, self.model_path), self.save_local_path)

                # early stop
                if epoch - self.best_epoch >= 20:
                    break

    def get_loss(self, batch_x, recon_x, loss_p, relation, batch_y=None, x_pad_mask=None, epoch=0, generate=False, unlabel=False):
        if isinstance(self.model, torch.nn.DataParallel):
            common_loss, dis_loss = self.model.module.loss_function(batch_x, recon_x, loss_p,
                                                                    relation, batch_y,
                                                                    x_pad_mask=x_pad_mask,
                                                                    generate=generate,
                                                                    unlabel=unlabel)  # 计算loss
        else:
            common_loss, dis_loss = self.model.loss_function(batch_x, recon_x, loss_p,
                                                             relation, batch_y,
                                                             x_pad_mask=x_pad_mask,
                                                             epoch=epoch,
                                                             generate=generate,
                                                             unlabel=unlabel)  # 计算loss
        return common_loss, dis_loss

    def one_epoch_train(self, generate, unlabel, epoch):
        # init loss
        contrastive_loss = 0
        vq_loss = 0
        discriminator_loss = 0
        # start train
        self.model.train()
        self.logger.info('Start epoch %d' % epoch)
        final_targets = []
        final_probabs = []
        # first batch to generate
        self.first_batch = True
        if generate is False and unlabel is False:
            data_loader = zip(self.train_loader)
            length = len(self.train_loader)
        elif generate is True and unlabel is False:
            data_loader = zip(cycle(self.train_loader), self.generated_loader)
            length = max(len(self.train_loader), len(self.generated_loader))
        elif generate is False and unlabel is True:
            data_loader = zip(cycle(self.train_loader), cycle(self.generated_loader), self.unlabel_loader)
            length = max(len(self.train_loader), len(self.generated_loader), len(self.unlabel_loader))
        elif generate is True and unlabel is True:
            data_loader = zip(cycle(self.train_loader), cycle(self.generated_loader), self.unlabel_loader)
            length = max(len(self.train_loader), len(self.generated_loader), len(self.unlabel_loader))
        for step, batch in tqdm(enumerate(data_loader), total=length, leave=True):
            train_batch = batch[0]
            loss_dict, final_targets, final_probabs, total_loss = self.one_batch_train(batch=train_batch,
                                                                                       epoch=epoch,
                                                                                       final_targets=final_targets,
                                                                                       final_probabs=final_probabs,)
            contrastive_loss += loss_dict['contrastive_loss']
            vq_loss += loss_dict['vq_loss']
            discriminator_loss += loss_dict['discriminator_loss']

            # generate train
            if generate:
                generate_batch = batch[1]
                g_dis_loss = self.one_generate_batch(generate_batch)
                total_loss += g_dis_loss

            # unlabeled train
            if unlabel:
                unlabel_batch = batch[2]
                unlabel_loss = self.one_unlabel_batch(unlabel_batch)
                total_loss += unlabel_loss

            # backward
            total_loss.backward()

            self.optimizer.step()
            for param in self.model.parameters():
                param.grad = None

            self.first_batch = False

        vq_loss = vq_loss/(step+1)
        contrastive_loss = contrastive_loss/(step+1)
        discriminator_loss = discriminator_loss/(step+1)

        loss_dict = {
            'contrastive_loss': contrastive_loss,
            'vq_loss': vq_loss,
            'discriminator_loss': discriminator_loss
        }
        return loss_dict, final_targets, final_probabs

    def one_batch_train(self, batch, epoch, final_targets, final_probabs, is_train=True):
        '''
        common training
        '''
        # loss function init
        self.contras_loss_fun = ContrastiveLoss().to(device)
        self.contras_p_loss_fun = ContrastiveLoss().to(device)
        # self.contras_p_loss_fun = ContrastiveLoss_Neg(self.label_mat).to(device)
        # return loss
        contrastive_loss = 0
        discriminator_loss = 0
        vq_loss = 0
        total_loss = None
        # deal batch data
        x, y, x_da = batch
        batch_y = y.to(device)
        batch_x, x_pad_mask = self.emb_data(x)
        batch_x_da, x_da_pad_mask = self.emb_data(x_da)

        # get result
        relation, recon_x, z_t, z_p, loss_p = self.model(batch_x, x_pad_mask,
                                                         batch_y.to(batch_x_da.device))
        # get da result
        relation_da, recon_x_da, z_t_da, z_p_da, loss_p_da = self.model(batch_x_da,
                                                                        x_da_pad_mask,
                                                                        batch_y.to(batch_x_da.device))
        final_targets, final_probabs = self.get_prob_and_target(y, relation, final_targets, final_probabs)

        if is_train:
            # 检测生成的句子
            if self.output_generate and self.first_batch:
                sent_list, _, _ = self.generate_sentence(recon_x)
                for i in range(len(sent_list)):
                    print('epoch:', self.epoch, 'recon句子', sent_list[i], 'origin sentence', x[i])
            # print(torch.topk(result, 10))
            common_loss, dis_loss = self.get_loss(batch_x, recon_x, loss_p, relation, batch_y, x_pad_mask=x_pad_mask,
                                                  epoch=epoch, generate=False)
            common_da_loss, dis_da_loss = self.get_loss(batch_x_da, recon_x_da, loss_p_da, 
                                                        relation_da, batch_y,
                                                        x_pad_mask=x_da_pad_mask,
                                                        epoch=epoch,
                                                        generate=False)
            # epoch 为 10， 20， 30的时候输出一次句子
            # 检测生成的句子
            # if self.output_generate and epoch in [10, 20, 30]:
            #     sent_list = self.generate_sentence(recon_x)
            #     print('生成句子', sent_list, '原句', x)

            # start deal loss
            total_loss = common_loss + common_da_loss

            # calculate contrastive loss
            if self.model_config.is_contrastive_t:
                contrastive_loss_t = self.contras_loss_fun(z_t, z_t_da)
                contrastive_loss += contrastive_loss_t.item()
                total_loss += contrastive_loss_t
                print('contrastive loss t is', contrastive_loss_t)
            if self.model_config.is_contrastive_p:
                contrastive_loss_p = -self.contras_loss_fun(z_p, z_p_da)
                contrastive_loss += contrastive_loss_p.item()
                print('contrastive loss p is', contrastive_loss_p)
                total_loss += contrastive_loss_p

            discriminator_loss = torch.mean(dis_loss).item() + torch.mean(dis_da_loss).item()
            vq_loss += torch.mean(common_loss).item() + torch.mean(common_da_loss).item()

        loss_dict = {
            'contrastive_loss': contrastive_loss,
            'vq_loss': vq_loss,
            'discriminator_loss': discriminator_loss
        }
        return loss_dict, final_targets, final_probabs, total_loss

    def one_unlabel_batch(self, batch):
        # 处理原文本以及da数据
        text, text_da = batch
        batch_x, x_pad_mask = self.emb_data(text)
        batch_x_da, x_da_pad_mask = self.emb_data(text_da)
        # model reconstruction
        easy = True
        # get result
        relation, recon_x, z_t, z_p, loss_p = self.model(batch_x, x_pad_mask)
        relation_da, recon_x_da, z_t_da, z_p_da, loss_p_da = self.model(batch_x, x_pad_mask)
        # get loss, 设置unlabel为True，不计算cross entropy loss
        common_loss, _ = self.get_loss(batch_x, recon_x, loss_p, relation,
                                       x_pad_mask=x_pad_mask,
                                       generate=False,
                                       unlabel=True)
        common_da_loss, _ = self.get_loss(batch_x_da,
                                          recon_x_da,
                                          loss_p_da,
                                          relation_da,
                                          x_pad_mask=x_da_pad_mask,
                                          generate=False,
                                          unlabel=True)
        total_loss = common_loss + common_da_loss
        # get contrastive loss
        if self.model_config.is_contrastive_t:
            contrastive_loss_t = self.contras_loss_fun(z_t, z_t_da)
            total_loss += contrastive_loss_t
            print('contrastive loss t is', contrastive_loss_t)
        if self.model_config.is_contrastive_p:
            contrastive_loss_p = -self.contras_loss_fun(z_p, z_p_da)
            print('contrastive loss p is', contrastive_loss_p)
            total_loss += contrastive_loss_p

        return total_loss

    def one_generate_batch(self, batch, n=1):
        '''
        用生成的数据训练classifier
        '''
        y_idx, y, y_text = batch
        y_idx = y_idx.to(device)
        y = y.to(device)
        if isinstance(self.model, torch.nn.DataParallel):
            x, y_idx, p_idx = self.model.module.sample_result(y_idx, device=device, n=n)  # type: ignore
        else:
            x, y_idx, p_idx = self.model.sample_result(y_idx, device=device, n=n)
        x = self.emb_layer.emb_model.cut_and_pad(x)
        sent_list, true_idx_list, pad_mask = self.generate_sentence(x)

        if self.output_generate and self.first_batch:
            for i in range(len(sent_list)):
                print('epoch:', self.epoch, 'generate 句子', sent_list[i], 'label', y_text[i], p_idx[i])
        # 从句子中重新生成batch
        batch_x, x_pad_mask = self.emb_data(sent_list)
        # get result
        relation, recon_x, z_t, z_p, loss_p = self.model(batch_x, x_pad_mask)
        # 检测生成的句子
        if self.output_generate and self.first_batch:
            sent_list, _, _ = self.generate_sentence(recon_x)
            print('epoch:', self.epoch, 'generate结果', sent_list)
        common_loss, dis_loss = self.get_loss(x, recon_x, loss_p, relation, y,
                                              x_pad_mask=x_pad_mask,
                                              epoch=100, generate=True)
        return dis_loss

    def evaluate(self, loader, get_loss=False):
        self.logger.info('Start model test with name: %s' % (self.model_path))
        final_targets = []
        final_probabs = []
        self.model.eval()
        eval_loss = 0
        with torch.no_grad():
            if get_loss:
                for batch in tqdm(self.eval_loader):
                    loss_dict, final_targets, final_probabs, _ = self.one_batch_train(batch=batch,
                                                                                   epoch=100,
                                                                                   final_targets=final_targets,
                                                                                   final_probabs=final_probabs,
                                                                                   is_train=False)
                    # calculate eval loss
                    eval_loss += loss_dict['discriminator_loss']
                eval_loss = eval_loss/len(self.eval_loader.dataset)
            else:
                # Set the model to evaluation mode
                for batch in loader:
                    x, y, _ = batch
                    batch_x, x_pad_mask = self.emb_data(x)
                    # get result
                    relation, _, _, _, _ = self.model(batch_x)
                    final_targets, final_probabs = self.get_prob_and_target(y, relation, final_targets, final_probabs)

        print('max is', max(final_probabs))
        print('min is', min(final_probabs))
        return final_probabs, final_targets, eval_loss

    def generate_sentence(self, recon_x):
        '''
        从recon x中生成句子
        '''
        batch_num = recon_x.shape[0]
        print('recon shape is', recon_x.shape)
        sent_list = []
        true_idx_list = []
        pad_mask_list = []

        for i in range(batch_num):
            self.logger.info('start sentence %d' % i)
            sentence = []
            one_sentence_output = recon_x[i, :, :]
            sentence, true_idx, pad_mask = self.emb_layer.emb_model.find_sim_word(one_sentence_output)
            sent_list.append(sentence)
            true_idx_list.append(true_idx)
            pad_mask_list.append(pad_mask)
        return sent_list, torch.tensor(true_idx_list), torch.tensor(pad_mask_list).float()

    def eval(self, name='dev'):
        seen_idx_list = []
        unseen_idx_list = []
        index = 0
        for batch in self.eval_loader:
            x, y, _ = batch
            for i in range(y.shape[0]):
                if y[i].item() in self.seen_train:
                    seen_idx_list.append(index)
                else:
                    unseen_idx_list.append(index)
                index += 1

        self.logger.info('Start model eval with name: %s' % (self.model_path))
        probabs, targets, eval_loss = self.evaluate(loader=self.eval_loader, get_loss=True)
        apk_1 = self.get_metric_result(probabs, targets, name=name)
        seen_prob = [probabs[i] for i in seen_idx_list]
        seen_targets = [targets[i] for i in seen_idx_list]
        self.get_metric_result(seen_prob, seen_targets, name='eval seen')
        if len(unseen_idx_list) > 0:
            unseen_prob = [probabs[i] for i in unseen_idx_list]
            unseen_targets = [targets[i] for i in unseen_idx_list]
            self.get_metric_result(unseen_prob, unseen_targets, name='eval unseen')
        return apk_1

    def generate_all(self):
        with torch.no_grad():
            for step, batch in tqdm(enumerate(self.generated_loader), total=len(self.generated_loader), leave=True):
                y_idx, y, y_text = batch
                y_idx = y_idx.to(device)
                y = y.to(device)
                for i in range(y.shape[0]):
                    one_y_idx = y_idx[i].unsqueeze(0)
                    if isinstance(self.model, torch.nn.DataParallel):
                        x, one_y_idx, p_idx = self.model.module.vq_vae.sample_all(one_y_idx, device=device)  # type: ignore
                    else:
                        x, one_y_idx, p_idx = self.model.vq_vae.sample_all(one_y_idx, device=device)

                    sent_list, true_idx_list, pad_mask = self.generate_sentence(x, need_build=True)
                    # if self.output_generate and self.first_batch:
                    if one_y_idx.item() in self.seen_train:
                        seen_type = 'train'
                    elif one_y_idx.item() in self.seen_eval:
                        seen_type = 'eval'
                    else:
                        seen_type = 'unseen'
                    print('epoch', self.epoch, seen_type, 'label is', y_text[i])
                    for j in range(len(sent_list)):
                        print('epoch:', self.epoch, 'pidx', p_idx[j], 'generate句子', sent_list[j])


class OneModelTrainerParallel_Fix(OneModelTrainer_Fix):
    def __init__(self, model_config, logger, dataloader_list, label_mat, generated_dataset, emb_layer=None, local=False, **kwargs):
        super().__init__(model_config, logger, dataloader_list, label_mat, generated_dataset, emb_layer=emb_layer, local=local, **kwargs)

    def build_model(self):
        '''
        initialize model
        '''
        super().build_model()
        self.model = torch.nn.DataParallel(self.model.to(device))
        after_init_linear1 = torch.cuda.memory_allocated()
        print('model usage', after_init_linear1)
