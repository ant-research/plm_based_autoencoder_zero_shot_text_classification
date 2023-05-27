import torch
import torch.nn.functional as F
from tqdm import tqdm
from model.new_model.utils.metric import compute_scores, compute_scores_single

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


class Trainbasic(object):
    '''
    训练模型
    '''
    def __init__(self, model_config, logger, emb_layer, local=False, parent_adj=None, child_adj=None, **kwargs):
        # init params
        self.logger = logger
        self.emb_layer = emb_layer
        self.model_config = model_config
        self.local = local  # save model local or to oss
        self.parent_adj = parent_adj
        self.child_adj = child_adj

    def get_seen_class_id(self):
        '''
        区分不同dataloader中的seen和unseen class
        '''
        def get_one_seen_class(data_loader):
            seen_list = []
            for batch in tqdm(data_loader):
                result_dict = batch
                y = result_dict['y_data']
                for i in range(y.shape[0]):
                    seen_list.append(y[i].item())
            return list(set(seen_list))
        self.seen_train = get_one_seen_class(self.train_loader)
        self.seen_eval = get_one_seen_class(self.eval_loader)
        self.seen_test = get_one_seen_class(self.test_loader)

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

    def emb_data(self, x, get_idx=True):
        '''
        emb_data
        '''
        if self.emb_layer:
            if get_idx is False:
                result = []
                padding_result = []
                for text in x:
                    vector, padding_mask = self.emb_layxwwwer.emb_one_text(text)
                    result.append(vector.unsqueeze(0).to(device))
                    padding_result.append(padding_mask.unsqueeze(0).to(device))
                result = torch.cat(result, dim=0).to(device)
                padding_result = torch.cat(padding_result, dim=0).to(device)
                return result, padding_result
            else:
                result = []
                padding_result = []
                for text in x:
                    input_list, padding_mask = self.emb_layer.emb_model.tokenize(text, unk=True)
                    result.append(input_list.unsqueeze(0).to(device))
                    padding_result.append(padding_mask.unsqueeze(0).to(device))
                result = torch.cat(result, dim=0).to(device)
                padding_result = torch.cat(padding_result, dim=0).to(device)
                return result, padding_result
        else:
            return x.to(device), None


    def test(self, name='test'):
        self.logger.info('Start model test with name: %s, epoch %d' % (self.model_path, self.best_epoch))
        try:
            self.model.load_state_dict(torch.load(self.save_local_path)['state_dict'])
        except RuntimeError:
            self.model.module.load_state_dict(torch.load(self.save_local_path)['state_dict'])
        self.logger.info('start test')
        probabs, targets, _ = self.evaluate(loader=self.test_loader, get_loss=False)
        self.get_metric_result(probabs, targets, name=name)
        return probabs, targets

    def eval(self, name='dev'):
        self.logger.info('Start model eval with name: %s' % (self.model_path))  # type: ignore
        probabs, targets, eval_loss = self.evaluate(loader=self.eval_loader, get_loss=True)  # type: ignore
        apk_1 = self.get_metric_result(probabs, targets, name=name)
        return apk_1

    def get_prob_and_target(self, y, relation, final_target, final_probab, need_deal=True):
        '''
        get proability and targets
        '''
        if self.model_config.task_type == 'Multi_Label':
            final_target.extend(y.tolist())
            final_probab.extend(torch.sigmoid(relation).detach().cpu().tolist())
        else:
            y = F.one_hot(y, num_classes=self.class_num)
            final_target.extend(y.tolist())
            if need_deal:
                final_probab.extend(F.softmax(relation, dim=1).detach().cpu().tolist())
            else:
                final_probab.extend(relation.detach().cpu().tolist())
        return final_target, final_probab

    def get_metric_result(self, probabs, targets, name='dev'):
        '''
        get metrics
        '''
        if self.model_config.task_type == 'Multi_Label':
            apk_n = compute_scores(probabs, targets, name=name)
        else:
            apk_n = compute_scores_single(probabs, targets, name=name)
        return apk_n

    def generate_sentence(self, recon_x, need_build=True):
        """给定reconstruct结果，通过emb layer还原句子

        Args:
            recon_x (_type_): 2d tensor or 3d tensor
            need_build (bool, optional): if 2d tensor with idx, false, else need to do argmax, input true. Defaults to True.

        Returns:
            sent_list: sentence result
            true idx list: predict出的idx，加上padding后的结果
            pad mask: padding mask
        """
        if need_build:
            recon_x = recon_x.argmax(dim=2)
        sent_list = []
        true_idx_list = []
        pad_mask_list = []
        for i in range(recon_x.shape[0]):
            sentence_idx = recon_x[i, :].cpu().tolist()
            sentence, true_idx, pad_mask = self.emb_layer.emb_model.get_sentence_by_index(sentence_idx)
            sent_list.append(sentence)
            true_idx_list.append(true_idx)
            pad_mask_list.append(pad_mask)
        return sent_list, torch.tensor(true_idx_list), torch.tensor(pad_mask_list).float()