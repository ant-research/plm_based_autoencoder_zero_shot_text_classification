from turtle import distance
import torch
import torch.nn as nn
from typing import Union
import numpy as np
import random
from model.new_model.utils.loss import BCEFocalLoss, GraphCELoss


class ClassifierBase(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.loss_func = BCEFocalLoss()
    
    def loss_function(self, relation, y):
        '''
        '''
        # print('y is', y)
        d_loss = self.loss_func(relation, y)
        return d_loss


class PolicyNetwork(ClassifierBase):
    def __init__(self, input_size, **kwargs):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 600)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(600, 1)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, z_t, label_mat=None):
        '''
        z: [batch, distil_size]
        label_mat: [label_count, emb_size]
        '''
        # result_m = z_t/torch.norm(z_t, p=2, dim=1, keepdim=True)
        # label_cos = torch.matmul(result_m, result_m.t())
        # print('z_t before relu cos similarity', label_cos)

        z_t = self.relu(self.fc1(z_t))

        # result_m = z_t/torch.norm(z_t, p=2, dim=1, keepdim=True)
        # label_cos = torch.matmul(result_m, result_m.t())
        # print('z_t after relu cos similarity', label_cos)

        relation = self.fc2(z_t)

        # relation = self.sigmoid(relation)
        # print('relation max is', torch.max(relation, dim=1))
        # print('relation min is', torch.min(relation, dim=1))
        y = torch.ones(relation.shape[0]).to(relation.device)
        loss = self.loss_func(relation, y)
        return relation, loss

    def policy_gradient(self, rewards, gamma=0.99):
        r = np.array([gamma ** i * rewards[i] for i in range(len(rewards))])
        r = r[::-1].cumsum()[::-1]
        return r - r.mean()


class PolicyNetwork_Test(object):
    def __init__(self):
        # self.taxonomy_id = taxonomy_id
        # self.right_node_reward = right_node_reward
        self.wrong_node_reward = -0.1
        # self.path_max_length = path_max_length
    
    def init_state(self, current_node, mode):
        if mode == 'greedy':
            self.greedy_state = [ {} for x in current_node ]
            self.state = self.greedy_state
        elif mode == 'random':
            self.random_state = [ {} for x in current_node ]
            self.state = self.random_state
        self.level = 0
    
    def update_state(self, index, prob, current_node, batch_index):
        length = index.size()[0]
        for i in range(length):
            node = current_node[index[i]]
            node_state = prob[index[i]]
            batch_num = batch_index[index[i]]
            self.state[int(batch_num)][int(node)] = {'prob': node_state, 'level': self.level}
        self.level += 1
    
    def select_current_node(self, index, text, current_node, batch_index):
        text_ret = [text[i] for i in index]
        current_node_ret = [current_node[i] for i in index]
        batch_index_ret = [batch_index[i] for i in index]
        return text_ret, np.asarray(current_node_ret), np.asarray(batch_index_ret)

    def get_children(self, text, current_node, batch_index):
        new_text = []
        new_node = []
        new_batch_index = []
        for i, node in enumerate(current_node):
            children = self.taxonomy_id[node]['children']
            new_node.extend(children)
            new_text += [text[i] for x in children]
            new_batch_index += [batch_index[i]] * len(children)
        return new_text, np.asarray(new_node), np.asarray(new_batch_index)
    
    def get_greedy_action(self, prob):
        greedy_index = torch.nonzero(prob[:, 1] > prob[:, 0], as_tuple=False)
        return greedy_index.squeeze(1)
    
    def get_random_action(self, prob, batch_index, batch_size, K):
        prob = prob.detach().cpu().numpy()
        action = []
        batch_prob = [[] for i in range(batch_size)]
        batch_action = [[] for i in range(batch_size)]
        for i in range(len(prob)):
            index_ = batch_index[i]
            batch_prob[index_].append(prob[i])
            batch_action[index_].append(i)
        for i in range(batch_size):
            batch_selected_actions = []
            for j, one_weight in enumerate(batch_prob[i]):
                one_action = np.random.choice([0, 1], p=one_weight)
                if one_action == 1:
                    batch_selected_actions.append(batch_action[i][j])
            if len(batch_selected_actions) > K:
                batch_selected_actions = random.sample(batch_selected_actions, K)
            action.extend(batch_selected_actions)
        action = sorted(action)
        return torch.Tensor(action).to(torch.int64)

    def get_path(self, state):
        node_level = {}
        for key, val in state.items():
            level = val['level']
            if level not in node_level:
                node_level[level] = []
            node_level[level].append(key)
        node_rec = {one_node: False for one_node in state.keys()}
        path = []
        sorted_level = sorted(node_level.keys(), reverse=True)
        for level in sorted_level:
            level_node = node_level[level]
            for one_node in level_node:
                if node_rec[one_node]:
                    continue
                current_node = one_node
                one_path = []
                get_end = False
                while True:
                    if current_node == 0:
                        get_end = True
                        break
                    if current_node not in state:
                        break
                    one_path.append(current_node)
                    current_node = self.taxonomy_id[current_node]['parent']
                if get_end:
                    one_path.reverse()
                    path.append(one_path)
                    for one_node in one_path:
                        node_rec[one_node] = True
        rest_node = []
        for level, level_node in node_level.items():
            for one_node in level_node:
                if not node_rec[one_node]:
                    rest_node.append(one_node) 
        return path, rest_node
    
    def path_reward(self, state, category, path):  
        def judge_path(path, all_path):
            for one_path in all_path:
                flag = True
                for node in path:
                    if node not in one_path:
                        flag = False
                        break
                if flag:
                    return True
                else:
                    continue
            return False         
        reward = []
        for i, one_state in enumerate(state):
            predicted_path, rest_node = self.get_path(one_state)
            batch_reward = 0.
            for one_path in predicted_path:
                path_reward = 0.
                if judge_path(one_path, path[i]):
                    for one_node in one_path:
                        path_reward += self.right_node_reward * (-one_state[one_node]['prob'][1])
                else:
                    for one_node in one_path:
                        path_reward += self.wrong_node_reward * (-one_state[one_node]['prob'][1])
                batch_reward += path_reward
            for one_node in rest_node:
                batch_reward += -1 * (-one_state[one_node]['prob'][1])
            reward.append(batch_reward)
        return reward

    def get_reward(self, state, category, path):
        return self.path_reward(state, category, path)