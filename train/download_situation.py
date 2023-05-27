import torch
import argparse
from ..config import config
import sys
sys.path.append(config.path)
# print(sys.path)
from codes.trainflow import SituationFlow


parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", default=-1, type=int)
parser.add_argument("--emb_type", default='w2v', type=str)
parser.add_argument("--init", default='False', type=str)
args = parser.parse_args()
try:
    torch.cuda.set_device(args.local_rank)
except:
    print('use one card')


if __name__ == '__main__':
    print('start download data')
    print(args.emb_type)
    print('download init status', args.init)
    download = True
    init = args.init
    if init == 'False':
        init = False
    elif init == 'True':
        init = True
    else:
        raise TypeError
    dataset_dict = {'easy_label': True}

                        
    model_dict = {'contras_p': True,
                    'contras_t': True,
                    'generate': True,
                    'unlabel': True,
                    'lr': 0.003,
                    'disperse_vq_type': 'DVQ',
                    'distil_vq_type': 'Striaght',
                    'classifier_type': 'Euclidean',
                    'discri_type': 'Linear',
                    'graph_type': 'GAT',
                    'decoder_type': 'Transformer',
                    'encoder_type': 'Transformer',
                    'encoder_p_type': 'Transformer',
                    'start_unlabel_epoch': 0,
                    'start_generate_epoch': 0,
                    'batch_size': 40,
                    'generate_size': 16,
                    'start_vq_epoch': 0,
                    'easy_decoder': False,
                    'use_memory': False,
                    'num_layers': 4,
                    'disper_num': 16,
                    'use_w2v_weight': True
                    }         
    print('###########################################this task set is:', model_dict, dataset_dict)
    
    emb_type = args.emb_type
    
    flow = SituationFlow(dataset_dict=dataset_dict, model_dict=model_dict, emb_type=emb_type,
                    parallel=True, model='Entailment', pre_idx=False,
                    debug=False, init=init, local=False, download=download, last_label=True, decoder_gpt=True)

    flow.run(train=False, test=False)
    download = False  # only need download in the first time
    init = False
