import torch
import argparse
from config import config
import sys
sys.path.append(config.path)
sys.path.append(config.parent_path)
print(sys.path)
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from codes.trainflow import SituationFlow

try:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", default=-1, type=int)
    args = parser.parse_args()

    torch.cuda.set_device(args.local_rank)
except:
    print('use one card')

if __name__ == '__main__':
    print('start train and task')
    download = False
    init = False
    dataset_dict = {'easy_label': True, "train_type": "v0"}

    for lr in [5e-5, 3e-5, 2e-5]:
        model_dict = {'contras_p': False,
                        'contras_t': True,
                        'ablation_generate_train': True,
                        'use_new_self_training': True,
                        'use_pseudo_label': True,
                        'ablation_discriminator': True,
                        'generate': True,
                        'unlabel': True,
                        'lr': lr,
                        'max_epochs': 10,
                        'disperse_vq_type': 'GS',
                        # 'disperse_vq_type': 'DVQ',
                        'distil_vq_type': 'Fix',
                        'graph_type': 'Straight',
                        'decoder_type': 'GPT2',
                        'encoder_type': 'Bert_Entail_CLS',
                        'encoder_p_type': 'Bert_Pooler',
                        'start_unlabel_epoch': 0,
                        'start_generate_epoch': 0,
                        'batch_size': 14,
                        'generate_size': 10,
                        'start_vq_epoch': 0,
                        'easy_decoder': True,
                        'use_memory': False,
                        'num_layers': 4,
                        'disper_num': 64,
                        'encoder_output_size': 768,
                        'decoder_input_size': 64,
                        'pseudo_label_threshold': 0.8
                        }         
        print('###########################################this task set is:', model_dict, dataset_dict)
        emb_type = 'bert'
        flow = SituationFlow(dataset_dict=dataset_dict, model_dict=model_dict, emb_type=emb_type,
                        parallel=True, 
                        model = 'Entailment',
                        debug=False, init=init, local=True, download=download, last_label=True, decoder_gpt=True,
                        entailment=True, pre_idx=True)

        flow.run()
        download = False  # only need download in the first time
        init = False
