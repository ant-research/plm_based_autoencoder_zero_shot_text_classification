import torch
import argparse
import sys
sys.path.append('..')
from codes.trainflow import WOSFlow, TopicFlow

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
    dataset_dict = {'easy_label': True}

    for lr in [5e-5, 3e-5, 2e-5]:        
        model_dict = {'lr': lr,
                      'batch_size': 20,
                      'max_epochs': 100
                        }         
        print('###########################################this task set is:', model_dict, dataset_dict)
        emb_type = 'bert'
        flow = TopicFlow(dataset_dict=dataset_dict, model_dict=model_dict, emb_type=emb_type,
                        parallel=True, model='BERTClassifier',
                        debug=False, init=init, local=False, download=download, last_label=True, decoder_gpt=True,
                        entailment=True, pre_idx=True)

        flow.run()
        download = False  # only need download in the first time
        init = False
