import torch
import argparse
import sys
sys.path.append('..')
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from codes.trainflow import SituationFlow

try:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", default=-1, type=int)
    args = parser.parse_args()

    torch.cuda.set_device(args.local_rank)
except:
    print('use one card')

if __name__ == '__main__':
    download = False
    dataset_dict = {}
    model_dict = {'lr': 5e-5,
                  'max_epochs': 10
                    }       
    #for model in ['EASYSIM', 'AttnXML', 'TransICD', 'FZML']:
    # for model in ['Bert_RL', 'ZeroGen', 'BERTClassifier']:
    for model in ['ZeroGen']:
        flow = SituationFlow(dataset_dict=dataset_dict, model_dict=model_dict, emb_type='bert', parallel=False, model=model,
                            debug=False, init=False, local=True, download=download, decoder_gpt=True,entailment=True, pre_idx=True)
        flow.run()
        download = False  # only need download in the first time
    # for model in ['FZML']:
    #     flow = SituationFlow(dataset_dict=dataset_dict, model_dict=model_dict, emb_type='w2v', parallel=False, model=model,
    #                         debug=False, init=False, local=False, download=download)
    #     flow.run()
    #     download = False  # only need download in the first time
    
