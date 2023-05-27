from codes.topic.model_config import (
    NewModel_Config,
    NewFixModel_Config,
    TransICD_Config,
    FZML_Config,
    AttentiveXML_Config,
    EASYSIM_Config,
    DisVAE_Config,
    )

class NewModel_Config_Sit(NewModel_Config):
    def __init__(self, class_num=8549, emb_size=768, seq_len=512, contras_t=True, contras_p=True, generate=True,
                 disperse_vq_type='Hard', distil_vq_type='Fix', vae_type='Fix_VQ_VAE', lr=0.0005, unlabel=True,
                 output_generate=True, decoder_type='GPT_EMB_EASY', encoder_type='Transformer',
                 encoder_p_type='Transformer', start_unlabel_epoch=3, start_generate_epoch=20, batch_size=32,
                 generate_size=16, pretrain=False, pretrained_epochs=30):
        # save params
        self.save_model_oss_path = f'model_output/simple_model_vq_situation/'
        super().__init__(class_num=class_num, emb_size=emb_size, seq_len=seq_len, contras_t=contras_t, contras_p=contras_p, generate=generate,
                 disperse_vq_type=disperse_vq_type, distil_vq_type=distil_vq_type, vae_type=vae_type, lr=lr, unlabel=unlabel,
                 output_generate=output_generate, decoder_type=decoder_type, encoder_type=encoder_type,
                 encoder_p_type=encoder_p_type, start_unlabel_epoch=start_unlabel_epoch, start_generate_epoch=start_generate_epoch, batch_size=batch_size,
                 generate_size=generate_size, pretrain=pretrain, pretrained_epochs=pretrained_epochs)


class NewFixModel_Config_Sit(NewFixModel_Config):
    def __init__(self, class_num=8549, emb_size=200, seq_len=512, contras_t=True, contras_p=True, generate=True,
                 disperse_vq_type='Hard', distil_vq_type='Fix', vae_type='Fix_VQ_VAE', lr=0.0005, unlabel=True,
                 output_generate=True, decoder_type='GPT_EMB_EASY', encoder_type='Transformer',
                 encoder_p_type='Transformer', start_unlabel_epoch=3, start_generate_epoch=20, batch_size=32,
                 generate_size=16, pretrain=False, pretrained_epochs=30, graph_type='GAT', discri_type='Linear',
                 classifier_type='Euclidean', start_vq_epoch=10, easy_decoder=False, use_memory=False, num_layers=1, 
                 disper_num=40, use_w2v_weight=True, encoder_output_size=200, decoder_input_size=32, max_epochs=10, **kwargs):
        
        """
        Global Model setting
        """

        super().__init__(class_num=class_num, emb_size=emb_size, seq_len=seq_len, contras_t=contras_t, contras_p=contras_p, generate=generate,
                 disperse_vq_type=disperse_vq_type, distil_vq_type=distil_vq_type, vae_type=vae_type, lr=lr, unlabel=unlabel,
                 output_generate=output_generate, decoder_type=decoder_type, encoder_type=encoder_type,
                 encoder_p_type=encoder_p_type, start_unlabel_epoch=start_unlabel_epoch, start_generate_epoch=start_generate_epoch, batch_size=batch_size,
                 generate_size=generate_size, pretrain=pretrain, pretrained_epochs=pretrained_epochs,  graph_type=graph_type, discri_type=discri_type,
                 classifier_type=classifier_type, start_vq_epoch=start_vq_epoch, easy_decoder=easy_decoder, use_memory=use_memory, num_layers=num_layers, 
                 disper_num=disper_num, use_w2v_weight=use_w2v_weight, encoder_output_size=encoder_output_size, decoder_input_size=decoder_input_size,
                 max_epochs=max_epochs, **kwargs)

        # save params
        self.save_model_oss_path = f'model_output/simple_model_fix_situation/'
        
        self.detect = False
        self.use_scheduler = True
        
        self.split_str = 'is:'
        self.add_str = ' is:'
        
        self.dataset_name = 'situation'



ModelConfig = {'New_VQ': NewModel_Config_Sit,
               'TransICD': TransICD_Config,
               'Fix': NewFixModel_Config_Sit,
               'FixIdx': NewFixModel_Config_Sit,
               'FixGCN': NewFixModel_Config_Sit,
               'GAN_VQ': NewFixModel_Config_Sit,
               'FZML': FZML_Config,
               'AttnXML': AttentiveXML_Config,
               'EASYSIM': EASYSIM_Config,
               'DisVAE': DisVAE_Config,
               'Semi': NewFixModel_Config_Sit,
               'SemiLabel': NewFixModel_Config_Sit,
               'SimpleLabel':NewFixModel_Config_Sit,
               'GPT2Classifier': NewFixModel_Config_Sit,
               'BERTClassifier': NewFixModel_Config_Sit,
               'Entailment': NewFixModel_Config_Sit,
               'Other': NewFixModel_Config_Sit
               }
