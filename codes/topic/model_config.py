class NewModel_Config:
    def __init__(self, class_num=8549, emb_size=768, seq_len=512, contras_t=True, contras_p=True, generate=True,
                 disperse_vq_type='Hard', distil_vq_type='Fix', vae_type='Fix_VQ_VAE', lr=0.0005, unlabel=True,
                 output_generate=True, decoder_type='GPT_EMB_EASY', encoder_type='Transformer',
                 encoder_p_type='Transformer', start_unlabel_epoch=3, start_generate_epoch=20, batch_size=32,
                 generate_size=16, pretrain=False, pretrained_epochs=30):

        # preprocess params:
        self.da = 'EDA'  # EDA, UDA, SimCSE
        self.class_num = class_num
        # 是否要把generate的句子输出
        self.output_generate = output_generate

        # training params
        self.lr = lr
        self.batch_size = batch_size
        self.generate_size = generate_size
        self.epochs = 10000

        # pretrained_decoder
        self.pretrain = pretrain
        self.pretrain_epochs = pretrained_epochs

        # save params
        self.save_model_oss_path = f'model_output/simple_model_vq_kesu/'

        # model params
        self.emb_size = emb_size  # 单个词输入的embedding size
        self.seq_len = seq_len  # transformer的最大长度
        self.dropout_rate = 0.1

        # vq vae params
        self.vae_type = vae_type
        self.distil_vq_type = distil_vq_type
        self.disperse_vq_type = disperse_vq_type

        self.disper_num = 200
        self.distill_num = 256
        self.distil_size = self.emb_size
        self.disper_size = self.emb_size
        self.decoder_type = decoder_type
        self.encoder_type = encoder_type
        self.encoder_p_type = encoder_p_type
        self.vq_coef = 1
        self.comit_coef = 0.25  # commitment loss weight
        self.other_coef = 1

        # start encoder
        # transformer params
        self.num_heads = 4
        self.forward_expansion = 4  # transformer feedforward expansion
        self.num_layers = 1  # transformer layers
        # self.pad_idx = 0

        # cnn params
        self.cnn_internel_size = 512  # internel size of cnn channel

        # descrminator
        self.classifier_type = 'RelationNet'
        self.gamma = 2

        # abletion study
        self.is_contrastive_p = contras_p
        self.is_contrastive_t = contras_t
        self.is_generate_train = generate
        self.generate_epoch = start_generate_epoch
        self.is_unlabel_train = unlabel
        self.unlabel_epoch = start_unlabel_epoch

        self.set_model_name()

    def set_model_name(self):
        # name
        arg_list = [self.distil_vq_type, self.disperse_vq_type, self.lr, self.epochs, self.batch_size,
                    self.encoder_type, self.decoder_type, self.classifier_type, self.is_contrastive_p,
                    self.is_contrastive_t, self.is_generate_train]
        self.model_name = 'model'
        for arg in arg_list:
            self.model_name = self.model_name + f'_{arg}'
        self.model_name = self.model_name + '.pt'


class NewFixModel_Config(NewModel_Config):
    def __init__(self, class_num=8549, emb_size=200, seq_len=512, contras_t=True, contras_p=True, generate=True,
                 disperse_vq_type='Hard', distil_vq_type='Fix', vae_type='Fix_VQ_VAE', lr=0.0005, unlabel=True,
                 output_generate=True, decoder_type='GPT_EMB_EASY', encoder_type='Transformer',
                 encoder_p_type='Transformer', start_unlabel_epoch=3, start_generate_epoch=20, batch_size=32,
                 generate_size=16, pretrain=False, pretrained_epochs=30, graph_type='GAT', discri_type='Linear',
                 classifier_type='Euclidean', start_vq_epoch=10, easy_decoder=False, use_memory=False, num_layers=1, 
                 disper_num=40, use_w2v_weight=True, encoder_output_size=200, decoder_input_size=32, max_epochs=1000, 
                 ablation_generate_train=True, use_new_self_training=True, use_pseudo_label=True, use_discriminator_train=True, 
                 pseudo_label_threshold=0.9, **kwargs):
        # preprocess params:
        self.da = 'EDA'  # EDA, UDA, SimCSE
        self.emb_size = emb_size  # 单个词输入的embedding size
        self.encoder_output_size = encoder_output_size
        self.decoder_input_size = decoder_input_size
        self.seq_len = seq_len  # transformer的最大长度
        self.class_num = class_num
        """
        Training setting
        """
        # pretrained_decoder
        self.pretrain = pretrain
        self.pretrain_epochs = pretrained_epochs

        # 是否要把generate的句子输出
        self.output_generate = output_generate
        self.easy_decoder = easy_decoder
        self.use_memory = use_memory
        self.detect = False
        self.use_scheduler = True

        # abletion study
        self.is_contrastive_p = contras_p
        self.is_contrastive_t = contras_t
        self.ablation_discriminator = use_discriminator_train
        self.ablation_generate_train = ablation_generate_train
        self.ablation_use_new_self_training = use_new_self_training
        self.ablation_use_pseudo_label = use_pseudo_label
        self.is_generate_train = generate
        self.generate_epoch = start_generate_epoch
        self.is_unlabel_train = unlabel
        self.unlabel_epoch = start_unlabel_epoch

        # training params
        self.lr = lr
        self.batch_size = batch_size
        self.generate_size = generate_size
        self.epochs = max_epochs
        self.dropout_rate = 0.2
        self.grad_norm = 5
        self.class_loss_decay = True  # 是否把非classifer的loss的权重下调
        self.pseudo_label_threshold = pseudo_label_threshold
        """
        Global Model setting
        """
        # save params
        self.save_model_oss_path = f'model_output/simple_model_fix_topic/'

        """
        Classifier: Euclidean
        """
        self.classifier_type = classifier_type
        """
        Discriminator Model: Linear
        used in GAN type model
        """
        self.discri_type = discri_type
        """
        Graph Model: GAT
        """
        self.graph_type = graph_type
        """
        Encoder: Transformer, Transformer_P
        """
        self.encoder_type = encoder_type
        self.encoder_p_type = encoder_p_type
        self.use_w2v_weight = use_w2v_weight
        # transformer params
        self.num_heads = 4
        self.forward_expansion = 4  # transformer feedforward expansion
        self.num_layers = num_layers  # transformer layers
        """
        Decoder: GPT, GPT_Match
        """
        self.decoder_type = decoder_type
        # cnn params
        self.cnn_internel_size = 512  # internel size of cnn channel
        """
        VAE type: VQ_VAE, VQ_VAE_Idx
        """
        self.vae_type = vae_type
        self.distil_vq_type = distil_vq_type
        self.disperse_vq_type = disperse_vq_type


        # vq parameters
        self.start_vq_epoch = start_vq_epoch
        self.decompose_number = 1 # disperse decompose number
        self.disper_num = disper_num
        self.distil_size = self.encoder_output_size
        self.disper_size = self.encoder_output_size
        """
        concrete vq vae setting
        """
        self.concrete_tau_init = 0.67
        self.concrete_tau_mode = 'fix'
        self.concrete_tau_anneal_base = 0.5
        self.concrete_tau_anneal_rate = 1e-4
        self.concrete_tau_anneal_interval = 1000
        
        self.concrete_kl_type = 'categorical'
        self.concrete_kl_prior_logits = 'uniform'
        self.concrete_kl_prior_tau = 'train'
        self.concrete_kl_beta = 1
        self.concrete_kl_fbp_threshold = 0
        self.concrete_kl_fbp_ratio = 0.6

        self.concrete_hard = 1
        """
        Other loss setting
        """
        self.gamma = 2
        self.other_coef = 1
        self.comit_coef = 0.25  # commitment loss weight
        self.vq_coef = 1  # total vq loss weight
        self.recon_coef = 1  # reconstruction loss weight
        self.classifier_coef = 1  # classifier loss weight
        self.discriminator_coef = 1  # discriminator loss weight
        self.contrastive_t_coef = 1
        self.contrastive_p_coef = 1

        self.set_model_name()
        
        self.split_str = 'news is:'
        self.add_str = ' news is:'
        
        self.dataset_name = 'topic'


class DisVAE_Config:
    def __init__(self, batch_size=30, class_num=8549, num_layers=1, seq_len=512, unlabel=True, generate=False, lr=0.001, unlabel_epoch=0, quantizer_type='concrete',
                 easy_decoder=False, use_memory=False, **kwargs):
        # preprocess params:
        self.da = 'EDA'  # EDA, UDA, SimCSE
        self.emb_size = 64  # 单个词输入的embedding size
        self.seq_len = seq_len  # transformer的最大长度
        self.class_num = class_num

        """
        Training setting
        """
        # abletion study
        self.is_generate_train = generate
        self.output_generate = True
        self.is_unlabel_train = unlabel
        self.unlabel_epoch = unlabel_epoch
        self.easy_decoder = easy_decoder
        self.use_memory = use_memory

        # training params
        self.lr = lr
        self.batch_size = batch_size
        self.generate_size = 32
        self.epochs = 100000
        self.dropout_rate = 0
        self.grad_norm = 5
        """
        Global Model setting
        """
        # save params
        self.save_model_oss_path = f'model_output/simple_model_fix_kesu/'

        # transformer params
        self.num_heads = 4
        self.forward_expansion = 4  # transformer feedforward expansion
        self.num_layers = num_layers  # transformer layers

        # vq parameters
        self.quantizer_type = quantizer_type
        self.decompose_number = 1 # disperse decompose number
        self.disper_num = 160
        self.distil_size = self.emb_size
        self.disper_size = self.emb_size
        """
        Other loss setting
        """
        self.gamma = 2
        self.other_coef = 1
        self.comit_coef = 1e-3  # commitment loss weight
        self.vq_coef = 1  # total vq loss weight
        self.recon_coef = 1  # reconstruction loss weight

        """
        concrete vq vae setting
        """
        self.concrete_tau_init = 0.67
        self.concrete_tau_mode = 'fix'
        self.concrete_tau_anneal_base = 0.5
        self.concrete_tau_anneal_rate = 1e-4
        self.concrete_tau_anneal_interval = 1000
        
        self.concrete_kl_type = 'categorical'
        self.concrete_kl_prior_logits = 'uniform'
        self.concrete_kl_prior_tau = 'train'
        self.concrete_kl_beta = 100
        self.concrete_kl_fbp_threshold = 0
        self.concrete_kl_fbp_ratio = 0.6

        self.concrete_hard = 1
        
        self.set_model_name()
        
    def set_model_name(self):
        # name
        arg_list = [self.lr, self.epochs, self.batch_size,
                    self.is_generate_train]
        self.model_name = 'model'
        for arg in arg_list:
            self.model_name = self.model_name + f'_{arg}'
        self.model_name = self.model_name + '.pt'
    

class TransICD_Config:
    def __init__(self, class_num=8549, emb_size=768, seq_len=512):
        # preprocess params:
        self.da = 'EDA'  # EDA, UDA, SimCSE
        self.class_num = class_num

        # training params
        self.lr = 0.001
        self.batch_size = 24
        self.generate_size = 8
        self.epochs = 40

        # save params
        self.save_model_oss_path = 'model_output/transicd_model_kesu/'

        # model params
        self.emb_size = emb_size  # 单个词输入的embedding size
        self.seq_len = seq_len  # transformer的最大长度
        self.dropout_rate = 0.1

        # start encoder
        # transformer params
        self.num_heads = 8
        self.forward_expansion = 4  # transformer feedforward expansion
        self.num_layers = 2  # transformer layers
        self.pad_idx = 0

        # attention params
        self.attn_expansion = 2

        # name
        self.model_name = f'transicd_{self.lr}_{self.epochs}_{self.batch_size}.pt'


class FZML_Config:
    def __init__(self, class_num=8549, emb_size=768, seq_len=512):
        # preprocess params:
        self.da = 'EDA'  # EDA, UDA, SimCSE
        self.class_num = class_num

        self.adj_matrix_oss_path = 'kesu/cut_adj_dict.pth'
        self.adj_matrix_local_path = './data/cut_adj_dict.pth'
        # training params
        self.lr = 0.001
        self.batch_size = 24
        self.generate_size = 8
        self.epochs = 40

        # save params
        self.save_model_oss_path = 'model_output/fzml_model_kesu/'

        # model params
        self.emb_size = emb_size  # 单个词输入的embedding size
        self.seq_len = seq_len  # transformer的最大长度
        self.dropout_rate = 0.1

        # start encoder
        # transformer params
        self.num_heads = 8
        self.forward_expansion = 4  # transformer feedforward expansion
        self.num_layers = 2  # transformer layers
        self.pad_idx = 0

        # attention params
        self.attn_expansion = 2

        # name
        self.model_name = f'fzml_{self.lr}_{self.epochs}_{self.batch_size}.pt'


class AttentiveXML_Config:
    def __init__(self, class_num=8549, emb_size=768, seq_len=512):
        # preprocess params:
        self.da = 'EDA'  # EDA, UDA, SimCSE
        self.class_num = class_num

        self.adj_matrix_oss_path = 'kesu/cut_adj_dict.pth'
        self.adj_matrix_local_path = './data/cut_adj_dict.pth'
        # training params
        self.lr = 0.001
        self.batch_size = 40
        self.generate_size = 8
        self.epochs = 30
        self.swa_warmup = 10

        # save params
        self.save_model_oss_path = 'model_output/attnxml_model_kesu/'

        # model params
        self.emb_size = emb_size  # 单个词输入的embedding size
        self.seq_len = seq_len  # transformer的最大长度
        self.dropout_rate = 0.5
        self.hidden_size = 256
        self.layers_num = 1
        self.linear_size = [256]

        # start encoder
        # transformer params
        self.num_heads = 8
        self.forward_expansion = 4  # transformer feedforward expansion
        self.num_layers = 2  # transformer layers
        self.pad_idx = 0

        # attention params
        self.attn_expansion = 2

        # name
        self.model_name = f'attnxml_{self.lr}_{self.epochs}_{self.batch_size}.pt'


class EASYSIM_Config():
    def __init__(self, class_num=8549, emb_size=768, seq_len=512):
        # preprocess params:
        self.da = 'EDA'  # EDA, UDA, SimCSE
        self.class_num = class_num

        self.adj_matrix_oss_path = 'kesu/cut_adj_dict.pth'
        self.adj_matrix_local_path = './data/cut_adj_dict.pth'
        # training params
        self.lr = 0.001
        self.batch_size = 40
        self.generate_size = 8
        self.epochs = 30
        self.swa_warmup = 10

        # save params
        self.save_model_oss_path = 'model_output/attnxml_model_kesu/'

        # model params
        self.emb_size = emb_size  # 单个词输入的embedding size

        # name
        self.model_name = f'attnxml_{self.lr}_{self.epochs}_{self.batch_size}.pt'


ModelConfig = {'New_VQ': NewModel_Config,
               'TransICD': TransICD_Config,
               'Fix': NewFixModel_Config,
               'FixIdx': NewFixModel_Config,
               'FixGCN': NewFixModel_Config,
               'GAN_VQ': NewFixModel_Config,
               'FZML': FZML_Config,
               'AttnXML': AttentiveXML_Config,
               'EASYSIM': EASYSIM_Config,
               'DisVAE': DisVAE_Config,
               'Semi': NewFixModel_Config,
               'SemiLabel': NewFixModel_Config,
               'SimpleLabel':NewFixModel_Config,
               'GPT2Classifier': NewFixModel_Config,
               'BERTClassifier': NewFixModel_Config,
               'Entailment': NewFixModel_Config,
               'Other': NewFixModel_Config
               }
