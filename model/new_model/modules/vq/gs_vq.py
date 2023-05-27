import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import OneHotCategorical, kl_divergence
import numpy as np

# from torch.distributions import RelaxedOneHotCategorical
from torch.distributions.relaxed_categorical import RelaxedOneHotCategorical
from model.new_model.modules.simple_modules import one_hot_argmax, PackedSequneceUtil


class AnnealingTemperature(object):
    """
    tau is updated every N step
    tau = max(base_tau, init_tau * exp (anneal_rate * step))
    """

    def __init__(self, init_tau=1.0, base_tau=0.5, anneal_rate=1e-3, N=500):

        self.init_tau = init_tau
        self.base_tau = base_tau
        self.anneal_rate = anneal_rate
        self.N = N

        self._tau = init_tau
        self._step = 0

    def step(self):
        self._step = self._step + 1
        if self._step % self.N == 0:
            self._tau = np.maximum(
                self.init_tau * np.exp(-self.anneal_rate * self._step), self.base_tau
            )

        return self._tau


# TODO: if train tau_p: min value > 0, init
class KLConcrete(nn.Module):
    """
    calculate kl (q||p)
        - posterior q (z|x)
        - prior p (z)
    careful that tau_p should be positive
    """

    def __init__(
        self,
        K,  # number of classes
        M,  # number of splits
        kl_type="categorical",  # 'categorical', 'relaxed'
        logits_p="train",  # 'train', 'uniform'
        tau_p=1.0,  # 'train', positive float value
    ):
        super().__init__()

        l = torch.ones(M, K)
        # torch.nn.init.orthogonal_(l)
        if logits_p == "uniform":
            self.register_buffer('logits_p', l)
        elif logits_p == "train":
            self.logits_p = nn.parameter.Parameter(l)

        self.kl_type = kl_type
        t = torch.ones(M, 1)
        if kl_type == "relaxed":
            if tau_p == "train":
                # torch.nn.init.uniform_(t, a=0.1, b=10)
                self.tau_p = nn.parameter.Parameter(t)
            else:
                assert type(tau_p) in [int, float] and tau_p > 0
                tau_p = t * tau_p
                self.register_buffer('tau_p', tau_p)

    def forward(self, q, z, logits_q):
        """
        q: relaxed catergorical posterior
        z: bsz × M × K, sample from q
        logits_q: bsz × M × K
        """
        if self.kl_type == "categorical":
            kl = self.kl_categorical(logits_q)
        elif self.kl_type == "relaxed":
            kl = self.kl_relaxed(q, z)
        else:
            raise KeyError
        return kl

    def kl_relaxed(self, q, z):
        """
        Monte Carlo KL with relaxed prior
        it is important to have not too small temperature for q, otherwise log_prob produces nan
        0.01 is very dangerous, 0.1 sometimes nan, 0.3 generally safe
        small temperature is dangerous for q but fine for p
        """

        self.tau_p = self.tau_p.data.clamp(min=1e-2, max=10.0)
        logits = self.logits_p.expand_as(z)
        p = RelaxedOneHotCategorical(
            logits=logits, temperature=self.tau_p
        )
        # single sample
        KL_qp = q.log_prob(z) - p.log_prob(z)

        # Raise exception if contains nan
        if torch.isnan(KL_qp).any():
            nan_mask = torch.isnan(KL_qp)
            KL_qp = torch.where(torch.isnan(KL_qp), torch.zeros_like(KL_qp), KL_qp)
            raise RuntimeError("nan log probibility for relaxed kl")

        return KL_qp

    def kl_categorical(self, logits_q):
        # Analytical KL with categorical prior
        logits = self.logits_p.expand_as(logits_q)
        p_cat = OneHotCategorical(logits=logits)
        q_cat = OneHotCategorical(logits=logits_q)
        KL_qp = kl_divergence(q_cat, p_cat)
        return KL_qp


# TODO: if train tau: min value > 0, init
class ConcreteRelaxation(nn.Module):
    """
    z: latent sampled from posterior q (z|x)
    """

    def __init__(
        self,
        hard=True,  # True - Straight Through Gumbel, False - GumbelSoftmax
        tau_mode="fix",  # 'fix', 'anneal', 'train'
        # for q temperature
        init_tau=1.0,  # for anneal, fix
        base_tau=0.5,  # for anneal
        anneal_rate=1e-4,  # for anneal
        update_tau_every=1000,  # for anneal
        # for KL
        K=10,
        M=2,
        kl_type="categorical",  # 'relaxed', 'categorical'
        logits_p="train",  # 'train', 'uniform'
        tau_p=1.0,  # 'train', fixed positive float
    ):
        super().__init__()

        self.hard = hard
        if hard:
            assert kl_type == "categorical"

        self.KL = KLConcrete(K=K, M=M, kl_type=kl_type, logits_p=logits_p, tau_p=tau_p)

        self.tau_mode = tau_mode
        t = torch.ones(M, 1)
        if tau_mode == "train":
            # torch.nn.init.uniform_(t, a=0.5, b=10)
            # t.fill_(1.0)
            t = t * init_tau
            self.tau = nn.parameter.Parameter(t)
        else:
            assert type(init_tau) in [int, float] and init_tau > 0
            tau = t * init_tau
            self.register_buffer('tau', tau)
            if tau_mode == "fix":
                pass
            elif tau_mode == "anneal":
                self.tau_scheduler = AnnealingTemperature(
                    init_tau=init_tau,
                    base_tau=base_tau,
                    anneal_rate=anneal_rate,
                    N=update_tau_every,
                )

    def forward(self, logits_q):
        """
        Draw sample and calculate KL
        logits_q: bsz × M × K
        """

        self.tau.data = self.tau.data.clamp(min=3e-1, max=10.0)
        q = RelaxedOneHotCategorical(logits=logits_q, temperature=self.tau)
        # draw soft sample with reparameterization, preserve gradient
        z = q.sample()
        # cast into hard sample
        if self.hard:
            z_hard = one_hot_argmax(z)
            # straight through gradient, hard sample has same gradient as soft sample
            z = (z_hard - z).detach() + z

        # calculate kl
        kl = self.KL(q, z, logits_q)

        # update tau during training, CAUTION: only call forward once within a batch
        # do not update if eval
        if self.tau_mode == "anneal" and self.training:
            tau_value = self.tau_scheduler.step()
            # self.tau = self.tau.fill_(tau_value)
            self.tau = torch.ones(self.tau.shape).to(self.tau.device) * tau_value

        return {"z": z, "kl": kl}

    def check_paramater(self, config):
        tau_q = self.tau.data.squeeze()
        print("\ntau_q mode: ", config.concrete_tau_mode)
        print("tau_q: ", tau_q)

        logits_p = self.KL.logits_p
        print("\nlogits_p mode: ", config.concrete_kl_prior_logits)
        print("probs_p: ", torch.softmax(logits_p, dim=-1)[0])

        if hasattr(self.KL, "tau_p"):
            tau_p = self.KL.tau_p.data.squeeze()
            print("\ntau_p mode: ", config.concrete_kl_prior_tau)
            print("tau_p: ", tau_p)


class ConcreteQuantizer(nn.Module):
    """
    Continuous relaxation of categorical distrubution (the two are equivalent)
        Concrete distribution: https://arxiv.org/pdf/1611.00712.pdf
        Gumbel-Softmax distribution: https://arxiv.org/abs/1611.01144
    """

    def __init__(self, num_embeddings, embedding_dim, config, **kwargs):
        super().__init__()

        self.K = num_embeddings
        self.D = embedding_dim
        self.M = config.decompose_number
        print('K', 'D', 'M', self.K, self.D, self.M)

        self.concrete = ConcreteRelaxation(
            hard=(config.concrete_hard == 1),
            tau_mode=config.concrete_tau_mode,
            init_tau=config.concrete_tau_init,
            base_tau=config.concrete_tau_anneal_base,
            anneal_rate=config.concrete_tau_anneal_rate,
            update_tau_every=config.concrete_tau_anneal_interval,
            K=self.K,
            M=self.M,
            kl_type=config.concrete_kl_type,
            logits_p=config.concrete_kl_prior_logits,
            tau_p=config.concrete_kl_prior_tau,
        )

        self.embeddings = nn.parameter.Parameter(torch.randn(self.M, self.K, self.D))

        # set to true when classification
        # freeze this part, but train modules on top
        self.force_eval = False

    def quantize_embedding(self, indice: torch.Tensor) -> torch.Tensor:
        """return the quantized vector by input indices

        Args:
            indice (torch.Tensor): [B, M] M is the decompose number

        Returns:
            torch.Tensor: [B, M, E], quantize vector
        """     
        result_list = []
        for i in range(indice.shape[1]):
            quantized = self.embeddings[i, indice[:, i], :]
            result_list.append(quantized.unsqueeze(1))
        return torch.cat(result_list, dim=1)

    def forward(self, inputs, **kwargs):
        """
        Case 1: sequence of vector
        Arg: logits - bsz x T x (M * K), tensor
                    - or N x (M * K), packed sequence
        Return: quantized - bsz x T x (M * D)
        Case 2: single vector
        Arg: logits - bsz x (M * K), tensor
        Return: quantized - bsz x (M * D)
        """
        # support PackedSequence
        packed_seq_util = PackedSequneceUtil()
        logits = packed_seq_util.preprocess(inputs)
        # reshape logits: B_flatten x M x K
        #       with B_flatten = bsz, or bsz * T, or N
        bsz = logits.shape[0]
        assert logits.shape[-1] == self.M * self.K
        logits = logits.view(-1, self.M, self.K)

        # z: B_flatten x M x K
        if self.training and not self.force_eval:
            # use concrete when train
            out = self.concrete(logits)
            z = out["z"]
            kl = out["kl"].sum()
            # z = one_hot_argmax(logits)
            # kl = 0.0
            # out = None
        else:
            # simple argmax when eval
            z = one_hot_argmax(logits)
            kl = 0.0
            out = None

        # B_flatten x M x D
        quantized_stack = z.transpose(0, 1).bmm(self.embeddings).transpose(0, 1)
        # if prob is not one-hot, this is not exact index
        encoding_indices = torch.argmax(z, dim=-1)

        if packed_seq_util.is_packed:
            quantized_stack = packed_seq_util.postprocess(quantized_stack, pad=0.0)
            quantized = quantized_stack.view(
                [*quantized_stack.shape[:-2]] + [self.M * self.D]
            )
            encoding_indices = packed_seq_util.postprocess(encoding_indices, pad=-1)
            z = packed_seq_util.postprocess(z, pad=0.0)
        else:
            quantized_stack = quantized_stack.view(bsz, -1, self.M, self.D).squeeze(1)
            quantized = quantized_stack.reshape(bsz, -1, self.M * self.D).squeeze(1)
            encoding_indices = encoding_indices.view(bsz, -1, self.M).squeeze(1)
            z = z.view(bsz, -1, self.M, self.K).squeeze(1)


        return {
            # B x T (optional) x (M * D)
            "quantized": quantized,
            # B x T (optional) x M x D
            "quantized_stack": quantized_stack,
            # B x T (optional) x M
            "encoding_indices": encoding_indices,
            # kl sum
            "loss": kl,
        }