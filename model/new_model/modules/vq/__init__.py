from model.new_model.modules.vq.gs_vq import ConcreteQuantizer
from model.new_model.modules.vq.simple_vq import StraightForwardZ
from model.new_model.modules.vq.dvq import DVQ
from model.new_model.modules.vq.fixed_vq import FixedVectorQuantizer, FixedVectorQuantizerClassifier

VQ = {
    'GS': ConcreteQuantizer,
    'Striaght': StraightForwardZ,
    'DVQ': DVQ,
    'Fix': FixedVectorQuantizer,
    'FixClassifier': FixedVectorQuantizerClassifier
}