import torch
from algos.ssl.simclr import SimCLR
import copy

# Assume models stored in models/...
path='models/model_10epochs.tar'
model=torch.load(path)
hps=model['hps']

# Correct path to itself
hps.load_checkpoint=path
simclr=SimCLR(torch.device("cuda:0"),hps)

dimensionRep=simclr.model.tv_backbone_inner_fc_dim

input=torch.zeros(1,10,120,120).to('cuda:0')

print(dimensionRep)

torch.onnx.export(simclr.model.backbone,args=torch.zeros(1,10,120,120).to('cuda:0'),f='ssl_backbone.onnx',verbose=False,input_names=['input_sentinel2_10_bands_120'],output_names=['representation_'+str(dimensionRep)],dynamic_axes={"input_sentinel2_10_bands_120":{0:'batch_size'},'representation_'+str(dimensionRep):{0:'batch_size'}})