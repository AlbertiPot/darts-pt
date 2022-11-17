import torch

path = '/data/gbc/Workspace/darts-pt/rdarts_ckpt/search_rdarts_c10_r6p8/search-c10_v1cls_maskr0.6p8c10seed2-20220607-074706/weights.pt'

pth = torch.load(path)
keys = list(pth.keys())
print(len(keys))

rec_keys= [item for item in keys if item if 'rec_decoder' in item]
print(rec_keys)

for item in rec_keys:
    pth.pop(item)

keys = list(pth.keys())
print(len(keys))
print(keys)

print(pth)

# ['rec_decoder.inter_conv.0.weight', 
# 'rec_decoder.inter_conv.0.bias', 
# 'rec_decoder.inter_conv.1.weight', 
# 'rec_decoder.inter_conv.1.bias', 
# 'rec_decoder.decoder.0.weight', 
# 'rec_decoder.decoder.0.bias', 
# 'rec_decoder.decoder.1.weight', 
# 'rec_decoder.decoder.1.bias', 
# 'rec_decoder.decoder.1.running_mean', 
# 'rec_decoder.decoder.1.running_var', 
# 'rec_decoder.decoder.1.num_batches_tracked', 
# 'rec_decoder.decoder.3.weight', 
# 'rec_decoder.decoder.3.bias', 
# 'rec_decoder.decoder.4.weight', 
# 'rec_decoder.decoder.4.bias', 
# 'rec_decoder.decoder.4.running_mean', 
# 'rec_decoder.decoder.4.running_var', 
# 'rec_decoder.decoder.4.num_batches_tracked', 
# 'rec_decoder.decoder.6.weight', 
# 'rec_decoder.decoder.6.bias', 
# 'rec_decoder.decoder.7.weight', 
# 'rec_decoder.decoder.7.bias', 
# 'rec_decoder.decoder.7.running_mean', 
# 'rec_decoder.decoder.7.running_var', 
# 'rec_decoder.decoder.7.num_batches_tracked', 
# 'rec_decoder.decoder.9.weight', 
# 'rec_decoder.decoder.9.bias', 
# 'rec_decoder.decoder.10.weight', 
# 'rec_decoder.decoder.10.bias', 
# 'rec_decoder.decoder.10.running_mean', 
# 'rec_decoder.decoder.10.running_var', 
# 'rec_decoder.decoder.10.num_batches_tracked', 
# 'rec_decoder.decoder.12.weight', 
# 'rec_decoder.decoder.12.bias']