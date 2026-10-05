# local -> enclosing -> global -> built-in
import torch
import torch.nn as nn

p = print

# -----------

rms_norm = nn.RMSNorm([2, 3])
input = torch.randn(2, 2, 3)
output = rms_norm(input)
p(output.shape)
