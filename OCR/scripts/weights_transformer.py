# Weights files contain not only the model's state, but also states for optimizer,
# scheduler and some other training paramaters. We don't need if for the
# submissin, so we can leave only the model's state.

import torch

PATHTOLOAD = "/home/toefl/K/nto/checkpoints/optical_distortion/model-54-3.9064.ckpt"
PATHTOSAVE = "/home/toefl/K/sub/rus_model.ckpt"

model = torch.load(PATHTOLOAD, map_location="cuda:0")["model"]
torch.save({"model": model}, PATHTOSAVE)