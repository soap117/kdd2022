import torch
import torch.nn as nn
from transformers import GPT2Model
from transformers.models.gpt2 import modeling_gpt2
from pytorch_metric_learning import losses
from pytorch_metric_learning import distances
temp = GPT2Model.from_pretrained()