import torch
import torch.nn as nn
from collections import OrderedDict

class CoOp(nn.Module):
	def __init__(self, emb_size, tokenizer, encoder, device):
		super().__init__()
		self.device = device
		ctx_init = "A photo of a"
		text = tokenizer(ctx_init, return_tensors="pt", padding=True).to(self.device)
		embedding = encoder(
			text.input_ids,
			attention_mask=text.attention_mask,
			return_dict=True,
			mode="text",
		)
		ctx_vectors = embedding[0][0, 1:, :]

		self.ctx = nn.Parameter(ctx_vectors)
		self.meta_net = nn.Sequential(OrderedDict([
			("linear1", nn.Linear(emb_size, emb_size // 16)),
			("relu", nn.ReLU(inplace=True)),
			("linear2", nn.Linear(emb_size // 16, emb_size))
		]))

	def forward(self, im_features):
		ctx = self.ctx  # (n_ctx, ctx_dim)
		bias = self.meta_net(im_features)  # (batch, ctx_dim)
		bias = bias.unsqueeze(1)  # (batch, 1, ctx_dim)
		ctx = ctx.unsqueeze(0)  # (1, n_ctx, ctx_dim)
		ctx_shifted = ctx + bias  # (batch, n_ctx, ctx_dim)
		return ctx_shifted