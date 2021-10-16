import torch
import torch.nn as nn
from models.model_utils import index_select_ND
from typing import Tuple


class DGATGRU(nn.Module):
    """GRU Message Passing layer."""
    def __init__(self, args, input_size: int, h_size: int, depth: int):
        super().__init__()
        self.args = args

        self.input_size = input_size
        self.h_size = h_size
        self.depth = depth

        self._build_layer_components()
        self._build_attention()

    def _build_layer_components(self) -> None:
        """Build layer components."""
        self.W_z = nn.Linear(self.input_size + self.h_size, self.h_size)
        self.W_r = nn.Linear(self.input_size, self.h_size, bias=False)
        self.U_r = nn.Linear(self.h_size, self.h_size)
        self.W_h = nn.Linear(self.input_size + self.h_size, self.h_size)

    def _build_attention(self) -> None:
        self.leaky_relu = nn.LeakyReLU()
        self.head_count = self.args.encoder_attn_heads
        self.dim_per_head = self.h_size // self.head_count

        self.attn_alpha = nn.Parameter(
            torch.Tensor(1, 1, self.head_count, 2 * self.dim_per_head), requires_grad=True)
        self.attn_bias = nn.Parameter(
            torch.Tensor(1, 1, self.head_count), requires_grad=True)

        self.attn_W_q = nn.Linear(self.input_size, self.h_size, bias=True)
        self.attn_W_k = nn.Linear(self.h_size, self.h_size, bias=True)
        self.attn_W_v = nn.Linear(self.h_size, self.h_size, bias=True)

        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(self.args.dropout)
        self.attn_dropout = nn.Dropout(self.args.attn_dropout)

    def GRU(self, x: torch.Tensor, h_nei: torch.Tensor) -> torch.Tensor:
        """Implements the GRU gating equations.

        Parameters
        ----------
            x: torch.Tensor, input tensor
            h_nei: torch.Tensor, hidden states of the neighbors
        """
        # attention-based aggregation
        n_node, max_nn, h_size = h_nei.size()
        head_count = self.head_count
        dim_per_head = self.dim_per_head

        q = self.attn_W_q(x)                            # (n_node, input) -> (n_node, h)
        q = q.unsqueeze(1).repeat(1, max_nn, 1)         # -> (n_node, max_nn, h)
        q = q.reshape(
            n_node, max_nn, head_count, dim_per_head)   # -> (n_node, max_nn, head, h/head)

        k = self.attn_W_k(h_nei)                        # (n_node, max_nn, h)
        k = k.reshape(
            n_node, max_nn, head_count, dim_per_head)   # -> (n_node, max_nn, head, h/head)

        v = self.attn_W_v(h_nei)                        # (n_node, max_nn, h)
        v = v.reshape(
            n_node, max_nn, head_count, dim_per_head)   # -> (n_node, max_nn, head, h/head)

        qk = torch.cat([q, k], dim=-1)                  # -> (n_node, max_nn, head, 2*h/head)
        qk = self.leaky_relu(qk)

        attn_score = qk * self.attn_alpha               # (n_node, max_nn, head, 2*h/head)
        attn_score = torch.sum(attn_score, dim=-1)      # (n_node, max_nn, head, 2*h/head) -> (n_node, max_nn, head)
        attn_score = attn_score + self.attn_bias        # (n_node, max_nn, head)

        attn_mask = (h_nei.sum(dim=2) == 0
                     ).unsqueeze(2)                     # (n_node, max_nn, h) -> (n_node, max_nn, 1)
        attn_score = attn_score.masked_fill(attn_mask, -1e18)

        attn_weight = self.softmax(attn_score)          # (n_node, max_nn, head), softmax over dim=1
        attn_weight = attn_weight.unsqueeze(3)          # -> (n_node, max_nn, head, 1)

        attn_context = attn_weight * v                  # -> (n_node, max_nn, head, h/head)
        attn_context = attn_context.reshape(
            n_node, max_nn, h_size)                     # -> (n_node, max_nn, h)

        sum_h = attn_context.sum(dim=1)                 # -> (n_node, h)

        # GRU
        z_input = torch.cat([x, sum_h], dim=1)          # x = [x_u; x_uv]
        z = torch.sigmoid(self.W_z(z_input))            # (10)

        r_1 = self.W_r(x)                               # (n_node, h) -> (n_node, h)
        r_2 = self.U_r(sum_h)                           # (n_node, h) -> (n_node, h)
        r = torch.sigmoid(r_1 + r_2)                    # (11) r_ku = f_r(x; m_ku) = W_r(x) + U_r(m_ku)

        sum_gated_h = r * sum_h                         # (n_node, h)
        h_input = torch.cat([x, sum_gated_h], dim=1)
        pre_h = torch.tanh(self.W_h(h_input))           # (13)
        new_h = (1.0 - z) * sum_h + z * pre_h           # (14)

        return new_h

    def forward(self, fmess: torch.Tensor, bgraph: torch.Tensor) -> torch.Tensor:
        """Forward pass of the RNN

        Parameters
        ----------
            fmess: torch.Tensor, contains the initial features passed as messages
            bgraph: torch.Tensor, bond graph tensor. Contains who passes messages to whom.
        """
        h = torch.zeros(fmess.size()[0], self.h_size, device=fmess.device)
        mask = torch.ones(h.size()[0], 1, device=h.device)
        mask[0, 0] = 0      # first message is padding

        for i in range(self.depth):
            h_nei = index_select_ND(h, 0, bgraph)
            h = self.GRU(fmess, h_nei)
            h = h * mask
        return h


class DGATEncoder(nn.Module):
    """MessagePassing Network based encoder. Messages are updated using an RNN
    and the final message is used to update atom embeddings."""
    def __init__(self, args, input_size: int, node_fdim: int):
        super().__init__()
        self.args = args

        self.h_size = args.encoder_hidden_size
        self.depth = args.encoder_num_layers
        self.input_size = input_size
        self.node_fdim = node_fdim
        self.head_count = args.encoder_attn_heads
        self.dim_per_head = self.h_size // self.head_count

        self.leaky_relu = nn.LeakyReLU()

        self._build_layers()
        self._build_attention()

    def _build_layers(self) -> None:
        """Build layers associated with the MPNEncoder."""
        self.W_o = nn.Sequential(nn.Linear(self.node_fdim + self.h_size, self.h_size), nn.GELU())
        self.rnn = DGATGRU(self.args, self.input_size, self.h_size, self.depth)

    def _build_attention(self) -> None:
        self.attn_alpha = nn.Parameter(
            torch.Tensor(1, 1, self.head_count, 2 * self.dim_per_head), requires_grad=True)
        self.attn_bias = nn.Parameter(
            torch.Tensor(1, 1, self.head_count), requires_grad=True)

        self.attn_W_q = nn.Linear(self.node_fdim, self.h_size, bias=True)
        self.attn_W_k = nn.Linear(self.h_size, self.h_size, bias=True)
        self.attn_W_v = nn.Linear(self.h_size, self.h_size, bias=True)

        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(self.args.dropout)
        self.attn_dropout = nn.Dropout(self.args.attn_dropout)

    def forward(self, fnode: torch.Tensor, fmess: torch.Tensor,
                agraph: torch.Tensor, bgraph: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Forward pass of the MPNEncoder.

        Parameters
        ----------
            fnode: torch.Tensor, node feature tensor
            fmess: torch.Tensor, message features
            agraph: torch.Tensor, neighborhood of an atom
            bgraph: torch.Tensor, neighborhood of a bond,
                except the directed bond from the destination node to the source node
            mask: torch.Tensor, masks on nodes
        """
        h = self.rnn(fmess, bgraph)
        nei_message = index_select_ND(h, 0, agraph)

        # attention-based aggregation
        n_node, max_nn, h_size = nei_message.size()
        head_count = self.head_count
        dim_per_head = self.dim_per_head

        q = self.attn_W_q(fnode)                        # (n_node, h)
        q = q.unsqueeze(1).repeat(1, max_nn, 1)         # -> (n_node, max_nn, h)
        q = q.reshape(
            n_node, max_nn, head_count, dim_per_head)   # (n_node, max_nn, h) -> (n_node, max_nn, head, h/head)

        k = self.attn_W_k(nei_message)                  # (n_node, max_nn, h)
        k = k.reshape(
            n_node, max_nn, head_count, dim_per_head)   # -> (n_node, max_nn, head, h/head)

        v = self.attn_W_v(nei_message)                  # (n_node, max_nn, h)
        v = v.reshape(
            n_node, max_nn, head_count, dim_per_head)   # -> (n_node, max_nn, head, h/head)

        qk = torch.cat([q, k], dim=-1)                  # -> (n_node, max_nn, head, 2*h/head)
        qk = self.leaky_relu(qk)

        attn_score = qk * self.attn_alpha               # (n_node, max_nn, head, 2*h/head)
        attn_score = torch.sum(attn_score, dim=-1)      # (n_node, max_nn, head, 2*h/head) -> (n_node, max_nn, head)
        attn_score = attn_score + self.attn_bias        # (n_node, max_nn, head)

        attn_mask = (nei_message.sum(dim=2) == 0
                     ).unsqueeze(2)                     # (n_node, max_nn, h) -> (n_node, max_nn, 1)
        attn_score = attn_score.masked_fill(attn_mask, -1e18)

        attn_weight = self.softmax(attn_score)          # (n_node, max_nn, head), softmax over dim=1
        attn_weight = attn_weight.unsqueeze(3)          # -> (n_node, max_nn, head, 1)

        attn_context = attn_weight * v                  # -> (n_node, max_nn, head, h/head)
        attn_context = attn_context.reshape(
            n_node, max_nn, h_size)                     # -> (n_node, max_nn, h)

        nei_message = attn_context.sum(dim=1)           # -> (n_node, h)

        # readout
        node_hiddens = torch.cat([fnode, nei_message], dim=1)
        node_hiddens = self.W_o(node_hiddens)

        if mask is None:
            mask = torch.ones(node_hiddens.size(0), 1, device=fnode.device)
            mask[0, 0] = 0      # first node is padding

        return node_hiddens * mask, h
