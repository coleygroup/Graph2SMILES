import torch
import torch.nn as nn
from models.model_utils import index_select_ND
from typing import Tuple


class DGCNGRU(nn.Module):
    """GRU Message Passing layer."""
    def __init__(self, args, input_size: int, h_size: int, depth: int):
        super().__init__()
        self.args = args

        self.input_size = input_size
        self.h_size = h_size
        self.depth = depth

        self._build_layer_components()

    def _build_layer_components(self) -> None:
        """Build layer components."""
        self.W_z = nn.Linear(self.input_size + self.h_size, self.h_size)
        self.W_r = nn.Linear(self.input_size, self.h_size, bias=False)
        self.U_r = nn.Linear(self.h_size, self.h_size)
        self.W_h = nn.Linear(self.input_size + self.h_size, self.h_size)

    def GRU(self, x: torch.Tensor, h_nei: torch.Tensor) -> torch.Tensor:
        """Implements the GRU gating equations.

        Parameters
        ----------
            x: torch.Tensor, input tensor
            h_nei: torch.Tensor, hidden states of the neighbors
        """
        sum_h = h_nei.sum(dim=1)                        # (9)
        z_input = torch.cat([x, sum_h], dim=1)          # x = [x_u; x_uv]
        z = torch.sigmoid(self.W_z(z_input))            # (10)

        r_1 = self.W_r(x).view(-1, 1, self.h_size)
        r_2 = self.U_r(h_nei)
        r = torch.sigmoid(r_1 + r_2)                    # (11) r_ku = f_r(x; m_ku) = W_r(x) + U_r(m_ku)

        gated_h = r * h_nei
        sum_gated_h = gated_h.sum(dim=1)                # (12)
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


class DGCNEncoder(nn.Module):
    """MessagePassing Network based encoder. Messages are updated using an RNN
    and the final message is used to update atom embeddings."""
    def __init__(self, args, input_size: int, node_fdim: int):
        super().__init__()
        self.args = args

        self.h_size = args.encoder_hidden_size
        self.depth = args.encoder_num_layers
        self.input_size = input_size
        self.node_fdim = node_fdim

        self._build_layers()

    def _build_layers(self) -> None:
        """Build layers associated with the MPNEncoder."""
        self.W_o = nn.Sequential(nn.Linear(self.node_fdim + self.h_size, self.h_size), nn.GELU())
        self.rnn = DGCNGRU(self.args, self.input_size, self.h_size, self.depth)

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
        nei_message = nei_message.sum(dim=1)
        node_hiddens = torch.cat([fnode, nei_message], dim=1)
        node_hiddens = self.W_o(node_hiddens)

        if mask is None:
            mask = torch.ones(node_hiddens.size(0), 1, device=fnode.device)
            mask[0, 0] = 0      # first node is padding

        return node_hiddens * mask, h
