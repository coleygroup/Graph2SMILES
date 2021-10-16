import torch
import torch.nn as nn
from models.dgat import DGATEncoder
from models.dgcn import DGCNEncoder
from typing import Tuple
from utils.data_utils import G2SBatch


class GraphFeatEncoder(nn.Module):
    """
    GraphFeatEncoder encodes molecules by using features of atoms and bonds,
    instead of a vocabulary, which is used for generation tasks.
    Adapted from Somnath et al. (2020): https://grlplus.github.io/papers/61.pdf
    """

    def __init__(self, args, n_atom_feat: int, n_bond_feat: int):
        super().__init__()
        self.args = args

        self.n_atom_feat = n_atom_feat
        self.n_bond_feat = n_bond_feat

        if args.mpn_type == "dgcn":
            MPNClass = DGCNEncoder
        elif args.mpn_type == "dgat":
            MPNClass = DGATEncoder
        else:
            raise NotImplemented(f"Unsupported mpn_type: {args.mpn_type}!")

        self.mpn = MPNClass(
            args,
            input_size=n_atom_feat + n_bond_feat,
            node_fdim=n_atom_feat
        )

    def forward(self, reaction_batch: G2SBatch) -> Tuple[torch.Tensor, None]:
        """
        Forward pass of the graph encoder. First the feature vectors are extracted,
        and then encoded. This has been modified to pass data via the G2SBatch datatype
        """
        fnode = reaction_batch.fnode
        fmess = reaction_batch.fmess
        agraph = reaction_batch.agraph
        bgraph = reaction_batch.bgraph

        # embed graph, note that for directed graph, fess[any, 0:2] = u, v
        hnode = fnode.clone()
        fmess1 = hnode.index_select(index=fmess[:, 0].long(), dim=0)
        fmess2 = fmess[:, 2:].clone()
        hmess = torch.cat([fmess1, fmess2], dim=-1)         # hmess = x = [x_u; x_uv]

        # encode
        hatom, _ = self.mpn(hnode, hmess, agraph, bgraph, mask=None)
        hmol = None

        return hatom, hmol
