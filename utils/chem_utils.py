import numpy as np
from rdkit import Chem
from typing import List


# Symbols for different atoms
ATOM_LIST = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe',
             'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti',
             'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb',
             'W', 'Ru', 'Nb', 'Re', 'Te', 'Rh', 'Ta', 'Tc', 'Ba', 'Bi', 'Hf', 'Mo', 'U', 'Sm', 'Os', 'Ir',
             'Ce', 'Gd', 'Ga', 'Cs', '*', 'unk']
ATOM_DICT = {symbol: i for i, symbol in enumerate(ATOM_LIST)}

MAX_NB = 10
DEGREES = list(range(MAX_NB))
HYBRIDIZATION = [Chem.rdchem.HybridizationType.SP,
                 Chem.rdchem.HybridizationType.SP2,
                 Chem.rdchem.HybridizationType.SP3,
                 Chem.rdchem.HybridizationType.SP3D,
                 Chem.rdchem.HybridizationType.SP3D2]
HYBRIDIZATION_DICT = {hb: i for i, hb in enumerate(HYBRIDIZATION)}

FORMAL_CHARGE = [-1, -2, 1, 2, 0]
FC_DICT = {fc: i for i, fc in enumerate(FORMAL_CHARGE)}

VALENCE = [0, 1, 2, 3, 4, 5, 6]
VALENCE_DICT = {vl: i for i, vl in enumerate(VALENCE)}

NUM_Hs = [0, 1, 3, 4, 5]
NUM_Hs_DICT = {nH: i for i, nH in enumerate(NUM_Hs)}

CHIRAL_TAG = [Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
              Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
              Chem.rdchem.ChiralType.CHI_UNSPECIFIED]
CHIRAL_TAG_DICT = {ct: i for i, ct in enumerate(CHIRAL_TAG)}

RS_TAG = ["R", "S", "None"]
RS_TAG_DICT = {rs: i for i, rs in enumerate(RS_TAG)}

BOND_TYPES = [None,
              Chem.rdchem.BondType.SINGLE,
              Chem.rdchem.BondType.DOUBLE,
              Chem.rdchem.BondType.TRIPLE,
              Chem.rdchem.BondType.AROMATIC]
BOND_FLOAT_TO_TYPE = {
    0.0: BOND_TYPES[0],
    1.0: BOND_TYPES[1],
    2.0: BOND_TYPES[2],
    3.0: BOND_TYPES[3],
    1.5: BOND_TYPES[4],
}

BOND_STEREO = [Chem.rdchem.BondStereo.STEREOE,
               Chem.rdchem.BondStereo.STEREOZ,
               Chem.rdchem.BondStereo.STEREONONE]

BOND_DELTAS = {-3: 0, -2: 1, -1.5: 2, -1: 3, -0.5: 4, 0: 5, 0.5: 6, 1: 7, 1.5: 8, 2: 9, 3: 10}
BOND_FLOATS = [0.0, 1.0, 2.0, 3.0, 1.5]

RXN_CLASSES = list(range(10))

# ATOM_FDIM = len(ATOM_LIST) + len(DEGREES) + len(FORMAL_CHARGE) + len(HYBRIDIZATION) \
#             + len(VALENCE) + len(NUM_Hs) + 1
ATOM_FDIM = [len(ATOM_LIST), len(DEGREES), len(FORMAL_CHARGE), len(HYBRIDIZATION), len(VALENCE),
             len(NUM_Hs), len(CHIRAL_TAG), len(RS_TAG), 2]
# BOND_FDIM = 6
BOND_FDIM = 9
BINARY_FDIM = 5 + BOND_FDIM
INVALID_BOND = -1


def get_atom_features_sparse(atom: Chem.Atom, rxn_class: int = None, use_rxn_class: bool = False) -> List[int]:
    """Get atom features as sparse idx.

    Parameters
    ----------
    atom: Chem.Atom,
        Atom object from RDKit
    rxn_class: int, None
        Reaction class the molecule was part of
    use_rxn_class: bool, default False,
        Whether to use reaction class as additional input
    """
    feature_array = []
    symbol = atom.GetSymbol()
    symbol_id = ATOM_DICT.get(symbol, ATOM_DICT["unk"])
    feature_array.append(symbol_id)

    if symbol in ["*", "unk"]:
        padding = [999999999] * len(ATOM_FDIM) if use_rxn_class else [999999999] * (len(ATOM_FDIM) - 1)
        feature_array.extend(padding)

    else:
        degree_id = atom.GetDegree()
        if degree_id not in DEGREES:
            degree_id = 9
        formal_charge_id = FC_DICT.get(atom.GetFormalCharge(), 4)
        hybridization_id = HYBRIDIZATION_DICT.get(atom.GetHybridization(), 4)
        valence_id = VALENCE_DICT.get(atom.GetTotalValence(), 6)
        num_h_id = NUM_Hs_DICT.get(atom.GetTotalNumHs(), 4)
        chiral_tag_id = CHIRAL_TAG_DICT.get(atom.GetChiralTag(), 2)

        rs_tag = atom.GetPropsAsDict().get("_CIPCode", "None")
        rs_tag_id = RS_TAG_DICT.get(rs_tag, 2)

        is_aromatic = int(atom.GetIsAromatic())
        feature_array.extend([degree_id, formal_charge_id, hybridization_id,
                              valence_id, num_h_id, chiral_tag_id, rs_tag_id, is_aromatic])

        if use_rxn_class:
            feature_array.append(rxn_class)

    return feature_array


def get_bond_features(bond: Chem.Bond) -> List[int]:
    """Get bond features.

    Parameters
    ----------
    bond: Chem.Bond,
        bond object
    """
    bt = bond.GetBondType()
    bond_features = [int(bt == bond_type) for bond_type in BOND_TYPES[1:]]
    bs = bond.GetStereo()
    bond_features.extend([int(bs == bond_stereo) for bond_stereo in BOND_STEREO])
    bond_features.extend([int(bond.GetIsConjugated()), int(bond.IsInRing())])

    return bond_features
