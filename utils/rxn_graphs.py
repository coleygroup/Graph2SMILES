import networkx as nx
from utils.chem_utils import BOND_TYPES
from rdkit import Chem
from typing import List, Tuple, Union


def get_sub_mol(mol, sub_atoms):
    new_mol = Chem.RWMol()
    atom_map = {}
    for idx in sub_atoms:
        atom = mol.GetAtomWithIdx(idx)
        atom_map[idx] = new_mol.AddAtom(atom)

    sub_atoms = set(sub_atoms)
    for idx in sub_atoms:
        a = mol.GetAtomWithIdx(idx)
        for b in a.GetNeighbors():
            if b.GetIdx() not in sub_atoms:
                continue
            bond = mol.GetBondBetweenAtoms(a.GetIdx(), b.GetIdx())
            bt = bond.GetBondType()
            if a.GetIdx() < b.GetIdx():  # each bond is enumerated twice
                new_mol.AddBond(atom_map[a.GetIdx()], atom_map[b.GetIdx()], bt)

    return new_mol.GetMol()


class RxnGraph:
    """
    RxnGraph is an abstract class for storing all elements of a reaction, like
    reactants, products and fragments. The edits associated with the reaction
    are also captured in edit labels. One can also use h_labels, which keep track
    of atoms with hydrogen changes. For reactions with multiple edits, a done
    label is also added to account for termination of edits.
    """

    def __init__(self,
                 prod_mol: Chem.Mol = None,
                 frag_mol: Chem.Mol = None,
                 reac_mol: Chem.Mol = None,
                 rxn_class: int = None) -> None:
        """
        Parameters
        ----------
        prod_mol: Chem.Mol,
            Product molecule
        frag_mol: Chem.Mol, default None
            Fragment molecule(s)
        reac_mol: Chem.Mol, default None
            Reactant molecule(s)
        rxn_class: int, default None,
            Reaction class for this reaction.
        """
        if prod_mol is not None:
            self.prod_mol = RxnElement(mol=prod_mol, rxn_class=rxn_class)
        if frag_mol is not None:
            self.frag_mol = MultiElement(mol=frag_mol, rxn_class=rxn_class)
        if reac_mol is not None:
            self.reac_mol = MultiElement(mol=reac_mol, rxn_class=rxn_class)
        self.rxn_class = rxn_class

    def get_attributes(self, mol_attrs: Tuple = ('prod_mol', 'frag_mol', 'reac_mol')) -> Tuple:
        """
        Returns the different attributes associated with the reaction graph.

        Parameters
        ----------
        mol_attrs: Tuple,
            Molecule objects to return
        """
        return tuple(getattr(self, attr) for attr in mol_attrs if hasattr(self, attr))


class RxnElement:
    """
    RxnElement is an abstract class for dealing with single molecule. The graph
    and corresponding molecule attributes are built for the molecule. The constructor
    accepts only mol objects, sidestepping the use of SMILES string which may always
    not be achievable, especially for a unkekulizable molecule.
    """

    def __init__(self, mol: Chem.Mol, rxn_class: int = None) -> None:
        """
        Parameters
        ----------
        mol: Chem.Mol,
            Molecule
        rxn_class: int, default None,
            Reaction class for this reaction.
        """
        self.mol = mol
        self.rxn_class = rxn_class
        self._build_mol()
        self._build_graph()

    def _build_mol(self) -> None:
        """Builds the molecule attributes."""
        self.num_atoms = self.mol.GetNumAtoms()
        self.num_bonds = self.mol.GetNumBonds()
        self.amap_to_idx = {atom.GetAtomMapNum(): atom.GetIdx()
                            for atom in self.mol.GetAtoms()}
        self.idx_to_amap = {value: key for key, value in self.amap_to_idx.items()}

    def _build_graph(self) -> None:
        """Builds the graph attributes."""
        self.G_undir = nx.Graph(Chem.rdmolops.GetAdjacencyMatrix(self.mol))
        self.G_dir = nx.DiGraph(Chem.rdmolops.GetAdjacencyMatrix(self.mol))

        for atom in self.mol.GetAtoms():
            self.G_undir.nodes[atom.GetIdx()]['label'] = atom.GetSymbol()
            self.G_dir.nodes[atom.GetIdx()]['label'] = atom.GetSymbol()

        for bond in self.mol.GetBonds():
            a1 = bond.GetBeginAtom().GetIdx()
            a2 = bond.GetEndAtom().GetIdx()
            btype = BOND_TYPES.index(bond.GetBondType())
            self.G_undir[a1][a2]['label'] = btype
            self.G_dir[a1][a2]['label'] = btype
            self.G_dir[a2][a1]['label'] = btype

        self.atom_scope = (0, self.num_atoms)
        self.bond_scope = (0, self.num_bonds)

    def update_atom_scope(self, offset: int) -> Union[List, Tuple]:
        """Updates the atom indices by the offset.

        Parameters
        ----------
        offset: int,
            Offset to apply
        """
        # Note that the self. reference to atom_scope is dropped to keep self.atom_scope non-dynamic
        if isinstance(self.atom_scope, list):
            atom_scope = [(st + offset, le) for st, le in self.atom_scope]
        else:
            st, le = self.atom_scope
            atom_scope = (st + offset, le)

        return atom_scope

    def update_bond_scope(self, offset: int) -> Union[List, Tuple]:
        """Updates the bond indices by the offset.

        Parameters
        ----------
        offset: int,
            Offset to apply
        """
        # Note that the self. reference to bond_scope is dropped to keep self.bond_scope non-dynamic
        if isinstance(self.bond_scope, list):
            bond_scope = [(st + offset, le) for st, le in self.bond_scope]
        else:
            st, le = self.bond_scope
            bond_scope = (st + offset, le)

        return bond_scope


class MultiElement(RxnElement):
    """
    MultiElement is an abstract class for dealing with multiple molecules. The graph
    is built with all molecules, but different molecules and their sizes are stored.
    The constructor accepts only mol objects, sidestepping the use of SMILES string
    which may always not be achievable, especially for an invalid intermediates.
    """

    def _build_graph(self) -> None:
        """Builds the graph attributes."""
        self.G_undir = nx.Graph(Chem.rdmolops.GetAdjacencyMatrix(self.mol))
        self.G_dir = nx.DiGraph(Chem.rdmolops.GetAdjacencyMatrix(self.mol))

        for atom in self.mol.GetAtoms():
            self.G_undir.nodes[atom.GetIdx()]['label'] = atom.GetSymbol()
            self.G_dir.nodes[atom.GetIdx()]['label'] = atom.GetSymbol()

        for bond in self.mol.GetBonds():
            a1 = bond.GetBeginAtom().GetIdx()
            a2 = bond.GetEndAtom().GetIdx()
            btype = BOND_TYPES.index(bond.GetBondType())
            self.G_undir[a1][a2]['label'] = btype
            self.G_dir[a1][a2]['label'] = btype
            self.G_dir[a2][a1]['label'] = btype

        frag_indices = [c for c in nx.strongly_connected_components(self.G_dir)]
        self.mols = [get_sub_mol(self.mol, sub_atoms) for sub_atoms in frag_indices]

        atom_start = 0
        bond_start = 0
        self.atom_scope = []
        self.bond_scope = []

        for mol in self.mols:
            self.atom_scope.append((atom_start, mol.GetNumAtoms()))
            self.bond_scope.append((bond_start, mol.GetNumBonds()))
            atom_start += mol.GetNumAtoms()
            bond_start += mol.GetNumBonds()
