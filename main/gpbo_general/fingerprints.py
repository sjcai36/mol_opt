""" Code for molecular fingerprints """

import functools

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit.DataStructs.cDataStructs import ConvertToNumpyArray

standard_fingerprint = functools.partial(
    rdMolDescriptors.GetMorganFingerprintAsBitVect, radius=2, nBits=1024
)


def _fp_to_array(fp):
    fp_arr = np.zeros((1,), dtype=np.int8)
    ConvertToNumpyArray(fp, fp_arr)
    return fp_arr


def smiles_to_fp_array(smiles: str, fingerprint_func: callable = None) -> np.array:
    """Convert individual SMILES into a 1D fingerprint array"""
    if fingerprint_func is None:
        fingerprint_func = standard_fingerprint
    mol = Chem.MolFromSmiles(smiles)

    # return fingerprint only if valid smiles
    if mol:
        fp = fingerprint_func(mol)
        return _fp_to_array(fp).flatten()


def smiles_to_fingerprint_stack(smiles_list, config, dtype=None):
    # Returns stacked np fingerprint representations of valid smiles and index values of invalid smiles

    fingerprint_func = functools.partial(
        rdMolDescriptors.GetMorganFingerprintAsBitVect,
        radius=config["fp_radius"],
        nBits=config["fp_nbits"],
    )
    smiles_to_np_fingerprint = functools.partial(
        smiles_to_fp_array, fingerprint_func=fingerprint_func
    )

    fp_stack = []
    invalid_idx = []

    for i, s in enumerate(smiles_list):
        fp = smiles_to_np_fingerprint(s)
        if fp is not None:
            fp_stack.append(fp)
        else:
            invalid_idx.append(i)

    fp_stack = np.stack(fp_stack)
    if dtype:
        fp_stack = fp_stack.astype(dtype)

    return fp_stack, invalid_idx
