import torch
from torch.utils.data import Dataset
import numpy as np


## SMILES Label Encoding
CHARISOSMISET = {"#": 29, "%": 30, ")": 31, "(": 1, "+": 32, "-": 33, "/": 34, ".": 2,
                 "1": 35, "0": 3, "3": 36, "2": 4, "5": 37, "4": 5, "7": 38, "6": 6,
                 "9": 39, "8": 7, "=": 40, "A": 41, "@": 8, "C": 42, "B": 9, "E": 43,
                 "D": 10, "G": 44, "F": 11, "I": 45, "H": 12, "K": 46, "M": 47, "L": 13,
                 "O": 48, "N": 14, "P": 15, "S": 49, "R": 16, "U": 50, "T": 17, "W": 51,
                 "V": 18, "Y": 52, "[": 53, "Z": 19, "]": 54, "\\": 20, "a": 55, "c": 56,
                 "b": 21, "e": 57, "d": 22, "g": 58, "f": 23, "i": 59, "h": 24, "m": 60,
                 "l": 25, "o": 61, "n": 26, "s": 62, "r": 27, "u": 63, "t": 28, "y": 64}
CHARISOSMILEN = 64

## ProteinSeq Label Encoding
CHARPROTSET = { "A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6,
                 "F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12,
                 "O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18,
                 "U": 19, "T": 20, "W": 21,
                 "V": 22, "Y": 23, "X": 24,
                 "Z": 25 }
CHARPROTLEN = 25

## ProteinSSE Label Encoding
CHARPROTSTRUSET = {"C": 0,"H": 1,"E": 2}
CHARPROTSTRULEN = 3

## Proteinphyche Encoding
CHARPHYCHESET = {"A":0, "C":0, "G":0, "I":0, "L":0, "M":0, "V":0, "F":0, "W":0,
                "S":1, "T":1, "N":1, "Q":1, "P":1, "U":1,
                "K":2, "R":2, "H":2,
                "D":3, "E":3,
                "F":4, "W":4, "Y":4,
                "C":5, "M":5,
                "X":6, "Z":6, "O":6}
CHARPHYCHELEN = 7


## Ligand SMILES
def label_smiles(line, smi_set, MAX_SMI_LEN=100):
    X = np.zeros(MAX_SMI_LEN,dtype=np.int64())
    for i, ch in enumerate(line[:MAX_SMI_LEN]):
        X[i] = smi_set[ch]
    return X


dataset = "Davis"
# dataset = "KIBA"
drugs = []
with open("../SMFF-DTA/Dataset/{}/ligands_can.txt".format(dataset), 'r', encoding='utf-8') as file:
    data = file.read()
    matches = eval(data)
    ## get all keys
    for key in matches.keys():
        drugs = list(matches.keys())
    
## Morgan Fingerprint 
def fingerprint(drug):
    if drug in drugs:
        file_path = f'../SMFF-DTA/Dataset/{dataset}/fingerprint/{drug}.npy'
        fp = np.load(file_path)
        fp_numpy = torch.from_numpy(fp)
    return fp_numpy

## Atom Features
def atomfeature(drug):
    if drug in drugs:
        file_path = f'../SMFF-DTA/Dataset/{dataset}/atomfeature/{drug}.npy'
        atom = np.load(file_path)
        atom_numpy = torch.from_numpy(atom)
    return atom_numpy

    
## Protein Sequence
def label_sequence(line, seq_set, MAX_SEQ_LEN=1200):
    X = np.zeros(MAX_SEQ_LEN, np.int64())
    for i, ch in enumerate(line[:MAX_SEQ_LEN]):
        X[i] = seq_set[ch]
    return X

## Protein SSE
def label_sse(line, sse_set, MAX_SSE_LEN=1200):
    X = np.zeros(MAX_SSE_LEN, np.int64())
    for i, ch in enumerate(line[:MAX_SSE_LEN]):
        X[i] = sse_set[ch]
    return X

## Protein phyche
def label_phyche(line, phyche_set, MAX_SSE_LEN=1200):
    X = np.zeros(MAX_SSE_LEN, np.int64())
    for i, ch in enumerate(line[:MAX_SSE_LEN]):
        X[i] = phyche_set[ch]
    return X


class CustomDataSet(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs

    def __getitem__(self, item):
        return self.pairs[item]

    def __len__(self):
        return len(self.pairs)
    

def collate_fn(batch_data, max_d=100, max_a=100, max_p=1200, max_fp=1024):
    N = len(batch_data)
    compound_new = torch.zeros((N, max_d), dtype=torch.long)
    compound_fp = torch.zeros((N, max_fp), dtype=torch.long)
    compound_atom = torch.zeros((N, max_a, 38), dtype=torch.float)
    protein_new = torch.zeros((N, max_p), dtype=torch.long)
    protein_sse = torch.zeros((N, max_p), dtype=torch.long)
    protein_phyche = torch.zeros((N, max_p), dtype=torch.long)
    labels_new = torch.zeros(N, dtype=torch.float)
    
    for i,pair in enumerate(batch_data):
        pair = pair.strip().split()

        drug = pair[0]
        compoundstr = pair[2]
        proteinstr = pair[3]
        label = pair[4]
        proteinSSE = pair[5]
        
        ## SMILES
        compoundint = torch.from_numpy(label_smiles(compoundstr, CHARISOSMISET, max_d))
        compound_new[i] = compoundint
        ## fingerprint
        compoundfp = fingerprint(drug)
        compound_fp[i] = compoundfp
        ## atom features
        compoundAtom = atomfeature(drug)
        compound_atom[i] = compoundAtom
        
        ## proteinSeq
        proteinint = torch.from_numpy(label_sequence(proteinstr, CHARPROTSET, max_p))
        protein_new[i] = proteinint
        ## proteinSSE
        proteinsse = torch.from_numpy(label_sse(proteinSSE, CHARPROTSTRUSET, max_p))
        protein_sse[i] = proteinsse
        ## proteinPHYCHE
        phyche = torch.from_numpy(label_phyche(proteinstr, CHARPHYCHESET, max_p))
        protein_phyche[i] = phyche
        
        labels_new[i] = np.float64(label)

        
    return (compound_new, compound_fp, compound_atom, protein_new, protein_sse, protein_phyche, labels_new)

