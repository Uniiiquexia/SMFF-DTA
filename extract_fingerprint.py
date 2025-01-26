import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect

# dataset = "Davis"
dataset = "KIBA"

input = "../SMFF-DTA/Dataset/{}/ligands_can.txt".format(dataset)
out = "../SMFF-DTA/Dataset/{}/fingerprint_temp.txt".format(dataset)

def smiles_to_fingerprint(line, radius=2, nBits=1024):

    # Miles → molecular
    mol = Chem.MolFromSmiles(line)
    if mol is None:
        raise ValueError("Invalid SMILES string")
    # generate Morgan Fingerprint
    fingerprint = GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nBits)
    # trans to numpy
    fingerprint_array = np.zeros((1,))
    Chem.DataStructs.ConvertToNumpyArray(fingerprint, fingerprint_array)
    # trans to tensor
    fingerprint_tensor = torch.from_numpy(fingerprint_array).float().unsqueeze(0)
    return fingerprint_tensor


davis_keys = []
with open(input, 'r', encoding='utf-8') as file:
    data = file.read()
    matches = eval(data)
    for key in matches.keys():
        davis_keys = list(matches.keys())
    with open(out, 'w+') as f1:
        for key in matches.keys():
            f1.write(f"{key} {matches[key]}\n")  
            
with open(out, 'r') as f:
    lines = f.readlines()
    for line in lines:
        columns = line.strip().split()
        if len(columns) >= 2:
            first_column = columns[0]
            second_column = columns[1]
            if first_column in davis_keys:
                file_path = f'../SMFF-DTA/Dataset/{dataset}/fingerprint/{first_column}.npy'
                fingerprint = smiles_to_fingerprint(second_column)
                np.save(file_path, fingerprint.numpy())

print("results saved!")
