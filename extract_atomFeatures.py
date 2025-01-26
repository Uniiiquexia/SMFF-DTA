import numpy as np
import re
from rdkit import Chem

atomic_num_list = [1, 6, 7, 8, 9, 15, 16, 17, 35, 53]   # 常见的原子序数
degree_list = [0, 1, 2, 3, 4, 5]                        # 键的数目
num_Hs_list = [0, 1, 2, 3, 4]                           # 氢原子数目
implicit_valence_list = [0, 1, 2, 3, 4, 5]              # 隐式价电子数
formal_charge_list = [-1, 0, 1]                         # 形式电荷
is_aromatic_list = [0, 1]                               # 是否芳香族
hybridization_list = [Chem.rdchem.HybridizationType.S, Chem.rdchem.HybridizationType.SP, 
                      Chem.rdchem.HybridizationType.SP2, Chem.rdchem.HybridizationType.SP3, 
                      Chem.rdchem.HybridizationType.SP3D, Chem.rdchem.HybridizationType.SP3D2]  # 杂化状态



## ONE-HOT ENCODING
def get_one_hot(value, category_list):
    one_hot = [0] * len(category_list)
    if value in category_list:
        one_hot[category_list.index(value)] = 1
    return one_hot

def get_atom_features(atom):
    features = []
    features.extend(get_one_hot(atom.GetAtomicNum(), atomic_num_list))  # 原子序数
    features.extend(get_one_hot(atom.GetDegree(), degree_list))  # 键的数目
    features.extend(get_one_hot(atom.GetTotalNumHs(), num_Hs_list))  # 氢原子数目
    features.extend(get_one_hot(atom.GetImplicitValence(), implicit_valence_list))  # 隐式价电子数
    features.extend(get_one_hot(atom.GetFormalCharge(), formal_charge_list))  # 形式电荷
    features.extend(get_one_hot(int(atom.GetIsAromatic()), is_aromatic_list))  # 是否芳香族
    features.extend(get_one_hot(atom.GetHybridization().real, hybridization_list))  # 杂化状态
    return features


def get_drug_atom_features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string")
    
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_features = get_atom_features(atom)
        atom_features_list.append(atom_features)
    
    atom_num = len(atom_features_list)

    atom_features_array = np.array(atom_features_list)
    return atom_num, atom_features_array



def atomFeatures(smiles, MAX_SMI_LEN=100):
    
    X = np.zeros((MAX_SMI_LEN, 38), dtype=np.int64)

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string")

    _, atom_features = get_drug_atom_features(smiles)
    atom_positions = []
    atom_count = 0

    atom_pattern = re.compile(r'\[[^\]]+\]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p')
    matches = atom_pattern.finditer(smiles)

    for match in matches:
        # print(f"Match: {match.group()}, Start: {match.start()}, End: {match.end()}")
        atom_positions.append(match.start())
        atom_count += 1

    current_feature = None
    atom_idx = 0

    for pos, char in enumerate(smiles):
        if pos >= MAX_SMI_LEN:
            break

        if pos in atom_positions:
            if atom_idx < len(atom_features):
                current_feature = atom_features[atom_idx]
                atom_idx += 1
            else:
                current_feature = None

        if current_feature is not None:
            X[pos, :] = current_feature
        else:
            if pos > 0 and np.any(X[pos-1, :] != 0):
                X[pos, :] = X[pos-1, :]
 
    X[len(smiles):, :] = 0
    
    return X



# dataset = "Davis"
dataset = "KIBA"
input = "../SMFF-DTA/Dataset/{}/ligands_can.txt".format(dataset)
out = "../SMFF-DTA/Dataset/{}/fingerprint_temp.txt".format(dataset)

keys = []
with open(input, 'r', encoding='utf-8') as file:
    data = file.read()
    matches = eval(data)
    for key in matches.keys():
        keys = list(matches.keys())
    with open(out, 'w+') as f1:
        for key in matches.keys():
            f1.write(f"{key} {matches[key]}\n")  
# print(keys)

with open(out, 'r') as f:
    lines = f.readlines()
    for line in lines:
        columns = line.strip().split()
        if len(columns) >= 2:
            first_column = columns[0]
            second_column = columns[1]
            if first_column in keys:
                file_path = f'/home/2023/23xzj/SMFF-DTA/Dataset/{dataset}/atomfeature/{first_column}.npy'
                atom = atomFeatures(second_column)
                np.save(file_path, atom)

print("results saved!")

