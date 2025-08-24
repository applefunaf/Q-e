
import csv
from rdkit import Chem

# 需要提取的 qm9 性质字段
QM9_FIELDS = [
    'rotational_constant_a', 'rotational_constant_b', 'rotational_constant_c',
    'dipole_moment', 'polarizability', 'homo', 'lumo', 'gap', 'r2',
    'zero_point_energy', 'u0', 'u298', 'h298', 'g298', 'heat_capacity'
]

def to_canonical_smiles(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        pass
    return None

# 读取 qm9_dataset.csv，建立 canonical_smiles 到性质的映射
def load_qm9_properties(qm9_path):
    smiles2props = {}
    with open(qm9_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            smiles = row['smiles']
            cano = to_canonical_smiles(smiles)
            if cano:
                props = {field: row[field] for field in QM9_FIELDS}
                smiles2props[cano] = props
    return smiles2props

# 处理 results-v4.csv，查找性质并生成新文件
def merge_properties(results_path, qm9_path, output_path):
    smiles2props = load_qm9_properties(qm9_path)
    with open(results_path, 'r', encoding='utf-8') as fin, \
         open(output_path, 'w', encoding='utf-8', newline='') as fout:
        reader = csv.DictReader(fin)
        fieldnames = reader.fieldnames + QM9_FIELDS
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        for row in reader:
            smiles = row['SMILES']
            cano = to_canonical_smiles(smiles)
            props = smiles2props.get(cano, {field: '' for field in QM9_FIELDS})
            row.update(props)
            writer.writerow(row)

if __name__ == '__main__':
    merge_properties(
        'results-v4.csv',
        'qm9_dataset.csv',
        'results-v4-with-qm9.csv'
    )
