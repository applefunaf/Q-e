import pubchempy as pcp
import requests
import csv

def get_smiles_with_new_logic(original_name):
    """
    最终版SMILES获取函数。
    - 能够处理以逗号分隔的物质名称。
    - 新逻辑：删除逗号并替换为'-'，然后尝试正向和反向两种组合。
    - 不再单独查询逗号前后的部分。
    """
    # 1. 生成要尝试的搜索词列表
    search_terms = [original_name]
    if ',' in original_name:
        # 清理每个部分前后的空格
        parts = [p.strip() for p in original_name.split(',')]
        
        if len(parts) > 1:
            # a. 生成正向连接的搜索词 ("A-B")
            forward_hyphenated = '-'.join(parts)
            search_terms.append(forward_hyphenated)
            
            # b. 生成反向连接的搜索词 ("B-A")
            reversed_hyphenated = '-'.join(reversed(parts))
            search_terms.append(reversed_hyphenated)

    # 打印出将要尝试的所有搜索组合，方便调试
    print(f"  -> 尝试的搜索组合: {search_terms}")

    # 2. 遍历列表，依次尝试每个搜索词
    for term in search_terms:
        # 步骤 A: 使用 pubchempy 安全地搜索化合物ID (CID)
        try:
            results = pcp.get_compounds(term, 'name')
            if not results:
                continue  # 尝试下一个搜索词
            
            cid = results[0].cid
            if not cid:
                continue
        except Exception:
            continue

        # 步骤 B: 使用 requests 手动、稳定地获取SMILES属性
        try:
            prop_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/IsomericSMILES/json"
            response = requests.get(prop_url, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                smiles = data['PropertyTable']['Properties'][0]['SMILES']
                
                print(f"  -> 使用 '{term}' 查询成功！")
                return smiles
            else:
                continue
        except Exception:
            continue
    
    # 3. 如果遍历完所有组合都失败了，则返回未找到
    return "Not Found"

def process_file(input_filename='compounds.txt', output_filename='results.csv'):
    """
    从 .txt 文件读取化合物名称，使用新逻辑函数查找SMILES，并将结果写入 .csv 文件。
    """
    try:
        with open(input_filename, 'r', encoding='utf-8') as f_in:
            compound_names = [line.strip() for line in f_in if line.strip()]
    except FileNotFoundError:
        print(f"错误: 输入文件 '{input_filename}' 未找到。请确保文件存在。")
        return

    print(f"开始处理 {len(compound_names)} 个化合物...")
    
    all_results = []
    for name in compound_names:
        print(f"正在处理: '{name}'")
        smiles_result = get_smiles_with_new_logic(name)
        all_results.append([name, smiles_result])

    try:
        with open(output_filename, 'w', newline='', encoding='utf-8') as f_out:
            writer = csv.writer(f_out)
            writer.writerow(['Original_Name', 'SMILES'])
            writer.writerows(all_results)
        print(f"\n处理完成！结果已成功保存到文件: {output_filename}")
    except IOError as e:
        print(f"错误: 无法写入CSV文件 '{output_filename}': {e}")


# --- 主程序入口 ---
if __name__ == '__main__':
    # 确保您的电脑上有一个名为 'compounds.txt' 的文件
    process_file()