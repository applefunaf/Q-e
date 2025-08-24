#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
合并Q-e.csv和results-v2.csv文件的脚本
将Q-e.csv中的Q和e值添加到results-v2.csv的每一行末尾
使用Python标准库，不依赖外部包
"""

import csv
import os

def merge_csv_files():
    # 读取Q-e.csv文件的Q和e值列表
    qe_data = []
    
    print("正在读取 Q-e.csv...")
    with open('Q-e.csv', 'r', encoding='utf-8') as qe_file:
        lines = qe_file.readlines()
        
        # 跳过表头行
        for line in lines[1:]:
            line = line.strip()
            if not line:
                continue
            
            # 按空格分割，获取最后两个值（Q和e）
            parts = line.split()
            if len(parts) >= 3:
                q_value = parts[-2]
                e_value = parts[-1]
                qe_data.append({'Q': q_value, 'e': e_value})
    
    print(f"已读取 {len(qe_data)} 行Q-e数据")
    
    # 读取results-v2.csv并写入新的results-v3.csv
    print("正在处理 results-v2.csv...")
    
    total_count = 0
    
    with open('results-v2.csv', 'r', encoding='utf-8') as input_file, \
         open('results-v3.csv', 'w', encoding='utf-8', newline='') as output_file:
        
        input_reader = csv.DictReader(input_file)
        
        # 创建输出文件的列名（原列名 + Q + e）
        fieldnames = input_reader.fieldnames + ['Q', 'e']
        output_writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        
        # 写入表头
        output_writer.writeheader()
        
        # 处理每一行数据
        for row_index, row in enumerate(input_reader):
            total_count += 1
            
            # 按行号直接对应Q-e数据
            if row_index < len(qe_data):
                row['Q'] = qe_data[row_index]['Q']
                row['e'] = qe_data[row_index]['e']
            else:
                # 如果Q-e数据行数不够，设为空值
                row['Q'] = ''
                row['e'] = ''
            
            # 写入到输出文件
            output_writer.writerow(row)
    
    print(f"处理完成！")
    print(f"总共处理了 {total_count} 行数据")
    print(f"Q-e数据共有 {len(qe_data)} 行")
    
    if total_count > len(qe_data):
        print(f"⚠️  results-v2.csv有 {total_count - len(qe_data)} 行数据没有对应的Q-e值")
    elif len(qe_data) > total_count:
        print(f"⚠️  Q-e.csv有 {len(qe_data) - total_count} 行数据没有被使用")
    else:
        print("✅ 两个文件的行数完全匹配")
    
    print(f"\n合并后的数据已保存到: results-v3.csv")
    
    # 显示前几行作为示例
    print("\n前5行数据预览:")
    with open('results-v3.csv', 'r', encoding='utf-8') as preview_file:
        preview_reader = csv.reader(preview_file)
        for i, row in enumerate(preview_reader):
            if i >= 6:  # 显示表头 + 前5行数据
                break
            print(f"  {','.join(row)}")

if __name__ == "__main__":
    # 确保在正确的目录中
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    try:
        merge_csv_files()
        print("\n✅ 任务完成成功！")
    except FileNotFoundError as e:
        print(f"❌ 文件未找到: {e}")
    except Exception as e:
        print(f"❌ 发生错误: {e}")