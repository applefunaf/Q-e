#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
提取results-v3.csv的第二、三、四列到results-v4.csv，并修正最后一列的非标准负号
"""
import csv
import re

def fix_minus(val):
    # 替换全角负号、长破折号等为标准负号
    if val is None:
        return val
    return re.sub(r"[−—‒–﹣－]", "-", val)

def extract_and_fix():
    with open('results-v3.csv', 'r', encoding='utf-8') as fin, \
         open('results-v4.csv', 'w', encoding='utf-8', newline='') as fout:
        reader = csv.reader(fin)
        writer = csv.writer(fout)
        header = next(reader)
        # 只保留SMILES, Q, e
        writer.writerow([header[1], header[2], header[3]])
        for row in reader:
            if len(row) < 4:
                continue
            smiles = row[1]
            q = row[2]
            e = fix_minus(row[3])
            writer.writerow([smiles, q, e])

if __name__ == "__main__":
    extract_and_fix()
    print("已生成results-v4.csv，且已修正非标准负号。")
