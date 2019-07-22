import sys
import pandas as pd
import json, csv

def json2tsv1(inf, outf):
    pd.read_json(inf).to_csv(outf, sep='\t', index=False)

def json2tsv2(inf, outf):
    lines=[]
    for line in open(inf, "r", encoding="utf-8"):
        line = json.loads(line)
        lines.append(line)
    for line in lines:
        print(line["premise"])
    return lines

if __name__ == "__main__":
    json2tsv2("/Users/ksboy/Datasets/SuperGLUE/CB/train.jsonl","/Users/ksboy/Datasets/SuperGLUE/CB/train.tsv")
