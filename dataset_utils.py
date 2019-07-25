
from collections import Counter
import numpy as np
import json
import os

def get_percent(a_list, num):
    count=0
    for a in a_list:
        if a <= num:
            count+=1
    return count/len(a_list)

def get_quantile(a_list, percent):
    a_set = list(set(a_list))
    a_set.sort()
    a_distribution = Counter(a_list)
    for i in range(len(a_set)):
        count=0
        for j in range(i+1):
            count+=a_distribution[a_set[j]]
        if count/len(a_list)>=percent:
            return a_set[i], get_percent(a_list,a_set[i])
    return a_set[-1], 1

def get_statistic_for_train_dev(input_file):
    label_list=[]
    text_a_list =[]
    text_b_list=[]
    for line in open(input_file, "r", encoding="utf-8"):
        line = json.loads(line)
        text_a_list.append(line["premise"])
        text_b_list.append(line["hypothesis"])
        label_list.append(line["label"])
    length_a_list= [len(text_a.split()) for text_a in text_a_list]
    length_b_list= [len(text_b.split()) for text_b in text_b_list]
    
    print("text_a")
    print("max",max(length_a_list))
    print(128, get_percent(length_a_list,128))
    print(0.95, get_quantile(length_a_list,0.95))
    
    print("text_b")
    print("max",max(length_b_list))
    print(128, get_percent(length_b_list,128))
    print(0.95, get_quantile(length_b_list,0.95))
    
    label_distribution = Counter(label_list)
    print("label_distribution: ",label_distribution)

def get_statistic_for_test(input_file):
    label_list=[]
    text_a_list =[]
    text_b_list=[]
    for line in open(input_file, "r", encoding="utf-8"):
        line = json.loads(line)
        text_a_list.append(line["premise"])
        text_b_list.append(line["hypothesis"])
    length_a_list= [len(text_a.split()) for text_a in text_a_list]
    length_b_list= [len(text_b.split()) for text_b in text_b_list]
    
    print("text_a")
    print("max",max(length_a_list))
    print(128, get_percent(length_a_list,128))
    print(0.95, get_quantile(length_a_list,0.95))
    
    print("text_b")
    print("max",max(length_b_list))
    print(128, get_percent(length_b_list,128))
    print(0.95, get_quantile(length_b_list,0.95))   

def get_statistic(input_file):
    quantile_list=[128]
    print("train")
    get_statistic_for_train_dev(os.path.join(input_file, "train.jsonl"))
    print("val")
    get_statistic_for_train_dev(os.path.join(input_file, "val.jsonl"))
    print("test")
    get_statistic_for_test(os.path.join(input_file, "test.jsonl"))

if __name__ == "__main__":
    get_statistic("../data-superglue/CB")