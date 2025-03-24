import os, json, re

from rank_bm25 import BM25Okapi
import numpy as np
import pymysql
from argparse import ArgumentParser
from base_llm_chat import llm_generate_response
import yaml
from string import Template

from tools import *
from nodes_tools import *
from workflow_tools import *


def build_bm25(corpus):
    tokenized_corpus = [doc.split(" ") for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25

def get_bm25_top(bm25, corpus, query, n =1):
    tokenized_query = query.split(" ")
    doc_scores = bm25.get_scores(tokenized_query)
    top_n = np.argsort(doc_scores)[::-1][:n]
    top_scores = [doc_scores[i] for i in top_n]
    top_docs = [corpus[i] for i in top_n]
    return top_n, top_scores, top_docs


def get_corpus(cur, block_label, nodes_id_list):
    # if block_label == "All":
    #     cur.execute("select id, node_name, node_label, node_des from retriever_nodes where id in %s", (nodes_id_list,))
    # else:
    #     block_label_list = ["Node / "+block_label, "Component / "+block_label]
    #     cur.execute("select id, node_name, node_label, node_des from retriever_nodes where id in %s and node_label in %s", (nodes_id_list, block_label_list,))
    # nodes_list = cur.fetchall()
    # print(len(nodes_list))
    corpus_list = []
    corpus_id = []
    corpus_name = []

    with open("data/nodes_data.json", "r", encoding='utf-8') as user_file:
        data = json.load(user_file)
    for each in data:
        node_id = each["node_id"]
        node_name = each["node_name"]
        node_des = each["node_des"]

    # for node_id, node_name, node_label, node_des in nodes_list:
        node_des = node_name + " " + node_des.split(".")[0].strip().replace("%%00010","")
        corpus_list.append(node_des)
        corpus_id.append(node_id)
        corpus_name.append(node_name)

    corpus_dict = {"corpus_id": corpus_id,
                   "corpus_name": corpus_name,
                   "corpus_list": corpus_list,
               }
    
    return corpus_dict


if __name__=='__main__':
    parser = ArgumentParser()
    parser.add_argument("--save_path", type=str, default="logs/result_bm25.json")
    parser.add_argument("--llm", type=str, default="gpt-4o")
    parser.add_argument("--Q2D", type=str, default="yes")

    args = parser.parse_args()

    llm_type = args.llm
    print("Using LLMs....", llm_type)
    base_prompts = yaml.safe_load(open("prompts.yaml"))
    task_apis_writer = base_prompts["Query2API"]

    con = pymysql.connect(host='localhost', port=3306, user='root', 
                    password='***', db='workflow', charset='utf8')
    cur = con.cursor()

    nodes_id_list = get_nodes_list(cur)

    All_corpus = get_corpus(cur, "All", nodes_id_list)

    bm25 = build_bm25(All_corpus["corpus_list"])

    src_list = []
    pred_list = []
    total_num = 0

    results_list = []
    json_file = open(args.save_path, 'w')

    with open("data/workflow_data.json", "r", encoding='utf-8') as user_file:
        data = json.load(user_file)
    for example in data:
        print(example)
        workflow_id = example["id"]
        workflow_des = example["workflow_des"]
        nodes_ids = example["nodes_ids"]
        nodes_names = example["nodes_names"]
        workflow_programer = example["workflow_programer"]

    # workflow_list =cur.fetchall()
    # for workflow_id, workflow_des, nodes_ids, nodes_names, workflow_programer in workflow_list:
        total_num += 1
        task_des = workflow_des.strip("\n") #.split("\n")[0]

        if args.Q2D == "yes":
            # task details expand
            task_arguments = {
                "task":task_des,
            }
            task_writer_inputs = Template(task_apis_writer).safe_substitute(task_arguments)
            print(task_writer_inputs)
            new_task = llm_generate_response(llm_type, task_writer_inputs)
            print("NL_task.....", new_task)
            task_list = Parse_TaskAPIs(new_task.replace("*",""))
            print("rewrite task....", task_list)
            task_des += (" "+" ".join(task_list).replace("*",""))
            # task_des += (" "+" ".join(task_list).replace("*",""))
            # task_des = " ".join(task_list).replace("*","")

        gold_nodes_ids = list(set([int(i) for i in nodes_ids.split(";")]))
        gold_nodes_names = list(set([n for n in nodes_names.split(";")]))

        predict_nodes_ids = []
        predict_nodes_names = []
        predict_nodes_des = []
        predict_nodes_scores = []

        top_n, predict_nodes_scores, predict_nodes_des = get_bm25_top(bm25, All_corpus["corpus_list"], task_des, n = 20)

        predict_nodes_ids = [All_corpus["corpus_id"][i] for i in top_n]
        predict_nodes_names = [All_corpus["corpus_name"][i] for i in top_n]

        print("++++++++++++++++++++++++++")
        print(workflow_id)
        print("workflow des: ", workflow_des)
        print(gold_nodes_ids)
        print(gold_nodes_names)
        print(predict_nodes_ids)
        print(predict_nodes_names)
        print(predict_nodes_des)
        print(predict_nodes_scores)
        src_list.append(gold_nodes_ids)
        pred_list.append(predict_nodes_ids)

        each_result = {
            "id": workflow_id,
            "task": task_des,
            "gold_nodes_ids": gold_nodes_ids,
            "gold_nodes_names": gold_nodes_names,
            "predict_nodes_ids": predict_nodes_ids,
            "predict_nodes_names": predict_nodes_names,
            "predict_nodes_scores": predict_nodes_scores,
            "gold_workflow": workflow_programer,
        }
        
        # write into file
        json_str = json.dumps(each_result)
        json_file.write(json_str)
        json_file.write('\n')
    
    results = calc_recall(src_list, pred_list, top_k=[10, 15, 20])
    print(results)









