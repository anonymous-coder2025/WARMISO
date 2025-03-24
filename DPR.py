from sentence_transformers import SentenceTransformer, util
import pymysql
from tools import *
from nodes_tools import *
from argparse import ArgumentParser

from base_llm_chat import llm_generate_response
import yaml
from string import Template

from tools import *
from nodes_tools import *
from workflow_tools import *

def calc_recall(src, pred, print_result=True, top_k=None):
    top_k = TOP_K if top_k is None else top_k
    recall_n = {x: 0 for x in top_k}
    precision_n = {x: 0 for x in top_k}

    for s, p in zip(src, pred):
        # cmd_name = s['cmd_name']
        oracle_man = s
        pred_man = p
        for tk in recall_n.keys():
            cur_result_vids = pred_man[:tk]
            cur_hit = sum([x in cur_result_vids for x in oracle_man]) # sum([x in cur_result_vids for x in oracle_man])
            # recall_n[tk] += cur_hit / (len(oracle_man) + 1e-10)
            recall_n[tk] += cur_hit / (len(oracle_man)) if len(oracle_man) else 1
            precision_n[tk] += cur_hit / tk
    recall_n = {k: v / len(pred) for k, v in recall_n.items()}
    precision_n = {k: v / len(pred) for k, v in precision_n.items()}

    if print_result:
        for k in sorted(recall_n.keys()):
            print(f"{recall_n[k] :.3f}", end="\t")
        print()
        for k in sorted(precision_n.keys()):
            print(f"{precision_n[k] :.3f}", end="\t")
        print()
        for k in sorted(recall_n.keys()):
            print(f"{2 * precision_n[k] * recall_n[k] / (precision_n[k] + recall_n[k] + 1e-10) :.3f}", end="\t")
        print()

    return {'recall': recall_n, 'precision': precision_n}


if __name__=='__main__':
    parser = ArgumentParser()
    parser.add_argument("--save_path", type=str, default="logs/result_dpr.json")
    parser.add_argument("--llm", type=str, default="gpt-4o")
    parser.add_argument("--Q2D", type=str, default="yes")
    args = parser.parse_args()
    # Download model
    model = SentenceTransformer(model_name_or_path="jina-embeddings-v3", 
                                trust_remote_code=True,
                                ) 
    llm_type = args.llm
    print("Using LLMs....", llm_type)
    base_prompts = yaml.safe_load(open("prompts.yaml"))
    task_apis_writer = base_prompts["Query2API"]

    con = pymysql.connect(host='localhost', port=3306, user='root', 
                            password='@123',db='workflow',charset='utf8')
    cur = con.cursor()

    # corpus_id, corpus_name, corpus_list, corpus_embeddings = get_corpus_embedding(cur, model)
    # get all the nodes APIs databases;
    nodeAPIs_dict_id, nodeAPIs_dict_name = get_nodeAPIs_database(con, cur)

    nodes_id_list = get_nodes_list(cur)
    # print(nodes_id_list)
    nodes_label_dict, nodes_name_dict = get_nodes_labels(cur, nodes_id_list)
    All_corpus = get_corpus_embedding(cur, model, "All", nodes_id_list)
    corpus_embeddings, corpus_id, corpus_name, corpus_list = All_corpus['corpus_embeddings'], All_corpus['corpus_id'], All_corpus['corpus_name'], All_corpus['corpus_list']

    results_list = []
    json_file = open(args.save_path, 'w')

    src_list = []
    pred_list = []
    num = 0 
    with open("data/workflow_sample.json", "r", encoding='utf-8') as user_file:
        data = json.load(user_file)
    for example in data:
        print(example)
        workflow_id = example["id"]
        workflow_des = example["workflow_des"]
        nodes_ids = example["nodes_ids"]
        nodes_names = example["nodes_names"]
        workflow_programer = example["workflow_programer"]
    # for workflow_id, workflow_des, nodes_ids, nodes_names, workflow_programer in workflow_list:

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
            task_des += (" "+" ".join(task_list).replace("*",""))
            # task_des = " ".join(task_list).replace("*","")
            print("rewrite task....", task_list)

        queries_embeddings = model.encode([task_des]) #, prompt_name="query")
        hits = util.semantic_search(queries_embeddings, corpus_embeddings, top_k=30)

        gold_nodes_ids = list(set([int(i) for i in nodes_ids.split(";")]))
        gold_nodes_names = list(set([n for n in nodes_names.split(";")]))
        predict_nodes_ids = []
        predict_nodes_names = []
        predict_nodes_des = []
        predict_nodes_scores = []
        for hit in hits[0]:
            predict_nodes_ids.append(corpus_id[hit['corpus_id']])
            predict_nodes_names.append(corpus_name[hit['corpus_id']])
            predict_nodes_des.append(corpus_list[hit['corpus_id']])
            predict_nodes_scores.append(hit['score'])

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

        num += 1
        # if num > 0:
        #     break
        each_result = {
            "id": workflow_id,
            "task": workflow_des,
            "gold_nodes_ids": gold_nodes_ids,
            "gold_nodes_names": gold_nodes_names,
            "predict_nodes_ids": predict_nodes_ids,
            "predict_nodes_names": predict_nodes_names,
            "predict_nodes_scores": predict_nodes_scores,
            "gold_workflow": workflow_programer,
        }
        
        # 
        json_str = json.dumps(each_result)
        json_file.write(json_str)
        json_file.write('\n')
    
    results = calc_recall(src_list, pred_list, top_k=[10, 15, 20])
    print(results)
