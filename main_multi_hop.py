import os, json, re

from base_llm_chat import llm_generate_response
import yaml
from string import Template
from argparse import ArgumentParser
from tools import *
from nodes_tools import *
from workflow_tools import *

import pymysql
from sentence_transformers import SentenceTransformer, util


if __name__=='__main__':
    parser = ArgumentParser()
    parser.add_argument("--llm", type=str, default="llama3-70b") 
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--save_path", type=str, default="logs/result_mt.json")
    parser.add_argument("--num_iter", type=int, default=2)
    parser.add_argument("--all_corpus", type=str, default="yes")

    args = parser.parse_args()

    # hard_list = []
    # with open("data/WG/hard.txt", "r") as hard_file:
    #     for line in hard_file:
    #         hard_list.append(int(line.strip()))

    # Download model
    # jinaai/jina-embeddings-v3
    model = SentenceTransformer(model_name_or_path="jinaai/jina-embeddings-v3", 
                                trust_remote_code=True,
                                ) 

    # get all the nodes APIs databases;
    nodeAPIs_dict_id, nodeAPIs_dict_name = get_nodeAPIs_database()

    llm_type = args.llm
    print("Using LLMs....", llm_type)

    nodes_id_list = get_nodes_list()
    # print(nodes_id_list)

    nodes_label_dict, nodes_name_dict = get_nodes_labels(nodes_id_list)

    base_prompts = yaml.safe_load(open("prompts.yaml"))
    task_step_writer = base_prompts["task_step_writer"]
    task_step_refine = base_prompts["task_step_refine"]
    APIs_rerank = base_prompts["APIs_Rerank"]
    APIs_Filter = base_prompts["APIs_Filter"]

    All_corpus = get_corpus_embedding(model, "All", nodes_id_list)

    src_list = []
    pred_list = []
    pred_list_rank = []
    pred_list_filter = []
    total_num = 0
    results_list = []

    results_list = []
    has_ids = []

    if os.path.exists(args.save_path):
        file_read = open(args.save_path, 'r')
        for line in file_read:
            tmp_data = json.loads(line.strip())
            results_list.append(tmp_data)
            has_ids.append(tmp_data["id"])
            src_list.append(tmp_data["gold_nodes_ids"])
            pred_list.append(tmp_data["predict_nodes_ids"])
            pred_list_rank.append(tmp_data["predict_nodes_ids_ranked"])
            pred_list_filter.append(tmp_data["predict_nodes_ids_filter"])
        file_read.close()

    json_file = open(args.save_path, 'a')

    with open("data/workflow_sample.json", "r", encoding='utf-8') as user_file:
        data = json.load(user_file)
    for example in data:
        print(example)
        workflow_id = example["id"]
        workflow_des = example["workflow_des"]
        nodes_ids = example["nodes_ids"]
        nodes_names = example["nodes_names"]
        workflow_programer = example["workflow_programer"]
        # workflow_class = example["workflow_class"]

    # cur.execute('''select id, workflow_des_new, nodes_ids, nodes_names, workflow_programer from workflows_knwf_new_test where''') #, (test_id,)) # where id =%s''', 34)
    # workflow_list = cur.fetchall()
    # for workflow_id, workflow_des, nodes_ids, nodes_names, workflow_programer, workflow_class in workflow_list:
        total_num += 1

        # if workflow_id not in hard_list:
        #     continue
        if workflow_id in has_ids:
            continue

        gold_nodes_ids = list(set([int(i) for i in nodes_ids.split(";")]))
        gold_nodes_names = list(set([n for n in nodes_names.split(";")]))

        task_des = workflow_des.strip("\n") #.split("\n")[0]
        
        # using predict Task Label
        Big_data = Get_TaskLabel(llm_type, task_des)

        clear_nodes_name = get_clear_nodes(task_des, All_corpus["corpus_name"])

        task_list = []

        RRFScore = {}
        predict_nodes_ids = []
        predict_nodes_names = []
        predict_nodes_des = []
        predict_nodes_scores = []

        for name in clear_nodes_name:
            name_idx = All_corpus["corpus_name"].index(name)
            if name not in predict_nodes_names:
                predict_nodes_ids.insert(0, All_corpus["corpus_id"][name_idx])
                predict_nodes_names.insert(0, name)
                predict_nodes_scores.insert(0, 1.0)

        max_steps = 10
        cur_step = 0
        while len(task_list)==0 or (len(task_list)>0 and "Finished" not in task_list[-1] and cur_step < max_steps):
            print("#########cur_step#############")
            # task details expand, generate the next step
            previous_steps = get_previous_steps(task_list) # "\n".join(task_list)
            task_arguments = {
                "task": task_des,
                "previous_steps": previous_steps
            }
            task_writer_inputs = Template(task_step_writer).safe_substitute(task_arguments)
            print(task_writer_inputs)
            new_task = llm_generate_response(llm_type, task_writer_inputs)
            print("NL_task.....", new_task)
            current_step = Parse_TaskStepAPI(new_task)
            print("to rewrite task....", current_step)
            task_list.append(current_step)
            if "Finished" not in current_step:
                related_APIs = []
                each_predict_nodes_ids = []
                each_predict_nodes_names = []
                each_predict_nodes_des = []
                each_predict_nodes_scores = []
                for iter in range(0, args.num_iter): 
                    if iter > 0:
                        # Refine current_step
                        task_block = current_step
                        API_name = task_block.split(": ")[0].strip().replace("*", "").replace("_", " ").strip("`")
                        API_des = task_block.split(": ")[-1].strip()
                        #print("task_list to compute previous steps", task_list[0:i])
                        previous_steps = get_previous_steps(task_list[:-1])
                        current_step = task_list[-1]
                        current_related_apis = "\n".join(related_APIs)
                        task_refine_arguments = {
                            "task": task_des,
                            "previous_steps": previous_steps,
                            "current_step": current_step,
                            "related_apis": current_related_apis,
                        }
                        task_refine_inputs = Template(task_step_refine).safe_substitute(task_refine_arguments)
                        print(task_refine_inputs)
                        # refine 
                        new_task = llm_generate_response(llm_type, task_refine_inputs)
                        print("refine task.....", new_task)
                        refine_task = Parse_TaskStepRefine(new_task)
                        print("refine task after parse....", refine_task)
                        if refine_task != "":
                            task_list[-1] = refine_task

                    # Retriever 
                    related_APIs = []
                    each_predict_nodes_ids = []
                    each_predict_nodes_names = []
                    each_predict_nodes_des = []
                    each_predict_nodes_scores = []
                    #for task_block in task_list:
                    task_block = task_list[-1]
                    API_name = task_block.split(": ")[0].strip().replace("*", "").replace("_", " ").strip("`")
                    API_des = task_block.split(": ")[-1].strip()
                    print("#################", API_name)
                    # if API_name in corpus_name, insert each_predict_nodes_ids 
                    if API_name in All_corpus["corpus_name"]:
                        print("***************", API_name)
                        api_index = All_corpus["corpus_name"].index(API_name)
                        each_predict_nodes_ids.insert(0, All_corpus["corpus_id"][api_index])
                        each_predict_nodes_names.insert(0, API_name)
                        each_predict_nodes_des.insert(0, All_corpus["corpus_list"][api_index])
                        each_predict_nodes_scores.insert(0, 1.0)

                        # if API_name not in predict_nodes_names:
                        #     predict_nodes_ids.insert(0, All_corpus["corpus_id"][api_index])
                        #     predict_nodes_names.insert(0, API_name)
                        #     predict_nodes_scores.insert(0, 1.0)
                        #related_APIs.append([])
                        # break
                    # else:
                    task_block = API_des

                    corpus_dict = All_corpus

                    queries_embeddings = model.encode([task_block])  # prompt_name="query"
                    if Big_data:
                        hits = util.semantic_search(queries_embeddings, corpus_dict["corpus_embeddings"], top_k=args.top_k)
                    else:
                        hits = util.semantic_search(queries_embeddings, corpus_dict["corpus_embeddings"], top_k=len(corpus_dict["corpus_embeddings"]))

                    idx = 1
                    for hit in hits[0]:
                        cur_corpus_name = corpus_dict["corpus_name"][hit['corpus_id']]
                        if not Big_data and ("DB " in cur_corpus_name or "Connector" in cur_corpus_name or "Spark" in cur_corpus_name):
                            continue
                        if idx > 21:
                            break
                        # if cur_corpus_name not in each_predict_nodes_names:
                        each_predict_nodes_ids.append(corpus_dict["corpus_id"][hit['corpus_id']])
                        each_predict_nodes_names.append(corpus_dict["corpus_name"][hit['corpus_id']])
                        each_predict_nodes_des.append(corpus_dict["corpus_list"][hit['corpus_id']])
                        each_predict_nodes_scores.append(hit['score'])
                        
                        idx += 1
                    print(each_predict_nodes_ids)
                    print(each_predict_nodes_names)
                    print(each_predict_nodes_des)
                    print(each_predict_nodes_scores)

                    for j in range(len(each_predict_nodes_ids)):
                        related_APIs.append(each_predict_nodes_names[j].replace(" ", "_")+": "+each_predict_nodes_des[j])
                            
                # update RRFScore
                idx = 1
                for k in range(len(each_predict_nodes_ids)):
                    cur_corpus_id = each_predict_nodes_ids[k]
                    if cur_corpus_id not in RRFScore:
                        # RRFScore[cur_corpus_id] = 1/(60+idx)
                        RRFScore[cur_corpus_id] = 1*each_predict_nodes_scores[k]/(0+idx)
                    else:
                        # RRFScore[cur_corpus_id] += 1/(60+idx)
                        RRFScore[cur_corpus_id] += 1*each_predict_nodes_scores[k]/(0+idx)
                    idx += 1
            cur_step += 1
        # print(RRFScore)
        sorted_RRFScore = sorted(RRFScore.items(), key=lambda item:item[1], reverse=True)
        print(sorted_RRFScore)

        num = 0
        len_sorted_RRFScore = len(sorted_RRFScore)
        if len_sorted_RRFScore > 0:
            while len(sorted_RRFScore) <= args.top_k:
                sorted_RRFScore.append(sorted_RRFScore[num%len_sorted_RRFScore])
                num += 1

            for i in range(args.top_k):
                #cprint(chose_n, i)
                predict_nodes_ids.append(sorted_RRFScore[i][0])
                predict_nodes_names.append(nodes_name_dict[sorted_RRFScore[i][0]])
                predict_nodes_scores.append(sorted_RRFScore[i][1])
        else:
            task_list = task_des # [task_des]
            corpus_dict = All_corpus

            queries_embeddings = model.encode([task_list]) # , prompt_name="query")
            hits = util.semantic_search(queries_embeddings, corpus_dict["corpus_embeddings"], top_k=args.top_k)

            for hit in hits[0]:
                # if corpus_dict["corpus_name"][hit['corpus_id']] not in predict_nodes_ids:
                predict_nodes_ids.append(corpus_dict["corpus_id"][hit['corpus_id']])
                predict_nodes_names.append(corpus_dict["corpus_name"][hit['corpus_id']])
                predict_nodes_des.append(corpus_dict["corpus_list"][hit['corpus_id']])
                predict_nodes_scores.append(hit['score'])

        print(predict_nodes_ids)
        print(predict_nodes_names)
        #print(predict_nodes_des)
        print(predict_nodes_scores)
        
        print("++++++++++++++++++++++++++")
        print(workflow_id)
        print("workflow des: ", workflow_des)
        print(gold_nodes_ids)
        nodes_names_list = [name.replace(" ", "_") for name in nodes_names.split(";")]
        node_dict = {}
        for node in nodes_names_list:
            node_dict[node] = nodeAPIs_dict_name[node]["node_dict"]

        for item in gold_nodes_ids:
            print(item, nodes_name_dict[item], nodes_label_dict[item])
        #gold_nodes_labels = [nodes_label_dict[item] for item in gold_nodes_ids]
        #print(gold_nodes_labels)
        print(gold_nodes_names)
        print(predict_nodes_ids)
        print(predict_nodes_names)
        print(predict_nodes_des)
        print(predict_nodes_scores)


        # Rerank for the top20
        candidate_apis = "\n".join(predict_nodes_des)
        candidate_APIs_list = predict_nodes_names
        APIs_rerank_arguments = {
            "task": task_des,
            "candidate_apis": candidate_apis,
            "candidate_APIs_list": candidate_APIs_list
        }

        APIs_rerank_inputs = Template(APIs_rerank).safe_substitute(APIs_rerank_arguments)
        print(APIs_rerank_inputs)

        new_APIs_list = llm_generate_response(llm_type, APIs_rerank_inputs)
        print("rerank APIs list.....", new_APIs_list)
        parsed_APIs_list = Parse_APIsRerank(new_APIs_list)
        print(parsed_APIs_list)

        new_predict_nodes_ids = []
        new_predict_nodes_names = []
        if len(parsed_APIs_list) > 0:
            for item in parsed_APIs_list:
                if item in All_corpus["corpus_name"]:
                    api_index = All_corpus["corpus_name"].index(item)
                    new_predict_nodes_names.append(item)
                    new_predict_nodes_ids.append(All_corpus["corpus_id"][api_index])
                
        src_list.append(gold_nodes_ids)
        pred_list.append(predict_nodes_ids)
        if len(new_predict_nodes_ids) == 20:
            pred_list_rank.append(new_predict_nodes_ids)
            predict_nodes_ids_ranked = new_predict_nodes_ids
            predict_nodes_names_ranked = new_predict_nodes_names
        else:
            pred_list_rank.append(predict_nodes_ids)
            predict_nodes_ids_ranked = predict_nodes_ids
            predict_nodes_names_ranked = predict_nodes_names

        # Filter for the top20
        candidate_apis = "\n".join(predict_nodes_des)
        candidate_APIs_list = predict_nodes_names  # before rerank 
        APIs_filter_arguments = {
            "task": task_des,
            "candidate_apis": candidate_apis,
            "candidate_APIs_list": candidate_APIs_list
        }

        APIs_filter_inputs = Template(APIs_Filter).safe_substitute(APIs_filter_arguments)
        print(APIs_filter_inputs)

        new_APIs_list = llm_generate_response(llm_type, APIs_filter_inputs)
        print("filter APIs list.....", new_APIs_list)
        parsed_APIs_list = Parse_APIsFilter(new_APIs_list)
        print(parsed_APIs_list)

        filter_predict_nodes_ids = []
        filter_predict_nodes_names = []
        if len(parsed_APIs_list) > 0:
            for item in parsed_APIs_list:
                if item in All_corpus["corpus_name"]:
                    api_index = All_corpus["corpus_name"].index(item)
                    filter_predict_nodes_names.append(item)
                    filter_predict_nodes_ids.append(All_corpus["corpus_id"][api_index])
        
        if len(filter_predict_nodes_ids) == 0:
            #pred_list.append(predict_nodes_ids)
            filter_predict_nodes_ids = predict_nodes_ids_ranked[0:10]
            filter_predict_nodes_names = predict_nodes_names_ranked[0:10]
        
        pred_list_filter.append(filter_predict_nodes_ids)

        each_result = {
            "id": workflow_id,
            "task": task_des,
            "gold_nodes_ids": gold_nodes_ids,
            "gold_nodes_names": gold_nodes_names,
            "predict_nodes_ids": predict_nodes_ids,
            "predict_nodes_names": predict_nodes_names,
            "predict_nodes_scores": predict_nodes_scores,
            "predict_nodes_ids_ranked": predict_nodes_ids_ranked,
            "predict_nodes_names_ranked": predict_nodes_names_ranked,
            "predict_nodes_ids_filter": filter_predict_nodes_ids,
            "predict_nodes_names_filter": filter_predict_nodes_names,
            "gold_workflow": workflow_programer,
        }

        # 
        json_str = json.dumps(each_result)
        json_file.write(json_str)
        json_file.write('\n')
        
        results_list.append(each_result)

        # if total_num > 0:
        #     break
    
    # json_file = open(args.save_path, 'w')
    # json.dump(results_list, json_file, indent=4)

    print("before ranking....")
    results = calc_recall(src_list, pred_list, top_k=[10, 15, 20])
    print(results)

    print("after ranking....")
    results_rank = calc_recall(src_list, pred_list_rank, top_k=[10, 15, 20])
    print(results_rank)

    print("after filter....")
    results_filter = calc_recall_filter(src_list, pred_list_filter)
    print(results_filter)

    
        






