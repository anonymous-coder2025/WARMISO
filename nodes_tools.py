import os, sys
import json

def get_clear_nodes(task_des, All_corpus_name):
    clear_nodes_name = []
    for name in All_corpus_name:
        if name in task_des:
            clear_nodes_name.append(name)
    return clear_nodes_name



def get_node_api(item):
    if 'input_ports' in item:
        input_ports = item['input_ports']
        input_str = ""
        for input in input_ports:
            input_str += input['port_name'].replace(" ", "_") +":"+input["port_type"]
            #   input_str += input['port_name'].replace(" ", "_")
            input_str += ", "
        input_str = input_str[:-2]
    else:
        input_str =""

    if 'output_ports' in item:
        output_ports = item['output_ports']
        output_str = ""
        for output in output_ports:
            output_str += output['port_name'].replace(" ", "_") +":"+output["port_type"]
            # output_str += output['port_name'].replace(" ", "_")
            output_str += ", "
        output_str = output_str[:-2]
    else:
        output_str=""

    # 
    if 'node_des' in item:
        node_des = item['node_des'].split(".")[0].strip().replace("This node","This function").replace("This component","This function").replace("%%00010","")+"."
    else:
        node_des = ""

    node_api = item['node_name'].replace(" ", "_")+"("+input_str+"): \n\tdescription: "+node_des+"\n\tparameters: ["+input_str+"]\n\treturns: ["+output_str+"]"

    return node_api

def get_apis_des(nodes_ids, nodeAPIs_dict):
    apis = ""
    nodes_list = nodes_ids.split(";")
    nodes_list = list(set(nodes_list))
    num_apis = len(nodes_list)
    for node in nodes_list:
        node_dict = nodeAPIs_dict[int(node)]["node_dict"]
        node_api = get_node_api(node_dict)
        apis += node_api +"\n\n"
    apis = apis.strip("\n")
    return apis, num_apis

def get_nodeAPIs_database():
    nodeAPIs_dict_id = {}
    nodeAPIs_dict_name = {}
    with open("data/nodes_sample.json", "r", encoding='utf-8') as user_file:
        data = json.load(user_file)
    for each in data:
        node_id = each["node_id"]
        node_name = each["node_name"]
        node_des = each["node_des"]
        node_label = each["node_label"]
        # node_url = each["node_url"]
        node_des = each["node_des"]
        input_ports = each["input_ports"]
        output_ports = each["output_ports"]

    # cur.execute("select id, node_name, node_label, node_url, node_des, input_ports, output_ports from retriever_nodes")
    # nodes_list =cur.fetchall()
    # for node_id, node_name, node_label, node_url, node_des, input_ports, output_ports in nodes_list:
        node_name = node_name.replace(" ", "_")
        node_des = node_des.split("\n")[0]
        node_dict = {}
        node_dict['node_id']=id
        node_dict['node_name']=node_name
        node_dict['node_label']=node_label
        # node_dict['node_url']=node_url
        node_dict['node_des']=node_des
        node_dict['input_ports']= input_ports #json.loads(input_ports)
        node_dict['output_ports']= output_ports #json.loads(output_ports)
        node_api = get_node_api(node_dict)

        nodeAPIs_dict_id[node_id]={"node_des":node_des, "node_dict": node_dict, "node_api": node_api}
        nodeAPIs_dict_name[node_name]={"node_des":node_des, "node_dict": node_dict, "node_api": node_api}
    return nodeAPIs_dict_id, nodeAPIs_dict_name


def get_nodes_list():
    nodes_id_list = set()
    with open("data/workflow_data.json", "r", encoding='utf-8') as user_file:
        data = json.load(user_file)
    for example in data:
        print(example)
        nodes_ids = example["nodes_ids"]
        # nodes_ids = nodes_ids.split(";")
        # for node in nodes_ids:
        nodes_id_list.add(nodes_ids)

    print(len(nodes_id_list))
    return list(nodes_id_list)

def get_nodes_labels(nodes_id_list):
    nodes_labels_dict = {}
    nodes_names_dict = {}
    with open("data/nodes_sample.json", "r", encoding='utf-8') as user_file:
        data = json.load(user_file)
    for each in data:
        id = each["node_id"]
        node_name = each["node_name"]
        node_label = each["node_label"]
        # if id in nodes_id_list: 
    # cur.execute("select id, node_label, node_name from retriever_nodes where id in %s", (nodes_id_list,))
    # nodes_list = cur.fetchall()
    # # print(len(nodes_list))
    # for id, node_label, node_name in nodes_list:
        node_label = node_label.split("/")[-1].strip()
        nodes_labels_dict[id] = node_label
        nodes_names_dict[id] = node_name

    
    return nodes_labels_dict, nodes_names_dict

def get_corpus_embedding(search_model, block_label, nodes_id_list):
    corpus_list = []
    corpus_id = []
    corpus_name = []

    with open("data/nodes_sample.json", "r", encoding='utf-8') as user_file:
        data = json.load(user_file)
    for each in data:
        node_id = each["node_id"]
        node_name = each["node_name"]
        node_label = each["node_label"]
        node_des = each["node_des"]

    # for node_id, node_name, node_label, node_des in nodes_list:
        # if node_id in nodes_id_list:
        node_des = node_name + " " + node_des.split(".")[0].strip().replace("%%00010","")
        corpus_list.append(node_des)
        corpus_id.append(node_id)
        corpus_name.append(node_name)
    corpus_embeddings = search_model.encode(corpus_list)

    corpus_dict = {"corpus_id": corpus_id,
                   "corpus_name": corpus_name,
                   "corpus_list": corpus_list,
                   "corpus_embeddings": corpus_embeddings
                }
    
    return corpus_dict
