
import re
from base_llm_chat import llm_generate_response
import yaml
from string import Template
import ast

# split task into blocks
def split_task(task_coder):
    task_coder = task_coder.strip("```")
    task_coder_list = task_coder.split("\n#")
    task_list = []
    for each in task_coder_list:
        each = "# "+each.strip()
        #print("****", each)
        bool_value = False
        each_list = each.split("\n")
        for item in each_list:
            if "(" in item and ")" in item:
                bool_value = True
        if bool_value:
            task_list.append(each)

    return task_list

def starts_with_number_dot(s):
    return bool(re.match(r'^\d+\.', s))

def Parse_subTasks(message):
    message = message.strip()
    sub_tasks = ""
    if "[The Start of sub-tasks]" in message and "[The End of sub-tasks]" in message:
        sub_tasks = message.split("[The Start of sub-tasks]")[-1].split("[The End of sub-tasks]")[0].strip()
    return sub_tasks


def Parse_TaskAPIs(message):
    message = message.strip()
    new_task_list = []
    for each in message.split("\n"):
        if starts_with_number_dot(each.strip()):
            temp_each = ".".join(each.split(".")[1:]).strip()
            if temp_each != "":
                new_task_list.append(temp_each)
    return new_task_list


def Parse_TaskStepAPI(message):
    parsed_message = ""
    message = message.strip()
    for each in message.split("\n"):
        if ": " in each:
            parsed_message = each.strip()
            break

    return parsed_message


def Parse_TaskStepRefine(message):
    refine_message = ""
    message = message.strip()
    for each in message.split("\n"):
        if ": " in each:
            refine_message = each.strip()
            break
            
    return refine_message

def Parse_APIsRerank(new_APIs_list):
    parsed_APIs_list = []
    match = re.search(r'Reranked List: \[(.*?)\]', new_APIs_list)

    if match:
        extracted_string = match.group(1)
        print("matched string....", extracted_string)
        parsed_APIs_list = [item.strip().strip("'") for item in extracted_string.split(",")]
        
    return parsed_APIs_list

def Parse_APIsFilter(new_APIs_list):
    parsed_APIs_list = []
    match = re.search(r'Selected List: \[(.*?)\]', new_APIs_list)
    if match:
        extracted_string = match.group(1)
        print("matched string....", extracted_string)
        parsed_APIs_list = [item.strip().strip("'") for item in extracted_string.split(",")]
        
    return parsed_APIs_list

def Parse_TaskRefine(message):
    message = message.strip()
    return message


def Parse_Taskspecify(message):
    message = message.strip()
    new_task_list = []
    if "[The Start of sub-tasks]" in message and "[The End of sub-tasks]" in message:
        new_task = message.split("[The Start of sub-tasks]")[-1].split("[The End of sub-tasks]")[0].strip()
        for each in new_task.split("\n"):
            if starts_with_number_dot(each.strip()):
                temp_each = ".".join(each.split(".")[1:]).strip()
                if temp_each != "":
                    new_task_list.append(temp_each)

    return new_task_list

def Parse_Taskspecify_bak(message):
    message = message.strip()
    new_task_list = []
    if "[The Start of sub-tasks]" in message and "[The End of sub-tasks]" in message:
        new_task = message.split("[The Start of sub-tasks]")[-1].split("[The End of sub-tasks]")[0].strip()
        for each in new_task.split("\n"):
            if each.startswith("Step") or each.startswith("Step") and ":" in each:
                temp_each = ":".join(each.split(":")[1:]).strip()
                if temp_each != "":
                    new_task_list.append(temp_each)

    return new_task_list

def Parse_TaskCoder(message):
    message = message.strip()
    if "[The Start of python code]" in message and "[The End of python code]" in message:
        new_task = message.split("[The Start of python code]")[-1].split("[The End of python code]")[0].strip()
    else:
        new_task =""
    return new_task

def Parse_BlockLabel(message):
    message = message.strip()
    if "categorize:" in message:
        block_label = message.split("categorize:")[-1].strip().split("\n")[0]
    else:
        block_label = ""
    return block_label

def Parse_Keywords(message):
    message = message.strip()
    if "Keywords:" in message:
        block_label = message.split("Keywords:")[-1].strip().split("\n")[0]
    else:
        block_label = ""
    return block_label

def Parse_TaskLabel(message):
    message = message.strip()
    if "Answer:" in message:
        task_label = message.split("Answer:")[-1].strip().split("\n")[0]
    else:
        task_label = ""
    return task_label

def Get_TaskLabel(llm_type, task_des):
    base_prompts = yaml.safe_load(open("prompts.yaml"))
    task_classify_all = base_prompts["task_classify_all"]

    task_arguments = {
            "task":task_des,
        }
    task_classify_inputs = Template(task_classify_all).safe_substitute(task_arguments)
    #print(task_classify_inputs)
    classify = llm_generate_response(llm_type, task_classify_inputs)
    #print("classify.....", classify)
    classify_result = Parse_TaskLabel(classify)
    if classify_result =="Yes":
        Big_data = True
    else:
        Big_data = False

    return Big_data


# Parse workflow code for the base method
def Parse_workflow_code(results):
    if "[The Start of workflow code]" in results and "[The End of workflow code]" in results:
        data_flow = results.split("[The Start of workflow code]")[-1].split("[The End of workflow code]")[0].strip().replace("\_","_")
    else:
        data_flow = results.replace("\_","_")
    
    return data_flow


# Parse ReAct Results
def Parse_ReAct_Results(data_flow, apis_name_list):
    # data_flow = Parse_workflow_code(results).replace("\n\n", "\n")
    data_flow_list = data_flow.split("\n")
    workflow_program = ""
    for i in range(len(data_flow_list)-2):
        if data_flow_list[i].startswith("Thought:") and data_flow_list[i+1].startswith("API:") and data_flow_list[i+2].startswith("Code:"):
            API = data_flow_list[i+1].split("API:")[-1].strip().split("(")[0]
            code = data_flow_list[i+2].split("Code:")[-1].strip()
            if API in apis_name_list:
                workflow_program += (code+"\n")
    return workflow_program


# remove "step, # " and so on lines
def remove_illegal(workflow_program):
    workflow_program = workflow_program.replace("\_","_")
    completion_new = ""
    completion_list = workflow_program.split("\n")
    # remove the lines with 'Step, #' char
    for each in completion_list:
        each = each.strip("\n").strip()
        # processing the line with multi-API calling
        
        if ("=" in each or ("(" in each and ")" in each)) and not each.startswith("Step:") and not each.startswith("#") and not each.startswith("print("):
            # each_inter = each.split("(")[-1].split(")")[0]
            # each_first = each.split("(")[0]
            # each_inter_args = ",".join([item.strip().split("=")[-1] for item in each_inter.split(",")])
            # completion_new += (each_first+"("+each_inter_args+")"+"\n")
            completion_new += (each+"\n")
    
    return completion_new.strip()


def get_previous_steps(task_list):
    previous_steps = ""
    for i in range(len(task_list)):
        previous_steps += "Step "+str(i+1)+". "+task_list[i]+"\n"
    
    previous_steps = previous_steps.strip("\n").strip()
    return previous_steps
