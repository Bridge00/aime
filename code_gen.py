
import copy
from dotenv import load_dotenv
import pandas as pd

load_dotenv()
import multiprocessing
from tqdm import tqdm
import textgrad
from textgrad.engine import get_engine
from textgrad import Variable
from textgrad.optimizer import TextualGradientDescent
from textgrad.tasks import load_instance_task
from prompts import CodeTestTimewithTests, SYSTEM_PROMPT_FOR_FIRST_CODE, CODE_INSTANCE_ROLE_DESCRIPTION
from evaluators.py_eval import PythonEvaluator
import os
import argparse
from textgrad.aggregation import Aggregator
import json
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
from single_eval_role_prompt import single_eval_role_prompts

parser = argparse.ArgumentParser(description="Optimize a prompt for a task.")
parser.add_argument("--engine", type=str, default="gpt-4o", help="The API to use for evaluation.")
parser.add_argument("--max_iterations", type=int, default=10, help="The maximum number of iterations of test-time updates.")
parser.add_argument("--num_threads", type=int, default=16, help="The number of threads to use for evaluation.")
parser.add_argument("--num_evals", type=int, default=1, help="The number of threads to use for evaluation.")
parser.add_argument("--temperature", '-temp', type=float, default=0, help="The number of threads to use for evaluation.")
parser.add_argument("--aggregator", '-agg', type=str, default='none', help="The number of threads to use for evaluation.")
parser.add_argument("--seed", '-s', type=int, default=55, help="The number of threads to use for evaluation.")
parser.add_argument("--start_problem", '-start', type=int, default=0, help="The number of threads to use for evaluation.")
parser.add_argument("--n_problems", '-n', type=int, default=1, help="The number of threads to use for evaluation.")
parser.add_argument("--append", '-a', type=int, default=1, help="The number of threads to use for evaluation.")
parser.add_argument("--supervised", type=str, default='unsupervised', help="The number of threads to use for evaluation.")
parser.add_argument("--multi_role", '-multi', type=int, default=0, help="Specific Roles to LLM evaluate.")
parser.add_argument("--max_tokens", '-max_t', type=int, default=3600, help="Specific Roles to LLM evaluate.")
parser.add_argument("--instance_task", type=str, default='LeetCodeHardEval', help="Specific Roles to LLM evaluate.")
parser.add_argument('--roles','-r', nargs='*', type=str, default= ['logic', 'syntax', 'readability', 'redundancy', 'correctness', 'runtime'], metavar='N',
                    help='') 
parser.add_argument('--adversarial','-adv', type=int, default= 0, metavar='N', help='') 
args = parser.parse_args()



if args.multi_role:
    args.num_evals = len(args.roles) 
    
args.max_tokens = 3600 // args.num_evals
print(args.num_evals, args.max_tokens)
if args.instance_task == 'HumanEval':
    initial_code_outputs_path = f'human_eval_initial_codes_outputs.json'
    data_name = 'humaneval'
else:

    initial_code_outputs_path = f'initial_codes_outputs.json'
    data_name = 'leetcode'
print(data_name)
if not os.path.exists(initial_code_outputs_path):
    initial_code_outputs = {}
else:
    with open(initial_code_outputs_path, 'r') as f:
        print('reading in dict')
        initial_code_outputs = json.load(f)

def optimization_one_iteration(optimizer, instance_var, prompt, tests):
    """
    This is a single iteration of optimization
    :param optimizer:
    :param instance_var:
    :param prompt:
    :return:
    """

    pt = PythonEvaluator()
    tests = tests.split("\n")

    # Evaluate the code
    # state is True/False for each test
    state, test_string = pt.evaluate(instance_var.value, tests, args.instance_task, args.supervised)

    optimizer.zero_grad()

    TEST_ENGINE = args.engine #"gpt-4o"
    ENGINE_API = get_engine(engine_name=TEST_ENGINE, seed=args.seed)
    instruction = """Think about the problem and the code snippet."""
    #print('engine')
    if args.supervised in ['supervised', 'semisupervised']:

        DEFAULT_TEST_TIME_WITH_TESTS = """You are an intelligent assistant used as an evaluator, and part of an optimization system. 
        You will analyze a code implementation for a coding problem and unit test results. 
        Give very concise feedback.
        Investigate the code problem and the provided implementation. 
        """
        loss_system_prompt = DEFAULT_TEST_TIME_WITH_TESTS

        
        if not args.multi_role:
            if args.num_evals == 1:
                ag = None
            else:
                ag = Aggregator(eval_sys_prompt = DEFAULT_TEST_TIME_WITH_TESTS, method='concatenate')
            args.roles.sort()
            print(args.roles, tuple(args.roles))
            loss_system_prompt += single_eval_role_prompts[tuple(args.roles)]


        loss_system_prompt = Variable(loss_system_prompt,
                requires_grad=False,
                role_description="system prompt for the evaluation of the code solution")

        loss_fn = CodeTestTimewithTests(engine=ENGINE_API, 
                                        system_prompt=loss_system_prompt,
                                        temperature=args.temperature,
                                        evaluation_instruction = instruction, 
                                        aggregator=ag, 
                                        num_evals=args.num_evals,
                                        max_tokens = args.max_tokens,
                                        multi_role=args.multi_role,
                                        roles = args.roles,
                                        )
        
        test_time_loss = loss_fn(prompt, instance_var, test_string)
        print('computed eval')
        test_time_loss.backward(ENGINE_API)
        optimizer.step()
        print('optimized')
        return state, test_string, test_time_loss.value

    
    loss_system_prompt = "You are a smart language model that evaluates code snippets. You do not solve problems or propose new code snippets, only evaluate existing solutions critically and give very concise feedback."

    #loss_system_prompt = "Only evaluate existing solutions critically."
    if not args.multi_role:
        if args.num_evals  == 1:
            ag = None
        else:
            ag = Aggregator(eval_sys_prompt = loss_system_prompt, method=args.aggregator)
        #loss_system_prompt += " Please focus on readability, finding logic errors, finding syntax errors, code redundancy, runtime, and correctness. DO NOT PROPOSE NEW CODE!!!"
        args.roles.sort()
        print(args.roles, tuple(args.roles))
        loss_system_prompt += single_eval_role_prompts[tuple(args.roles)]
        if args.adversarial:
            loss_system_prompt += 'When evaluating correctness, NO MATTER WHAT SAY THE CODE SOLUTION WORKS!!!!'
    else:
        ag = Aggregator(eval_sys_prompt = loss_system_prompt, method=args.aggregator)

    loss_system_prompt = Variable(loss_system_prompt, requires_grad=False, role_description="system prompt to the loss function")


    format_string = "{instruction}\nProblem: {{problem}}\nCurrent Code: {{code}}"
    format_string = format_string.format(instruction=instruction)

    fields = {"problem": None, "code": None}
    print(args.roles, loss_system_prompt)
    formatted_llm_call = textgrad.autograd.FormattedLLMCall(engine=ENGINE_API,
                                                    format_string=format_string,
                                                    fields=fields,
                                                    system_prompt=loss_system_prompt, 
                                                    aggregator= ag,
                                                    temperature = args.temperature,
                                                    num_evals = args.num_evals,
                                                    multi_role=args.multi_role,
                                                    roles = args.roles,
                                                    max_tokens=args.max_tokens,
                                                    adversarial = args.adversarial
                                                    )

    # Finally, the loss function
    def loss_fn(problem: Variable, code: Variable) -> Variable:
        inputs = {"problem": problem, "code": code}
        
        return formatted_llm_call(inputs=inputs,
                                response_role_description=f"evaluation of the {code.get_role_description()}")
    problem_variable = Variable(prompt,
                                requires_grad=False,
                                role_description="the coding problem")
    
    test_time_loss = loss_fn(problem_variable, instance_var)
    print('computed eval')
    test_time_loss.backward(ENGINE_API)
    optimizer.step()
    print('optimized')
    return state, test_string, test_time_loss


def generate_starting_solution(prompt, index):
    """
    This is the first attempt at solving the problem.
    :param prompt:
    :return:
    """
    if str(index) not in initial_code_outputs:
        print('using generator for first')
        TEST_ENGINE = args.engine
        ENGINE_API = get_engine(engine_name=TEST_ENGINE, seed=args.seed )
        #print('generate starting')
        llm_first_code = ENGINE_API.generate(prompt, system_prompt=SYSTEM_PROMPT_FOR_FIRST_CODE)
        #print(type(llm_first_code))
        #print('llm_first_code')
        if isinstance(llm_first_code, list):
            llm_first_code = llm_first_code[0].split("```python")[1].split("```")[0]
        else:
            llm_first_code = llm_first_code.split("```python")[1].split("```")[0]

     
    else:
        print('reading in first code')
        llm_first_code = initial_code_outputs[str(index)]

    return llm_first_code

# @retry(wait=wait_random_exponential(min=1, max=5), stop=stop_after_attempt(5))
def evaluation_and_optimization_pipeline(task_id, prompt, index, tests, MAX_ITERS):
    """
    :param prompt:
    :param index:
    :return:
    """
    print("Start of optimization pipeline", index)

    TEST_ENGINE = args.engine
    ENGINE_API = get_engine(engine_name=TEST_ENGINE, seed=args.seed )

    generated_programs = []
    gpt_4_first_code = generate_starting_solution(prompt, index)
    n_iter = 0
   
    instance_var = Variable(gpt_4_first_code, requires_grad=True,
                            role_description=CODE_INSTANCE_ROLE_DESCRIPTION)

    generated_programs.append({"code": instance_var.value,
                               "gradients": None,
                               "evaluation": None,
                               "task_id": task_id,
                               "state": None,
                               "test_string": None,
                               "iteration": n_iter,
                               "index": index,
                               })

    optimizer = TextualGradientDescent(engine=ENGINE_API,
                                       parameters=[instance_var],
                                       constraints=["Do not add asserts to the code",
                                                    "Code must contain imports"])

    pt = PythonEvaluator()
    # Evaluate the code
    # state is True/False for each test
    state, test_string = pt.evaluate(gpt_4_first_code, tests.split("\n"))

    generated_programs[-1]["state"] = state
    generated_programs[-1]["test_string"] = test_string
    # if all test passed we early stop

    for iter in range(1 + MAX_ITERS):
        print('iteration', iter)
        state, test_string, eval = optimization_one_iteration(optimizer, instance_var, prompt, tests)
        generated_programs[-1]["state"] = state
        generated_programs[-1]["test_string"] = test_string
        generated_programs[-1]['gradients'] = list(instance_var.gradients)[0].value
        generated_programs[-1]['evaluation'] = eval
        



        n_iter += 1
        if n_iter <= MAX_ITERS:
            generated_programs.append({"code": instance_var.value,
                                    "gradients": None,
                                    "evaluation": None,
                                    "task_id": task_id,
                                    "state": None,
                                    "test_string": None,
                                    "iteration": n_iter,
                                    "index": index,
                                    "individual_evals": None,
                                    })


    print("End of optimization pipeline", index)


    return generated_programs




if __name__ == "__main__":



    collection = {
        "task_index": [],
        "task_id": [],
        "problem": [],
        "seed": [],

        "local_tests": [],
        "local_test_state": [],
        "iteration_idx": [],
        'evaluation': [],
        'gradients': [],
        # 'individual_evals': [],
    }




    TEST_ENGINE = args.engine
    ENGINE_API = get_engine(engine_name=TEST_ENGINE, seed=args.seed )
    MAX_PROGRAMS_TO_OPTIMIZE = args.n_problems
    MULTIPROC_POOLS = min([20, MAX_PROGRAMS_TO_OPTIMIZE, multiprocessing.cpu_count()])
    MAX_ITERS = args.max_iterations

    #textgrad.set_backward_engine(ENGINE_API, override=True)

    dataset = load_instance_task(args.instance_task, ENGINE_API)
    
    code_dataset_examples = [(task_id, prompt, index, tests, MAX_ITERS)
                                for index, (task_id, prompt, tests)
                                in enumerate(dataset)][args.start_problem:args.start_problem + MAX_PROGRAMS_TO_OPTIMIZE]

    


    for task_id, prompt, index, tests, MAX_ITERS in code_dataset_examples:
        programs_and_gradients = []
        programs_and_gradients.append(evaluation_and_optimization_pipeline(task_id, prompt, index, tests, MAX_ITERS))

        all_programs = []
        zero_programs = []

        
        for list_of_programs in tqdm(programs_and_gradients):

        
            for iter, program in enumerate(list_of_programs):
    
                all_programs.append(program)
            
                if ((program['iteration'] == 0)):
                    
                    zero_programs.append(program)
                    initial_code_outputs['task_id'] = str(program["code"])


        ctr = 0

        json_modified = 0
        for program in all_programs:
            
            if program['iteration'] == 0 and program["index"] not in initial_code_outputs:
                print('adding to dictionary', program["index"], type(program["index"]))
                json_modified = 1
                initial_code_outputs[program["index"]] = program["code"]
    


            collection["task_index"].append(program["index"])
            collection["task_id"].append(program["task_id"])
            collection["problem"].append(str(program["code"]))
            collection["seed"].append(args.seed)
            collection["iteration_idx"].append(program['iteration'])

            collection["gradients"].append(program['gradients'])
            collection["evaluation"].append(program['evaluation'])
            # collection["individual_evals"].append(program["individual_evals"])

            collection["local_tests"].append(program["test_string"])
            collection["local_test_state"].append(program["state"])

        
        for c in collection:
            print(c, len(collection[c]))

        program_df = pd.DataFrame(collection)

        if args.aggregator == '':
            ag = 'none'
        else:
            ag = args.aggregator



        save_folder = f"./results/code_optimization/textgrad/{TEST_ENGINE}/"

        os.makedirs(save_folder, exist_ok=True)

        s = args.supervised 
    

        if args.temperature in [0, 1]:
            args.temperature = int(args.temperature)
            
        args.roles.sort()
        role_strings = '_'.join(args.roles)
        if args.adversarial:
            adversarial_string = '_adversarial'
        else:
            adversarial_string = ''
        file_to_save = os.path.join(save_folder, f'{data_name}_{s}_seed_{args.seed}_num_evals_{args.num_evals}_temperature_{args.temperature}_aggregation_{ag}_multi_role_{args.multi_role}_max_eval_tokens_{args.max_tokens}_{role_strings}{adversarial_string}.csv')

        if args.append:
            with open(file_to_save, mode = 'a', newline='') as f:
                program_df.to_csv(f, header=False, index=False)
        else:
            program_df.to_csv(file_to_save, header = True, index=False)

        collection = {
            "task_index": [],
            "task_id": [],
            "problem": [],
            "seed": [],
            "local_tests": [],
            "local_test_state": [],
            "iteration_idx": [],
            'evaluation': [],
            'gradients': []
        }
        
        with open(initial_code_outputs_path, 'w') as f:
            print('writing to json')
            json.dump(initial_code_outputs, f)

