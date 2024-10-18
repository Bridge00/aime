from evaluators.py_eval import PythonEvaluator
from evaluators.utils import timeout_exec_with_return
import io
import sys
from evoeval.data import get_evo_eval, write_jsonl
import json

def get_true_value(code_to_run, function_name_string, input): 
    pt = PythonEvaluator()

    exec_env = {}

    exec(code_to_run + '\nresult=' + function_name_string  + '(' + input +')', exec_env)

    output_value = exec_env['result']
    if isinstance(output_value, str):
        output_value = output_value.strip()
    return output_value


if __name__ == '__main__':

    for type in ['difficult', 'creative', 'subtle', 'combine', 'tool_use']:
        print(type)
        list_to_json = []
        evoeval_benchmark = f"EvoEval_{type}"
        dict_ = get_evo_eval(evoeval_benchmark)

        for key in dict_:
            print(key)
            try:
                prompt = dict_[key]['prompt']
                def_index = prompt.find('def')
                paran_index = prompt.find('(')
        
                function_name_string = prompt[def_index:paran_index].split()[1]

                values = [get_true_value(prompt + '\n' + dict_[key]['canonical_solution'], function_name_string, i) for i in dict_[key]['inputs']]
                tests = [f'assert {function_name_string}({i}) == {v}' for i, v in zip(dict_[key]['inputs'], values)]
                #print(tests)
                #print(prompt + '\n' + dict_[key]['canonical_solution'])
                dict_[key]['test'] = '\n'.join(tests)
                list_to_json.append(dict_[key])
            except Exception as e:
                print('something went wrong', e)
                pass
        
        with open(f'./data/{evoeval_benchmark}.jsonl', 'w') as file:
            for record in list_to_json:
                json_record = json.dumps(record)
                file.write(json_record + '\n')