single_eval_role_prompts ={
                             ('correctness', 'logic', 'readability', 'redundancy', 'runtime', 'syntax') : ' Please focus on readability, finding logic errors, finding syntax errors, code redundancy, runtime, and correctness. DO NOT PROPOSE NEW CODE!!!',
                             ('correctness', 'logic', 'syntax') : ' Please focus on finding logic errors, finding syntax errors, and correctness. DO NOT PROPOSE NEW CODE!!!',
                             ('readability', 'redundancy', 'runtime') : ' Please focus on readability, code redundancy, and runtime. DO NOT PROPOSE NEW CODE!!!',
                             ('correctness',) : ' Please focus on correctness. DO NOT PROPOSE NEW CODE!!!',
                             ('logic',) : ' Please focus on finding logic errors. DO NOT PROPOSE NEW CODE!!!',
                             ('syntax',) : ' Please focus on finding syntax errors. DO NOT PROPOSE NEW CODE!!!',
                             ('logic', 'readability') : ' Please focus on readability and finding logic errors. DO NOT PROPOSE NEW CODE!!!',
                             ('readability',) : ' Please focus on readability. DO NOT PROPOSE NEW CODE!!!'

                        }