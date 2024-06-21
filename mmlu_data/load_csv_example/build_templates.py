import sys
sys.path.append('../')

import utils
import json
import os

if __name__ == "__main__":

    initial_template = "Classify the {type_of_class} of the following {text_type} to one of these options: {classes}. Do not include any text other than one of these options.\n{text_type}: {text}\nAnswer:"

    temps = list(utils.generateNTemplates(initial_template, " ", 1))

    with open("templates.json", "w") as json_file:
        json.dump([template for template, _, _, _ in temps], json_file, indent=2)
    
    with open("templates_metadata.json", "w") as json_file:
        json.dump({template: {'sep': sep, 'space': space, 'op': op} for template, sep, space, op in temps}, json_file, indent=2)

    with open('./templates.json', 'r') as file:
        res = json.load(file)
    with open('./template_to_index.json', "w") as file:
        json.dump({res[i]: i for i in range(len(res))}, file, indent=2)