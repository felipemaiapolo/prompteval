import sys
sys.path.append('../')

import utils
import json

if __name__ == "__main__":

    initial_template = "The following are multiple choice questions (with answers) about {topic}.\n{question}.\nAnswers: \n{choices}.\nAnswer:"

    temps = list(utils.generateNTemplates(initial_template, "\n", 2))

    with open("templates.json", "w") as json_file:
        json.dump([template for template, _, _, _ in temps], json_file, indent=2)
    
    with open("templates_metadata.json", "w") as json_file:
        json.dump({template: {'sep': sep, 'space': space, 'op': op} for template, sep, space, op in temps}, json_file, indent=2)

    with open('./templates.json', 'r') as file:
        res = json.load(file)
    with open('./template_to_index.json', "w") as file:
        json.dump({res[i]: i for i in range(len(res))}, file, indent=2)