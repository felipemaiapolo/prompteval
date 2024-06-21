from unitxt.templates import MultipleChoiceTemplate
from unitxt.blocks import LoadHF, Set, TaskCard
from unitxt.catalog import add_to_catalog
from unitxt.operators import Copy, IndexOf, RenameFields

import json
import os

subtasks = ["ARC-Challenge", "ARC-Easy"]

unitxt_task_dir = '../lm-evaluation-harness-main/lm_eval/tasks/unitxt'

if __name__ == "__main__":

    with open("templates.json", "r") as file:
        raw_templates = json.load(file)

    template_list = [
        (MultipleChoiceTemplate(
            input_format=curr_template,
            target_field="answer",
            choices_separator="\n",
            postprocessors=["processors.first_character"],
        ), i)
        for i, curr_template in enumerate(raw_templates)
    ]

    cards = []
    card_names = []
    template_names = []
    for i, subtask in enumerate(subtasks):
        for j in range(len(template_list)):
            card = TaskCard(
                loader=LoadHF(path="ai2_arc", name=subtask),
                preprocess_steps=[
                    Set({"topic": "science"}),
                    RenameFields(field_to_field={"answerKey": "label", "choices": "_choices"}),
                    Copy(
                        field_to_field={"_choices/text": "choices", "_choices/label": "labels"}
                    ),
                    IndexOf(search_in="labels", index_of="label", to_field="answer"),
                ],
                task="tasks.qa.multiple_choice.with_topic",
                templates=template_list[j][0],
            )
            alphanumeric_subtask_name = subtask.replace('-', '_')
            card_suffix = f"arc.{alphanumeric_subtask_name}_{template_list[j][1]}"
            template_name = f"templates.qa.multiple_choice.with_topic.{alphanumeric_subtask_name}_{template_list[j][1]}"
            add_to_catalog(card, "cards." + card_suffix, overwrite=True)
            add_to_catalog(template_list[j][0], template_name, overwrite=True)
            card_names.append(card_suffix)
            template_names.append(template_name)

    with open(os.path.join(unitxt_task_dir, "arc_datasets"), "w") as file:
        for i, card_name in enumerate(card_names):
            file.write(card_name)
            if i != len(card_names) - 1:
                file.write("\n")

    card_to_template = {card_names[i]: template_names[i] for i in range(len(card_names))}

    with open(os.path.join(unitxt_task_dir, "card_to_template.json"), "w") as file:
        json.dump(card_to_template, file, indent=2)
