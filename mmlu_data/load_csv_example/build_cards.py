from unitxt import add_to_catalog
from unitxt.blocks import (
    MapInstanceValues,
    RenameFields,
    Set,
    SplitRandomMix,
    TaskCard,
)
from unitxt.loaders import LoadCSV
from unitxt.templates import InputOutputTemplate

import json
import os

dataset_name = "medical_abstracts"


mappers = {
    "1": "neoplasms",
    "2": "digestive system diseases",
    "3": "nervous system diseases",
    "4": "cardiovascular diseases",
    "5": "general pathological conditions",
}

unitxt_task_dir = '../lm-evaluation-harness-main/lm_eval/tasks/unitxt'

if __name__ == "__main__":

    with open("templates.json", "r") as file:
        raw_templates = json.load(file)

    template_list = [
        (InputOutputTemplate(
            input_format=curr_template,
            output_format="{label}",
            postprocessors=[
                "processors.take_first_non_empty_line",
                "processors.lower_case_till_punc",
            ],
        ), i)
        for i, curr_template in enumerate(raw_templates)
    ]

    cards = []
    card_names = []
    template_names = []
    for j in range(len(template_list)):
        card = TaskCard(
            loader=LoadCSV(
                files={
                    "train": "https://raw.githubusercontent.com/sebischair/Medical-Abstracts-TC-Corpus/main/medical_tc_train.csv",
                    "test": "https://raw.githubusercontent.com/sebischair/Medical-Abstracts-TC-Corpus/main/medical_tc_test.csv",
                }
            ),
            preprocess_steps=[
                SplitRandomMix(
                    {"train": "train[90%]", "validation": "train[10%]", "test": "test"}
                ),
                RenameFields(
                    field_to_field={"medical_abstract": "text", "condition_label": "label"}
                ),
                MapInstanceValues(mappers={"label": mappers}),
                Set(
                    fields={
                        "classes": list(mappers.values()),
                        "text_type": "abstract",
                        "type_of_class": "topic",
                    }
                ),
            ],
            task="tasks.classification.multi_class",
            templates=template_list[j][0],
        )
        
        card_suffix = f"csv_example.card_{template_list[j][1]}"
        template_name = f"templates.classification.multi_class.csv_example.{template_list[j][1]}"
        add_to_catalog(card, "cards." + card_suffix, overwrite=True)
        add_to_catalog(template_list[j][0], template_name, overwrite=True)
        card_names.append(card_suffix)
        template_names.append(template_name)

    with open(os.path.join(unitxt_task_dir, "csv_example_datasets"), "w") as file:
        for i, card_name in enumerate(card_names):
            file.write(card_name)
            if i != len(card_names) - 1:
                file.write("\n")

    card_to_template = {card_names[i]: template_names[i] for i in range(len(card_names))}

    with open(os.path.join(unitxt_task_dir, "card_to_template.json"), "w") as file:
        json.dump(card_to_template, file, indent=2)
