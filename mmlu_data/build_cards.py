from unitxt import add_to_catalog
from unitxt.blocks import (
    LoadHF,
    TaskCard,
    Set,
)
from unitxt.templates import MultipleChoiceTemplate
from unitxt.splitters import RenameSplits

import json
import subprocess
import os

subtasks = [
    "abstract_algebra",
    "anatomy",
    "astronomy",
    "business_ethics",
    "clinical_knowledge",
    "college_biology",
    "college_chemistry",
    "college_computer_science",
    "college_mathematics",
    "college_medicine",
    "college_physics",
    "computer_security",
    "conceptual_physics",
    "econometrics",
    "electrical_engineering",
    "elementary_mathematics",
    "formal_logic",
    "global_facts",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_computer_science",
    "high_school_european_history",
    "high_school_geography",
    "high_school_government_and_politics",
    "high_school_macroeconomics",
    "high_school_mathematics",
    "high_school_microeconomics",
    "high_school_physics",
    "high_school_psychology",
    "high_school_statistics",
    "high_school_us_history",
    "high_school_world_history",
    "human_aging",
    "human_sexuality",
    "international_law",
    "jurisprudence",
    "logical_fallacies",
    "machine_learning",
    "management",
    "marketing",
    "medical_genetics",
    "miscellaneous",
    "moral_disputes",
    "moral_scenarios",
    "nutrition",
    "philosophy",
    "prehistory",
    "professional_accounting",
    "professional_law",
    "professional_medicine",
    "professional_psychology",
    "public_relations",
    "security_studies",
    "sociology",
    "us_foreign_policy",
    "virology",
    "world_religions",
]

unitxt_task_dir = './lm-evaluation-harness-main/lm_eval/tasks/unitxt'

if __name__ == "__main__":

    with open("mmlu_templates.json", "r") as file:
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
                loader=LoadHF(path="cais/mmlu", name=subtask),
                preprocess_steps=[
                    RenameSplits({"dev": "train"}),
                    Set({"topic": subtask.replace("_", " ")}),
                ],
                task="tasks.qa.multiple_choice.with_topic",
                templates=template_list[j][0],
            )
            card_suffix = f"mmlu.{subtask}_{template_list[j][1]}"
            template_name = f"templates.qa.multiple_choice.with_topic.{subtask}_{template_list[j][1]}"
            add_to_catalog(card, "cards." + card_suffix, overwrite=True)
            add_to_catalog(template_list[j][0], template_name, overwrite=True)
            card_names.append(card_suffix)
            template_names.append(template_name)

    with open(os.path.join(unitxt_task_dir, "mmlu_datasets"), "w") as file:
        for i, card_name in enumerate(card_names):
            file.write(card_name)
            if i != len(card_names) - 1:
                file.write("\n")

    card_to_template = {card_names[i]: template_names[i] for i in range(len(card_names))}

    with open(os.path.join(unitxt_task_dir, "card_to_template.json"), "w") as file:
        json.dump(card_to_template, file, indent=2)
