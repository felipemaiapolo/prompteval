# Data Generation

## Pre-requisites

Navigate to `prompteval/lm-evaluation-harness` and run `pip install -e ".[unitxt]"`.

## Building Templates and Cards

Run `build_templates.py` and `build_cards.py`. Navigate to `prompteval/lm-evaluation-harness/lm_eval/tasks/unitxt` and run `generate_yamls.py`.

## Running lm_eval

Evaluate on MMLU by running lm_eval with the following task parameter. You can use other templates by changing the index at the end of each task name, i.e. swap all 0s with 3s to evaluate template 3.

`--tasks mmlu.abstract_algebra_0,mmlu.anatomy_0,mmlu.astronomy_0,mmlu.business_ethics_0,mmlu.clinical_knowledge_0,mmlu.college_biology_0,mmlu.college_chemistry_0,mmlu.college_computer_science_0,mmlu.college_mathematics_0,mmlu.college_medicine_0,mmlu.college_physics_0,mmlu.computer_security_0,mmlu.conceptual_physics_0,mmlu.econometrics_0,mmlu.electrical_engineering_0,mmlu.elementary_mathematics_0,mmlu.formal_logic_0,mmlu.global_facts_0,mmlu.high_school_biology_0,mmlu.high_school_chemistry_0,mmlu.high_school_computer_science_0,mmlu.high_school_european_history_0,mmlu.high_school_geography_0,mmlu.high_school_government_and_politics_0,mmlu.high_school_macroeconomics_0,mmlu.high_school_mathematics_0,mmlu.high_school_microeconomics_0,mmlu.high_school_physics_0,mmlu.high_school_psychology_0,mmlu.high_school_statistics_0,mmlu.high_school_us_history_0,mmlu.high_school_world_history_0,mmlu.human_aging_0,mmlu.human_sexuality_0,mmlu.international_law_0,mmlu.jurisprudence_0,mmlu.logical_fallacies_0,mmlu.machine_learning_0,mmlu.management_0,mmlu.marketing_0,mmlu.medical_genetics_0,mmlu.miscellaneous_0,mmlu.moral_disputes_0,mmlu.moral_scenarios_0,mmlu.nutrition_0,mmlu.philosophy_0,mmlu.prehistory_0,mmlu.professional_accounting_0,mmlu.professional_law_0,mmlu.professional_medicine_0,mmlu.professional_psychology_0,mmlu.public_relations_0,mmlu.security_studies_0,mmlu.sociology_0,mmlu.us_foreign_policy_0,mmlu.virology_0,mmlu.world_religions_0 --log_samples`

