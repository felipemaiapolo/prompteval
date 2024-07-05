from datasets import load_dataset
from unitxt.text_utils import print_dict


dataset = load_dataset(
    "unitxt/data",
    "card=cards.mmlu.college_biology,template=templates.qa.multiple_choice.with_topic.mmlu,num_demos=5,demos_pool_size=5",
    trust_remote_code=True,
)


print_dict(dataset["test"][0])