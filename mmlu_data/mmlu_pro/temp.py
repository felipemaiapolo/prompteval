import evaluate
from datasets import load_dataset
from unitxt.text_utils import print_dict


dataset = load_dataset(
    "unitxt/data",
    "card=cards.mmlu_pro.0,template=templates.qa.multiple_choice.with_topic.pro_0,num_demos=5,demos_pool_size=20",
    trust_remote_code=True,
)


print_dict(dataset["test"][0])