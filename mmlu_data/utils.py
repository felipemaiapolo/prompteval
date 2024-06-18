import random
import sys

CHOSEN_SEPARATOR_LIST = [":", "-", "|", "<sep>", ".", "]", "/", "\\", "!", "'", '"']
CHOSEN_SPACE_LIST = ["", " ", "\n", " \n", "  ", "; \n", ", ", " , ", "\n "]

OPERATORS_LIST = [
    (lambda x: x, "lambda x: x"),
    (lambda x: x.title(), "lambda x: x.title()"),
    (lambda x: x.upper(), "lambda x: x.upper()"),
    (lambda x: x.lower(), "lambda x: x.lower()"),
]


def has_alpha_characters(input_string):
    return any(char.isalpha() for char in input_string)


def call_operator_fn(string, operator_fn, space, initial_space):
    words = string.split(initial_space)
    result = []

    for word in words:
        if len(word) == 0:
            continue
        if "{" in word or word[0].isalpha():
            result.append(word)
        else:
            result.append(operator_fn(word))

    return space.join(result)


def generateNTemplates(initial_template, initial_separator, n=100):
    random.seed(0)
    templates_metadata = set([(initial_template, '\n', initial_separator, "lambda x: x")])
    templates = set([initial_template])
    agenda = [(initial_template, 0, initial_separator)]

    while len(agenda) > 0:

        if n is not None and len(templates_metadata) >= n:
            break

        curr_template, depth, initial_space = random.choice(agenda)
        agenda.remove((curr_template, depth, initial_space))

        # curr_template, depth, initial_space = agenda.pop(0)

        existing_item = None
        for e in CHOSEN_SEPARATOR_LIST:
            if curr_template.find(e) != -1:
                existing_item = e

        if existing_item is None:
            continue

        for sep in CHOSEN_SEPARATOR_LIST:
            operator_fn = random.choice(OPERATORS_LIST)
            space = random.choice(CHOSEN_SPACE_LIST)
            new_template = call_operator_fn(
                curr_template.replace(existing_item, sep), operator_fn[0], space, initial_space
            )

            if new_template not in templates:
                templates_metadata.add((new_template, sep, space, operator_fn[1]))
                templates.add(new_template)
                if space != "":
                    agenda.append((new_template, depth + 1, space))
                
                if n is not None and len(templates_metadata) >= n:
                    break

        print(len(templates))
        sys.stdout.flush()

    for t in templates_metadata:
        print('t', t)
    return sorted(list(templates_metadata))
