import json
from datasets import load_dataset, Dataset
from generate_data import (
    build_expression,
    build_code,
    generate_relation_problem,
    generate_bool_expression,
    generate_linear_equations,
)

def generate_training_dataset(
    example_number: int = 100,
    data_name_or_path: str = "data/code.json",
):
    all_data = []
    # generate expressions
    for _ in range(example_number):
        text, result = build_expression()
        all_data.append(text + str(result))
    
    # generate code
    hint = "The code is: \n"
    instruction = "The input is: "
    answer = ", so the output is: "
    dataset = load_dataset("json", data_files=data_name_or_path, split="train")
    for pos in range(example_number):
        input_code, input_value, real_result = build_code(dataset[pos%168])
        all_data.append(hint + input_code + instruction + str(input_value) + answer + str(result))

    # generate relation
    instruction = " So 'the city A and Z is connected' is "
    for _ in range(example_number):
        question, answer = generate_relation_problem()
        answer = "True" if answer==1 else "False"
        all_data.append(question + "\n" + instruction + str(answer))

    # generate boolean
    instruction = " The result is: "
    for _ in range(example_number):
        question, answer = generate_bool_expression(randoms=True)
        answer = "True" if answer==1 else "False"
        all_data.append(question + "\n" + instruction + answer)

    # # generate algebre
    # instruction = " The solutions are: "
    # for _ in range(example_number):
    #     question, solutions = generate_linear_equations()
    #     answer = "a = " + str(solutions["a"]) + ", b = " + str(solutions["b"])
    #     all_data.append(question + "\n" + instruction + answer)
    
    return all_data

my_list = generate_training_dataset(400)

# 将列表写入 JSON 文件
# 将列表转换为一个字典，其中键是列名，值是列表
data_dict = {"text": my_list}

# 创建Dataset对象
dataset = Dataset.from_dict(data_dict)

# 将Dataset对象保存为JSON Lines文件
dataset.to_json('/home/qingyu_yin/project/pattern/data/ft_data_1k6.json', orient='records', lines=True)
