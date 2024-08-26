import torch
import argparse
import random
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.utils import get_precision

from src.generate_data import (
    build_expression,
    build_code,
    generate_relation_problem, 
    generate_bool_expression,
    generate_linear_equations,
)

def build_code(inputs, range: int = 10):
    input_value = random.randint(0, range)

    if inputs["type"] == "+":
        result = input_value + int(inputs["number"])
    elif inputs["type"] == "-":
        result = input_value - int(inputs["number"])
    elif inputs["type"] == "*":
        result = input_value * int(inputs["number"])
    elif inputs["type"] == "/":
        result = input_value / int(inputs["number"])
    else:
        print(inputs["type"])
        raise ValueError("Not a valid type")
    
    return inputs["code"], input_value, result

def test_expression(
    model, tokenizer, precision,
    test_len: int = 199,
    shot_num: int = 5,
    generation_len: int = 3,
    need_false: bool = False,
    need_print: bool = False,
):

    prompt = "Now you need to calculate the answer of some mathematic equations. Here are some examples: \n"
    instruction = ""
    answer = ""

    count_has_zero = 0
    count_no_zero = 0

    for _ in range(test_len):
        text_has_zero = prompt
        text_no_zero = prompt
        for __ in range(shot_num):
            expression_has_zero, result_has_zero = build_expression(
                min_terms = 2,
                max_terms = 4,
                num_range = 20,
            )
            text_has_zero += expression_has_zero + str(result_has_zero) + " \n "
        text_has_zero += instruction
        expression_has_zero, result_has_zero = build_expression(
            min_terms = 2,
            max_terms = 4,
            num_range = 20,
            need_false = need_false,
        )
        # print(text_has_zero)
        text_has_zero += expression_has_zero

        inputs = tokenizer(text_has_zero, return_tensors="pt").to("cuda")
        outputs = model.generate(
            inputs["input_ids"], 
            max_new_tokens = generation_len, 
            num_return_sequences=1, 
            pad_token_id=tokenizer.eos_token_id
            )
        
        outputs = tokenizer.decode(outputs[0], skip_special_tokens=True)[len(text_has_zero):]

        # print(text_has_zero)

        if str(result_has_zero) in outputs:
            count_has_zero += 1

    if need_print:
        print("shot num:", shot_num)
        print(count_has_zero/test_len)

    return count_has_zero/test_len

def test_code(
    model, tokenizer, precision,
    test_len: int = 199,
    shot_num: int = 5,
    generation_len: int = 2,
    data_name_or_path: str = "data/code.json",
    need_false: bool = False,
):
    prompt = "Now you need to give me the printed result after running this python code . Here are some examples: \n"
    hint = "The code is: \n"
    instruction = "The input is: "
    answer = ", so the output is: "

    count_has_zero = 0
    # count_no_zero = 0

    dataset = load_dataset("json", data_files=data_name_or_path, split="train")

    for pos in range(test_len):
        text_has_zero = prompt
        real_result = None
        # text_no_zero = prompt
        if shot_num == 0:
            text_has_zero += instruction
            # text_no_zero += instruction
            input_code, input_value, real_result = build_code(dataset[pos+1])
            text_has_zero += hint + input_code + instruction + str(input_value) + answer

        for idx in range(shot_num):
            input_code, input_value, result = build_code(dataset[pos+idx])
            text_has_zero += hint + input_code + instruction + str(input_value) + answer + str(result) + "\n" + "Here is next example:" + "\n"
            if idx == shot_num-1:
                text_has_zero += instruction
                # text_no_zero += instruction
                input_code, input_value, real_result = build_code(dataset[pos+idx+1])
                text_has_zero += hint + input_code + instruction + str(input_value) + answer

        # print(text_has_zero)

        inputs = tokenizer(text_has_zero, return_tensors="pt").to("cuda")
        outputs = model.generate(
            inputs["input_ids"], 
            max_new_tokens = generation_len, 
            num_return_sequences=1, 
            pad_token_id=tokenizer.eos_token_id
            )
        
        outputs = tokenizer.decode(outputs[0], skip_special_tokens=True)[len(text_has_zero):]

        # print(text_has_zero, outputs)

        if str(real_result) in outputs:
            count_has_zero += 1
        
        # inputs = tokenizer(text_no_zero, return_tensors="pt").to("cuda")
        # outputs = model.generate(
        #     inputs["input_ids"], 
        #     max_new_tokens = generation_len, 
        #     num_return_sequences=1, 
        #     pad_token_id=tokenizer.eos_token_id
        #     )
        
        # outputs = tokenizer.decode(outputs[0], skip_special_tokens=True)[len(text_no_zero):]

        # # print(text_no_zero, outputs)

        # if str(result_has_zero) in outputs:
        #     count_no_zero += 1

    # print("shot num:", shot_num)
    # print(count_has_zero/test_len)
    # print(count_no_zero/test_len)
    
    return count_has_zero/test_len

def test_relation(
    model, tokenizer, precision,
    test_len: int = 199,
    shot_num: int = 5,
    generation_len: int = 2,
    need_false: bool = False,
    need_print: bool = True,
):

    prompt = "Here are some cities expressed as A, B, C, etc. I will show some connection relations, and you need to tell me if city A and city Z are connected (Answer True or False). Here are some examples: \n"
    instruction = " So 'the city A and Z is connected' is "
    answer = ""

    count_has_zero = 0
    count_no_zero = 0

    for _ in range(test_len):
        text_has_zero = prompt

        for __ in range(shot_num):
            question, answer = generate_relation_problem()
            text_has_zero += question + "\n" + instruction + str(answer) + "\n"

        question, answer = generate_relation_problem(need_false=need_false)
        text_has_zero += question + "\n" + instruction

        inputs = tokenizer(text_has_zero, return_tensors="pt").to("cuda")
        outputs = model.generate(
            inputs["input_ids"], 
            max_new_tokens = generation_len, 
            num_return_sequences=1, 
            pad_token_id=tokenizer.eos_token_id
            )
        
        # print(text_has_zero, tokenizer.decode(outputs[0], skip_special_tokens=True))
        outputs = tokenizer.decode(outputs[0], skip_special_tokens=True)[len(text_has_zero):]

        if str(answer) in outputs:
            count_has_zero += 1
        else:
            answer = 1 if answer else 0
            if str(answer) in outputs:
                count_has_zero += 1

    # if need_print:
    #     print("shot num:", shot_num)
    #     print(count_has_zero/test_len)
    # print(count_no_zero/test_len)

    return count_has_zero/test_len

def test_bool(
    model, tokenizer, precision,
    test_len: int = 199,
    shot_num: int = 5,
    generation_len: int = 1,
    need_false: bool = False,
    need_print: bool = True,
):

    prompt = "Here are some boolean expressions, you need to directly tell me the result. If it is true, print True, else print False. Here are some examples: \n"
    instruction = " The result is: "

    count_has_zero = 0
    # count_no_zero = 0

    for _ in range(test_len):
        text_has_zero = prompt

        for __ in range(shot_num):
            question, answer = generate_bool_expression(randoms=True)
            text_has_zero += question + "\n" + instruction + '1' if answer else '0' + "\n"

        question, answer = generate_bool_expression(randoms=True, need_false=need_false)
        text_has_zero += question + "\n" + instruction

        # print(text_has_zero)

        inputs = tokenizer(text_has_zero, return_tensors="pt").to("cuda")
        outputs = model.generate(
            inputs["input_ids"], 
            max_new_tokens = generation_len, 
            num_return_sequences=1, 
            pad_token_id=tokenizer.eos_token_id
            )
        
        outputs = tokenizer.decode(outputs[0], skip_special_tokens=True)[len(text_has_zero):]

        # print(text_has_zero, outputs)

        if ('1' if answer else '0') in outputs:
            count_has_zero += 1

    # if need_print: 
    #     print("shot num:", shot_num)
    #     print(count_has_zero/test_len)
    # print(count_no_zero/test_len)

    return count_has_zero/test_len

def test_algebra(
    model, tokenizer, precision,
    test_len: int = 199,
    shot_num: int = 5,
    generation_len: int = 1,
    need_false: bool = False,
    need_print: bool = True,
):

    prompt = "Here are some equations, you need to find all solutions of given variables. Here are some examples: \n"
    instruction = " The solutions are: "

    count_has_zero = 0
    # count_no_zero = 0

    for _ in tqdm(range(test_len)):
        text_has_zero = prompt
        for __ in range(shot_num):
            question, solutions = generate_linear_equations(
                variables = 2, 
                equations = 3, 
                dependent = 1,
            )
            answer = "a = " + str(solutions["a"]) + ", b = " + str(solutions["b"])
            text_has_zero += question + "\n" + instruction + answer + "\n"

        question, solutions = generate_linear_equations(
            variables = 2, 
            equations = 4 if need_false else 3, 
            dependent = 0 if need_false else 1,
        )
        answer = "a = " + str(solutions["a"]) + ", b = "
        text_has_zero += question + "\n" + instruction + answer


        inputs = tokenizer(text_has_zero, return_tensors="pt").to("cuda")
        outputs = model.generate(
            inputs["input_ids"], 
            max_new_tokens = generation_len, 
            num_return_sequences=1, 
            pad_token_id=tokenizer.eos_token_id
            )
        
        outputs = tokenizer.decode(outputs[0], skip_special_tokens=True)[len(text_has_zero):]

        # print(text_has_zero, outputs)

        if str(solutions["b"]) in outputs:
            count_has_zero += 1

    if need_print:
        print("shot num:", shot_num)
        print(count_has_zero/test_len)
    # print(count_no_zero/test_len)

    return count_has_zero/test_len

def print_list(all_list: list = None):
    for name, data in all_list:
        print(name)
        for task, each_data in data:
            print(task, each_data)


@torch.no_grad()
def group_test(
    precision,
):
    tokenizer_list = [
        # 70B series

        # 7b series
        # "/home/qingyu_yin/model/llama-2-7b-hf",
        # "/home/qingyu_yin/model/Mistral-7B-v0.2",
       
        # "/home/qingyu_yin/model/gemma-7b",
        # "/home/qingyu_yin/model/Yi-6B",

        # 1b series
        
        # "/home/qingyu_yin/model/mamba-1.4b-hf",
       
       
    ]

    model_list = [
        ("/home/qingyu_yin/model/Qwen1.5-7B", "Qwen1.5-7B-2024_06_06-01_38_09"),
        ("/home/qingyu_yin/model/Qwen1.5-1.8B", "Qwen1.5-1.8B"),
        ("/home/qingyu_yin/model/pythia-1.4b-v0", "pythia-1.4b-v0-2024_06_06-02_50_24"),
        ("/home/qingyu_yin/model/gpt-neo-1.3B", "gpt-neo-1.3B-2024_06_06-01_39_48"),
        ("/home/qingyu_yin/model/Qwen1.5-0.5B", "Qwen1.5-0.5B"),
    ]

    test_list = [
        test_algebra,
        test_bool,
        test_expression,
        test_relation,
    ]

    shot_list = [0, 1, 2, 4, 8, 16, 32]
    all_list_baseline = []
    all_list_ft = []
    all_list_icl = []
    for tokenizer_name, model in model_list:
        print("Now testing:", model)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        model_name = f"/home/qingyu_yin/project/pattern/result/{model}/checkpoint-100"
        result_list = []
        print("Now fine-tuned:", model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=precision).to("cuda")
        for test in test_list:
            result_correct = test(
                model=model,
                tokenizer=tokenizer,
                precision=precision,
                shot_num=0,
                need_print=False,
            )
            result_misleading = test(
                model=model,
                tokenizer=tokenizer,
                precision=precision,
                shot_num=0,
                need_false=True,
                need_print=False,
            )
            result_list.append([str(test.__name__), result_correct, result_misleading])
        print(result_list)
        all_list_ft.append((tokenizer_name, result_list))
        
        result_list = []
        model_name = tokenizer_name
        print("Now baseline")
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=precision).to("cuda")
        for test in test_list:
            result_correct = test(
                model=model,
                tokenizer=tokenizer,
                precision=precision,
                shot_num=0,
                need_print=False
            )
            result_misleading = test(
                model=model,
                tokenizer=tokenizer,
                precision=precision,
                shot_num=0,
                need_false=True,
                need_print=False,
            )
            result_list.append([str(test.__name__), result_correct, result_misleading])
        all_list_baseline.append((tokenizer_name, result_list))

        print(result_list)
        result_list=[]
        print("Now ICL")
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=precision).to("cuda")
        for test in test_list:
            result_correct = -1
            result_misleading = -1
            for shot_num in [1, 2, 4, 8, 16]:
                result_correct = max(result_correct, test(
                    model=model,
                    tokenizer=tokenizer,
                    precision=precision,
                    shot_num=shot_num,
                    need_print=False
                ))
                result_misleading = max(result_misleading, test(
                    model=model,
                    tokenizer=tokenizer,
                    precision=precision,
                    shot_num=shot_num,
                    need_false=True,
                    need_print=False,
                ))
            result_list.append([str(test.__name__), result_correct, result_misleading])
        print(result_list)
        all_list_icl.append((tokenizer_name, result_list))
        # model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=precision).to("cuda")
        # for test in test_list:
        #     result = []
        #     for shot_num in shot_list:
        #         result.append(
        #             test(
        #                 model=model,
        #                 tokenizer=tokenizer,
        #                 precision=precision,
        #                 shot_num=shot_num,
        #             )
        #         )
        #     print(result)
    print_list(all_list_baseline)
    print_list(all_list_ft)
    print_list(all_list_icl)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="/home/qingyu_yin/model/llama-2-7b-hf")
    parser.add_argument("--tokenizer", type=str, default="/home/qingyu_yin/model/llama-2-7b-hf")
    parser.add_argument("--data_name_or_path", type=str, default="relation")
    parser.add_argument("--context_len", type=int, default=1024)
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--precision", type=str, default="fp16")
    parser.add_argument("--len", type=int, default=16) 
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--shot_num", type=int, default=32)
    parser.add_argument("--seed", type=int, default=114514)
    parser.add_argument("--need_false", type=bool, default=False) 
    args = parser.parse_args()

    precision = get_precision(args.precision)
    random.seed(args.seed)

    if args.data_name_or_path == "group": 
        group_test(precision)
    else:
        if args.data_name_or_path == "expression":
            test = test_expression
        elif args.data_name_or_path == "code":
            test = test_code
        elif args.data_name_or_path == "relation":
            test = test_relation
        elif args.data_name_or_path == "bool":
            test = test_bool
        elif args.data_name_or_path == "algebra":
            test = test_algebra
        else:
            raise ValueError("Not a valid data type")
        
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, torch_dtype=precision).to("cuda")
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

        with torch.no_grad():
            results = []
            shot_num_list = [0, 1, 2, 4, 8, 16, 32] if "result" not in args.model_name_or_path else [0]
            for shot_num in tqdm(shot_num_list):
                if shot_num > args.shot_num:
                    break
                result = test(
                    model=model,
                    tokenizer=tokenizer,
                    precision=precision,
                    shot_num=shot_num,
                    need_false=args.need_false
                )
                results.append((shot_num, result))
        
        # print the result in format like shot_num, result
        print("###########")
        for shot_num, result in results:
            print(shot_num, ":", result)
        print("###########")

