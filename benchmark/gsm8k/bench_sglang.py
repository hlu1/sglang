import argparse
import ast
import json
import os
import re
import time

import numpy as np

from sglang.lang.api import set_default_backend
from sglang.test.test_utils import (
    add_common_sglang_args_and_parse,
    dump_bench_raw_result,
    select_sglang_backend,
)
from sglang.utils import download_and_cache_file, dump_state_text, read_jsonl

INVALID = -9999999


def get_one_example(lines, i, include_answer):
    ret = "Question: " + lines[i]["question"] + "\nAnswer:"
    if include_answer:
        ret += " " + lines[i]["answer"]
    return ret


def get_few_shot_examples(lines, k):
    ret = ""
    for i in range(k):
        ret += get_one_example(lines, i, True) + "\n\n"
    return ret


def get_answer_value(answer_str):
    answer_str = answer_str.replace(",", "")
    numbers = re.findall(r"\d+", answer_str)
    if len(numbers) < 1:
        return INVALID
    try:
        return ast.literal_eval(numbers[-1])
    except SyntaxError:
        return INVALID


def main(args):
    # Select backend
    set_default_backend(select_sglang_backend(args))

    # Read data
    data_path = args.data_path
    url = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"
    if not os.path.isfile(data_path):
        data_path = download_and_cache_file(url)
    lines = list(read_jsonl(data_path))

    # Construct prompts
    num_questions = args.num_questions
    num_shots = args.num_shots
    few_shot_examples = get_few_shot_examples(lines, num_shots)

    # questions = []
    # labels = []
    # for i in range(len(lines[:num_questions])):
    #     questions.append(get_one_example(lines, i, False))
    #     labels.append(get_answer_value(lines[i]["answer"]))
    # assert all(l != INVALID for l in labels)
    # arguments = [{"question": q} for q in questions]


    bad_questions = [34, 37, 46, 58, 62, 93, 97, 102, 107, 108, 119, 149, 155, 157, 172, 175, 189, 193, 198, 199, 204, 205, 209, 210, 219, 236, 238, 241, 242, 243, 255, 257, 267, 281, 290, 293, 298, 308, 313, 331, 342, 353, 357, 368, 371, 382, 392, 394, 403, 406, 409, 423, 425, 427, 428, 432, 439, 454, 457, 459, 466, 468, 470, 502, 528, 539, 564, 568, 570, 574, 580, 590, 635, 640, 641, 649, 651, 652, 675, 696, 727, 738, 739, 749, 754, 755, 768, 780, 782, 785, 796, 802, 809, 814, 815, 823, 827, 831, 835, 857, 858, 877, 883, 886, 894, 897, 900, 912, 923, 925, 926, 931, 938, 943, 951, 952, 953, 956, 960, 962, 972, 976, 979, 984, 991, 997, 1014, 1016, 1019, 1021, 1035, 1038, 1039, 1042, 1046, 1048, 1059, 1068, 1070, 1074, 1079, 1088, 1110, 1118, 1120, 1137, 1145, 1157, 1159, 1161, 1166, 1174, 1181, 1182, 1183, 1190, 1191, 1199, 1212, 1216, 1231, 1249, 1251, 1273, 1281, 1288, 1298, 1302, 1306, 1309, 1310]

    questions = []
    labels = []
    for i in range(len(bad_questions)):
        questions.append(get_one_example(lines, i, False))
        labels.append(get_answer_value(lines[i]["answer"]))
    assert all(l != INVALID for l in labels)
    arguments = [{"question": q} for q in questions]



    #####################################
    ######### SGL Program Begin #########
    #####################################

    import sglang as sgl

    @sgl.function
    def few_shot_gsm8k(s, question):
        s += few_shot_examples + question
        s += sgl.gen(
            "answer", max_tokens=512, stop=["Question", "Assistant:", "<|separator|>"]
        )

    #####################################
    ########## SGL Program End ##########
    #####################################

    # Run requests
    tic = time.perf_counter()
    states = few_shot_gsm8k.run_batch(
        arguments,
        temperature=0,
        num_threads=args.parallel,
        progress_bar=True,
    )
    latency = time.perf_counter() - tic

    preds = []
    for i in range(len(states)):
        preds.append(get_answer_value(states[i]["answer"]))

    #     if preds[i] == labels[i]:
    #         print(f"Question: {questions[i]}")
    #         print(f"Pred: {preds[i]}")
    #         print(f"Label: {labels[i]}")
    #         print(f"State: {states[i]}")
    #         print(f"--------------------------------")

    bad_questions_res = []
    print("Incorrect predictions:")
    for i in range(len(states)):
        if preds[i] != labels[i]:
            bad_questions_res.append(bad_questions[i])
    print(bad_questions_res)


    print("Debug notes:")
    for i in range(len(states)):
        if preds[i] != labels[i]:
            print(f"Question: {questions[i]}")
            print(f"Pred: {preds[i]}")
            print(f"Label: {labels[i]}")
            print(f"State: {states[i]}")
            print(f"--------------------------------")

    # Compute accuracy
    acc = np.mean(np.array(preds) == np.array(labels))
    invalid = np.mean(np.array(preds) == INVALID)

    # Compute speed
    num_output_tokens = sum(
        s.get_meta_info("answer")["completion_tokens"] for s in states
    )
    output_throughput = num_output_tokens / latency

    # Print results
    print(f"Accuracy: {acc:.3f}")
    print(f"Invalid: {invalid:.3f}")
    print(f"Latency: {latency:.3f} s")
    print(f"Output throughput: {output_throughput:.3f} token/s")

    # Dump results
    dump_state_text(f"tmp_output_{args.backend}.txt", states)
    dump_bench_raw_result(
        path=args.raw_result_file,
        states=states,
        preds=preds,
        labels=labels,
    )

    with open(args.result_file, "a") as fout:
        value = {
            "task": "gsm8k",
            "backend": args.backend,
            "num_gpus": 1,
            "latency": round(latency, 3),
            "accuracy": round(acc, 3),
            "num_requests": args.num_questions,
            "other": {
                "num_questions": args.num_questions,
                "parallel": args.parallel,
            },
        }
        fout.write(json.dumps(value) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-shots", type=int, default=5)
    parser.add_argument("--data-path", type=str, default="test.jsonl")
    parser.add_argument("--num-questions", type=int, default=200)
    args = add_common_sglang_args_and_parse(parser)
    main(args)
