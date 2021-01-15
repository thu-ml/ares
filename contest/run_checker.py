import argparse
import copy
import os
import sys

BASE_PATH = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(BASE_PATH)

import json
import time
import uuid
import importlib
import subprocess32
from contest.check.config import model_parameters, TOTAL_TIME, STATUS_NO_FINISH, STATUS_SUCCESS, STATUS_FILED, \
    BASE_PATH, TASK_LOG_PATH

importlib.reload(sys)

task_id = uuid.uuid4()


def cul_score(result):
    score = sum([i["success_count"] / i["dataset_size"] for i in result.values()]) / len(result.values())
    print("task_id: {} => score {}".format(task_id, str(score)))
    return "%.4f" % (round(score, 6) * 100)


def attack():
    surplus_time = copy.deepcopy(TOTAL_TIME)
    data_bus = {"status": STATUS_NO_FINISH, "data": "", "info": ""}
    data_tmp = {}
    for model, task_info in model_parameters.items():
        try:
            print("task_id: {} => start model {} attack".format(task_id, model))
            start_time = time.time()
            result_file = os.path.join(TASK_LOG_PATH, "{}.txt".format(task_id))
            task_info["result_file"] = result_file
            cmd = "python3 {} {}".format(os.path.join(BASE_PATH, "contest", "run.py"),
                                         " ".join(["--{}={}".format(k.replace("_", "-"), v) for k, v in
                                                   task_info.items()])).split(" ")
            status = subprocess32.check_call(cmd, stderr=subprocess32.STDOUT, timeout=surplus_time)
            total_time = time.time() - start_time
            print(
                "task_id: {} => attack, finish call scipt model {} attack, total_time {}S".format(task_id, model, total_time))
        except subprocess32.TimeoutExpired as time_e:
            print("task_id: {} => attack err, model {} attack timeout return".format(task_id, model))
            data_bus["status"] = STATUS_NO_FINISH
            return data_bus
        except subprocess32.CalledProcessError as call_e:
            data_bus["status"] = STATUS_FILED
            print(
                "task_id: {} => attack err, model {} attack error {}".format(task_id, model, str(call_e)))
            data_bus["info"] = "attack err, attack program exit(1)"
            return data_bus
        else:
            result_file = task_info["result_file"]
            with open(result_file, "r") as f:
                lines = f.readlines()
                last_line = lines[-1]
                result = json.loads(last_line.replace("\n", ""))
                success, model_path, data_set, expense_time, success_count, err = \
                    result.get('success'), result.get('model'), result.get('dataset'), result.get('total_time'), \
                    result.get('success_count'), result.get('err_msg')

            if success == "0":
                data_bus["status"] = STATUS_FILED
                data_bus["info"] = "model err {}".format(err)
                print("task_id: {} => model {} attack error {}".format(task_id, model, str(err)))
                return data_bus

            expense_time = float(expense_time)
            surplus_time = surplus_time - expense_time
            print(
                "task_id: {} => finish model {} attack, expense_time {}, surplus time {}s"
                    .format(task_id, model, expense_time, round(surplus_time, 5)))
            task_info_record = copy.deepcopy(task_info)
            task_info_record["expense_time"] = expense_time
            task_info_record["success_count"] = int(success_count)
            data_tmp[model] = task_info_record

    data_bus["status"] = STATUS_SUCCESS
    data_bus["data"] = json.dumps(data_tmp)
    return data_bus


def main():
    score = "0.0"
    out_info = ""

    try:
        # start attack
        result_dict = attack()
        if result_dict.get("status") == STATUS_NO_FINISH:
            raise ValueError("error attack timeout")
        if result_dict.get("status") == STATUS_FILED:
            raise ValueError("error attack model, {}".format(result_dict.get("info")))

        # calculate score
        score = cul_score(json.loads(result_dict["data"]))
    except Exception as e:
        out_info = {"score": score, "message": str(e), "status": "-1"}
    else:
        out_info = {"score": score, "message": "", "status": "0"}
    finally:
        print(json.dumps(out_info))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main()
