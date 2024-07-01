from src.test.hotpot.hotpot_agent import single_agent
from src.test.hotpot.configs import REDIS_SETTING
from src.eval.eval_plannings import EvaluatePlanning
import json
import time
import datetime
import sys
import threading
from src.utils import ROOT_DIR
import os
from src.utils.redis_tools import RedisWrapper


def init(data_set_path):
    with open(ROOT_DIR + data_set_path, 'r') as f:
        data_list = json.load(f)

    data_l = []
    for item in data_list:
        res, msg, task = EvaluatePlanning().reset(
            id=1,
            question=item,
            planning_name="AskIsALlYouNeed",
            pointer='memory',
            planning_graph=[],
            task_name='Hotpot QA'
        )
        if res:
            data_l.append(task)

    # append to list
    client = RedisWrapper(REDIS_SETTINGS=REDIS_SETTING,setting_name='')
    client.list_push("agent_process", *data_l, side='l')

def process_agent():
    client = RedisWrapper(REDIS_SETTINGS=REDIS_SETTING, setting_name='')
    res, msg, team = single_agent()
    agent_name = 'Hotpot Agent'
    group_name = 'Hotpot Q&A'
    if res:
        group = team.find_node("group_name", group_name).metadata
        if group:
            agent = group.agent_pools[agent_name]
            while True:
                remain_process = len(client.lrange("agent_process"))
                if remain_process == 0:
                    print("Currently No Tasks..............")
                    time.sleep(60)
                    continue

                else:
                    task = client.list_pop("agent", 'r', 1)
                    agent.process_task(task)
        else:
            print("no such group")
    else:
        print(msg)



    pass


def process_message():
    pass

