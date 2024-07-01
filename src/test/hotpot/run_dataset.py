import pprint

from src.test.hotpot.hotpot_agent import single_agent
from src.test.hotpot.configs import REDIS_SETTINGS
from src.eval.eval_plannings import EvaluatePlanning
import json
import time
import threading
import numpy as np
from src.utils import ROOT_DIR
from src.utils.redis_tools import RedisWrapper


def init(data_set_path):
    with open(ROOT_DIR + data_set_path, 'r') as f:
        data_list = json.load(f)

    data_l = []
    for item in data_list[:10]:
        res, msg, task = EvaluatePlanning().reset(
            id=np.random.randint(0, 10000000),
            question=item,
            planning_name="AskIsALlYouNeed",
            pointer='memory',
            planning_graph={},
            task_name='Hotpot QA'
        )
        if res:
            data_l.append(task)
        else:
            print(msg)

    # append to list
    client = RedisWrapper(REDIS_SETTINGS=REDIS_SETTINGS, setting_name='tasks')
    client.list_push("agent_process", *data_l, side='l')
    print(f"Init Success")


def process_agent():
    res, msg, team = single_agent()
    agent_name = 'Hotpot Agent'
    group_name = 'Hotpot Q&A'
    if res:
        group = team.find_node("group_name", group_name).metadata
        if group:
            agent = group.agent_pools[agent_name]
            while True:
                remain_process = len(agent.cache.lrange("agent_process"))
                if remain_process == 0:
                    print("Currently No Tasks..............")
                    time.sleep(60)
                    continue
                else:
                    task = agent.cache.list_pop("agent_process", 'r', 1)[0]
                    pprint.pprint(task)
                    agent.process_task(task)
        else:
            print("no such group")
    else:
        print(msg)


def process_message():
    client = RedisWrapper(REDIS_SETTINGS=REDIS_SETTINGS, setting_name='tasks')
    while True:
        remain_process = len(client.lrange("human_process"))
        if remain_process == 0:
            print("Currently No Tasks..............")
            time.sleep(60)
            continue
        else:
            task = client.list_pop("human_process", 'r', 1)
            print(f"Process task: {task['id']}-{task['task'][0]}")
            traj = task['traj']
            pprint.pprint(traj)
            response = input("Give instruction: ")
            task['message'].append({"role": "user", "content": response})
            task['human_counts'] += 1
            task['msg_status'] = 0
            client.list_push("agent_process", *[task], side='l')


if __name__ == "__main__":
    # init('/data/hotpot/hotpot_train_v1_simplified.json')
    task = {'answer': None,
             'done': False,
             'error': None,
             'human_counts': 0,
             'id': 664294,
             'messages': [],
             'msg_status': 0,
             'planning_name': 'AskIsALlYouNeed',
             'planning_status': {},
             'pointer': 'memory',
             'status': 0,
             'step': 0,
             'task': ['What is the length of the track where the 2013 Liqui Moly Bathurst '
                      '12 Hour was staged?',
                      '6.213 km long'],
             'task_name': 'Hotpot QA',
             'traj': []
    }
    res, msg, team = single_agent()
    agent_name = 'Hotpot Agent'
    group_name = 'Hotpot Q&A'
    if res:
        group = team.find_node("group_name", group_name).metadata
        if group:
            agent = group.agent_pools[agent_name]
            agent.process_task(task, False)
