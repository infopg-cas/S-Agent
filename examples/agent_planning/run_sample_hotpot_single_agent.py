

if __name__ == "__main__":
    from src.test.hotpot.hotpot_agent import single_agent
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