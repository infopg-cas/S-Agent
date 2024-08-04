


def process_ob(ob):
    if ob.startswith("You arrive at loc "):
        ob = ob[ob.find(". ") + 2:]
    return ob


def update_action(action, env, **kwargs):
    observation, reward, done, info = env.step([action])
    observation, reward, done = (
        process_ob(observation[0]),
        info["won"][0],
        done[0],
    )
    payload = {
        "o": observation,
        "r": reward,
        "d": done
    }
    return True, payload