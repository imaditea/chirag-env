def grade(obs, action, reward, info):
    if info.get("difficulty") == 2:
        if reward == 15.0:
            return 1.0
        elif reward == 5.0:
            return 0.5
        else:
            return 0.0
    return reward / 15.0
