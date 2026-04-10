def grade(obs, action, reward, info):
    if info.get("difficulty") == 2:
        if reward == 15.0:
            return 0.999
        elif reward == 5.0:
            return 0.5
        else:
            return 0.001
    return max(0.001, min(0.999, reward / 15.0))
