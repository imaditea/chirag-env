def grade(obs, action, reward, info):
    if info.get("difficulty") == 3:
        if action == 3 and info.get("success"):
            return 1.0
        else:
            return 0.0
    return reward / 10.0
