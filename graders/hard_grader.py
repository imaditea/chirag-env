def grade(obs, action, reward, info):
    if info.get("difficulty") == 3:
        if action == 3 and info.get("success"):
            return 0.999
        else:
            return 0.001
    return max(0.001, min(0.999, reward / 10.0))
