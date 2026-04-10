def grade(obs, action, reward, info):
    if info.get("difficulty") == 1:
        if info.get("success") and action != 3:
            return 0.999
        elif action == 3:
            return 0.001
        else:
            return 0.2
    return max(0.001, min(0.999, reward / 10.0))
