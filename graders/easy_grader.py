def grade(obs, action, reward, info):
    if info.get("difficulty") == 1:
        if info.get("success") and action != 3:
            return 1.0
        elif action == 3:
            return 0.0
        else:
            return 0.2
    return reward / 10.0
