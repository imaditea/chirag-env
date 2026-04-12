import sys
from chirag_env import CHIRAGEnv

def run_inference():
    env = CHIRAGEnv()
    obs, info = env.reset()
    
    print("[START] task=CHIRAG", flush=True)
    
    total_reward = 0
    step = 0
    
    for step in range(20):
        try:
            # Simple rule based agent - no network needed
            difficulty = info.get('difficulty', 1)
            if difficulty == 1:
                action = 2  # hybrid search for easy
            elif difficulty == 2:
                action = 1  # semantic for medium
            else:
                action = 3  # say IDK for hard
            
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            print(f"[STEP] step={step+1} action={info['action_taken']} reward={reward} success={info['success']} streak={info['streak']} difficulty={info['difficulty']}", flush=True)
            
            if terminated or truncated:
                break
                
        except Exception as e:
            print(f"[ERROR] step={step+1} error={str(e)}", flush=True)
            break
    
    print(f"[END] task=CHIRAG score={total_reward} steps={step+1} streak={info.get('streak', 0)}", flush=True)
    return total_reward

if __name__ == "__main__":
    run_inference()
