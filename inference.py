import sys
import requests
import time

ENV_URL = "http://localhost:8000"

def run_inference():
    # Wait for env server to be ready
    for i in range(10):
        try:
            r = requests.get(f"{ENV_URL}/health", timeout=2)
            if r.status_code == 200:
                break
        except:
            time.sleep(1)
    
    print("[START] task=CHIRAG", flush=True)
    
    # Reset environment
    r = requests.post(f"{ENV_URL}/reset")
    data = r.json()
    info = data.get("info", {})
    
    total_reward = 0
    step = 0
    
    for step in range(20):
        try:
            difficulty = info.get("difficulty", 1)
            if difficulty == 1:
                action = 2
            elif difficulty == 2:
                action = 1
            else:
                action = 3
            
            r = requests.post(f"{ENV_URL}/step", json={"action": action})
            result = r.json()
            
            reward = result.get("reward", 0)
            terminated = result.get("terminated", False)
            truncated = result.get("truncated", False)
            info = result.get("info", {})
            total_reward += reward
            
            print(f"[STEP] step={step+1} action={info.get('action_taken','unknown')} reward={reward} success={info.get('success',False)} streak={info.get('streak',0)} difficulty={info.get('difficulty',1)}", flush=True)
            
            if terminated or truncated:
                break
                
        except Exception as e:
            print(f"[ERROR] step={step+1} error={str(e)}", flush=True)
            break
    
    print(f"[END] task=CHIRAG score={total_reward} steps={step+1} streak={info.get('streak',0)}", flush=True)

if __name__ == "__main__":
    run_inference()
