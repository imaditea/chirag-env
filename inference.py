import sys
import requests
import time

ENV_URL = "http://localhost:8000"

def wait_for_server(max_wait=30):
    for i in range(max_wait):
        try:
            r = requests.get(f"{ENV_URL}/health", timeout=2)
            if r.status_code == 200:
                print(f"[INFO] Server ready after {i+1}s", flush=True)
                return True
        except Exception:
            pass
        time.sleep(1)
    print("[ERROR] Server never became ready", flush=True)
    return False

def run_inference():
    if not wait_for_server():
        sys.exit(1)

    print("[START] task=CHIRAG", flush=True)

    try:
        r = requests.post(f"{ENV_URL}/reset", timeout=10)
        data = r.json()
        info = data.get("info", {})
    except Exception as e:
        print(f"[ERROR] reset failed: {e}", flush=True)
        sys.exit(1)

    total_reward = 0
    last_info = info

    for step in range(20):
        try:
            difficulty = last_info.get("difficulty", 1)
            if difficulty == 1:
                action = 2
            elif difficulty == 2:
                action = 1
            else:
                action = 3

            r = requests.post(
                f"{ENV_URL}/step",
                json={"action": action},
                timeout=10
            )
            result = r.json()

            reward = result.get("reward", 0)
            terminated = result.get("terminated", False)
            truncated = result.get("truncated", False)
            last_info = result.get("info", {})
            total_reward += reward

            print(f"[STEP] step={step+1} action={last_info.get('action_taken','unknown')} reward={reward} success={last_info.get('success',False)} streak={last_info.get('streak',0)} difficulty={last_info.get('difficulty',1)}", flush=True)

            if terminated or truncated:
                break

        except Exception as e:
            print(f"[ERROR] step={step+1} error={str(e)}", flush=True)
            break

    print(f"[END] task=CHIRAG score={total_reward} steps={step+1} streak={last_info.get('streak',0)}", flush=True)

if __name__ == "__main__":
    run_inference()
