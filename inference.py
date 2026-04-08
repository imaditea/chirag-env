import os
import json
import sys
from openai import OpenAI
from chirag_env import CHIRAGEnv

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN     = os.getenv("HF_TOKEN", "")

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN if HF_TOKEN else "sk-placeholder")

env = CHIRAGEnv(seed=42)

ACTION_MAP = {
    "Keyword Search": 0,
    "Semantic Search": 1,
    "Hybrid Search": 2,
    "Say I Don't Know": 3
}

DIFFICULTY_ACTION = {
    1: "Semantic Search",
    2: "Hybrid Search", 
    3: "Say I Don't Know"
}

obs, info = env.reset()

print(json.dumps({"type": "[START]", "task": "CHIRAG", "question": obs["question"]}))
sys.stdout.flush()

for step in range(20):
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            max_tokens=50,
            messages=[
                {"role": "system", "content": "You are a retrieval agent. Reply with exactly one of: Keyword Search, Semantic Search, Hybrid Search, Say I Don't Know"},
                {"role": "user", "content": f"Question: {obs['question']}\nDifficulty: {obs['difficulty_level']}\nWhich action?"}
            ]
        )
        action_name = response.choices[0].message.content.strip()
        if action_name not in ACTION_MAP:
            action_name = DIFFICULTY_ACTION.get(obs["difficulty_level"], "Semantic Search")
    except Exception as e:
        print(json.dumps({"type": "LLM_ERROR", "step": step + 1, "error": str(e)}))
        sys.stdout.flush()
        action_name = DIFFICULTY_ACTION.get(obs["difficulty_level"], "Semantic Search")

    action = ACTION_MAP[action_name]
    obs, reward, terminated, truncated, info = env.step(action)

    print(json.dumps({
        "type":       "[STEP]",
        "step":       step + 1,
        "action":     info["action_taken"],
        "question":   info["question"],
        "success":    info["success"],
        "reward":     reward,
        "streak":     info["streak"],
        "difficulty": info["difficulty"],
    }))
    sys.stdout.flush()

    if terminated or truncated:
        break

print(json.dumps({
    "type":         "[END]",
    "total_steps":  step + 1,
    "final_streak": info["streak"],
    "final_reward": reward,
}))
sys.stdout.flush()
