from fastapi import FastAPI
from chirag_env import CHIRAGEnv
import uvicorn

app = FastAPI()
env = CHIRAGEnv(seed=42)
obs_store = {}
obs_store["obs"], _ = env.reset()

@app.get("/")
def root():
    return {"status": "ok", "name": "CHIRAG OpenEnv"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/reset")
def reset():
    obs_store["obs"], info = env.reset()
    return {"observation": obs_store["obs"], "info": info}

@app.post("/step")
def step(body: dict):
    obs, reward, terminated, truncated, info = env.step(body.get("action", 0))
    obs_store["obs"] = obs
    return {"observation": obs, "reward": reward, "terminated": terminated, "truncated": truncated, "info": info}

@app.get("/state")
def state():
    return {"observation": obs_store["obs"]}

@app.get("/tasks")
def tasks():
   return {
    "tasks": [
        {"id": "task_easy", "name": "Easy - Direct Factual Retrieval", "difficulty": "easy", "reward_range": [0.0, 1.0], "grader": "graders/easy_grader.py"},
        {"id": "task_medium", "name": "Medium - Multi-Chunk Reasoning", "difficulty": "medium", "reward_range": [0.0, 1.0], "grader": "graders/medium_grader.py"},
        {"id": "task_hard", "name": "Hard - Ambiguity Detection", "difficulty": "hard", "reward_range": [0.0, 1.0], "grader": "graders/hard_grader.py"}
    ]
}

def main():
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
