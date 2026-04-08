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
    return {"tasks": env.get_tasks()}

def main():
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
