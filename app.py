from fastapi import FastAPI
from fastapi.responses import JSONResponse
from chirag_env import CHIRAGEnv
import uvicorn

app = FastAPI()
env = CHIRAGEnv(seed=42)
obs, info = env.reset()
current_obs = obs

@app.get("/")
def root():
    return {"status": "ok", "name": "CHIRAG OpenEnv"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/reset")
def reset():
    global current_obs
    current_obs, info = env.reset()
    return {"observation": current_obs, "info": info}

@app.post("/step")
def step(action: dict):
    global current_obs
    action_id = action.get("action", 0)
    current_obs, reward, terminated, truncated, info = env.step(action_id)
    return {
        "observation": current_obs,
        "reward": reward,
        "terminated": terminated,
        "truncated": truncated,
        "info": info
    }

@app.get("/state")
def state():
    return {"observation": current_obs}

@app.get("/tasks")
def tasks():
    return {"tasks": env.get_tasks()}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
