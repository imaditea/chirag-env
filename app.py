import gradio as gr
from chirag_env import CHIRAGEnv

env = CHIRAGEnv()
obs, info = env.reset()

def run_chirag(action_name):
    action_map = {
        "Keyword Search": 0,
        "Semantic Search": 1,
        "Hybrid Search": 2,
        "Say I Don't Know": 3
    }
    action = action_map[action_name]
    obs, reward, terminated, truncated, info = env.step(action)
    
    return f"""Question: {info['question']}
Action: {info['action_taken']}
Success: {info['success']}
Reward: {reward}
Streak: {info['streak']}/10
Difficulty: {info['difficulty']}/3"""

demo = gr.Interface(
    fn=run_chirag,
    inputs=gr.Dropdown(
        ["Keyword Search", "Semantic Search", 
         "Hybrid Search", "Say I Don't Know"],
        label="Choose Retrieval Strategy"
    ),
    outputs=gr.Textbox(label="CHIRAG Response"),
    title="CHIRAG - IIT Student Query Resolution",
    description="Built by Aditi Singh | IIT Guwahati"
)

demo.launch()