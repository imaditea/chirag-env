import numpy as np
import gymnasium as gym
from gymnasium import spaces
import random

# 20 real IIT online degree FAQ questions with answers
FAQ_DATA = [
    {
        "question": "How is attendance calculated?",
        "answer": "Attendance is calculated based on the number of live sessions attended out of total sessions conducted.",
        "keywords": ["attendance", "calculated", "sessions"]
    },
    {
        "question": "What happens if I fail a subject?",
        "answer": "If you fail a subject you will get a backlog and must appear in the next semester exam for that subject.",
        "keywords": ["fail", "backlog", "subject", "exam"]
    },
    {
        "question": "How do I get an NOC?",
        "answer": "You can request an NOC by submitting a form through the student portal with your roll number and reason.",
        "keywords": ["NOC", "portal", "form", "request"]
    },
    {
        "question": "Can I visit the IIT campus?",
        "answer": "Yes, online degree students can visit campus during designated events with prior approval and valid ID.",
        "keywords": ["visit", "campus", "approval", "ID"]
    },
    {
        "question": "What is the fee payment deadline?",
        "answer": "Fee payment deadline is typically before the start of each semester as mentioned in the academic calendar.",
        "keywords": ["fee", "payment", "deadline", "semester"]
    },
    {
        "question": "How do I apply for a certificate?",
        "answer": "Certificates can be requested through the student portal under the documents section using your roll number.",
        "keywords": ["certificate", "portal", "documents", "roll number"]
    },
    {
        "question": "What is the grading system?",
        "answer": "The grading system uses a 10 point scale where each grade corresponds to a specific marks range.",
        "keywords": ["grading", "scale", "marks", "grade"]
    },
    {
        "question": "How many subjects are there per semester?",
        "answer": "Typically there are 4 to 5 subjects per semester depending on the program and year of study.",
        "keywords": ["subjects", "semester", "program"]
    },
    {
        "question": "Can I change my registered email?",
        "answer": "Yes you can update your registered email through the profile settings section of the student portal.",
        "keywords": ["email", "change", "update", "profile"]
    },
    {
        "question": "What happens if I miss an exam?",
        "answer": "Missing an exam without prior approval results in zero marks for that exam and you must apply for a makeup.",
        "keywords": ["miss", "exam", "makeup", "marks"]
    },
    {
        "question": "How do I access study materials?",
        "answer": "Study materials are available on the learning management system accessible through your student login.",
        "keywords": ["study", "materials", "LMS", "login"]
    },
    {
        "question": "What is the minimum attendance requirement?",
        "answer": "The minimum attendance requirement is 75 percent of all live sessions conducted in a semester.",
        "keywords": ["minimum", "attendance", "75", "requirement"]
    },
    {
        "question": "How do I contact my faculty?",
        "answer": "You can contact faculty through the discussion forum on the portal or through official email during office hours.",
        "keywords": ["contact", "faculty", "forum", "email"]
    },
    {
        "question": "What is the duration of the program?",
        "answer": "The online degree program is typically 3 to 4 years depending on the specific degree and institution.",
        "keywords": ["duration", "program", "years", "degree"]
    },
    {
        "question": "How are assignments submitted?",
        "answer": "Assignments are submitted through the learning portal before the deadline specified by the faculty.",
        "keywords": ["assignment", "submit", "portal", "deadline"]
    },
    {
        "question": "Can I take a semester break?",
        "answer": "Semester breaks require formal approval from the academic committee with valid documented reasons.",
        "keywords": ["break", "semester", "approval", "committee"]
    },
    {
        "question": "How do I get my hall ticket?",
        "answer": "Hall tickets are generated automatically on the student portal before exams once fees are paid.",
        "keywords": ["hall ticket", "exam", "portal", "fees"]
    },
    {
        "question": "What is the backlog policy?",
        "answer": "Students can clear backlogs in subsequent semesters with no limit on the number of attempts allowed.",
        "keywords": ["backlog", "policy", "attempts", "clear"]
    },
    {
        "question": "How do I update my mobile number?",
        "answer": "Mobile number can be updated by submitting a change request through the student portal settings.",
        "keywords": ["mobile", "number", "update", "settings"]
    },
    {
        "question": "Are there any scholarships available?",
        "answer": "Scholarship information is available on the official program website and is based on merit and need.",
        "keywords": ["scholarship", "merit", "need", "available"]
    },
]

class CHIRAGEnv(gym.Env):
    """
    CHIRAG - Student Query Resolution RL Environment
    
    An AI agent learns to retrieve accurate answers for IIT online
    degree students by choosing the best retrieval strategy.
    
    State: current question + difficulty level
    Actions: 0=keyword search, 1=semantic search, 
             2=hybrid search, 3=say I don't know
    Reward: based on answer accuracy
    """
    
    metadata = {"render_modes": ["human"]}
    
    def __init__(self, render_mode=None):
        super().__init__()
        
        self.render_mode = render_mode
        self.faq_data = FAQ_DATA
        self.current_question = None
        self.current_answer = None
        self.current_keywords = None
        self.difficulty = 1
        self.steps = 0
        self.correct_streak = 0
        self.max_steps = 50
        
        # 4 actions: keyword, semantic, hybrid, dont know
        self.action_space = spaces.Discrete(4)
        
        # State: [difficulty, question_length, keyword_match_score]
        self.observation_space = spaces.Box(
            low=np.array([1, 0, 0]),
            high=np.array([3, 100, 1]),
            dtype=np.float32
        )
    
    def _get_obs(self):
        if self.current_question is None:
            return np.array([self.difficulty, 0, 0], dtype=np.float32)
        
        question_len = min(len(self.current_question.split()), 100)
        keyword_score = self._keyword_match_score()
        
        return np.array([
            self.difficulty,
            question_len,
            keyword_score
        ], dtype=np.float32)
    
    def _keyword_match_score(self):
        if not self.current_question or not self.current_keywords:
            return 0.0
        question_lower = self.current_question.lower()
        matches = sum(1 for kw in self.current_keywords 
                     if kw.lower() in question_lower)
        return matches / len(self.current_keywords)
    
    def _simulate_retrieval(self, action):
        """Simulate different retrieval strategies"""
        keyword_score = self._keyword_match_score()
        
        if action == 0:  # keyword search
            # Works well when keywords match directly
            success_prob = 0.9 if keyword_score > 0.5 else 0.4
            if self.difficulty == 2:
                success_prob *= 0.8
            elif self.difficulty == 3:
                success_prob *= 0.5
                
        elif action == 1:  # semantic search
            # More consistent across difficulties
            success_prob = 0.75
            if self.difficulty == 1:
                success_prob = 0.8
            elif self.difficulty == 3:
                success_prob = 0.6
                
        elif action == 2:  # hybrid search
            # Best overall but agent must learn this
            success_prob = 0.85
            if self.difficulty == 3:
                success_prob = 0.7
                
        else:  # say I don't know (action == 3)
            # Only good for unanswerable questions
            if self.difficulty == 3 and random.random() < 0.3:
                return True, "idk"  # sometimes correct to say IDK
            return False, "idk"
        
        success = random.random() < success_prob
        return success, self.current_answer if success else "wrong answer"
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Pick random question
        idx = self.np_random.integers(0, len(self.faq_data))
        self.current_question = self.faq_data[idx]["question"]
        self.current_answer = self.faq_data[idx]["answer"]
        self.current_keywords = self.faq_data[idx]["keywords"]
        
        self.steps = 0
        self.correct_streak = 0
        
        # Adjust difficulty
        if self.correct_streak >= 5:
            self.difficulty = min(3, self.difficulty + 1)
        
        obs = self._get_obs()
        info = {
            "question": self.current_question,
            "difficulty": self.difficulty
        }
        
        return obs, info
    
    def step(self, action):
        self.steps += 1
        
        success, retrieved = self._simulate_retrieval(action)
        
        # Calculate reward
        if action == 3:  # said I don't know
            if self.difficulty == 3 and retrieved == "idk":
                reward = 10.0  # correctly admitted uncertainty
            else:
                reward = -5.0  # wrong to say IDK
        elif success:
            if self.difficulty == 1:
                reward = 10.0
            elif self.difficulty == 2:
                reward = 15.0
            else:
                reward = 20.0
            self.correct_streak += 1
        else:
            reward = -5.0  # hallucination penalty
            self.correct_streak = 0
        
        # Check if done
        terminated = self.correct_streak >= 10
        truncated = self.steps >= self.max_steps
        
        # Pick next question
        idx = self.np_random.integers(0, len(self.faq_data))
        self.current_question = self.faq_data[idx]["question"]
        self.current_answer = self.faq_data[idx]["answer"]
        self.current_keywords = self.faq_data[idx]["keywords"]
        
        obs = self._get_obs()
        info = {
            "question": self.current_question,
            "action_taken": ["keyword", "semantic", "hybrid", "idk"][action],
            "success": success,
            "reward": reward,
            "streak": self.correct_streak,
            "difficulty": self.difficulty
        }
        
        if self.render_mode == "human":
            self.render()
        
        return obs, reward, terminated, truncated, info
    
    def render(self):
        action_names = ["Keyword Search", "Semantic Search", 
                       "Hybrid Search", "Say I Don't Know"]
        print(f"\n{'='*50}")
        print(f"Question: {self.current_question}")
        print(f"Difficulty: {self.difficulty}/3")
        print(f"Correct Streak: {self.correct_streak}/10")
        print(f"{'='*50}")
    
    def get_tasks(self):
        return [
            {
                "id": 1,
                "name": "Basic Query Resolution",
                "difficulty": "easy",
                "description": "Answer simple factual IIT student queries",
                "success_condition": "5 correct answers in a row",
                "reward_target": 50
            },
            {
                "id": 2,
                "name": "Multi-topic Queries",
                "difficulty": "medium", 
                "description": "Handle queries requiring combined knowledge",
                "success_condition": "7 correct answers in a row",
                "reward_target": 100
            },
            {
                "id": 3,
                "name": "Ambiguous Queries",
                "difficulty": "hard",
                "description": "Handle unclear queries, know when to say IDK",
                "success_condition": "10 correct answers in a row",
                "reward_target": 200
            }
        ]
    
    def close(self):
        pass


# Test the environment
if __name__ == "__main__":
    print("Testing CHIRAG Environment...")
    print("="*50)
    
    env = CHIRAGEnv(render_mode="human")
    obs, info = env.reset()
    
    print(f"First Question: {info['question']}")
    print(f"Difficulty: {info['difficulty']}")
    print(f"Observation: {obs}")
    print(f"\nAvailable Tasks:")
    for task in env.get_tasks():
        print(f"  Task {task['id']}: {task['name']} ({task['difficulty']})")
    
    print("\nRunning 5 test steps...")
    total_reward = 0
    
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        print(f"\nStep {i+1}:")
        print(f"  Question: {info['question']}")
        print(f"  Action: {info['action_taken']}")
        print(f"  Success: {info['success']}")
        print(f"  Reward: {reward}")
        print(f"  Streak: {info['streak']}")
        
        if terminated or truncated:
            break
    
    print(f"\nTotal Reward: {total_reward}")
    print("\nCHIRAG Environment working correctly!")
    env.close()