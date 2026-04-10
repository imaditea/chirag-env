"""
CHIRAGEnv: Contextual Helper for IIT Resolution and Guidance
A reinforcement learning environment simulating a student query resolution agent
for IIT online degree programs. Built for the Scaler Meta PyTorch Hackathon.

Framework: OpenEnv (gym-compatible interface)
Author: CHIRAGEnv Team
"""

import numpy as np
import random
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Hardcoded dataset: 20 student questions with ground truth & document chunks
# ---------------------------------------------------------------------------

DATASET = [
    # EASY (difficulty=1)
    {
        "question": "What is the minimum attendance required to appear for the end-term exam?",
        "ground_truth": "75%",
        "chunks": [
            "Students must maintain a minimum attendance of 75% in each course to be eligible for the end-term examination.",
            "Attendance is calculated based on the total number of live sessions conducted during the semester.",
        ],
        "difficulty": 1,
    },
    {
        "question": "When is the fee payment deadline for the current semester?",
        "ground_truth": "last working day of the first month of the semester",
        "chunks": [
            "Fee must be paid by the last working day of the first month of the semester to avoid late charges.",
            "A late fee of Rs. 500 per week is applicable for payments made after the deadline.",
        ],
        "difficulty": 1,
    },
    {
        "question": "How many backlogs is a student allowed to carry to the next semester?",
        "ground_truth": "2",
        "chunks": [
            "A student can carry a maximum of 2 backlogs to the next semester without academic probation.",
            "If backlogs exceed 2, the student must appear for a re-examination before semester registration.",
        ],
        "difficulty": 1,
    },
    {
        "question": "What grade is awarded if a student scores between 80 and 89 percent?",
        "ground_truth": "A",
        "chunks": [
            "Grading scheme: O (90-100), A (80-89), B (70-79), C (60-69), D (50-59), F (below 50).",
            "The final grade is calculated as 40% internal assessment + 60% end-term examination.",
        ],
        "difficulty": 1,
    },
    {
        "question": "What does NOC stand for in the context of IIT online programs?",
        "ground_truth": "No Objection Certificate",
        "chunks": [
            "NOC stands for No Objection Certificate. It is required when a student wishes to pursue an additional certification or internship.",
            "NOC requests must be submitted to the academic office at least 15 days before the activity begins.",
        ],
        "difficulty": 1,
    },
    {
        "question": "What is the passing grade in the IIT online degree program?",
        "ground_truth": "D (50%)",
        "chunks": [
            "The minimum passing grade is D, which corresponds to a score of 50% or above in the course.",
            "Students who score below 50% are awarded an F grade and must re-appear for the supplementary examination.",
        ],
        "difficulty": 1,
    },
    {
        "question": "How many days in advance must a student apply for an NOC?",
        "ground_truth": "15 days",
        "chunks": [
            "NOC requests must be submitted to the academic office at least 15 days before the activity begins.",
            "Late NOC applications may not be processed and can result in the activity being considered unauthorised.",
        ],
        "difficulty": 1,
    },
    # MEDIUM (difficulty=2)
    {
        "question": "What happens if a student fails to pay fees on time AND has more than 2 backlogs?",
        "ground_truth": "late fee and academic probation",
        "chunks": [
            "A late fee of Rs. 500 per week is applicable for payments made after the deadline.",
            "If backlogs exceed 2, the student must appear for a re-examination before semester registration.",
        ],
        "difficulty": 2,
    },
    {
        "question": "Describe the complete grading breakdown and how internal assessment contributes.",
        "ground_truth": "40% internal 60% end-term; O A B C D F grades",
        "chunks": [
            "The final grade is calculated as 40% internal assessment + 60% end-term examination.",
            "Grading scheme: O (90-100), A (80-89), B (70-79), C (60-69), D (50-59), F (below 50).",
        ],
        "difficulty": 2,
    },
    {
        "question": "Can a student with 70% attendance and 3 backlogs register for the next semester?",
        "ground_truth": "No; fails both attendance and backlog criteria",
        "chunks": [
            "Students must maintain a minimum attendance of 75% in each course to be eligible for the end-term examination.",
            "If backlogs exceed 2, the student must appear for a re-examination before semester registration.",
        ],
        "difficulty": 2,
    },
    {
        "question": "What are the consequences of not submitting an NOC application 15 days in advance?",
        "ground_truth": "activity considered unauthorised and NOC may not be processed",
        "chunks": [
            "Late NOC applications may not be processed and can result in the activity being considered unauthorised.",
            "NOC requests must be submitted to the academic office at least 15 days before the activity begins.",
        ],
        "difficulty": 2,
    },
    {
        "question": "How is the end-term examination eligibility and final grade jointly determined?",
        "ground_truth": "75% attendance needed; grade = 40% internal + 60% end-term",
        "chunks": [
            "Students must maintain a minimum attendance of 75% in each course to be eligible for the end-term examination.",
            "The final grade is calculated as 40% internal assessment + 60% end-term examination.",
        ],
        "difficulty": 2,
    },
    {
        "question": "What steps must a student follow if they score below 50% and also need an NOC?",
        "ground_truth": "appear for supplementary exam and apply for NOC 15 days in advance",
        "chunks": [
            "Students who score below 50% are awarded an F grade and must re-appear for the supplementary examination.",
            "NOC requests must be submitted to the academic office at least 15 days before the activity begins.",
        ],
        "difficulty": 2,
    },
    {
        "question": "Explain the late fee policy and how it interacts with semester registration.",
        "ground_truth": "Rs. 500 per week late fee; unpaid dues block registration",
        "chunks": [
            "A late fee of Rs. 500 per week is applicable for payments made after the deadline.",
            "Students with outstanding dues are not permitted to register for the subsequent semester.",
        ],
        "difficulty": 2,
    },
    # HARD (difficulty=3)
    {
        "question": "Can a student negotiate a custom fee installment plan with the institute?",
        "ground_truth": "I do not know",
        "chunks": [
            "Fee must be paid by the last working day of the first month of the semester to avoid late charges.",
            "A late fee of Rs. 500 per week is applicable for payments made after the deadline.",
        ],
        "difficulty": 3,
    },
    {
        "question": "What is the exact process for appealing a grade that a student believes is incorrect?",
        "ground_truth": "I do not know",
        "chunks": [
            "Grading scheme: O (90-100), A (80-89), B (70-79), C (60-69), D (50-59), F (below 50).",
            "The final grade is calculated as 40% internal assessment + 60% end-term examination.",
        ],
        "difficulty": 3,
    },
    {
        "question": "Does the program allow a student to take a semester break without losing enrollment?",
        "ground_truth": "I do not know",
        "chunks": [
            "Students must maintain a minimum attendance of 75% in each course to be eligible for the end-term examination.",
            "A student can carry a maximum of 2 backlogs to the next semester without academic probation.",
        ],
        "difficulty": 3,
    },
    {
        "question": "Are there scholarships available for students from economically weaker sections?",
        "ground_truth": "I do not know",
        "chunks": [
            "Fee must be paid by the last working day of the first month of the semester to avoid late charges.",
            "Students with outstanding dues are not permitted to register for the subsequent semester.",
        ],
        "difficulty": 3,
    },
    {
        "question": "What happens if a student is caught plagiarising their internal assessment?",
        "ground_truth": "I do not know",
        "chunks": [
            "The final grade is calculated as 40% internal assessment + 60% end-term examination.",
            "Students who score below 50% are awarded an F grade and must re-appear for the supplementary examination.",
        ],
        "difficulty": 3,
    },
    {
        "question": "Can students transfer credits from another university to this IIT online program?",
        "ground_truth": "I do not know",
        "chunks": [
            "NOC stands for No Objection Certificate. It is required when a student wishes to pursue an additional certification or internship.",
            "Attendance is calculated based on the total number of live sessions conducted during the semester.",
        ],
        "difficulty": 3,
    },
]


# ---------------------------------------------------------------------------
# Action constants
# ---------------------------------------------------------------------------
ACTION_KEYWORD_SEARCH  = 0
ACTION_SEMANTIC_SEARCH = 1
ACTION_HYBRID_SEARCH   = 2
ACTION_IDK             = 3

ACTION_NAMES = {
    ACTION_KEYWORD_SEARCH:  "Keyword Search",
    ACTION_SEMANTIC_SEARCH: "Semantic Search",
    ACTION_HYBRID_SEARCH:   "Hybrid Search",
    ACTION_IDK:             "I Do Not Know",
}


# ---------------------------------------------------------------------------
# CHIRAGEnv
# ---------------------------------------------------------------------------

class CHIRAGEnv:
    """
    Contextual Helper for IIT Resolution and Guidance - RL Environment.
    """

    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)
        random.seed(seed)
        self.action_space_n = 4
        self._dataset             = DATASET.copy()
        self._current_idx         = 0
        self._consecutive_correct = 0
        self._last_action         = None
        self._last_reward         = 0.0
        self._last_chunks         = []
        self._last_question       = ""
        self._last_difficulty     = 1
        self._episode_step        = 0
        self.state: Dict[str, Any] = {}

    def reset(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        random.shuffle(self._dataset)
        self._current_idx         = 0
        self._consecutive_correct = 0
        self._last_action         = None
        self._last_reward         = 0.0
        self._last_chunks         = []
        self._last_question       = ""
        self._last_difficulty     = 1
        self._episode_step        = 0
        self.state = self._build_state()
        info = {
            "question":   self.state["question"],
            "difficulty": self.state["difficulty_level"],
            "streak":     0,
        }
        return self.state, info

    def step(self, action: int) -> Tuple[Dict[str, Any], float, bool, bool, Dict]:
        assert 0 <= action < self.action_space_n, \
            f"Invalid action {action}. Must be in [0, {self.action_space_n - 1}]."

        record       = self._dataset[self._current_idx]
        difficulty   = record["difficulty"]
        ground_truth = record["ground_truth"].lower()

        retrieved_chunks      = self._simulate_retrieval(action, record)
        self._last_chunks     = retrieved_chunks
        self._last_question   = record["question"]
        self._last_difficulty = difficulty

        combined_text = " ".join(retrieved_chunks).lower()
        answer_found  = ground_truth in combined_text

        reward, correct = self._compute_reward(
            action, answer_found, ground_truth, difficulty
        )

        self._last_action  = action
        self._last_reward  = reward
        self._episode_step += 1

        if correct:
            self._consecutive_correct += 1
        else:
            self._consecutive_correct = 0

        terminated = self._consecutive_correct >= 10
        truncated  = False

        self._current_idx = (self._current_idx + 1) % len(self._dataset)
        self.state = self._build_state()

        info = {
            "difficulty":          difficulty,
            "answer_found":        answer_found,
            "consecutive_correct": self._consecutive_correct,
            "episode_step":        self._episode_step,
            "ground_truth":        record["ground_truth"],
            "question":            record["question"],
            "action_taken":        ACTION_NAMES.get(action, str(action)),
            "success":             correct,
            "streak":              self._consecutive_correct,
        }

        return self.state, reward, terminated, truncated, info

    def render(self, mode: str = "human") -> None:
        sep = "-" * 60
        print(f"\n{sep}")
        print(f"  CHIRAG Env  |  Step {self._episode_step}")
        print(sep)
        print(f"  Question   : {self._last_question or 'N/A'}")
        print(f"  Difficulty : {self._last_difficulty}")
        print(f"  Action     : [{self._last_action}] {ACTION_NAMES.get(self._last_action, 'N/A')}")
        print(f"  Reward     : {self._last_reward:+.1f}")
        print(f"  Consecutive Correct: {self._consecutive_correct}")
        print(sep)

    def get_tasks(self) -> List[Dict[str, Any]]:
        """Return the 3 task descriptors with grader information."""
        return [
            {
                "id":          "task_easy",
                "name":        "Easy - Direct Factual Retrieval",
                "difficulty":  "easy",
                "reward_range": [0.0, 1.0],
                "grader":      "graders/easy_grader.py",
                "description": "Answer simple factual questions whose answer is directly contained in a single document chunk.",
            },
            {
                "id":          "task_medium",
                "name":        "Medium - Multi-Chunk Reasoning",
                "difficulty":  "medium",
                "reward_range": [0.0, 1.0],
                "grader":      "graders/medium_grader.py",
                "description": "Answer multi-part questions requiring information from two different document chunks.",
            },
            {
                "id":          "task_hard",
                "name":        "Hard - Ambiguity Detection",
                "difficulty":  "hard",
                "reward_range": [0.0, 1.0],
                "grader":      "graders/hard_grader.py",
                "description": "Handle ambiguous questions where the answer is NOT in any document chunk.",
            },
        ]

    def _build_state(self) -> Dict[str, Any]:
        record = self._dataset[self._current_idx]
        return {
            "question":         record["question"],
            "retrieved_chunks": [],
            "difficulty_level": record["difficulty"],
        }

    def _simulate_retrieval(self, action: int, record: Dict[str, Any]) -> List[str]:
        chunks = record["chunks"]
        if action == ACTION_KEYWORD_SEARCH:
            return [chunks[0]]
        elif action == ACTION_SEMANTIC_SEARCH:
            return list(chunks)
        elif action == ACTION_HYBRID_SEARCH:
            noise = "General program information is available on the official portal."
            return list(chunks) + [noise]
        elif action == ACTION_IDK:
            return ["[Agent elected not to retrieve - responding with I do not know]"]
        return []
def _compute_reward(self, action: int, answer_found: bool, ground_truth: str, difficulty: int) -> Tuple[float, bool]:
    if difficulty == 3:
        if action == ACTION_IDK:
            return 0.999, True
        else:
            return 0.001, False
    if difficulty == 1:
        if answer_found and action != ACTION_IDK:
            return 0.999, True
        elif action == ACTION_IDK:
            return 0.001, False
        else:
            return 0.001, False
    if difficulty == 2:
        if action == ACTION_IDK:
            return 0.001, False
        if answer_found:
            if action in (ACTION_SEMANTIC_SEARCH, ACTION_HYBRID_SEARCH):
                return 0.999, True
            else:
                return 0.5, False
        else:
            return 0.001, False
    return 0.5, False
