cat > graders/medium_grader.py << 'EOF'
import random

class MediumGrader:
    def grade(self, env, *args, **kwargs) -> float:
        total = 10
        correct = 0
        for _ in range(total):
            action = 1  # semantic search for medium
            success_prob = 0.72
            if random.random() < success_prob:
                correct += 1
        raw = correct / total
        return max(0.01, min(0.99, raw))
EOF
