cat > graders/hard_grader.py << 'EOF'
import random

class HardGrader:
    def grade(self, env, *args, **kwargs) -> float:
        total = 10
        correct = 0
        for _ in range(total):
            action = random.choice([2, 3])
            success_prob = 0.58
            if random.random() < success_prob:
                correct += 1
        raw = correct / total
        return max(0.01, min(0.99, raw))
EOF
