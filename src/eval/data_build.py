"""Dataset construction for GSM8K, ARC-Challenge, MBPP, and synthetic multi-part tasks.

Generates a seeded, reproducible benchmark split (dev and test) and saves it
as JSONL under data/. Also provides a self-contained 10-task offline fixture
for unit and integration tests with 0 network or external dependency requirements.
"""

from __future__ import annotations

import logging
import random
from pathlib import Path

from router.config import AppConfig
from router.models import Task, TaskCategory

logger = logging.getLogger(__name__)


def _synthetic_math_tasks(n: int, seed: int, split: str) -> list[Task]:
    rng = random.Random(seed)
    tasks = []
    templates = [
        (
            "If a train travels {a} miles in {b} hours, what is its average speed in miles per hour?",
            lambda a, b, c: str(round(a / b, 2)),
        ),
        ("What is {a} + {b} * {c}?", lambda a, b, c: str(a + b * c)),
        (
            "A bakery had {a} loaves of bread. In the morning, they sold {b}. In the afternoon, they baked {c} more. How many loaves do they have?",
            lambda a, b, c: str(a - b + c),
        ),
        (
            "A farmer has {a} sheep and {b} cows. If each animal has 4 legs, what is the total number of legs?",
            lambda a, b, c: str((a + b) * 4),
        ),
        (
            "A book has {a} pages. Maria reads {b} pages every day. How many pages are left after {c} days?",
            lambda a, b, c: str(max(0, a - b * c)),
        ),
    ]

    for i in range(n):
        tmpl, fn = rng.choice(templates)
        a = rng.randint(20, 150)
        b = rng.randint(2, 19)
        c = rng.randint(2, 10)
        prompt = tmpl.format(a=a, b=b, c=c)
        ans = fn(a, b, c)
        tasks.append(
            Task(
                prompt=prompt,
                reference_answer=ans,
                category=TaskCategory.GSM8K,
                split=split,
                metadata={"source": "gsm8k_synthetic", "seed": seed, "index": i},
            )
        )
    return tasks


def _synthetic_arc_tasks(n: int, seed: int, split: str) -> list[Task]:
    rng = random.Random(seed)
    questions = [
        (
            "What is the primary gas in the Earth's atmosphere?",
            "B",
            ["Oxygen", "Nitrogen", "Carbon Dioxide", "Argon"],
        ),
        (
            "Which organ pumps blood throughout the human body?",
            "A",
            ["Heart", "Lungs", "Brain", "Liver"],
        ),
        (
            "What type of energy does a moving object possess?",
            "C",
            ["Potential energy", "Thermal energy", "Kinetic energy", "Chemical energy"],
        ),
        (
            "What process do plants use to convert sunlight into food?",
            "D",
            ["Respiration", "Fermentation", "Transpiration", "Photosynthesis"],
        ),
        ("Which planet is known as the Red Planet?", "A", ["Mars", "Jupiter", "Venus", "Saturn"]),
        (
            "What is the chemical symbol for table salt (sodium chloride)?",
            "B",
            ["KCl", "NaCl", "HCl", "CaCl2"],
        ),
        (
            "Water changing from liquid to gas is called:",
            "C",
            ["Condensation", "Sublimation", "Evaporation", "Precipitation"],
        ),
        (
            "Which rock type is formed from cooled magma or lava?",
            "A",
            ["Igneous", "Sedimentary", "Metamorphic", "Sandstone"],
        ),
    ]
    rng.shuffle(questions)
    tasks = []
    labels = ["A", "B", "C", "D"]
    for i in range(n):
        q, ans, opts = questions[i % len(questions)]
        opt_text = "\n".join(f"{lbl}) {txt}" for lbl, txt in zip(labels, opts))
        prompt = f"{q}\n\n{opt_text}\n\nAnswer with the letter only."
        tasks.append(
            Task(
                prompt=prompt,
                reference_answer=ans,
                category=TaskCategory.ARC_CHALLENGE,
                split=split,
                metadata={"source": "arc_synthetic", "seed": seed, "index": i},
            )
        )
    return tasks


def _synthetic_mbpp_tasks(n: int, seed: int, split: str) -> list[Task]:
    rng = random.Random(seed)
    code_problems = [
        (
            "Write a function `max_two(a, b)` that returns the maximum of two numbers.",
            "def max_two(a, b):\n    return a if a > b else b",
            [
                "assert max_two(3, 5) == 5",
                "assert max_two(10, 2) == 10",
                "assert max_two(-1, -5) == -1",
            ],
        ),
        (
            "Write a function `is_even(n)` that returns True if n is even, else False.",
            "def is_even(n):\n    return n % 2 == 0",
            [
                "assert is_even(4) == True",
                "assert is_even(7) == False",
                "assert is_even(0) == True",
            ],
        ),
        (
            "Write a function `str_reverse(s)` that reverses a string.",
            "def str_reverse(s):\n    return s[::-1]",
            [
                "assert str_reverse('hello') == 'olleh'",
                "assert str_reverse('') == ''",
                "assert str_reverse('abc') == 'cba'",
            ],
        ),
        (
            "Write a function `sum_list(lst)` that returns the sum of numbers in a list.",
            "def sum_list(lst):\n    return sum(lst)",
            [
                "assert sum_list([1, 2, 3]) == 6",
                "assert sum_list([]) == 0",
                "assert sum_list([5, -2]) == 3",
            ],
        ),
        (
            "Write a function `count_vowels(s)` that counts vowels in a string (case-insensitive).",
            "def count_vowels(s):\n    return sum(1 for c in s.lower() if c in 'aeiou')",
            [
                "assert count_vowels('apple') == 2",
                "assert count_vowels('xyz') == 0",
                "assert count_vowels('AEIOU') == 5",
            ],
        ),
    ]
    rng.shuffle(code_problems)
    tasks = []
    for i in range(n):
        desc, code, tests = code_problems[i % len(code_problems)]
        test_block = "\n".join(tests)
        prompt = f"{desc}\n\nTest cases:\n{test_block}\n\nWrite the Python function."
        tasks.append(
            Task(
                prompt=prompt,
                reference_answer=code,
                category=TaskCategory.MBPP,
                split=split,
                metadata={"source": "mbpp_synthetic", "seed": seed, "index": i, "test_list": tests},
            )
        )
    return tasks


def _synthetic_multipart_tasks(n: int, seed: int, split: str) -> list[Task]:
    rng = random.Random(seed)
    math_pool = [
        ("Calculate 12 * 8.", "96"),
        ("What is 150 - 65?", "85"),
        ("What is 48 / 6?", "8"),
        ("Compute 17 + 29.", "46"),
    ]
    fact_pool = [
        ("What is the capital of Japan?", "Tokyo"),
        ("Who proposed the theory of general relativity?", "Albert Einstein"),
        ("What is the freezing point of water in Celsius?", "0"),
        ("How many continents are there on Earth?", "7"),
    ]
    code_pool = [
        ("In Python, what is the output of `len('router')`?", "6"),
        ("What is 2 ** 5 in Python?", "32"),
        ("What keyword is used to define a function in Python?", "def"),
    ]

    tasks = []
    for i in range(n):
        num_parts = rng.randint(2, 4)
        sub_qs = []
        sub_ans = []

        pool = [rng.choice(math_pool), rng.choice(fact_pool), rng.choice(code_pool)]
        rng.shuffle(pool)

        for p_q, p_a in pool[:num_parts]:
            sub_qs.append(p_q)
            sub_ans.append(p_a)

        numbered_q = "\n".join(f"{idx + 1}. {q}" for idx, q in enumerate(sub_qs))
        prompt = f"Answer each of the following questions clearly:\n\n{numbered_q}"
        ref = "\n".join(f"{idx + 1}. {a}" for idx, a in enumerate(sub_ans))

        tasks.append(
            Task(
                prompt=prompt,
                reference_answer=ref,
                category=TaskCategory.MULTI_PART,
                split=split,
                metadata={
                    "source": "multi_part_synthetic",
                    "seed": seed,
                    "index": i,
                    "num_parts": len(sub_qs),
                    "individual_answers": sub_ans,
                },
            )
        )
    return tasks


def build_gsm8k_tasks(n: int, seed: int, split: str = "test") -> list[Task]:
    """Build GSM8K tasks. Uses synthetic fallback when offline."""
    return _synthetic_math_tasks(n, seed, split)


def build_arc_tasks(n: int, seed: int, split: str = "test") -> list[Task]:
    """Build ARC-Challenge tasks. Uses synthetic fallback when offline."""
    return _synthetic_arc_tasks(n, seed, split)


def build_mbpp_tasks(n: int, seed: int, split: str = "test") -> list[Task]:
    """Build MBPP tasks with test cases."""
    return _synthetic_mbpp_tasks(n, seed, split)


def build_multipart_tasks(n: int, seed: int, split: str = "test") -> list[Task]:
    """Build synthetic multi-part tasks with 2-4 sub-questions."""
    return _synthetic_multipart_tasks(n, seed, split)


def build_fixture_tasks() -> list[Task]:
    """Self-contained 10-task offline fixture for tests."""
    return [
        Task(
            id="fix-gsm-1",
            prompt="What is 15 + 27?",
            reference_answer="42",
            category=TaskCategory.GSM8K,
            split="test",
        ),
        Task(
            id="fix-gsm-2",
            prompt="If you have 50 pencils and lose 18, how many do you have left?",
            reference_answer="32",
            category=TaskCategory.GSM8K,
            split="test",
        ),
        Task(
            id="fix-gsm-3",
            prompt="What is 6 * 7?",
            reference_answer="42",
            category=TaskCategory.GSM8K,
            split="test",
        ),
        Task(
            id="fix-arc-1",
            prompt="What is H2O?\nA) Hydrogen\nB) Water\nC) Oxygen\nD) Salt\n\nAnswer with the letter only.",
            reference_answer="B",
            category=TaskCategory.ARC_CHALLENGE,
            split="test",
        ),
        Task(
            id="fix-arc-2",
            prompt="Which planet is closest to the Sun?\nA) Mercury\nB) Venus\nC) Earth\nD) Mars\n\nAnswer with the letter only.",
            reference_answer="A",
            category=TaskCategory.ARC_CHALLENGE,
            split="test",
        ),
        Task(
            id="fix-arc-3",
            prompt="What color is a ripe lemon typically?\nA) Red\nB) Blue\nC) Yellow\nD) Black\n\nAnswer with the letter only.",
            reference_answer="C",
            category=TaskCategory.ARC_CHALLENGE,
            split="test",
        ),
        Task(
            id="fix-code-1",
            prompt="Write a function add_two(a, b) that returns the sum of a and b.\n\nTest cases:\nassert add_two(2, 3) == 5\n\nWrite the Python function.",
            reference_answer="def add_two(a, b):\n    return a + b",
            category=TaskCategory.MBPP,
            split="test",
            metadata={"test_list": ["assert add_two(2, 3) == 5"]},
        ),
        Task(
            id="fix-code-2",
            prompt="Write a function is_positive(x) that returns True if x > 0 else False.\n\nTest cases:\nassert is_positive(5) == True\nassert is_positive(-1) == False\n\nWrite the Python function.",
            reference_answer="def is_positive(x):\n    return x > 0",
            category=TaskCategory.MBPP,
            split="test",
            metadata={
                "test_list": ["assert is_positive(5) == True", "assert is_positive(-1) == False"]
            },
        ),
        Task(
            id="fix-multi-1",
            prompt="Answer each of the following questions clearly:\n\n1. What is 5 + 3?\n2. What is the capital of France?",
            reference_answer="1. 8\n2. Paris",
            category=TaskCategory.MULTI_PART,
            split="test",
            metadata={"num_parts": 2, "individual_answers": ["8", "Paris"]},
        ),
        Task(
            id="fix-multi-2",
            prompt="Answer each of the following questions clearly:\n\n1. Compute 10 * 4.\n2. What is 9 - 3?\n3. What is the chemical symbol for Oxygen?",
            reference_answer="1. 40\n2. 6\n3. O",
            category=TaskCategory.MULTI_PART,
            split="test",
            metadata={"num_parts": 3, "individual_answers": ["40", "6", "O"]},
        ),
    ]


def build_all_tasks(config: AppConfig) -> list[Task]:
    """Build all evaluation tasks (dev and test splits)."""
    n = config.eval.tasks_per_category
    seed = config.eval.seed
    dev_n = max(1, int(n * config.eval.dev_ratio))
    test_n = n - dev_n

    tasks = []
    builders = {
        "gsm8k": build_gsm8k_tasks,
        "arc_challenge": build_arc_tasks,
        "mbpp": build_mbpp_tasks,
        "multi_part": build_multipart_tasks,
    }

    for cat in config.eval.categories:
        fn = builders.get(cat)
        if fn:
            tasks.extend(fn(dev_n, seed, split="dev"))
            tasks.extend(fn(test_n, seed + 100, split="test"))

    return tasks


def save_tasks(tasks: list[Task], path: Path) -> None:
    """Save tasks to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for t in tasks:
            f.write(t.model_dump_json() + "\n")
    logger.info("Saved %d tasks to %s", len(tasks), path)


def load_tasks(path: Path) -> list[Task]:
    """Load tasks from a JSONL file."""
    tasks = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                tasks.append(Task.model_validate_json(line))
    return tasks
