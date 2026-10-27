"""Data pipeline for the edge router model.

Builds a separate training set from the train splits of GSM8K, ARC-Challenge,
and MBPP (disjoint from evaluation dev/test splits, verified via hash checking),
labels them using actual local model execution (or mock client in offline mode),
and caches the results on disk.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import logging
import random
import sys
from pathlib import Path
from typing import Any

from eval.scoring import score_task
from router.clients.base import LocalClient
from router.models import Task, TaskCategory

logger = logging.getLogger(__name__)


def _normalize_prompt(text: str) -> str:
    """Normalize whitespace and case for robust collision checking."""
    return " ".join(text.strip().lower().split())


def _prompt_hash(text: str) -> str:
    """SHA-256 hash of normalized prompt text."""
    return hashlib.sha256(_normalize_prompt(text).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# 20-task bundled offline fixture (zero network, instant execution)
# ---------------------------------------------------------------------------


def build_edge_fixture_tasks() -> list[Task]:
    """Return a fixed 20-prompt fixture (7 GSM8K, 7 ARC, 6 MBPP) for offline testing."""
    tasks: list[Task] = [
        # GSM8K (7 math prompts)
        Task(
            id="edge-fix-gsm-1",
            prompt="A bakery prepares 45 loaves of sourdough. They sell 28 in the morning and bake 15 more in the afternoon. How many loaves are currently in the bakery?",
            reference_answer="32",
            category=TaskCategory.GSM8K,
            split="train",
        ),
        Task(
            id="edge-fix-gsm-2",
            prompt="Marcus buys 4 packs of batteries for $6 each and a flashlight for $14. What is the total cost in dollars?",
            reference_answer="38",
            category=TaskCategory.GSM8K,
            split="train",
        ),
        Task(
            id="edge-fix-gsm-3",
            prompt="A delivery van carries 80 parcels. At the first stop, the driver delivers 25 parcels. At the second stop, 18 parcels are delivered. How many parcels remain in the van?",
            reference_answer="37",
            category=TaskCategory.GSM8K,
            split="train",
        ),
        Task(
            id="edge-fix-gsm-4",
            prompt="An athlete runs 6 miles every day from Monday to Friday, and 10 miles on Saturday. How many total miles does the athlete run that week?",
            reference_answer="40",
            category=TaskCategory.GSM8K,
            split="train",
        ),
        Task(
            id="edge-fix-gsm-5",
            prompt="A school library has 120 science books. The librarian buys 35 new books and donates 15 damaged ones. How many science books are in the library now?",
            reference_answer="140",
            category=TaskCategory.GSM8K,
            split="train",
        ),
        Task(
            id="edge-fix-gsm-6",
            prompt="Sophia earns $15 per hour gardening. If she works 5 hours on Saturday and 4 hours on Sunday, how much money does she make in total?",
            reference_answer="135",
            category=TaskCategory.GSM8K,
            split="train",
        ),
        Task(
            id="edge-fix-gsm-7",
            prompt="A carton contains 36 eggs. A baker uses 8 eggs for a cake and 12 eggs for muffins. How many eggs are left in the carton?",
            reference_answer="16",
            category=TaskCategory.GSM8K,
            split="train",
        ),
        # ARC-Challenge (7 reasoning/science prompts)
        Task(
            id="edge-fix-arc-1",
            prompt="Which layer of Earth's interior is believed to generate its magnetic field?\nA) Crust\nB) Outer Core\nC) Inner Core\nD) Asthenosphere\n\nAnswer with the letter only.",
            reference_answer="B",
            category=TaskCategory.ARC_CHALLENGE,
            split="train",
        ),
        Task(
            id="edge-fix-arc-2",
            prompt="What form of energy is stored in the chemical bonds of glucose molecules?\nA) Nuclear energy\nB) Kinetic energy\nC) Potential energy\nD) Thermal energy\n\nAnswer with the letter only.",
            reference_answer="C",
            category=TaskCategory.ARC_CHALLENGE,
            split="train",
        ),
        Task(
            id="edge-fix-arc-3",
            prompt="Which organelle is responsible for synthesizing proteins inside animal cells?\nA) Lysosome\nB) Ribosome\nC) Vacuole\nD) Centrosome\n\nAnswer with the letter only.",
            reference_answer="B",
            category=TaskCategory.ARC_CHALLENGE,
            split="train",
        ),
        Task(
            id="edge-fix-arc-4",
            prompt="What state of matter exists at extremely high temperatures where electrons are stripped from nuclei?\nA) Plasma\nB) Gas\nC) Solid\nD) Supercritical fluid\n\nAnswer with the letter only.",
            reference_answer="A",
            category=TaskCategory.ARC_CHALLENGE,
            split="train",
        ),
        Task(
            id="edge-fix-arc-5",
            prompt="Which gas constitutes approximately 21 percent of the volume of dry air in Earth's atmosphere?\nA) Nitrogen\nB) Oxygen\nC) Carbon Dioxide\nD) Argon\n\nAnswer with the letter only.",
            reference_answer="B",
            category=TaskCategory.ARC_CHALLENGE,
            split="train",
        ),
        Task(
            id="edge-fix-arc-6",
            prompt="When liquid water turns into ice, what happens to its density?\nA) Increases\nB) Decreases\nC) Remains constant\nD) Fluctuates randomly\n\nAnswer with the letter only.",
            reference_answer="B",
            category=TaskCategory.ARC_CHALLENGE,
            split="train",
        ),
        Task(
            id="edge-fix-arc-7",
            prompt="Which blood vessels carry oxygen-depleted blood back toward the heart from body tissues?\nA) Arteries\nB) Veins\nC) Capillaries\nD) Arterioles\n\nAnswer with the letter only.",
            reference_answer="B",
            category=TaskCategory.ARC_CHALLENGE,
            split="train",
        ),
        # MBPP (6 coding prompts)
        Task(
            id="edge-fix-mbpp-1",
            prompt="Write a function `square(n)` that returns the square of an integer.\n\nTest cases:\nassert square(4) == 16\nassert square(-3) == 9\n\nWrite the Python function.",
            reference_answer="def square(n):\n    return n * n",
            category=TaskCategory.MBPP,
            split="train",
            metadata={"test_list": ["assert square(4) == 16", "assert square(-3) == 9"]},
        ),
        Task(
            id="edge-fix-mbpp-2",
            prompt="Write a function `is_odd(n)` that returns True if n is odd, else False.\n\nTest cases:\nassert is_odd(5) == True\nassert is_odd(8) == False\n\nWrite the Python function.",
            reference_answer="def is_odd(n):\n    return n % 2 != 0",
            category=TaskCategory.MBPP,
            split="train",
            metadata={"test_list": ["assert is_odd(5) == True", "assert is_odd(8) == False"]},
        ),
        Task(
            id="edge-fix-mbpp-3",
            prompt="Write a function `find_min(lst)` that returns the smallest number in a non-empty list.\n\nTest cases:\nassert find_min([5, 2, 9, 1]) == 1\nassert find_min([-3, -7, 0]) == -7\n\nWrite the Python function.",
            reference_answer="def find_min(lst):\n    return min(lst)",
            category=TaskCategory.MBPP,
            split="train",
            metadata={"test_list": ["assert find_min([5, 2, 9, 1]) == 1", "assert find_min([-3, -7, 0]) == -7"]},
        ),
        Task(
            id="edge-fix-mbpp-4",
            prompt="Write a function `product_list(lst)` that computes the product of all integers in a list.\n\nTest cases:\nassert product_list([2, 3, 4]) == 24\nassert product_list([5, 0]) == 0\n\nWrite the Python function.",
            reference_answer="def product_list(lst):\n    p = 1\n    for x in lst:\n        p *= x\n    return p",
            category=TaskCategory.MBPP,
            split="train",
            metadata={"test_list": ["assert product_list([2, 3, 4]) == 24", "assert product_list([5, 0]) == 0"]},
        ),
        Task(
            id="edge-fix-mbpp-5",
            prompt="Write a function `string_length(s)` that returns the length of a string without using built-in len().\n\nTest cases:\nassert string_length('hello') == 5\nassert string_length('') == 0\n\nWrite the Python function.",
            reference_answer="def string_length(s):\n    c = 0\n    for _ in s:\n        c += 1\n    return c",
            category=TaskCategory.MBPP,
            split="train",
            metadata={"test_list": ["assert string_length('hello') == 5", "assert string_length('') == 0"]},
        ),
        Task(
            id="edge-fix-mbpp-6",
            prompt="Write a function `count_digits(s)` that counts how many numeric digit characters are in string s.\n\nTest cases:\nassert count_digits('item456') == 3\nassert count_digits('word') == 0\n\nWrite the Python function.",
            reference_answer="def count_digits(s):\n    return sum(1 for c in s if c.isdigit())",
            category=TaskCategory.MBPP,
            split="train",
            metadata={"test_list": ["assert count_digits('item456') == 3", "assert count_digits('word') == 0"]},
        ),
    ]
    return tasks


# ---------------------------------------------------------------------------
# Synthetic prompt generators (disjoint from eval set templates/seeds)
# ---------------------------------------------------------------------------


def _generate_synthetic_gsm8k(n: int, seed: int, forbidden_hashes: set[str]) -> list[Task]:
    rng = random.Random(seed)
    tasks: list[Task] = []
    templates = [
        (
            "A grocery store had {a} boxes of cereal. They received a delivery of {b} more boxes. By evening, they sold {c} boxes. How many boxes remain?",
            lambda a, b, c: str(a + b - c),
        ),
        (
            "Liam buys {a} notebooks at ${b} each and {c} pens at $2 each. What is the total cost in dollars?",
            lambda a, b, c: str(a * b + c * 2),
        ),
        (
            "A warehouse holds {a} pallets of goods. A forklift can transport {b} pallets at a time. After {c} trips, how many pallets are still in the warehouse?",
            lambda a, b, c: str(max(0, a - b * c)),
        ),
        (
            "An orchard harvests {a} kilograms of apples. They discard {b} kilograms of damaged apples and pack the rest equally into {c} crates. How many kilograms are in each crate?",
            lambda a, b, c: str(round((a - b) / c, 2)),
        ),
        (
            "A cyclist travels {a} miles in the morning at an average speed of {b} miles per hour. In the afternoon, they travel another {c} miles. How many total miles were traveled?",
            lambda a, b, c: str(a + c),
        ),
        (
            "A classroom has {a} students. If {b} students are absent today and {c} students join from another room, how many students are present?",
            lambda a, b, c: str(a - b + c),
        ),
    ]

    attempt = 0
    while len(tasks) < n and attempt < n * 30:
        attempt += 1
        tmpl, fn = rng.choice(templates)
        a = rng.randint(25, 200)
        b = rng.randint(3, 20)
        c = rng.randint(2, 12)
        if tmpl.find("crates") != -1:
            a = b + c * rng.randint(2, 10)
        prompt = tmpl.format(a=a, b=b, c=c)
        ans = fn(a, b, c)
        p_hash = _prompt_hash(prompt)
        if p_hash in forbidden_hashes:
            continue
        forbidden_hashes.add(p_hash)
        tasks.append(
            Task(
                id=f"edge-gsm-{len(tasks) + 1}",
                prompt=prompt,
                reference_answer=ans,
                category=TaskCategory.GSM8K,
                split="train",
                metadata={"source": "gsm8k_train_synthetic", "seed": seed, "index": len(tasks)},
            )
        )
    return tasks


def _generate_synthetic_arc(n: int, seed: int, forbidden_hashes: set[str]) -> list[Task]:
    rng = random.Random(seed)
    tasks: list[Task] = []

    elements = [
        ("Hydrogen", "H"), ("Helium", "He"), ("Lithium", "Li"), ("Carbon", "C"),
        ("Nitrogen", "N"), ("Oxygen", "O"), ("Sodium", "Na"), ("Magnesium", "Mg"),
        ("Aluminum", "Al"), ("Silicon", "Si"), ("Phosphorus", "P"), ("Sulfur", "S"),
        ("Chlorine", "Cl"), ("Potassium", "K"), ("Calcium", "Ca"), ("Iron", "Fe"),
        ("Copper", "Cu"), ("Zinc", "Zn"), ("Silver", "Ag"), ("Gold", "Au"),
    ]
    units = [
        ("force", "Newton"), ("energy", "Joule"), ("power", "Watt"),
        ("frequency", "Hertz"), ("pressure", "Pascal"), ("electric current", "Ampere"),
        ("voltage", "Volt"), ("electrical resistance", "Ohm"), ("temperature", "Kelvin"),
    ]
    planets = [
        ("closest planet to the Sun", "Mercury"), ("hottest planet in the solar system", "Venus"),
        ("planet known for prominent rings", "Saturn"), ("largest planet in the solar system", "Jupiter"),
        ("planet known as the Red Planet", "Mars"), ("farthest planet from the Sun", "Neptune"),
    ]
    organs = [
        ("pumps blood throughout the circulatory system", "Heart"),
        ("filters waste from blood to form urine", "Kidney"),
        ("produces bile and detoxifies chemicals", "Liver"),
        ("exchanges oxygen and carbon dioxide with atmosphere", "Lungs"),
        ("secretes digestive enzymes and insulin", "Pancreas"),
    ]
    facts = [
        ("Which layer of the Earth is composed predominantly of liquid iron and nickel?", "Outer Core", ["Crust", "Outer Core", "Inner Core", "Lithosphere"]),
        ("What type of electromagnetic radiation has wavelengths shorter than visible light but longer than X-rays?", "Ultraviolet light", ["Ultraviolet light", "Infrared radiation", "Radio waves", "Microwaves"]),
        ("In cellular respiration, what organelle produces the majority of ATP?", "Mitochondria", ["Ribosome", "Endoplasmic reticulum", "Mitochondria", "Golgi apparatus"]),
        ("Which mineral has a hardness rating of 10 on the Mohs scale?", "Diamond", ["Diamond", "Corundum", "Topaz", "Quartz"]),
        ("When warm air rises and cool air sinks, heat is transferred by which mechanism?", "Convection", ["Conduction", "Radiation", "Convection", "Advection"]),
        ("What subatomic particle carries a negative electrical charge?", "Electron", ["Electron", "Proton", "Neutron", "Positron"]),
    ]

    attempt = 0
    max_attempts = n * 50
    labels = ["A", "B", "C", "D"]

    while len(tasks) < n and attempt < max_attempts:
        attempt += 1
        q_type = rng.choice(["element", "unit", "planet", "organ", "fact"])
        if q_type == "element":
            elem, sym = rng.choice(elements)
            wrong = [s for _, s in elements if s != sym]
            rng.shuffle(wrong)
            opts = [sym] + wrong[:3]
            rng.shuffle(opts)
            correct_lbl = labels[opts.index(sym)]
            opt_text = "\n".join(f"{lbl}) {txt}" for lbl, txt in zip(labels, opts))
            prompt = f"What is the chemical symbol for {elem}?\n\n{opt_text}\n\nAnswer with the letter only."
            ans = correct_lbl
        elif q_type == "unit":
            quant, unit = rng.choice(units)
            wrong = [u for _, u in units if u != unit]
            rng.shuffle(wrong)
            opts = [unit] + wrong[:3]
            rng.shuffle(opts)
            correct_lbl = labels[opts.index(unit)]
            opt_text = "\n".join(f"{lbl}) {txt}" for lbl, txt in zip(labels, opts))
            prompt = f"In the International System of Units (SI), what is the standard unit of {quant}?\n\n{opt_text}\n\nAnswer with the letter only."
            ans = correct_lbl
        elif q_type == "planet":
            desc, name = rng.choice(planets)
            wrong = [p for _, p in planets if p != name]
            rng.shuffle(wrong)
            opts = [name] + wrong[:3]
            rng.shuffle(opts)
            correct_lbl = labels[opts.index(name)]
            opt_text = "\n".join(f"{lbl}) {txt}" for lbl, txt in zip(labels, opts))
            prompt = f"Which celestial body in our solar system is the {desc}?\n\n{opt_text}\n\nAnswer with the letter only."
            ans = correct_lbl
        elif q_type == "organ":
            func, organ_name = rng.choice(organs)
            wrong = [o for _, o in organs if o != organ_name]
            rng.shuffle(wrong)
            opts = [organ_name] + wrong[:3]
            rng.shuffle(opts)
            correct_lbl = labels[opts.index(organ_name)]
            opt_text = "\n".join(f"{lbl}) {txt}" for lbl, txt in zip(labels, opts))
            prompt = f"Which human organ {func}?\n\n{opt_text}\n\nAnswer with the letter only."
            ans = correct_lbl
        else:
            q_text, correct_txt, choices = rng.choice(facts)
            opts = list(choices)
            rng.shuffle(opts)
            correct_lbl = labels[opts.index(correct_txt)]
            opt_text = "\n".join(f"{lbl}) {txt}" for lbl, txt in zip(labels, opts))
            prompt = f"{q_text}\n\n{opt_text}\n\nAnswer with the letter only."
            ans = correct_lbl

        p_hash = _prompt_hash(prompt)
        if p_hash in forbidden_hashes:
            continue
        forbidden_hashes.add(p_hash)
        tasks.append(
            Task(
                id=f"edge-arc-{len(tasks) + 1}",
                prompt=prompt,
                reference_answer=ans,
                category=TaskCategory.ARC_CHALLENGE,
                split="train",
                metadata={"source": "arc_train_synthetic", "seed": seed, "index": len(tasks)},
            )
        )
    return tasks


def _generate_synthetic_mbpp(n: int, seed: int, forbidden_hashes: set[str]) -> list[Task]:
    rng = random.Random(seed)
    tasks: list[Task] = []

    base_templates = [
        ("Write a function `multiply_{v}(n)` that multiplies an integer by {v}.", "def multiply_{v}(n):\n    return n * {v}", ["assert multiply_{v}(3) == {ans1}", "assert multiply_{v}(-2) == {ans2}"]),
        ("Write a function `add_{v}(n)` that adds {v} to an integer.", "def add_{v}(n):\n    return n + {v}", ["assert add_{v}(5) == {ans1}", "assert add_{v}(0) == {ans2}"]),
        ("Write a function `is_divisible_by_{v}(n)` that returns True if n is divisible by {v}, else False.", "def is_divisible_by_{v}(n):\n    return n % {v} == 0", ["assert is_divisible_by_{v}({v}) == True", "assert is_divisible_by_{v}({v1}) == False"]),
        ("Write a function `power_{v}(n)` that raises an integer n to the power {v}.", "def power_{v}(n):\n    return n ** {v}", ["assert power_{v}(2) == {ans1}", "assert power_{v}(3) == {ans2}"]),
        ("Write a function `clamp_max_{v}(lst)` that replaces all numbers in list greater than {v} with {v}.", "def clamp_max_{v}(lst):\n    return [min(x, {v}) for x in lst]", ["assert clamp_max_{v}([1, {v1}, 0]) == [1, {v}, 0]"]),
        ("Write a function `drop_first_{v}(lst)` that drops the first {v} elements of a list.", "def drop_first_{v}(lst):\n    return lst[{v}:]", ["assert drop_first_{v}([1, 2, 3, 4, 5]) == [1, 2, 3, 4, 5][{v}:]"]),
    ]

    attempt = 0
    max_attempts = n * 50

    while len(tasks) < n and attempt < max_attempts:
        attempt += 1
        desc_tmpl, code_tmpl, test_tmpls = rng.choice(base_templates)
        v = rng.randint(2, 50)
        v1 = v + 1
        ans1_mul = 3 * v
        ans2_mul = -2 * v
        ans1_add = 5 + v
        ans2_add = v
        ans1_pow = 2 ** min(v % 6 + 1, 5)
        ans2_pow = 3 ** min(v % 6 + 1, 5)

        subs = {
            "v": v, "v1": v1,
            "ans1": ans1_mul if "multiply" in desc_tmpl else (ans1_add if "add" in desc_tmpl else ans1_pow),
            "ans2": ans2_mul if "multiply" in desc_tmpl else (ans2_add if "add" in desc_tmpl else ans2_pow),
        }
        desc = desc_tmpl.format(**subs)
        code = code_tmpl.format(**subs)
        tests = [t.format(**subs) for t in test_tmpls]
        test_block = "\n".join(tests)
        prompt = f"{desc}\n\nTest cases:\n{test_block}\n\nWrite the Python function."

        p_hash = _prompt_hash(prompt)
        if p_hash in forbidden_hashes:
            continue
        forbidden_hashes.add(p_hash)
        tasks.append(
            Task(
                id=f"edge-mbpp-{len(tasks) + 1}",
                prompt=prompt,
                reference_answer=code,
                category=TaskCategory.MBPP,
                split="train",
                metadata={"source": "mbpp_train_synthetic", "seed": seed, "index": len(tasks), "test_list": tests},
            )
        )
    return tasks


# ---------------------------------------------------------------------------
# HuggingFace loader with explicit 60s timeout and unbuffered progress
# ---------------------------------------------------------------------------


def _load_hf_dataset_with_timeout(name: str, config_name: str | None, split: str, timeout: float = 60.0) -> Any:
    """Load a HuggingFace dataset inside a ThreadPoolExecutor with timeout."""
    def _loader() -> Any:
        from datasets import load_dataset  # type: ignore[import-untyped]
        if config_name:
            return load_dataset(name, config_name, split=split, trust_remote_code=True)
        return load_dataset(name, split=split, trust_remote_code=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_loader)
        return future.result(timeout=timeout)


# ---------------------------------------------------------------------------
# Main Router-Training Set Builder
# ---------------------------------------------------------------------------


def build_router_training_set(
    n_per_category: int = 120,
    seed: int = 2026,
    eval_tasks_path: Path = Path("data/tasks.jsonl"),
    cache_path: Path = Path("data/router_train_cache.jsonl"),
    offline: bool = True,
    use_fixture: bool = False,
) -> list[Task]:
    """Build ~300-500 prompts from train splits disjoint from eval tasks.

    Parameters
    ----------
    n_per_category:
        Number of prompts per category (GSM8K, ARC, MBPP).
    seed:
        RNG seed for reproducibility.
    eval_tasks_path:
        Path to evaluation dev/test tasks for collision verification.
    cache_path:
        Path to local disk cache (reused automatically if present).
    offline:
        If True, never connects to HuggingFace and uses synthetic/fixture generation.
    use_fixture:
        If True, returns the fixed 20-prompt fixture immediately.
    """
    if use_fixture:
        sys.stdout.write("[edge-data] Using bundled 20-prompt fixture (offline mode)\n")
        sys.stdout.flush()
        return build_edge_fixture_tasks()

    # 1. Check local cache first
    if cache_path.exists():
        cached_tasks: list[Task] = []
        with open(cache_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    cached_tasks.append(Task.model_validate_json(line))
        if len(cached_tasks) >= n_per_category * 3:
            sys.stdout.write(f"[edge-data] Loaded {len(cached_tasks)} tasks from local cache: {cache_path}\n")
            sys.stdout.flush()
            return cached_tasks

    # 2. Collect all forbidden hashes from existing eval tasks (dev + test)
    forbidden_hashes: set[str] = set()
    if eval_tasks_path.exists():
        with open(eval_tasks_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    prompt_text = item.get("prompt", "")
                    if prompt_text:
                        forbidden_hashes.add(_prompt_hash(prompt_text))
        sys.stdout.write(f"[edge-data] Loaded {len(forbidden_hashes)} evaluation task hashes for collision checking\n")
        sys.stdout.flush()

    tasks: list[Task] = []

    # 3. Try HuggingFace only if offline is False
    hf_success: bool = False
    if not offline:
        try:
            # GSM8K
            sys.stdout.write("[edge-data] Loading gsm8k train split from HuggingFace (60s timeout)...\n")
            sys.stdout.flush()
            ds_gsm = _load_hf_dataset_with_timeout("gsm8k", "main", "train", timeout=60.0)
            sys.stdout.write("[edge-data] Successfully loaded gsm8k train split\n")
            sys.stdout.flush()

            rng_gsm = random.Random(seed)
            indices = rng_gsm.sample(range(len(ds_gsm)), min(len(ds_gsm), n_per_category * 3))
            count = 0
            for i in indices:
                row = ds_gsm[i]
                p = row["question"]
                if _prompt_hash(p) not in forbidden_hashes:
                    forbidden_hashes.add(_prompt_hash(p))
                    ans = row["answer"].split("####")[-1].strip()
                    tasks.append(Task(id=f"edge-gsm-hf-{count + 1}", prompt=p, reference_answer=ans, category=TaskCategory.GSM8K, split="train", metadata={"source": "gsm8k_hf_train", "index": i}))
                    count += 1
                    if count >= n_per_category:
                        break

            # ARC-Challenge
            sys.stdout.write("[edge-data] Loading allenai/ai2_arc train split from HuggingFace (60s timeout)...\n")
            sys.stdout.flush()
            ds_arc = _load_hf_dataset_with_timeout("allenai/ai2_arc", "ARC-Challenge", "train", timeout=60.0)
            sys.stdout.write("[edge-data] Successfully loaded allenai/ai2_arc train split\n")
            sys.stdout.flush()

            rng_arc = random.Random(seed + 1)
            indices = rng_arc.sample(range(len(ds_arc)), min(len(ds_arc), n_per_category * 3))
            count = 0
            for i in indices:
                row = ds_arc[i]
                choices = row["choices"]
                opt_text = "\n".join(f"{lbl}) {txt}" for lbl, txt in zip(choices["label"], choices["text"]))
                p = f"{row['question']}\n\n{opt_text}\n\nAnswer with the letter only."
                if _prompt_hash(p) not in forbidden_hashes:
                    forbidden_hashes.add(_prompt_hash(p))
                    tasks.append(Task(id=f"edge-arc-hf-{count + 1}", prompt=p, reference_answer=row["answerKey"], category=TaskCategory.ARC_CHALLENGE, split="train", metadata={"source": "arc_hf_train", "index": i}))
                    count += 1
                    if count >= n_per_category:
                        break

            # MBPP
            sys.stdout.write("[edge-data] Loading mbpp train split from HuggingFace (60s timeout)...\n")
            sys.stdout.flush()
            ds_mbpp = _load_hf_dataset_with_timeout("mbpp", "sanitized", "train", timeout=60.0)
            sys.stdout.write("[edge-data] Successfully loaded mbpp train split\n")
            sys.stdout.flush()

            rng_mbpp = random.Random(seed + 2)
            indices = rng_mbpp.sample(range(len(ds_mbpp)), min(len(ds_mbpp), n_per_category * 3))
            count = 0
            for i in indices:
                row = ds_mbpp[i]
                tests = row.get("test_list", [])[:3]
                test_block = "\n".join(tests)
                p = f"{row['text']}\n\nTest cases:\n{test_block}\n\nWrite the Python function."
                if _prompt_hash(p) not in forbidden_hashes:
                    forbidden_hashes.add(_prompt_hash(p))
                    tasks.append(Task(id=f"edge-mbpp-hf-{count + 1}", prompt=p, reference_answer=row.get("code", ""), category=TaskCategory.MBPP, split="train", metadata={"source": "mbpp_hf_train", "index": i, "test_list": tests}))
                    count += 1
                    if count >= n_per_category:
                        break

            if len(tasks) >= n_per_category * 3:
                hf_success = True
                sys.stdout.write(f"[edge-data] Built {len(tasks)} tasks from HuggingFace train splits\n")
                sys.stdout.flush()
        except Exception as exc:
            sys.stdout.write(f"[edge-data] HuggingFace loading error or timeout: {exc}. Falling back to synthetic generators.\n")
            sys.stdout.flush()

    # 4. Fallback to synthetic generators
    if not hf_success:
        sys.stdout.write(f"[edge-data] Generating {n_per_category * 3} seeded synthetic router-training tasks (offline mode)...\n")
        sys.stdout.flush()
        tasks = []
        tasks.extend(_generate_synthetic_gsm8k(n_per_category, seed=seed + 10, forbidden_hashes=forbidden_hashes))
        tasks.extend(_generate_synthetic_arc(n_per_category, seed=seed + 20, forbidden_hashes=forbidden_hashes))
        tasks.extend(_generate_synthetic_mbpp(n_per_category, seed=seed + 30, forbidden_hashes=forbidden_hashes))

    # 5. Collision verification check
    eval_hashes: set[str] = set()
    if eval_tasks_path.exists():
        with open(eval_tasks_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    eval_hashes.add(_prompt_hash(item.get("prompt", "")))

    collisions = [t for t in tasks if _prompt_hash(t.prompt) in eval_hashes]
    if collisions:
        raise ValueError(f"CRITICAL: Found {len(collisions)} overlapping prompts between edge train set and evaluation benchmark!")

    sys.stdout.write(f"[edge-data] Hash check passed: 0 collisions with {len(eval_hashes)} evaluation tasks\n")
    sys.stdout.flush()

    # 6. Save to cache
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        for t in tasks:
            f.write(t.model_dump_json() + "\n")
    sys.stdout.write(f"[edge-data] Cached {len(tasks)} router-training tasks to {cache_path}\n")
    sys.stdout.flush()

    return tasks


# ---------------------------------------------------------------------------
# Labeling pipeline with local model execution and caching
# ---------------------------------------------------------------------------


def label_router_training_set(
    tasks: list[Task],
    local_client: LocalClient,
    cache_path: Path,
    force_refresh: bool = False,
) -> list[dict[str, Any]]:
    """Execute local model on each task, score answer, and record outcome label.

    Outcome Label:
      1 = Local model answer failed quality score (< 1.0) -> must escalate
      0 = Local model answer passed quality score (== 1.0) -> resolved locally

    Proxy Label (secondary baseline comparison):
      1 = Task is MBPP/Multi-Part or word count > 40
      0 = Otherwise
    """
    if cache_path.exists() and not force_refresh:
        cached_results: list[dict[str, Any]] = []
        with open(cache_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    cached_results.append(json.loads(line))
        if len(cached_results) == len(tasks):
            sys.stdout.write(f"[edge-label] Loaded {len(cached_results)} labeled tasks from cache: {cache_path}\n")
            sys.stdout.flush()
            return cached_results

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    labeled_tasks: list[dict[str, Any]] = []

    sys.stdout.write(f"[edge-label] Labeling {len(tasks)} tasks using model: {local_client.model_name}...\n")
    sys.stdout.flush()

    for idx, task in enumerate(tasks):
        responses = local_client.generate(task.prompt, temperature=0.0)
        local_answer = responses[0].text if responses else ""

        quality_score = score_task(task, local_answer)
        outcome_label = 1 if quality_score < 1.0 else 0

        is_complex_cat = task.category in (TaskCategory.MBPP, TaskCategory.MULTI_PART)
        is_long = len(task.prompt.split()) > 40
        proxy_label = 1 if (is_complex_cat or is_long) else 0

        record: dict[str, Any] = {
            "id": task.id,
            "prompt": task.prompt,
            "reference_answer": task.reference_answer,
            "category": task.category.value,
            "split": task.split,
            "local_answer": local_answer,
            "quality_score": quality_score,
            "outcome_label": outcome_label,
            "proxy_label": proxy_label,
            "metadata": task.metadata,
        }
        labeled_tasks.append(record)

    with open(cache_path, "w", encoding="utf-8") as f:
        for record in labeled_tasks:
            f.write(json.dumps(record) + "\n")

    sys.stdout.write(f"[edge-label] Saved {len(labeled_tasks)} labeled tasks to {cache_path}\n")
    sys.stdout.flush()
    return labeled_tasks


def get_label_distribution_summary(labeled_tasks: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute class balance and agreement statistics."""
    total = len(labeled_tasks)
    if total == 0:
        return {"total": 0}

    outcome_pos = sum(1 for t in labeled_tasks if t["outcome_label"] == 1)
    outcome_neg = total - outcome_pos
    proxy_pos = sum(1 for t in labeled_tasks if t["proxy_label"] == 1)
    proxy_neg = total - proxy_pos

    agreement = sum(
        1 for t in labeled_tasks if t["outcome_label"] == t["proxy_label"]
    ) / total

    return {
        "total": total,
        "outcome_positive_escalate": outcome_pos,
        "outcome_negative_local": outcome_neg,
        "outcome_positive_pct": round(outcome_pos / total * 100, 2),
        "proxy_positive_escalate": proxy_pos,
        "proxy_negative_local": proxy_neg,
        "proxy_positive_pct": round(proxy_pos / total * 100, 2),
        "outcome_vs_proxy_agreement_pct": round(agreement * 100, 2),
    }
