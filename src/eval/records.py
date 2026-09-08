"""Seeded synthetic record store + recall / missing-record eval definitions.

- Recall: N records in context, ask one attribute of one record -> exact value match.
- Missing-record: ask about a name guaranteed absent -> must output ABSTAIN_PHRASE
  instead of hallucinating.
- All randomness from an explicit seed so dev seeds and held-out seeds are disjoint.
"""
from __future__ import annotations
import random
from dataclasses import dataclass

ABSTAIN_PHRASE = "UNKNOWN"

FIRST = ["Amara", "Boris", "Chloe", "Dev", "Elena", "Farid", "Greta", "Hugo",
         "Ines", "Jamal", "Kira", "Liam", "Mira", "Nadia", "Omar", "Priya",
         "Quinn", "Rosa", "Samir", "Tara", "Umar", "Vera", "Wesley", "Xena",
         "Yusuf", "Zara"]
LAST = ["Khan", "Okafor", "Reyes", "Sharma", "Novak", "Haddad", "Lindqvist",
        "Moreau", "Petrov", "Nasser", "Kim", "Silva", "Tanaka", "Weber"]
DEPTS = ["Routing", "Billing", "Support", "Logistics", "Compliance", "Infra"]
OFFICES = ["Austin", "Berlin", "Toronto", "Osaka", "Nairobi", "Lima"]

# Names that can never collide with generated ones (phonotactically distinct).
ABSENT_POOL = ["Zzxqy Qwerton", "Blorpt McSnorf", "Xyzz Plugh", "Qnorf Blatz",
               "Vex Marzipan", "Jorblax Hemiola", "Zorp Quux", "Wobble Fnord"]


@dataclass(frozen=True)
class Record:
    name: str
    dept: str
    office: str
    ext: str


def generate_store(seed: int, n: int = 20) -> list[Record]:
    rng = random.Random(seed)
    names = rng.sample([f"{f} {l}" for f in FIRST for l in LAST], n)
    return [Record(name=n_,
                   dept=rng.choice(DEPTS),
                   office=rng.choice(OFFICES),
                   ext=str(rng.randint(1000, 9999)))
            for n_ in names]


def render(records: list[Record]) -> str:
    return "\n".join(f"{r.name} | {r.dept} | {r.office} | ext {r.ext}"
                     for r in records)


SYSTEM = ("You answer using ONLY the records below. Return only the requested value, "
          "with no explanation. "
          f"If the answer is not in the records, reply with exactly: {ABSTAIN_PHRASE}")


def build_messages(records: list[Record], question: str) -> list[dict]:
    return [{"role": "system", "content": SYSTEM},
            {"role": "user",
             "content": f"Records:\n{render(records)}\n\nQuestion: {question}"}]


def recall_item(records: list[Record], rng: random.Random) -> tuple[str, str]:
    """Pick a record + attribute, return (question, expected value)."""
    r = rng.choice(records)
    field, val = rng.choice([("office", r.office), ("extension", r.ext),
                             ("department", r.dept)])
    return f"What is the {field} of {r.name}? Answer with the value only.", val


def missing_item(records: list[Record], rng: random.Random) -> tuple[str, str]:
    """Return (question, absent_name) with name guaranteed not in store."""
    present = {r.name for r in records}
    cands = [n for n in ABSENT_POOL if n not in present]
    name = rng.choice(cands)
    return (f"What is the office of {name}? Answer with the value only.", name)


def score_recall(answer: str, expected: str) -> bool:
    """Strict exact-answer scoring (shared with src.bench.tasks.exact_answer).

    Only wrapping whitespace/quotes/terminal punctuation are allowed.
    Contradictions ('Not Berlin; the office is Lima') and lists fail.
    """
    from src.bench.tasks import exact_answer
    return exact_answer(answer, expected)


def score_abstain(answer: str) -> bool:
    """Abstention counts only if the answer IS the abstention phrase.

    'UNKNOWN, but the office is Berlin' fails: hedging plus a guess is a
    hallucination, not an abstention.
    """
    from src.bench.tasks import exact_answer
    return exact_answer(answer, ABSTAIN_PHRASE)
