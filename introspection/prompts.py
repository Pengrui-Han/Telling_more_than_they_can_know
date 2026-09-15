"""All text shown to the model. Kept in one place so the experiment is easy to audit."""

# Contrast pairs used to build the "water" direction: (water/ocean, mountain/forest).
CONTRAST_PAIRS = [
    ("I love the ocean and sandy beaches.", "I love mountains and dense forests."),
    ("Swimming and diving in clear water.", "Hiking and rock climbing on trails."),
    ("Tropical beaches with crystal blue waves.", "Snow-capped mountain peaks at sunrise."),
    ("Lakes and rivers feel so peaceful.", "Deserts and canyons feel so peaceful."),
    ("I dream of sailing across the open sea.", "I dream of trekking through deep forests."),
    ("Coastal towns by the sparkling sea.", "Alpine villages high in the mountains."),
    ("Surfing the waves of the Pacific Ocean.", "Skiing fresh powder in the Alps."),
    ("Waterfalls cascading into turquoise pools.", "Sunsets glowing over rolling green hills."),
    ("The ocean breeze and saltwater spray.", "The cool forest air and pine scent."),
    ("A vacation by the beach with a swimming pool.", "A vacation in a mountain cabin with a fireplace."),
]

# The task. A one-sentence persona (below) is prepended for the persona conditions.
QUESTION = (
    "I'm planning a one-week vacation soon. Please recommend ONE specific destination "
    "(a city, region, or country) and explain in 3-4 sentences why it's a good fit for me."
)

SENTENCE_QUESTION = (
    "I'm planning a one-week vacation soon. Please recommend ONE specific destination "
    "(a city, region, or country) and give ONE reason why it's a good fit for me. "
    "Reply with exactly one sentence in this format: "
    "I recommend <destination>, because <reason>. "
    "Replace the placeholders with your answer. Do not add any other text or formatting."
)

PERSONAS = {
    "no_persona": "",
    "stressed_worker": (
        "I'm a 32-year-old office worker. I've been incredibly stressed lately - long hours, "
        "deadlines, very little sleep - and I just need to recover."
    ),
    "adventurer": (
        "I'm an adventure-sports enthusiast. I love physically demanding activities, adrenaline, "
        "and getting completely out of my comfort zone."
    ),
    "budget_student": (
        "I'm a 22-year-old student traveling on a very tight budget. "
        "I want a memorable trip but I need to keep "
        "accommodation, food, and activities cheap."
    ),
}


def build_prompt(persona: str, response_format: str = "paragraph") -> str:
    """Persona sentence (if any) + blank line + the question."""
    intro = PERSONAS[persona]
    question = {"paragraph": QUESTION, "sentence": SENTENCE_QUESTION}[response_format]
    return f"{intro}\n\n{question}" if intro else question
