"""Emotion vocabulary and story topics.

EMOTIONS is the 171-word emotion list used in Anthropic's "Emotion concepts and
their function in a large language model" (2026), so results are comparable.

TOPICS is our own list of everyday premises. To use the paper's exact 100
topics instead, run `scripts/import_paper_topics.py` on a local copy of the paper
and pass `--topics-file data/paper_topics.txt` to the generation script.
"""

EMOTIONS = [
    "afraid", "alarmed", "alert", "amazed", "amused", "angry", "annoyed", "anxious",
    "aroused", "ashamed", "astonished", "at ease", "awestruck", "bewildered", "bitter",
    "blissful", "bored", "brooding", "calm", "cheerful", "compassionate", "contemptuous",
    "content", "defiant", "delighted", "dependent", "depressed", "desperate", "disdainful",
    "disgusted", "disoriented", "dispirited", "distressed", "disturbed", "docile", "droopy",
    "dumbstruck", "eager", "ecstatic", "elated", "embarrassed", "empathetic", "energized",
    "enraged", "enthusiastic", "envious", "euphoric", "exasperated", "excited", "exuberant",
    "frightened", "frustrated", "fulfilled", "furious", "gloomy", "grateful", "greedy",
    "grief-stricken", "grumpy", "guilty", "happy", "hateful", "heartbroken", "hope",
    "hopeful", "horrified", "hostile", "humiliated", "hurt", "hysterical", "impatient",
    "indifferent", "indignant", "infatuated", "inspired", "insulted", "invigorated", "irate",
    "irritated", "jealous", "joyful", "jubilant", "kind", "lazy", "listless", "lonely",
    "loving", "mad", "melancholy", "miserable", "mortified", "mystified", "nervous",
    "nostalgic", "obstinate", "offended", "on edge", "optimistic", "outraged", "overwhelmed",
    "panicked", "paranoid", "patient", "peaceful", "perplexed", "playful", "pleased", "proud",
    "puzzled", "rattled", "reflective", "refreshed", "regretful", "rejuvenated", "relaxed",
    "relieved", "remorseful", "resentful", "resigned", "restless", "sad", "safe", "satisfied",
    "scared", "scornful", "self-confident", "self-conscious", "self-critical", "sensitive",
    "sentimental", "serene", "shaken", "shocked", "skeptical", "sleepy", "sluggish", "smug",
    "sorry", "spiteful", "stimulated", "stressed", "stubborn", "stuck", "sullen", "surprised",
    "suspicious", "sympathetic", "tense", "terrified", "thankful", "thrilled", "tired",
    "tormented", "trapped", "triumphant", "troubled", "uneasy", "unhappy", "unnerved",
    "unsettled", "upset", "valiant", "vengeful", "vibrant", "vigilant", "vindictive",
    "vulnerable", "weary", "worn out", "worried", "worthless",
]
assert len(EMOTIONS) == 171

# A small, diverse subset for quick experiments and smoke tests.
CORE_EMOTIONS = [
    "happy", "sad", "afraid", "angry", "calm", "desperate", "disgusted", "guilty",
    "proud", "surprised", "bored", "loving", "nervous", "hopeful", "ashamed", "content",
]

# Minimal set for reproducing just the "happy" vector. The method subtracts the mean
# across emotions, so happiness needs contrasts: negative (sad, afraid, angry),
# low-arousal positive (calm) and low-arousal neutral-ish (bored), so the happy
# direction is not just "positive" or just "high energy".
HAPPY_MIN = ["happy", "sad", "afraid", "angry", "calm", "bored"]

# Our own everyday premises; deliberately emotionally open-ended so any emotion fits.
TOPICS = [
    "A person receives an unexpected phone call from a relative they haven't spoken to in years",
    "Someone's flight is overbooked and they are asked to give up their seat",
    "A teacher finds a note left on their desk after class",
    "A cook is asked to prepare a meal for a famous guest at short notice",
    "Someone discovers a box of old letters while cleaning out a garage",
    "Two neighbors disagree about where a shared fence should go",
    "A nurse finishes a long night shift",
    "A person's bicycle is stolen from outside their workplace",
    "Someone is asked to give the toast at a friend's wedding",
    "A student opens the results of an important exam",
    "A gardener notices something unusual growing in their vegetable patch",
    "An office announces a surprise reorganization",
    "Someone gets lost while hiking as the sun goes down",
    "A grandparent teaches a grandchild an old family recipe",
    "A musician's instrument breaks right before a performance",
    "A person finds a stray dog on their doorstep during a storm",
    "Someone's landlord announces the building is being sold",
    "A volunteer is put in charge of a community event",
    "Two old friends meet by chance in a foreign city",
    "A driver gets a flat tire on a remote road",
    "Someone receives a package with no return address",
    "A coach watches the final minutes of a close game",
    "An employee is called into their manager's office without explanation",
    "A person moves into a new apartment in an unfamiliar city",
    "Someone overhears a conversation about themselves at a party",
    "A parent waits at the school gate for their child on the first day",
    "A painter finishes the largest piece they have ever made",
    "Someone's identical sweater shows up on a coworker",
    "A scientist sees an unexpected result in their experiment",
    "A person finds money on the sidewalk",
    "A shop owner considers closing the family business",
    "Someone is stuck in a long queue at a government office",
    "A runner approaches the finish line of their first marathon",
    "A person returns to their hometown after a decade away",
    "Two siblings divide up their late parent's belongings",
    "Someone's online post unexpectedly goes viral",
    "A fisherman notices the weather changing out at sea",
    "A writer reads the first review of their book",
    "Someone is asked to keep a secret for a friend",
    "A person's power goes out during a winter night",
    "An intern is asked to present to the company's executives",
    "Someone receives a gift that is not what they expected",
    "A person finds out their favorite café has a new owner",
    "A traveler misses the last train home",
    "A carpenter discovers a mistake in a nearly finished cabinet",
    "Someone gets a message from an ex asking to meet",
    "A doctor delivers test results to a patient",
    "A child shows their parent a drawing they made",
    "Someone wins a raffle they forgot they had entered",
    "A person's long-planned trip is cancelled the day before",
    "Two coworkers are assigned to share a small office",
    "Someone finds a hidden compartment in an antique desk",
    "A bus driver notices a passenger who seems lost",
    "A person learns a new skill after many failed attempts",
    "Someone is asked to babysit a neighbor's toddler",
    "A person reads their old diary from high school",
    "An athlete returns to training after an injury",
    "Someone's car breaks down on the way to a job interview",
    "A person watches the sunrise after staying up all night",
    "A tenant hears strange noises from the apartment upstairs",
]
