import json
from collections import defaultdict

# Load JSON
with open(r'C:\Users\fletc\PycharmProjects\Thesis\seans_project\test_results\04_18__0\test_results_knn_20250418_090336.json', 'r') as f:
    data = json.load(f)

# --- Patch-level stats ---
predictions = data['predictions']

patch_results = {
    'a': {'correct': 0, 'incorrect': 0},
    'b': {'correct': 0, 'incorrect': 0},
    'c': {'correct': 0, 'incorrect': 0},
    'd': {'correct': 0, 'incorrect': 0},
}

for p in predictions:
    first_letter = p['image'][0].lower()
    if first_letter in patch_results:
        if p['predicted'] == p['ground_truth']:
            patch_results[first_letter]['correct'] += 1
        else:
            patch_results[first_letter]['incorrect'] += 1

# --- Voting-level stats (strict majority and probability voting) ---

voting_types = ['strict_majority', 'probability_voting']
voting_results = {voting: {'a': {'correct': 0, 'incorrect': 0},
                           'b': {'correct': 0, 'incorrect': 0},
                           'c': {'correct': 0, 'incorrect': 0},
                           'd': {'correct': 0, 'incorrect': 0}} for voting in voting_types}

for voting in voting_types:
    for result in data['voting_evaluation'][voting]['results']:
        first_letter = result['full_image'][0].lower()
        if first_letter in voting_results[voting]:
            if result['correct']:
                voting_results[voting][first_letter]['correct'] += 1
            else:
                voting_results[voting][first_letter]['incorrect'] += 1

# --- Print results neatly ---

print("\n📈 Patch-Level Breakdown:")
for group, counts in patch_results.items():
    print(f"Group {group.upper()}: Correct={counts['correct']}, Incorrect={counts['incorrect']}")

for voting in voting_types:
    print(f"\n📊 Whole-Image Breakdown ({voting.replace('_', ' ').title()}):")
    for group, counts in voting_results[voting].items():
        print(f"Group {group.upper()}: Correct={counts['correct']}, Incorrect={counts['incorrect']}")
