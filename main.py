from TreeOfThoughts import TreeOfThoughts, Thought
from typing import Dict


def main():

    test_sentences = [
        "The crane was standing near the water.",
        "I saw the man with the telescope.",
        "Everyone loves someone.",
        "The bat flew out of the cave.",
        "Young dogs and cats need special vitamins."
    ]

    # Initialize ToT
    tot = TreeOfThoughts(
        generator_model="qwen2.5:0.5b",
        evaluator_model="gemma3:latest"
    )

    # Run on first sentence as example
    sentence = test_sentences[1]
    results = tot.run_tot(
        sentence=sentence,
        initial_k=5,  # Generate 5 initial interpretations
        refinement_k=2,  # Generate 2 refinements for each
        beam_width=3  # Keep top 3 for refinement
    )

    analyze_results(results)


# To run on all sentences:
# all_results = []
# for sentence in test_sentences:
#     results = tot.run_tot(sentence, initial_k=5, refinement_k=2, beam_width=3)
#     all_results.append(results)
#     time.sleep(2)

def analyze_results(results: Dict):
    """Analyze and print statistics about the ToT results"""
    print(f"\n{'=' * 60}")
    print("ANALYSIS")
    print(f"{'=' * 60}")

    d1_scores = results['all_scores_d1']
    d2_scores = results['all_scores_d2']

    print(f"\nDepth 1 Statistics:")
    print(f"  Mean score: {sum(d1_scores) / len(d1_scores):.2f}")
    print(f"  Max score: {max(d1_scores):.2f}")
    print(f"  Min score: {min(d1_scores):.2f}")
    print(f"  Score range: {max(d1_scores) - min(d1_scores):.2f}")

    print(f"\nDepth 2 Statistics:")
    print(f"  Mean score: {sum(d2_scores) / len(d2_scores):.2f}")
    print(f"  Max score: {max(d2_scores):.2f}")
    print(f"  Min score: {min(d2_scores):.2f}")
    print(f"  Improvement: {max(d2_scores) - max(d1_scores):.2f}")

    print(f"\nInterpretation Diversity:")
    print(f"  Generated {len(d1_scores)} initial interpretations")
    print(f"  Explored {len(d2_scores)} refined interpretations")

if __name__ == "__main__":
    main()