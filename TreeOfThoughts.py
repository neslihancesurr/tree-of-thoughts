import time
from dataclasses import dataclass
from typing import Tuple, List, Dict
import json

import ollama


@dataclass
class Thought:
    """Represents a thought/interpretation nodes in the tree"""
    content: str
    score: float
    depth: int
    parent: str = None

class TreeOfThoughts:
    def __init__(self, generator_model="qwen2.5:0.5b", evaluator_model="gemma3:latest"):
        """Initialize generator and evaluator models. Initialize the prompts"""

        self.generator_model = generator_model
        self.evaluator_model = evaluator_model
        self.generation_prompt = """You are responsible for describing a situation in a given sentence.  
        Describe what the event is and who/what the participants are clearly in three sentences.

        Sentence: {sentence} """
        self.refinement_prompt = """You previously interpreted the given sentence as follows:
        Sentence: {sentence}
        Previous interpretation: {previous_interpretation}

        What can you improve in your interpretation. Determine what might be missing in terms of possible scenarios.
        Refine your interpretation to include more possible scenarios.
        """
        self.evaluation_prompt = """You are a fair judge who decides whether a given interpretation of a sentence
        captures the linguistic ambiguity in the meaning. Evaluate the given sentence and list all possible meanings.
        Then, rate the following interpretation of the given sentence on a scale of 1-10 based on how well it captures the ambiguity.

        Sentence: {sentence}
        Interpretation: {interpretation}

        Respond with ONLY a JSON object in this exact format with no additional text:
        {{"score": <number between 1-10>, "justification": "<brief explanation>"}}"""

    def generate_thought(self, prompt: str, temperature: float = 0.8) -> str:
        """This function is to generate one single thought. There will be a wrapper function
        using this to generate as many thoughts as one likes. It should be done by the generator model"""
        response = ollama.generate(
            model = self.generator_model,
            prompt = prompt,
            options = {
                "temperature": temperature,
            },
            )
        return response['response'].strip()

    def evaluate_thought(self, sentence: str, interpretation: str) -> Tuple[float, str]:
        """This function is for evaluating a single thought"""
        prompt = self.evaluation_prompt.format(sentence=sentence, interpretation= interpretation)

        response = ollama.generate(
            model=self.evaluator_model,
            prompt=prompt,
            options={
                'temperature': 0.1
            }
        )

        response_text = response['response'].strip()

        # Remove markdown code blocks if present
        if '```json' in response_text:
            response_text = response_text.split('```json')[1].split('```')[0]
        elif '```' in response_text:
            response_text = response_text.split('```')[1].split('```')[0]

        result = json.loads(response_text)
        return result['score'], result['justification']

    def generate_initial_thoughts(self, sentence: str, k: int = 2) -> List[Thought]:
        """Generate k initial thoughts for depth 1"""

        thoughts = []
        prompt = self.generation_prompt.format(sentence=sentence)
        print(f"\n{'=' * 60}")
        print(f"Generating {k} initial interpretations for: '{sentence}'")
        print(f"{'=' * 60}")

        for i in range(k):

            temp = 0.7 + (i * 0.1)

            print(f"\n{'-' * 10}")
            print(f"Generated thought {i+1} for: '{sentence}' with temperature: {temp}")
            print(f"{'-' * 10}")

            interpretation = self.generate_thought(prompt=prompt, temperature=temp)

            if interpretation:
                print("The interpretation:", interpretation)
                score, justification = self.evaluate_thought(sentence=sentence, interpretation=interpretation)

                print(f"\n{'-' * 10}")
                print(f"Evaluation {i + 1} for: '{sentence}'")
                print(f"{'-' * 10}")
                print(f"Score: {score}/10 - {justification}")

                thoughts.append(Thought(
                    content=interpretation,
                    depth=1,
                    parent=None,
                    score=score )
                )

                time.sleep(1)

        return thoughts

    def refine_thoughts(self, sentence: str, parent_thoughts: List[Thought], k: int) -> List[Thought]:
        """For each thought, generate k refined thoughts."""

        refined_thoughts = []
        print(f"\n{'=' * 60}")
        print(f"Refining top interpretations (depth 2)")
        print(f"{'=' * 60}")

        for idx, parent in enumerate(parent_thoughts):
            print(f"Refining interpretation for thought {idx+1}")
            print(f"Parent content: {parent.content}")

            for i in range(k):
                prompt = self.refinement_prompt.format(sentence= sentence, previous_interpretation=parent.content)

                refinement = self.generate_thought(prompt=prompt, temperature=0.7 + (i * 0.15))

                if refinement:
                    print(f"  Refinement {i + 1}: {refinement}...")
                    score, justification = self.evaluate_thought(sentence, refinement)
                    print(f"  Score: {score}/10")

                    refined_thoughts.append(Thought(
                        content=refinement,
                        score=score,
                        depth=2,
                        parent=parent.content
                    ))

                time.sleep(0.5)

        return refined_thoughts

    def beam_search(self, thoughts: List[Thought], beam_width: int = 1) -> List[Thought]:
        """Select top beam_width thoughts based on scores"""
        sorted_thoughts = sorted(thoughts, key=lambda t: t.score, reverse=True)
        return sorted_thoughts[:beam_width]


    def run_tot(self, sentence: str, initial_k: int = 5, refinement_k: int = 2,
                beam_width: int = 3) -> Dict:

        print(f"\n{'#' * 60}")
        print(f"STARTING TREE OF THOUGHTS")
        print(f"Sentence: {sentence}")
        print(f"Generator: {self.generator_model}")
        print(f"Evaluator: {self.evaluator_model}")
        print(f"{'#' * 60}")

        # Depth 1: Generate initial interpretations
        depth1_thoughts = self.generate_initial_thoughts(sentence, k=initial_k)
        top_thoughts = self.beam_search(depth1_thoughts, beam_width=beam_width)

        print(f"\n{'=' * 60}")
        print(f"Selected top {beam_width} interpretations for refinement:")
        for i, t in enumerate(top_thoughts):
            print(f"{i + 1}. Score: {t.score}/10")
            print(f"   {t.content}...")

        # Depth 2: Refine top thoughts
        depth2_thoughts = self.refine_thoughts(sentence, top_thoughts, k=refinement_k)
        best_thought = self.beam_search(depth2_thoughts, beam_width=1)[0]

        print(f"\n{'#' * 60}")
        print(f"FINAL BEST INTERPRETATION")
        print(f"Score: {best_thought.score}/10")
        print(f"{'#' * 60}")
        print(best_thought.content)
        print(f"{'#' * 60}\n")

        return {
            'sentence': sentence,
            'depth1_thoughts': depth1_thoughts,
            'depth2_thoughts': depth2_thoughts,
            'best_interpretation': best_thought,
            'all_scores_d1': [t.score for t in depth1_thoughts],
            'all_scores_d2': [t.score for t in depth2_thoughts]
        }


