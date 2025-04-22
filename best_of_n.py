import re
import time
import os
import random 
from groq import Groq


GROQ_API_KEY = ""
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY environment variable not set.")

def safe_exec(code_string: str, result_dict: dict):
    """
    Executes code safely in a restricted environment.
    Stores execution result ('output' or 'error') in result_dict.
    WARNING: This is a placeholder. Real sandboxing is complex.
             Libraries like 'restrictedpython' or running in a separate
             process/container are recommended for production.
    """
    try:
        exec_globals = {}
        exec(code_string, exec_globals)
        output = exec_globals.get('result', 'Code executed successfully (no specific output captured).')
        result_dict['output'] = output
        result_dict['error'] = None
    except Exception as e:
        result_dict['output'] = None
        result_dict['error'] = f"{type(e).__name__}: {e}"


class BestOfNCodeGenerator:
    def __init__(self, initial_prompt: str, num_candidates: int, groq_client: Groq, code_mode: bool = True):
        """
        Initializes the Best-of-N generator.

        Args:
            initial_prompt: The starting prompt.
            num_candidates: The number (N) of independent candidates to generate.
            groq_client: An initialized Groq client.
            code_mode: Whether to extract and execute Python code for scoring.
        """
        if num_candidates < 1:
            raise ValueError("Number of candidates (N) must be at least 1.")
        self.initial_prompt = initial_prompt
        self.n_candidates = num_candidates
        self.client = groq_client
        self.code_mode = code_mode
        # Regex to find Python code blocks
        self.pattern = r"```(?:python|Python)?\s*([\s\S]*?)```"

    def _generate_n_candidates(self, temperature: float = 0.7) -> list[str]:
        """Generates N independent candidate responses."""
        candidates = []
        print(f"Generating {self.n_candidates} candidates (temperature={temperature})...")
        for i in range(self.n_candidates):
            try:
              
                chat_completion = self.client.chat.completions.create(
                    messages=[{"role": "user", "content": self.initial_prompt}],
                    model="llama3-70b-8192",
                    temperature=temperature, 
                    n=1 
                )
                result = chat_completion.choices[0].message.content
                candidates.append(result)
                print(f"  Generated candidate {i+1}/{self.n_candidates}")
            except Exception as e:
                print(f"  API call failed for candidate {i+1}: {e}")
         
                candidates.append(None)

        return candidates

    def _score_sequence(self, sequence: str) -> tuple[float, str, str, str]:
   
        score = 0.0
        status = "no_code"
        code = None
        detail = "No code block found in the sequence."
        exec_result = {}

        if self.code_mode:
            code_blocks = re.findall(self.pattern, sequence, flags=re.MULTILINE)
            if code_blocks:
       
                code = code_blocks[-1].strip()
                if code: 
                    safe_exec(code, exec_result) 
                    if exec_result.get("error") is None:
                        score = 1.0 + 0.1 
                        status = "success"
                        detail = f"Execution success. Output: {exec_result.get('output', 'N/A')}"
                    else:
                        score = -0.5
                        status = "error"
                        detail = f"Execution error: {exec_result['error']}"
                else:
                    status = "empty_code"
                    detail = "Found empty code block."
                    score = -0.1 
       
        else:

             score = 0.0
             status = "not_applicable"
             detail = "Code execution scoring not applicable."


        return score, status, code, detail

    def generate_and_select(self, temperature: float = 0.7) -> tuple[str | None, float]:

        candidates = self._generate_n_candidates(temperature)

        best_score = -float('inf')
        best_sequence = None
        scored_candidates = []

        print("\n--- Scoring Candidates ---")
        for i, candidate in enumerate(candidates):
            if candidate is None:
                print(f"Candidate {i+1}: Generation failed.")
                scored_candidates.append({'id': i+1, 'sequence': None, 'score': -float('inf'), 'status': 'generation_failed', 'code': None, 'detail': 'API call failed'})
                continue

            score, status, code, detail = self._score_sequence(candidate)
            print(f"Candidate {i+1}: Score={score:.2f}, Status={status}, Detail={detail}")
            scored_candidates.append({'id': i+1, 'sequence': candidate, 'score': score, 'status': status, 'code': code, 'detail': detail})

            if score > best_score:
                best_score = score
                best_sequence = candidate

   
        if best_sequence is None:
             print("\nWarning: All candidate generations or scorings failed.")
             return None, -float('inf')
        else:
        
            pass

        return best_sequence, best_score



if __name__ == "__main__":
    initial_prompt = "Calculate the definite integral: $$ \int_{0}^{3 / 2} \\frac{x^{2} \cdot d x}{\sqrt{9-x^{2}}} $$ Use Python code with scipy.integrate.quad to find the numerical solution and print the result clearly, for example 'The result is: [value]'."
    num_candidates_N = 5  
    generation_temperature = 0.8 

    print(f"--- Starting Best-of-N Generation ---")
    print(f"Initial Prompt: {initial_prompt}")
    print(f"Number of Candidates (N): {num_candidates_N}")
    print(f"Generation Temperature: {generation_temperature}")
    print("-" * 30)

    client = Groq(api_key=GROQ_API_KEY)
    generator = BestOfNCodeGenerator(initial_prompt, num_candidates_N, client, code_mode=True)

    start_time = time.time()
    best_sequence, best_score = generator.generate_and_select(temperature=generation_temperature)
    end_time = time.time()

    print("\n--- Best-of-N Finished ---")
    print(f"Time taken: {end_time - start_time:.2f} seconds")

    if best_sequence is not None:
        print(f"Best Score Achieved: {best_score:.2f}")
        print("\nBest Sequence Found:\n")
        print(best_sequence)


        print("\n--- Final Code Execution Check ---")
        final_code_blocks = re.findall(generator.pattern, best_sequence, flags=re.MULTILINE)
        if final_code_blocks:
            final_code = final_code_blocks[-1].strip()
            if final_code:
                print("Executing final code block:")
                print("```python")
                print(final_code)
                print("```")
                final_result = {}
                safe_exec(final_code, final_result)
                if final_result.get('error') is None:
                    print(f"Final Execution Success. Output: {final_result.get('output', 'N/A')}")
                else:
                    print(f"Final Execution Error: {final_result['error']}")
            else:
                print("Best sequence ended with an empty code block.")
        else:
            print("Best sequence contains no code blocks.")
    else:
        print("No successful sequence was generated.")
