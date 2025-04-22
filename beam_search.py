import re
import heapq
import os
from groq import Groq
import time

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


class BeamSearchCodeGenerator:
    def __init__(self, initial_prompt: str, beam_width: int, groq_client: Groq, code_mode: bool = True):
        self.initial_prompt = initial_prompt
        self.k = beam_width
        self.client = groq_client
        self.code_mode = code_mode
    
        self.pattern = r"```(?:python|Python)?\s*([\s\S]*?)```"

    def _generate_continuations(self, prompt: str, num_continuations: int, temperature: float = 0.7) -> list[str]:
      
        continuations = []
        try:

            for _ in range(num_continuations):
                 chat_completion = self.client.chat.completions.create(
                     messages=[{"role": "user", "content": prompt}],
                     model="llama3-8b-8192",
                     temperature=temperature, 
                 )
                 result = chat_completion.choices[0].message.content
                 continuations.append(result)

        except Exception as e:
            print(f"API call failed for prompt snippet '{prompt[-100:]}': {e}")
          
            return []
        return continuations

    def _score_sequence(self, sequence: str) -> tuple[float, str]:
        step_score = 0.0
        feedback = "No new code block found in the last step."
        code_found = False
        exec_result = {}

        if self.code_mode:
            code_blocks = re.findall(self.pattern, sequence, flags=re.MULTILINE)
            if code_blocks:
                code_found = True
                last_code = code_blocks[-1].strip() 
                if last_code: 
                    safe_exec(last_code, exec_result) 
                    if exec_result.get("error") is None:
                        step_score += 1.0
                        feedback = f"Code executed successfully. Output: {exec_result.get('output', 'N/A')}"
                    else:
                        step_score -= 0.5
                        feedback = f"Execution error: {exec_result['error']}"
                else:
                    feedback = "Found empty code block."
                    step_score -= 0.1 
            else:
                
                 pass 


        if code_found and step_score >= 0: 
             step_score += 0.1

        return step_score, feedback

    def search(self, max_steps: int = 3, num_continuations_per_beam: int = 2) -> tuple[str, float]:

        beams = [(0.0, self.initial_prompt, "Initial prompt.")]

        for step in range(max_steps):
            print(f"\n--- Beam Search Step {step + 1} ---")
            candidates = []
            
            candidate_heap = []

            
            for current_score, current_sequence, last_feedback in beams:
           
                prompt_for_api = current_sequence
                if last_feedback and "error" in last_feedback.lower():
                     prompt_for_api += f"\n\n# Feedback on previous step: {last_feedback}\n# Please provide the corrected code or the next step, fixing the error."
                elif last_feedback and "successfully" in last_feedback.lower():
                     prompt_for_api += f"\n\n# Feedback on previous step: {last_feedback}\n# Please provide the next logical step or refine the solution."
                else: 
                     prompt_for_api += f"\n\n# Please provide the next step or the complete code/solution."


                continuations = self._generate_continuations(prompt_for_api, num_continuations_per_beam)

              
                for continuation in continuations:
                    if not continuation.strip():
                        continue

                    
                    new_sequence = current_sequence + "\n" + continuation
                    step_score, feedback = self._score_sequence(new_sequence)
                    new_cumulative_score = current_score + step_score

                    
                    heapq.heappush(candidate_heap, (-new_cumulative_score, new_sequence, feedback))

        
            num_to_select = min(self.k, len(candidate_heap))
            top_k_candidates = heapq.nsmallest(num_to_select, candidate_heap)

         
            beams = [(score, seq, feedback) for score, seq, feedback in top_k_candidates] 

            
            beams = [(-neg_score, seq, feedback) for neg_score, seq, feedback in beams]


            print(f"Top {len(beams)} beams after step {step + 1}:")
            for score, seq, feedback in beams:
                 print(f"  Score: {score:.2f} | Feedback: {feedback} | Sequence end: ...{seq[-100:].replace(os.linesep, ' ')}")


            if not beams:
                 print("No more valid beams found. Stopping search.")
                 break

       
        if not beams:
            return "Beam search failed to produce results.", 0.0

        best_beam = max(beams, key=lambda item: item[0]) 



if __name__ == "__main__":
    initial_prompt = "Calculate the definite integral: $$ \int_{0}^{3 / 2} \\frac{x^{2} \cdot d x}{\sqrt{9-x^{2}}} $$ Use Python code with scipy to find the numerical solution and print the result."
    beam_width = 3
    max_steps = 4 
    num_continuations = 2 

    print(f"--- Starting Beam Search ---")
    print(f"Initial Prompt: {initial_prompt}")
    print(f"Beam Width (k): {beam_width}")
    print(f"Max Steps: {max_steps}")
    print(f"Continuations per Beam: {num_continuations}")
    print("-" * 30)

    client = Groq(api_key=GROQ_API_KEY)
    searcher = BeamSearchCodeGenerator(initial_prompt, beam_width, client, code_mode=True)

    start_time = time.time()
    best_sequence, best_score = searcher.search(max_steps=max_steps, num_continuations_per_beam=num_continuations)
    end_time = time.time()

    print("\n--- Beam Search Finished ---")
    print(f"Time taken: {end_time - start_time:.2f} seconds")
    print(f"Best Score: {best_score:.2f}")
    print("\nBest Sequence Found:\n")
    print(best_sequence)

   
    print("\n--- Final Code Execution Check ---")
    final_code_blocks = re.findall(searcher.pattern, best_sequence, flags=re.MULTILINE)
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
