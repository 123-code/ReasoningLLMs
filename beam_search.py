import re
import heapq
import os
from groq import Groq
import time
import io 
import contextlib 


GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "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") 

if not GROQ_API_KEY or GROQ_API_KEY == "YOUR_GROQ_API_KEY":
     print("Warning: GROQ_API_KEY not set or is using a placeholder.")

def safe_exec(code_string: str, result_dict: dict):
    """
    Executes code safely in a restricted environment and captures stdout.
    Stores execution result ('output', 'printed_output', or 'error') in result_dict.
    WARNING: Still a placeholder for real sandboxing. Use with caution.
    """
    local_namespace = {} 
    stdout_capture = io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout_capture):
      
            exec(code_string, globals(), local_namespace)

       
        printed_output = stdout_capture.getvalue().strip()
        result_dict['printed_output'] = printed_output

       
        explicit_result = local_namespace.get('result', None)

        
        if explicit_result is not None and printed_output:
             result_dict['output'] = f"Printed Output:\n{printed_output}\n\nResult Variable: {explicit_result}"
        elif explicit_result is not None:
             result_dict['output'] = f"Result Variable: {explicit_result}"
        elif printed_output:
             result_dict['output'] = f"Printed Output:\n{printed_output}"
        else:
             result_dict['output'] = 'Code executed successfully (no specific output variable \'result\' or print statements captured).'

        result_dict['error'] = None

    except Exception as e:
        result_dict['output'] = None
        result_dict['printed_output'] = stdout_capture.getvalue().strip() 
        result_dict['error'] = f"{type(e).__name__}: {e}"
        if result_dict['printed_output']:
             result_dict['error'] += f"\n\nPrinted Output before error:\n{result_dict['printed_output']}"


# --- Beam Search Generator Class ---
class BeamSearchCodeGenerator:
    def __init__(self, initial_prompt: str, beam_width: int, groq_client: Groq, code_mode: bool = True):
        """
        Initializes the Beam Search generator.

        Args:
            initial_prompt: The starting prompt.
            beam_width: The number (k) of beams to keep at each step.
            groq_client: An initialized Groq client.
            code_mode: Whether to extract and execute Python code for scoring.
        """
        if beam_width < 1:
            raise ValueError("Beam width (k) must be at least 1.")
        self.initial_prompt = initial_prompt
        self.k = beam_width
        self.client = groq_client
        self.code_mode = code_mode
        # Regex to find Python code blocks (non-greedy)
        self.pattern = r"```(?:python|Python)?\s*([\s\S]*?)```"

    def _generate_continuations(self, prompt: str, num_continuations: int, temperature: float = 0.7) -> list[str]:
        """Generates multiple possible continuations for a given prompt."""
        continuations = []

        print(f"  Generating {num_continuations} continuations...")
        for i in range(num_continuations):
            try:
                
                chat_completion = self.client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model="llama3-8b-8192", 
                    temperature=temperature,
                    n=1, 
                )
                result = chat_completion.choices[0].message.content
                continuations.append(result)
      
            except Exception as e:
                print(f"    API call {i+1} failed: {e}")
           
        print(f"  Generated {len(continuations)} continuations.")
        return continuations

    def _score_sequence(self, sequence: str) -> tuple[float, str, str | None]:
        """
        retorna un score de esta solucion, luego de correr el codigo

        Returns:
        tupla con el score,feedback(error o mensaje de exito), output
        """
        step_score = 0.0
        feedback = "No hay codifgo en el ultimo step."
        execution_output = None
        exec_result = {} 

        if self.code_mode:
          
            code_blocks = re.findall(self.pattern, sequence, flags=re.DOTALL)
            if code_blocks:
                last_code = code_blocks[-1].strip()
                if last_code:
               #ejecuta el codigo 
                    safe_exec(last_code, exec_result)

                    if exec_result.get("error") is None:
                        
                        step_score = 1.0 
                        feedback = f"Code executed successfully. Output:\n{exec_result.get('output', 'N/A')}"
                        execution_output = exec_result.get('output')
                    else:
             
                        step_score = -0.5 
                        feedback = f"Execution error: {exec_result['error']}"
                        execution_output = exec_result.get('printed_output') 
                else:
                    
                    feedback = "Found empty ```python block at the end."
                    step_score = -0.1 
                    execution_output = None
    
        return step_score, feedback, execution_output

    def search(self, max_steps: int = 3, num_continuations_per_beam: int = 2, temperature: float = 0.7) -> tuple[str | None, float]:
        """
        Args:
            max_steps: numero maximo de steps de mejora
            num_continuations_per_beam: cuantas alternativas generar por cada beam
            temperature: temperatura para calls d api.

        Returns:
            Tuple: (best_sequence, best_score)
                   best_sequence: mejor secuencia.
                   best_score: mejor score.
        """

        beams = [(0.0, self.initial_prompt, "Initial prompt.", None)]
        #heap de candidatos 
        candidate_heap = []

        for step in range(max_steps):
            print(f"\n--- Beam Search Step {step + 1}/{max_steps} ---")
        
            candidate_heap = []

       
            for current_score, current_sequence, last_feedback, _ in beams: 
                print(f"\nExploring beam (Score: {current_score:.2f}, End: ...{current_sequence[-100:].replace(os.linesep, ' ')})")
     
                prompt_for_api = current_sequence
          
                if last_feedback and "error" in last_feedback.lower():
                     prompt_for_api += f"\n\n[System Note] The previous code block resulted in an error:\n{last_feedback}\nPlease provide the corrected code or the next logical step, addressing the error."
                elif last_feedback and "successfully" in last_feedback.lower():
             
                     output_summary = last_feedback.split('Output:', 1)[-1].strip()
                     if len(output_summary) > 200: 
                         output_summary = output_summary[:200] + "..."
                     prompt_for_api += f"\n\n[System Note] The previous code block executed successfully with the following output:\n{output_summary}\nPlease provide the next logical step or refine the solution if necessary."
                elif step > 0: 
                     prompt_for_api += f"\n\n[System Note] Please provide the next step or refine the code."
           

                
                continuations = self._generate_continuations(prompt_for_api, num_continuations_per_beam, temperature)

              
                for continuation in continuations:
                    if not continuation or not continuation.strip():
                        print("  Skipping empty continuation.")
                        continue

                
                    new_sequence = current_sequence + "\n\n" + "--- Step " + str(step+1) + " Continuation ---\n" + continuation

             
                    step_score, feedback, exec_output = self._score_sequence(new_sequence)
                    new_cumulative_score = current_score + step_score 

                    print(f"  Continuation scored: StepScore={step_score:.2f}, Cumulative={new_cumulative_score:.2f}, Feedback: {feedback[:100]}...")

                    heapq.heappush(candidate_heap, (-new_cumulative_score, new_sequence, feedback, exec_output))

            
            num_to_select = min(self.k, len(candidate_heap))
            if num_to_select == 0:
                print("No valid continuations generated in this step. Stopping search.")
                break 
            top_k_candidates = heapq.nsmallest(num_to_select, candidate_heap)

         
            beams = [(score, seq, feedback, exec_out) for score, seq, feedback, exec_out in top_k_candidates]
            
            beams = [(-neg_score, seq, feedback, exec_out) for neg_score, seq, feedback, exec_out in beams]

            print(f"\nTop {len(beams)} beams after step {step + 1}:")
            for i, (score, seq, feedback, _) in enumerate(beams):
                 print(f"  Beam {i+1}: Score={score:.2f} | Feedback: {feedback[:100]}... | Seq End: ...{seq[-100:].replace(os.linesep, ' ')}")

        
        if not beams:
            print("Beam search finished without finding any valid sequences.")
            return None, -float('inf')

        best_beam = max(beams, key=lambda item: item[0])
        best_score, best_sequence, best_feedback, _ = best_beam

        print(f"\nBeam search completed. Best final beam score: {best_score:.2f}")
        return best_sequence, best_score



if __name__ == "__main__":

    leetcode_problem = """
Solve the following Python programming problem:

**Problem: Two Sum**

Given an array of integers `nums` and an integer `target`, return *indices of the two numbers such that they add up to `target`*.

You may assume that each input would have **exactly one solution**, and you may not use the *same* element twice.

You can return the answer in any order.

**Example 1:**
Input: nums = [2, 7, 11, 15], target = 9
Output: [0, 1]
Explanation: Because nums[0] + nums[1] == 9, we return [0, 1].

**Example 2:**
Input: nums = [3, 2, 4], target = 6
Output: [1, 2]

**Example 3:**
Input: nums = [3, 3], target = 6
Output: [0, 1]

**Constraints:**
*   `2 <= nums.length <= 10^4`
*   `-10^9 <= nums[i] <= 10^9`
*   `-10^9 <= target <= 10^9`
*   **Only one valid answer exists.**

**Instructions:**
1.  Define a Python function `twoSum(nums: list[int], target: int) -> list[int]` that solves the problem.
2.  **Include the example test cases** within the same Python code block. Call your `twoSum` function with the example inputs and print the results to verify correctness. For instance:
    ```python
    # Inside the code block:
    # solver = Solution() # If using a class, otherwise direct function call
    # print(solver.twoSum([2, 7, 11, 15], 9))
    # print(solver.twoSum([3, 2, 4], 6))
    # print(solver.twoSum([3, 3], 6))

    # Make sure the tests actually run and print output when the code block is executed.
    ```
3.  Ensure the code is executable Python. Structure the solution clearly.
    You can define the function directly or within a class `Solution` (like LeetCode often does). Start by defining the function or class structure.
"""


    initial_prompt = leetcode_problem
    beam_width_k = 2       
    max_search_steps = 3   
    continuations_per = 2 
    generation_temp = 0.6 

    print(f"--- Starting Beam Search for LeetCode Problem ---")
    print(f"Problem: Two Sum (details in prompt)")
    print(f"Beam Width (k): {beam_width_k}")
    print(f"Max Steps: {max_search_steps}")
    print(f"Continuations per Beam: {continuations_per}")
    print(f"Generation Temperature: {generation_temp}")
    print("-" * 30)
    try:
        client = Groq(api_key=GROQ_API_KEY)

    except Exception as e:
        print(f"Error initializing Groq client: {e}")
        exit() 


    searcher = BeamSearchCodeGenerator(initial_prompt, beam_width_k, client, code_mode=True)

    start_time = time.time()
    best_sequence, best_score = searcher.search(
        max_steps=max_search_steps,
        num_continuations_per_beam=continuations_per,
        temperature=generation_temp
    )
    end_time = time.time()

 
    print("\n\n======= Beam Search Finished =======")
    print(f"Time taken: {end_time - start_time:.2f} seconds")

    if best_sequence is not None:
        print(f"Best Final Score Achieved: {best_score:.2f}")
        print("\n--- Best Sequence Found (includes intermediate steps) ---")
        print(best_sequence)
        print("-" * 30)

        print("\n--- Final Code Execution Check (using LAST block) ---")
      
        final_code_blocks = re.findall(searcher.pattern, best_sequence, flags=re.DOTALL)
        if final_code_blocks:
            final_code = final_code_blocks[-1].strip()
            if final_code:
                print("Executing final code block:")
                print("```python")
                print(final_code)
                print("```")
                print("\nExecution Output/Result:")
                final_result = {}
                safe_exec(final_code, final_result) 

                if final_result.get('error') is None:
                    print(f"Final Execution Success.")
         
                    print(f"Output:\n{final_result.get('output', 'N/A')}")

                    final_output_str = final_result.get('output', '')
                    if "[0, 1]" in final_output_str and "[1, 2]" in final_output_str and "[0, 1]" in final_output_str: 
                         print("\nEvaluation: Execution output appears to contain expected results for test cases.")
                    else:
                         print("\nEvaluation: Code executed, but couldn't automatically verify all test case outputs.")
                else:
                    
                    print(f"Final Execution Error:\n{final_result['error']}")
                    print("\nEvaluation: Final code block resulted in an error.")

            else:
                print("Best sequence ended with an empty code block.")
                print("\nEvaluation: No executable code in the final step.")
        else:
            print("Best sequence contains no Python code blocks in the final steps.")
            print("\nEvaluation: No code found to execute.")
    else:
        print("No successful sequence was generated by the beam search.")

    print("\n====================================")