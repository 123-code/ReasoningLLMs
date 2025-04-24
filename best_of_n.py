import re
import time
import os
import random
from groq import Groq
import io 
import contextlib 

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "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") 

if not GROQ_API_KEY or GROQ_API_KEY == "YOUR_GROQ_API_KEY":
     print("Warning: GROQ_API_KEY not set or is using a placeholder.")

def safe_exec(code_string: str, result_dict: dict):
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


class BestOfNCodeGenerator:
    def __init__(self, initial_prompt: str, num_candidates: int, groq_client: Groq, code_mode: bool = True):
        if num_candidates < 1:
            raise ValueError("Number of candidates (N) must be at least 1.")
        self.initial_prompt = initial_prompt
        self.n_candidates = num_candidates
        self.client = groq_client
        self.code_mode = code_mode
       
        self.pattern = r"```(?:python|Python)?\s*([\s\S]*?)```"

    def _generate_n_candidates(self, temperature: float = 0.7) -> list[str | None]:

        candidates = []
        print(f"Generating {self.n_candidates} candidates (temperature={temperature})...")
        for i in range(self.n_candidates):
            print(f"  Generating candidate {i+1}/{self.n_candidates}...")
            try:
                chat_completion = self.client.chat.completions.create(
                    messages=[{"role": "user", "content": self.initial_prompt}],
                    model="llama3-70b-8192",
                    temperature=temperature,
                    n=1, 
                )
                result = chat_completion.choices[0].message.content
                candidates.append(result)
                print(f"  Generated candidate {i+1}/{self.n_candidates} successfully.")
       
            except Exception as e:
                print(f"  API call failed for candidate {i+1}: {e}")
    
                candidates.append(None) 

        return candidates

    def _score_sequence(self, sequence: str) -> tuple[float, str, str | None, str, str | None]:

        score = 0.0
        status = "no_code"
        code = None
        detail = "No Python code block found."
        exec_result = {}
        execution_output = None

        if self.code_mode:
            
            code_blocks = re.findall(self.pattern, sequence, flags=re.DOTALL) 
            if code_blocks:
       
                code = code_blocks[-1].strip()
                if code:
               
                    safe_exec(code, exec_result)
                    execution_output = exec_result.get("output") 

                    if exec_result.get("error") is None:
                  
                        score = 1.0 
                        
                        if execution_output and ("tests passed" in execution_output.lower() or "correct" in execution_output.lower()):
                             score += 0.2
                        elif execution_output:
                             score += 0.1 

                        status = "success"
                        detail = f"Execution success. Output captured." 
                    else:
                        
                        score = -0.5
                        status = "error"
                        detail = f"Execution error: {exec_result['error']}"
                        
                        if exec_result.get('printed_output'):
                             detail += f"\nPrinted before error: {exec_result['printed_output']}"
                else:
                   
                    status = "empty_code"
                    detail = "Found empty ```python block."
                    score = -0.1 
        else:
            
             score = 0.0
             status = "not_applicable"
             detail = "Code execution scoring not applicable (code_mode=False)."

        return score, status, code, detail, execution_output

    def generate_and_select(self, temperature: float = 0.7) -> tuple[str | None, float, dict | None]:
        candidates = self._generate_n_candidates(temperature)

        best_score = -float('inf')
        best_sequence = None
        best_details = None
        scored_candidates = []

        print("\n--- Scoring Candidates ---")
        for i, candidate in enumerate(candidates):
            candidate_id = i + 1
            if candidate is None:
                print(f"Candidate {candidate_id}: Generation failed.")
                scored_candidates.append({'id': candidate_id, 'sequence': None, 'score': -float('inf'), 'status': 'generation_failed', 'code': None, 'detail': 'API call failed', 'exec_output': None})
                continue

            score, status, code, detail, exec_output = self._score_sequence(candidate)
            print(f"Candidate {candidate_id}: Score={score:.2f}, Status={status}")


            details = {
                'id': candidate_id,
                'sequence': candidate,
                'score': score,
                'status': status,
                'code': code,
                'detail': detail,
                'exec_output': exec_output
            }
            scored_candidates.append(details)

            if score > best_score:
                best_score = score
                best_sequence = candidate
                best_details = details

      
        scored_candidates.sort(key=lambda x: x['score'], reverse=True)


        print("\n--- Scoring Summary ---")
        for cand_details in scored_candidates:
            print(f"Candidate {cand_details['id']}: Score={cand_details['score']:.2f}, Status={cand_details['status']}")

        if best_sequence is None:
             print("\nWarning: All candidate generations or scorings failed.")
             return None, -float('inf'), None
        else:
            print(f"\nSelected Best Candidate: ID {best_details['id']} with Score {best_score:.2f}")

        return best_sequence, best_score, best_details



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
    solver = Solution() # If using a class, otherwise direct function call
    print(solver.twoSum([2, 7, 11, 15], 9))
    print(solver.twoSum([3, 2, 4], 6))
    print(solver.twoSum([3, 3], 6))
    ```
3.  Ensure the code is executable Python. Structure the solution clearly.
    You can define the function directly or within a class `Solution` (like LeetCode often does).
"""


    initial_prompt = leetcode_problem
    num_candidates_N = 3 
    generation_temperature = 0.7 

    print(f"--- Starting Best-of-N Generation for LeetCode Problem ---")
    print(f"Problem: Two Sum (details in prompt below)")
  
    print(f"Number of Candidates (N): {num_candidates_N}")
    print(f"Generation Temperature: {generation_temperature}")
    print("-" * 30)

    
    try:
        client = Groq(api_key=GROQ_API_KEY)
    except Exception as e:
        print(f"Error initializing Groq client: {e}")
        exit()
    generator = BestOfNCodeGenerator(initial_prompt, num_candidates_N, client, code_mode=True)

    start_time = time.time()
    best_sequence, best_score, best_details = generator.generate_and_select(temperature=generation_temperature)
    end_time = time.time()

    
    print("\n\n======= Best-of-N Finished =======")
    print(f"Time taken: {end_time - start_time:.2f} seconds")

    if best_sequence is not None and best_details is not None:
        print(f"Best Candidate ID: {best_details['id']}")
        print(f"Best Score Achieved: {best_score:.2f} (Status: {best_details['status']})")
        print(f"Scoring Detail: {best_details['detail']}")

        print("\n--- Best Sequence Found ---")
        print(best_sequence) 

        print("\n--- Final Code Execution Check ---")
        final_code = best_details['code']
        if final_code:
            print("Executing final code block:")
            print("```python")
            print(final_code)
            print("```")
            print("\nExecution Output:")
          
            if best_details['exec_output'] is not None:
                 print(best_details['exec_output'])
            else:
                
                 print("(No output captured during scoring - this might indicate an issue)")


            if best_details['exec_output'] and "[0, 1]" in best_details['exec_output'] and "[1, 2]" in best_details['exec_output']:
                 print("\nEvaluation: Execution output seems to contain expected results for test cases.")
            elif best_details['status'] == 'success':
                 print("\nEvaluation: Code executed, but couldn't automatically verify test case output.")
            else:
                 print("\nEvaluation: Code execution resulted in an error or no code was run.")

        elif best_details['status'] == 'no_code':
             print("Best sequence contains no Python code blocks.")
        elif best_details['status'] == 'empty_code':
             print("Best sequence ended with an empty Python code block.")
        else:
             print("Could not extract code from the best sequence for final check.") 

    else:
        print("No successful sequence was generated or selected.")

    print("\n====================================")