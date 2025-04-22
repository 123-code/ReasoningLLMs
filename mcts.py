import subprocess
import logging
from groq import Groq
import numpy as np
import random
import math
import re
import os
import time


GROQ_API_KEY = "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"
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY environment variable not set.")


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def safe_exec(code_string: str, result_dict: dict):
    
  
    local_scope = {}
    try:
        
        exec(code_string, {"__builtins__": {}}, local_scope) 
    
        output = local_scope.get('result', 'Code executed successfully (no specific output variable \'result\' captured).')
        result_dict['output'] = output
        result_dict['error'] = None
    except Exception as e:
        result_dict['output'] = None
        result_dict['error'] = f"{type(e).__name__}: {e}"
        logger.warning(f"Execution failed: {e}", exc_info=False) 
    except SystemExit:
       
        result_dict['output'] = None
        result_dict['error'] = "SystemExit called, execution halted."
        logger.warning("Attempted SystemExit caught during safe_exec.")



def ask_llm(prompt: str, temperature: float = 0.7, max_tokens: int = 500) -> str:
    """Sends a prompt to the Groq LLM and returns the response."""
    client = Groq(api_key=GROQ_API_KEY)
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "user", "content": prompt}
            ],
            model="llama3-70b-8192",
            temperature=temperature,
            max_tokens=max_tokens,
            n=1, 
            stop=None 
        )
        response = chat_completion.choices[0].message.content
        logger.debug(f"LLM Raw Response: {response}")
        return response
    except Exception as e:
        logger.error(f"LLM API call failed: {e}")
        return "" 
def extract_code(text: str) -> str | None:
    """Extracts the last Python code block from the text."""
    pattern = r"```(?:python|Python)?\s*([\s\S]*?)```"
    code_blocks = re.findall(pattern, text, flags=re.MULTILINE)
    if code_blocks:
        return code_blocks[-1].strip()

    elif "def solution(" in text or "import " in text:
      
         potential_code = text.strip()
       
         if '\n' in potential_code and ('def ' in potential_code or 'return ' in potential_code or 'import ' in potential_code):
              logger.warning("No ```python ``` block found, attempting to treat entire response as code.")
              return potential_code
    return None


class Node:
    def __init__(self, parent=None, state="", prior=0.0):
        self.parent = parent
        self.state = state  
        self.children = []
        self.visits = 0
        self.value = 0.0  
        self.prior = prior
        self.is_terminal = False 
        self.code_executes = None 

    def is_fully_expanded(self, max_children):
        return len(self.children) >= max_children

    def get_best_child_ucb1(self, exploration_rate=1.41):
        """Selects the best child using the UCB1 formula."""
        if not self.children:
            return None

        best_child = None
        best_score = -float('inf')

        for child in self.children:
            if child.visits == 0:
          
                score = float('inf')
            else:
                exploit_term = child.value / child.visits
                explore_term = exploration_rate * math.sqrt(math.log(self.visits) / child.visits)
                score = exploit_term + explore_term
                logger.debug(f"  Child score: {score:.3f} (Value: {child.value:.2f}, Visits: {child.visits})")


            if score > best_score:
                best_score = score
                best_child = child

        if best_child is None: 
             return random.choice(self.children) if self.children else None

        logger.debug(f"Selected best child with score {best_score:.3f} (Visits: {best_child.visits}, Value: {best_child.value:.2f})")
        return best_child



class MCTS:
    def __init__(self, initial_prompt: str, iterations: int = 20, max_children: int = 3, exploration_rate: float = 1.41):
        self.initial_prompt = initial_prompt
        self.iterations = iterations
        self.max_children = max_children
        self.exploration_rate = exploration_rate
        self.root = Node(state=initial_prompt) 
        self.llm_calls = 0
        self.code_executions = 0
        self.best_code_found = None
        self.best_exec_result = None

    def _select(self, node: Node) -> Node:
        """Phase 1: Selection - Traverse the tree using UCB1 until a leaf or non-fully expanded node."""
        logger.info("--- Selection Phase ---")
        current_node = node
        while not current_node.is_terminal:
            if not current_node.is_fully_expanded(self.max_children):
                logger.info(f"Node not fully expanded (children: {len(current_node.children)}/{self.max_children}). Expanding.")
                return self._expand(current_node) 
            else:
                logger.info("Node fully expanded, selecting best child.")
                best_child = current_node.get_best_child_ucb1(self.exploration_rate)
                if best_child is None:
                     logger.warning("Selection reached fully expanded node with no children?")
                     return current_node 
                current_node = best_child
                logger.info(f"Moved to child with state ending: ...{current_node.state[-100:].replace(os.linesep, ' ')}")

        logger.info("Selection reached a terminal or leaf node.")
        return current_node 

    def _expand(self, node: Node) -> Node:
        """Phase 2: Expansion - Add a new child node by generating a continuation."""
        logger.info("--- Expansion Phase ---")
      
        prompt = f"{node.state}\n\n# Continue the solution above. Provide the next logical step, refinement, or code segment.\n# If the solution seems complete, write the final code.\n# Ensure code is in a python block like ```python ... ```"

        self.llm_calls += 1
        logger.info("Calling LLM for expansion...")
        llm_response = ask_llm(prompt, temperature=0.7) 

        if not llm_response:
            logger.warning("Expansion failed: LLM returned empty response.")
            node.is_terminal = True 
            return node 
        new_state = node.state + "\n\n" + llm_response.strip()
        new_child = Node(parent=node, state=new_state)
        node.children.append(new_child)
        logger.info(f"Expanded with new child. State ends: ...{new_child.state[-100:].replace(os.linesep, ' ')}")
        return new_child

    def _simulate(self, node: Node) -> float:
  
        logger.info("--- Simulation Phase ---")

        if node.code_executes is not None:
             logger.info("Using cached execution result.")
             return 1.0 if node.code_executes else -0.5 

        code = extract_code(node.state)
        reward = 0.0

        if code:
            logger.info("Code found, attempting execution.")
            self.code_executions += 1
            exec_result = {}
            safe_exec(code, exec_result)
            if exec_result.get('error') is None:
                logger.info(f"Execution successful. Output: {exec_result.get('output', 'N/A')}")
                reward = 1.0 
                node.code_executes = True
           
                if self.best_code_found is None:
                     self.best_code_found = code
                     self.best_exec_result = exec_result.get('output', 'N/A')

            else:
                logger.warning(f"Execution error: {exec_result['error']}")
                reward = -0.5 
                node.code_executes = False
        else:
            logger.info("No executable code found in this node's state.")
           
            node.code_executes = False 

        if reward == 1.0:
            node.is_terminal = True
            logger.info("Marking node as terminal due to successful execution.")

        return reward


    def _backpropagate(self, node: Node, reward: float):
        """Phase 4: Backpropagation - Update visits and value up the tree."""
        logger.info(f"--- Backpropagation Phase (Reward: {reward:.2f}) ---")
        current_node = node
        path_length = 0
        while current_node is not None:
            current_node.visits += 1
            current_node.value += reward
            logger.debug(f"  Updated Node (Depth {path_length}): Visits={current_node.visits}, Value={current_node.value:.2f}")
            current_node = current_node.parent
            path_length += 1
        logger.info(f"Backpropagated up {path_length} levels.")


    def search(self):
        """Runs the MCTS search for a specified number of iterations."""
        logger.info(f"Starting MCTS search with {self.iterations} iterations.")

        for i in range(self.iterations):
            start_iter_time = time.time()
            logger.info(f"\n===== Iteration {i + 1}/{self.iterations} =====")

            selected_node = self._select(self.root)

        
      
            simulation_reward = self._simulate(selected_node)

            self._backpropagate(selected_node, simulation_reward)

            end_iter_time = time.time()
            logger.info(f"Iteration {i+1} finished in {end_iter_time - start_iter_time:.2f} seconds.")

         

        best_node = self.get_best_final_node()
        final_code = extract_code(best_node.state) if best_node else None

        logger.info("MCTS search completed.")
        return best_node.state if best_node else "No satisfactory solution found.", \
               best_node.value / best_node.visits if best_node and best_node.visits > 0 else 0, \
               final_code, \
               self.best_exec_result if best_node and best_node.code_executes else "N/A"

    def get_best_final_node(self) -> Node | None:
        """Selects the best child of the root based on robustness (visits) or value."""
        if not self.root.children:
            return None

       
        best_child = max(self.root.children, key=lambda c: c.value / c.visits if c.visits > 0 else -float('inf'))

        logger.info(f"Best Root Child: Visits={best_child.visits}, Avg Value={best_child.value / best_child.visits if best_child.visits > 0 else 'N/A'}")
     
        return best_child



if __name__ == "__main__":
    initial_question = "crea codigo en python, para un bubble sort de una lista de numeros"
    iterations = 10
    max_children_per_node = 3 

    mcts = MCTS(initial_question, iterations=iterations, max_children=max_children_per_node)

    start_search_time = time.time()
   
    best_state, best_avg_reward, final_code, final_output = mcts.search()
    end_search_time = time.time()

    print("\n--- MCTS Result ---")
    print(f"Search Time: {end_search_time - start_search_time:.2f} seconds")
    print(f"Total LLM Calls: {mcts.llm_calls}")
    print(f"Total Code Executions: {mcts.code_executions}")
    print(f"Best Average Reward (Root Child): {best_avg_reward:.4f}")
    print("\nBest Final State (Sequence):")
    print(best_state)
    print("\nExtracted Code from Best State:")
    print("```python")
    print(final_code if final_code else "# No code found or extracted.")
    print("```")
    print(f"\nExecution Result of Best Code: {final_output}")


    if final_code:
         print("\n--- Verifying final code again ---")
         verify_result = {}
         safe_exec(final_code, verify_result)
         print(f"Verification Output: {verify_result.get('output', 'N/A')}")
         print(f"Verification Error: {verify_result.get('error', 'None')}")
