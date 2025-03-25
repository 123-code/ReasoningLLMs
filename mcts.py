import subprocess
from venv import logger
from groq import Groq
import numpy as np
import random
import math
import re
from sympy import symbols,integrate,sin,cos,diff

opciones_solucion = []


def estimate_code_quality(code):
    try:
        if "think" in code:
            value = 0.9
        return value
    except Exception as e:
        return 0.1



def clean_code(code):
    pattern = r"```(?:python)?\s*([\s\S]*?)```"
    code_blocks = re.findall(pattern,code,flags=re.MULTILINE)
    code = "\n".join(code_blocks)
    return code

def execute_code(code):
    try:
        process = subprocess.Popen(['python', '-c', code], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = process.communicate(timeout=10)
        if process.returncode == 0:
            return "success", stdout if stdout else "No output"
        else:
            return "error", stderr
    except subprocess.TimeoutExpired:
        process.kill()
        return "timeout", ""
    except Exception as e:
        return "error", str(e)

def create_response(user_prompt: str):
    client = Groq(api_key="AGREGAR API KEY")
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": """
                 Eres un programador en python, que ejecuta acciones, te dan una opcion de solucion, y tu la escribes el codigo necesario para ejecutarla
                """
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        model="llama3-70b-8192",
        temperature = 1
    )
    response = chat_completion.choices[0].message.content
    print(response)
    return response


def reason_on_response(output_esperado:str,opciones_anteriores:list):
        client = Groq(api_key="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")
        chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": """
                Razonas posibles caminos hacia la solucion de cierto problema, te dan una lista de opciones ya enlistadas, y ofreces otras alternativas
                """
            },
            {
                "role": "user",
                "content": f"queremos hacer:{output_esperado}, y tenemos por ahora estas ideas:{opciones_anteriores}, por favor sugiereme otra alternativa (unicamente una) para resolver el problema."
            }
        ],
        model="llama3-70b-8192",
        temperature = 1
        )
        response = chat_completion.choices[0].message.content
        opciones_solucion.append(response)
        logger.info(f"opcion hacia la solucion: {response}")
        print(response)
        return response


def get_critique(question: str, answer: str):
    prompt = f"""
    Pregunta: {question}
    Respuesta: {answer}

    Por favor dime cómo mejorarías esta respuesta a la pregunta indicada. Da instrucciones breves y concisas.
    """
    return create_response(prompt)



def improve_answer(question: str, answer: str):
    critique = get_critique(question, answer)
    prompt = f"""
    Pregunta: {question}
    Respuesta: {answer}
    Crítica: {critique}

    Por favor provee una respuesta mejorada a la pregunta indicada, basado en la crítica.
    Para cada respuesta, dame cuál fue tu proceso de razonamiento. Usa el formato:
    Razonamiento: <Proceso de razonamiento>
    Verificación: <Verificación de la respuesta>
    Respuesta final: <la respuesta final verificada>
    """
    return create_response(prompt)


max_children = 5

class Node:
    def __init__(self, question, answer, parent=None):
        self.question = question
        self.answer = answer
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0

    def is_fully_expanded(self):
        return len(self.children) >= max_children

    def get_best_child(self, exploration_rate=1.41):
        choices_weights = []
        for child in self.children:
            if child.visits == 0:
                weight = float('inf')
            else:
                weight = (child.value / child.visits) + exploration_rate * math.sqrt((2 * math.log(self.visits) / child.visits))
            choices_weights.append(weight)
        return self.children[np.argmax(choices_weights)]

    def add_child(self, child_node):
        self.children.append(child_node)




class MCTS:
    def __init__(self, question, iterations=5,max_children=5):
        self.question = question
        self.iterations = iterations
        self.max_children=max_children
        self.root = self.create_root_node()


    def create_root_node(self):
        root_node = Node(self.question, "")

        for x in range(self.max_children):
            solution_option = reason_on_response(self.question,opciones_solucion)
            child_node = Node(self.question, solution_option, parent=root_node)
            root_node.add_child(child_node)
        return root_node

    def select(self, node):
        while node.is_fully_expanded() and node.children:
            node = node.get_best_child()
        return node

    def expand(self, node):
        for j in range(max_children - len(node.children)):
            child_node = Node(self.question, "", parent=node)
            improved = improve_answer(self.question, node.answer)
            child_node.answer = improved
            node.add_child(child_node)
        return random.choice(node.children)

    def search(self):
        best_overall_child = None
        best_overall_score = float('-inf')
        best_output = ""

        for x in range(self.iterations):
            node = self.select(self.root)
            if not node.is_fully_expanded():
                node = self.expand(node)
            reward, output = self.simulate(node)
            self.backpropagate(node, reward)
            best_child = self.root.get_best_child()
            if best_child.value > best_overall_score:
                best_overall_score = best_child.value
                best_overall_child = best_child
                best_output = output if reward == 1.0 else ""
            yield {
                "best_answer": best_child.answer,
                "success_rate": (best_child.value / best_child.visits * 100 if best_child.visits > 0 else 0),
                "output": output if reward == 1.0 else ""
            }
            if reward == 1.0:  # Stop on first success
                break

        # Final result (optional, if needed)
        success_rate = (best_overall_child.value / best_overall_child.visits * 100 
                        if best_overall_child and best_overall_child.visits > 0 else 0)
        yield {
            "best_answer": best_overall_child.answer if best_overall_child else "No solution found",
            "success_rate": success_rate,
            "output": best_output
        }


    def simulate(self, node):
        improved_answer = improve_answer(self.question, node.answer)
        node.answer = improved_answer
        status, output = execute_code(clean_code(improved_answer))
        return 1.0 if status == "success" else 0.0, output
       


    def backpropagate(self, node, reward):
        while node is not None:
            node.visits += 1
            node.value += reward
            node = node.parent


if __name__ == "__main__":
    question = "crea codigo en python, para un bubble sort de una lista de numeros"
    mcts = MCTS(question, iterations=5, max_children=5)
    best_answer = mcts.search()
    logger.info(f"Best answer: {best_answer}")
