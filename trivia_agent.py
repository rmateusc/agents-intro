from dotenv import load_dotenv
from IPython.display import Image, display
from langchain.schema import HumanMessage, SystemMessage
from langchain_core.runnables.graph import CurveStyle, MermaidDrawMethod, NodeStyles
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from pydantic import BaseModel

# Initialize LLM
load_dotenv(".env", override=True)
llm = ChatOpenAI(model_name="gpt-4", temperature=0.7)


class QuizState(BaseModel):
    question: str = ""
    score: int = 0


def generate_question(topic="general knowledge"):
    """Generates a trivia question based on a topic."""
    prompt = [
        SystemMessage(
            content=f"You are a trivia quiz master. Ask a multiple-choice question about {topic}."
        )
    ]
    response = llm.invoke(prompt)
    return response.content


def check_answer(question, user_answer):
    """Validates the user's answer."""
    prompt = [
        SystemMessage(
            content=f"You are a trivia judge. Given this question: '{question}', is the answer '{user_answer}' correct? Respond with 'Yes' or 'No'."
        )
    ]
    response = llm.invoke(prompt)
    return response.content.strip().lower() == "yes"


# LangGraph setup
class QuizGraph:
    def __init__(self):
        self.workflow = StateGraph(QuizState)
        self.workflow.add_node("ask_question", self.ask_question)
        self.workflow.add_node("evaluate_answer", self.evaluate_answer)
        self.workflow.set_entry_point("ask_question")
        self.workflow.add_edge("ask_question", "evaluate_answer")
        self.workflow.add_edge("evaluate_answer", "ask_question")
        self.workflow.add_edge("ask_question", END)

        # Compile the workflow
        self.compiled_workflow = self.workflow.compile()

        compiled_workflow = self.compiled_workflow

        # render and save the graph
        react_graph = compiled_workflow

        # Render and save graph
        graph_data = react_graph.get_graph(xray=True)
        mermaid_code = graph_data.draw_mermaid_png()
        with open("workflow_graph.png", "wb") as f:
            f.write(mermaid_code)

    def ask_question(self, state: QuizState):
        state.question = generate_question()
        return state

    def evaluate_answer(self, state: QuizState):
        user_answer = input(f"{state.question}\nYour answer: ")
        if check_answer(state.question, user_answer):
            state.score += 1
            print("Correct! 🎉")
        else:
            print("Wrong answer. ❌")
        print(f"Current Score: {state.score}\n")
        return state

    def run(self, state: QuizState):
        return self.compiled_workflow.invoke(state)


# Run the quiz
if __name__ == "__main__":
    quiz = QuizGraph()
    state = QuizState()

    for _ in range(3):  # Run 3 rounds of quiz
        _, state = quiz.run(state)
