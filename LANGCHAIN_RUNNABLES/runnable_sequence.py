from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence
from dotenv import load_dotenv

load_dotenv()

prompt = PromptTemplate(
    template="Write a joke about {topic}",
    input_variables=["topic"]
)

model = ChatOpenAI()

parser = StrOutputParser()

chain = RunnableSequence(prompt, model, parser)



prompt2 = PromptTemplate(
    template="Explain the joke:{text}",
    input_variables=["text"]
)
chain = RunnableSequence(prompt, model, parser, prompt2, model, parser)
result = chain.invoke({"topic": "programming"})
print(result)