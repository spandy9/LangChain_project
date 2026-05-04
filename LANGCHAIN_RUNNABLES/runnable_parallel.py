from langchain_core.runnables import RunnableSequence, RunnableParallel
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

prompt1 = PromptTemplate(
    template="Write a tweet about {topic}",
    input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template="Generate a LinkedIn post about {topic}",
    input_variables=["topic"]
)

model = ChatOpenAI()

parser = StrOutputParser()

chain = RunnableParallel({
    "tweet": RunnableSequence(prompt1, model, parser),
    "linkedin": RunnableSequence(prompt2, model, parser)
}
)

result = chain.invoke({"topic":"AI"})
print(result)

