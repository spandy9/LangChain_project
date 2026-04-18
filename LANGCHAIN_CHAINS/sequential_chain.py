from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
model = ChatOpenAI()
parser = StrOutputParser()
prompt1 = PromptTemplate(
    template="Generate a detailed report about {topic}",
    input_variables=["topic"]
)
prompt2 = PromptTemplate(
    template="Summarize the following report in 5 bullet points:\n\n{report}",
    input_variables=["report"]
)
chain = prompt1 | model | parser | prompt2 | model | parser

result = chain.invoke({"topic": "Unemployment in India"})

print(result)

chain.get_graph().print_ascii()