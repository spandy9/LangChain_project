from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI()

parser = StrOutputParser()

prompt1 = PromptTemplate(
    template = "Generate a short note on the following text:\n {text}",
    input_variables=["text"]
)

prompt2 = PromptTemplate(
    template = "Generate a set of short 3 question and answers based on the following text:\n {text}",
    input_variables=["text"]
)

prompt3 = PromptTemplate(
    template = "Merge the short notes and set of three questions into a single document \n {notes} and {quiz}",
    input_variables=["notes", "quiz"]
)

parallel_chain = RunnableParallel(
    {
        "notes": prompt1 | model | parser,
        "quiz": prompt2 | model | parser
    }
)

final_chain = parallel_chain | prompt3 | model | parser

result = final_chain.invoke({"text": "The Great Wall of China is a series of fortifications that were built across the historical northern borders of China to protect against invasions. The wall stretches over 13,000 miles and is one of the most iconic landmarks in the world. It was constructed over several centuries, with the earliest sections dating back to the 7th century BC. The Great Wall is not a single continuous wall but rather a collection of walls and fortifications built by different dynasties. It is a UNESCO World Heritage Site and attracts millions of tourists each year."})

print(result)

final_chain.get_graph().print_ascii()