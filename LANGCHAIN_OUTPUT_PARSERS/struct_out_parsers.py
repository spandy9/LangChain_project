from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI()
# schema
schema = [
    ResponseSchema(name="fact_1", description="Write down the first fact about the topic"),
    ResponseSchema(name="fact_2", description="Write down the second fact about the topic"),
    ResponseSchema(name="fact_3", description="Write down the third fact about the topic")
]

parser = StructuredOutputParser.from_response_schemas(schema)

template = PromptTemplate(
    template="Write down facts about {topic} /n {format_instructions}",
    input_variables = ["topic"],
    partial_variables = {"format_instructions": parser.get_format_instructions()}
)

chain = template | model | parser

result = chain.invoke({"topic": "black holes"})

print(result)

