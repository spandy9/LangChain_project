from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

model = ChatOpenAI()

# schema
class Person(BaseModel):

    name: str = Field(description="Write the name of the person")
    age: int = Field(gt = 18, description="Write the age of the person")
    city: str = Field(description="Write the city where the person lives")

parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template="Give the name, age and city of a {region} person /n {format_instructions}",
    input_variables=["region"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

chain = template | model | parser

result = chain.invoke({"region": "European"})

print(result)