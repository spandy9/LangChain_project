from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableBranch, RunnableLambda
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI()

class ReviewSentiment(BaseModel):
    sentiment: Literal["positive", "negative"] = Field(description="The sentiment of the review, either positive or negative")

parser1 = PydanticOutputParser(pydantic_object=ReviewSentiment)
parser2 = StrOutputParser()

prompt1 = PromptTemplate(
    template = "Determine the sentiment of the following review: \n {review} \n {format_instructions}",
    input_variables=["review"],
    partial_variables={"format_instructions": parser1.get_format_instructions()}
)

prompt_positive = PromptTemplate(
    template = "Provide an appropriate response to the positive review: \n {review}",
    input_variables=["review"]
)

prompt_negative = PromptTemplate(
    template = "Provide an appropriate response to the negative review: \n {review}",
    input_variables=["review"]
)

classifier_chain = prompt1 | model | parser1

branch_chain = RunnableBranch(
    (lambda x: x.sentiment == "positive", prompt_positive | model | parser2),
    (lambda x: x.sentiment == "negative", prompt_negative | model | parser2),
    RunnableLambda(lambda x: "Invalid sentiment")
)

final_chain = classifier_chain | branch_chain

result = final_chain.invoke({"review": "This smartphone is the worst smartphone I have ever owned!"})

print(result)

final_chain.get_graph().print_ascii()