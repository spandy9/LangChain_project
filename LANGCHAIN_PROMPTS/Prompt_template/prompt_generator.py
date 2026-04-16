from langchain_core.prompts import PromptTemplate

template = PromptTemplate(
    template="""
    Please summarize the research paper "{paper_input}" with the following specifications:
    Explanation Style: {style_input}
    Explanation Length: {length_input}
    1. Mathematical Details:
        - Include relevant mathematical equations if available in the paper
        - Explain the mathematical concepts using simple intuitive code snippets where applicable
    2. Anologies:
        - Use analogies where possible to simplify complex ideas
    If certain information is not available in the paper, respond with: "Insufficient information" and do not hallucinate.
    Ensure the summary is clear, accurate and aligned with the provided style and length
    """,
    input_variables=['paper_input', 'style_input', 'length_input'],
    validate_template= True
)

template.save('template.json')
