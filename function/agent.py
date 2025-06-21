# domain generation agent
# ****related****
from typing import TypedDict
from langgraph.graph import MessagesState
from langchain_core.prompts import PromptTemplate
from .generation import get_model_response
from langgraph.graph import StateGraph, START, END
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain_core.utils.json import OutputParserException
from typing import Literal
from langchain.llms.huggingface_pipeline import HuggingFacePipeline

#---node 1 prompt---
system_prompt_node1 ="""You are required to generate domain names based on a given topic. \
For every request, you will be provided with a topic, and your task \
is to generate domain names that include the word 'DOMAIN' before \
each suggested name. Below are examples of how the input request \
and your response should look like:\

Example:
\
Request: Please help me generate 3 domain names about "cooking" with tld .com
Response:
DOMAIN: TastyHub.com
DOMAIN: RecipeRush.com
DOMAIN: TheCookingCoach.com"""

user_prompt_node1 = "Please help me generate {number} domains related to \"{topic}\" with tld \"{tld}\" and please do not generate domain that already exisit"

# -----end of node1-----

# -----prompt for node2(extraction)-----

# what information to extract
info = "- **domain**: Identify all domain names mentioned in the text and return them as a list. "

review_template1 = """\
For the given text, extract the following information:

{info}

**Text To Analyze:**
{text}
**End Of Text**
{format_instructions}

Please output the result strictly in valid JSON format, without any additional text, commentary, or explanation.

"""

format_instructions = """\
here is a example output:
```
{
	"domain": string  //     Extract the domain names mentioned in the text.
    For example, if the text contains "The domains are hello.com and byebye.net",
    the output should be a list: ["hello.com", "byebye.net"].
    
}
```

"""

review_template1 = """\
For the given text, extract the following information:

- **domain**: Identify all domain names mentioned in the text and return them as a list. 

**Text To Analyze:**
{text}
**End Of Text**
here is a example output:
```
{{
	"domain": string  //     Extract the domain names mentioned in the text.
    For example, if the text contains "The domains are hello.com and byebye.net",
    the output should be a list: ["hello.com", "byebye.net"].
    
}}
```
Please output the result strictly in valid JSON format, without any additional text, commentary, or explanation.

"""

# ----- end of node2 -----

# ---- prompt for parse_domain_correct----

system_parse_domain_correct = """
Here’s a revised version of your prompt that improves clarity and grammar:

Your task is to transform input data into a specific format. Specifically, fix outputs that do not conform to the required structure:
```
{
	"domain": [string]
}
```
Example

Input:
```
[
  "ChallengeAccepted.com",
  "ChallengeHub.com",
  "TheChallengeCenter.com"
]
```
Problem:
The input is missing the "domain" key.

Correct Output:
```
{
	"domain": [
		"ChallengeAccepted.com",
		"ChallengeHub.com",
		"TheChallengeCenter.com"
	]
}
```
Ensure all outputs follow the specified format and please do not need any additional comment.
"""

# ---- end of parse_domain_correct ----

# ---- prompt for domain_generate_decision ----
system_domain_generate_decision = """\
You are tasked with inspecting the output of another model to determine if it contains a domain.
Follow these steps to arrive at your answer:
1.Analyze the input step by step to check if it contains any pattern that resembles a domain.
2.Conclude by determining whether the input contains a domain.
Your final response must be formatted as follows:

Reasoning: (Whether there is a valid domain inside the text)
Answer: [TRUE, FALSE]
for example:
input:
the text you need to analyze is "
DOMAIN: TastyHub.com
DOMAIN: RecipeRush.com
DOMAIN: TheCookingCoach.com"
your response should be: 
Resoning: there are domain which fit the format of a standard domain inside the text. 
So the Answer should be: TRUE
or input:
the text you need to analyze is "hello, how can i help you today?"
your response should be: 
Resaoning: there is no domain which fit the format of a standard domain inside the text.
So the Answer should be: FALSE
"""
user_domain_generate_decision = """\
the text you need to analyze is "{text}"
"""

class Agent_State(TypedDict):
    state: MessagesState
    number: int
    topic: str
    tld: str
    domain: dict
    llm: HuggingFacePipeline

# generate domain
def domain_generate(agent_state: Agent_State):
    print("---Generating domain---")
    # context
    messages = agent_state["state"]["messages"]
    # initialize the prompt
    node1_prompt = PromptTemplate(
        template=user_prompt_node1
    )
    llm = agent_state['llm']
    prompt_input = node1_prompt.format(
    number = agent_state['number'],
    topic = agent_state['topic'],
    tld = agent_state['tld'],
    )
    # print("the input message:", prompt_input)    
    # input the message to the model
    output = get_model_response(prompt_input, system_prompt_node1, llm)
    # print("model response: ", output)
    # print("messages[:1]:", messages[:1])
    new_messages = messages[:1] + [output] + messages[1:]
    print("new_messages:", new_messages)
    updated_state = MessagesState(messages=new_messages)
    
    # update the state object to output
    return {
      "state": updated_state
    }


def parse_domain(agent_state: Agent_State):
    print('Now we try to parse the output to structured format and inspect the result.\n')
    print("Prompt Template:\n", review_template1)

    messages = agent_state["state"]["messages"]
    text_to_analyze = messages[1]

    domain_schema = ResponseSchema(
        name="domain",
        description="""\
        Extract the domain names mentioned in the text.
        For example, if the text contains: "The domains are hello.com and byebye.net",
        the output should be a list like: ["hello.com", "byebye.net"].
        """
    )
    response_schemas = [domain_schema]

    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

    review_prompt = PromptTemplate(template=review_template1)
    prompt_input = review_prompt.format(text=text_to_analyze)
    print("Formatted prompt:\n", prompt_input)

    llm = agent_state['llm']
    output = get_model_response(prompt_input, llm=llm)
    print("Raw output from model:\n", output)

    try:
        parsed_output = output_parser.parse(output)
        print("Parsed output:", parsed_output)
        domain_list = parsed_output.get("domain", [])
    except Exception as e:
        print("Failed to parse model output:", e)
        domain_list = []

    new_messages = messages[:2] + [output] + messages[2:]
    updated_state = MessagesState(messages=new_messages)

    return {
        "state": updated_state,
        "domain": domain_list
    }


def structed_output(agent_state: Agent_State)->Literal['generate_domain', 'finish', 'parse_domain_correct']:
    domain_schema = ResponseSchema(
    name="domain",
    description="""\
    Was the domain name of the text \
    For example, if the text contains The domains are "hello.com and byebye.net"  \
    the output should be a list: ["hello.com", "byebye.net"].
    """
    )
    response_schemas = [domain_schema,]
    # create a parser object which can parse the output
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    
    try:
        text_to_analyze = agent_state["state"]["messages"][2]
        # parse the response to dictionary
        output_dict = output_parser.parse(text_to_analyze)
        
        # check whether output_dict is empty and also output_dict['domain'] is empty
        if not output_dict or not output_dict.get('domain'):
          print("The output is null or empty.")
          return 'generate_domain'
        else:
          print("The output has valid data:", output_dict)
          return 'finish'
    except OutputParserException as e:
        message = "The input format does not fit the format"
        print("The input format does not fit the format")
        return 'parse_domain_correct'

def parse_domain_correct(agent_state: Agent_State):
    text_to_fix = agent_state["state"]["messages"][2]
    messages = agent_state["state"]["messages"]
    llm = agent_state['llm']
    output = get_model_response(text_to_fix, system_parse_domain_correct, llm=llm)
    
    new_messages = messages[:2] + [output] + messages[2:]
    print("new_messages:", new_messages)
    updated_state = MessagesState(messages=new_messages)
    num = agent_state['number'] - 1
    return {
        "state": updated_state,
        "number": num
    }

def domain_generate_decision(agent_state: Agent_State)->Literal['parse_domain', 'shut_down']:
    text_to_check = agent_state["state"]["messages"][1]
    context = agent_state["state"]["messages"]
    user_template = PromptTemplate(template = user_domain_generate_decision)
    prompt_input = user_template.format(text = text_to_check)
    llm = agent_state['llm']
    output = get_model_response(prompt_input, system_domain_generate_decision, llm=llm)
    if 'TRUE' in output:
        return 'parse_domain'
    else:
        return 'shut_down'

def shut_down(agent_state: Agent_State):
    print('operation failed, there is no domain inside this ')
    output_dict = {}
    output_dict['domain'] = ['google.com']
    return{
        'domain': output_dict
    }
    
def finish(agent_state: Agent_State):
    print('operation are finished!')
    domain_schema = ResponseSchema(
    name="domain",
    description="""\
    Was the domain name of the text \
    For example, if the text contains The domains are "hello.com and byebye.net"  \
    the output should be a list: ["hello.com", "byebye.net"].
    """
    )
    response_schemas = [domain_schema,]
    # create a parser object which can parse the output
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    
    text_to_analyze = agent_state["state"]["messages"][2]
    print(text_to_analyze)
    # parse the response to dictionary
    output_dict = output_parser.parse(text_to_analyze)
    output_dict['domain'] = [domain.lower() for domain in output_dict['domain']]
    return{
        'domain': output_dict
    }


def build_agent():
    builder = StateGraph(Agent_State)
    builder.add_node("generate_domain", domain_generate)
    builder.add_node("parse_domain", parse_domain)
    builder.add_node("parse_domain_correct", parse_domain_correct)
    builder.add_node("shut_down", shut_down)
    builder.add_node("finish", finish)

    builder.add_edge(START, "generate_domain")
    builder.add_conditional_edges("generate_domain", domain_generate_decision)
    builder.add_conditional_edges("parse_domain", structed_output)
    builder.add_edge("parse_domain_correct", "generate_domain")
    builder.add_edge("shut_down", END)
    builder.add_edge("finish", END)
    graph = builder.compile()
    return graph

# view
# display(Image(graph.get_graph().draw_mermaid_png()))