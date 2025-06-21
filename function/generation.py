import torch
from langchain_huggingface import HuggingFacePipeline
from transformers import (AutoTokenizer, 
                          AutoModelForCausalLM,
                          BitsAndBytesConfig,
                          pipeline,
                          TextStreamer)

from langchain_core.prompts import PromptTemplate


def init_model():
    """
    initialize the model by using hugging face pipeline api and linked it with the langchain
    """
    model_id = "meta-llama/Llama-3.2-3B-Instruct"
    # default tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    streamer = TextStreamer(tokenizer)

    # model configuration
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        # quantization_config = quantization_config,
        torch_dtype=torch.float32
    )
    pipe = pipeline(
        "text-generation",
        model=model, # use the model in the above
        tokenizer=tokenizer,
        max_new_tokens=512, # these are the generative config
        pad_token_id=tokenizer.eos_token_id,
        do_sample = False,
        streamer=streamer
    )
    # create a chain for us to use langchain in further process
    llm = HuggingFacePipeline(pipeline=pipe)

    return llm

def get_model_response(user_prompt, system_prompt=None, llm=None):
    if llm is None:
        raise "please input the llm object for generation, llm cannot be null"

    # Base template without system prompt
    template_user_only = """
        <|begin_of_text|>
        <|start_header_id|>user<|end_header_id|>
        {user_prompt}
        <|eot_id|>
        <|start_header_id|>assistant<|end_header_id|>
        """

    # Template with system prompt
    template_with_system = """
        <|begin_of_text|>
        <|start_header_id|>system<|end_header_id|>
        {system_prompt}
        <|eot_id|>
        <|start_header_id|>user<|end_header_id|>
        {user_prompt}
        <|eot_id|>
        <|start_header_id|>assistant<|end_header_id|>
        """

    # Select template based on whether system_prompt is provided
    if system_prompt:
        template = template_with_system
    else:
        template = template_user_only

    # Create the prompt using the selected template
    prompt = PromptTemplate(
        input_variables=["user_prompt"] if not system_prompt else ["system_prompt", "user_prompt"],
        template=template
    )
    
    # Format the prompt and get the response
    if system_prompt:
        formatted_prompt = prompt.format(system_prompt=system_prompt, user_prompt=user_prompt)
    else:
        formatted_prompt = prompt.format(user_prompt=user_prompt)
    response = llm.invoke(formatted_prompt)
    response_content = response.split("<|start_header_id|>assistant<|end_header_id|>")[-1].strip()
    return response_content
