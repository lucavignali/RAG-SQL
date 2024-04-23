import boto3
import sys

import langchain

import langchain
from langchain_community.retrievers import AmazonKendraRetriever

from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_aws import BedrockLLM
from langchain.chains import StuffDocumentsChain, LLMChain
from langchain.prompts import PromptTemplate

# bedrock client of langchain
llm = BedrockLLM(model_id="anthropic.claude-v2")

llm.model_kwargs = {"temperature": 0.0, "max_tokens_to_sample": 4096}

conversation = ConversationChain(
    llm=llm, verbose=True, memory=ConversationBufferMemory()
)

if len(sys.argv)-1 > 0:
    query = sys.argv[1]
    print("Q:", query)
    retriever = AmazonKendraRetriever(index_id="04b7df84-ddab-4c84-962e-3734de1f48c1",region_name ="eu-west-1")
    documents = retriever.get_relevant_documents(query)
    print("Found {} documents on Kendra".format(len(documents)))
    
    document_prompt = PromptTemplate(
        input_variables=["page_content"], template="{page_content}"
    )

    document_variable_name = "context"

    stuff_prompt_override = """Given this text extracts:
        -----
        {context}
        -----
        Please answer the following question with plenty of details and explain your answer:
        {query}"""


    prompt = PromptTemplate(
        template=stuff_prompt_override, input_variables=["context", "query"]
    )
    
     # Instantiate the chain
    llm_chain = LLMChain(llm=llm, prompt=prompt)

    chain = StuffDocumentsChain(
       llm_chain=llm_chain,
       document_prompt=document_prompt,
       document_variable_name=document_variable_name,
    )
    
    input_data = {
    'input_documents': documents,
    'query': query,
    }
    
    print(chain.invoke(input = input_data)['output_text'])


