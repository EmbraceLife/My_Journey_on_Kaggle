The chatbot repo on huggingface https://huggingface.co/spaces/hwchase17/chat-langchain/tree/main
My replicate of this repo above https://huggingface.co/spaces/DanielCL/chat-langchain/settings
- add repo secret for `WEAVIATE_URL`

## <mark style="background: #FFB8EBA6;">Q & A from chatGPT and chatLangChain</mark> 

#### Qs on #QA_chatGPT


#### What does weaviate library do?

 >Weaviate is an open-source library that allows users to create a semantic search engine using a graph database. It allows you to organize your data in a way that makes it easy to understand the relationships between different pieces of information, and then use that understanding to perform more accurate and relevant searches. It can be used to build a variety of applications, such as chatbots, recommendation systems, and knowledge graphs. Weaviate is built on top of Apache Lucene, an open-source search engine library, which provides powerful search capabilities.


#### what does `WEAVIATE_URL = os.environ["WEAVIATE_URL"]` do?
 
`>WEAVIATE_URL = os.environ["WEAVIATE_URL"]` is a line of code that assigns the value of the environment variable "WEAVIATE_URL" to the variable "WEAVIATE_URL". 

>The `os.environ` object is a built-in Python module that allows you to access environment variables. In this case, the code is accessing the environment variable named "WEAVIATE_URL" and assigning its value to the variable "WEAVIATE_URL" .

>This line of code is typically used when the Weaviate instance is running in a different environment such as a different server or container, allowing the code to connect to the Weaviate instance via the URL specified in the environment variable, rather than hardcoding the URL in the scrip t.

>It's a way to specify the endpoint of the Weaviate instance in a dynamic way, so the same script can be used in different environments with different Weaviate instances.



#### What does `WEAVIATE_URL` look li ke?

>The format of the `WEAVIATE_URL` variable would typically be a string in the form of a URL, starting with the protocol (http or https) followed by the domain name or IP address and the port number of the Weaviate ser ver.

>For example, if Weaviate is running on a server with IP address 192.168.1.100, and it's using the default port number, the `WEAVIATE_URL` woul d be:

>Copy code

`http://192.168.1.100 :8080`

>It's also possible that the Weaviate instance is running behind a reverse proxy, in this case the `WEAVIATE_URL` would be the URL that the reverse proxy is exposing, for example:

Copy code

`http://weaviate.examp le.com`

>It is important to note that the exact format of the `WEAVIATE_URL` will depend on how you have set up and configured your Weaviate instance.



#### What is `WEAVIATE_URL` and why we  need it? 🔥

>`WEAVIATE_URL` is a variable that holds the endpoint URL of a Weaviate server, which is a semantic search engine that allows you to organize your data in a way that makes it easy to understand the relationships between different pieces of information, and then use that understanding to perform more accurate and relevant  searches.

>The `WEAVIATE_URL` variable is used to specify the endpoint of the Weaviate server in a dynamic way, so the same script can be used in different environments with different Weaviate  instances.

>The `WEAVIATE_URL` is needed in order to connect to the Weaviate instance in order to perform various operations such as adding, deleting or querying data, via the Weaviate API. Without specifying the `WEAVIATE_URL` the script wouldn't know where to  connect to.

>In summary, `WEAVIATE_URL` is a variable that contains the endpoint of the Weaviate instance and is used to connect to it, allowing the script to interact with the Weaviate instance via the API.


Then I searched a [blog](https://towardsdatascience.com/getting-started-with-weaviate-python-client-e85d14f19e4f) on how to get started with Weaviate

#### what does library beautifulsoup4 do?

> The `beautifulsoup4` library is used for web scraping in Python. It allows a user to parse HTML and XML documents and extract useful data from them, such as text and attributes of specific elements. The library can also be used to navigate and search the parse tree, making it a powerful tool for extracting and manipulating data from web pages.


#### what do libraries like `black`, `isort`, `Flask` do?

 >`beautifulsoup4` is a library for pulling data out of HTML and XML files.
 
>`black` is a Python code formatter, which automatically reformats code to conform to a consistent style. 

> `isort` is a library for sorting Python imports alphabetically, and automatically separating them into sections .

>`Flask` is a micro web framework for Python. It is used for building web applications and APIs. It is easy to use, lightweight, and has a small and easy-to-extend core. It is also compatible with a number of different templating engines, and allows for easy integration with other libraries and tools.


### How to understand `from langchain.vectorstores import Weaviate`

#QA_chatLangChain
> `from langchain.vectorstores import Weaviate` imports the Weaviate wrapper from the LangChain vectorstores module. This wrapper allows you to use Weaviate as a vectorstore for semantic search or example selection. For more information, see the [Weaviate page](https://langchain.readthedocs.io/en/latest/ecosystem/weaviate.html) in the LangChain documentation.

#QA_chatGPT 
**what does it mean by "This wrapper allows you to use Weaviate as a vectorstore for semantic search or example selection."**

> This statement is saying that the wrapper is a tool that allows the use of Weaviate, which is a vector store, to perform semantic search or selection of examples. In other words, it enables the user to use Weaviate's vector representation capabilities to search for and retrieve relevant information based on the meaning of the query, rather than just keyword matching. It also allows for selection of examples that are semantically similar to a given query.


""" it enables the user to use Weaviate's vector representation capabilities to search for and retrieve relevant information based on the meaning of the query, rather than just keyword matching. It also allows for selection of examples that are semantically similar to a given query. """ 
**Can you explain it as if I am a total beginner?**

> Sure, the text is saying that this <mark style="background: #FF5582A6;">wrapper</mark> (a piece of code that connects to another piece of software) allows you to use Weaviate, which is a tool for understanding and organizing data, to search for information in a way that <mark style="background: #ADCCFFA6;">understands the meaning of what you're looking for</mark> , instead of just matching keywords. It also allows you to find examples that are similar in meaning to what you're looking for. So it's a way to make <mark style="background: #D2B3FFA6;">searching</mark> for information more <mark style="background: #D2B3FFA6;">intelligent and accurate</mark> .


#### What does `from chain import get_new_chain1` do?


---


## <mark style="background: #FFB8EBA6;">codes in the repo</mark> 

```txt requirements.txt
langchain==0.0.64
beautifulsoup4
weaviate-client
openai
black
isort
Flask
transformers
gradio
```

```python app.py
import datetime
import os

import gradio as gr
import langchain
import weaviate
from langchain.vectorstores import Weaviate

from chain import get_new_chain1

WEAVIATE_URL = os.environ["WEAVIATE_URL"]


def get_weaviate_store():
    client = weaviate.Client(
        url=WEAVIATE_URL,
        additional_headers={"X-OpenAI-Api-Key": os.environ["OPENAI_API_KEY"]},
    )
    return Weaviate(client, "Paragraph", "content", attributes=["source"])


def set_openai_api_key(api_key, agent):
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
        vectorstore = get_weaviate_store()
        qa_chain = get_new_chain1(vectorstore)
        os.environ["OPENAI_API_KEY"] = ""
        return qa_chain


def chat(inp, history, agent):
    history = history or []
    if agent is None:
        history.append((inp, "Please paste your OpenAI key to use"))
        return history, history
    print("\n==== date/time: " + str(datetime.datetime.now()) + " ====")
    print("inp: " + inp)
    history = history or []
    output = agent({"question": inp, "chat_history": history})
    answer = output["answer"]
    history.append((inp, answer))
    print(history)
    return history, history


block = gr.Blocks(css=".gradio-container {background-color: lightgray}")

with block:
    with gr.Row():
        gr.Markdown("<h3><center>LangChain AI</center></h3>")

        openai_api_key_textbox = gr.Textbox(
            placeholder="Paste your OpenAI API key (sk-...)",
            show_label=False,
            lines=1,
            type="password",
        )

    chatbot = gr.Chatbot()

    with gr.Row():
        message = gr.Textbox(
            label="What's your question?",
            placeholder="What's the answer to life, the universe, and everything?",
            lines=1,
        )
        submit = gr.Button(value="Send", variant="secondary").style(full_width=False)

    gr.Examples(
        examples=[
            "What are agents?",
            "How do I summarize a long document?",
            "What types of memory exist?",
        ],
        inputs=message,
    )

    gr.HTML(
        """
    This simple application is an implementation of ChatGPT but over an external dataset (in this case, the LangChain documentation)."""
    )

    gr.HTML(
        "<center>Powered by <a href='https://github.com/hwchase17/langchain'>LangChain 🦜️🔗</a></center>"
    )

    state = gr.State()
    agent_state = gr.State()

    submit.click(chat, inputs=[message, state, agent_state], outputs=[chatbot, state])
    message.submit(chat, inputs=[message, state, agent_state], outputs=[chatbot, state])

    openai_api_key_textbox.change(
        set_openai_api_key,
        inputs=[openai_api_key_textbox, agent_state],
        outputs=[agent_state],
    )

block.launch(debug=True)

```

```python chain.py
import json
import os
import pathlib
from typing import Dict, List, Tuple

import weaviate
from langchain import OpenAI, PromptTemplate
from langchain.chains import LLMChain
from langchain.chains.base import Chain
from langchain.chains.combine_documents.base import BaseCombineDocumentsChain
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.chains.question_answering import load_qa_chain
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from langchain.prompts.example_selector import \
    SemanticSimilarityExampleSelector
from langchain.vectorstores import FAISS, Weaviate
from pydantic import BaseModel




class CustomChain(Chain, BaseModel):

    vstore: Weaviate
    chain: BaseCombineDocumentsChain
    key_word_extractor: Chain

    @property
    def input_keys(self) -> List[str]:
        return ["question"]

    @property
    def output_keys(self) -> List[str]:
        return ["answer"]

    def _call(self, inputs: Dict[str, str]) -> Dict[str, str]:
        question = inputs["question"]
        chat_history_str = _get_chat_history(inputs["chat_history"])
        if chat_history_str:
            new_question = self.key_word_extractor.run(
                question=question, chat_history=chat_history_str
            )
        else:
            new_question = question
        print(new_question)
        docs = self.vstore.similarity_search(new_question, k=4)
        new_inputs = inputs.copy()
        new_inputs["question"] = new_question
        new_inputs["chat_history"] = chat_history_str
        answer, _ = self.chain.combine_docs(docs, **new_inputs)
        return {"answer": answer}


def get_new_chain1(vectorstore) -> Chain:
    WEAVIATE_URL = os.environ["WEAVIATE_URL"]
    client = weaviate.Client(
        url=WEAVIATE_URL,
        additional_headers={"X-OpenAI-Api-Key": os.environ["OPENAI_API_KEY"]},
    )

    _eg_template = """## Example:

    Chat History:
    {chat_history}
    Follow Up Input: {question}
    Standalone question: {answer}"""
    _eg_prompt = PromptTemplate(
        template=_eg_template,
        input_variables=["chat_history", "question", "answer"],
    )

    _prefix = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question. You should assume that the question is related to LangChain."""
    _suffix = """## Example:

    Chat History:
    {chat_history}
    Follow Up Input: {question}
    Standalone question:"""
    eg_store = Weaviate(
        client,
        "Rephrase",
        "content",
        attributes=["question", "answer", "chat_history"],
    )
    example_selector = SemanticSimilarityExampleSelector(vectorstore=eg_store, k=4)
    prompt = FewShotPromptTemplate(
        prefix=_prefix,
        suffix=_suffix,
        example_selector=example_selector,
        example_prompt=_eg_prompt,
        input_variables=["question", "chat_history"],
    )
    llm = OpenAI(temperature=0, model_name="text-davinci-003")
    key_word_extractor = LLMChain(llm=llm, prompt=prompt)

    EXAMPLE_PROMPT = PromptTemplate(
        template=">Example:\nContent:\n---------\n{page_content}\n----------\nSource: {source}",
        input_variables=["page_content", "source"],
    )
    template = """You are an AI assistant for the open source library LangChain. The documentation is located at https://langchain.readthedocs.io.
You are given the following extracted parts of a long document and a question. Provide a conversational answer with a hyperlink to the documentation.
You should only use hyperlinks that are explicitly listed as a source in the context. Do NOT make up a hyperlink that is not listed.
If the question includes a request for code, provide a code block directly from the documentation.
If you don't know the answer, just say "Hmm, I'm not sure." Don't try to make up an answer.
If the question is not about LangChain, politely inform them that you are tuned to only answer questions about LangChain.
Question: {question}
=========
{context}
=========
Answer in Markdown:"""
    PROMPT = PromptTemplate(template=template, input_variables=["question", "context"])
    doc_chain = load_qa_chain(
        OpenAI(temperature=0, model_name="text-davinci-003", max_tokens=-1),
        chain_type="stuff",
        prompt=PROMPT,
        document_prompt=EXAMPLE_PROMPT,
    )
    return CustomChain(chain=doc_chain, vstore=vectorstore, key_word_extractor=key_word_extractor)


def _get_chat_history(chat_history: List[Tuple[str, str]]):
    buffer = ""
    for human_s, ai_s in chat_history:
        human = f"Human: " + human_s
        ai = f"Assistant: " + ai_s
        buffer += "\n" + "\n".join([human, ai])
    return buffer

```

```python ingest.py
"""Load html from files, clean up, split, ingest into Weaviate."""
import os
from pathlib import Path

import weaviate
from bs4 import BeautifulSoup
from langchain.text_splitter import CharacterTextSplitter


def clean_data(data):
    soup = BeautifulSoup(data)
    text = soup.find_all("main", {"id": "main-content"})[0].get_text()
    return "\n".join([t for t in text.split("\n") if t])


docs = []
metadatas = []
for p in Path("langchain.readthedocs.io/en/latest/").rglob("*"):
    if p.is_dir():
        continue
    with open(p) as f:
        docs.append(clean_data(f.read()))
        metadatas.append({"source": p})


text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
)

documents = text_splitter.create_documents(docs, metadatas=metadatas)


WEAVIATE_URL = os.environ["WEAVIATE_URL"]
client = weaviate.Client(
    url=WEAVIATE_URL,
    additional_headers={"X-OpenAI-Api-Key": os.environ["OPENAI_API_KEY"]},
)

client.schema.delete_class("Paragraph")
client.schema.get()
schema = {
    "classes": [
        {
            "class": "Paragraph",
            "description": "A written paragraph",
            "vectorizer": "text2vec-openai",
            "moduleConfig": {
                "text2vec-openai": {
                    "model": "ada",
                    "modelVersion": "002",
                    "type": "text",
                }
            },
            "properties": [
                {
                    "dataType": ["text"],
                    "description": "The content of the paragraph",
                    "moduleConfig": {
                        "text2vec-openai": {
                            "skip": False,
                            "vectorizePropertyName": False,
                        }
                    },
                    "name": "content",
                },
                {
                    "dataType": ["text"],
                    "description": "The link",
                    "moduleConfig": {
                        "text2vec-openai": {
                            "skip": True,
                            "vectorizePropertyName": False,
                        }
                    },
                    "name": "source",
                },
            ],
        },
    ]
}

client.schema.create(schema)

with client.batch as batch:
    for text in documents:
        batch.add_data_object(
            {"content": text.page_content, "source": str(text.metadata["source"])},
            "Paragraph",
        )

```

```sh ingest.sh
# Bash script to ingest data
# This involves scraping the data from the web and then cleaning up and putting in Weaviate.
!set -eu
wget -r -A.html https://langchain.readthedocs.io/en/latest/
python3 ingest.py
python3 ingest_examples.py

```

```python  ingest_examples.py
"""Ingest examples into Weaviate."""
import os
from pathlib import Path

import weaviate

WEAVIATE_URL = os.environ["WEAVIATE_URL"]
client = weaviate.Client(
    url=WEAVIATE_URL,
    additional_headers={"X-OpenAI-Api-Key": os.environ["OPENAI_API_KEY"]},
)

client.schema.delete_class("Rephrase")
client.schema.delete_class("QA")
client.schema.get()
schema = {
    "classes": [
        {
            "class": "Rephrase",
            "description": "Rephrase Examples",
            "vectorizer": "text2vec-openai",
            "moduleConfig": {
                "text2vec-openai": {
                    "model": "ada",
                    "modelVersion": "002",
                    "type": "text",
                }
            },
            "properties": [
                {
                    "dataType": ["text"],
                    "moduleConfig": {
                        "text2vec-openai": {
                            "skip": False,
                            "vectorizePropertyName": False,
                        }
                    },
                    "name": "content",
                },
                {
                    "dataType": ["text"],
                    "description": "The link",
                    "moduleConfig": {
                        "text2vec-openai": {
                            "skip": True,
                            "vectorizePropertyName": False,
                        }
                    },
                    "name": "question",
                },
                {
                    "dataType": ["text"],
                    "description": "The link",
                    "moduleConfig": {
                        "text2vec-openai": {
                            "skip": True,
                            "vectorizePropertyName": False,
                        }
                    },
                    "name": "answer",
                },
                {
                    "dataType": ["text"],
                    "description": "The link",
                    "moduleConfig": {
                        "text2vec-openai": {
                            "skip": True,
                            "vectorizePropertyName": False,
                        }
                    },
                    "name": "chat_history",
                },
            ],
        },
    ]
}

client.schema.create(schema)

documents = [
    {
        "question": "how do i load those?",
        "chat_history": "Human: What types of memory exist?\nAssistant: \n\nThere are a few different types of memory: Buffer, Summary, and Conversational Memory.",
        "answer": "How do I load Buffer, Summary, and Conversational Memory",
    },
    {
        "question": "how do i install this package?",
        "chat_history": "",
        "answer": "How do I install langchain?",
    },
    {
        "question": "how do I set serpapi_api_key?",
        "chat_history": "Human: can you write me a code snippet for that?\nAssistant: \n\nYes, you can create an Agent with a custom LLMChain in LangChain. Here is a [link](https://langchain.readthedocs.io/en/latest/modules/agents/examples/custom_agent.html) to the documentation that provides a code snippet for creating a custom Agent.",
        "answer": "How do I set the serpapi_api_key?",
    },
    {
        "question": "What are some methods for data augmented generation?",
        "chat_history": "Human: List all methods of an Agent class please\nAssistant: \n\nTo answer your question, you can find a list of all the methods of the Agent class in the [API reference documentation](https://langchain.readthedocs.io/en/latest/modules/agents/reference.html).",
        "answer": "What are some methods for data augmented generation?",
    },
    {
        "question": "can you write me a code snippet for that?",
        "chat_history": "Human: how do I create an agent with custom LLMChain?\nAssistant: \n\nTo create an Agent with a custom LLMChain in LangChain, you can use the [Custom Agent example](https://langchain.readthedocs.io/en/latest/modules/agents/examples/custom_agent.html). This example shows how to create a custom LLMChain and use an existing Agent class to parse the output. For more information on Agents and Tools, check out the [Key Concepts](https://langchain.readthedocs.io/en/latest/modules/agents/key_concepts.html) documentation.",
        "answer": "Can you provide a code snippet for creating an Agent with a custom LLMChain?",
    },
]
from langchain.prompts.example_selector.semantic_similarity import \
    sorted_values

for d in documents:
    d["content"] = " ".join(sorted_values(d))
with client.batch as batch:
    for text in documents:
        batch.add_data_object(
            text,
            "Rephrase",
        )

client.schema.get()
schema = {
    "classes": [
        {
            "class": "QA",
            "description": "Rephrase Examples",
            "vectorizer": "text2vec-openai",
            "moduleConfig": {
                "text2vec-openai": {
                    "model": "ada",
                    "modelVersion": "002",
                    "type": "text",
                }
            },
            "properties": [
                {
                    "dataType": ["text"],
                    "moduleConfig": {
                        "text2vec-openai": {
                            "skip": False,
                            "vectorizePropertyName": False,
                        }
                    },
                    "name": "content",
                },
                {
                    "dataType": ["text"],
                    "description": "The link",
                    "moduleConfig": {
                        "text2vec-openai": {
                            "skip": True,
                            "vectorizePropertyName": False,
                        }
                    },
                    "name": "question",
                },
                {
                    "dataType": ["text"],
                    "description": "The link",
                    "moduleConfig": {
                        "text2vec-openai": {
                            "skip": True,
                            "vectorizePropertyName": False,
                        }
                    },
                    "name": "answer",
                },
                {
                    "dataType": ["text"],
                    "description": "The link",
                    "moduleConfig": {
                        "text2vec-openai": {
                            "skip": True,
                            "vectorizePropertyName": False,
                        }
                    },
                    "name": "summaries",
                },
                {
                    "dataType": ["text"],
                    "description": "The link",
                    "moduleConfig": {
                        "text2vec-openai": {
                            "skip": True,
                            "vectorizePropertyName": False,
                        }
                    },
                    "name": "sources",
                },
            ],
        },
    ]
}

client.schema.create(schema)

documents = [
    {
        "question": "how do i install langchain?",
        "answer": "```pip install langchain```",
        "summaries": ">Example:\nContent:\n---------\nYou can pip install langchain package by running 'pip install langchain'\n----------\nSource: foo.html",
        "sources": "foo.html",
    },
    {
        "question": "how do i import an openai LLM?",
        "answer": "```from langchain.llm import OpenAI```",
        "summaries": ">Example:\nContent:\n---------\nyou can import the open ai wrapper (OpenAI) from the langchain.llm module\n----------\nSource: bar.html",
        "sources": "bar.html",
    },
]
from langchain.prompts.example_selector.semantic_similarity import \
    sorted_values

for d in documents:
    d["content"] = " ".join(sorted_values(d))
with client.batch as batch:
    for text in documents:
        batch.add_data_object(
            text,
            "QA",
        )

```


