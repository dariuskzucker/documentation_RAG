### RAG used for LLM to answer questions regarding your package/library, based on documentation.

### This LLM will now effectively answer questions regarding the syntax of [dataspace](https://pypi.org/project/dataspace/).



```python
%pip install --upgrade langchain openai weaviate-client
%pip install weaviate
%pip install tiktoken
```


```python
import getpass
import os

os.environ["OPENAI_API_KEY"] = "your api key here"
```


```python
import dotenv
dotenv.load_dotenv()
```


```python
import requests

def get_json_urls(repo_owner, repo_name, path):
    """
    Get the URLs of all JSON files in a specific path within a GitHub repository.
    
    :param repo_owner: GitHub username or organization name
    :param repo_name: Repository name
    :param path: Path within the repository
    :return: List of URLs of JSON files
    """
    api_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/contents/{path}"
    response = requests.get(api_url)
    urls = []

    if response.status_code == 200:
        contents = response.json()
        for item in contents:
            if item['type'] == 'file' and item['name'].endswith('.json'):
                urls.append(item['download_url'])
            elif item['type'] == 'dir':
                urls.extend(get_json_urls(repo_owner, repo_name, item['path']))

    return urls

# Usage for dataspace python library (https://pypi.org/project/dataspace/)
repo_owner = 'synw'
repo_name = 'dataspace'
path = 'docsite/public/doc/doc'
json_urls = get_json_urls(repo_owner, repo_name, path)
print(json_urls)
```


```python
import requests
from langchain.document_loaders import TextLoader

documents = []

# Loop through each documentation URL
for url in json_urls:
    response = requests.get(url)
    
    # Check if the request was successful
    if response.status_code == 200:
        file_content = response.text
        
        # Save the file content to a temporary file
        with open("temp.txt", "w") as file:
            file.write(file_content)
        
        # Load the document using TextLoader
        loader = TextLoader("temp.txt")
        document = loader.load()

        # Add the loaded document to the documents list
        documents.extend(document)
```


```python
from langchain.text_splitter import CharacterTextSplitter
text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=50)
chunks = text_splitter.split_documents(documents)
```


```python
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Weaviate
import weaviate
from weaviate.embedded import EmbeddedOptions

client = weaviate.Client(
  embedded_options = EmbeddedOptions()
)

vectorstore = Weaviate.from_documents(
    client = client,    
    documents = chunks,
    embedding = OpenAIEmbeddings(),
    by_text = False
)
```


```python
retriever = vectorstore.as_retriever()

```


```python
from langchain.prompts import ChatPromptTemplate

template = """
Given provided context, answer the question.
If the answer is still unknown, write an error.
Question: {question} 
Context: {context} 
Answer:
"""

prompt = ChatPromptTemplate.from_template(template)

print(prompt)
```


```python
from langchain.chat_models import ChatOpenAI
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

rag_chain = (
    {"context": retriever,  "question": RunnablePassthrough()} 
    | prompt 
    | llm
    | StrOutputParser() 
)

query = "What is the right syntax to add a column to my dataspace dataframe?"
rag_chain.invoke(query)
```


```python

```
