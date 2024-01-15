{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "77a1fcc1",
   "metadata": {},
   "source": [
    "### Quick example to show RAG used for LLM to answer questions regarding your package/library, based on documentation.\n",
    "\n",
    "### This example answers questions regarding the syntax of [dataspace](https://pypi.org/project/dataspace/).\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "41beadae",
   "metadata": {},
   "outputs": [],
   "source": [
    "%pip install --upgrade langchain openai weaviate-client\n",
    "%pip install weaviate\n",
    "%pip install tiktoken"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "d5fd7564",
   "metadata": {},
   "outputs": [],
   "source": [
    "import getpass\n",
    "import os\n",
    "\n",
    "os.environ[\"OPENAI_API_KEY\"] = \"your openai api key\""
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "72dd9b4a",
   "metadata": {},
   "outputs": [],
   "source": [
    "import dotenv\n",
    "dotenv.load_dotenv()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "8326ae20",
   "metadata": {},
   "outputs": [],
   "source": [
    "import requests\n",
    "\n",
    "def get_json_urls(repo_owner, repo_name, path):\n",
    "    \"\"\"\n",
    "    Get the URLs of all JSON files in a specific path within a GitHub repository.\n",
    "    \n",
    "    :param repo_owner: GitHub username or organization name\n",
    "    :param repo_name: Repository name\n",
    "    :param path: Path within the repository\n",
    "    :return: List of URLs of JSON files\n",
    "    \"\"\"\n",
    "    api_url = f\"https://api.github.com/repos/{repo_owner}/{repo_name}/contents/{path}\"\n",
    "    response = requests.get(api_url)\n",
    "    urls = []\n",
    "\n",
    "    if response.status_code == 200:\n",
    "        contents = response.json()\n",
    "        for item in contents:\n",
    "            if item['type'] == 'file' and item['name'].endswith('.json'):\n",
    "                urls.append(item['download_url'])\n",
    "            elif item['type'] == 'dir':\n",
    "                urls.extend(get_json_urls(repo_owner, repo_name, item['path']))\n",
    "\n",
    "    return urls\n",
    "\n",
    "# Usage for dataspace python library (https://pypi.org/project/dataspace/)\n",
    "repo_owner = 'synw'\n",
    "repo_name = 'dataspace'\n",
    "path = 'docsite/public/doc/doc'\n",
    "json_urls = get_json_urls(repo_owner, repo_name, path)\n",
    "print(json_urls)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "0f9a012b",
   "metadata": {},
   "outputs": [],
   "source": [
    "import requests\n",
    "from langchain.document_loaders import TextLoader\n",
    "\n",
    "documents = []\n",
    "\n",
    "# Loop through each documentation URL\n",
    "for url in json_urls:\n",
    "    response = requests.get(url)\n",
    "    \n",
    "    # Check if the request was successful\n",
    "    if response.status_code == 200:\n",
    "        file_content = response.text\n",
    "        \n",
    "        # Save the file content to a temporary file\n",
    "        with open(\"temp.txt\", \"w\") as file:\n",
    "            file.write(file_content)\n",
    "        \n",
    "        # Load the document using TextLoader\n",
    "        loader = TextLoader(\"temp.txt\")\n",
    "        document = loader.load()\n",
    "\n",
    "        # Add the loaded document to the documents list\n",
    "        documents.extend(document)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "2772f82b",
   "metadata": {},
   "outputs": [],
   "source": [
    "from langchain.text_splitter import CharacterTextSplitter\n",
    "text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=50)\n",
    "chunks = text_splitter.split_documents(documents)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "d4549c1c",
   "metadata": {},
   "outputs": [],
   "source": [
    "from langchain.embeddings import OpenAIEmbeddings\n",
    "from langchain.vectorstores import Weaviate\n",
    "import weaviate\n",
    "from weaviate.embedded import EmbeddedOptions\n",
    "\n",
    "client = weaviate.Client(\n",
    "  embedded_options = EmbeddedOptions()\n",
    ")\n",
    "\n",
    "vectorstore = Weaviate.from_documents(\n",
    "    client = client,    \n",
    "    documents = chunks,\n",
    "    embedding = OpenAIEmbeddings(),\n",
    "    by_text = False\n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "8bb715df",
   "metadata": {},
   "outputs": [],
   "source": [
    "retriever = vectorstore.as_retriever()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "7a4e0fe6",
   "metadata": {},
   "outputs": [],
   "source": [
    "from langchain.prompts import ChatPromptTemplate\n",
    "\n",
    "template = \"\"\"\n",
    "Given provided context, answer the question.\n",
    "If the answer is still unknown, write an error.\n",
    "Question: {question} \n",
    "Context: {context} \n",
    "Answer:\n",
    "\"\"\"\n",
    "\n",
    "prompt = ChatPromptTemplate.from_template(template)\n",
    "\n",
    "print(prompt)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "afb945c8",
   "metadata": {},
   "outputs": [],
   "source": [
    "from langchain.chat_models import ChatOpenAI\n",
    "from langchain.schema.runnable import RunnablePassthrough\n",
    "from langchain.schema.output_parser import StrOutputParser\n",
    "\n",
    "llm = ChatOpenAI(model_name=\"gpt-3.5-turbo\", temperature=0)\n",
    "\n",
    "rag_chain = (\n",
    "    {\"context\": retriever,  \"question\": RunnablePassthrough()} \n",
    "    | prompt \n",
    "    | llm\n",
    "    | StrOutputParser() \n",
    ")\n",
    "\n",
    "query = \"What is the right syntax to add a column to my dataspace dataframe?\"\n",
    "rag_chain.invoke(query)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "60c83911",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}