<h1 align="center"> Retrieval-Augmented Generation with Gradio and Groq API Key</h1>
<p align="center"> Natural Language Processing Project</p>

<div align="center">

<img src="https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54">

</div>

### Name : Samuel Farrel Bagasputra
### Tech Stack : Python, Gradio, LangChain, HuggingFace Embedding, FAISS vector store

---

### 1. Analysis about how the project works
This Retrieval-Augmented Generation (RAG) application combines the capabilities of large language models (LLMs) with a vector-based document retrieval system to provide contextually grounded answers from user-supplied PDFs:

- **User Interface (Gradio):**  
  A friendly web UI allows users to upload a PDF and ask natural language questions about its content.

- **Document Loading & Splitting:**  
  `PyPDFLoader` reads the PDF, and `CharacterTextSplitter` breaks the text into chunks (1,000 characters with 200-character overlap) for efficient embedding and retrieval.

- **Embedding Generation:**  
  Each text chunk is transformed into a fixed-size vector using a HuggingFace embedding model (`all-MiniLM-L6-v2`), capturing semantic similarity.

- **Vector Store (FAISS):**  
  The chunk embeddings are indexed in FAISS for fast nearest-neighbor search. At query time, the user’s question is also embedded and used to retrieve the top-k (default k=4) most relevant chunks.

- **RetrievalQA Chain:**  
  Retrieved chunks are concatenated and passed alongside the user question to the ChatGroq LLM (e.g., `llama-3.3-70b-versatile`) to generate a precise answer, reducing model hallucination.


### 2. Analysis about how different every model works on Retrieval-Augmented Generation

```python
def get_llm():
    return ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="llama-3.3-70b-versatile", # Change the model in the code
        temperature=0.2
    )
```
- Model used : ```[llama-3.3-70b-versatile, deepseek-r1-distill-llama-70b, gemma2-9b-it]```

2.1 Analysis on ```llama-3.3-70b-versatile``` : 
- **Quality:** High-fidelity reasoning, deep contextual understanding, excels at multi-hop inference.  
- **Latency & Cost:** Larger model size leads to higher latency (~1.5–2× slower than 70B distilled) and increased compute expense.  
- **Use Cases:** Research-level QA, long-document summarization, complex analytical queries.

2.2 Analysis on ```deepseek-r1-distill-llama-70b``` : 
- **Quality:** Distilled version of 70B shows ~90–95% of full model performance on benchmarks, slightly less precise on long contexts.  
- **Latency & Cost:** ~30% faster inference, reduced memory footprint, more cost-effective for moderate-volume usage.  
- **Use Cases:** Real-time chatbots, interactive assistants, when response speed is critical.

2.3 Analysis on ```gemma2-9b-it``` : 
- **Quality:** A 9B-parameter model fine-tuned for instruction following. Performs well on straightforward queries but can miss nuances in long or complex texts.  
- **Latency & Cost:** Fastest inference and lowest cost among the three, suitable for large-scale deployments.  
- **Use Cases:** FAQ bots, internal knowledge base assistants, educational tutors for simple topics.

### 3. Analysis about how temperature works

```python
def get_llm():
    return ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="llama-3.3-70b-versatile",
        temperature=0.2 # Change the temperature value here and analzye
    )
```

3.1 Analysis on higher temperature 
- Behavior: Generates more varied and creative responses, useful for brainstorming or creative writing tasks.
- Trade-Off: Increased risk of hallucinations and reduced factual precision when chaining with retrieved context.

3.2 Analysis on lower temperature
- Behavior: Produces deterministic, focused answers closely aligned with retrieved context.
- Benefit: Ideal for technical Q&A, legal/medical information retrieval, and any scenario requiring high precision.

### 4. How to run the project

- Clone this repository with : 

```git
git clone https://github.com/arifian853/RAG_with_GroqAPI.git
```

- Copy the ```.env.example``` file and rename it to ```.env```

```
GROQ_API_KEY=your-groq-api-key
```

- Fill the ```GROQ_API_KEY``` with your Groq API Key, find it here : https://console.groq.com/keys

- Create a virtual enviroment
- Install Dependencies `pip install -r requirements.txt`
- Adjust the model you want to use
- Run the program
- Open `127.0.0.1:7860` in your browser
- Upload and process your pdf file
- Enter prompt, analyze, and do some experiments
