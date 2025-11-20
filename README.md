# Ex.No.6 Development of Python Code Compatible with Multiple AI Tools

# Date:
# Register no.
# Aim: Write and implement Python code that integrates with multiple AI tools to automate the task of interacting with APIs, comparing outputs, and generating actionable insights with Multiple AI Tools

# AI Tools Required:
## ⚙️ Required AI Tools
To run this project effectively, you’ll need access to some or all of the following AI tools:

## 1. OpenAI API (GPT, DALL·E, Whisper)
Use case: Text generation, summarization, conversation, embeddings, image generation, speech-to-text.
Integration: openai Python package.

## 2. Anthropic Claude API
Use case: Structured reasoning, summarization, safer text generation.
Integration: anthropic Python package.

## 3. Cohere API
Use case: Text classification, embeddings, clustering, semantic search.
Integration: cohere Python package.

## 4. Hugging Face Transformers
Use case: Open-source models for NLP, vision, and speech tasks.
Integration: transformers + datasets.

## 5. Google Generative AI (Gemini / PaLM)
Use case: Text processing, code generation, multimodal AI.
Integration: google-generativeai Python package.

## 6. LangChain
Use case: Orchestration framework for combining multiple AI tools.
Integration: langchain Python package.

## 7. Vector Databases (Optional)
Examples: Pinecone, Weaviate, ChromaDB.
Use case: Storing embeddings for semantic search and retrieval-augmented generation (RAG).

## 🚀 Getting Started
Follow these steps to set up and run the project:

### 1. Clone the Repository
```
git clone https://github.com/your-username/multi-ai-tools-automation.git
cd multi-ai-tools-automation
```
### 2. Install Dependencies
Install the required Python packages:
```
pip install openai cohere anthropic langchain transformers datasets google-generativeai
```

### 3. Set API Keys
Create a .env file in the project root and add your API keys:
```

OPENAI_API_KEY=your_openai_api_key
COHERE_API_KEY=your_cohere_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
GOOGLE_API_KEY=your_google_api_key
```

### 4. Run the Example
Run the sample workflow:

```
python main.py
```

### 5. Extend the Project
Add more AI tools by creating new query functions.
Integrate vector databases (Pinecone, Weaviate, ChromaDB) for retrieval tasks.
Enhance the logic layer to analyze and visualize insights.

# Explanation:
Experiment the persona pattern as a programmer for any specific applications related with your interesting area. 
Generate the outoput using more than one AI tool and based on the code generation analyse and discussing that. 

# Conclusion:

## Multi-AI Tools Integration & Comparison
This project demonstrates how to integrate multiple AI APIs, compare outputs, and generate actionable insights automatically. The provided Python script shows a practical way to call different providers, normalize their responses, evaluate similarities, and produce a final report.

## Features
Send the same prompt to multiple AI providers.
Normalize outputs into a common structure.
Compare results using similarity, latency, and token usage.
Generate insights and recommendations (fastest provider, most verbose, agreement/disagreement).
Save results in JSON for further use.


## Architecture
```

+-----------------+   prompt   +----------------+   +-----------------+
|   Client CLI    | --------> |  Controller    |-->| Provider A API  |
+-----------------+           |  (compare)     |   +-----------------+
                              |                |   +-----------------+
                              |                |-->| Provider B API  |
                              |                |   +-----------------+
                              +----------------+
                                      |
                                      v
                              +-----------------+
                              | Normalizer /    |
                              | Comparator      |
                              +-----------------+
                                      |
                                      v
                              +-----------------+
                              | Insights +      |
                              | Report (JSON)   |
                              +-----------------+


```


## Python Implementation (multi_ai_compare.py)

```
# multi_ai_compare.py
import os, time, json, argparse, requests
from dataclasses import dataclass, asdict
from typing import Dict, Any, List
from dotenv import load_dotenv
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
load_dotenv()


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
COHERE_API_KEY = os.getenv("COHERE_API_KEY", "")


@dataclass
class NormalizedResponse:
provider: str
text: str
token_count: int
latency_ms: float
metadata: Dict[str, Any]


def call_openai_like(prompt: str, model="gpt-4o", max_tokens=256) -> NormalizedResponse:
url = "https://api.openai.com/v1/chat/completions"
headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
payload = {"model": model, "messages":[{"role":"user","content":prompt}], "max_tokens":max_tokens}
start = time.time()
res = requests.post(url, headers=headers, json=payload, timeout=30)
latency = (time.time() - start) * 1000
data = res.json()
text = data["choices"][0]["message"]["content"].strip()
tokens = data.get("usage",{}).get("total_tokens",-1)
return NormalizedResponse("openai", text, tokens, latency, {"raw":data})


def call_cohere_like(prompt: str, model="xlarge", max_tokens=256) -> NormalizedResponse:
url = "https://api.cohere.ai/generate"
headers = {"Authorization": f"Bearer {COHERE_API_KEY}", "Content-Type": "application/json"}
payload = {"model":model,"prompt":prompt,"max_tokens":max_tokens}
start = time.time()
res = requests.post(url, headers=headers, json=payload, timeout=30)
latency = (time.time() - start) * 1000
data = res.json()
text = data["generations"][0]["text"].strip()
return NormalizedResponse("cohere", text, -1, latency, {"raw":data})


EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")


def compare_responses(responses: List[NormalizedResponse]) -> Dict[str,Any]:
texts = [r.text for r in responses]
embeddings = EMBED_MODEL.encode(texts, convert_to_numpy=True)
sim = cosine_similarity(embeddings)
pairwise = {}
for i in range(len(responses)):
for j in range(i+1,len(responses)):
pairwise[f"{responses[i].provider}vs{responses[j].provider}"]=float(sim[i,j])
best_latency = min(responses,key=lambda r:r.latency_ms).provider
return {"pairwise_similarity":pairwise,"best_latency_provider":best_latency}


def main():
parser = argparse.ArgumentParser()
parser.add_argument("--prompt",type=str,required=True)
args=parser.parse_args()
responses=[]
try: responses.append(call_openai_like(args.prompt))
except Exception as e: print("OpenAI failed",e)
try: responses.append(call_cohere_like(args.prompt))
except Exception as e: print("Cohere failed",e)
if not responses: return
comparison=compare_responses(responses)
report={"responses":[asdict(r) for r in responses],"comparison":comparison}
print(json.dumps(report,indent=2))


if name=="main":
main()

```

<img width="674" height="297" alt="image" src="https://github.com/user-attachments/assets/8f6b4b8a-4898-4220-89d9-cb1cfe2c775e" />


### Extending

1.Add more provider wrappers (Anthropic, Hugging Face Inference API, etc.).
2.Implement retries and error handling.
3.Store results in a database for analytics.
4.Visualize similarity and latency with charts.

### License
MIT License

# Result: The corresponding Prompt is executed successfully.
