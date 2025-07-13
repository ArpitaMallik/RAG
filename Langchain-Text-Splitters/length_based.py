from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("Reasoning with large language models for medical.pdf")

docs = loader.load()


# text = """Recommendation: Subtask 2 – Image Captioning Models Evaluation
# This is the better fit for you because:

# 🎯 Why it suits your LLM interests:
# You can fine-tune a multimodal LLM (like Qwen 2.5-VL 7B) on the Arabic image captioning dataset.

# You’ll be working with zero-shot and few-shot prompting, or training from scratch — both are LLM techniques.

# Evaluation is done using GPT-4o as a judge, so you'll get to work with evaluation aligned with human-level understanding.

# You’ll gain experience in multimodal LLMs (vision + text), which is highly valuable and trending.

# 💡 What You Can Do in Subtask 2:
# Use existing Colab notebooks to understand baselines.

# Apply prompt engineering, few-shot examples, or fine-tune Qwen 2.5-VL 7B or similar models.

# Explore how semantic alignment between image and caption can be improved via better LLM reasoning.

# Optionally, test with other multilingual models (e.g., mPLUG-Owl2, Gemini, or LLaVA-Mistral if permitted).

# Bonus Tip:
# If you still want to explore both subtasks, you could try Subtask 1 as a warm-up — it gives you insight into what makes a “good caption” culturally — which helps in making better LLM-generated captions in Subtask 2."""

splitter = CharacterTextSplitter(
    chunk_size = 100,
    chunk_overlap = 0,
    separator = ""
)

# result = splitter.split_text(text)
result = splitter.split_documents(docs)
print(result[0].page_content)