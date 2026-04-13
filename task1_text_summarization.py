"""
TASK 1: TEXT SUMMARIZATION TOOL
Uses NLP techniques to summarize lengthy articles.
Libraries: transformers (HuggingFace), nltk, sumy
"""


import nltk
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from collections import defaultdict
import string

def extractive_summarize(text: str, num_sentences: int = 3) -> str:
    """Rank sentences by word frequency and pick top N."""
    sentences = sent_tokenize(text)
    if len(sentences) <= num_sentences:
        return text

    stop_words = set(stopwords.words('english'))
    word_freq: dict = defaultdict(int)

    for word in word_tokenize(text.lower()):
        if word not in stop_words and word not in string.punctuation:
            word_freq[word] += 1

    
    max_freq = max(word_freq.values(), default=1)
    word_freq = {w: f / max_freq for w, f in word_freq.items()}

    
    sentence_scores: dict = defaultdict(float)
    for sent in sentences:
        for word in word_tokenize(sent.lower()):
            if word in word_freq:
                sentence_scores[sent] += word_freq[word]

    
    top_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)[:num_sentences]
    summary = ' '.join([s for s in sentences if s in top_sentences])
    return summary



def abstractive_summarize(text: str, max_length: int = 130, min_length: int = 30) -> str:
    """Use facebook/bart-large-cnn for abstractive summarization."""
    try:
        from transformers import pipeline
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        
        result = summarizer(text[:3000], max_length=max_length,
                            min_length=min_length, do_sample=False)
        return result[0]['summary_text']
    except ImportError:
        return "❌  Install transformers & torch: pip install transformers torch"



SAMPLE_ARTICLE = """
Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to
natural intelligence displayed by animals including humans. AI research has been defined
as the field of study of intelligent agents, which refers to any system that perceives
its environment and takes actions that maximize its chance of achieving its goals.

The term "artificial intelligence" had previously been used to describe machines that
mimic and display human cognitive skills associated with the human mind, such as learning
and problem-solving. This definition has since been rejected by major AI researchers who
now describe AI in terms of rationality and acting rationally, which does not limit how
intelligence can be articulated.

AI applications include advanced web search engines (e.g., Google Search), recommendation
systems (used by YouTube, Amazon, and Netflix), understanding human speech (such as Siri
and Alexa), self-driving cars (e.g., Waymo), generative or creative tools (ChatGPT and
AI art), automated decision-making, and competing at the highest level in strategic game
systems (such as chess and Go).

As machines become increasingly capable, tasks considered to require intelligence are
often removed from the definition of AI, a phenomenon known as the AI effect. For
instance, optical character recognition is frequently excluded from things considered to
be AI, having become a routine technology. Modern machine capabilities generally classified
as AI include successfully understanding human speech, competing at the highest level in
strategic game systems, and also impersonating human capabilities.
"""

if __name__ == "__main__":
    print("=" * 60)
    print("         TASK 1 — TEXT SUMMARIZATION TOOL")
    print("=" * 60)

    print("\n📄  ORIGINAL TEXT  ({} words)".format(len(SAMPLE_ARTICLE.split())))
    print("-" * 60)
    print(SAMPLE_ARTICLE.strip())

    print("\n\n✂️   EXTRACTIVE SUMMARY (NLTK — 3 sentences)")
    print("-" * 60)
    ext_summary = extractive_summarize(SAMPLE_ARTICLE, num_sentences=3)
    print(ext_summary)

    print("\n\n🤖  ABSTRACTIVE SUMMARY (BART — requires download)")
    print("-" * 60)
    abs_summary = abstractive_summarize(SAMPLE_ARTICLE)
    print(abs_summary)

    print("\n" + "=" * 60)
    print("✅  Summarization complete!")
    print("=" * 60)
