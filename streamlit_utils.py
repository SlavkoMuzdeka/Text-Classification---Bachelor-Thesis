import torch, streamlit as st
import torch.nn.functional as F

from textblob import TextBlob


def split_text_into_sentences(text):
    blob = TextBlob(text)
    sentences = [str(sentence) for sentence in blob.sentences]
    return sentences


def summarize_text(text, summarizer, max_length=512):
    summary = summarizer(text, max_length=max_length, min_length=50, do_sample=False)[
        0
    ]["summary_text"]
    return summary


def recursive_summarize_and_tokenize(text, tokenizer, summarizer):
    inputs = tokenizer(text, return_tensors="pt", truncation=False)
    input_length = inputs["input_ids"].shape[1]

    if input_length <= 512:
        return inputs

    if input_length <= 1024:
        with st.spinner("Summarizing text..."):
            summarized_text = summarize_text(text, summarizer)
            return recursive_summarize_and_tokenize(
                summarized_text, tokenizer, summarizer
            )

    with st.spinner("Summarizing text in parts..."):
        sentences = split_text_into_sentences(text)
        text_parts = []
        current_chunk = ""

        for sentence in sentences:
            if (
                len(
                    tokenizer(
                        current_chunk + " " + sentence,
                        return_tensors="pt",
                        truncation=False,
                    )["input_ids"][0]
                )
                <= 512
            ):
                current_chunk += " " + sentence
            else:
                text_parts.append(current_chunk.strip())
                current_chunk = sentence

        if current_chunk:
            text_parts.append(current_chunk.strip())

        summarized_parts = [summarize_text(part, summarizer) for part in text_parts]
        combined_summary = " ".join(summarized_parts)

        return recursive_summarize_and_tokenize(combined_summary, tokenizer, summarizer)


def tokenize_text(tokenizer, summarizer, text):
    return recursive_summarize_and_tokenize(text, tokenizer, summarizer)


def classify_text(tokenizer, model, summarizer, text):
    inputs = tokenize_text(tokenizer, summarizer, text)

    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    probabilities = F.softmax(logits, dim=-1)
    return probabilities
