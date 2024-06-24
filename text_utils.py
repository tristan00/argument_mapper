import re
import re
from typing import List
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer


lemmatizer = WordNetLemmatizer()

from bs4 import BeautifulSoup


def lemmatize_docs(docs: List[str]):
    docs = [[lemmatizer.lemmatize(token) for token in doc] for doc in docs]
    return docs


def tokenize_doc(input_str) -> List[str]:
    return word_tokenize(input_str)


def lemmatize_str(input_str) -> str:
    return " ".join(
        [lemmatizer.lemmatize(i) for i in tokenize_doc(str(input_str).lower())]
    )


def encode_decode(s, encoding):
    return s.encode(
        encoding, "replace"
    ).decode()


def text_preprocessing(text):
    modified_text = text.strip()
    modified_text = re.sub(r'\s*hide$', '', modified_text)
    modified_text = re.sub(r'\s*hide,$', '', modified_text)
    modified_text = re.sub(r'\s*HIDE$', '', modified_text)
    modified_text = re.sub(r'\s*HIDE,$', '', modified_text)

    modified_text = modified_text.replace('\n\n', '\n')
    modified_text = modified_text.replace(' .\n', '.\n')
    return modified_text


def transform_html_to_source(html_message):
    soup = BeautifulSoup(html_message, 'html.parser')

    p_tags = soup.find_all('p')
    result = []
    for p in p_tags:
        # Optionally, remove or handle non-relevant tags
        for non_relevant in p.find_all(['span', 'div']):  # Add more tags if needed
            non_relevant.decompose()  # This removes the tag from the tree

        for a in p.find_all('a'):
            href = a.get('href', '')
            text = a.get_text(strip=True)
            if text:  # Only add the markdown link if there is text
                markdown_link = f"[{text}]({href})"
                a.replace_with(markdown_link)
            else:
                a.decompose()  # Remove <a> tags that don't contribute to visible text

        # Find and replace <em> tags with Markdown italics
        for em in p.find_all('em'):
            em_text = em.get_text(strip=True)
            markdown_em = f"*{em_text}*"
            em.replace_with(markdown_em)

        inner_text = p.get_text(" ", strip=True)
        result.append(inner_text)

    transformed_text = '\n'.join(result).strip()

    transformed_text += '\nhide'

    return transformed_text