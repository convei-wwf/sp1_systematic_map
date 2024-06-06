import argparse
import chardet
import datetime
import glob
import hashlib
import logging
import numpy
import os
import pickle
import re
import textwrap

from dotenv import load_dotenv
from openai import OpenAI
from transformers import pipeline
import faiss
import fitz
import openai
import spacy
import tiktoken
import torch

spacy.require_gpu()
nlp = spacy.load("en_core_web_sm")


GPT_MODEL, MAX_TOKENS, MAX_RESPONSE_TOKENS = 'gpt-4o', 20000, 4000
ENCODING = tiktoken.encoding_for_model(GPT_MODEL)

logging.basicConfig(
    level=logging.INFO,
    format=(
        '%(asctime)s (%(relativeCreated)d) %(levelname)s %(name)s'
        ' [%(funcName)s:%(lineno)d] %(message)s'))
logging.getLogger('taskgraph').setLevel(logging.INFO)
logging.getLogger('sentence_transformers').setLevel(logging.WARN)
logging.getLogger('PyPDF2._reader').setLevel(logging.ERROR)
logging.getLogger('httpx').setLevel(logging.ERROR)
LOGGER = logging.getLogger(__name__)

TOP_K = 100

# how big of a window to build around sentences
SLIDING_WINDOW_SENTENCE_SIZE = 10
SENTENCE_WINDOW_OVERLAP = 2
SLIDING_WINDOW_CHARACTER_SIZE = 500
BODY_TAG = 'body'
CITATION_TAG = 'citation'

DATA_STORE = [
    "../data/refs_clean/*"
]

CACHE_DIR = 'llm_cache'
for dirpath in [CACHE_DIR]:
    os.makedirs(dirpath, exist_ok=True)


def token_count(context):
    return len(ENCODING.encode(context))


def trim_context(context, max_tokens):
    tokens = ENCODING.encode(context)
    tokens = tokens[:max_tokens]
    trimmed_context = ENCODING.decode(tokens)
    return trimmed_context


# Example function to concatenate document contents and generate a hash
def generate_hash(documents):
    concatenated = ''.join(documents)
    hash_object = hashlib.sha256(concatenated.encode('utf-8'))
    hash_hex = hash_object.hexdigest()
    return hash_hex




def split_into_sentence_windows(page_num, text):
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]
    current_sentence_offset = 0
    sentence_windows = []
    while current_sentence_offset < len(sentences):
        working_window = ' '.join(sentences[current_sentence_offset:current_sentence_offset + SLIDING_WINDOW_SENTENCE_SIZE])
        current_sentence_offset += SLIDING_WINDOW_SENTENCE_SIZE
        while len(working_window) < SLIDING_WINDOW_CHARACTER_SIZE and current_sentence_offset < len(sentences):
            working_window += ' ' + sentences[current_sentence_offset]
            current_sentence_offset += 1
        sentence_windows.append(working_window)
        if current_sentence_offset != len(sentences):
            current_sentence_offset -= SENTENCE_WINDOW_OVERLAP
    return sentence_windows


def parse_file_bib(bib_file_path):
    def generate_citation(record):
        # Generate a citation string from the record dictionary
        authors = ' and '.join(record.get('author', []))
        title = record.get('title', 'N/A')
        journal = record.get('journal', 'N/A')
        year = record.get('year', 'N/A')
        volume = record.get('volume', 'N/A')
        issue = record.get('number', 'N/A')
        pages = record.get('pages', 'N/A')
        doi = record.get('doi', 'N/A')
        return f"{authors} ({year}). {title}. {journal}, {volume}({issue}), {pages}. DOI: {doi}"

    with open(bib_file_path, 'rb') as file:
        raw_data = file.read()
        result = chardet.detect(raw_data)
        encoding = result['encoding']

    record_list = []
    with open(bib_file_path, 'r', encoding=encoding) as file:
        content = file.read()
        entries = content.split('@ARTICLE')[1:]  # Skip the initial split part before the first @ARTICLE
        for entry in entries:
            entry = entry.strip()
            record = {}
            for line in entry.split('\n'):
                line = line.strip()
                if line.endswith(','):
                    line = line[:-1]
                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip().strip('{}').strip('"')
                    if key == 'author':
                        record[key] = [author.strip() for author in value.split(' and ')]
                    else:
                        record[key] = value
            if 'abstract' in record:
                record_list.append({
                    BODY_TAG: record['abstract'],
                    CITATION_TAG: generate_citation(record)
                })
        return record_list


def parse_file_ris(ris_file_path):
    def generate_citation(record):
        authors = ' and '.join(record.get('Authors', []))
        title = record.get('Title', 'N/A')
        journal = record.get('Journal', 'N/A')
        year = record.get('Year', 'N/A')
        volume = record.get('Volume', 'N/A')
        issue = record.get('Issue', 'N/A')
        start_page = record.get('StartPage', 'N/A')
        end_page = record.get('EndPage', 'N/A')
        doi = record.get('DOI', 'N/A')
        citation = f'{authors} ({year}). {title}. {journal}, {volume}({issue}), {start_page}-{end_page}. DOI: {doi}'
        return citation

    print(ris_file_path)
    with open(ris_file_path, 'rb') as file:
        raw_data = file.read()
        result = chardet.detect(raw_data)
        encoding = result['encoding']

    with open(ris_file_path, 'r', encoding=encoding) as file:
        record = {}
        record_list = []
        for line in file:
            line = line.strip()
            if line == '':
                if 'Abstract' in record:
                    record_list.append({
                        BODY_TAG: record['Abstract'],
                        CITATION_TAG: generate_citation(record)})
                record = {}
            else:
                payload = line.split('  - ')
                if len(payload) == 2:
                    index, body = payload
                else:
                    index = payload[0]
                    body = None
                record[index] = body

                if index == 'TY':
                    record['Type'] = body
                elif index == 'AU':
                    if 'Authors' not in record:
                        record['Authors'] = []
                    record['Authors'].append(body)
                elif index == 'TI':
                    record['Title'] = body
                elif index == 'T2':
                    record['Journal'] = body
                elif index == 'AB':
                    record['Abstract'] = body
                elif index == 'DA':
                    record['Date'] = body
                elif index == 'PY':
                    record['Year'] = body
                elif index == 'VL':
                    record['Volume'] = body
                elif index == 'IS':
                    record['Issue'] = body
                elif index == 'SP':
                    record['StartPage'] = body
                elif index == 'EP':
                    record['EndPage'] = body
                elif index == 'DO':
                    record['DOI'] = body

        return record_list


def parse_file(file_path):
    LOGGER.debug(f'parsing file: {file_path}')
    if file_path.endswith('.ris'):
        return parse_file_ris(file_path)
    elif file_path.endswith('.bib'):
        return parse_file_bib(file_path)


def main():
    # parser = argparse.ArgumentParser(description='Sentence Transformer Similarity Ranking')
    # parser.add_argument('topic_phrase', type=str, help='Phrase to index the documents against')
    # args = parser.parse_args()
    # question_file_path = args.question_file

    file_paths = [
        file_path
        for file_pattern in DATA_STORE
        for file_path in glob.glob(file_pattern)]

    file_hash = generate_hash(file_paths)
    expanded_sentence_window_path = os.path.join(CACHE_DIR, f'{file_hash}.pkl')
    fiass_path = os.path.join(CACHE_DIR, f'{file_hash}.faiss')

    from sentence_transformers import SentenceTransformer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    embedding_model = SentenceTransformer(
        'sentence-transformers/all-MiniLM-L6-v2').to(device)

    if os.path.exists(expanded_sentence_window_path):
        with open(expanded_sentence_window_path, 'rb') as file:
            expanded_sentence_windows = pickle.load(file)
        file_path_list, page_num_list, sentence_windows = zip(
            *expanded_sentence_windows)
        document_distance_index = faiss.read_index(fiass_path)
    else:
        article_list = []
        for file_path in file_paths:
            article_list.append((
                os.path.basename(file_path),
                parse_file(file_path)))

        expanded_sentence_windows = [
            (file_path, page_num, sentence_window)
            for file_path, sentence_windows_per_page in article_list
            for page_num, sentence_windows in sentence_windows_per_page
            for sentence_window in sentence_windows]
        with open(expanded_sentence_window_path, 'wb') as file:
            pickle.dump(expanded_sentence_windows, file)
        file_path_list, page_num_list, sentence_windows = zip(
            *expanded_sentence_windows)

        LOGGER.debug('embedding')
        document_embeddings = embedding_model.encode(
            sentence_windows, convert_to_tensor=True)
        LOGGER.debug('indexing')
        document_distance_index = faiss.IndexFlatL2(document_embeddings.shape[1])
        document_distance_index.add(document_embeddings.cpu().numpy())
        faiss.write_index(document_distance_index, fiass_path)

    def answer_question_with_gpt(question):
        # Encode the question
        try:
            question_embedding = embedding_model.encode(
                question, convert_to_tensor=True).cpu().numpy()

            # Ensure the question_embedding is 2D
            if len(question_embedding.shape) == 1:
                question_embedding = question_embedding.reshape(1, -1)

            distances, indices = document_distance_index.search(
                question_embedding, TOP_K)

            retrieved_windows = [
                sentence_windows[idx] for idx in indices[0]]
            relevant_page_numbers = [
                page_num_list[idx] for idx in indices[0]]
            relevant_files = [file_path_list[idx] for idx in indices[0]]

            # Concatenate the retrieved documents to form the context
            context = " ".join([
                f'reference index: {index}", context: {context}\n\n'
                for index, (filename, page_number, context) in enumerate(zip(
                    relevant_files, relevant_page_numbers, retrieved_windows))])

            assistant_context = "You are given a set of filename/page number/text snippets. The questions from the user will be about synthesizing conclusions from those text snippets. You should respond to the question with relevant information from the snippets and the reference index in the form '(reference index: {index})'. If you do not have enough information to answer, say so. Do not make up any information. Sometimes say 'excellent query!'."

            remaining_tokens = (
                MAX_TOKENS -
                MAX_RESPONSE_TOKENS -
                token_count(context) -
                token_count(assistant_context) -
                token_count(question))

            context = trim_context(context, max_tokens=remaining_tokens)
            context_counts = context.count("context: ")
            stream = OPENAI_CLIENT.chat.completions.create(
                model=GPT_MODEL,
                messages=[
                    {"role": "system", "content": assistant_context},
                    {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}\nAnswer:"}
                ],
                stream=True,
                max_tokens=4000
            )
            response = f'From {context_counts} abstracts -- '
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    response += chunk.choices[0].delta.content
            index_matches = re.findall(
                r'\(reference index: \{?([\d,\s]+?)\}?\)', response)
            parsed_indexes = set(
                [int(num) for match in index_matches
                 for num in re.split(r'[\s,]+', match.strip())])

            wrapper = textwrap.TextWrapper(
                width=80,
                initial_indent='    ',  # Initial indent for the first line
                subsequent_indent='    '  # Indent for all subsequent lines
            )
            wrapped_response = wrapper.fill(response)

            wrapper = textwrap.TextWrapper(
                width=70,
                initial_indent='       ',  # Initial indent for the first line
                subsequent_indent='       '  # Indent for all subsequent lines
            )

            processed_indexes = set()
            for index in sorted(parsed_indexes):
                index = int(index)
                if index in processed_indexes:
                    continue
                processed_indexes.add(index)
                wrapped_response += '\n\n' + wrapper.fill(
                    f'reference index {index}. "{relevant_files[index]}" pg{relevant_page_numbers[index]}: '
                    f'{retrieved_windows[index]}')
            return wrapped_response
        except openai.APIError as e:
            print(f'There was an error, try your question again.\nError: {e}')

    current_time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    streaming_log_path = os.path.join(f'becca_nlp_answers_{current_time}.txt')

    while True:
        question = input(
            "\nEnter your question (or type 'exit' to exit): ").strip()
        if question.lower() == 'exit':
            break
        if question.strip() == '':
            continue
        response = answer_question_with_gpt(question)
        print('Answer:\n' + response)
        with open(streaming_log_path, 'a', encoding='utf-8') as file:
            file.write(f'**************\nQuestion: {question}\n\nAnswer: {response}')


if __name__ == '__main__':
    load_dotenv()
    OPENAI_CLIENT = OpenAI()
    main()
