import chardet
import datetime
import glob
import hashlib
import logging
import os
import pickle
import re

import faiss
import spacy
import tiktoken
import torch

# spacy.require_gpu()
nlp = spacy.load("en_core_web_sm")


GPT_MODEL, MAX_TOKENS, MAX_RESPONSE_TOKENS = 'gpt-4o', 20000, 4000
ENCODING = tiktoken.encoding_for_model(GPT_MODEL)

logging.basicConfig(
    level=logging.INFO,
    format=(
        '%(asctime)s (%(relativeCreated)d) %(levelname)s %(name)s'
        ' [%(funcName)s:%(lineno)d] %(message)s'))
logging.getLogger('sentence_transformers').setLevel(logging.WARN)
LOGGER = logging.getLogger(__name__)

BODY_TAG = 'body'
CITATION_TAG = 'cite_key'

REPO_DIR = 'Documents/github/sp1_systematic_map'
DATA_STORE = [
    os.path.join(REPO_DIR, "_data/3_refs_clean/*.ris")
]

CACHE_DIR = os.path.join(REPO_DIR, 'llm_cache')
for dirpath in [CACHE_DIR]:
    os.makedirs(dirpath, exist_ok=True)


def generate_hash(documents):
    concatenated = ''.join(documents)
    hash_object = hashlib.sha256(concatenated.encode('utf-8'))
    hash_hex = hash_object.hexdigest()
    return hash_hex


def parse_file_bib(bib_file_path):
    def generate_citation(record):
        authors = ' and '.join(record.get('author', []))
        title = record.get('title', 'N/A')
        journal = record.get('journal', 'N/A')
        year = record.get('year', 'N/A')
        # volume = record.get('volume', 'N/A')
        # issue = record.get('number', 'N/A')
        # pages = record.get('pages', 'N/A')
        doi = record.get('doi', 'N/A')
        return f"{authors} | {year} | {title} | {journal} | DOI: {doi}"

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
        # volume = record.get('Volume', 'N/A')
        # issue = record.get('Issue', 'N/A')
        # start_page = record.get('StartPage', 'N/A')
        # end_page = record.get('EndPage', 'N/A')
        doi = record.get('DOI', 'N/A')
        # citation = f"{authors} | {year} | {title} | {journal} | DOI: {doi}"
        citation = record.get('Notes', 'N/A')
        return citation

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
                elif index == 'N1':
                    record['Notes'] = body
                # elif index == 'VL':
                #     record['Volume'] = body
                # elif index == 'IS':
                #     record['Issue'] = body
                # elif index == 'SP':
                #     record['StartPage'] = body
                # elif index == 'EP':
                #     record['EndPage'] = body
                elif index == 'DO':
                    record['DOI'] = body

        return record_list


def parse_file(file_path):
    LOGGER.info(f'parsing file: {file_path}')
    if file_path.endswith('.ris'):
        return parse_file_ris(file_path)
    elif file_path.endswith('.bib'):
        return parse_file_bib(file_path)

def main():
    file_paths = [
        file_path
        for file_pattern in DATA_STORE

        for file_path in glob.glob(file_pattern)]
    file_hash = generate_hash(file_paths)
    parsed_articles_path = os.path.join(CACHE_DIR, f'{file_hash}.pkl')
    faiss_path = os.path.join(CACHE_DIR, f'{file_hash}.faiss')

    from sentence_transformers import SentenceTransformer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    embedding_model = SentenceTransformer(
        'sentence-transformers/all-MiniLM-L6-v2').to(device)

    if os.path.exists(parsed_articles_path):
        with open(parsed_articles_path, 'rb') as file:
            (abstract_list, citation_list) = pickle.load(file)
        document_distance_index = faiss.read_index(faiss_path)
    else:
        article_list = []
        LOGGER.info(f'about to process {file_paths}')
        for file_path in file_paths:
            LOGGER.info(f'processing single file {file_path}')
            article_list.extend(parse_file(file_path))
        abstract_list, citation_list = zip(
            *[(article[BODY_TAG], article[CITATION_TAG])
              for article in article_list])
        with open(parsed_articles_path, 'wb') as file:
            pickle.dump((abstract_list, citation_list), file)

        LOGGER.debug('embedding')
        document_embeddings = embedding_model.encode(
            abstract_list, convert_to_tensor=True)
        LOGGER.debug('indexing')
        document_distance_index = faiss.IndexFlatL2(document_embeddings.shape[1])
        document_distance_index.add(document_embeddings.cpu().numpy())
        faiss.write_index(document_distance_index, faiss_path)

    def find_relevant_articles(start_index, stop_index, question):
        question_embedding = embedding_model.encode(
            question, convert_to_tensor=True).cpu().numpy()

        # Ensure the question_embedding is 2D
        if len(question_embedding.shape) == 1:
            question_embedding = question_embedding.reshape(1, -1)

        distances, indices = document_distance_index.search(
            question_embedding, stop_index + 1)

        retrieved_citations = [
            citation_list[idx] for idx in
            indices[:, start_index:stop_index + 1].flatten()]
        retrieved_abstracts = [
            abstract_list[idx] for idx in
            indices[:, start_index:stop_index + 1].flatten()]

        # Concatenate the retrieved documents to form the context
        context = "".join([
            f'reference index: {index+start_index}, distance (smaller is better): {distance}\n\t{citation}\n\n{abstract}\n\n'
            for index, (distance, citation, abstract) in enumerate(zip(
                distances[0], retrieved_citations, retrieved_abstracts))])
        return context

    current_time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    streaming_log_path = os.path.join(f'becca_nlp_answers_{current_time}.txt')

    pattern = re.compile(r'(\d+)\s+(\d+)\s+(.*)')
    while True:
        question = input(
            "\nEnter your [start index] [stop index] [question] (or type 'exit' to exit): ").strip()
        if question.lower() == 'exit':
            break
        if question.strip() == '':
            continue
        match = pattern.match(question)
        if match:
            start_index, stop_index, question = match.groups()
            response = find_relevant_articles(int(start_index), int(stop_index), question)
            print('Answer:\n' + response)
            with open(streaming_log_path, 'a', encoding='utf-8') as file:
                file.write(f'**************\nQuestion: {question}\n\nAnswer: {response}')
        else:
            print(f'Could not match a [start index] [stop index] [question] pattern from "{question}"')


if __name__ == '__main__':
    main()
