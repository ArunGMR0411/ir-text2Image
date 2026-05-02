"""
Clean transcript_text and raw_text columns for BM25 indexing and BGE embedding.

Cleaning steps (in order):
1. Lowercase all characters
2. Remove all punctuation
3. Tokenise using nltk.word_tokenize()
4. Remove English stopwords
5. Rejoin tokens into cleaned string
"""

import csv
import string
import os
import multiprocessing as mp
from functools import partial

import nltk

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

STOPWORDS = set(stopwords.words('english'))


def clean_text(text: str) -> str:
    """Apply cleaning pipeline to a single text string."""
    if not text or not isinstance(text, str):
        return ""
    
    text = text.lower()
    
    # strip punctuation before tokenising so it doesn't attach to words
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    tokens = word_tokenize(text)
    
    # remove stopwords — they add noise without helping BM25 or BGE
    tokens = [t for t in tokens if t not in STOPWORDS]
    
    return ' '.join(tokens)


def clean_row(row: dict) -> dict:
    """Clean both transcript_text and raw_text columns in a row."""
    row = row.copy()
    row['transcript_text'] = clean_text(row.get('transcript_text', ''))
    row['raw_text'] = clean_text(row.get('raw_text', ''))
    return row


def main():
    base_path = '/home/arun-gmr/MSC_Projects/MOS'
    input_csv = os.path.join(base_path, 'index/retrieval/transcript_aligned.csv')
    output_csv = os.path.join(base_path, 'index/indexing/transcript_cleaned.csv')
    
    rows = []
    with open(input_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    # smoke-test the cleaning pipeline on a small sample before the full run
    test_rows = rows[:100]
    
    num_workers = os.cpu_count() or 4
    with mp.Pool(num_workers) as pool:
        test_cleaned = pool.map(clean_row, test_rows)
    
    for i in range(5):
        orig = test_rows[i].get('transcript_text', '')[:100]
        cleaned = test_cleaned[i].get('transcript_text', '')[:100]
    
    # process the full dataset in parallel
    with mp.Pool(num_workers) as pool:
        cleaned_rows = pool.map(clean_row, rows)
    
    # count rows where cleaning produced an empty string
    empty_count = sum(1 for r in cleaned_rows if not r.get('transcript_text', '').strip())
    
    assert len(cleaned_rows) == len(rows), f"Row count mismatch! {len(cleaned_rows)} vs {len(rows)}"
    
    total_words = sum(len(r['transcript_text'].split()) for r in cleaned_rows if r['transcript_text'])
    avg_words_after = total_words / len(cleaned_rows)
    
    fieldnames = ['frame_id', 'day', 'stream_name', 'hour', 'frame_index', 
                  'time_offset_sec', 'transcript_text', 'raw_text']
    
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(cleaned_rows)
    
    for row in cleaned_rows[:5]:
        pass


if __name__ == "__main__":
    main()
