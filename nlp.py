import spacy
def filter_substrings(noun_chunks):
    noun_chunks = list(set(noun_chunks))
    filtered_chunks = []
    for i, chunk in enumerate(noun_chunks):
        is_substring = False
        for j, other_chunk in enumerate(noun_chunks):
            if i != j and chunk in other_chunk.split():
                is_substring = True
                break
        if not is_substring and not chunk in ['a', 'an', 'the', 'background']:
            filtered_chunks.append(chunk)
    return filtered_chunks

def remove_articles(noun_chunks):
    clean_chunks = []
    for chunk in noun_chunks:
        if chunk.lower().startswith('the '):
            clean_chunks.append(chunk[4:])
        elif chunk.lower().startswith('a '):
            clean_chunks.append(chunk[2:])
        elif chunk.lower().startswith('an '):
            clean_chunks.append(chunk[3:])
        else:
            clean_chunks.append(chunk)
    return clean_chunks

def remove_repeated_words(cap):
    chunk_text = cap.split()
    chunk_text_no_repeats = [word for i, word in enumerate(chunk_text) if (word != chunk_text[i-1] and i>0) or i==0]
    chunk = ' '.join(chunk_text_no_repeats)
    return chunk
def load_spacy(model = "en_core_web_trf"):
    spacy_model = spacy.load(model)
    return spacy_model
def get_noun_chunks(captions, spacy_model):
    all_chunks = []
    for cap in captions:
        cap = remove_repeated_words(cap)
        doc = spacy_model(cap)
        chunks = [str(chunk) for chunk in doc.noun_chunks]
        #chunks = remove_repeated_words(chunks)
        chunks = remove_articles(chunks)
        all_chunks += chunks
    all_chunks = filter_substrings(all_chunks)

    return all_chunks

