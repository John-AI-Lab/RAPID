CONTEXT_STRUCTURE = '''<Retrieved Context>
{rag_context}
</Retrieved Context>

<Full Long Context>
{long_context}
</Full Long Context>

{query}
'''

RAG_PROMPT_TEMPLATE = '''<Retrieved Context>
{rag_context}
</Retrieved Context>'''

# RAG_PROMPT_TEMPLATE_WITH_FULL = '''<Retrieved Context>
# {rag_context}
# </Retrieved Context>

# <Full Long Context>
# {long_context}
# </Full Long Context>'''


LONG_CONTEXT_PROMPT_TEMPLATE = '''<Context>
{long_context}
</Context>'''


RAG_INSTRUCTION = '''Please answer the question based on the <Retrieved Context> provided above. \
If the context doesn't contain all necessary information, please make reasonable inferences based on the available information.'''

FULL_INSTRUCTION = '''Please answer the question based on the <Context> provided above.'''


# FULL_INSTRUCTION_WITH_RAG = '''Given the provided contexts above:
# 1. The <Retrieved Context> contains relevant excerpts selected from the full text specifically related to the question.
# 2. The <Full Long Context> contains the complete original full text.

# Please use both contexts to provide a precise answer:
# - First, focus on the key information from the retrieved context if it can directly address the question.
# - If the <Retrieved Context> is insufficient, supplement or validate this information using the full context.
# - Ensure your answer is accurate by cross-referencing both contexts.
# - If you notice any important details in the full context that were not included in the retrieved portions but are relevant to the question, please utilize them to solve the question.'''




def split_into_chunks(document, tokenizer, max_tokens=128):
    """
    Splits a document into chunks using Langchain's text splitter, 
    ensuring each chunk does not exceed the maximum number of tokens.
    
    Parameters:
    - document: str, the input text to be split.
    - tokenizer: a tokenizer object from the transformers library.
    - max_tokens: int, the maximum number of tokens per chunk.
    
    Returns:
    - chunks: list of str, each being a chunk of text that respects the token limit.
    """
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    
    # Initialize Langchain's RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer,
        separators=[
            # English separators
            "\n\n", "\n", ". ", "! ", "? ", "; ", ", ",
            # Chinese separators
            "。", "！", "？", "；", "，",
            # Space for English words
            " ",
            # Character level (last resort)
            ""
        ],
        chunk_size=max_tokens,    # Start with a reasonable chunk size
        chunk_overlap=0   # Slight overlap to maintain context
    )
    
    # Perform an initial split into chunks
    initial_chunks = text_splitter.split_text(document)
    chunks_with_positions = []
    current_position = 0
    for chunk in initial_chunks:
        # Find the start position of this chunk in the original text
        chunk_position = document.find(chunk, current_position)
        chunks_with_positions.append({
            'text': chunk,
            'position': chunk_position
        })
        current_position = chunk_position + 1
    
    return chunks_with_positions
    
    # return initial_chunks


import faiss
def retrieve_short_context(embed_model, tokenizer, chunks_with_positions, query, top_k=5, rag_threshold=0.3):
    # Split context into chunks
    chunks_text = [chunk['text'] for chunk in chunks_with_positions]
    # Encode the chunks using the model, utilizing the GPU
    chunk_embeddings = embed_model.encode(chunks_text, normalize_embeddings=True)

    # Move embeddings to CPU as FAISS operates on CPU tensors
    # chunk_embeddings = chunk_embeddings

    # Initialize FAISS on the GPU
    d = chunk_embeddings.shape[1]  # Dimension of embeddings
    index = faiss.IndexFlatIP(d)   # Inner product
    # gpu_index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, index)

    # Add embeddings to the FAISS index
    index.add(chunk_embeddings)

    # Define a retrieval function
    query_embedding = embed_model.encode([query], normalize_embeddings=True)
    # query_embedding = query_embedding.cpu().detach().numpy()

    distances, indices = index.search(query_embedding, top_k)
    # return [chunks_with_positions[i] for i in indices[0]]
    # Filter results based on similarity threshold
    filtered_results = []
    for dist, idx in zip(distances[0], indices[0]):
        if dist >= rag_threshold:  # Since we're using inner product with normalized vectors, 
                     # the similarity score will be between -1 and 1
            filtered_results.append(chunks_with_positions[idx])
    
    return filtered_results


from rank_bm25 import BM25Okapi
from langchain_community.retrievers import BM25Retriever
from nltk.tokenize import word_tokenize


def retrieve_short_context_bm25(tokenizer, chunks_with_positions, query, top_k=5):
    chunks_text = [chunk['text'] for chunk in chunks_with_positions]
    retriever = BM25Retriever.from_texts(chunks_text, preprocess_func=word_tokenize, k=top_k)
    result = retriever.invoke(query)
    # return [item.page_content for item in result]
    retrieved_chunks = []
    for item in result:
        for chunk in chunks_with_positions:
            if chunk['text'] == item.page_content:
                retrieved_chunks.append(chunk)
                break
    return retrieved_chunks


from sentence_transformers import SentenceTransformer
def get_rag_context(long_context, query, embed_model_path, tokenizer, num_to_retrieve, max_chunk_token=128, rag_threshold=0.1):
    '''
    Parameters:
    - long_context: str, the long context to retrieve short contexts from.
    - query: str, the query to retrieve short contexts for.
    - embed_model_path: str, the path to the RAG model.
    - tokenizer: the tokenizer of LLM.
    - num_to_retrieve: int, the number of short context chunks to retrieve.
    - max_chunk_token: int, the maximum number of tokens per chunk.
    '''
    chunks = split_into_chunks(long_context, tokenizer, max_tokens=max_chunk_token)
    embed_model = SentenceTransformer(embed_model_path)
    model_retrieved_chunks = retrieve_short_context(embed_model, tokenizer, chunks, query, top_k=num_to_retrieve, rag_threshold=rag_threshold)
    bm25_retrieval_chunks = retrieve_short_context_bm25(tokenizer, chunks, query, top_k=5)
    print("bm25_retrieval_chunks: ", len(bm25_retrieval_chunks))
    all_chunks = model_retrieved_chunks + bm25_retrieval_chunks
    unique_chunks = {chunk['position']: chunk for chunk in all_chunks}.values()
    
    # Sort by original position
    sorted_chunks = sorted(unique_chunks, key=lambda x: x['position'])
    
    # Join the sorted chunks
    retrieved_context = ' '.join(chunk['text'] for chunk in sorted_chunks)
    return retrieved_context



def get_input_prompts(
        long_context: str,
        query: str,
        embed_model_path: str,
        num_chunks_to_retrieve: int,
        max_chunk_len: int,
        tokenizer,
        global_instruction: str = None,
        rag_threshold: float = 0.3):
    if global_instruction:
        global_instruction = "\n" + global_instruction
    retrieved_context = get_rag_context(long_context, query, embed_model_path, tokenizer, num_chunks_to_retrieve, max_chunk_len, rag_threshold)
    rag_prompt = RAG_PROMPT_TEMPLATE.format(rag_context=retrieved_context)
    # rag_prompt_with_full = RAG_PROMPT_TEMPLATE_WITH_FULL.format(rag_context=retrieved_context, long_context=long_context)
    full_context = LONG_CONTEXT_PROMPT_TEMPLATE.format(long_context=long_context)
    full_input = full_context + "\n" + query + "\n" + FULL_INSTRUCTION + global_instruction

    # full_input = rag_prompt_with_full + "\n" + query + "\n" + FULL_INSTRUCTION + global_instruction
    rag_input = rag_prompt + "\n" + query + "\n" + RAG_INSTRUCTION + global_instruction
    return rag_input, full_input