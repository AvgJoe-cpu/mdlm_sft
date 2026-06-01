import hashlib


def add_hash_id(example, text_field="text"):
    """
    Generate a SHA-256 hash of the text as a unique ID.
    
    Args:
        example: Dataset example
        text_field: Name of the text field to hash
    
    Returns:
        Dict with 'id' key containing the hash
    """
    # Handle different possible text field names
    text = example.get(text_field) or example.get("text") or example.get("story") or example.get("prompt", "")
    text_bytes = str(text).encode('utf-8')
    hash_id = hashlib.sha256(text_bytes).hexdigest()
    return {"id": hash_id}


def count_tokens(example, tokenizer, text_field="text"):
    """
    Count total tokens in the text (NOT sentences).
    
    Args:
        example: Dataset example
        tokenizer: Tokenizer to use
        text_field: Name of the text field to tokenize
    
    Returns:
        Dict with 'token_count' key
    """
    # Handle different possible text field names
    text = example.get(text_field) or example.get("text") or example.get("story") or example.get("prompt", "")
    token_count = len(tokenizer.encode(str(text), add_special_tokens=False))
    return {"token_count": token_count}


def count_sentence_tokens(example, tokenizer, sentences_field="sentences"):
    """
    Count total tokens across all non-empty sentences.
    
    Args:
        example: Dataset example with a list of sentences
        tokenizer: Tokenizer to use
        sentences_field: Name of the field containing sentences
    
    Returns:
        Dict with 'token_count' key
    """
    total_tokens = 0
    for sent in example.get(sentences_field, []):
        if sent.strip():  # Skip empty sentences
            total_tokens += len(tokenizer.encode(sent, add_special_tokens=False))
    return {"token_count": total_tokens}