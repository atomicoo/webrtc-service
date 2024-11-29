import sys, hashlib

def get_unique_id_from_keyword(keyword: str, max_value: int = sys.maxsize) -> int:
    hash_object = hashlib.sha256(keyword.encode())
    hex_digits = hash_object.hexdigest()
    unique_id = int(hex_digits, 16) % max_value
    print(f'keyword: {keyword}; unique_id: {unique_id}')
    return unique_id