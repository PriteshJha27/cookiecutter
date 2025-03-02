
import base64
import re

def decode_base64_list(string_list):
    """
    Checks if each string in the input list is Base64 encoded.
    If a string is Base64 encoded, decodes it to a normal string.
    Otherwise, returns the original string.
    
    Args:
        string_list (list): A list of strings
        
    Returns:
        list: A list of strings with Base64 encoded strings decoded
    """
    result = []
    
    for item in string_list:
        # Check if the string is potentially Base64 encoded
        # Base64 strings should only contain A-Z, a-z, 0-9, +, /, and = (padding)
        if re.match(r'^[A-Za-z0-9+/]+={0,2}$', item):
            try:
                # Try to decode the string
                decoded = base64.b64decode(item).decode('utf-8')
                result.append(decoded)
            except Exception:
                # If decoding fails, it's not a valid Base64 string
                result.append(item)
        else:
            # If the pattern doesn't match, it's not Base64
            result.append(item)
    
    return result

# Example usage
if __name__ == "__main__":
    # Example list with mixed strings and Base64 encoded strings
    test_list = [
        "SGVsbG8gV29ybGQ=",  # "Hello World" in Base64
        "Regular text",
        "VGhpcyBpcyBhIHRlc3Q=",  # "This is a test" in Base64
        "12345",
        "QmFzZTY0IEVuY29kaW5n"  # "Base64 Encoding" in Base64
    ]
    
    decoded_list = decode_base64_list(test_list)
    print("Original list:", test_list)
    print("Decoded list:", decoded_list)
