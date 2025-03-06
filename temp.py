import base64
import re

class Base64Encoding:
    """
    A class that provides methods for working with Base64 encoding.
    """

    @staticmethod
    def is_base64(input_string):
        """
        Checks if a string is Base64 encoded.
        
        Args:
            input_string (str): The string to check.
            
        Returns:
            bool: True if the string is Base64 encoded, False otherwise.
        """
        # Check if the string is potentially a base64 string
        # Base64 strings should only contain alphanumeric chars, '+', '/' and possibly '=' padding
        pattern = r'^[A-Za-z0-9+/]+={0,2}$'
        
        # First check basic pattern
        if not re.match(pattern, input_string):
            return False
            
        # Check if length is valid (must be multiple of 4)
        if len(input_string) % 4 != 0:
            return False
            
        # Try to decode it
        try:
            # Attempt to decode and encode back to see if we get the same result
            decoded = base64.b64decode(input_string)
            encoded = base64.b64encode(decoded).decode('utf-8')
            return encoded == input_string
        except Exception:
            return False
    
    @staticmethod
    def encode(input_string):
        """
        Converts a regular string to a Base64 encoded string.
        
        Args:
            input_string (str): The string to encode.
            
        Returns:
            str: The Base64 encoded string.
        """
        # Convert the string to bytes and then to base64
        string_bytes = input_string.encode('utf-8')
        encoded_bytes = base64.b64encode(string_bytes)
        
        # Convert back to string and return
        return encoded_bytes.decode('utf-8')
    
    @staticmethod
    def decode(base64_string):
        """
        Converts a Base64 encoded string back to a regular string.
        
        Args:
            base64_string (str): The Base64 encoded string to decode.
            
        Returns:
            str: The decoded string.
            
        Raises:
            ValueError: If the input is not a valid Base64 encoded string.
        """
        try:
            # Convert the base64 string to bytes, then decode it
            decoded_bytes = base64.b64decode(base64_string)
            
            # Convert back to string and return
            return decoded_bytes.decode('utf-8')
        except Exception as e:
            raise ValueError(f"Invalid Base64 string: {e}")



#---------------------------------------------------------------------------------------


# Create an instance
b64 = Base64Encoding()

# Check if a string is base64 encoded
is_encoded = b64.is_base64("SGVsbG8gV29ybGQ=")  # Returns True

# Encode a string to base64
encoded = b64.encode("Hello World")  # Returns "SGVsbG8gV29ybGQ="

# Decode a base64 string
decoded = b64.decode("SGVsbG8gV29ybGQ=")  # Returns "Hello World"



# -----------------------------------------------------------------------------------
# Static method usage
encoded = Base64Encoding.encode("Hello World")
decoded = Base64Encoding.decode(encoded)
print(decoded)  # "Hello World"
