from collections import Counter

text = """
image7.669ff4a706f8f3.27064217.png
image7.669ff4a706f8f3.27064217.png
image7.669f9bcb736c78.09388737.png
image7.669f9bcb736c78.09388737.png
image7.669f9bcb736c78.09388737.png
image7.669ff4a706f8f3.27064217.png
image7.669f9bcb736c78.09388737.png
image7.669f9bcb736c78.09388737.png
image7.669f9bcb736c78.09388737.png
image7.669ff4a706f8f3.27064217.png
image7.669ff4a706f8f3.27064217.png
image7.669ff4a706f8f3.27064217.png
image7.669f9bcb736c78.09388737.png
image7.669f9bcb736c78.09388737.png
image7.669ff4a706f8f3.27064217.png
image7.669f9bcb736c78.09388737.png
image7.669ff4a706f8f3.27064217.png
image7.669ff4a706f8f3.27064217.png
image7.669f9bcb736c78.09388737.png
image7.669f9bcb736c78.09388737.png
image7.669f9bcb736c78.09388737.png
image7.669f9bcb736c78.09388737.png
image7.669f9bcb736c78.09388737.png
image7.669f9bcb736c78.09388737.png
image7.669f9bcb736c78.09388737.png
image7.669f9bcb736c78.09388737.png
image7.669f9bcb736c78.09388737.png
image7.669f9bcb736c78.09388737.png
image7.669f9bcb736c78.09388737.png
image7.669ff4a706f8f3.27064217.png
image7.669ff4a706f8f3.27064217.png
image7.669f9bcb736c78.09388737.png
image7.669ff4a706f8f3.27064217.png
image7.669f9bcb736c78.09388737.png
image7.669f9bcb736c78.09388737.png
image7.669f9bcb736c78.09388737.png
image7.669f9bcb736c78.09388737.png
image7.669f9bcb736c78.09388737.png
"""

def count_image_names(text):
    counter = Counter(text.split())
    return counter

# Call the function and print the result
image_counts = count_image_names(text)
print(image_counts)