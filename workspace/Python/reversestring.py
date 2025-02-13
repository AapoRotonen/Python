def reverse(wordsString):
    # Split the string into a list of words
    words = wordsString.split()
    # Reverse the list of words
    reversed_words = words[::-1]
    # Join the words back into a single string
    return ' '.join(reversed_words)

# Example usage
input_string = "Hello world this is Python"
print(reverse(input_string))  # Output: "Python is this world Hello"
