def is_palindrome(text: str) -> bool:
    # Extract only Unicode letters and digits, convert to lowercase
    cleaned = []
    for char in text:
        if char.isalnum():
            cleaned.append(char.lower())
    s = ''.join(cleaned)
    return s == s[::-1]


# Test cases
assert is_palindrome("A man a plan a canal Panama") == True
assert is_palindrome("race a car") == False
assert is_palindrome("Was it a car or a cat I saw?") == True
assert is_palindrome("Madam") == True
assert is_palindrome("你好好你") == True
