import re

test_text_1 = '"A tingling feeling in my toes" → find the clip "walking barefoot on broken glass" → find the clip "always worse at night"'
test_text_2 = '"I couldn\'t stand anything" -> find a clip "touching my feet" -> find the clip "not even a bed sheet"'
test_text_3 = 'Normal text without any instructions.'
test_text_4 = 'Text with "quotes" but not instructions.'

# Regex to remove:
# 1. Optional arrow: (→|->)\s*
# 2. Key phrase: (find (the|a) clip)\s*
# 3. Quoted content: ".*?"
# 4. Trailing spaces

# Combining them:
patterns = [
    r'→\s*find (the|a) clip\s*".*?"',
    r'->\s*find (the|a) clip\s*".*?"'
]

def clean_text(text):
    cleaned = text
    # We can use a single regex for both arrow types or just replace multiple times
    # Using a comprehensive regex:
    # (→|->)\s*find (the|a) clip\s*".*?"
    
    regex = r'(→|->)\s*find (the|a) clip\s*".*?"'
    cleaned = re.sub(regex, '', cleaned, flags=re.IGNORECASE)
    
    # Also clean up any resulting double spaces or trailing arrows if the regex missed something subtle
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned

print(f"Original 1: {test_text_1}")
print(f"Cleaned 1:  {clean_text(test_text_1)}")
print("-" * 20)
print(f"Original 2: {test_text_2}")
print(f"Cleaned 2:  {clean_text(test_text_2)}")
print("-" * 20)
print(f"Original 3: {test_text_3}")
print(f"Cleaned 3:  {clean_text(test_text_3)}")
print("-" * 20)
print(f"Original 4: {test_text_4}")
print(f"Cleaned 4:  {clean_text(test_text_4)}")
