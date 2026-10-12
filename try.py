from command_extractor import CommandExtractor

ext = CommandExtractor(model="qwen2.5:1.5b")

test_phrases = [
    "what's behind of me?",
    "read the menu",
    "how much battery do I have?",
    "help!",
    "what's to my right?",
    "read the board",
    "hello how are you, i miss u",   
]

for phrase in test_phrases:
    result = ext.extract(phrase)
    print(result)