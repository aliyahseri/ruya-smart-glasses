import json
from command_extractor import CommandExtractor

# Initialize with your chosen model
ext = CommandExtractor(model="llama3.2:1b")

print("Live Command Extractor (type 'exit' to quit)\n")

while True:
    # read input from user
    user_input = input("You: ").strip()
    
    if user_input.lower() == "exit":
        print("Goodbye!")
        break
    
    # run the extractor
    result = ext.extract(user_input)
    
    # print the output as pretty JSON
    output = result.to_dict()
    print(json.dumps(output, indent=2), "\n")