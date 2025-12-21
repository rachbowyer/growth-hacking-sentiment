from ollama import chat

# response = chat(
#     model="llama4:scout",
#     messages=[
#         {"role": "user", "content": "Explain black holes simply."}
#     ]
# )

# print(response["message"]["content"])


response = chat(
    model="gpt-oss:20b",
    messages=[
        {"role": "user", "content": "Who are you?"}
    ]
)

print(response["message"]["content"])
