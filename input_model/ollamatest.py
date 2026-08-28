from ollama import chat
import time

start = time.perf_counter()

response = chat(
    model="qwen3:8b",
    messages=[
        {
            "role": "user",
            "content": "Return only the word: Cardiology"
        }
    ],
    keep_alive=-1
)

end = time.perf_counter()

print("Response:")
print(repr(response["message"]["content"]))

print(f"Runtime: {end - start:.2f} seconds")
