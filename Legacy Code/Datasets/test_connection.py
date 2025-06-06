# test_connection.py
from openai import OpenAI

# Configure client with your remote server
client = OpenAI(
    api_key="dummy-key",
    base_url="http://172.16.2.236:11434/v1/"  # Replace with your server IP
)

# Test the connection
try:
    response = client.chat.completions.create(
        model="llama3.2",  # or your model name
        messages=[
            {"role": "user", "content": "Hello, are you working?"}
        ]
    )
    print("Connection successful!")
    print("Response:", response.choices[0].message.content)
except Exception as e:
    print(f"Connection failed: {str(e)}")