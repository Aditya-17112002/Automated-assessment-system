import google.generativeai as genai

genai.configure(api_key="AIzaSyB0ZRg2g79v4RDHvSb9JewJE8XXfvMGwzk")
model = genai.GenerativeModel(model_name="gemini-1.5-flash")
response = model.generate_content("Explain how AI works")
print(response.text)


import google.generativeai as genai

# Configure API Key
genai.configure(api_key="AIzaSyB0ZRg2g79v4RDHvSb9JewJE8XXfvMGwzk")  # Replace with your API key

# Initialize the Gemini Model
model = genai.GenerativeModel("gemini-1.5-flash")

# Get the subject from the user
subject = input("Enter a subject: ")

# Define a prompt to generate five questions related to the subject
prompt = f"Generate five thought-provoking questions about {subject}."

# Generate the response
response = model.generate_content(prompt)

# Print the generated questions
print("\nHere are 5 questions related to:", subject)
print(response.text)


# import google.generativeai as genai

# # Configure API Key
# genai.configure(api_key="AIzaSyB0ZRg2g79v4RDHvSb9JewJE8XXfvMGwzk")

# # Use an available model (chat-bison-001)
# model = genai.GenerativeModel("models/chat-bison-001")

# # Generate response
# response = model.generate_content("Explain how AI works")

# # Print response
# print(response.text)

# import google.generativeai as genai

# # Step 1: Configure API Key
# genai.configure(api_key="AIzaSyB0ZRg2g79v4RDHvSb9JewJE8XXfvMGwzk")

# # Step 2: Verify Available Models
# available_models = [model.name for model in genai.list_models()]
# print("Available Models:", available_models)

# # Step 3: Use a Correct Model Name from the List
# if "gemini-1.5-flash" in available_models:
#     model = genai.GenerativeModel("gemini-1.5-flash")
#     prompt = "Explain how AI works"

#     # Step 4: Generate Response
#     response = model.generate_content(prompt)
    
#     # Step 5: Print Output
#     print("AI Response:", response.text)
# else:
#     print("Model 'gemini-1.5-flash' is not available. Check your API key access.")
