import google.generativeai as genai


genai.configure(api_key="")


available_models = genai.list_models()


print("Available Gemini Models:")
for model in available_models:
    print(model.name)