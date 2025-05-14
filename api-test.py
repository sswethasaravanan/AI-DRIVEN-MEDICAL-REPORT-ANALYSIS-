import google.generativeai as genai


genai.configure(api_key="AIzaSyBzJM5n3_f5d62hDTOOGLSkI3pBzUuUnlk")


available_models = genai.list_models()


print("Available Gemini Models:")
for model in available_models:
    print(model.name)