import openai

openai.api_key = 'XX'

def generate_personalized_content(user_profile, uploaded_text):
    # Create a prompt based on user profile
    prompt = f"""
    The user is a {user_profile['year']} year university student majoring in {user_profile['major']}.
    They prefer learning through {user_profile['preferred_method']} and have a {user_profile['learning_speed']} learning speed.
    Explain the following content in a way that suits their {user_profile['learning_style']} learning style:
    
    {uploaded_text}
    """
    
    response = openai.Completion.create(
        engine="text-davinci-004",
        prompt=prompt,
        max_tokens=500,
        n=1,
        stop=None,
        temperature=0.7
    )
    return response.choices[0].text.strip()

# Example user profile
user_profile = {
    'year': 'second',
    'major': 'Computer Science',
    'learning_speed': 'fast',
    'learning_style': 'visual',
    'preferred_method': 'videos'
}

# Example uploaded text
uploaded_text = """
Quantum mechanics is a fundamental theory in physics that provides a description of the physical properties of nature at the scale of atoms and subatomic particles.
"""

# Generate personalized content
personalized_content = generate_personalized_content(user_profile, uploaded_text)
print(personalized_content)