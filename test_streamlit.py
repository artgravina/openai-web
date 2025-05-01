import os
from dotenv import load_dotenv
import streamlit as st
import openai
from pinecone import Pinecone, ServerlessSpec

from openai import OpenAI

load_dotenv()  # load .env support

openai.api_key = os.getenv("OPENAI_API_KEY") # private key neessary to access openai

model_name = 'text-embedding-ada-002'
#model_name = 'text-embedding-3-small'
client = OpenAI()

pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone = Pinecone(
    api_key=os.environ.get(pinecone_api_key)
)
pinecone_index_name = "movie-recommendation"

if not pinecone.has_index(pinecone_index_name):
    print(f"Error: No movie recommendation index in pincone: {pinecone_index_name}")
    quit()

# if not there run 4_Movies/movie_recommend_3. Change line 23 the if
index = pinecone.Index('movie-recommendation')

# AI Functions ===================
print("Section 9 Web Interface (streamlit run 9_WebInterfaces/webinterface.py)")

st.sidebar.title("AI Apps")
application_choice = st.sidebar.radio("Choose an AI App", ("Blog Generator", "Image Generator", "Movie Recommender"))


def generate_blog(topic, additional_text):
    
    prompt = f"""
    You are a copy writer with years of experience writing impactful blog that converge and help elevate brands.
    Your task is to write a blog on any topic system provided to you. Make sure to write in a format that works for Medium.
    Each blog should be separated into segments that have titles and subtitles.

    Topic: {topic}
    Additiona pointers: {additional_text}
    """
    
    response = client.completions.create(
        model="gpt-3.5-turbo-instruct",
        prompt=prompt,
        max_tokens=700,
        temperature=0.9
    )
    
    return response
        
def generate_images(prompt, number_of_images):
    response = client.images.generate(
        prompt=prompt,
        n=number_of_images,
        size="512x512"
    )
    
    return response

# End AI Functions ==============

# main application logic

def main():
    if application_choice == "Blog Generator":
        st.header("Blog Gernerator")
        st.write("Input a topic to generate a blog about it using OpenAI API")
        
        topic = st.text_area("Topic", height=70)
        additional_text = st.text_area("Additional Text", height=70)
        
        if st.button("Genrate Blog"):
            with st.spinner("Generating...."):
                response = generate_blog(topic, additional_text)
                st.text_area("Generated blog", value=response.choices[0].text, height=700)
        
    elif application_choice == "Image Generator":
        st.header("Image Gernerator")
        st.write("Add a prompt to generate an image using OpenAI API and DALLE model")
        
        prompt = st.text_area("prompts", height=70)
        
        number_of_images = st.slider("Number of Images", 1, 5, 1)  
        
        if st.button("Generate Image") and prompt != "":
            with st.spinner("Generating...."):
                response = generate_images(prompt=prompt,number_of_images=number_of_images)
                
                for output in response.data:
                    st.image(output.url)
                            
    elif application_choice == "Movie Recommender":
        st.header("Movie Rcommender")
        st.write("Describe a movie that you would like to see")
        
        movie_description = st.text_area("Movie Description", height=70)
        
        if st.button("Get Movie Recommendations") and movie_description != "":
            
            with st.spinner("Loading..."):
                
            
                vector = client.embeddings.create(
                    model=model_name,
                    input=movie_description
                )
                
                result_vector = vector.data[0].embedding
                
                result = index.query(
                    vector=result_vector,
                    top_k=10,
                    include_metadata=True            
                )
                
                for movie in result.matches:
                    st.write(movie['metadata']['title'])

# Run the main function
if __name__ == "__main__":
    main()