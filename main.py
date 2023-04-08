import re
from typing import Union

from fastapi import FastAPI
from langchain import LLMChain
import openai
from pydantic import BaseModel
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from bs4 import BeautifulSoup

import os
import requests
import urllib.parse
import json

OPEN_AI_API_KEY = "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"
os.environ["OPENAI_API_KEY"] = OPEN_AI_API_KEY
openai.api_key = OPEN_AI_API_KEY #"ssk-eM518KKScep9NO4MmvbHT3BlbkFJLDY1PNzBDBHQSBdQv6IJ"

app = FastAPI()


meals = [
]

class Item(BaseModel):
    ingredients: str

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/ingredients")
def read_root():
    with open('ingredients.json', 'r') as f:
        ingredients_data = json.load(f)
    return  ingredients_data

@app.post("/meal/discover")
def read_root(item :Item):
    recipe_name = get_recipe_name(item.ingredients)
    recipe_details = get_recipe_details(recipe_name)
    recipe_image = generate_meal_image(recipe_name)
    return {"recipe_name": recipe_name, "recipe_details": recipe_details, "recipe_image": recipe_image}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}

def get_recipes_image_url(query):
    query = urllib.parse.quote(query)
    url = "https://www.google.com/search?q=" + query + "&source=lnms&tbm=isch"
    header = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'}
    response = requests.get(url, headers=header)
    soup = BeautifulSoup(response.content, 'html.parser')
    images = soup.find_all('img')
    image_url = images[1]['src']
    return image_url


def get_recipe_name(ingredients):
    llm = OpenAI(temperature=0.9)
    prompt = PromptTemplate(
        input_variables=["ingredients"],
        template="What is a good meal I can do with these ingredients {ingredients}? only provide the name",
    )
    chain = LLMChain(llm=llm, prompt=prompt)

    return chain.run(ingredients)

def get_recipe_details(recipe_name):
    llm = OpenAI(temperature=0.9)
    prompt = PromptTemplate(
        input_variables=["recipe_name"],
        template="Given that recipe {recipe_name}? provide details, difficulty and time it might take",
    )
    chain = LLMChain(llm=llm, prompt=prompt)

    return chain.run(recipe_name)

def generate_meal_image(name):
    # Generate image with DALL-E 2
    response = openai.Image.create(
        prompt=f"a {name}",
        n=1,
        size="512x512",
        response_format="url"
    )
    image_url = response['data'][0]['url']

    # Check if image is accessible
    response = requests.get(image_url)
    if response.status_code != 200:
        raise ValueError("Failed to download image")

    return image_url
def text_to_markdown(text):
    # Convert headers
    text = re.sub(r'^(#+)\s*(.*)', r'\1 \2', text, flags=re.MULTILINE)

    # Convert bold text
    text = re.sub(r'\*\*(.+?)\*\*', r'**\1**', text)

    # Convert italic text
    text = re.sub(r'\'\'(.+?)\'\'', r'*\1*', text)

    # Convert links
    text = re.sub(r'\[(.+?)\]\((.+?)\)', r'[\1](\2)', text)

    return text