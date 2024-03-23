# promptengineering-with-llama-2#!/usr/bin/env python
# coding: utf-8

# # Lesson 2

# ### Getting started with Llama 2
# 
# The code to call the Llama 2 models through the Together.ai hosted API service has been wrapped into a helper function called `llama`. You can take a look at this code if you like by opening the utils.py file using the File -> Open menu item above this notebook (the last optional lesson also covers the helper function in more detail).
# 
# Note: To see how to run Llama 2 locally on your own computer, you can go to the last section of this notebook.

# In[1]:


# import llama helper function
from utils import llama


# In[2]:


# define the prompt
prompt = "Help me write a birthday card for my dear friend Andrew."


# **Note:** LLMs can have different responses for the same prompt, which is why throughout the course, the responses you get might be slightly different than the ones in the lecture videos.

# In[ ]:


# pass prompt to the llama function, store output as 'response' then print
response = llama(prompt)
print(response)


# In[ ]:


# Set verbose to True to see the full prompt that is passed to the model.
prompt = "Help me write a birthday card for my dear friend Andrew."
response = llama(prompt, verbose=True)


# ### Chat vs. base models
# 
# Ask model a simple question to demonstrate the different behavior of chat vs. base models.

# In[ ]:


### chat model
prompt = "What is the capital of France?"
response = llama(prompt, 
                 verbose=True,
                 model="togethercomputer/llama-2-7b-chat")


# In[ ]:


print(response)


# In[ ]:


### base model
prompt = "What is the capital of France?"
response = llama(prompt, 
                 verbose=True,
                 add_inst=False,
                 model="togethercomputer/llama-2-7b")


# Note how the prompt **does not** include the `[INST]` and `[/INST]` tags as `add_inst` was set to `False`.

# In[ ]:


print(response)


# ### Changing the temperature setting

# In[ ]:


prompt = """
Help me write a birthday card for my dear friend Andrew.
Here are details about my friend:
He likes long walks on the beach and reading in the bookstore.
His hobbies include reading research papers and speaking at conferences.
His favorite color is light blue.
He likes pandas.
"""
response = llama(prompt, temperature=0.0)
print(response)


# In[ ]:


# Run the code again - the output should be identical
response = llama(prompt, temperature=0.0)
print(response)


# In[ ]:


prompt = """
Help me write a birthday card for my dear friend Andrew.
Here are details about my friend:
He likes long walks on the beach and reading in the bookstore.
His hobbies include reading research papers and speaking at conferences.
His favorite color is light blue.
He likes pandas.
"""
response = llama(prompt, temperature=0.9)
print(response)


# In[ ]:


# run the code again - the output should be different
response = llama(prompt, temperature=0.9)
print(response)


# ### Changing the max tokens setting

# In[ ]:


prompt = """
Help me write a birthday card for my dear friend Andrew.
Here are details about my friend:
He likes long walks on the beach and reading in the bookstore.
His hobbies include reading research papers and speaking at conferences.
His favorite color is light blue.
He likes pandas.
"""
response = llama(prompt,max_tokens=20)
print(response)


# The next cell reads in the text of the children's book *The Velveteen Rabbit* by Margery Williams, and stores it as a string named `text`. (Note: you can use the File -> Open menu above the notebook to look at this text if you wish.)

# In[ ]:


with open("TheVelveteenRabbit.txt", "r", encoding='utf=8') as file:
    text = file.read()


# In[ ]:


prompt = f"""
Give me a summary of the following text in 50 words:\n\n
{text}
"""
response = llama(prompt)


# In[ ]:


print(response)


# Running the cell above returns an error because we have too many tokens. 

# In[ ]:


# sum of input tokens (prompt + Velveteen Rabbit text) and output tokens
3974 + 1024


# For Llama 2 chat models, the sum of the input and max_new_tokens parameter must be <= 4097 tokens.

# In[ ]:


# calculate tokens available for response after accounting for 3974 input tokens
4097 - 3974


# In[ ]:


# set max_tokens to stay within limit on input + output tokens
prompt = f"""
Give me a summary of the following text in 50 words:\n\n
{text}
"""
response = llama(prompt,
                max_tokens=123)


# In[ ]:


print(response)


# In[ ]:


# increase max_tokens beyond limit on input + output tokens
prompt = f"""
Give me a summary of the following text in 50 words:\n\n
{text}
"""
response = llama(prompt,
                max_tokens=124)


# In[ ]:


print(response)


# ### Asking a follow up question

# In[ ]:


prompt = """
Help me write a birthday card for my dear friend Andrew.
Here are details about my friend:
He likes long walks on the beach and reading in the bookstore.
His hobbies include reading research papers and speaking at conferences.
His favorite color is light blue.
He likes pandas.
"""
response = llama(prompt)
print(response)


# In[ ]:


prompt_2 = """
Oh, he also likes teaching. Can you rewrite it to include that?
"""
response_2 = llama(prompt_2)
print(response_2)


# ### (Optional): Using Llama-2 on your own computer!
# - The small 7B Llama chat model is free to download on your own machine!
#   - **Note** that only the Llama 2 7B chat model (by default the 4-bit quantized version is downloaded) may work fine locally.
#   - Other larger sized models could require too much memory (13b models generally require at least 16GB of RAM and 70b models at least 64GB of RAM) and run too slowly.
#   - The Meta team still recommends using a hosted API service (in this case, the classroom is using Together.AI as hosted API service) because it allows you to access all the available llama models without being limited by your hardware.
#   - You can find more instructions on using the Together.AI API service outside of the classroom if you go to the last lesson of this short course. 
# - One way to install and use llama 7B on your computer is to go to https://ollama.com/ and download app. It will be like installing a regular application.
# - To use llama-2, the full instructions are here: https://ollama.com/library/llama2
# 
# 
# #### Here's an quick summary of how to get started:
#   - Follow the installation instructions (for Windows, Mac or Linux).
#   - Open the command line interface (CLI) and type `ollama run llama2`.
#   - The first time you do this, it will take some time to download the llama-2 model. After that, you'll see 
# > `>>> Send a message (/? for help)`
# 
# - You can type your prompt and the llama-2 model on your computer will give you a response!
# - To exit, type `/bye`.
# - For a list of other commands, type `/?`.
# 
# ![](ollama_example.png "")
# 
# 
# 

# In[ ]:



