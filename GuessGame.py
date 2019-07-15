# coding: utf-8

# In[1]:


import random


# In[2]:


price = random.randint(1, 100)


# In[ ]:





# In[3]:


while(True):
    guess = input("Enter a number:")
    guess_num = int(guess)
    if(guess_num > price):
        print("Too high")
    elif(guess_num < price):
        print("Too low")
    else:
        print("You got it!")
        break


# In[4]:


print(price)
print(guess)
print(guess_num)


# In[ ]:




