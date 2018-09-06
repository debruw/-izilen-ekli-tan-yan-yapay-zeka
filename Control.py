
# coding: utf-8

# Web sayfasından aldığımız çizim verisini kontrol etmek için yazılmış kodlar.
# Veriyi web sayfamızda numpy dizisi olarak kaydediyoruz. Burada ise mevcut diziyi okuyup ekrana çizdiriyoruz.

# In[1]:


import numpy as np 

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# In[2]:


image = np.load('C:/Users/Ebru/Desktop/Flask/bir.npy')


# In[3]:


print(image.shape)


# In[4]:


plt.imshow(image.reshape(28, 28), cmap='gray_r', interpolation='nearest')

