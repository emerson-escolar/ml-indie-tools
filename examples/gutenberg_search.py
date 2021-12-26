import sys
sys.path.append('../src/ml_indie_tools')
from env_tools import *
from Gutenberg_Dataset import *

ml_env = MLEnv()
desc = ml_env.describe()
print(desc)

gd = Gutenberg_Dataset()
sl = gd.search({'title': ['philo'], 'language': 'german'})
print(sl)
print(f"Length: {len(sl)}")

sl=gd.insert_book_texts(sl)
for t in sl:
    if 'title' in t:
        print(f"Title: {t['title']}")
    else:
        print("no title")
    if 'text' in t:
        print("has text")
    else:
        print("No text")
    print("")

