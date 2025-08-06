#------------------------ Question 3 ------------------------

# poem.txt Contains famous poem "Road not taken" by poet Robert Frost. 
# You have to read this file in python and print every word and its count as show below. 
# Think about the best data structure that you can use to solve this problem and figure 
# out why you selected that specific data structure.



from collections import Counter
#read txt file in python
def read_txt(file_path):
    with open(file_path, 'r') as file:
        text = file.read().lower()  # Read the file and convert to lowercase
        words = text.split()  # Split text into words
        word_count = Counter(words)  # Count occurrences of each word
    return word_count

file_path = 'poem.txt'
word_count = read_txt(file_path)

for word, count in word_count.items():
    print(f"{word}: {count}")

#--------------------- Question 4 ------------------------
class HashTable:  
    def __init__(self):
        self.MAX = 10
        self.arr = [[] for i in range(self.MAX)]
        
    def get_hash(self, key):
        hash = 0
        for char in key:
            hash += ord(char)
        return hash % self.MAX
    
    def __getitem__(self, key):
        arr_index = self.get_hash(key)
        for kv in self.arr[arr_index]:
            if kv[0] == key:
                return kv[1]
            
    def __setitem__(self, key, val):
        h = self.get_hash(key)
        found = False
        for idx, element in enumerate(self.arr[h]):
            if len(element)==2 and element[0] == key:
                self.arr[h][idx] = (key,val)
                found = True
        if not found:
            self.arr[h].append((key,val))
        
    def __delitem__(self, key):
        arr_index = self.get_hash(key)
        for index, kv in enumerate(self.arr[arr_index]):
            if kv[0] == key:
                print("del",index)
                del self.arr[arr_index][index]
                

t = HashTable()
t["march 6"] = 310
t["march 7"] = 420
t["march 8"] = 67
t["march 17"] = 63457

