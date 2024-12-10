import re

class TextTransform:
  ''' Map characters to integers and vice versa '''
  def __init__(self):
    self.char_map = {}
    self.char_map["<pad>"] = 0
    self.char_map["@"] = 1 #<s>
    self.char_map["!"] = 2 #</s>
    self.char_map["_"] = 3 #<blank>

    for i, char in enumerate(range(65, 91), 4):
      self.char_map[chr(char)] = i

    self.index_map = {} 
    for char, i in self.char_map.items():
      self.index_map[i] = char

  def text_to_int(self, text):
      ''' Map text string to an integer sequence '''
      int_sequence = []
      for c in text:
        ch = self.char_map[c]
        int_sequence.append(ch)
      return int_sequence

  def int_to_text(self, labels):
      ''' Map integer sequence to text string '''
      string = []
      for i in labels:
          if i == 3: # blank char
            continue
          else:
            string.append(self.index_map[i])
      return ''.join(string)


def preprocess(word):
    word = re.sub(r'[^\w\s]', '', word)
    word = word.upper()
    word = re.sub(" ", "_", word)
    word = "@" + word + "!"

    return word

#t = TextTransform()
#print(preprocess("Hello, 'world'! How's everything going? Great, I hope."))
#print(t.text_to_int(preprocess("Hello, 'world'! How's everything going? Great, I hope.")))
