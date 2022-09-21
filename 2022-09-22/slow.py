from torchtext.datasets import EnWik9
from torch.utils.data import DataLoader
from defusedxml.sax import parse
from xml.sax import ContentHandler, SAXParseException
#import mwparserfromhell

class WikiHandler(ContentHandler):
    def __init__(self):
        self.in_text = False
        self.articles = []
        self.strings = []

    def startElement(self, name, attrs):
        if name == 'text':
            self.in_text = True

    def endElement(self, name):
        if name == 'text':
            self.articles.append(''.join(self.strings))
            self.strings = []
            self.in_text = False

    def characters(self, content):
        if self.in_text:
            self.strings.append(content)

def main():
    h = WikiHandler()
    try:
        print("Reading data")
        parse('data/enwik9', h)
    except SAXParseException as e:
        pass
    print(len(h.articles))
    for a in h.articles[:20]:
        print(a)
        print()
        print()

if __name__ == '__main__':
    main()
