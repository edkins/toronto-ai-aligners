from torchtext.datasets import EnWik9
from torch.utils.data import DataLoader
from defusedxml.ElementTree import parse

def main():
    dp = EnWik9()
    print('created dp')
    i = 0
    for line in dp:
        print(i)
        print(line)
        i += 1
        if i > 100:
            break

if __name__ == '__main__':
    main()
