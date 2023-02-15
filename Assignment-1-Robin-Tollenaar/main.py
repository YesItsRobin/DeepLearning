from mynetworkp1 import MyNetwork1
from mynetworkp2 import MyNetwork2

def p1():
    net=MyNetwork1([1.1],[0],100,1)
    net.train([[[0,1,2,3],0],[[1,1,2,3],2],[[2,1,2,3],4],[[3,1,2,3],6],[[4,1,2,3],8]])
def p2():
    net=MyNetwork2([1.1],[0],100,1)
    net.train([[0,0],[1,2],[2,4],[3,6],[4,8]])

def main():
    p2()

if __name__ == "__main__":
    main()