from mynetwork import MyNetwork

p2=[[0,0],[1,2],[2,4],[3,6],[4,8]]

def mainp2():
    net=MyNetwork([1,1],[0,0],100,2,4,1)
    net.train(p2)

if __name__ == "__main__":
    mainp2()